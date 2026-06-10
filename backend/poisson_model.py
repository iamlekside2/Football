"""
Dixon-Coles Poisson model for football match outcomes.

Fits team attack / defence strengths plus a home-advantage term and the
Dixon-Coles low-score correlation correction (rho). From the fitted
parameters it produces a full score-probability matrix and from that the
1X2 (home / draw / away) and over/under 2.5 goal probabilities.

`find_value()` compares those model probabilities against bookmaker odds and
flags outcomes the model thinks are underpriced.

Reference: Dixon, M. & Coles, S. (1997), "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market".

This module is deliberately dependency-light (numpy + scipy) so it can be
driven by a walk-forward backtest (see backtest.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

# Outcome keys used throughout: home win / draw / away win.
OUTCOMES = ("H", "D", "A")


class DixonColesModel:
    """A Dixon-Coles bivariate-Poisson-style model fitted by maximum likelihood.

    Usage:
        m = DixonColesModel()
        m.fit(train_df)                       # train_df: HomeTeam, AwayTeam, FTHG, FTAG[, Date]
        probs = m.predict_proba("Arsenal", "Chelsea")
        # -> {"H":..., "D":..., "A":..., "over_2.5":..., "under_2.5":...}
    """

    def __init__(self, max_goals: int = 10):
        self.max_goals = int(max_goals)
        self.teams: Optional[List[str]] = None
        self.team_index: Dict[str, int] = {}
        self.attack: Optional[np.ndarray] = None
        self.defence: Optional[np.ndarray] = None
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self.fitted: bool = False

    # ------------------------------------------------------------------ #
    # Likelihood                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tau(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray,
             mu: np.ndarray, rho: float) -> np.ndarray:
        """Dixon-Coles low-score dependency correction, vectorised."""
        tau = np.ones_like(lam, dtype=float)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        return tau

    def _unpack(self, params: np.ndarray, n: int):
        """Rebuild (attack, defence, home_adv, rho) from the flat vector.

        Attack has a sum-to-zero constraint to remove the additive
        redundancy between attack and defence; we optimise n-1 free attack
        values and derive the last as the negative of their sum.
        """
        attack_free = params[: n - 1]
        attack = np.concatenate([attack_free, [-attack_free.sum()]])
        defence = params[n - 1: 2 * n - 1]
        home_adv = params[-2]
        rho = params[-1]
        return attack, defence, home_adv, rho

    def _neg_log_likelihood(self, params, home_idx, away_idx, hg, ag, weights, n):
        attack, defence, home_adv, rho = self._unpack(params, n)
        lam = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
        mu = np.exp(attack[away_idx] + defence[home_idx])
        # Poisson log-pmf computed directly (avoids per-row scipy overhead).
        log_pois = (hg * np.log(lam) - lam - gammaln(hg + 1.0)) + \
                   (ag * np.log(mu) - mu - gammaln(ag + 1.0))
        tau = self._tau(hg, ag, lam, mu, rho)
        # Guard against the correction driving a probability non-positive.
        tau = np.clip(tau, 1e-12, None)
        ll = log_pois + np.log(tau)
        return -np.sum(weights * ll)

    # ------------------------------------------------------------------ #
    # Fitting                                                            #
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, decay: float = 0.0,
            ref_date: Optional[pd.Timestamp] = None) -> "DixonColesModel":
        """Fit by maximum likelihood on the supplied matches.

        df must contain HomeTeam, AwayTeam, FTHG, FTAG. If `decay` > 0 a
        match played `t` days before `ref_date` is weighted exp(-decay * t),
        down-weighting older results (Dixon-Coles time weighting). `decay`
        is per-day; ~0.0018 gives a half-life of about one year.
        """
        required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"fit() is missing required columns: {sorted(missing)}")

        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        if df.empty:
            raise ValueError("fit() received no usable rows after dropping NaNs.")

        self.teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.team_index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        home_idx = df["HomeTeam"].map(self.team_index).to_numpy()
        away_idx = df["AwayTeam"].map(self.team_index).to_numpy()
        hg = df["FTHG"].to_numpy(dtype=float)
        ag = df["FTAG"].to_numpy(dtype=float)

        # Time-decay weights.
        if decay and decay > 0 and "Date" in df.columns:
            ref = ref_date if ref_date is not None else df["Date"].max()
            age_days = (pd.Timestamp(ref) - df["Date"]).dt.days.to_numpy(dtype=float)
            age_days = np.clip(age_days, 0, None)
            weights = np.exp(-decay * age_days)
        else:
            weights = np.ones(len(df), dtype=float)

        # Parameter vector: [attack_free (n-1), defence (n), home_adv, rho].
        x0 = np.concatenate([
            np.zeros(n - 1),     # attack_free
            np.zeros(n),         # defence
            [0.25],              # home advantage (typical log-scale value)
            [-0.05],             # rho (typically slightly negative)
        ])
        bounds = (
            [(-3, 3)] * (n - 1) +    # attack_free
            [(-3, 3)] * n +          # defence
            [(-1, 1)] +              # home_adv
            [(-0.2, 0.2)]            # rho (kept small so tau stays positive)
        )

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, hg, ag, weights, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-7},
        )

        attack, defence, home_adv, rho = self._unpack(result.x, n)
        self.attack = attack
        self.defence = defence
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Prediction                                                         #
    # ------------------------------------------------------------------ #
    def knows_team(self, team: str) -> bool:
        return self.fitted and team in self.team_index

    def _rates(self, home: str, away: str):
        i, j = self.team_index[home], self.team_index[away]
        lam = float(np.exp(self.attack[i] + self.defence[j] + self.home_adv))
        mu = float(np.exp(self.attack[j] + self.defence[i]))
        return lam, mu

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """Return P(home goals = x, away goals = y) as a matrix M[x, y]."""
        if not self.knows_team(home) or not self.knows_team(away):
            raise ValueError(f"Unseen team(s): {home!r} / {away!r}. Fit on prior matches first.")
        lam, mu = self._rates(home, away)
        rng = np.arange(0, self.max_goals + 1)
        home_pmf = poisson.pmf(rng, lam)
        away_pmf = poisson.pmf(rng, mu)
        m = np.outer(home_pmf, away_pmf)
        # Apply the Dixon-Coles correction to the four low-score cells.
        m[0, 0] *= 1.0 - lam * mu * self.rho
        m[0, 1] *= 1.0 + lam * self.rho
        m[1, 0] *= 1.0 + mu * self.rho
        m[1, 1] *= 1.0 - self.rho
        m = np.clip(m, 0.0, None)
        total = m.sum()
        if total <= 0:
            raise ValueError("Degenerate score matrix (sum <= 0).")
        return m / total

    def predict_proba(self, home: str, away: str, ou_line: float = 2.5) -> Dict[str, float]:
        """1X2 and over/under probabilities for a single fixture."""
        m = self.score_matrix(home, away)
        home_win = float(np.tril(m, -1).sum())   # x > y
        draw = float(np.trace(m))                 # x == y
        away_win = float(np.triu(m, 1).sum())     # x < y

        size = m.shape[0]
        xs = np.arange(size)[:, None]
        ys = np.arange(size)[None, :]
        totals = xs + ys
        over = float(m[totals > ou_line].sum())
        under = float(m[totals < ou_line].sum())

        return {
            "H": home_win,
            "D": draw,
            "A": away_win,
            f"over_{ou_line}": over,
            f"under_{ou_line}": under,
        }


# ---------------------------------------------------------------------- #
# Value detection                                                        #
# ---------------------------------------------------------------------- #
@dataclass
class ValueBet:
    outcome: str          # e.g. "H", "D", "A" (or "over_2.5")
    model_prob: float     # model's probability for the outcome
    implied_prob: float   # bookmaker implied probability (1 / odds, incl. margin)
    odds: float           # decimal odds offered
    edge: float           # model_prob - implied_prob
    ev: float             # expected profit per 1-unit stake = model_prob * odds - 1

    def as_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "model_prob": self.model_prob,
            "implied_prob": self.implied_prob,
            "odds": self.odds,
            "edge": self.edge,
            "ev": self.ev,
        }


def find_value(model_probs: Dict[str, float],
               odds: Dict[str, float],
               threshold: float = 0.05) -> List[ValueBet]:
    """Flag outcomes where the model probability beats the bookmaker's.

    edge = model_prob - (1 / decimal_odds). Using the *raw* implied
    probability (which includes the bookmaker margin) keeps this honest:
    a positive edge under this definition is exactly a positive-EV bet at
    the offered price (model_prob * odds > 1).

    Only outcomes present in both dicts with valid odds (> 1.0) are
    considered. Returns the qualifying bets sorted by descending edge.
    """
    bets: List[ValueBet] = []
    for outcome, price in odds.items():
        if outcome not in model_probs:
            continue
        if price is None or not np.isfinite(price) or price <= 1.0:
            continue
        p = model_probs[outcome]
        implied = 1.0 / price
        edge = p - implied
        if edge >= threshold:
            bets.append(ValueBet(
                outcome=outcome,
                model_prob=p,
                implied_prob=implied,
                odds=float(price),
                edge=edge,
                ev=p * price - 1.0,
            ))
    bets.sort(key=lambda b: b.edge, reverse=True)
    return bets


if __name__ == "__main__":
    # Tiny smoke test on synthetic data so the module is runnable standalone.
    rng = np.random.default_rng(0)
    teams = [f"Team{i}" for i in range(6)]
    strength = {t: rng.normal(0, 0.4) for t in teams}
    rows = []
    for _ in range(300):
        h, a = rng.choice(teams, size=2, replace=False)
        lam = np.exp(0.2 + strength[h] - strength[a] + 0.1)
        mu = np.exp(strength[a] - strength[h])
        rows.append({
            "HomeTeam": h, "AwayTeam": a,
            "FTHG": rng.poisson(lam), "FTAG": rng.poisson(mu),
        })
    df = pd.DataFrame(rows)
    model = DixonColesModel().fit(df)
    print("home advantage:", round(model.home_adv, 3), "rho:", round(model.rho, 3))
    probs = model.predict_proba(teams[0], teams[1])
    print("sample 1X2:", {k: round(v, 3) for k, v in probs.items()})
    print("sum 1X2:", round(probs["H"] + probs["D"] + probs["A"], 6))
    demo_odds = {"H": 2.10, "D": 3.40, "A": 3.60}
    for vb in find_value(probs, demo_odds, threshold=0.02):
        print("value:", vb.as_dict())
