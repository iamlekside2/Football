"""
ml_model.py -- gradient-boosted (XGBoost) football match-outcome predictor.

This is the "machine learning" engine for the prediction site. It engineers
pre-match features with **no lookahead** (every feature for a match is computed
from information available strictly before kick-off), trains a 3-class
Home / Draw / Away classifier, and -- because honesty is the whole ethos of
this project -- evaluates itself out-of-sample against the bookmaker's closing
line so we never overstate how good it is.

Features engineered per match (home & away):
  * Elo rating (carried across seasons with regression to the mean)
  * Recent form: points from the last N matches (overall + venue-specific)
  * Rolling goals for / against
  * Rolling shots-on-target for / against (proxy for underlying quality)
  * Rest days since the team's previous match
  * Matches played so far (a confidence signal early in a season)

Run standalone for an honest report:
    python ml_model.py
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest import _read_csv, _parse_dates, _pick_odds_columns
from diagnostics import URL_TEMPLATE, DEFAULT_SEASONS

OUTCOME_TO_CLASS = {"H": 0, "D": 1, "A": 2}
CLASS_TO_OUTCOME = {0: "H", 1: "D", 2: "A"}

# Competitions we support. Each is its own model. Two data formats:
#   "main"  -> mmz4281/{season}/{code}.csv  (European leagues, has shots)
#   "extra" -> new/{file}.csv               (rest of world, no shots)
# `api_names` are the league names as API-Football reports them (for live
# fixture matching); defaults to [name] when omitted.
LEAGUES = {
    # --- European, main format ---
    "E0":  {"name": "Premier League", "country": "England", "fmt": "main"},
    "E1":  {"name": "Championship", "country": "England", "fmt": "main"},
    "SP1": {"name": "La Liga", "country": "Spain", "fmt": "main"},
    "D1":  {"name": "Bundesliga", "country": "Germany", "fmt": "main"},
    "I1":  {"name": "Serie A", "country": "Italy", "fmt": "main"},
    "F1":  {"name": "Ligue 1", "country": "France", "fmt": "main"},
    "N1":  {"name": "Eredivisie", "country": "Netherlands", "fmt": "main"},
    "P1":  {"name": "Primeira Liga", "country": "Portugal", "fmt": "main"},
    "B1":  {"name": "Jupiler Pro League", "country": "Belgium", "fmt": "main",
            "api_names": ["Jupiler Pro League", "First Division A", "Pro League"]},
    "T1":  {"name": "Super Lig", "country": "Turkey", "fmt": "main",
            "api_names": ["Süper Lig", "Super Lig"]},
    # --- Rest of world, extra format (in-season during the European summer) ---
    "BRA": {"name": "Serie A", "country": "Brazil", "fmt": "extra",
            "file": "BRA", "match": "Serie A", "api_names": ["Serie A", "Brasileirao"]},
    "USA": {"name": "MLS", "country": "USA", "fmt": "extra",
            "file": "USA", "match": "MLS", "api_names": ["Major League Soccer", "MLS"]},
    "NOR": {"name": "Eliteserien", "country": "Norway", "fmt": "extra",
            "file": "NOR", "match": "Eliteserien"},
    "SWE": {"name": "Allsvenskan", "country": "Sweden", "fmt": "extra",
            "file": "SWE", "match": "Allsvenskan"},
    "MEX": {"name": "Liga MX", "country": "Mexico", "fmt": "extra",
            "file": "MEX", "match": "Liga MX", "api_names": ["Liga MX"]},
    "ARG": {"name": "Liga Profesional", "country": "Argentina", "fmt": "extra",
            "file": "ARG", "match": "Liga Profesional",
            "api_names": ["Liga Profesional Argentina", "Primera Division"]},
}
LEAGUE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
EXTRA_URL = "https://www.football-data.co.uk/new/{file}.csv"

FEATURES = [
    "elo_diff", "elo_home", "elo_away",
    "form_home", "form_away",
    "venue_form_home", "venue_form_away",
    "gf_home", "ga_home", "gf_away", "ga_away",
    "sot_home", "sota_home", "sot_away", "sota_away",
    "rest_home", "rest_away",
    "played_home", "played_away",
]

FORM_N = 6          # rolling window length
ELO_K = 20.0        # Elo update rate
ELO_HOME_ADV = 65.0 # Elo home-advantage points
ELO_REGRESS = 0.75  # between-season regression toward 1500


# ---------------------------------------------------------------------- #
# Data loading (richer than backtest.load_season: keeps shots + FTR)     #
# ---------------------------------------------------------------------- #
_MATCH_CACHE: Dict[tuple, pd.DataFrame] = {}


def _finalise(out: pd.DataFrame) -> pd.DataFrame:
    out = out.dropna(subset=["Date", "FTHG", "FTAG"]).copy()
    out["FTHG"] = out["FTHG"].astype(int)
    out["FTAG"] = out["FTAG"].astype(int)
    out["FTR"] = np.where(out["FTHG"] > out["FTAG"], "H",
                          np.where(out["FTHG"] < out["FTAG"], "A", "D"))
    return out


def _load_main(season_codes: List[str], league: str, verbose: bool) -> pd.DataFrame:
    frames = []
    for code in season_codes:
        try:
            raw = _read_csv(LEAGUE_URL.format(code=code, league=league), None)
        except Exception as exc:
            if verbose:
                print(f"  load failed for {league} {code}: {exc}")
            continue
        if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(raw.columns):
            continue
        try:
            h, d, a = _pick_odds_columns(raw.columns)
        except ValueError:
            h = d = a = None

        def col(name):
            return pd.to_numeric(raw[name], errors="coerce") if name in raw.columns else np.nan

        frames.append(pd.DataFrame({
            "Date": _parse_dates(raw["Date"]),
            "season": code, "league": league,
            "HomeTeam": raw["HomeTeam"].astype(str).str.strip(),
            "AwayTeam": raw["AwayTeam"].astype(str).str.strip(),
            "FTHG": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "FTAG": pd.to_numeric(raw["FTAG"], errors="coerce"),
            "HST": col("HST"), "AST": col("AST"),
            "odds_H": pd.to_numeric(raw[h], errors="coerce") if h else np.nan,
            "odds_D": pd.to_numeric(raw[d], errors="coerce") if d else np.nan,
            "odds_A": pd.to_numeric(raw[a], errors="coerce") if a else np.nan,
        }))
    if not frames:
        raise RuntimeError(f"No data for league {league}.")
    out = _finalise(pd.concat(frames, ignore_index=True))
    rank = {c: i for i, c in enumerate(season_codes)}
    out["_r"] = out["season"].map(rank).fillna(0)
    return out.sort_values(["_r", "Date"]).drop(columns="_r").reset_index(drop=True)


def _load_extra(info: dict, league: str, n_seasons: int) -> pd.DataFrame:
    """Rest-of-world format: new/{file}.csv with Country/League/Season/Home/...
    No shots columns; the model falls back to neutral shot priors."""
    raw = _read_csv(EXTRA_URL.format(file=info["file"]), None)
    raw["League"] = raw["League"].astype(str).str.strip()
    sub = raw[raw["League"] == info["match"]].copy()
    if sub.empty:
        raise RuntimeError(f"No rows for {league} ({info['match']}).")
    sub["Season"] = sub["Season"].astype(str)
    keep = set(sorted(sub["Season"].unique())[-n_seasons:])
    sub = sub[sub["Season"].isin(keep)]

    def oc(pref, alt):
        s = pd.to_numeric(sub[pref], errors="coerce") if pref in sub else pd.Series(np.nan, index=sub.index)
        if alt in sub:
            s = s.fillna(pd.to_numeric(sub[alt], errors="coerce"))
        return s

    out = _finalise(pd.DataFrame({
        "Date": _parse_dates(sub["Date"]),
        "season": sub["Season"].values, "league": league,
        "HomeTeam": sub["Home"].astype(str).str.strip(),
        "AwayTeam": sub["Away"].astype(str).str.strip(),
        "FTHG": pd.to_numeric(sub["HG"], errors="coerce"),
        "FTAG": pd.to_numeric(sub["AG"], errors="coerce"),
        "HST": np.nan, "AST": np.nan,
        "odds_H": oc("B365CH", "AvgCH"), "odds_D": oc("B365CD", "AvgCD"),
        "odds_A": oc("B365CA", "AvgCA"),
    }))
    return out.sort_values("Date").reset_index(drop=True)


def load_matches(season_codes: List[str], league: str = "E0",
                 verbose: bool = False) -> pd.DataFrame:
    """Load matches for one competition. `league` is a key in LEAGUES.
    Cached per (season-codes, league)."""
    cache_key = (tuple(season_codes), league)
    if cache_key in _MATCH_CACHE:
        return _MATCH_CACHE[cache_key].copy()
    info = LEAGUES.get(league, {"fmt": "main"})
    if info.get("fmt") == "extra":
        out = _load_extra(info, league, n_seasons=max(2, len(season_codes)))
    else:
        out = _load_main(season_codes, league, verbose)
    _MATCH_CACHE[cache_key] = out
    return out.copy()


# ---------------------------------------------------------------------- #
# Per-team rolling state                                                 #
# ---------------------------------------------------------------------- #
@dataclass
class TeamState:
    elo: float = 1500.0
    form: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    home_form: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    away_form: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    gf: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    ga: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    sot: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    sota: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    last_date: Optional[pd.Timestamp] = None
    played: int = 0
    last_season: Optional[str] = None


def _avg(d: Deque[float], default: float) -> float:
    return float(np.mean(d)) if len(d) else default


def _rest_days(state: TeamState, date: pd.Timestamp) -> float:
    if state.last_date is None:
        return 7.0
    return float(min((date - state.last_date).days, 21))


# ---------------------------------------------------------------------- #
# The model                                                              #
# ---------------------------------------------------------------------- #
class FootballMLModel:
    def __init__(self, n_estimators: int = 350, max_depth: int = 4,
                 learning_rate: float = 0.04):
        self.params = dict(
            objective="multi:softprob", num_class=3, n_estimators=n_estimators,
            max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            reg_lambda=1.5, eval_metric="mlogloss", n_jobs=4,
        )
        self.model = None
        self.state: Dict[str, TeamState] = {}
        self.teams: List[str] = []

    # -- feature engineering ------------------------------------------- #
    def _new_season_regress(self, st: TeamState, season: str):
        if st.last_season is not None and season != st.last_season:
            st.elo = 1500.0 + ELO_REGRESS * (st.elo - 1500.0)
        st.last_season = season

    def _row_features(self, hs: TeamState, as_: TeamState, date) -> Dict[str, float]:
        return {
            "elo_home": hs.elo, "elo_away": as_.elo,
            "elo_diff": (hs.elo + ELO_HOME_ADV) - as_.elo,
            "form_home": _avg(hs.form, 1.3), "form_away": _avg(as_.form, 1.3),
            "venue_form_home": _avg(hs.home_form, 1.5),
            "venue_form_away": _avg(as_.away_form, 1.1),
            "gf_home": _avg(hs.gf, 1.3), "ga_home": _avg(hs.ga, 1.3),
            "gf_away": _avg(as_.gf, 1.1), "ga_away": _avg(as_.ga, 1.4),
            "sot_home": _avg(hs.sot, 4.5), "sota_home": _avg(hs.sota, 4.5),
            "sot_away": _avg(as_.sot, 4.0), "sota_away": _avg(as_.sota, 4.5),
            "rest_home": _rest_days(hs, date), "rest_away": _rest_days(as_, date),
            "played_home": float(hs.played), "played_away": float(as_.played),
        }

    def _update(self, hs: TeamState, as_: TeamState, row):
        """Update both teams' state AFTER a match (so features stay pre-match)."""
        hg, ag = int(row.FTHG), int(row.FTAG)
        # Elo
        exp_h = 1.0 / (1.0 + 10 ** ((as_.elo - (hs.elo + ELO_HOME_ADV)) / 400.0))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 + (gd - 3) / 8.0))
        delta = ELO_K * mult * (s_h - exp_h)
        hs.elo += delta
        as_.elo -= delta
        # Points / form
        ph = 3 if hg > ag else (1 if hg == ag else 0)
        pa = 3 if ag > hg else (1 if hg == ag else 0)
        hs.form.append(ph); as_.form.append(pa)
        hs.home_form.append(ph); as_.away_form.append(pa)
        # Goals
        hs.gf.append(hg); hs.ga.append(ag)
        as_.gf.append(ag); as_.ga.append(hg)
        # Shots on target (if present)
        if not pd.isna(row.HST): hs.sot.append(row.HST); as_.sota.append(row.HST)
        if not pd.isna(row.AST): as_.sot.append(row.AST); hs.sota.append(row.AST)
        # Bookkeeping
        hs.last_date = as_.last_date = row.Date
        hs.played += 1; as_.played += 1

    def engineer(self, df: pd.DataFrame, reset_state: bool = True):
        """Return (X DataFrame, y array, meta DataFrame) with no lookahead."""
        if reset_state:
            self.state = defaultdict(TeamState)
        rows, ys, meta = [], [], []
        for row in df.itertuples(index=False):
            hs = self.state[row.HomeTeam]
            as_ = self.state[row.AwayTeam]
            self._new_season_regress(hs, row.season)
            self._new_season_regress(as_, row.season)
            feats = self._row_features(hs, as_, row.Date)
            rows.append(feats)
            ys.append(OUTCOME_TO_CLASS[row.FTR])
            meta.append({"Date": row.Date, "HomeTeam": row.HomeTeam,
                         "AwayTeam": row.AwayTeam, "FTR": row.FTR,
                         "odds_H": row.odds_H, "odds_D": row.odds_D, "odds_A": row.odds_A})
            self._update(hs, as_, row)
        X = pd.DataFrame(rows, columns=FEATURES)
        return X, np.array(ys), pd.DataFrame(meta)

    # -- training / prediction ----------------------------------------- #
    def fit(self, df: pd.DataFrame):
        import xgboost as xgb
        X, y, _ = self.engineer(df, reset_state=True)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
        self.teams = sorted(self.state.keys())
        return self

    def knows_team(self, team: str) -> bool:
        return team in self.state

    def predict_teams(self, home: str, away: str,
                      ref_date: Optional[pd.Timestamp] = None) -> Dict[str, float]:
        """Predict using each team's CURRENT state (post-training)."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        hs = self.state.get(home, TeamState())
        as_ = self.state.get(away, TeamState())
        date = ref_date if ref_date is not None else (
            max([s.last_date for s in (hs, as_) if s.last_date is not None],
                default=pd.Timestamp("2000-01-01")))
        feats = self._row_features(hs, as_, date)
        X = pd.DataFrame([feats], columns=FEATURES)
        proba = self.model.predict_proba(X)[0]
        return {"H": float(proba[0]), "D": float(proba[1]), "A": float(proba[2])}


# ---------------------------------------------------------------------- #
# Honest evaluation                                                      #
# ---------------------------------------------------------------------- #
def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _logloss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(np.mean(-np.log(p[np.arange(len(y)), y])))


def _book_probs(meta: pd.DataFrame) -> Optional[np.ndarray]:
    if meta[["odds_H", "odds_D", "odds_A"]].isna().any().any():
        return None
    raw = 1.0 / meta[["odds_H", "odds_D", "odds_A"]].to_numpy()
    return raw / raw.sum(axis=1, keepdims=True)


def evaluate(season_codes: List[str] = None, test_n: int = 2) -> dict:
    """Train on earlier seasons, test out-of-sample on the last `test_n`."""
    import xgboost as xgb
    season_codes = season_codes or list(DEFAULT_SEASONS)
    if len(season_codes) <= test_n:
        test_n = 1
    train_codes = season_codes[:-test_n]
    test_codes = season_codes[-test_n:]
    print(f"Train seasons: {train_codes}   Test (out-of-sample): {test_codes}\n")

    full = load_matches(season_codes, verbose=True)
    model = FootballMLModel()
    # Engineer over the FULL chronological history (state must roll through
    # train into test), then split by season for an honest holdout.
    X, y, meta = model.engineer(full, reset_state=True)
    test_mask = full["season"].isin(test_codes).to_numpy()
    train_mask = ~test_mask

    clf = xgb.XGBClassifier(**model.params)
    clf.fit(X[train_mask], y[train_mask])
    proba = clf.predict_proba(X[test_mask])
    y_test = y[test_mask]
    meta_test = meta[test_mask].reset_index(drop=True)

    acc = float((proba.argmax(axis=1) == y_test).mean())
    ml_brier, ml_ll = _brier(proba, y_test), _logloss(proba, y_test)

    book = _book_probs(meta_test)
    print("Out-of-sample results on the held-out season(s):")
    print(f"  test matches:      {len(y_test)}")
    print(f"  ML accuracy:       {acc * 100:.1f}%")
    print(f"  ML Brier:          {ml_brier:.4f}   (lower better)")
    print(f"  ML log-loss:       {ml_ll:.4f}")
    # Always-home baseline accuracy
    print(f"  baseline acc (all-home): {(y_test == 0).mean() * 100:.1f}%")
    if book is not None:
        bk_brier, bk_ll = _brier(book, y_test), _logloss(book, y_test)
        bk_acc = float((book.argmax(axis=1) == y_test).mean())
        print(f"\n  Bookmaker (no-vig) accuracy: {bk_acc * 100:.1f}%")
        print(f"  Bookmaker Brier:   {bk_brier:.4f}")
        print(f"  Bookmaker log-loss:{bk_ll:.4f}")
        verdict = ("The book is still better calibrated (expected) -- but check how "
                   "close the ML model gets." if bk_brier < ml_brier
                   else "The ML model edges the book on Brier here -- verify carefully.")
        print(f"\n  -> {verdict}")
    return {"accuracy": acc, "brier": ml_brier, "log_loss": ml_ll,
            "n_test": int(len(y_test))}


if __name__ == "__main__":
    evaluate()
