"""
worldcup_model.py -- national-team match predictor for the 2026 World Cup.

Trains on the full history of international football (the open
`martj42/international_results` dataset, 1872-present) using an Elo rating
that accounts for neutral venues and match importance, plus recent form, fed
into an XGBoost Win/Draw/Win classifier.

Conveniently, that dataset also carries the *upcoming* 2026 World Cup fixtures
(tournament = "FIFA World Cup", scores still blank), so we get the real
schedule for free with team names that already match our training data.

Run standalone for an honest out-of-sample report:
    python worldcup_model.py
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

OUTCOME_TO_CLASS = {"H": 0, "D": 1, "A": 2}
CLASS_TO_OUTCOME = {0: "H", 1: "D", 2: "A"}

FEATURES = [
    "elo_diff", "elo_home", "elo_away",
    "form_home", "form_away",
    "gf_home", "ga_home", "gf_away", "ga_away",
    "neutral", "tournament",
]

FORM_N = 8
ELO_K = 30.0
ELO_HOME_ADV = 65.0      # applied only when the match is NOT neutral
TRAIN_SINCE = "2002-01-01"   # modern-era window for the classifier
TEST_SINCE = "2023-01-01"    # out-of-sample holdout for evaluation

_CACHE: Dict[str, pd.DataFrame] = {}


def load_international(verbose: bool = False) -> pd.DataFrame:
    if "df" in _CACHE:
        return _CACHE["df"].copy()
    raw = pd.read_csv(RESULTS_URL)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "home_team", "away_team"]).reset_index(drop=True)
    raw["neutral"] = raw["neutral"].astype(bool)
    if verbose:
        print(f"Loaded {len(raw)} international matches "
              f"({raw['date'].min().date()} -> {raw['date'].max().date()})")
    _CACHE["df"] = raw
    return raw.copy()


def _tourn_weight(name: str) -> float:
    n = str(name).lower()
    if "friendly" in n:
        return 1.0
    if "qualification" in n or "qualifier" in n:
        return 1.5
    # World Cup, Euro, Copa America, AFCON, Nations League finals, etc.
    return 2.0


def _is_tournament(name: str) -> float:
    return 0.0 if "friendly" in str(name).lower() else 1.0


@dataclass
class NState:
    elo: float = 1500.0
    form: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    gf: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    ga: Deque[float] = field(default_factory=lambda: deque(maxlen=FORM_N))
    played: int = 0


def _avg(d: Deque[float], default: float) -> float:
    return float(np.mean(d)) if len(d) else default


class WorldCupModel:
    def __init__(self, n_estimators: int = 400, max_depth: int = 4,
                 learning_rate: float = 0.04):
        self.params = dict(
            objective="multi:softprob", num_class=3, n_estimators=n_estimators,
            max_depth=max_depth, learning_rate=learning_rate, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=4, reg_lambda=1.5,
            eval_metric="mlogloss", n_jobs=4,
        )
        self.model = None
        self.state: Dict[str, NState] = {}

    # -- features ------------------------------------------------------ #
    def _features(self, hs: NState, as_: NState, neutral: bool, tourn: float) -> Dict[str, float]:
        elo_h = hs.elo + (0.0 if neutral else ELO_HOME_ADV)
        return {
            "elo_home": hs.elo, "elo_away": as_.elo, "elo_diff": elo_h - as_.elo,
            "form_home": _avg(hs.form, 1.3), "form_away": _avg(as_.form, 1.3),
            "gf_home": _avg(hs.gf, 1.2), "ga_home": _avg(hs.ga, 1.2),
            "gf_away": _avg(as_.gf, 1.2), "ga_away": _avg(as_.ga, 1.2),
            "neutral": 1.0 if neutral else 0.0, "tournament": tourn,
        }

    def _update(self, hs: NState, as_: NState, hg: int, ag: int,
                neutral: bool, tournament: str):
        elo_h = hs.elo + (0.0 if neutral else ELO_HOME_ADV)
        exp_h = 1.0 / (1.0 + 10 ** ((as_.elo - elo_h) / 400.0))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        gd = abs(hg - ag)
        gd_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 + (gd - 3) / 8.0))
        delta = ELO_K * _tourn_weight(tournament) * gd_mult * (s_h - exp_h)
        hs.elo += delta
        as_.elo -= delta
        ph = 3 if hg > ag else (1 if hg == ag else 0)
        pa = 3 if ag > hg else (1 if hg == ag else 0)
        hs.form.append(ph); as_.form.append(pa)
        hs.gf.append(hg); hs.ga.append(ag)
        as_.gf.append(ag); as_.ga.append(hg)
        hs.played += 1; as_.played += 1

    @staticmethod
    def _result(hg: float, ag: float) -> str:
        return "H" if hg > ag else ("A" if hg < ag else "D")

    def engineer(self, df: pd.DataFrame, train_since: str = TRAIN_SINCE):
        self.state = defaultdict(NState)
        df = df.sort_values("date")
        since = pd.Timestamp(train_since)
        rows, ys, meta = [], [], []
        for r in df.itertuples(index=False):
            completed = pd.notna(r.home_score) and pd.notna(r.away_score)
            hs = self.state[r.home_team]
            as_ = self.state[r.away_team]
            feats = self._features(hs, as_, bool(r.neutral), _is_tournament(r.tournament))
            if completed and r.date >= since:
                rows.append(feats)
                ys.append(OUTCOME_TO_CLASS[self._result(r.home_score, r.away_score)])
                meta.append({"date": r.date, "home": r.home_team, "away": r.away_team,
                             "tournament": r.tournament})
            if completed:
                self._update(hs, as_, int(r.home_score), int(r.away_score),
                             bool(r.neutral), r.tournament)
        return pd.DataFrame(rows, columns=FEATURES), np.array(ys), pd.DataFrame(meta)

    def fit(self, df: pd.DataFrame):
        import xgboost as xgb
        X, y, _ = self.engineer(df)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def knows_team(self, team: str) -> bool:
        return team in self.state

    @property
    def teams(self) -> List[str]:
        return sorted(self.state.keys())

    def predict(self, home: str, away: str, neutral: bool = True) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        hs = self.state.get(home, NState())
        as_ = self.state.get(away, NState())
        feats = self._features(hs, as_, neutral, 1.0)
        proba = self.model.predict_proba(pd.DataFrame([feats], columns=FEATURES))[0]
        return {"H": float(proba[0]), "D": float(proba[1]), "A": float(proba[2])}

    def elo_of(self, team: str) -> Optional[float]:
        st = self.state.get(team)
        return round(st.elo, 0) if st else None


# ---------------------------------------------------------------------- #
# Upcoming World Cup fixtures (already in the dataset, scores blank)      #
# ---------------------------------------------------------------------- #
def upcoming_world_cup(df: pd.DataFrame, today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    today = today or pd.Timestamp.today().normalize()
    wc = df[(df["tournament"] == "FIFA World Cup")
            & (df["home_score"].isna())
            & (df["date"] >= today)].copy()
    return wc.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------- #
# Honest evaluation                                                      #
# ---------------------------------------------------------------------- #
def _brier(p, y):
    oh = np.eye(3)[y]
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def _logloss(p, y):
    p = np.clip(p, 1e-12, 1.0)
    return float(np.mean(-np.log(p[np.arange(len(y)), y])))


def evaluate():
    import xgboost as xgb
    df = load_international(verbose=True)
    model = WorldCupModel()
    X, y, meta = model.engineer(df)
    test_mask = (meta["date"] >= pd.Timestamp(TEST_SINCE)).to_numpy()
    train_mask = ~test_mask
    print(f"\nTrain matches: {train_mask.sum()}   Test (>= {TEST_SINCE}): {test_mask.sum()}")

    clf = xgb.XGBClassifier(**model.params)
    clf.fit(X[train_mask], y[train_mask])
    proba = clf.predict_proba(X[test_mask])
    y_test = y[test_mask]

    acc = float((proba.argmax(axis=1) == y_test).mean())
    # Baseline: always predict the home/first team to win.
    base = float((y_test == 0).mean())
    print("\nOut-of-sample (international matches since 2023):")
    print(f"  accuracy:      {acc * 100:.1f}%")
    print(f"  baseline(home):{base * 100:.1f}%")
    print(f"  Brier:         {_brier(proba, y_test):.4f}")
    print(f"  log-loss:      {_logloss(proba, y_test):.4f}")

    # Show a few upcoming WC predictions.
    model.fit(df)
    wc = upcoming_world_cup(df)
    print(f"\nUpcoming World Cup fixtures found: {len(wc)}")
    for r in wc.head(6).itertuples(index=False):
        p = model.predict(r.home_team, r.away_team, neutral=bool(r.neutral))
        pick = CLASS_TO_OUTCOME[int(np.argmax([p['H'], p['D'], p['A']]))]
        winner = r.home_team if pick == "H" else (r.away_team if pick == "A" else "Draw")
        print(f"  {r.date.date()}  {r.home_team} vs {r.away_team}: "
              f"{p['H']*100:.0f}/{p['D']*100:.0f}/{p['A']*100:.0f}  -> {winner}")


if __name__ == "__main__":
    evaluate()
