"""
prediction_service.py -- ties the ML classifier and Dixon-Coles together
into the predictions the website serves, with heavy work cached in-process.

* Head-to-head / current predictions use a model trained on ALL history
  (most up-to-date team state).
* The fixtures board and the model-quality (transparency) numbers come from an
  honest out-of-sample holdout: train on earlier seasons, predict the latest
  season's matches using only pre-match information.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import difflib
import re

from diagnostics import DEFAULT_SEASONS
from poisson_model import DixonColesModel
from ml_model import (
    FootballMLModel, load_matches, FEATURES, LEAGUES,
    _brier, _logloss, _book_probs, CLASS_TO_OUTCOME, FORM_N,
)
import fixtures_provider
import worldcup_model as wc_mod
from worldcup_model import (
    WorldCupModel, load_international, upcoming_world_cup,
    _brier as _wc_brier, _logloss as _wc_logloss,
)

# ---- module-level caches (populated lazily on first request) ---------- #
_FULL: Optional[dict] = None       # model trained on all seasons + DC + data
_HOLDOUT: Optional[dict] = None     # out-of-sample board + quality metrics

HOLDOUT_TEST = DEFAULT_SEASONS[-1:]  # latest season is the holdout


# ---------------------------------------------------------------------- #
# Full-history model (for current head-to-head predictions)              #
# ---------------------------------------------------------------------- #
def _get_full() -> dict:
    global _FULL
    if _FULL is None:
        df = load_matches(DEFAULT_SEASONS, verbose=False)
        ml = FootballMLModel().fit(df)
        # Dixon-Coles on the two most recent seasons -> current-ish scorelines.
        recent = df[df["season"].isin(DEFAULT_SEASONS[-2:])]
        dc = DixonColesModel().fit(recent)
        _FULL = {"df": df, "ml": ml, "dc": dc}
    return _FULL


def _team_snapshot(ml: FootballMLModel, team: str) -> Optional[dict]:
    st = ml.state.get(team)
    if st is None:
        return None
    return {
        "elo": round(st.elo, 1),
        "form_ppg": round(float(np.mean(st.form)), 2) if len(st.form) else None,
        "played": st.played,
    }


def _assemble(ml: FootballMLModel, dc, home: str, away: str) -> dict:
    if not ml.knows_team(home) or not ml.knows_team(away):
        raise ValueError(f"Unknown team(s). Known teams: {ml.teams}")
    wdl = ml.predict_teams(home, away)
    scoreline = score_prob = over25 = under25 = None
    if dc is not None and dc.knows_team(home) and dc.knows_team(away):
        mat = dc.score_matrix(home, away)
        i, j = np.unravel_index(int(np.argmax(mat)), mat.shape)
        scoreline, score_prob = f"{i}-{j}", float(mat[i, j])
        dcp = dc.predict_proba(home, away)
        over25, under25 = dcp.get("over_2.5"), dcp.get("under_2.5")
    return {
        "home": home, "away": away, "wdl": wdl,
        "pick": CLASS_TO_OUTCOME[int(np.argmax([wdl["H"], wdl["D"], wdl["A"]]))],
        "scoreline": scoreline, "scoreline_prob": score_prob,
        "over_2_5": over25, "under_2_5": under25,
        "home_snapshot": _team_snapshot(ml, home),
        "away_snapshot": _team_snapshot(ml, away),
    }


def predict_match(home: str, away: str) -> dict:
    full = _get_full()
    return _assemble(full["ml"], full["dc"], home, away)


def predict_league(code: str, home: str, away: str) -> dict:
    if code not in LEAGUES:
        raise ValueError(f"Unknown league {code}.")
    reg = _get_league_model(code)
    out = _assemble(reg["ml"], reg.get("dc"), home, away)
    out["league"] = reg["name"]
    out["country"] = reg["country"]
    return out


def list_teams() -> List[str]:
    return _get_full()["ml"].teams


# ---------------------------------------------------------------------- #
# Honest out-of-sample holdout (board + transparency metrics)            #
# ---------------------------------------------------------------------- #
def _reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> List[dict]:
    # Pool the three class probabilities vs whether that class occurred.
    p = probs.reshape(-1)
    onehot = np.eye(3)[y].reshape(-1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(mask.sum()),
            "mean_pred": float(p[mask].mean()),
            "observed": float(onehot[mask].mean()),
            "gap": float(p[mask].mean() - onehot[mask].mean()),
        })
    return rows


def _build_holdout() -> dict:
    import xgboost as xgb
    df = load_matches(DEFAULT_SEASONS, verbose=False)
    model = FootballMLModel()
    X, y, meta = model.engineer(df, reset_state=True)
    test_mask = df["season"].isin(HOLDOUT_TEST).to_numpy()
    train_mask = ~test_mask

    clf = xgb.XGBClassifier(**model.params)
    clf.fit(X[train_mask], y[train_mask])
    proba = clf.predict_proba(X[test_mask])
    y_test = y[test_mask]
    meta_test = meta[test_mask].reset_index(drop=True)

    # Per-match board records (most recent first).
    matches = []
    for k in range(len(meta_test)):
        row = meta_test.iloc[k]
        pr = proba[k]
        pred_cls = int(pr.argmax())
        matches.append({
            "date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
            "home": row["HomeTeam"], "away": row["AwayTeam"],
            "wdl": {"H": float(pr[0]), "D": float(pr[1]), "A": float(pr[2])},
            "pick": CLASS_TO_OUTCOME[pred_cls],
            "actual": row["FTR"],
            "correct": bool(CLASS_TO_OUTCOME[pred_cls] == row["FTR"]),
        })
    matches.sort(key=lambda m: m["date"], reverse=True)

    # Metrics
    acc = float((proba.argmax(axis=1) == y_test).mean())
    ml_brier, ml_ll = _brier(proba, y_test), _logloss(proba, y_test)
    base_acc = float((y_test == 0).mean())  # always-home
    book = _book_probs(meta_test)
    book_metrics = None
    if book is not None:
        book_metrics = {
            "accuracy": float((book.argmax(axis=1) == y_test).mean()),
            "brier": _brier(book, y_test), "log_loss": _logloss(book, y_test),
        }

    return {
        "season": HOLDOUT_TEST[0],
        "n_test": int(len(y_test)),
        "matches": matches,
        "metrics": {
            "ml": {"accuracy": acc, "brier": ml_brier, "log_loss": ml_ll},
            "baseline_home_accuracy": base_acc,
            "book": book_metrics,
        },
        "reliability": _reliability(proba, y_test),
    }


def _get_holdout() -> dict:
    global _HOLDOUT
    if _HOLDOUT is None:
        _HOLDOUT = _build_holdout()
    return _HOLDOUT


def get_board(limit: int = 12) -> dict:
    h = _get_holdout()
    return {
        "season": h["season"],
        "note": "Out-of-sample: each prediction used only data available before "
                "kick-off; actual results shown for transparency.",
        "matches": h["matches"][:limit],
    }


def get_model_quality() -> dict:
    h = _get_holdout()
    return {
        "holdout_season": h["season"],
        "n_test": h["n_test"],
        "metrics": h["metrics"],
        "reliability": h["reliability"],
        "features": FEATURES,
        "form_window": FORM_N,
    }


# ---------------------------------------------------------------------- #
# Multi-league daily predictions                                         #
# ---------------------------------------------------------------------- #
# Last 4 seasons per league -> current-ish team strength, fewer downloads.
DAILY_SEASONS = list(DEFAULT_SEASONS[-4:])
_LEAGUE_REG: Dict[str, dict] = {}

_SUFFIXES = {"fc", "cf", "afc", "sc", "ac", "cd", "ud", "ssc", "us", "as",
             "calcio", "club", "sv", "vfl", "vfb", "fsv", "tsg", "bk", "if"}


def _norm_team(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    tokens = [t for t in s.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


def _get_league_model(code: str) -> dict:
    if code not in _LEAGUE_REG:
        df = load_matches(DAILY_SEASONS, league=code, verbose=False)
        ml = FootballMLModel().fit(df)
        try:
            dc = DixonColesModel().fit(df)
        except Exception:
            dc = None
        norm = {_norm_team(t): t for t in ml.teams}
        _LEAGUE_REG[code] = {
            "ml": ml, "dc": dc, "teams": ml.teams, "norm": norm,
            "name": LEAGUES[code]["name"], "country": LEAGUES[code]["country"],
        }
    return _LEAGUE_REG[code]


def league_teams(code: str) -> List[str]:
    if code not in LEAGUES:
        raise ValueError(f"Unknown league {code}.")
    return _get_league_model(code)["teams"]


def _match_league(country: Optional[str], league_name: Optional[str]) -> Optional[str]:
    c = (country or "").strip().lower()
    n = (league_name or "").strip().lower()
    for code, info in LEAGUES.items():
        if info["country"].lower() != c:
            continue
        names = [info["name"]] + info.get("api_names", [])
        if any(n == x.strip().lower() for x in names):
            return code
    return None


def _match_team(reg: dict, api_name: str) -> Optional[str]:
    n = _norm_team(api_name)
    if n in reg["norm"]:
        return reg["norm"][n]
    cand = difflib.get_close_matches(n, list(reg["norm"].keys()), n=1, cutoff=0.82)
    return reg["norm"][cand[0]] if cand else None


def _predict_in_league(code: str, home: str, away: str) -> dict:
    ml: FootballMLModel = _get_league_model(code)["ml"]
    wdl = ml.predict_teams(home, away)
    pick = CLASS_TO_OUTCOME[int(np.argmax([wdl["H"], wdl["D"], wdl["A"]]))]
    return {"wdl": wdl, "pick": pick}


def supported_leagues() -> List[dict]:
    return [{"code": c, "name": i["name"], "country": i["country"]}
            for c, i in LEAGUES.items()]


def _demo_fixtures(limit: int) -> List[dict]:
    """Sample fixtures from known teams, used until an API key is set."""
    out: List[dict] = []
    for code in ("E0", "BRA", "USA"):
        reg = _get_league_model(code)
        teams = reg["teams"]
        for i in range(0, min(len(teams) - 1, 8), 2):
            home, away = teams[i], teams[i + 1]
            pred = _predict_in_league(code, home, away)
            out.append({
                "country": reg["country"], "league": reg["name"],
                "home": home, "away": away, "kickoff": None,
                "status": "predicted", "demo": True, **pred,
            })
            if len(out) >= limit:
                return out
    return out


def get_daily(date_str: Optional[str] = None, demo_limit: int = 12) -> dict:
    base = {"supported_leagues": supported_leagues()}

    if not fixtures_provider.has_key():
        return {
            **base, "mode": "demo", "date": fixtures_provider.today_str(),
            "note": "Demo fixtures. Set API_FOOTBALL_KEY (free api-sports.io) "
                    "to show today's real matches.",
            "matches": _demo_fixtures(demo_limit),
        }

    try:
        raw = fixtures_provider.fetch_fixtures(date_str)
    except Exception as exc:
        return {**base, "mode": "error", "error": str(exc), "matches": []}

    matches, predicted = [], 0
    for fx in raw:
        rec = {"country": fx["country"], "league": fx["league_name"],
               "home": fx["home"], "away": fx["away"], "kickoff": fx["kickoff"]}
        code = _match_league(fx["country"], fx["league_name"])
        if code is None:
            rec["status"] = "unsupported_league"
            matches.append(rec)
            continue
        reg = _get_league_model(code)
        h = _match_team(reg, fx["home"])
        a = _match_team(reg, fx["away"])
        if not h or not a:
            rec["status"] = "insufficient_data"
            matches.append(rec)
            continue
        rec.update({"status": "predicted", **_predict_in_league(code, h, a)})
        predicted += 1
        matches.append(rec)

    # Predicted matches first, then the rest.
    matches.sort(key=lambda m: 0 if m.get("status") == "predicted" else 1)
    return {**base, "mode": "live", "date": date_str or fixtures_provider.today_str(),
            "total_fixtures": len(raw), "predicted": predicted, "matches": matches}


# ---------------------------------------------------------------------- #
# World Cup 2026 (national teams)                                        #
# ---------------------------------------------------------------------- #
_WC: Optional[dict] = None
_WC_QUALITY: Optional[dict] = None


def _get_wc() -> dict:
    global _WC
    if _WC is None:
        df = load_international(verbose=False)
        model = WorldCupModel().fit(df)
        _WC = {"df": df, "model": model}
    return _WC


def _wc_pick(p: dict) -> str:
    return CLASS_TO_OUTCOME[int(np.argmax([p["H"], p["D"], p["A"]]))]


def wc_teams() -> List[str]:
    """The 48 teams in the upcoming World Cup (falls back to recently active)."""
    wc = _get_wc()
    up = upcoming_world_cup(wc["df"])
    teams = sorted(set(up["home_team"]) | set(up["away_team"]))
    if teams:
        return teams
    recent = wc["df"][wc["df"]["date"] >= pd.Timestamp("2022-01-01")]
    return sorted(set(recent["home_team"]) | set(recent["away_team"]))


def wc_predict(home: str, away: str, neutral: bool = True) -> dict:
    model: WorldCupModel = _get_wc()["model"]
    if not model.knows_team(home) or not model.knows_team(away):
        raise ValueError("Unknown national team(s).")
    p = model.predict(home, away, neutral=neutral)
    return {"home": home, "away": away, "wdl": p, "pick": _wc_pick(p),
            "neutral": neutral, "elo_home": model.elo_of(home),
            "elo_away": model.elo_of(away)}


def wc_fixtures(limit: int = 48) -> dict:
    wc = _get_wc()
    model: WorldCupModel = wc["model"]
    up = upcoming_world_cup(wc["df"])
    matches = []
    for r in up.head(limit).itertuples(index=False):
        p = model.predict(r.home_team, r.away_team, neutral=bool(r.neutral))
        matches.append({
            "date": r.date.strftime("%Y-%m-%d"),
            "home": r.home_team, "away": r.away_team,
            "city": getattr(r, "city", None), "country": getattr(r, "country", None),
            "wdl": p, "pick": _wc_pick(p),
            "elo_home": model.elo_of(r.home_team), "elo_away": model.elo_of(r.away_team),
        })
    return {"count": int(len(up)), "matches": matches}


def wc_quality() -> dict:
    global _WC_QUALITY
    if _WC_QUALITY is None:
        import xgboost as xgb
        df = _get_wc()["df"]
        m = WorldCupModel()
        X, y, meta = m.engineer(df)
        test_mask = (meta["date"] >= pd.Timestamp(wc_mod.TEST_SINCE)).to_numpy()
        clf = xgb.XGBClassifier(**m.params)
        clf.fit(X[~test_mask], y[~test_mask])
        proba = clf.predict_proba(X[test_mask])
        yt = y[test_mask]
        _WC_QUALITY = {
            "accuracy": float((proba.argmax(axis=1) == yt).mean()),
            "baseline": float((yt == 0).mean()),
            "brier": _wc_brier(proba, yt), "log_loss": _wc_logloss(proba, yt),
            "n_test": int(test_mask.sum()), "test_since": wc_mod.TEST_SINCE,
        }
    return _WC_QUALITY
