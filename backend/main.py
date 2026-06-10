"""
FastAPI backend for the football value-model research dashboard.

This is a *research* API: it exposes the existing Dixon-Coles model and the
walk-forward backtest so the React frontend can visualise edge (or the lack
of it) across seasons. It deliberately does NOT place bets or pull live odds
-- that infrastructure waits until the backtest demonstrates a real edge.

Run (from the backend/ directory):
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from poisson_model import DixonColesModel, find_value, OUTCOMES
from backtest import (
    load_season, run_backtest, favourite_outcome,
    _summary, baseline_favourite, baseline_random, baseline_always_home,
    edge_bucket_breakdown,
)
from diagnostics import (
    URL_TEMPLATE, DEFAULT_SEASONS,
    brier_multiclass, log_loss_multiclass, reliability_table,
)
import prediction_service

app = FastAPI(title="Football Value Model API", version="0.1.0")

# Dev CORS: allow the Vite dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SEASON_LABELS = {c: f"20{c[:2]}/{c[2:]}" for c in DEFAULT_SEASONS}


# ---------------------------------------------------------------------- #
# Caching (research tool: a backtest downloads + refits ~95 times)       #
# ---------------------------------------------------------------------- #
_SEASON_CACHE: dict = {}
_BT_CACHE: dict = {}


def get_season(code: str) -> pd.DataFrame:
    if code not in _SEASON_CACHE:
        try:
            _SEASON_CACHE[code] = load_season(url=URL_TEMPLATE.format(code=code),
                                              verbose=False)
        except Exception as exc:
            raise HTTPException(status_code=502,
                                detail=f"Could not load season {code}: {exc}")
    return _SEASON_CACHE[code]


def get_backtest(code, edge, min_train, decay, max_goals, collect=False):
    key = (code, round(edge, 4), min_train, round(decay, 6), max_goals, collect)
    if key not in _BT_CACHE:
        df = get_season(code)
        _BT_CACHE[key] = run_backtest(
            df, edge_threshold=edge, min_train_matches=min_train,
            decay=decay, max_goals=max_goals,
            collect_predictions=collect, verbose=False)
    return _BT_CACHE[key]


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def _f(x) -> Optional[float]:
    """JSON-safe float (NaN/inf -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _verdict(model: dict, fav: dict, rnd: dict) -> dict:
    roi = model["roi"]
    if model["n"] == 0:
        return {"level": "none", "text": "No value bets were flagged at this threshold."}
    beats_fav = np.isfinite(fav["roi"]) and roi > fav["roi"]
    beats_rnd = np.isfinite(rnd["roi"]) and roi > rnd["roi"]
    if roi <= 0:
        return {"level": "bad",
                "text": "ROI is negative or zero -- no edge on this data. Do NOT "
                        "build live infrastructure or risk money yet."}
    if roi < 0.02:
        return {"level": "weak",
                "text": "ROI is positive but within noise (<2%). Treat as no "
                        "demonstrated edge; one season can't distinguish this from luck."}
    if not (beats_fav and beats_rnd):
        return {"level": "weak",
                "text": "ROI is positive but does not clearly beat the naive "
                        "baselines. Validate on more seasons before trusting it."}
    return {"level": "good",
            "text": f"ROI is {roi * 100:+.1f}% and beats favourite & random "
                    "baselines. Encouraging on one season -- confirm out-of-sample."}


def _cumulative(bets_df: pd.DataFrame) -> List[dict]:
    if bets_df.empty:
        return []
    out = []
    for _, r in bets_df.iterrows():
        out.append({"date": pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"),
                    "cum_profit": _f(r["cum_profit"]),
                    "home": r["HomeTeam"], "away": r["AwayTeam"],
                    "outcome": r["outcome"], "won": bool(r["won"])})
    return out


def _favourite_cumulative(eligible_df: pd.DataFrame) -> List[dict]:
    if eligible_df.empty:
        return []
    elig = eligible_df.sort_values("Date")
    cum, out = 0.0, []
    for _, r in elig.iterrows():
        odds = {"H": r["odds_H"], "D": r["odds_D"], "A": r["odds_A"]}
        pick = favourite_outcome(odds)
        if pick is None:
            continue
        cum += (odds[pick] - 1.0) if pick == r["result"] else -1.0
        out.append({"date": pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"),
                    "cum_profit": _f(cum)})
    return out


def _summary_json(s: dict) -> dict:
    return {"n": int(s["n"]), "hit_rate": _f(s["hit_rate"]),
            "total_profit": _f(s["total_profit"]), "staked": _f(s["staked"]),
            "roi": _f(s["roi"])}


# ---------------------------------------------------------------------- #
# Request models                                                         #
# ---------------------------------------------------------------------- #
class BacktestRequest(BaseModel):
    season_code: str = "2324"
    edge_threshold: float = Field(0.05, ge=0.0, le=0.5)
    min_train_matches: int = Field(80, ge=10, le=300)
    decay: float = Field(0.0, ge=0.0, le=0.05)
    max_goals: int = Field(10, ge=5, le=15)


class CalibrationRequest(BaseModel):
    season_codes: List[str] = Field(default_factory=lambda: list(DEFAULT_SEASONS))
    min_train_matches: int = Field(80, ge=10, le=300)
    decay: float = Field(0.0, ge=0.0, le=0.05)
    max_goals: int = Field(10, ge=5, le=15)


# ---------------------------------------------------------------------- #
# Routes                                                                 #
# ---------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/seasons")
def seasons():
    return [{"code": c, "label": SEASON_LABELS.get(c, c)} for c in DEFAULT_SEASONS]


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    bets, elig, _ = get_backtest(req.season_code, req.edge_threshold,
                                 req.min_train_matches, req.decay, req.max_goals)
    model = _summary(bets["profit"].to_numpy()) if not bets.empty else _summary(np.array([]))
    fav = baseline_favourite(elig)
    home = baseline_always_home(elig)
    rnd = baseline_random(elig)

    buckets = []
    bb = edge_bucket_breakdown(bets)
    if not bb.empty:
        for _, r in bb.iterrows():
            buckets.append({"edge_bucket": str(r["edge_bucket"]), "n": int(r["n"]),
                            "hit_rate": _f(r["hit_rate"]),
                            "total_profit": _f(r["total_profit"]), "roi": _f(r["roi"])})

    return {
        "params": req.model_dump(),
        "season_label": SEASON_LABELS.get(req.season_code, req.season_code),
        "fixtures_evaluated": int(len(elig)),
        "model": _summary_json(model),
        "baselines": {
            "favourite": _summary_json(fav),
            "always_home": _summary_json(home),
            "random": _summary_json(rnd),
        },
        "edge_buckets": buckets,
        "cumulative": _cumulative(bets),
        "favourite_cumulative": _favourite_cumulative(elig),
        "verdict": _verdict(model, fav, rnd),
    }


@app.post("/api/calibration")
def calibration(req: CalibrationRequest):
    frames = []
    used, failed = [], []
    for code in req.season_codes:
        try:
            _, _, preds = get_backtest(code, 0.0, req.min_train_matches,
                                       req.decay, req.max_goals, collect=True)
        except HTTPException:
            failed.append(code)
            continue
        if not preds.empty:
            frames.append(preds)
            used.append(code)
    if not frames:
        raise HTTPException(status_code=502,
                            detail="Could not load any season data (network).")

    preds = pd.concat(frames, ignore_index=True)
    model_cols, book_cols = ["m_H", "m_D", "m_A"], ["b_H", "b_D", "b_A"]

    def rel(cols):
        return [{"bin": r["bin"], "n": int(r["n"]), "mean_pred": _f(r["mean_pred"]),
                 "observed": _f(r["observed"]), "gap": _f(r["gap"])}
                for _, r in reliability_table(preds, cols).iterrows()]

    return {
        "n_fixtures": int(len(preds)),
        "seasons": used,
        "seasons_failed": failed,
        "scores": {
            "model": {"brier": _f(brier_multiclass(preds, model_cols)),
                      "log_loss": _f(log_loss_multiclass(preds, model_cols))},
            "book": {"brier": _f(brier_multiclass(preds, book_cols)),
                     "log_loss": _f(log_loss_multiclass(preds, book_cols))},
        },
        "model_reliability": rel(model_cols),
        "book_reliability": rel(book_cols),
    }


@app.get("/api/predict")
def predict(home: str, away: str):
    """Head-to-head prediction from the ML model (W/D/L) + Dixon-Coles
    (most-likely scoreline and over/under), using each team's current state."""
    try:
        return prediction_service.predict_match(home, away)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/ml-teams")
def ml_teams():
    """Teams the ML predictor knows (across all loaded seasons)."""
    return prediction_service.list_teams()


@app.get("/api/board")
def board(limit: int = 12):
    """Fixtures board: recent matches with the model's pre-match prediction
    and the actual result (out-of-sample, no lookahead)."""
    return prediction_service.get_board(limit=limit)


@app.get("/api/model-quality")
def model_quality():
    """Transparency: how good the ML model is out-of-sample, vs the bookmaker."""
    return prediction_service.get_model_quality()


@app.get("/api/daily")
def daily(date: Optional[str] = None):
    """Today's (or a given date's) football fixtures with predictions.
    Live when API_FOOTBALL_KEY is set; otherwise demo fixtures."""
    return prediction_service.get_daily(date_str=date)


@app.get("/api/leagues")
def leagues():
    return prediction_service.supported_leagues()


@app.get("/api/league-teams")
def league_teams(league: str):
    try:
        return prediction_service.league_teams(league)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/predict-league")
def predict_league(league: str, home: str, away: str):
    """Head-to-head prediction within a specific league's model."""
    try:
        return prediction_service.predict_league(league, home, away)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/provider-status")
def provider_status():
    import fixtures_provider
    return fixtures_provider.quota_status()


@app.get("/api/wc/fixtures")
def wc_fixtures(limit: int = 48):
    """Upcoming 2026 World Cup fixtures with national-team predictions."""
    return prediction_service.wc_fixtures(limit=limit)


@app.get("/api/wc/teams")
def wc_teams():
    return prediction_service.wc_teams()


@app.get("/api/wc/predict")
def wc_predict(home: str, away: str, neutral: bool = True):
    try:
        return prediction_service.wc_predict(home, away, neutral=neutral)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/wc/quality")
def wc_quality():
    return prediction_service.wc_quality()


@app.get("/api/teams")
def teams(season_code: str):
    df = get_season(season_code)
    return sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))


# ---------------------------------------------------------------------- #
# Serve the built frontend (single-origin deploy).                       #
# Mounted LAST so it never shadows the /api/* routes above. Requires     #
# `npm run build` in frontend/ to have produced frontend/dist.           #
# ---------------------------------------------------------------------- #
_DIST = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
