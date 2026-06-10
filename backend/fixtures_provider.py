"""
fixtures_provider.py -- fetch the day's real football fixtures.

Primary source: API-Football (api-sports.io). Set the env var
API_FOOTBALL_KEY to your free api-sports.io key to go live:

    setx API_FOOTBALL_KEY "your_key_here"     (Windows, new shell after)
    export API_FOOTBALL_KEY=your_key_here     (macOS/Linux)

Without a key the provider reports `mode="demo"` so the rest of the app can
still be exercised; callers then synthesise sample fixtures from known teams.

We only normalise fixtures down to what the predictor needs: competition
(country + league name) and the two team names. League/team name matching to
our trained models happens in prediction_service.
"""

from __future__ import annotations

import os
from datetime import date
from typing import List, Optional

import requests

API_BASE = "https://v3.football.api-sports.io"
KEY_ENV = "API_FOOTBALL_KEY"


def api_key() -> Optional[str]:
    key = os.environ.get(KEY_ENV)
    return key.strip() if key else None


def has_key() -> bool:
    return bool(api_key())


def today_str() -> str:
    return date.today().isoformat()


def fetch_fixtures(date_str: Optional[str] = None) -> List[dict]:
    """Return normalised fixtures for a date from API-Football.

    Raises RuntimeError if no key is configured (caller decides on demo mode).
    Each item: {country, league_name, home, away, kickoff, status}.
    """
    key = api_key()
    if not key:
        raise RuntimeError("No API_FOOTBALL_KEY configured.")
    date_str = date_str or today_str()

    resp = requests.get(
        f"{API_BASE}/fixtures",
        params={"date": date_str},
        headers={"x-apisports-key": key},
        timeout=25,
    )
    resp.raise_for_status()
    payload = resp.json()
    out: List[dict] = []
    for item in payload.get("response", []):
        league = item.get("league", {}) or {}
        teams = item.get("teams", {}) or {}
        fixture = item.get("fixture", {}) or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        if not home or not away:
            continue
        out.append({
            "country": league.get("country"),
            "league_name": league.get("name"),
            "home": home,
            "away": away,
            "kickoff": fixture.get("date"),
            "status": (fixture.get("status") or {}).get("short"),
        })
    return out


def quota_status() -> dict:
    """Lightweight health/quota probe (also validates the key)."""
    key = api_key()
    if not key:
        return {"configured": False}
    try:
        resp = requests.get(f"{API_BASE}/status",
                            headers={"x-apisports-key": key}, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("response", {})
        reqs = (data.get("requests") or {})
        return {
            "configured": True,
            "account": (data.get("account") or {}).get("email"),
            "plan": (data.get("subscription") or {}).get("plan"),
            "requests_used": reqs.get("current"),
            "requests_limit": reqs.get("limit_day"),
        }
    except Exception as exc:
        return {"configured": True, "error": str(exc)}
