"""
Honest walk-forward backtest for the Dixon-Coles value model.

The whole point of this script is to find out whether the model in
`poisson_model.py` has any real edge BEFORE you build scraping or risk
money. It is built to avoid lookahead bias: for every match it predicts,
the model is trained only on matches that kicked off strictly earlier.

What it does
------------
1.  Loads a real league season from football-data.co.uk (or a local CSV),
    detecting the actual column names rather than hardcoding them.
2.  Walks forward chronologically. On each match date it (re)fits the model
    on all prior matches, then predicts that date's fixtures.
3.  Runs the value filter (poisson_model.find_value) against the closing
    1X2 odds and records every flagged bet.
4.  Settles each bet at a flat 1-unit stake using the real result.
5.  Reports bet count, hit rate, ROI, total P/L, an edge-bucket breakdown,
    and baseline returns for (a) always backing the favourite and (b)
    random betting, so you can see whether the model beats either.

Honesty guardrails
------------------
* No lookahead: training set is always `Date < match_date`.
* The edge threshold is NOT tuned against the reported season. If you want
  to tune it, pass a *separate* season via --tune-csv/--tune-url; tuning on
  the test season and then reporting that ROI is self-deception.
* If ROI is negative or near zero the script says so plainly. A negative
  result here is a success: it saves you building infrastructure for an
  edge that does not exist.

Run:
    python backtest.py                       # downloads EPL 2023/24, default settings
    python backtest.py --csv E0.csv          # use a local file
    python backtest.py --edge-threshold 0.08 --min-train-matches 80
"""

from __future__ import annotations

import argparse
import io
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from poisson_model import DixonColesModel, find_value, OUTCOMES

DEFAULT_URL = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"

# Preferred 1X2 closing-odds column triples, best first. football-data.co.uk
# column names drift between seasons/leagues, so we try several. The "C"
# variants (B365CH, PSCH, ...) are *closing* odds; we prefer those.
ONEX2_ODDS_PREFS: List[Tuple[str, str, str]] = [
    ("B365CH", "B365CD", "B365CA"),  # Bet365 closing
    ("B365H", "B365D", "B365A"),     # Bet365
    ("PSCH", "PSCD", "PSCA"),        # Pinnacle closing
    ("PSH", "PSD", "PSA"),          # Pinnacle
    ("AvgCH", "AvgCD", "AvgCA"),     # market average closing
    ("AvgH", "AvgD", "AvgA"),       # market average
    ("BbAvH", "BbAvD", "BbAvA"),     # Betbrain average (older files)
    ("MaxCH", "MaxCD", "MaxCA"),     # best price closing
    ("MaxH", "MaxD", "MaxA"),       # best price
]


# ---------------------------------------------------------------------- #
# Data loading                                                           #
# ---------------------------------------------------------------------- #
def _read_csv(url: Optional[str], csv: Optional[str]) -> pd.DataFrame:
    if csv:
        # Try utf-8 then latin-1 (older football-data files).
        for enc in ("utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(csv, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(csv, encoding="latin-1")

    import time
    import requests
    # football-data.co.uk intermittently drops TLS connections under burst
    # requests; retry a few times with backoff before giving up.
    last_exc = None
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            text = resp.content.decode("utf-8-sig", errors="replace")
            return pd.read_csv(io.StringIO(text))
        except Exception as exc:  # network / SSL / HTTP
            last_exc = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse football-data dates, which are dd/mm/yyyy or dd/mm/yy."""
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    if d.isna().any():
        d2 = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
        d = d.fillna(d2)
    if d.isna().any():
        # Final fallback: let pandas infer (day-first), silencing nothing.
        d = d.fillna(pd.to_datetime(s, dayfirst=True, errors="coerce"))
    return d


def _pick_odds_columns(columns) -> Tuple[str, str, str]:
    cols = set(columns)
    for triple in ONEX2_ODDS_PREFS:
        if set(triple).issubset(cols):
            return triple
    raise ValueError(
        "Could not find a usable 1X2 odds column triple. "
        f"Looked for {ONEX2_ODDS_PREFS}. Available columns: {sorted(cols)}"
    )


def load_season(url: Optional[str] = DEFAULT_URL,
                csv: Optional[str] = None,
                verbose: bool = True) -> pd.DataFrame:
    """Return a clean DataFrame: Date, HomeTeam, AwayTeam, FTHG, FTAG,
    odds_H, odds_D, odds_A — sorted chronologically.

    Prints the raw columns and which odds set was chosen so column drift is
    visible rather than silently mis-mapped.
    """
    raw = _read_csv(url, csv)
    if verbose:
        src = csv if csv else url
        print(f"Loaded {len(raw)} rows from {src}")
        print(f"Columns present ({len(raw.columns)}): {list(raw.columns)}\n")

    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Source is missing required columns {missing}.")

    h_col, d_col, a_col = _pick_odds_columns(raw.columns)
    if verbose:
        print(f"Using 1X2 odds columns: home={h_col}, draw={d_col}, away={a_col}\n")

    df = pd.DataFrame({
        "Date": _parse_dates(raw["Date"]),
        "HomeTeam": raw["HomeTeam"].astype(str).str.strip(),
        "AwayTeam": raw["AwayTeam"].astype(str).str.strip(),
        "FTHG": pd.to_numeric(raw["FTHG"], errors="coerce"),
        "FTAG": pd.to_numeric(raw["FTAG"], errors="coerce"),
        "odds_H": pd.to_numeric(raw[h_col], errors="coerce"),
        "odds_D": pd.to_numeric(raw[d_col], errors="coerce"),
        "odds_A": pd.to_numeric(raw[a_col], errors="coerce"),
    })

    before = len(df)
    df = df.dropna(subset=["Date", "FTHG", "FTAG"]).copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df = df.sort_values("Date").reset_index(drop=True)
    if verbose and before != len(df):
        print(f"Dropped {before - len(df)} rows with missing date/score.\n")
    return df


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def actual_result(fthg: int, ftag: int) -> str:
    if fthg > ftag:
        return "H"
    if fthg < ftag:
        return "A"
    return "D"


def favourite_outcome(odds: Dict[str, float]) -> Optional[str]:
    valid = {k: v for k, v in odds.items()
             if v is not None and np.isfinite(v) and v > 1.0}
    if not valid:
        return None
    return min(valid, key=valid.get)


# ---------------------------------------------------------------------- #
# Walk-forward backtest                                                  #
# ---------------------------------------------------------------------- #
def run_backtest(df: pd.DataFrame,
                 edge_threshold: float = 0.05,
                 min_train_matches: int = 80,
                 max_goals: int = 10,
                 decay: float = 0.0,
                 collect_predictions: bool = False,
                 verbose: bool = True):
    """Walk forward over each match date, returning (bets_df, eligible_df, preds_df).

    bets_df    : one row per flagged value bet, settled at 1-unit flat stake.
    eligible_df: one row per match the model actually predicted (used for the
                 favourite / random baselines, so they cover the same games).
    preds_df   : one row per evaluated fixture with the model's H/D/A
                 probabilities alongside the bookmaker's implied (raw and
                 vig-removed) probabilities. Empty unless collect_predictions.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    unique_dates = list(pd.unique(df["Date"]))

    bets: List[dict] = []
    eligible: List[dict] = []
    predictions: List[dict] = []
    skipped_unseen = 0
    skipped_odds = 0
    predicted_dates = 0

    for d in unique_dates:
        train = df[df["Date"] < d]
        if len(train) < min_train_matches:
            continue
        test = df[df["Date"] == d]

        model = DixonColesModel(max_goals=max_goals)
        try:
            model.fit(train, decay=decay, ref_date=d)
        except Exception as exc:  # pragma: no cover - defensive
            if verbose:
                print(f"  fit failed on {pd.Timestamp(d).date()}: {exc}")
            continue
        predicted_dates += 1

        for _, row in test.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            if not (model.knows_team(home) and model.knows_team(away)):
                skipped_unseen += 1
                continue

            odds = {"H": row["odds_H"], "D": row["odds_D"], "A": row["odds_A"]}
            if any((v is None or not np.isfinite(v) or v <= 1.0) for v in odds.values()):
                skipped_odds += 1
                continue

            result = actual_result(row["FTHG"], row["FTAG"])
            eligible.append({
                "Date": d, "HomeTeam": home, "AwayTeam": away,
                "result": result,
                "odds_H": odds["H"], "odds_D": odds["D"], "odds_A": odds["A"],
            })

            probs = model.predict_proba(home, away)

            if collect_predictions:
                raw = {k: 1.0 / odds[k] for k in OUTCOMES}      # incl. margin
                overround = sum(raw.values())
                novig = {k: raw[k] / overround for k in OUTCOMES}  # margin-removed
                predictions.append({
                    "Date": d, "HomeTeam": home, "AwayTeam": away,
                    "result": result,
                    "m_H": probs["H"], "m_D": probs["D"], "m_A": probs["A"],
                    "b_H": novig["H"], "b_D": novig["D"], "b_A": novig["A"],
                    "braw_H": raw["H"], "braw_D": raw["D"], "braw_A": raw["A"],
                    "odds_H": odds["H"], "odds_D": odds["D"], "odds_A": odds["A"],
                    "overround": overround,
                })

            for vb in find_value(probs, odds, threshold=edge_threshold):
                won = (vb.outcome == result)
                profit = (vb.odds - 1.0) if won else -1.0
                bets.append({
                    "Date": d, "HomeTeam": home, "AwayTeam": away,
                    "outcome": vb.outcome, "result": result,
                    "model_prob": vb.model_prob, "implied_prob": vb.implied_prob,
                    "odds": vb.odds, "edge": vb.edge, "ev": vb.ev,
                    "won": won, "stake": 1.0, "profit": profit,
                })

    bets_df = pd.DataFrame(bets)
    eligible_df = pd.DataFrame(eligible)
    preds_df = pd.DataFrame(predictions)
    if not bets_df.empty:
        bets_df = bets_df.sort_values("Date").reset_index(drop=True)
        bets_df["cum_profit"] = bets_df["profit"].cumsum()

    if verbose:
        print(f"Predicted on {predicted_dates} match dates; "
              f"{len(eligible_df)} fixtures evaluated.")
        if skipped_unseen:
            print(f"Skipped {skipped_unseen} fixtures with an unseen team "
                  "(no prior matches to learn from).")
        if skipped_odds:
            print(f"Skipped {skipped_odds} fixtures with missing/invalid odds.")
        print()

    return bets_df, eligible_df, preds_df


# ---------------------------------------------------------------------- #
# Metrics & baselines                                                    #
# ---------------------------------------------------------------------- #
def _summary(profit: np.ndarray) -> dict:
    n = len(profit)
    staked = float(n)  # 1 unit per bet
    total = float(profit.sum())
    return {
        "n": n,
        "hit_rate": float((profit > 0).mean()) if n else float("nan"),
        "total_profit": total,
        "staked": staked,
        "roi": (total / staked) if staked else float("nan"),
    }


def baseline_favourite(eligible_df: pd.DataFrame) -> dict:
    if eligible_df.empty:
        return _summary(np.array([]))
    profit = []
    for _, r in eligible_df.iterrows():
        odds = {"H": r["odds_H"], "D": r["odds_D"], "A": r["odds_A"]}
        pick = favourite_outcome(odds)
        if pick is None:
            continue
        won = (pick == r["result"])
        profit.append((odds[pick] - 1.0) if won else -1.0)
    return _summary(np.array(profit, dtype=float))


def baseline_random(eligible_df: pd.DataFrame, seed: int = 42) -> dict:
    if eligible_df.empty:
        return _summary(np.array([]))
    rng = np.random.default_rng(seed)
    profit = []
    for _, r in eligible_df.iterrows():
        odds = {"H": r["odds_H"], "D": r["odds_D"], "A": r["odds_A"]}
        pick = OUTCOMES[rng.integers(0, 3)]
        won = (pick == r["result"])
        profit.append((odds[pick] - 1.0) if won else -1.0)
    return _summary(np.array(profit, dtype=float))


def baseline_always_home(eligible_df: pd.DataFrame) -> dict:
    if eligible_df.empty:
        return _summary(np.array([]))
    profit = [(r["odds_H"] - 1.0) if r["result"] == "H" else -1.0
              for _, r in eligible_df.iterrows()]
    return _summary(np.array(profit, dtype=float))


def edge_bucket_breakdown(bets_df: pd.DataFrame) -> pd.DataFrame:
    if bets_df.empty:
        return pd.DataFrame()
    bins = [0.05, 0.10, 0.15, 0.20, np.inf]
    labels = ["0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20+"]
    # Edge can be below the default 0.05 if a custom lower threshold was used.
    lo = min(0.0, float(bets_df["edge"].min()))
    bins = [lo] + bins if lo < 0.05 else bins
    labels = ["<0.05"] + labels if lo < 0.05 else labels
    cats = pd.cut(bets_df["edge"], bins=bins, labels=labels, right=False)
    rows = []
    for label, grp in bets_df.groupby(cats, observed=True):
        prof = grp["profit"].to_numpy()
        rows.append({
            "edge_bucket": label,
            "n": len(grp),
            "hit_rate": float((prof > 0).mean()),
            "total_profit": float(prof.sum()),
            "roi": float(prof.sum() / len(grp)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# Reporting                                                              #
# ---------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%" if np.isfinite(x) else "n/a"


def print_report(bets_df, eligible_df, edge_threshold):
    print("=" * 64)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 64)

    if bets_df.empty:
        print(f"\nNo value bets were flagged at edge threshold {edge_threshold:.3f}.")
        print("The model never disagreed with the bookmaker by enough to bet.")
        print("Try a lower --edge-threshold, but treat that as exploration, "
              "not a validated edge.")
        return

    model = _summary(bets_df["profit"].to_numpy())
    fav = baseline_favourite(eligible_df)
    rnd = baseline_random(eligible_df)
    home = baseline_always_home(eligible_df)

    print(f"\nEdge threshold:        {edge_threshold:.3f} "
          "(model_prob - implied_prob)")
    print(f"Value bets flagged:    {model['n']}")
    print(f"Hit rate:              {model['hit_rate'] * 100:.1f}%")
    print(f"Total staked:          {model['staked']:.0f} units")
    print(f"Total profit/loss:     {model['total_profit']:+.2f} units")
    print(f"ROI:                   {_fmt_pct(model['roi'])}   <-- the number that matters")

    print("\n--- Edge-bucket breakdown (do bigger edges do better?) ---")
    breakdown = edge_bucket_breakdown(bets_df)
    if not breakdown.empty:
        print(f"{'bucket':<12}{'n':>5}{'hit%':>9}{'profit':>11}{'ROI':>10}")
        for _, r in breakdown.iterrows():
            print(f"{str(r['edge_bucket']):<12}{int(r['n']):>5}"
                  f"{r['hit_rate'] * 100:>8.1f}%{r['total_profit']:>11.2f}"
                  f"{_fmt_pct(r['roi']):>10}")

    print("\n--- Baselines over the same fixtures (sanity checks) ---")
    print(f"{'strategy':<22}{'n':>5}{'hit%':>9}{'profit':>11}{'ROI':>10}")
    for name, s in [("Model value bets", model),
                    ("Always favourite", fav),
                    ("Always home", home),
                    ("Random 1X2", rnd)]:
        print(f"{name:<22}{s['n']:>5}{s['hit_rate'] * 100:>8.1f}%"
              f"{s['total_profit']:>11.2f}{_fmt_pct(s['roi']):>10}")

    # Plain-spoken verdict.
    print("\n" + "-" * 64)
    roi = model["roi"]
    beats_fav = np.isfinite(fav["roi"]) and roi > fav["roi"]
    beats_rnd = np.isfinite(rnd["roi"]) and roi > rnd["roi"]
    if roi <= 0:
        print("VERDICT: ROI is NEGATIVE OR ZERO. On this data the model shows "
              "no edge.")
        print("This is the most useful possible result: do NOT build scraping "
              "or risk money on this yet.")
    elif roi < 0.02:
        print("VERDICT: ROI is positive but within noise (< 2%). Treat as NO "
              "demonstrated edge.")
        print("A 1-season sample at this size cannot distinguish this from luck.")
    elif not (beats_fav and beats_rnd):
        print("VERDICT: ROI is positive but the model does NOT clearly beat the "
              "naive baselines.")
        print("Be skeptical; validate on more seasons before trusting it.")
    else:
        print(f"VERDICT: ROI is {_fmt_pct(roi)} and beats the favourite and "
              "random baselines.")
        print("Encouraging on ONE season. Confirm on additional seasons "
              "(out-of-sample) before risking money.")
    print("Reminder: this is a single season; edge thresholds were not tuned "
          "on it. One season is not proof.")
    print("-" * 64)


def save_plot(bets_df, eligible_df, path: str):
    if bets_df.empty:
        print("No bets to plot.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    x = bets_df["Date"]
    ax.plot(x, bets_df["cum_profit"], label="Model value bets", linewidth=2)

    # Overlay the favourite baseline accumulated over the same timeline.
    if not eligible_df.empty:
        elig = eligible_df.sort_values("Date").copy()
        fav_profit = []
        for _, r in elig.iterrows():
            odds = {"H": r["odds_H"], "D": r["odds_D"], "A": r["odds_A"]}
            pick = favourite_outcome(odds)
            p = ((odds[pick] - 1.0) if pick == r["result"] else -1.0) if pick else 0.0
            fav_profit.append(p)
        ax.plot(elig["Date"], np.cumsum(fav_profit),
                label="Always favourite", alpha=0.6, linestyle="--")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative profit (flat 1-unit stakes)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative profit (units)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved cumulative-profit plot to {path}")


# ---------------------------------------------------------------------- #
# Optional threshold tuning on a SEPARATE season (held out)              #
# ---------------------------------------------------------------------- #
def tune_threshold(df: pd.DataFrame, candidates, min_train_matches, max_goals, decay):
    """Pick the edge threshold with the best ROI on a separate season.

    This must be run on data OTHER than the season you report, or you are
    overfitting the threshold to noise. Returns (best_threshold, table).
    """
    rows = []
    for thr in candidates:
        bets_df, _, _ = run_backtest(df, edge_threshold=thr,
                                     min_train_matches=min_train_matches,
                                     max_goals=max_goals, decay=decay, verbose=False)
        s = _summary(bets_df["profit"].to_numpy()) if not bets_df.empty else _summary(np.array([]))
        rows.append({"threshold": thr, "n": s["n"], "roi": s["roi"],
                     "total_profit": s["total_profit"]})
    table = pd.DataFrame(rows)
    valid = table[table["n"] >= 20]  # ignore thresholds with too few bets
    if valid.empty:
        return candidates[0], table
    best = valid.loc[valid["roi"].idxmax(), "threshold"]
    return float(best), table


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="football-data.co.uk season CSV URL (test season).")
    parser.add_argument("--csv", default=None,
                        help="Local CSV path for the test season (overrides --url).")
    parser.add_argument("--edge-threshold", type=float, default=0.05,
                        help="Minimum (model_prob - implied_prob) to flag a bet.")
    parser.add_argument("--min-train-matches", type=int, default=80,
                        help="Minimum matches before the model starts predicting "
                             "(~first 8 rounds of a 20-team league).")
    parser.add_argument("--max-goals", type=int, default=10,
                        help="Score-matrix truncation.")
    parser.add_argument("--decay", type=float, default=0.0,
                        help="Per-day time-decay for training weights "
                             "(0 = none; ~0.0018 ~ 1-year half-life).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the random-betting baseline.")
    parser.add_argument("--plot-file", default="backtest_cumulative_profit.png",
                        help="Where to save the cumulative-profit PNG.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the plot.")
    parser.add_argument("--tune-url", default=None,
                        help="SEPARATE season URL to tune the edge threshold on "
                             "(held out from the reported season).")
    parser.add_argument("--tune-csv", default=None,
                        help="SEPARATE season CSV to tune the edge threshold on.")
    args = parser.parse_args(argv)

    df = load_season(url=args.url, csv=args.csv, verbose=True)

    edge_threshold = args.edge_threshold
    if args.tune_url or args.tune_csv:
        print("Tuning edge threshold on a SEPARATE held-out season...")
        tune_df = load_season(url=args.tune_url, csv=args.tune_csv, verbose=True)
        candidates = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
        edge_threshold, table = tune_threshold(
            tune_df, candidates, args.min_train_matches, args.max_goals, args.decay)
        print("Tuning results (on the held-out season):")
        print(table.to_string(index=False))
        print(f"Selected edge threshold: {edge_threshold:.3f}\n")

    bets_df, eligible_df, _ = run_backtest(
        df,
        edge_threshold=edge_threshold,
        min_train_matches=args.min_train_matches,
        max_goals=args.max_goals,
        decay=args.decay,
        verbose=True,
    )
    print_report(bets_df, eligible_df, edge_threshold)

    if not args.no_plot:
        save_plot(bets_df, eligible_df, args.plot_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
