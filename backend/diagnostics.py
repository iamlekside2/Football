"""
Diagnostics for the Dixon-Coles value model.

Two questions, after the single-season backtest came back at -31% ROI:

1.  ROBUSTNESS  -- is the negative result a one-season fluke, or does it
    hold across many seasons (and does time-decay weighting rescue it)?

2.  CALIBRATION -- *why* is the model losing? We compare the model's H/D/A
    probabilities against the bookmaker's vig-removed closing probabilities
    using Brier score, log-loss, and a reliability table. If the closing
    line is better calibrated than the model (it almost certainly is), there
    is no edge to extract and the value bets are just noise pointed at the
    market's least efficient-looking — but actually correctly priced — sides.

Run:
    python diagnostics.py                      # all default seasons + calibration
    python diagnostics.py --seasons 2223 2324  # specific seasons
    python diagnostics.py --no-decay-compare   # skip the decay column (faster)
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd

from backtest import (
    load_season, run_backtest, baseline_favourite, _summary, _fmt_pct,
)
from poisson_model import OUTCOMES

URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
DEFAULT_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]


# ---------------------------------------------------------------------- #
# 1. Multi-season robustness                                             #
# ---------------------------------------------------------------------- #
def multi_season(seasons: List[str], edge_threshold: float,
                 min_train_matches: int, max_goals: int,
                 decay_compare: bool, decay: float) -> pd.DataFrame:
    print("=" * 72)
    print("MULTI-SEASON ROBUSTNESS  (EPL, edge threshold "
          f"{edge_threshold:.2f}, flat 1u stakes)")
    print("=" * 72)

    rows = []
    pooled_profit_nodecay = []
    pooled_profit_decay = []
    for code in seasons:
        url = URL_TEMPLATE.format(code=code)
        season_label = f"20{code[:2]}/{code[2:]}"
        try:
            df = load_season(url=url, verbose=False)
        except Exception as exc:
            print(f"  {season_label}: load failed ({exc}) -- skipping")
            continue

        bets, elig, _ = run_backtest(
            df, edge_threshold=edge_threshold,
            min_train_matches=min_train_matches, max_goals=max_goals,
            decay=0.0, verbose=False)
        m = _summary(bets["profit"].to_numpy()) if not bets.empty else _summary(np.array([]))
        fav = baseline_favourite(elig)
        pooled_profit_nodecay.extend(bets["profit"].tolist() if not bets.empty else [])

        row = {
            "season": season_label,
            "bets": m["n"],
            "model_roi": m["roi"],
            "model_pl": m["total_profit"],
            "fav_roi": fav["roi"],
        }

        if decay_compare:
            bets_d, _, _ = run_backtest(
                df, edge_threshold=edge_threshold,
                min_train_matches=min_train_matches, max_goals=max_goals,
                decay=decay, verbose=False)
            md = _summary(bets_d["profit"].to_numpy()) if not bets_d.empty else _summary(np.array([]))
            row["decay_roi"] = md["roi"]
            pooled_profit_decay.extend(bets_d["profit"].tolist() if not bets_d.empty else [])

        rows.append(row)
        msg = (f"  {season_label}:  bets={m['n']:>3}  "
               f"model ROI={_fmt_pct(m['roi']):>8}  fav ROI={_fmt_pct(fav['roi']):>8}")
        if decay_compare:
            msg += f"  decay ROI={_fmt_pct(row['decay_roi']):>8}"
        print(msg)

    table = pd.DataFrame(rows)

    # Pooled, all seasons together (the honest aggregate).
    print("-" * 72)
    if pooled_profit_nodecay:
        agg = _summary(np.array(pooled_profit_nodecay))
        print(f"  POOLED (no decay):  bets={agg['n']}  "
              f"hit={agg['hit_rate'] * 100:.1f}%  P/L={agg['total_profit']:+.1f}u  "
              f"ROI={_fmt_pct(agg['roi'])}")
    if decay_compare and pooled_profit_decay:
        aggd = _summary(np.array(pooled_profit_decay))
        print(f"  POOLED (decay {decay}):  bets={aggd['n']}  "
              f"hit={aggd['hit_rate'] * 100:.1f}%  P/L={aggd['total_profit']:+.1f}u  "
              f"ROI={_fmt_pct(aggd['roi'])}")
    print()
    return table


# ---------------------------------------------------------------------- #
# 2. Calibration: model vs bookmaker                                     #
# ---------------------------------------------------------------------- #
def _pool(preds: pd.DataFrame, cols):
    """Stack the three 1X2 outcomes into (prob, hit) arrays."""
    p = np.concatenate([preds[c].to_numpy() for c in cols])
    y = np.concatenate([(preds["result"] == o).to_numpy() for o in OUTCOMES])
    return p, y.astype(float)


def brier_multiclass(preds: pd.DataFrame, cols) -> float:
    y = np.stack([(preds["result"] == o).to_numpy().astype(float) for o in OUTCOMES], axis=1)
    p = preds[cols].to_numpy()
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def log_loss_multiclass(preds: pd.DataFrame, cols) -> float:
    y = np.stack([(preds["result"] == o).to_numpy().astype(float) for o in OUTCOMES], axis=1)
    p = np.clip(preds[cols].to_numpy(), 1e-12, 1.0)
    return float(np.mean(-np.sum(y * np.log(p), axis=1)))


def reliability_table(preds: pd.DataFrame, cols, n_bins: int = 10) -> pd.DataFrame:
    p, y = _pool(preds, cols)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
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
            "observed": float(y[mask].mean()),
            "gap": float(p[mask].mean() - y[mask].mean()),
        })
    return pd.DataFrame(rows)


def calibration_report(seasons: List[str], edge_threshold: float,
                       min_train_matches: int, max_goals: int, decay: float):
    print("=" * 72)
    print("CALIBRATION: model probabilities vs bookmaker closing line")
    print("=" * 72)

    frames = []
    for code in seasons:
        try:
            df = load_season(url=URL_TEMPLATE.format(code=code), verbose=False)
        except Exception:
            continue
        _, _, preds = run_backtest(
            df, edge_threshold=edge_threshold,
            min_train_matches=min_train_matches, max_goals=max_goals,
            decay=decay, collect_predictions=True, verbose=False)
        if not preds.empty:
            frames.append(preds)

    if not frames:
        print("No predictions collected -- cannot assess calibration.")
        return

    preds = pd.concat(frames, ignore_index=True)
    print(f"Pooled fixtures across seasons: {len(preds)}\n")

    model_cols = ["m_H", "m_D", "m_A"]
    book_cols = ["b_H", "b_D", "b_A"]

    print("Scoring (lower = better) -- a fair test of who predicts results better:")
    print(f"{'':<22}{'Brier':>12}{'LogLoss':>12}")
    print(f"{'Model (Dixon-Coles)':<22}"
          f"{brier_multiclass(preds, model_cols):>12.4f}"
          f"{log_loss_multiclass(preds, model_cols):>12.4f}")
    print(f"{'Bookmaker (no-vig)':<22}"
          f"{brier_multiclass(preds, book_cols):>12.4f}"
          f"{log_loss_multiclass(preds, book_cols):>12.4f}")

    m_brier = brier_multiclass(preds, model_cols)
    b_brier = brier_multiclass(preds, book_cols)
    print()
    if b_brier < m_brier:
        print("-> The bookmaker's closing line is BETTER calibrated than the model.")
        print("   The model cannot have a real 1X2 edge against this line: its")
        print("   'value' flags are where it is most wrong, not where the book is.")
    else:
        print("-> The model scores better than the closing line here -- surprising;")
        print("   double-check for leakage before believing it.")

    print("\nReliability table -- MODEL (pooled H/D/A): does P(x) ~ observed freq?")
    rt_m = reliability_table(preds, model_cols)
    print(f"{'prob bin':<12}{'n':>7}{'mean_pred':>12}{'observed':>12}{'gap':>9}")
    for _, r in rt_m.iterrows():
        print(f"{r['bin']:<12}{int(r['n']):>7}{r['mean_pred']:>12.3f}"
              f"{r['observed']:>12.3f}{r['gap']:>+9.3f}")

    print("\nReliability table -- BOOKMAKER no-vig (for comparison):")
    rt_b = reliability_table(preds, book_cols)
    print(f"{'prob bin':<12}{'n':>7}{'mean_pred':>12}{'observed':>12}{'gap':>9}")
    for _, r in rt_b.iterrows():
        print(f"{r['bin']:<12}{int(r['n']):>7}{r['mean_pred']:>12.3f}"
              f"{r['observed']:>12.3f}{r['gap']:>+9.3f}")

    print("\nReading the MODEL table: a large positive 'gap' in the high-probability")
    print("bins means the model is OVERCONFIDENT (says e.g. 0.6, happens 0.45) --")
    print("classic source of bleeding money by backing its own over-priced picks.")
    return preds


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS,
                        help="football-data season codes, e.g. 2324 2425.")
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--min-train-matches", type=int, default=80)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0.0018,
                        help="Per-day decay used for the decay column / calibration "
                             "(~1-year half-life).")
    parser.add_argument("--no-decay-compare", action="store_true",
                        help="Skip the decay-weighted column in the robustness table.")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_robustness:
        multi_season(args.seasons, args.edge_threshold, args.min_train_matches,
                     args.max_goals, not args.no_decay_compare, args.decay)

    if not args.skip_calibration:
        # Calibration uses no decay by default (assess the base model honestly).
        calibration_report(args.seasons, args.edge_threshold,
                            args.min_train_matches, args.max_goals, decay=0.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
