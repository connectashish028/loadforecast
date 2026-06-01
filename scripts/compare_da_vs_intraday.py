"""Phase B.4 step 3 — same architecture, two markets.

Score the two production XGBoost quantile models on the same 61-day
Mar-Apr 2026 holdout:

  * xgboost_price_v1     targets day-ahead clearing price
  * xgboost_intraday_v1  targets intraday continuous average price

For each delivery day, run both models, score P50 MAE vs the relevant
realised series, score vs the relevant naive baseline (yesterday-same-
quarter-hour of the same market), and run the dispatch sim on the
relevant actuals. Output a wide CSV with one row per quarter-hour and
a daily summary.

The interesting interpretation isn't absolute MAE (intraday is harder
because the realised series carries more noise) — it's:

  1. **Skill vs naive** on each market (can we beat trader-naive in
     either market?)
  2. **Dispatch P&L** when the *forecast for that market* is dispatched
     against the *realised prices of that market* (the actually-relevant
     P&L for someone running a battery on that book).

Outputs:
  backtest_results/da_vs_intraday_per_qh.csv     (per-quarter-hour)
  backtest_results/da_vs_intraday_daily.csv      (per-day summary)

Run from repo root:
    PYTHONPATH=src python scripts/compare_da_vs_intraday.py
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.dispatch import BatterySpec, dispatch_pnl
from loadforecast.models.predict import (
    xgboost_intraday_predict_full,
    xgboost_price_predict_full,
)

PARQUET = "smard_merged_15min.parquet"
DA_COL = "price__germany_luxembourg"
ID_COL = "price__intraday_continuous_de_lu"
OUT_QH = Path("backtest_results/da_vs_intraday_per_qh.csv")
OUT_DAILY = Path("backtest_results/da_vs_intraday_daily.csv")

HOLDOUT_START = date(2026, 3, 1)
HOLDOUT_END = date(2026, 4, 30)


def _drange(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def main() -> None:
    print("Loading parquet + models...")
    df = load_smard_15min(PARQUET)
    if ID_COL not in df.columns:
        raise SystemExit(
            f"Intraday column {ID_COL!r} missing. Run "
            f"scripts/backfill_intraday_continuous.py first."
        )

    dates = _drange(HOLDOUT_START, HOLDOUT_END)
    print(f"Scoring {len(dates)} days ({dates[0]} -> {dates[-1]})...")

    qh_rows = []
    daily_rows = []
    spec = BatterySpec()
    for i, d in enumerate(dates):
        if i % 10 == 0:
            print(f"  day {i+1}/{len(dates)}: {d}")
        issue = issue_time_for(d)

        # Skip days where any input is incomplete.
        da_fc = xgboost_price_predict_full(df, issue)
        id_fc = xgboost_intraday_predict_full(df, issue)
        idx = da_fc.index
        if not idx.equals(id_fc.index):
            print(f"    skip {d}: forecast indices disagree")
            continue
        da_actual = df[DA_COL].reindex(idx).to_numpy()
        id_actual = df[ID_COL].reindex(idx).to_numpy()
        if np.isnan(da_actual).any() or np.isnan(id_actual).any():
            print(f"    skip {d}: missing actuals")
            continue

        # Naive baselines = yesterday-same-quarter-hour of the SAME market.
        prev_idx = idx - pd.Timedelta(days=1)
        da_naive = df[DA_COL].reindex(prev_idx).to_numpy()
        id_naive = df[ID_COL].reindex(prev_idx).to_numpy()
        if np.isnan(da_naive).any() or np.isnan(id_naive).any():
            print(f"    skip {d}: missing naive baseline")
            continue

        # Per-quarter-hour rows
        for j, ts in enumerate(idx):
            qh_rows.append({
                "issue_date": str(d),
                "target_ts": ts,
                "da_actual": da_actual[j],
                "id_actual": id_actual[j],
                "da_naive": da_naive[j],
                "id_naive": id_naive[j],
                "da_p10": float(da_fc["p10"].iloc[j]),
                "da_p50": float(da_fc["p50"].iloc[j]),
                "da_p90": float(da_fc["p90"].iloc[j]),
                "id_p10": float(id_fc["p10"].iloc[j]),
                "id_p50": float(id_fc["p50"].iloc[j]),
                "id_p90": float(id_fc["p90"].iloc[j]),
            })

        # Daily dispatch on each market: forecast vs naive vs perfect-foresight
        da_oracle = dispatch_pnl(da_actual, da_actual, da_actual, spec)["net_pnl"]
        da_naive_pnl = dispatch_pnl(da_naive, da_naive, da_actual, spec)["net_pnl"]
        da_model_pnl = dispatch_pnl(
            da_fc["p50"].to_numpy(), da_fc["p50"].to_numpy(), da_actual, spec,
        )["net_pnl"]

        id_oracle = dispatch_pnl(id_actual, id_actual, id_actual, spec)["net_pnl"]
        id_naive_pnl = dispatch_pnl(id_naive, id_naive, id_actual, spec)["net_pnl"]
        id_model_pnl = dispatch_pnl(
            id_fc["p50"].to_numpy(), id_fc["p50"].to_numpy(), id_actual, spec,
        )["net_pnl"]

        daily_rows.append({
            "issue_date": str(d),
            "da_mae_model":  float(np.abs(da_actual - da_fc["p50"].to_numpy()).mean()),
            "da_mae_naive":  float(np.abs(da_actual - da_naive).mean()),
            "id_mae_model":  float(np.abs(id_actual - id_fc["p50"].to_numpy()).mean()),
            "id_mae_naive":  float(np.abs(id_actual - id_naive).mean()),
            "da_coverage":   float(
                ((da_actual >= da_fc["p10"].to_numpy())
                 & (da_actual <= da_fc["p90"].to_numpy())).mean()
            ),
            "id_coverage":   float(
                ((id_actual >= id_fc["p10"].to_numpy())
                 & (id_actual <= id_fc["p90"].to_numpy())).mean()
            ),
            "da_oracle_pnl": da_oracle,
            "da_naive_pnl":  da_naive_pnl,
            "da_model_pnl":  da_model_pnl,
            "id_oracle_pnl": id_oracle,
            "id_naive_pnl":  id_naive_pnl,
            "id_model_pnl":  id_model_pnl,
        })

    qh = pd.DataFrame(qh_rows)
    daily = pd.DataFrame(daily_rows)
    OUT_QH.parent.mkdir(parents=True, exist_ok=True)
    qh.to_csv(OUT_QH, index=False)
    daily.to_csv(OUT_DAILY, index=False)
    print(f"\nWrote {OUT_QH}  ({len(qh):,} rows)")
    print(f"Wrote {OUT_DAILY}  ({len(daily)} rows)")
    print()

    # ---- Summary ----
    n_days = len(daily)
    print("=" * 70)
    print(f"DA vs Intraday — same architecture, two markets ({n_days} days)")
    print("=" * 70)
    da_mae = daily["da_mae_model"].mean()
    id_mae = daily["id_mae_model"].mean()
    da_naive_mae = daily["da_mae_naive"].mean()
    id_naive_mae = daily["id_mae_naive"].mean()
    da_skill = (1 - da_mae / da_naive_mae) * 100
    id_skill = (1 - id_mae / id_naive_mae) * 100
    print("\nP50 MAE (EUR/MWh):")
    print(f"  Day-ahead:   model={da_mae:>6.2f}  naive={da_naive_mae:>6.2f}  skill={da_skill:>+5.1f} %")
    print(f"  Intraday:    model={id_mae:>6.2f}  naive={id_naive_mae:>6.2f}  skill={id_skill:>+5.1f} %")
    print("\n80%-band coverage:")
    print(f"  Day-ahead:   {daily['da_coverage'].mean()*100:>5.1f} %  (target 80 %)")
    print(f"  Intraday:    {daily['id_coverage'].mean()*100:>5.1f} %  (target 80 %)")
    print("\nBattery dispatch P&L:")
    s = daily[
        ["da_oracle_pnl", "da_naive_pnl", "da_model_pnl",
         "id_oracle_pnl", "id_naive_pnl", "id_model_pnl"]
    ].sum()
    print("  Day-ahead market:")
    print(f"    perfect-foresight: EUR {s.da_oracle_pnl:>10,.0f}")
    print(f"    naive:             EUR {s.da_naive_pnl:>10,.0f}  ({s.da_naive_pnl/s.da_oracle_pnl*100:.1f} % of oracle)")
    print(f"    model P50:         EUR {s.da_model_pnl:>10,.0f}  ({s.da_model_pnl/s.da_oracle_pnl*100:.1f} % of oracle)")
    print(f"    model uplift vs naive: EUR {s.da_model_pnl - s.da_naive_pnl:>+8,.0f}")
    print("  Intraday market:")
    print(f"    perfect-foresight: EUR {s.id_oracle_pnl:>10,.0f}")
    print(f"    naive:             EUR {s.id_naive_pnl:>10,.0f}  ({s.id_naive_pnl/s.id_oracle_pnl*100:.1f} % of oracle)")
    print(f"    model P50:         EUR {s.id_model_pnl:>10,.0f}  ({s.id_model_pnl/s.id_oracle_pnl*100:.1f} % of oracle)")
    print(f"    model uplift vs naive: EUR {s.id_model_pnl - s.id_naive_pnl:>+8,.0f}")


if __name__ == "__main__":
    main()
