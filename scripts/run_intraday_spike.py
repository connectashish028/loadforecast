"""Phase B.3 — intraday re-forecast spike.

Proof-of-concept that the production forecasting infrastructure
(feature pipeline, quantile model, leakage tests, drift monitor)
extends to a finer issue-time cadence than the current single D-1 12:00
gate. Calls the existing XGBoost price model at four issue times within
the day-before-delivery window:

    D-1 06:00 Berlin   (six hours before the gate)
    D-1 12:00 Berlin   (the production gate — baseline)
    D-1 18:00 Berlin   (six hours after the gate)
    D-1 21:00 Berlin   (three hours before midnight)

For a curated set of delivery days. Saves per-quarter-hour forecasts
(all 4 issue times × 96 quarter-hours per day) and a per-day MAE
summary so the dashboard can both plot the forecast-evolution chart
and surface the per-issue MAE table.

What this proves:
  - The feature builder + predict wrapper handle non-noon issue times
    without code changes — `build_target_day_features(df, T)` resolves
    the target day from T's local date and applies the same leakage
    rules.
  - Per-issue forecasts differ as later-D-1 actuals enter the feature
    window (rolling stats, lagged residual-error features).

What this does NOT prove:
  - That the model extracts much value from the cadence. It was
    trained on a single D-1 12:00 issue, so the predictive distribution
    barely shifts across issue times. A production intraday model would
    need (a) training windows sampled at multiple issue times, and
    (b) intraday-specific features (auction prints, NWP-update recency,
    recent imbalance signals).

Run from repo root:
    PYTHONPATH=src python scripts/run_intraday_spike.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from loadforecast.backtest import load_smard_15min
from loadforecast.models.predict import xgboost_price_predict_full

PARQUET = "smard_merged_15min.parquet"
PRICE_COL = "price__germany_luxembourg"
NAIVE_LAG_DAYS = 1

FORECASTS_CSV = Path("backtest_results/xgboost_intraday_spike_forecasts.csv")
SUMMARY_CSV = Path("backtest_results/xgboost_intraday_spike_summary.csv")

# Mix of normal days, high-VRE days, and an extreme tail day so the
# spike can be inspected across regimes.
SPIKE_DAYS: list[date] = [
    date(2026, 4, 15),
    date(2026, 4, 18),
    date(2026, 5, 1),    # the -500 EUR/MWh holiday + PV record
    date(2026, 5, 10),
    date(2026, 5, 20),
]

ISSUE_OFFSETS: list[tuple[str, int]] = [
    ("D-1 06:00", 6),
    ("D-1 12:00 (gate)", 12),
    ("D-1 18:00", 18),
    ("D-1 21:00", 21),
]


def main() -> None:
    print(f"Loading parquet: {PARQUET}")
    df = load_smard_15min(PARQUET)
    berlin = ZoneInfo("Europe/Berlin")

    forecast_rows = []
    summary_rows = []

    for delivery in SPIKE_DAYS:
        actual_start_utc = pd.Timestamp(delivery, tz=berlin).tz_convert("UTC")
        actual_end_utc = (
            pd.Timestamp(delivery, tz=berlin) + pd.Timedelta(days=1)
        ).tz_convert("UTC")
        # pandas loc[a:b] is inclusive on both sides; trim to 96 quarter-hours
        actual = df[PRICE_COL].loc[
            actual_start_utc:actual_end_utc - pd.Timedelta(minutes=15)
        ]
        if len(actual) < 96 or actual.isna().any():
            print(f"  skip {delivery}: no realised prices yet ({len(actual)} rows, "
                  f"{int(actual.isna().sum())} NaN)")
            continue

        # Naive baseline = D-1 same quarter-hour
        prev_start = actual_start_utc - pd.Timedelta(days=NAIVE_LAG_DAYS)
        prev_end = actual_end_utc - pd.Timedelta(days=NAIVE_LAG_DAYS)
        naive = df[PRICE_COL].loc[prev_start:prev_end - pd.Timedelta(minutes=15)]
        if len(naive) < 96 or naive.isna().any():
            print(f"  skip {delivery}: naive baseline incomplete")
            continue

        d_minus_1 = pd.Timestamp(delivery, tz=berlin) - pd.Timedelta(days=1)
        print(f"  delivery {delivery} : actual range "
              f"[{float(actual.min()):.1f}, {float(actual.max()):.1f}] EUR/MWh")
        for label, hour_local in ISSUE_OFFSETS:
            issue_local = d_minus_1.replace(hour=hour_local)
            issue_utc = issue_local.tz_convert("UTC")
            fc = xgboost_price_predict_full(df, issue_utc)
            # Reindex actual + naive to the forecast's target_ts index
            y = actual.reindex(fc.index).to_numpy()
            y_naive = naive.set_axis(fc.index).to_numpy()
            err_model = np.abs(y - fc["p50"].to_numpy())
            err_naive = np.abs(y - y_naive)
            mae = float(np.nanmean(err_model))
            mae_naive = float(np.nanmean(err_naive))
            in_band = ((y >= fc["p10"].to_numpy()) & (y <= fc["p90"].to_numpy())).mean()
            summary_rows.append({
                "delivery_date": delivery,
                "issue_label": label,
                "issue_time_utc": issue_utc,
                "mae_eur": mae,
                "naive_mae_eur": mae_naive,
                "skill_vs_naive_pct": (1 - mae / mae_naive) * 100,
                "band_coverage_pct": float(in_band) * 100,
                "p50_mean": float(fc["p50"].mean()),
                "band_width_mean": float((fc["p90"] - fc["p10"]).mean()),
            })
            for ts, row in fc.iterrows():
                forecast_rows.append({
                    "delivery_date": delivery,
                    "issue_label": label,
                    "issue_time_utc": issue_utc,
                    "target_ts": ts,
                    "p10": float(row["p10"]),
                    "p50": float(row["p50"]),
                    "p90": float(row["p90"]),
                    "y_true": float(actual.loc[ts]) if ts in actual.index else np.nan,
                    "naive": float(naive.set_axis(fc.index).loc[ts]),
                })
            print(f"    {label:<18}  MAE={mae:>6.2f}  naive_skill={(1-mae/mae_naive)*100:>+5.1f}%  "
                  f"coverage={float(in_band)*100:>5.1f}%")

    pd.DataFrame(forecast_rows).to_csv(FORECASTS_CSV, index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print()
    print(f"Wrote {FORECASTS_CSV}  ({len(forecast_rows)} rows)")
    print(f"Wrote {SUMMARY_CSV}    ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
