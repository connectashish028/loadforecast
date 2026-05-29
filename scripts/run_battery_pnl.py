"""Run the battery-dispatch P&L simulation across the holdout (Mar–Apr 2026).

Compares 4 strategies on the same 10 MW / 20 MWh battery:

  1. ORACLE      — perfect-foresight dispatch on actual prices (theoretical max)
  2. NAIVE       — yesterday-same-quarter-hour as the forecast
  3. MODEL_P50   — model's median forecast, used for both charge & discharge
  4. MODEL_P10/90 — model's P10 for charging, P90 for discharging
                    (the recommended use of the quantile bands)

Output: backtest_results/battery_pnl_daily.csv  (one row per delivery day)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from loadforecast.dispatch import BatterySpec, dispatch_pnl

BACKTEST_CSV = Path("backtest_results/price_quantile_holdout.csv")
OUT_CSV = Path("backtest_results/battery_pnl_daily.csv")


def main() -> None:
    bt = pd.read_csv(BACKTEST_CSV, parse_dates=["target_ts"])
    spec = BatterySpec()

    rows = []
    for issue_date, day in bt.groupby("issue_date"):
        if len(day) != 96:
            continue
        actual = day["y_true"].to_numpy()
        p10 = day["p10"].to_numpy()
        p50 = day["p50"].to_numpy()
        p90 = day["p90"].to_numpy()
        naive = day["naive_1d"].to_numpy()

        # If naive has any NaN (e.g. first holdout day with no D-1 actuals),
        # skip — we can't run dispatch on partial data.
        if np.isnan(naive).any():
            continue

        oracle = dispatch_pnl(actual, actual, actual, spec)
        naive_run = dispatch_pnl(naive, naive, actual, spec)
        model_p50 = dispatch_pnl(p50, p50, actual, spec)
        model_band = dispatch_pnl(p10, p90, actual, spec)

        rows.append({
            "issue_date": issue_date,
            "oracle_pnl": oracle["net_pnl"],
            "naive_pnl": naive_run["net_pnl"],
            "model_p50_pnl": model_p50["net_pnl"],
            "model_band_pnl": model_band["net_pnl"],
            "actual_spread": float(actual.max() - actual.min()),
        })

    df = pd.DataFrame(rows).sort_values("issue_date").reset_index(drop=True)
    df["pct_oracle_naive"] = df["naive_pnl"] / df["oracle_pnl"]
    df["pct_oracle_p50"] = df["model_p50_pnl"] / df["oracle_pnl"]
    df["pct_oracle_band"] = df["model_band_pnl"] / df["oracle_pnl"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}  ({len(df)} days)")
    print()

    # ---- Summary table ----
    summary_eur = df[["oracle_pnl", "naive_pnl", "model_p50_pnl", "model_band_pnl"]].sum()
    print(f"Total P&L over {len(df)} days (€):")
    print(f"  Perfect-foresight  : {summary_eur['oracle_pnl']:>10,.0f}")
    print(f"  Naive yesterday    : {summary_eur['naive_pnl']:>10,.0f}  "
          f"({summary_eur['naive_pnl']/summary_eur['oracle_pnl']*100:.1f} % of perfect-foresight)")
    print(f"  Model P50 only     : {summary_eur['model_p50_pnl']:>10,.0f}  "
          f"({summary_eur['model_p50_pnl']/summary_eur['oracle_pnl']*100:.1f} % of perfect-foresight)")
    print(f"  Model P10 / P90    : {summary_eur['model_band_pnl']:>10,.0f}  "
          f"({summary_eur['model_band_pnl']/summary_eur['oracle_pnl']*100:.1f} % of perfect-foresight)")
    print()

    # ---- Per-day per-MWh-of-capacity ----
    cap = spec.capacity_mwh
    n_days = len(df)
    eur_per_mwh_per_day = summary_eur / (cap * n_days)
    print("€ per day per MWh of battery capacity:")
    print(f"  Perfect-foresight  : {eur_per_mwh_per_day['oracle_pnl']:>7.2f}")
    print(f"  Naive yesterday    : {eur_per_mwh_per_day['naive_pnl']:>7.2f}")
    print(f"  Model P50          : {eur_per_mwh_per_day['model_p50_pnl']:>7.2f}")
    print(f"  Model P10/P90      : {eur_per_mwh_per_day['model_band_pnl']:>7.2f}")
    print()

    # ---- Uplift framing ----
    naive_total = summary_eur["naive_pnl"]
    band_total = summary_eur["model_band_pnl"]
    p50_total = summary_eur["model_p50_pnl"]
    print("Uplift over naive:")
    print(f"  Model P50          : {(p50_total - naive_total):>+9,.0f} €  "
          f"({(p50_total / naive_total - 1) * 100:+.1f} %)")
    print(f"  Model P10/P90      : {(band_total - naive_total):>+9,.0f} €  "
          f"({(band_total / naive_total - 1) * 100:+.1f} %)")


if __name__ == "__main__":
    main()
