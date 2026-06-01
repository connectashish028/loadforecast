"""Apply split-conformal band calibration to the XGBoost price model
and re-run the risk-aware dispatch comparison.

Hypothesis: the B.1 risk-aware experiment failed because the model's
quantile bands have ~71 % coverage vs 80 % nominal. Conformal calibration
widens the bands to hit 80 % with a finite-sample guarantee; with
calibrated bands, the width-penalty / skip-wide policies should at
least become *informative* (operating on signal rather than noise).

Split: first CAL_DAYS days of the 61-day holdout = calibration set,
remaining = test set. Two variants (marginal + adaptive) compared.

Outputs:
- backtest_results/xgboost_conformal_summary.csv (one row per variant)
- backtest_results/xgboost_conformal_policies.csv (per-day P&L for all
  five risk-aware policies, evaluated on the test split, with both
  un-calibrated and calibrated bands)

Run from repo root:
    PYTHONPATH=src python scripts/run_conformal_calibration.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from loadforecast import conformal
from loadforecast.dispatch import BatterySpec, dispatch_pnl, risk_aware_signals

BACKTEST_CSV = Path("backtest_results/xgboost_price_holdout.csv")
SUMMARY_CSV = Path("backtest_results/xgboost_conformal_summary.csv")
POLICIES_CSV = Path("backtest_results/xgboost_conformal_policies.csv")

CAL_DAYS = 20      # first 20 days of the 61-day holdout
TARGET_ALPHA = 0.20  # 80 % nominal bands

POLICIES = [
    ("greedy_p50", {}),
    ("p10p90", {}),
    ("robust_p90p10", {}),
    ("width_penalty", {"lambda_width": 0.10}),
    ("skip_wide_slots", {"bandwidth_quantile": 0.80}),
]


def coverage(y: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    mask = ~(np.isnan(y) | np.isnan(p10) | np.isnan(p90))
    return float(((y[mask] >= p10[mask]) & (y[mask] <= p90[mask])).mean())


def run_policies(test_df: pd.DataFrame, p10_col: str, p50_col: str, p90_col: str) -> dict:
    """Run all 5 policies on test_df using the given band columns; return totals."""
    spec = BatterySpec()
    totals = {f"pnl_{name}": 0.0 for name, _ in POLICIES}
    naive_total = 0.0
    oracle_total = 0.0
    daily_uplifts: dict[str, list[float]] = {f"pnl_{name}": [] for name, _ in POLICIES}
    for _, day in test_df.groupby("issue_date"):
        if len(day) != 96:
            continue
        actual = day["y_true"].to_numpy()
        p10 = day[p10_col].to_numpy()
        p50 = day[p50_col].to_numpy()
        p90 = day[p90_col].to_numpy()
        naive = day["naive_1d"].to_numpy()
        if np.isnan(naive).any():
            continue
        oracle = dispatch_pnl(actual, actual, actual, spec)["net_pnl"]
        naive_pnl = dispatch_pnl(naive, naive, actual, spec)["net_pnl"]
        oracle_total += oracle
        naive_total += naive_pnl
        for name, kwargs in POLICIES:
            charge, discharge = risk_aware_signals(p10, p50, p90, name, **kwargs)
            pnl = dispatch_pnl(charge, discharge, actual, spec)["net_pnl"]
            totals[f"pnl_{name}"] += pnl
            daily_uplifts[f"pnl_{name}"].append(pnl - naive_pnl)
    return {
        "totals": totals,
        "naive_total": naive_total,
        "oracle_total": oracle_total,
        "daily_uplifts": {k: np.array(v) for k, v in daily_uplifts.items()},
    }


def main() -> None:
    bt = pd.read_csv(BACKTEST_CSV, parse_dates=["target_ts"])
    bt["issue_date"] = pd.to_datetime(bt["issue_date"])

    # Split: first CAL_DAYS issue_dates = calibration, rest = test
    all_dates = sorted(bt["issue_date"].unique())
    cal_dates = set(all_dates[:CAL_DAYS])
    test_dates = set(all_dates[CAL_DAYS:])
    cal_df = bt[bt["issue_date"].isin(cal_dates)].copy()
    test_df = bt[bt["issue_date"].isin(test_dates)].copy()
    print(f"Calibration: {len(cal_dates)} days, {len(cal_df):,} quarter-hours")
    print(f"Test:        {len(test_dates)} days, {len(test_df):,} quarter-hours")
    print()

    # Fit both conformal variants on the calibration set
    cal_y = cal_df["y_true"].to_numpy()
    cal_p10 = cal_df["p10"].to_numpy()
    cal_p90 = cal_df["p90"].to_numpy()
    marg = conformal.fit(cal_y, cal_p10, cal_p90, target_alpha=TARGET_ALPHA, variant="marginal")
    adapt = conformal.fit(cal_y, cal_p10, cal_p90, target_alpha=TARGET_ALPHA, variant="adaptive")
    print(f"Marginal q_hat:  {marg.q_hat:.2f} EUR/MWh (cal coverage pre: {marg.cal_coverage_pre*100:.1f} %)")
    print(f"Adaptive q_hat:  {adapt.q_hat:.4f} (multiplier on band width; cal coverage pre: {adapt.cal_coverage_pre*100:.1f} %)")
    print()

    # Apply both calibrations to the test set
    test_y = test_df["y_true"].to_numpy()
    test_p10 = test_df["p10"].to_numpy()
    test_p90 = test_df["p90"].to_numpy()

    cov_pre = coverage(test_y, test_p10, test_p90)
    p10_m, p90_m = conformal.apply(test_p10, test_p90, marg)
    cov_m = coverage(test_y, p10_m, p90_m)
    p10_a, p90_a = conformal.apply(test_p10, test_p90, adapt)
    cov_a = coverage(test_y, p10_a, p90_a)
    print("Test-set 80 %-band coverage:")
    print(f"  uncalibrated:        {cov_pre*100:.1f} %  (nominal 80 %)")
    print(f"  marginal-calibrated: {cov_m*100:.1f} %")
    print(f"  adaptive-calibrated: {cov_a*100:.1f} %")
    print()

    # Run the 5 policies on test set, three bands: uncalibrated / marginal / adaptive
    test_df = test_df.copy()
    test_df["p10_m"] = test_df["p10"] - marg.q_hat
    test_df["p90_m"] = test_df["p90"] + marg.q_hat
    width = test_df["p90"] - test_df["p10"]
    test_df["p10_a"] = test_df["p10"] - adapt.q_hat * width
    test_df["p90_a"] = test_df["p90"] + adapt.q_hat * width

    print("Risk-aware dispatch comparison on the test split "
          f"({len(test_dates)} days):")
    print()
    print(f"  {'bands':<14} {'policy':<24} {'total':>11} {'pct_oracle':>11} "
          f"{'vs_naive':>11} {'worst_day':>11} {'lost':>7}")
    print(f"  {'-'*14} {'-'*24} {'-'*11} {'-'*11} {'-'*11} {'-'*11} {'-'*7}")
    rows = []
    for band_label, p10c, p50c, p90c in [
        ("uncalibrated", "p10",   "p50", "p90"),
        ("marginal",     "p10_m", "p50", "p90_m"),
        ("adaptive",     "p10_a", "p50", "p90_a"),
    ]:
        r = run_policies(test_df, p10c, p50c, p90c)
        oracle_total = r["oracle_total"]
        naive_total = r["naive_total"]
        for name, _ in POLICIES:
            col = f"pnl_{name}"
            total = r["totals"][col]
            pct = total / oracle_total * 100
            uplift = total - naive_total
            uplifts = r["daily_uplifts"][col]
            worst = float(uplifts.min())
            lost = int((uplifts < 0).sum())
            print(f"  {band_label:<14} {name:<24} {total:>11,.0f} "
                  f"{pct:>10.1f}% {uplift:>+11,.0f} {worst:>+11,.0f} "
                  f"{lost:>3d}/{len(uplifts):d}")
            rows.append({
                "bands": band_label,
                "policy": name,
                "total_pnl": total,
                "pct_oracle": pct,
                "uplift_vs_naive": uplift,
                "worst_day_vs_naive": worst,
                "days_lost": lost,
                "n_days": len(uplifts),
            })

    pd.DataFrame(rows).to_csv(POLICIES_CSV, index=False)

    # Summary row: which (bands, policy) combo wins on each metric?
    pd.DataFrame([
        {"metric": "test_days", "value": len(test_dates)},
        {"metric": "cal_days", "value": len(cal_dates)},
        {"metric": "target_alpha", "value": TARGET_ALPHA},
        {"metric": "marginal_q_hat_eur", "value": marg.q_hat},
        {"metric": "adaptive_q_hat_mult", "value": adapt.q_hat},
        {"metric": "test_coverage_uncalibrated", "value": cov_pre},
        {"metric": "test_coverage_marginal", "value": cov_m},
        {"metric": "test_coverage_adaptive", "value": cov_a},
    ]).to_csv(SUMMARY_CSV, index=False)
    print()
    print(f"Wrote {POLICIES_CSV}")
    print(f"Wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
