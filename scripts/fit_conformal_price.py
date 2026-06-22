"""Fit split-conformal calibration for the production price model's P10/P90
bands and save it to the checkpoint as conformal.json.

The raw XGBoost quantile bands are under-calibrated (~71 % coverage vs the
80 % nominal). Marginal split-conformal widens them by a single scalar q_hat
with a finite-sample coverage guarantee. P50 is untouched, so dispatch P&L
is unchanged — this only makes the uncertainty bands trustworthy.

Calibration set: a recent trailing window of realised delivery days. This
is the production-correct choice — the bands' miscalibration is regime-
dependent (well-calibrated on the calm Jan–Feb validation window, under-
covered on the volatile Mar–Apr holdout), so a static calibration from an
old window is a no-op. Calibrating on recent realised data captures the
current regime and applies forward. Forward-tested on the most recent days.

Run:  PYTHONPATH=src python scripts/fit_conformal_price.py
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from loadforecast import conformal
from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.predict import xgboost_price_predict_full

PARQUET = "smard_merged_15min.parquet"
PRICE_COL = "price__germany_luxembourg"
OUT = Path("model_checkpoints/xgboost_price_v1/conformal.json")

CAL = (date(2026, 3, 1), date(2026, 5, 31))     # recent trailing window (current regime)
TEST = (date(2026, 6, 1), date(2026, 6, 20))    # forward slice — verify only
TARGET_ALPHA = 0.20                              # 80 % bands


def _drange(a, b):
    out, d = [], a
    while d <= b:
        out.append(d)
        d += timedelta(days=1)
    return out


def _collect(df, dates):
    """Return flat (y, p10, p90) over `dates` using RAW (uncalibrated) bands."""
    ys, p10s, p90s = [], [], []
    for d in dates:
        fc = xgboost_price_predict_full(df, issue_time_for(d), apply_conformal=False)
        y = df[PRICE_COL].reindex(fc.index).to_numpy()
        if np.isnan(y).any() or fc["p50"].isna().any():
            continue
        ys.append(y)
        p10s.append(fc["p10"].to_numpy())
        p90s.append(fc["p90"].to_numpy())
    return np.concatenate(ys), np.concatenate(p10s), np.concatenate(p90s)


def _coverage(y, p10, p90):
    return float(((y >= p10) & (y <= p90)).mean())


def main() -> None:
    df = load_smard_15min(PARQUET)

    print(f"Calibration window {CAL[0]}..{CAL[1]} ...")
    yc, p10c, p90c = _collect(df, _drange(*CAL))
    cal = conformal.fit(yc, p10c, p90c, target_alpha=TARGET_ALPHA, variant="marginal")
    print(f"  n={cal.cal_size}  pre-coverage={cal.cal_coverage_pre*100:.1f} %  "
          f"q_hat={cal.q_hat:.2f} EUR/MWh")

    OUT.write_text(json.dumps({
        "variant": cal.variant,
        "target_alpha": cal.target_alpha,
        "q_hat": cal.q_hat,
        "cal_size": cal.cal_size,
        "cal_coverage_pre": cal.cal_coverage_pre,
        "cal_window": [CAL[0].isoformat(), CAL[1].isoformat()],
        "note": "marginal split-conformal; widens P10/P90 by +/- q_hat; P50 untouched",
    }, indent=2))
    print(f"  wrote {OUT}")

    # Forward verify: coverage before vs after on the most recent days.
    print(f"\nForward verify on {TEST[0]}..{TEST[1]} ...")
    yt, p10t, p90t = _collect(df, _drange(*TEST))
    cov_pre = _coverage(yt, p10t, p90t)
    p10a, p90a = conformal.apply(p10t, p90t, cal)
    cov_post = _coverage(yt, p10a, p90a)
    mean_width_pre = float((p90t - p10t).mean())
    mean_width_post = float((p90a - p10a).mean())
    print(f"  holdout 80%-band coverage: {cov_pre*100:.1f} % -> {cov_post*100:.1f} %  (target 80 %)")
    print(f"  mean band width:           {mean_width_pre:.1f} -> {mean_width_post:.1f} EUR/MWh "
          f"(+{mean_width_post-mean_width_pre:.1f})")


if __name__ == "__main__":
    main()
