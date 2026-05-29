"""Calibrate the extreme-tail clip (Price M10).

Domain rule, not ML. The pinball-loss head structurally can't reach
−500 EUR/MWh on rare regime days (the conditional median given features
stays around the conditional median). This calibrates a single offset
to apply at the PV-trough hours on holiday/weekend × top-1%-VRE days.

Trigger (computed from features the model already has):
  - delivery_day is a federal holiday OR Saturday/Sunday
  - max(vre_percentile_decoder) > VRE_PCTILE_TRIGGER (forecast VRE
    exceeds the 90-day rolling q90 by some margin)

Calibration: on the 2024-2025 window (out-of-holdout), find days where
min(actual) < EXTREME_MIN_THRESHOLD AND the trigger fires. Δ = median
gap between v4 P50 trough and actual trough across those days.

Output: writes config to `model_checkpoints/price_quantile_v4/extreme_clip.json`
which the predict-time wrapper reads.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import holidays as hols
import numpy as np
import pandas as pd

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.dataset import FeatureScaler
from loadforecast.models.price_dataset import build_price_window

PARQUET = "smard_merged_15min.parquet"
MODEL_DIR = Path("model_checkpoints/price_quantile_v4")
OUT_PATH = MODEL_DIR / "extreme_clip.json"

# Trigger thresholds (same definition used at inference).
VRE_PCTILE_TRIGGER = 1.2     # forecast VRE > 120% of 90-day q90
EXTREME_MIN_THRESHOLD = -50.0  # EUR/MWh — bottom ~2 % of days

# Decoder feature index for vre_percentile (must match price_dataset.py).
VRE_PCTILE_COL_IDX = 4

# How many trough hours to clip. The actual negative-price episodes
# typically last 4-6 hours around mid-day; 24 quarter-hours = 6 hours.
N_TROUGH_QH = 24


def _drange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _load_model():
    from tensorflow import keras
    z = np.load(MODEL_DIR / "scaler.npz")
    s = FeatureScaler(
        enc_mean=z["enc_mean"], enc_std=z["enc_std"],
        dec_mean=z["dec_mean"], dec_std=z["dec_std"],
        y_mean=float(z["y_mean"]), y_std=float(z["y_std"]),
    )
    m = keras.models.load_model(MODEL_DIR / "model.keras", compile=False)
    return m, s


def _predict_one(model, scaler, df, issue):
    w = build_price_window(df, issue, include_weather=True)
    if np.isnan(w.X_enc).any() or np.isnan(w.X_dec).any():
        return None
    Xe, Xd = scaler.transform(w.X_enc[None], w.X_dec[None])
    raw = model.predict([Xe, Xd], verbose=0)[0]
    y = scaler.inverse_y(raw)
    return {
        "p10": y[:, 0], "p50": y[:, 1], "p90": y[:, 2],
        "y_true": w.y_price,
        "vre_percentile_max": float(w.X_dec[:, VRE_PCTILE_COL_IDX].max()),
    }


def main() -> None:
    print("Loading parquet + model...")
    df = load_smard_15min(PARQUET)
    model, scaler = _load_model()
    de_hols = hols.country_holidays("DE", years=range(2022, 2027))

    # Calibration window: out-of-holdout (holdout is Mar-Apr 2026).
    cal_start = date(2024, 1, 8)
    cal_end = date(2025, 12, 31)
    print(f"Calibration window: {cal_start} -> {cal_end}")

    rows = []
    for d in _drange(cal_start, cal_end):
        is_weekend = d.weekday() in (5, 6)
        is_holiday = d in de_hols
        if not (is_weekend or is_holiday):
            continue
        issue = issue_time_for(d)
        out = _predict_one(model, scaler, df, issue)
        if out is None or np.isnan(out["y_true"]).any():
            continue
        if out["vre_percentile_max"] < VRE_PCTILE_TRIGGER:
            continue
        actual_min = float(out["y_true"].min())
        if actual_min >= EXTREME_MIN_THRESHOLD:
            continue
        # Trough metric: p50 at the 24 quarter-hours where p50 is lowest
        # (mirrors what the wrapper will clip).
        trough_idx = np.argsort(out["p50"])[:N_TROUGH_QH]
        p50_trough = float(out["p50"][trough_idx].mean())
        actual_trough = float(out["y_true"][trough_idx].mean())
        gap = p50_trough - actual_trough  # positive = model under-predicts the negative
        rows.append({
            "date": d,
            "is_holiday": is_holiday,
            "is_weekend": is_weekend,
            "vre_pctile_max": out["vre_percentile_max"],
            "actual_min": actual_min,
            "actual_trough_mean": actual_trough,
            "p50_trough_mean": p50_trough,
            "gap": gap,
        })

    if not rows:
        print("No calibration days match the trigger. Loosen thresholds.")
        return

    cal = pd.DataFrame(rows)
    print(f"\nCalibration days matching trigger: {len(cal)}")
    print(cal[["date", "actual_min", "p50_trough_mean", "actual_trough_mean", "gap"]].to_string(index=False))

    delta = float(cal["gap"].median())
    delta_p25 = float(cal["gap"].quantile(0.25))
    delta_p75 = float(cal["gap"].quantile(0.75))

    print(f"\n  median gap (Delta): {delta:.1f} EUR/MWh")
    print(f"  IQR gap:            [{delta_p25:.1f}, {delta_p75:.1f}]")

    config = {
        "delta_eur_mwh": delta,
        "vre_pctile_trigger": VRE_PCTILE_TRIGGER,
        "extreme_min_threshold": EXTREME_MIN_THRESHOLD,
        "n_trough_qh": N_TROUGH_QH,
        "p10_multiplier": 1.5,
        "calibration_window": f"{cal_start} -> {cal_end}",
        "n_calibration_days": int(len(cal)),
        "calibration_dates": [str(d) for d in cal["date"].tolist()],
        "iqr_gap_p25_p75": [delta_p25, delta_p75],
    }
    OUT_PATH.write_text(json.dumps(config, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
