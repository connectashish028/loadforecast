"""Train the day-ahead price forecaster.

Mirrors train_lstm_quantile.py (the load model) but with:
  - target = raw price (not load residual)
  - encoder feature mix from price_dataset.py
  - training window starting 2024-01-01 (avoid the 2022 €500 outlier era)

v3 (current): adds feature-dropout augmentation on `tso_vre_fc` so the
model learns to fall back to weather + load + calendar when SMARD's VRE
day-ahead forecast hasn't published yet (production pattern).

Saves to model_checkpoints/price_quantile_v3/.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import holidays as hols
import numpy as np
import pandas as pd

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.dataset import FeatureScaler
from loadforecast.models.lstm_quantile import (
    QUANTILES,
    build_lstm_quantile,
    compile_lstm_quantile,
)
from loadforecast.models.price_dataset import (
    PRICE_DEC_FEATURE_NAMES,
    PRICE_ENC_FEATURE_NAMES,
    build_price_dataset,
)


def _sample_weight(issue_time, federal_holidays) -> float:
    """Per-window weight: 0.5× for 2022-2023 (recovered tail data,
    down-weighted to avoid spike domination), 3× for federal holidays
    and Sundays (the rare regime that blew up on May 1, 2026)."""
    delivery_local = (issue_time.tz_convert("Europe/Berlin").normalize()
                       + pd.Timedelta(days=1))
    w = 1.0
    if delivery_local.year in (2022, 2023):
        w *= WEIGHT_OLD_YEARS
    if delivery_local.weekday() == 6 or delivery_local.date() in federal_holidays:
        w *= WEIGHT_HOLIDAY_OR_SUN
    return w

PARQUET = "smard_merged_15min.parquet"
OUT_DIR = Path("model_checkpoints/price_quantile_v4")

# Feature-dropout augmentation: fraction of training windows where we mask
# tso_vre_fc (set to 0 + present-flag to 0). Teaches the model to handle
# the case where SMARD's VRE day-ahead forecast hasn't published yet —
# the production-grade pattern. Tuned conservatively at 30 %; higher
# (~50 %) trades full-feature accuracy for robustness.
VRE_DROPOUT_FRAC = 0.30
# Decoder feature indices (must match PRICE_DEC_FEATURE_NAMES order).
VRE_FC_COL_IDX = 1   # tso_vre_fc
VRE_PRESENT_COL_IDX = 2  # tso_vre_fc_present
VRE_RATIO_COL_IDX = 3    # vre_to_load_ratio (engineered, derives from tso_vre_fc)
VRE_PCTILE_COL_IDX = 4   # vre_percentile    (engineered, derives from tso_vre_fc)

# M9: sample weighting strategy. We re-include 2022-2023 to recover the
# negative-price events that v3's 2024+ window cut out, but down-weight
# them to avoid being dominated by 2022's €500 spike outliers. We also
# upweight federal holidays + Sundays — that's exactly the regime that
# blew up on May 1, 2026 (federal holiday + 54 GW PV record → −500 €/MWh).
WEIGHT_OLD_YEARS = 0.5     # multiplier for 2022-2023 windows
WEIGHT_HOLIDAY_OR_SUN = 3.0  # multiplier for federal holidays + Sundays


def _drange(start: date, end: date, step: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def main() -> None:
    print("Loading parquet...")
    df = load_smard_15min(PARQUET)

    # M9: extend train window back to 2022-01-08. v3 cut at 2024 to avoid
    # the 2022 €500 spike outlier era, but that also dropped most of the
    # negative-price events. Bringing them back with a 0.5× weight buys
    # ~2× the holiday/extreme examples without spike domination.
    train_dates = [issue_time_for(d) for d in _drange(date(2022, 1, 8),  date(2025, 12, 31))]
    val_dates   = [issue_time_for(d) for d in _drange(date(2026, 1, 1),  date(2026, 2, 28))]

    print(f"\nBuilding price windows: {len(train_dates)} train, {len(val_dates)} val")
    Xe_tr, Xd_tr, Y_tr, kept_tr = build_price_dataset(
        df, train_dates, include_weather=True,
    )
    Xe_va, Xd_va, Y_va, kept_va = build_price_dataset(
        df, val_dates,   include_weather=True,
    )
    print(f"  kept: train={len(kept_tr)}, val={len(kept_va)}")
    print(f"  shapes: X_enc={Xe_tr.shape}  X_dec={Xd_tr.shape}  Y={Y_tr.shape}")
    print(f"  encoder features: {PRICE_ENC_FEATURE_NAMES} + 4 weather")
    print(f"  decoder features: {PRICE_DEC_FEATURE_NAMES} + 4 weather")
    print(f"  target stats: mean={Y_tr.mean():.1f} std={Y_tr.std():.1f} "
          f"min={Y_tr.min():.1f} max={Y_tr.max():.1f}")

    # ---- Per-window sample weights (M9) ----------------------------
    federal_holidays = hols.country_holidays("DE", years=range(2022, 2027))
    sample_w = np.array(
        [_sample_weight(t, federal_holidays) for t in kept_tr], dtype=np.float32,
    )
    n_holiday_or_sun = int((sample_w > 1.5).sum())  # weight ≥ 3 (or 1.5 for old-year holidays)
    n_old = int((sample_w < 1.0).sum())             # weight ≤ 0.5 means 2022-23 non-holiday
    print(f"  sample weights: {n_holiday_or_sun} upweighted (holiday/Sun, "
          f"3x), {n_old} downweighted (2022-23, 0.5x)")

    # ---- Feature-dropout augmentation (industry pattern) -----------
    # Append masked copies of a random subset of training windows so the
    # model sees both "VRE forecast present" and "VRE forecast missing"
    # regimes. Augmentation is applied BEFORE scaling so the scaler
    # learns the bimodal distribution of the present-flag and adjusted
    # mean/std for the imputed-zero VRE column.
    n_aug = int(VRE_DROPOUT_FRAC * len(Xe_tr))
    rng = np.random.RandomState(42)
    aug_idx = rng.choice(len(Xe_tr), size=n_aug, replace=False)
    Xd_aug = Xd_tr[aug_idx].copy()
    # Zero everything derived from tso_vre_fc — the present flag is the
    # only signal the model gets that VRE info is unavailable.
    Xd_aug[..., VRE_FC_COL_IDX] = 0.0
    Xd_aug[..., VRE_PRESENT_COL_IDX] = 0.0
    Xd_aug[..., VRE_RATIO_COL_IDX] = 0.0
    Xd_aug[..., VRE_PCTILE_COL_IDX] = 0.0
    Xe_tr = np.concatenate([Xe_tr, Xe_tr[aug_idx]], axis=0)
    Xd_tr = np.concatenate([Xd_tr, Xd_aug], axis=0)
    Y_tr  = np.concatenate([Y_tr,  Y_tr[aug_idx]],  axis=0)
    sample_w = np.concatenate([sample_w, sample_w[aug_idx]], axis=0)
    print(f"  augmented with {n_aug} VRE-masked copies "
          f"({VRE_DROPOUT_FRAC:.0%}) -> train n={len(Xe_tr)}")

    scaler = FeatureScaler.fit(Xe_tr, Xd_tr, Y_tr)
    Xe_tr_n, Xd_tr_n, Y_tr_n = scaler.transform(Xe_tr, Xd_tr, Y_tr)
    Xe_va_n, Xd_va_n, Y_va_n = scaler.transform(Xe_va, Xd_va, Y_va)

    print("\nBuilding model...")
    from tensorflow import keras
    model = compile_lstm_quantile(
        build_lstm_quantile(
            hidden=64,
            enc_features=Xe_tr.shape[-1],
            dec_features=Xd_tr.shape[-1],
        ),
        lr=1e-3,
    )
    model.summary(line_length=100)
    print(f"\nQuantiles: {QUANTILES}")

    print("\nTraining...")
    t0 = time.time()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5, verbose=1,
        ),
    ]
    history = model.fit(
        [Xe_tr_n, Xd_tr_n], Y_tr_n,
        sample_weight=sample_w,
        validation_data=([Xe_va_n, Xd_va_n], Y_va_n),
        epochs=60, batch_size=32, callbacks=callbacks, verbose=2,
    )
    train_time = time.time() - t0
    print(f"\nTrained in {train_time:.0f}s ({len(history.epoch)} epochs)")

    # Validation diagnostics — same shape as the load quantile script.
    pred_va_n = model.predict([Xe_va_n, Xd_va_n], verbose=0)
    pred_va = scaler.inverse_y(pred_va_n)
    p10 = pred_va[..., 0]
    p50 = pred_va[..., 1]
    p90 = pred_va[..., 2]

    val_p50_mae = float(np.abs(Y_va - p50).mean())
    val_mean_abs_y = float(np.abs(Y_va).mean())
    inside = float(((Y_va >= p10) & (Y_va <= p90)).mean())
    avg_width = float((p90 - p10).mean())
    crossings = float(((p90 < p50) | (p50 < p10)).mean())

    print(f"\nValidation P50 MAE (full features): {val_p50_mae:>7.2f} €/MWh")
    print(f"Validation P50 / mean |y|:         {val_p50_mae / val_mean_abs_y * 100:>7.2f} %")
    print("\nInterval [P10, P90]:")
    print(f"  Empirical coverage:    {inside:.3%}   (target ~80%)")
    print(f"  Mean width:            {avg_width:.1f} €/MWh")
    print(f"  Quantile crossings:    {crossings:.3%}")

    # ---- Degraded-mode validation (no fc_gen) ----------------------
    # Mask the VRE forecast on the entire val set and re-score. This is
    # the "rendered before SMARD published" scenario — we want to know
    # the cost of running without the dominant feature.
    Xd_va_masked = Xd_va.copy()
    Xd_va_masked[..., VRE_FC_COL_IDX] = 0.0
    Xd_va_masked[..., VRE_PRESENT_COL_IDX] = 0.0
    Xd_va_masked[..., VRE_RATIO_COL_IDX] = 0.0
    Xd_va_masked[..., VRE_PCTILE_COL_IDX] = 0.0
    Xe_va_n2, Xd_va_masked_n, _ = scaler.transform(Xe_va, Xd_va_masked, Y_va)
    pred_va_masked = scaler.inverse_y(model.predict([Xe_va_n2, Xd_va_masked_n], verbose=0))
    val_p50_mae_masked = float(np.abs(Y_va - pred_va_masked[..., 1]).mean())
    print(f"\nValidation P50 MAE (VRE masked):    {val_p50_mae_masked:>7.2f} EUR/MWh")
    print(f"  delta vs full features:           +{val_p50_mae_masked - val_p50_mae:>6.2f} EUR/MWh "
          f"({(val_p50_mae_masked / val_p50_mae - 1) * 100:+.1f} %)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "model.keras")
    np.savez(
        OUT_DIR / "scaler.npz",
        enc_mean=scaler.enc_mean, enc_std=scaler.enc_std,
        dec_mean=scaler.dec_mean, dec_std=scaler.dec_std,
        y_mean=scaler.y_mean, y_std=scaler.y_std,
    )
    meta = {
        "model": "price_quantile_v4",
        "vre_dropout_frac": VRE_DROPOUT_FRAC,
        "weight_old_years": WEIGHT_OLD_YEARS,
        "weight_holiday_or_sun": WEIGHT_HOLIDAY_OR_SUN,
        "val_p50_mae_eur_mwh_masked": val_p50_mae_masked,
        "val_p50_mae_delta_eur_mwh": val_p50_mae_masked - val_p50_mae,
        "target": "price__germany_luxembourg",
        "include_weather": True,
        "quantiles": list(QUANTILES),
        "hidden": 64,
        "enc_features": int(Xe_tr.shape[-1]),
        "dec_features": int(Xd_tr.shape[-1]),
        "epochs_run": len(history.epoch),
        "train_time_s": train_time,
        "train_n": int(len(kept_tr)),
        "val_n": int(len(kept_va)),
        "train_window": "2024-01-08 to 2025-12-31",
        "val_window": "2026-01-01 to 2026-02-28",
        "val_p50_mae_eur_mwh": val_p50_mae,
        "val_interval_coverage": inside,
        "val_interval_width_eur_mwh": avg_width,
        "val_quantile_crossings": crossings,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
