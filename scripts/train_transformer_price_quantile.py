"""Train the day-ahead price TRANSFORMER (comparison-baseline experiment).

Identical to `train_lstm_price_quantile.py` — same train window, sample
weighting (0.5x 2022-23, 3x holidays/Sundays), 30% VRE-dropout augmentation,
scaler, pinball loss, callbacks, and degraded-mode diagnostics — the only
change is the architecture: `build_transformer_quantile` instead of
`build_lstm_quantile`, sized for ~LSTM parameter parity
(d_model=32, num_blocks=1, ff_dim=64).

Saves to model_checkpoints/transformer_price_quantile_v1/.
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
from loadforecast.models.lstm_quantile import QUANTILES, compile_lstm_quantile
from loadforecast.models.price_dataset import (
    PRICE_DEC_FEATURE_NAMES,
    PRICE_ENC_FEATURE_NAMES,
    build_price_dataset,
)
from loadforecast.models.transformer_quantile import build_transformer_quantile

PARQUET = "smard_merged_15min.parquet"
OUT_DIR = Path("model_checkpoints/transformer_price_quantile_v1")
ARCH = dict(d_model=32, num_heads=4, num_blocks=1, ff_dim=64, dropout=0.1)

VRE_DROPOUT_FRAC = 0.30
VRE_FC_COL_IDX = 1        # tso_vre_fc
VRE_PRESENT_COL_IDX = 2   # tso_vre_fc_present
VRE_RATIO_COL_IDX = 3     # vre_to_load_ratio
VRE_PCTILE_COL_IDX = 4    # vre_percentile
WEIGHT_OLD_YEARS = 0.5
WEIGHT_HOLIDAY_OR_SUN = 3.0


def _sample_weight(issue_time, federal_holidays) -> float:
    delivery_local = (issue_time.tz_convert("Europe/Berlin").normalize()
                       + pd.Timedelta(days=1))
    w = 1.0
    if delivery_local.year in (2022, 2023):
        w *= WEIGHT_OLD_YEARS
    if delivery_local.weekday() == 6 or delivery_local.date() in federal_holidays:
        w *= WEIGHT_HOLIDAY_OR_SUN
    return w


def _drange(start: date, end: date, step: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def main() -> None:
    print("Loading parquet...")
    df = load_smard_15min(PARQUET)

    train_dates = [issue_time_for(d) for d in _drange(date(2022, 1, 8),  date(2025, 12, 31))]
    val_dates   = [issue_time_for(d) for d in _drange(date(2026, 1, 1),  date(2026, 2, 28))]

    print(f"\nBuilding price windows: {len(train_dates)} train, {len(val_dates)} val")
    Xe_tr, Xd_tr, Y_tr, kept_tr = build_price_dataset(df, train_dates, include_weather=True)
    Xe_va, Xd_va, Y_va, kept_va = build_price_dataset(df, val_dates,   include_weather=True)
    print(f"  kept: train={len(kept_tr)}, val={len(kept_va)}")
    print(f"  shapes: X_enc={Xe_tr.shape}  X_dec={Xd_tr.shape}  Y={Y_tr.shape}")
    print(f"  encoder features: {PRICE_ENC_FEATURE_NAMES} + 4 weather")
    print(f"  decoder features: {PRICE_DEC_FEATURE_NAMES} + 4 weather")

    # Per-window sample weights.
    federal_holidays = hols.country_holidays("DE", years=range(2022, 2027))
    sample_w = np.array(
        [_sample_weight(t, federal_holidays) for t in kept_tr], dtype=np.float32,
    )
    n_holiday_or_sun = int((sample_w > 1.5).sum())
    n_old = int((sample_w < 1.0).sum())
    print(f"  sample weights: {n_holiday_or_sun} upweighted (3x), {n_old} downweighted (0.5x)")

    # Feature-dropout augmentation (VRE masked) — applied before scaling.
    n_aug = int(VRE_DROPOUT_FRAC * len(Xe_tr))
    rng = np.random.RandomState(42)
    aug_idx = rng.choice(len(Xe_tr), size=n_aug, replace=False)
    Xd_aug = Xd_tr[aug_idx].copy()
    Xd_aug[..., VRE_FC_COL_IDX] = 0.0
    Xd_aug[..., VRE_PRESENT_COL_IDX] = 0.0
    Xd_aug[..., VRE_RATIO_COL_IDX] = 0.0
    Xd_aug[..., VRE_PCTILE_COL_IDX] = 0.0
    Xe_tr = np.concatenate([Xe_tr, Xe_tr[aug_idx]], axis=0)
    Xd_tr = np.concatenate([Xd_tr, Xd_aug], axis=0)
    Y_tr  = np.concatenate([Y_tr,  Y_tr[aug_idx]],  axis=0)
    sample_w = np.concatenate([sample_w, sample_w[aug_idx]], axis=0)
    print(f"  augmented with {n_aug} VRE-masked copies -> train n={len(Xe_tr)}")

    scaler = FeatureScaler.fit(Xe_tr, Xd_tr, Y_tr)
    Xe_tr_n, Xd_tr_n, Y_tr_n = scaler.transform(Xe_tr, Xd_tr, Y_tr)
    Xe_va_n, Xd_va_n, Y_va_n = scaler.transform(Xe_va, Xd_va, Y_va)

    print("\nBuilding model...")
    from tensorflow import keras
    model = compile_lstm_quantile(
        build_transformer_quantile(
            enc_features=Xe_tr.shape[-1],
            dec_features=Xd_tr.shape[-1],
            **ARCH,
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

    pred_va = scaler.inverse_y(model.predict([Xe_va_n, Xd_va_n], verbose=0))
    p10, p50, p90 = pred_va[..., 0], pred_va[..., 1], pred_va[..., 2]
    val_p50_mae = float(np.abs(Y_va - p50).mean())
    val_mean_abs_y = float(np.abs(Y_va).mean())
    inside = float(((Y_va >= p10) & (Y_va <= p90)).mean())
    avg_width = float((p90 - p10).mean())
    crossings = float(((p90 < p50) | (p50 < p10)).mean())

    print(f"\nValidation P50 MAE (full features): {val_p50_mae:>7.2f} EUR/MWh")
    print(f"Validation P50 / mean |y|:         {val_p50_mae / val_mean_abs_y * 100:>7.2f} %")
    print("\nInterval [P10, P90]:")
    print(f"  Empirical coverage:    {inside:.3%}   (target ~80%)")
    print(f"  Mean width:            {avg_width:.1f} EUR/MWh")
    print(f"  Quantile crossings:    {crossings:.3%}")

    # Degraded-mode validation (VRE masked).
    Xd_va_masked = Xd_va.copy()
    Xd_va_masked[..., VRE_FC_COL_IDX] = 0.0
    Xd_va_masked[..., VRE_PRESENT_COL_IDX] = 0.0
    Xd_va_masked[..., VRE_RATIO_COL_IDX] = 0.0
    Xd_va_masked[..., VRE_PCTILE_COL_IDX] = 0.0
    Xe_va_n2, Xd_va_masked_n, _ = scaler.transform(Xe_va, Xd_va_masked, Y_va)
    pred_va_masked = scaler.inverse_y(model.predict([Xe_va_n2, Xd_va_masked_n], verbose=0))
    val_p50_mae_masked = float(np.abs(Y_va - pred_va_masked[..., 1]).mean())
    print(f"\nValidation P50 MAE (VRE masked):    {val_p50_mae_masked:>7.2f} EUR/MWh "
          f"(+{val_p50_mae_masked - val_p50_mae:.2f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "model.keras")
    np.savez(
        OUT_DIR / "scaler.npz",
        enc_mean=scaler.enc_mean, enc_std=scaler.enc_std,
        dec_mean=scaler.dec_mean, dec_std=scaler.dec_std,
        y_mean=scaler.y_mean, y_std=scaler.y_std,
    )
    meta = {
        "model": "transformer_price_quantile",
        "arch": ARCH,
        "n_params": int(model.count_params()),
        "vre_dropout_frac": VRE_DROPOUT_FRAC,
        "weight_old_years": WEIGHT_OLD_YEARS,
        "weight_holiday_or_sun": WEIGHT_HOLIDAY_OR_SUN,
        "val_p50_mae_eur_mwh_masked": val_p50_mae_masked,
        "val_p50_mae_delta_eur_mwh": val_p50_mae_masked - val_p50_mae,
        "target": "price__germany_luxembourg",
        "include_weather": True,
        "quantiles": list(QUANTILES),
        "enc_features": int(Xe_tr.shape[-1]),
        "dec_features": int(Xd_tr.shape[-1]),
        "epochs_run": len(history.epoch),
        "train_time_s": train_time,
        "train_n": int(len(kept_tr)),
        "val_n": int(len(kept_va)),
        "train_window": "2022-01-08 to 2025-12-31",
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
