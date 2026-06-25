"""Train the seq2seq TRANSFORMER with quantile heads (P10/P50/P90) on the
load model's weather feature set.

Comparison-baseline experiment (feature/transformer branch). Identical to
`train_lstm_quantile.py` — same splits, scaler, pinball loss, callbacks — the
only change is the architecture: `build_transformer_quantile` instead of
`build_lstm_quantile`. The transformer is sized for ~LSTM parameter parity
(d_model=32, num_blocks=1, ff_dim=64 -> ~47k params vs the LSTM's ~39k) so the
comparison is about architecture, not capacity.

Saves to model_checkpoints/transformer_quantile_v1/.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.dataset import (
    WEATHER_COLS,
    FeatureScaler,
    build_dataset,
)
from loadforecast.models.lstm_quantile import QUANTILES, compile_lstm_quantile
from loadforecast.models.transformer_quantile import build_transformer_quantile

PARQUET = "smard_merged_15min.parquet"
OUT_DIR = Path("model_checkpoints/transformer_quantile_v1")
ARCH = dict(d_model=32, num_heads=4, num_blocks=1, ff_dim=64, dropout=0.1)


def _drange(start: date, end: date, step: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def main() -> None:
    print("Loading parquet...")
    df = load_smard_15min(PARQUET)
    missing = [c for c in WEATHER_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing weather columns: {missing}. Run refresh first.")

    train_dates = [issue_time_for(d) for d in _drange(date(2022, 1, 8),  date(2024, 12, 31))]
    val_dates   = [issue_time_for(d) for d in _drange(date(2025, 1, 1),  date(2025, 6, 30))]

    print(f"\nBuilding windows (with weather): {len(train_dates)} train, {len(val_dates)} val")
    Xe_tr, Xd_tr, Y_tr, kept_tr = build_dataset(df, train_dates, include_weather=True)
    Xe_va, Xd_va, Y_va, kept_va = build_dataset(df, val_dates,   include_weather=True)
    print(f"  kept: train={len(kept_tr)}, val={len(kept_va)}")
    print(f"  shapes: X_enc={Xe_tr.shape}  X_dec={Xd_tr.shape}  Y={Y_tr.shape}")

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
        validation_data=([Xe_va_n, Xd_va_n], Y_va_n),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - t0
    print(f"\nTrained in {train_time:.0f}s ({len(history.epoch)} epochs)")

    pred_va_n = model.predict([Xe_va_n, Xd_va_n], verbose=0)  # (N, 96, 3)
    pred_va = scaler.inverse_y(pred_va_n)
    p10 = pred_va[..., 0]
    p50 = pred_va[..., 1]
    p90 = pred_va[..., 2]

    val_p50_mae = float(np.abs(Y_va - p50).mean())
    val_implied_skill = float(1 - val_p50_mae / np.abs(Y_va).mean())
    inside = ((Y_va >= p10) & (Y_va <= p90)).mean()
    avg_width = float((p90 - p10).mean())
    crossings = float(((p90 < p50) | (p50 < p10)).mean())

    print(f"\nValidation P50 residual MAE: {val_p50_mae:.1f} MW")
    print(f"  Implied P50 skill if generalised: {val_implied_skill:+.4f}")
    print("\nInterval [P10, P90]:")
    print(f"  Empirical coverage:   {inside:.3%}   (target ~80%)")
    print(f"  Mean width:           {avg_width:.1f} MW")
    print(f"  Quantile crossings:   {crossings:.3%}   (lower = better)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "model.keras")
    np.savez(
        OUT_DIR / "scaler.npz",
        enc_mean=scaler.enc_mean, enc_std=scaler.enc_std,
        dec_mean=scaler.dec_mean, dec_std=scaler.dec_std,
        y_mean=scaler.y_mean, y_std=scaler.y_std,
    )
    meta = {
        "model": "transformer_quantile",
        "include_weather": True,
        "quantiles": list(QUANTILES),
        "arch": ARCH,
        "n_params": int(model.count_params()),
        "enc_features": int(Xe_tr.shape[-1]),
        "dec_features": int(Xd_tr.shape[-1]),
        "epochs_run": len(history.epoch),
        "train_time_s": train_time,
        "train_n": int(len(kept_tr)),
        "val_n": int(len(kept_va)),
        "val_p50_residual_mae_mw": val_p50_mae,
        "val_p50_implied_skill": val_implied_skill,
        "val_interval_coverage": float(inside),
        "val_interval_width_mw": avg_width,
        "val_quantile_crossings": crossings,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
