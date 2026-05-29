"""M4 part 3 — feature ablation. HISTORICAL — output is archived.

This script ran once during the M4 LSTM exploration phase. It trained
five LSTM variants, each adding one feature group on top of the
previous, and scored them on the 70-date stratified holdout. The
output `backtest_results/ablation_summary.csv` is committed and
rendered by the dashboard's "Where the error reduction comes from"
chart. The script is preserved for reproducibility but is NOT part
of the production pipeline (XGBoost is the production model).

Variants (encoder / decoder, calendar always on):
  A  calendar only
  B  + load history (encoder)
  C  + residual lag (encoder)
  D  + TSO_fc (decoder)
  E  + weather (encoder + decoder)  -- equivalent to lstm_weather_v1

All variants train on the residual target (actual - TSO_fc), so skill
is comparable across the ladder.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.dataset import FeatureScaler, build_dataset, build_window
from loadforecast.models.lstm_plain import build_lstm_plain, compile_lstm

PARQUET = "smard_merged_15min.parquet"
OUT_ROOT = Path("model_checkpoints/ablation")
SUMMARY_CSV = Path("backtest_results/ablation_summary.csv")


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    include_load_history: bool
    include_residual: bool
    include_tso_fc_dec: bool
    include_weather: bool


VARIANTS = [
    Variant("A_calendar",     "calendar only",      False, False, False, False),
    Variant("B_load",         "+ load history",     True,  False, False, False),
    Variant("C_residual",     "+ residual lag",     True,  True,  False, False),
    Variant("D_tso_fc",       "+ TSO_fc decoder",   True,  True,  True,  False),
    Variant("E_weather",      "+ weather",          True,  True,  True,  True),
]


def _drange(start: date, end: date, step: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def _holdout_dates() -> list[date]:
    return list(_drange(date(2025, 1, 1), date(2026, 4, 30), 7))


def _train_one(df: pd.DataFrame, v: Variant, train_dates, val_dates, holdout_dates) -> dict:
    print(f"\n{'='*70}\n[{v.key}]  {v.label}\n{'='*70}")
    flags = dict(
        include_load_history=v.include_load_history,
        include_residual=v.include_residual,
        include_tso_fc_dec=v.include_tso_fc_dec,
        include_weather=v.include_weather,
    )

    Xe_tr, Xd_tr, Y_tr, kept_tr = build_dataset(df, train_dates, **flags)
    Xe_va, Xd_va, Y_va, kept_va = build_dataset(df, val_dates,   **flags)
    print(f"  train={len(kept_tr)}  val={len(kept_va)}  "
          f"X_enc={Xe_tr.shape}  X_dec={Xd_tr.shape}")

    scaler = FeatureScaler.fit(Xe_tr, Xd_tr, Y_tr)
    Xe_tr_n, Xd_tr_n, Y_tr_n = scaler.transform(Xe_tr, Xd_tr, Y_tr)
    Xe_va_n, Xd_va_n, Y_va_n = scaler.transform(Xe_va, Xd_va, Y_va)

    from tensorflow import keras
    keras.utils.set_random_seed(42)
    model = compile_lstm(
        build_lstm_plain(
            hidden=64,
            enc_features=Xe_tr.shape[-1],
            dec_features=Xd_tr.shape[-1],
        ),
        lr=1e-3,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3,
                                          factor=0.5, min_lr=1e-5, verbose=0),
    ]
    t0 = time.time()
    history = model.fit(
        [Xe_tr_n, Xd_tr_n], Y_tr_n,
        validation_data=([Xe_va_n, Xd_va_n], Y_va_n),
        epochs=60, batch_size=32, callbacks=callbacks, verbose=2,
    )
    train_time = time.time() - t0

    pred_va = scaler.inverse_y(model.predict([Xe_va_n, Xd_va_n], verbose=0))
    val_mae = float(np.abs(Y_va - pred_va).mean())
    val_skill = float(1 - val_mae / np.abs(Y_va).mean())

    # Holdout: walk dates, predict, score residual MAE in load space.
    print("  scoring holdout...")
    abs_resid_errs: list[float] = []
    abs_tso_errs: list[float] = []
    n_days = 0
    for d in holdout_dates:
        issue = issue_time_for(d)
        w = build_window(df, issue, **flags)
        if np.isnan(w.X_enc).any() or np.isnan(w.X_dec).any() or np.isnan(w.y_resid).any():
            continue
        Xe, Xd = scaler.transform(w.X_enc[None, ...], w.X_dec[None, ...])
        pred_resid = scaler.inverse_y(model.predict([Xe, Xd], verbose=0)[0])
        abs_resid_errs.extend(np.abs(w.y_resid - pred_resid).tolist())
        abs_tso_errs.extend(np.abs(w.y_resid).tolist())  # |actual - tso_fc|
        n_days += 1

    holdout_mae = float(np.mean(abs_resid_errs))
    tso_mae = float(np.mean(abs_tso_errs))
    holdout_skill = float(1 - holdout_mae / tso_mae)
    print(f"  -> val_skill={val_skill:+.4f}  holdout_skill={holdout_skill:+.4f}  "
          f"holdout_mae={holdout_mae:.1f}  tso_mae={tso_mae:.1f}  n_days={n_days}")

    out_dir = OUT_ROOT / v.key
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "variant": v.key,
        "label": v.label,
        "flags": flags,
        "enc_features": int(Xe_tr.shape[-1]),
        "dec_features": int(Xd_tr.shape[-1]),
        "epochs_run": len(history.epoch),
        "train_time_s": train_time,
        "val_residual_mae_mw": val_mae,
        "val_implied_skill": val_skill,
        "holdout_n_days": n_days,
        "holdout_residual_mae_mw": holdout_mae,
        "holdout_tso_mae_mw": tso_mae,
        "holdout_skill": holdout_skill,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    print("Loading parquet...")
    df = load_smard_15min(PARQUET)
    train_dates = [issue_time_for(d) for d in _drange(date(2022, 1, 8),  date(2024, 12, 31))]
    val_dates   = [issue_time_for(d) for d in _drange(date(2025, 1, 1),  date(2025, 6, 30))]
    holdout_dates = _holdout_dates()
    print(f"  train={len(train_dates)}  val={len(val_dates)}  holdout={len(holdout_dates)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = [_train_one(df, v, train_dates, val_dates, holdout_dates) for v in VARIANTS]
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV}")
    print(summary[["variant", "label", "enc_features", "dec_features",
                   "val_implied_skill", "holdout_skill"]].to_string(index=False))


if __name__ == "__main__":
    main()
