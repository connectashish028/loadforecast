"""Train an XGBoost quantile baseline for the LOAD model.

Mirrors the LSTM's residual-learning target: predicts (actual - TSO_forecast),
then adds the correction at inference. Same train/val splits, same data
layer, same leakage-safe windowing — the only difference is the model
class. Used to validate whether the seq2seq LSTM earns its complexity
vs a standard gradient-boosted baseline on tabular features.

Features come from `loadforecast.features.build.build_target_day_features`
(47 features: calendar, TSO forecast, lag features for D-2/D-7/D-14,
rolling stats, neighbour-zone prices, and the lagged-TSO-error feature
that was the LSTM ablation's biggest single lift).

One XGBoost regressor per quantile (3 models). XGBoost native missing-
value support means no row-drops for sparse columns.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.features.build import build_target_day_features

PARQUET = "smard_merged_15min.parquet"
OUT_DIR = Path("model_checkpoints/xgboost_load_v1")
ACTUAL_LOAD = "actual_cons__grid_load"
TSO_FC = "fc_cons__grid_load"
QUANTILES = (0.10, 0.50, 0.90)


def _drange(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def build_xgb_dataset(df: pd.DataFrame, dates: list[date]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack one (96, n_features) feature matrix per issue time. Returns
    (X, y, feature_cols)."""
    X_list, y_list, feature_cols = [], [], None
    for d in dates:
        issue = issue_time_for(d)
        try:
            features = build_target_day_features(df, issue)
        except Exception:
            continue
        actual = df[ACTUAL_LOAD].reindex(features.index).to_numpy()
        tso = df[TSO_FC].reindex(features.index).to_numpy()
        if np.isnan(actual).any() or np.isnan(tso).any():
            continue
        residual = actual - tso
        if feature_cols is None:
            feature_cols = features.columns.tolist()
        X_list.append(features.to_numpy(dtype=np.float32))
        y_list.append(residual.astype(np.float32))
    if not X_list:
        raise RuntimeError("No usable training samples — check date range.")
    return np.vstack(X_list), np.concatenate(y_list), feature_cols


def main() -> None:
    print("Loading parquet...")
    df = load_smard_15min(PARQUET)

    # Same period as the LSTM: 2022-2024 train, 2025-H1 val.
    train_dates = _drange(date(2022, 1, 15), date(2024, 12, 31))
    val_dates = _drange(date(2025, 1, 1), date(2025, 6, 30))

    print("\nBuilding datasets...")
    X_train, y_train, feature_cols = build_xgb_dataset(df, train_dates)
    X_val, y_val, _ = build_xgb_dataset(df, val_dates)
    print(f"  train: X={X_train.shape}, y={y_train.shape}")
    print(f"  val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  features: {len(feature_cols)}")
    print(f"  target stats: mean={y_train.mean():.1f} std={y_train.std():.1f}")

    # Three quantile regressors. XGBoost 3.x supports quantile loss via
    # objective="reg:quantileerror" with quantile_alpha.
    models = {}
    t0 = time.time()
    for q in QUANTILES:
        print(f"\nTraining quantile {q}...")
        reg = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            learning_rate=0.05,
            max_depth=6,
            n_estimators=600,
            tree_method="hist",
            early_stopping_rounds=25,
            random_state=42,
        )
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models[q] = reg
        print(f"  best iter: {reg.best_iteration}, "
              f"best val: {reg.best_score:.2f}")
    train_time = time.time() - t0
    print(f"\nTrained 3 quantile models in {train_time:.0f}s")

    # Validation diagnostics.
    pred_val = np.stack([models[q].predict(X_val) for q in QUANTILES], axis=1)
    p10, p50, p90 = pred_val[:, 0], pred_val[:, 1], pred_val[:, 2]
    val_p50_mae = float(np.abs(y_val - p50).mean())
    coverage = float(((y_val >= p10) & (y_val <= p90)).mean())
    crossings = float(((p90 < p50) | (p50 < p10)).mean())

    print(f"\nValidation P50 MAE:               {val_p50_mae:>7.2f} MW")
    print(f"Validation 80% band coverage:    {coverage*100:>7.2f} %  (target 80 %)")
    print(f"Quantile crossings:              {crossings*100:>7.2f} %")

    # Save models.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for q, reg in models.items():
        reg.save_model(OUT_DIR / f"xgb_q{int(q*100):02d}.json")

    meta = {
        "model": "xgboost_load_v1",
        "target": "residual (actual_load - TSO_forecast)",
        "quantiles": list(QUANTILES),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "train_window": "2022-01-15 to 2024-12-31",
        "val_window": "2025-01-01 to 2025-06-30",
        "n_train_rows": int(X_train.shape[0]),
        "n_val_rows": int(X_val.shape[0]),
        "val_p50_mae_mw": val_p50_mae,
        "val_interval_coverage": coverage,
        "val_quantile_crossings": crossings,
        "train_time_s": train_time,
        "hyperparams": {
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_estimators": 600,
            "tree_method": "hist",
            "early_stopping_rounds": 25,
        },
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
