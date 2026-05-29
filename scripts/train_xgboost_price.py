"""Train an XGBoost quantile baseline for the PRICE model.

Mirrors the v4 LSTM:
  - Target: raw EPEX day-ahead clearing price (not residual — no published baseline)
  - Features: 47 from features.build + 3 engineered VRE features
    (tso_vre_fc_present, vre_to_load_ratio, vre_percentile)
  - Sample weights: 3x holidays + Sundays, 0.5x 2022-2023 (recovers
    negative-price examples without spike domination)
  - Same train/val splits as the LSTM v4

Used in scripts/compare_lstm_vs_xgboost_price.py for the apples-to-apples
architecture comparison. No M10 clip — that's a post-processing layer
calibrated against the LSTM specifically; the comparison evaluates raw
model outputs.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import holidays as hols
import numpy as np
import pandas as pd
import xgboost as xgb

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.features.build import build_target_day_features

PARQUET = "smard_merged_15min.parquet"
OUT_DIR = Path("model_checkpoints/xgboost_price_v1")
PRICE_COL = "price__germany_luxembourg"
VRE_FC_COL = "fc_gen__photovoltaics_and_wind"
QUANTILES = (0.10, 0.50, 0.90)

WEIGHT_OLD_YEARS = 0.5
WEIGHT_HOLIDAY_OR_SUN = 3.0


def _drange(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _add_engineered_vre(features: pd.DataFrame, df: pd.DataFrame,
                       issue_time: pd.Timestamp) -> pd.DataFrame:
    """Add the 3 v4-specific VRE features to the existing 47-feature set."""
    out = features.copy()

    vre_fc = out["tso_vre_fc"]
    out["tso_vre_fc_present"] = (~vre_fc.isna()).astype(np.float32)
    out["tso_vre_fc"] = vre_fc.fillna(0.0)

    load_fc = out["tso_load_fc"]
    safe_load = load_fc.where(load_fc > 0, 1.0)
    out["vre_to_load_ratio"] = (out["tso_vre_fc"] / safe_load).astype(np.float32)

    ref_window = df[VRE_FC_COL].loc[
        issue_time - pd.Timedelta(days=90): issue_time
    ].dropna()
    q90 = float(ref_window.quantile(0.90)) if len(ref_window) > 100 else 1.0
    out["vre_percentile"] = (out["tso_vre_fc"] / max(q90, 1.0)).astype(np.float32)

    return out


def _sample_weight(delivery_date: date, federal_holidays: set) -> float:
    w = 1.0
    if delivery_date.year in (2022, 2023):
        w *= WEIGHT_OLD_YEARS
    if delivery_date.weekday() == 6 or delivery_date in federal_holidays:
        w *= WEIGHT_HOLIDAY_OR_SUN
    return w


def build_dataset(df: pd.DataFrame, dates: list[date]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Stack one (96, n_features) feature matrix per issue time. Returns
    (X, y, sample_weights, feature_cols). Target is raw price."""
    federal_holidays = hols.country_holidays("DE", years=range(2022, 2027))
    X_list, y_list, w_list, feature_cols = [], [], [], None
    for d in dates:
        issue = issue_time_for(d)
        try:
            features = build_target_day_features(df, issue)
        except Exception:
            continue
        features = _add_engineered_vre(features, df, issue)
        target = df[PRICE_COL].reindex(features.index).to_numpy()
        if np.isnan(target).any():
            continue
        if feature_cols is None:
            feature_cols = features.columns.tolist()
        X_list.append(features.to_numpy(dtype=np.float32))
        y_list.append(target.astype(np.float32))
        w = _sample_weight(d, federal_holidays)
        w_list.append(np.full(96, w, dtype=np.float32))
    if not X_list:
        raise RuntimeError("No usable training samples — check date range.")
    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.concatenate(w_list),
        feature_cols,
    )


def main() -> None:
    print("Loading parquet...")
    df = load_smard_15min(PARQUET)

    train_dates = _drange(date(2022, 1, 15), date(2025, 12, 31))
    val_dates = _drange(date(2026, 1, 1), date(2026, 2, 28))

    print("\nBuilding datasets (raw price target)...")
    X_train, y_train, w_train, feature_cols = build_dataset(df, train_dates)
    X_val, y_val, _, _ = build_dataset(df, val_dates)
    print(f"  train: X={X_train.shape}, y={y_train.shape}, weights={w_train.shape}")
    print(f"  val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  features: {len(feature_cols)}")
    print(f"  target stats: mean={y_train.mean():.1f} std={y_train.std():.1f} "
          f"min={y_train.min():.1f} max={y_train.max():.1f}")
    n_upweighted = int((w_train > 1.5).sum() // 96)  # day count
    n_downweighted = int((w_train < 1.0).sum() // 96)
    print(f"  sample weights: {n_upweighted} days upweighted (3x holiday/Sun), "
          f"{n_downweighted} days downweighted (0.5x 2022-23)")

    models = {}
    t0 = time.time()
    for q in QUANTILES:
        print(f"\nTraining quantile {q}...")
        reg = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            learning_rate=0.05,
            max_depth=6,
            n_estimators=800,
            tree_method="hist",
            early_stopping_rounds=30,
            random_state=42,
        )
        reg.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models[q] = reg
        print(f"  best iter: {reg.best_iteration}, "
              f"best val: {reg.best_score:.2f}")
    train_time = time.time() - t0
    print(f"\nTrained 3 quantile models in {train_time:.0f}s")

    pred_val = np.stack([models[q].predict(X_val) for q in QUANTILES], axis=1)
    p10, p50, p90 = pred_val[:, 0], pred_val[:, 1], pred_val[:, 2]
    val_p50_mae = float(np.abs(y_val - p50).mean())
    coverage = float(((y_val >= p10) & (y_val <= p90)).mean())
    crossings = float(((p90 < p50) | (p50 < p10)).mean())

    print(f"\nValidation P50 MAE:                {val_p50_mae:>7.2f} EUR/MWh")
    print(f"Validation 80% band coverage:     {coverage*100:>7.2f} %  (target 80 %)")
    print(f"Quantile crossings:               {crossings*100:>7.2f} %")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for q, reg in models.items():
        reg.save_model(OUT_DIR / f"xgb_q{int(q*100):02d}.json")

    meta = {
        "model": "xgboost_price_v1",
        "target": "raw EPEX clearing price (EUR/MWh)",
        "quantiles": list(QUANTILES),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "train_window": "2022-01-15 to 2025-12-31",
        "val_window": "2026-01-01 to 2026-02-28",
        "n_train_rows": int(X_train.shape[0]),
        "n_val_rows": int(X_val.shape[0]),
        "val_p50_mae_eur_mwh": val_p50_mae,
        "val_interval_coverage": coverage,
        "val_quantile_crossings": crossings,
        "train_time_s": train_time,
        "weight_old_years": WEIGHT_OLD_YEARS,
        "weight_holiday_or_sun": WEIGHT_HOLIDAY_OR_SUN,
        "hyperparams": {
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_estimators": 800,
            "tree_method": "hist",
            "early_stopping_rounds": 30,
        },
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
