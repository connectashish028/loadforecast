"""Drift monitor — daily passive check that production models still work.

Runs in the daily GitHub Action after smoke_tomorrow_predict.py. For
yesterday's delivery day (where actuals are now realised), computes
P50 MAE for FOUR predictors:

  - LSTM load     (MW per quarter-hour, vs realised load)
  - XGBoost load  (same target, architecture-comparison baseline)
  - LSTM price    (EUR/MWh, vs realised clearing price; with M10 clip)
  - XGBoost price (same target, architecture-comparison baseline)

Appends one row per day to `backtest_results/drift_log.csv`. Reading the
log gives the model-health trace over time AND a live architecture
comparison (LSTM vs XGBoost on the same daily data, no cherry-picking).

A 14-day rolling mean crossing 1.5× the original holdout baseline is
the signal that retraining is genuinely warranted. This is a one-way
observability layer — nothing breaks if drift fires.
"""
from __future__ import annotations

import datetime as dt
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xgboost as xgb

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.features.build import build_target_day_features
from loadforecast.models.predict import (
    lstm_quantile_predict_full,
    price_quantile_predict_full,
)

PARQUET = "smard_merged_15min.parquet"
ACTUAL_LOAD = "actual_cons__grid_load"
PRICE_COL = "price__germany_luxembourg"
VRE_FC_COL = "fc_gen__photovoltaics_and_wind"
TSO_FC = "fc_cons__grid_load"
LOG_CSV = Path("backtest_results/drift_log.csv")
XGB_LOAD_DIR = Path("model_checkpoints/xgboost_load_v1")
XGB_PRICE_DIR = Path("model_checkpoints/xgboost_price_v1")
QUANTILES = (0.10, 0.50, 0.90)


def _load_xgb(model_dir: Path):
    if not model_dir.exists():
        return None
    models = {}
    for q in QUANTILES:
        reg = xgb.XGBRegressor()
        reg.load_model(model_dir / f"xgb_q{int(q*100):02d}.json")
        models[q] = reg
    return models


def _add_engineered_vre(features, df, issue_time):
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


def _xgb_predict_load(models, df, issue_time):
    features = build_target_day_features(df, issue_time)
    X = features.to_numpy(dtype=np.float32)
    p50 = models[0.50].predict(X)
    tso = df[TSO_FC].reindex(features.index).to_numpy()
    if np.isnan(tso).any():
        return None
    return p50 + tso  # XGBoost predicts residual; add baseline


def _xgb_predict_price(models, df, issue_time):
    features = build_target_day_features(df, issue_time)
    features = _add_engineered_vre(features, df, issue_time)
    X = features.to_numpy(dtype=np.float32)
    return models[0.50].predict(X)  # raw price target

# Historical holdout baselines (the production-MAE numbers in the README).
LOAD_BASELINE_MAE_MW = 393.0     # 70-day load holdout
PRICE_BASELINE_MAE = 23.8        # 61-day price holdout, EUR/MWh
DRIFT_MULT = 1.5
ROLL_WINDOW_DAYS = 14


def _most_recent_delivery_with_full_actuals(df: pd.DataFrame) -> date | None:
    """Walk back from yesterday until we find a date with all 96 quarter-
    hour actuals realised (no NaN). Caps at 14 days back."""
    today = dt.datetime.now(ZoneInfo("Europe/Berlin")).date()
    for offset in range(1, 14):
        d = today - timedelta(days=offset)
        target_idx = pd.date_range(
            start=pd.Timestamp(d, tz="Europe/Berlin").tz_convert("UTC"),
            periods=96, freq="15min",
        )
        if df[ACTUAL_LOAD].reindex(target_idx).notna().all():
            return d
    return None


def _compute_day_mae(df: pd.DataFrame, delivery_date: date) -> dict:
    """Predict and score all four predictors for a delivery day.
    Returns None per predictor whose forecast couldn't be built."""
    issue = issue_time_for(delivery_date)
    target_idx = pd.date_range(
        start=pd.Timestamp(delivery_date, tz="Europe/Berlin").tz_convert("UTC"),
        periods=96, freq="15min",
    )

    load_actual = df[ACTUAL_LOAD].reindex(target_idx).to_numpy()
    price_actual = df[PRICE_COL].reindex(target_idx).to_numpy()

    # LSTM load
    load_lstm_fc = lstm_quantile_predict_full(df, issue)
    load_mae_lstm = (
        float(np.abs(load_actual - load_lstm_fc["p50"].to_numpy()).mean())
        if not load_lstm_fc["p50"].isna().any() and not np.isnan(load_actual).any()
        else None
    )

    # LSTM price (with M10 clip — the production path)
    price_lstm_fc = price_quantile_predict_full(df, issue)
    price_mae_lstm = (
        float(np.abs(price_actual - price_lstm_fc["p50"].to_numpy()).mean())
        if not price_lstm_fc["p50"].isna().any() and not np.isnan(price_actual).any()
        else None
    )

    # XGBoost load + price (architecture-comparison baseline). Wrapped
    # in try/except — if checkpoints missing or feature builder errors,
    # we still log the LSTM numbers and drop XGB for that day.
    load_mae_xgb = None
    price_mae_xgb = None
    xgb_load = _load_xgb(XGB_LOAD_DIR)
    if xgb_load is not None:
        try:
            xgb_load_p50 = _xgb_predict_load(xgb_load, df, issue)
            if xgb_load_p50 is not None and not np.isnan(load_actual).any():
                load_mae_xgb = float(np.abs(load_actual - xgb_load_p50).mean())
        except Exception:
            pass

    xgb_price = _load_xgb(XGB_PRICE_DIR)
    if xgb_price is not None:
        try:
            xgb_price_p50 = _xgb_predict_price(xgb_price, df, issue)
            if not np.isnan(price_actual).any():
                price_mae_xgb = float(np.abs(price_actual - xgb_price_p50).mean())
        except Exception:
            pass

    return {
        "load_mae_lstm_mw": load_mae_lstm,
        "load_mae_xgb_mw": load_mae_xgb,
        "price_mae_lstm_eur": price_mae_lstm,
        "price_mae_xgb_eur": price_mae_xgb,
    }


def _append_to_log(delivery: date, row: dict) -> pd.DataFrame:
    """Upsert one row per delivery date. Most recent at the bottom.
    Migrates legacy single-model schema if found."""
    new = {"delivery_date": str(delivery), **row}
    if LOG_CSV.exists():
        log = pd.read_csv(LOG_CSV)
        log = _migrate_legacy_columns(log)
        log = log[log["delivery_date"] != str(delivery)]
    else:
        log = pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([new])], ignore_index=True)
    log["delivery_date"] = pd.to_datetime(log["delivery_date"])
    log = log.sort_values("delivery_date").reset_index(drop=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_CSV, index=False)
    return log


def _migrate_legacy_columns(log: pd.DataFrame) -> pd.DataFrame:
    """Older drift_log.csv files used `load_mae_mw` / `price_mae_eur`.
    Migrate to the new dual-model schema (LSTM-specific columns) and
    leave XGBoost columns empty for those historical rows."""
    if "load_mae_mw" in log.columns and "load_mae_lstm_mw" not in log.columns:
        log = log.rename(columns={
            "load_mae_mw": "load_mae_lstm_mw",
            "price_mae_eur": "price_mae_lstm_eur",
        })
    for col in ("load_mae_xgb_mw", "price_mae_xgb_eur"):
        if col not in log.columns:
            log[col] = np.nan
    return log


def _report_drift(log: pd.DataFrame) -> None:
    """Print rolling stats for both architectures."""
    if len(log) < 5:
        print(f"  log has {len(log)} rows — need >=5 for rolling stats.")
        return

    rolls = {}
    for col, label, baseline, thresh in [
        ("load_mae_lstm_mw",  "load  LSTM", LOAD_BASELINE_MAE_MW, LOAD_BASELINE_MAE_MW * DRIFT_MULT),
        ("load_mae_xgb_mw",   "load  XGB ", LOAD_BASELINE_MAE_MW, LOAD_BASELINE_MAE_MW * DRIFT_MULT),
        ("price_mae_lstm_eur","price LSTM", PRICE_BASELINE_MAE,    PRICE_BASELINE_MAE * DRIFT_MULT),
        ("price_mae_xgb_eur", "price XGB ", PRICE_BASELINE_MAE,    PRICE_BASELINE_MAE * DRIFT_MULT),
    ]:
        if col not in log.columns:
            continue
        roll = log[col].rolling(ROLL_WINDOW_DAYS, min_periods=5).mean().iloc[-1]
        rolls[label] = (roll, baseline, thresh)

    print()
    for label, (roll, base, thresh) in rolls.items():
        if pd.isna(roll):
            print(f"  {label} 14d rolling: NaN  (not enough data yet)")
            continue
        alert = roll > thresh
        unit = "MW" if "load" in label else "EUR"
        print(f"  {label} 14d rolling: {roll:>6.2f} {unit}  "
              f"(baseline {base:.1f}, alert >{thresh:.1f})  "
              f"{'!! ALERT' if alert else 'ok'}")


def main() -> None:
    print(f"Drift monitor — loading {PARQUET}")
    df = load_smard_15min(PARQUET)

    delivery = _most_recent_delivery_with_full_actuals(df)
    if delivery is None:
        print("No recent date has full actuals. Skipping.")
        return

    print(f"Scoring delivery {delivery}...")
    row = _compute_day_mae(df, delivery)
    for label, key, unit in [
        ("load  LSTM", "load_mae_lstm_mw", "MW"),
        ("load  XGB ", "load_mae_xgb_mw", "MW"),
        ("price LSTM", "price_mae_lstm_eur", "EUR/MWh"),
        ("price XGB ", "price_mae_xgb_eur", "EUR/MWh"),
    ]:
        v = row.get(key)
        print(f"  {label}: " + (f"{v:.2f} {unit}" if v is not None else "NaN — model couldn't predict"))

    log = _append_to_log(delivery, row)
    print(f"Wrote {LOG_CSV} ({len(log)} rows)")

    _report_drift(log)


if __name__ == "__main__":
    main()
