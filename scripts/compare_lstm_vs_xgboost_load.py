"""Apples-to-apples comparison: LSTM vs XGBoost on the load model's
70-day holdout.

Uses the same dates as `backtest_results/lstm_weather_step7.csv` (the
existing LSTM holdout). For each date, runs both models, scores P50
MAE, and tabulates the comparison. Architecture-justification ablation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.features.build import build_target_day_features
from loadforecast.models.predict import lstm_quantile_predict_full

PARQUET = "smard_merged_15min.parquet"
ACTUAL_LOAD = "actual_cons__grid_load"
TSO_FC = "fc_cons__grid_load"
LSTM_HOLDOUT_CSV = "backtest_results/lstm_weather_step7.csv"
XGB_MODEL_DIR = Path("model_checkpoints/xgboost_load_v1")
OUT_CSV = Path("backtest_results/lstm_vs_xgboost_load.csv")

QUANTILES = (0.10, 0.50, 0.90)


def load_xgb_models() -> dict:
    """Load the three quantile regressors."""
    models = {}
    for q in QUANTILES:
        path = XGB_MODEL_DIR / f"xgb_q{int(q*100):02d}.json"
        reg = xgb.XGBRegressor()
        reg.load_model(path)
        models[q] = reg
    return models


def xgb_predict_for_day(df, issue_time, models) -> pd.DataFrame | None:
    """Returns the (96, 3) P10/P50/P90 frame for the delivery day, in MW.
    Adds the XGBoost residual prediction back to the TSO baseline."""
    features = build_target_day_features(df, issue_time)
    X = features.to_numpy(dtype=np.float32)
    preds = np.stack([models[q].predict(X) for q in QUANTILES], axis=1)  # (96, 3)
    tso = df[TSO_FC].reindex(features.index).to_numpy()
    if np.isnan(tso).any():
        return None
    return pd.DataFrame(
        {
            "p10": tso + preds[:, 0],
            "p50": tso + preds[:, 1],
            "p90": tso + preds[:, 2],
        },
        index=features.index,
    )


def main() -> None:
    print("Loading parquet + models...")
    df = load_smard_15min(PARQUET)
    xgb_models = load_xgb_models()

    # Match the LSTM holdout's exact 70 dates.
    bt = pd.read_csv(LSTM_HOLDOUT_CSV, parse_dates=["issue_date"])
    holdout_dates = sorted(bt["issue_date"].dt.date.unique())
    print(f"Scoring {len(holdout_dates)} holdout days...")

    rows = []
    for i, d in enumerate(holdout_dates):
        if i % 10 == 0:
            print(f"  day {i+1}/{len(holdout_dates)}: {d}")
        issue = issue_time_for(d)

        # LSTM forecast (P50 only; we already have the full LSTM holdout CSV
        # but easier to re-run for the comparison rather than join on it).
        lstm_fc = lstm_quantile_predict_full(df, issue)
        if lstm_fc["p50"].isna().any():
            continue

        # XGBoost forecast.
        xgb_fc = xgb_predict_for_day(df, issue, xgb_models)
        if xgb_fc is None or xgb_fc["p50"].isna().any():
            continue

        # Actuals (the truth).
        actual = df[ACTUAL_LOAD].reindex(lstm_fc.index).to_numpy()
        tso = df[TSO_FC].reindex(lstm_fc.index).to_numpy()
        if np.isnan(actual).any():
            continue

        for ts, lstm_p50, lstm_p10, lstm_p90, xgb_p50, xgb_p10, xgb_p90, a, t in zip(
            lstm_fc.index,
            lstm_fc["p50"], lstm_fc["p10"], lstm_fc["p90"],
            xgb_fc["p50"], xgb_fc["p10"], xgb_fc["p90"],
            actual, tso, strict=True,
        ):
            rows.append({
                "issue_date": str(d),
                "target_ts": ts,
                "y_true": float(a),
                "y_tso": float(t),
                "lstm_p10": float(lstm_p10),
                "lstm_p50": float(lstm_p50),
                "lstm_p90": float(lstm_p90),
                "xgb_p10": float(xgb_p10),
                "xgb_p50": float(xgb_p50),
                "xgb_p90": float(xgb_p90),
            })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # Summary metrics.
    out["abs_err_lstm"] = (out["y_true"] - out["lstm_p50"]).abs()
    out["abs_err_xgb"] = (out["y_true"] - out["xgb_p50"]).abs()
    out["abs_err_tso"] = (out["y_true"] - out["y_tso"]).abs()
    out["lstm_in_band"] = (out["y_true"] >= out["lstm_p10"]) & (out["y_true"] <= out["lstm_p90"])
    out["xgb_in_band"] = (out["y_true"] >= out["xgb_p10"]) & (out["y_true"] <= out["xgb_p90"])

    mae_lstm = float(out["abs_err_lstm"].mean())
    mae_xgb = float(out["abs_err_xgb"].mean())
    mae_tso = float(out["abs_err_tso"].mean())
    cov_lstm = float(out["lstm_in_band"].mean())
    cov_xgb = float(out["xgb_in_band"].mean())
    n_days = out["issue_date"].nunique()

    # Worst-10% days by TSO MAE (= the days where the model lift matters most)
    daily = out.groupby("issue_date").agg(
        mae_lstm=("abs_err_lstm", "mean"),
        mae_xgb=("abs_err_xgb", "mean"),
        mae_tso=("abs_err_tso", "mean"),
    )
    worst_n = max(1, int(0.1 * len(daily)))
    worst10_by_tso = daily.nlargest(worst_n, "mae_tso")
    mae_lstm_worst10 = float(worst10_by_tso["mae_lstm"].mean())
    mae_xgb_worst10 = float(worst10_by_tso["mae_xgb"].mean())
    mae_tso_worst10 = float(worst10_by_tso["mae_tso"].mean())

    # Hour-of-day MAE (where each model wins)
    out["target_ts"] = pd.to_datetime(out["target_ts"], utc=True)
    out["hour"] = out["target_ts"].dt.tz_convert("Europe/Berlin").dt.hour
    hour_mae = out.groupby("hour").agg(
        lstm=("abs_err_lstm", "mean"),
        xgb=("abs_err_xgb", "mean"),
        tso=("abs_err_tso", "mean"),
    )

    print()
    print("=" * 70)
    print(f"LOAD MODEL: LSTM vs XGBoost on {n_days}-day holdout ({len(out):,} 15-min slots)")
    print("=" * 70)
    print()
    print(f"  P50 MAE (MW):           LSTM = {mae_lstm:>6.1f}   "
          f"XGBoost = {mae_xgb:>6.1f}   TSO baseline = {mae_tso:>6.1f}")
    print(f"  Skill score vs TSO:     LSTM = {(1-mae_lstm/mae_tso)*100:>+5.1f} %   "
          f"XGBoost = {(1-mae_xgb/mae_tso)*100:>+5.1f} %")
    print(f"  Worst-10% MAE (MW):     LSTM = {mae_lstm_worst10:>6.1f}   "
          f"XGBoost = {mae_xgb_worst10:>6.1f}   TSO = {mae_tso_worst10:>6.1f}")
    print(f"  80% band coverage:      LSTM = {cov_lstm*100:>5.1f} %   "
          f"XGBoost = {cov_xgb*100:>5.1f} %")
    print()

    # Verdict.
    lstm_better = mae_lstm < mae_xgb
    gap_mw = abs(mae_lstm - mae_xgb)
    gap_pct = gap_mw / mae_xgb * 100 if not lstm_better else gap_mw / mae_lstm * 100
    if gap_mw < 5:
        verdict = f"Tied ({'LSTM' if lstm_better else 'XGBoost'} better by {gap_mw:.1f} MW = {gap_pct:.1f}%)"
    else:
        winner = "LSTM" if lstm_better else "XGBoost"
        verdict = f"{winner} wins by {gap_mw:.1f} MW ({gap_pct:.1f}%)"
    print(f"  Verdict: {verdict}")

    # Model size + training time from meta.json.
    xgb_meta = json.loads((XGB_MODEL_DIR / "meta.json").read_text())
    lstm_meta_path = Path("model_checkpoints/lstm_quantile_v1/meta.json")
    lstm_meta = json.loads(lstm_meta_path.read_text()) if lstm_meta_path.exists() else {}
    print()
    print(f"  Training time:          LSTM = {lstm_meta.get('train_time_s', '?')} s   "
          f"XGBoost = {xgb_meta['train_time_s']:.0f} s")

    print()
    print(f"Wrote {OUT_CSV}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
