"""3-way load comparison: Transformer vs XGBoost (production) vs LSTM on
the load model's 70-day holdout.

Reuses the architecture-agnostic predict wrappers: LSTM and Transformer
both go through `lstm_quantile_predict_full` (transformer via a different
model_dir); XGBoost via `xgboost_load_predict_full`. All return grid-load
P10/P50/P90 (TSO baseline + residual). Metrics: P50 MAE, skill vs the TSO
published forecast, 80% band coverage — computed generically over the
three models.

Output: backtest_results/transformer_3way_load.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import loadforecast.models.transformer_quantile  # noqa: F401  (registers PE layer for load_model)
from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.predict import (
    lstm_quantile_predict_full,
    xgboost_load_predict_full,
)

PARQUET = "smard_merged_15min.parquet"
ACTUAL_COL = "actual_cons__grid_load"
TSO_COL = "fc_cons__grid_load"
HOLDOUT_CSV = "backtest_results/lstm_weather_step7.csv"
TFM_MODEL_DIR = "model_checkpoints/transformer_quantile_v1"
OUT_CSV = Path("backtest_results/transformer_3way_load.csv")
MODELS = ["lstm", "xgb", "tfm"]


def main() -> None:
    print("Loading parquet + models...")
    df = load_smard_15min(PARQUET)

    bt = pd.read_csv(HOLDOUT_CSV, parse_dates=["issue_date"])
    holdout_dates = sorted(bt["issue_date"].dt.date.unique())
    print(f"Scoring {len(holdout_dates)} holdout days...")

    rows = []
    for i, d in enumerate(holdout_dates):
        if i % 10 == 0:
            print(f"  day {i+1}/{len(holdout_dates)}: {d}")
        issue = issue_time_for(d)
        fc = {
            "lstm": lstm_quantile_predict_full(df, issue),
            "tfm": lstm_quantile_predict_full(df, issue, model_dir=TFM_MODEL_DIR),
            "xgb": xgboost_load_predict_full(df, issue),
        }
        if any(f["p50"].isna().any() for f in fc.values()):
            continue
        idx = fc["lstm"].index
        actual = df[ACTUAL_COL].reindex(idx).to_numpy()
        tso = df[TSO_COL].reindex(idx).to_numpy()
        if np.isnan(actual).any() or np.isnan(tso).any():
            continue
        for j, ts in enumerate(idx):
            row = {"issue_date": str(d), "target_ts": ts,
                   "y_true": float(actual[j]), "y_tso": float(tso[j])}
            for m in MODELS:
                row[f"{m}_p10"] = float(fc[m]["p10"].iloc[j])
                row[f"{m}_p50"] = float(fc[m]["p50"].iloc[j])
                row[f"{m}_p90"] = float(fc[m]["p90"].iloc[j])
            rows.append(row)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    n_days = out["issue_date"].nunique()
    mae_tso = float((out["y_true"] - out["y_tso"]).abs().mean())

    # ---- Per-model metrics + worst-10% by TSO MAE ----
    daily_tso = out.groupby("issue_date").apply(
        lambda g: (g["y_true"] - g["y_tso"]).abs().mean(), include_groups=False)
    worst_n = max(1, int(0.1 * len(daily_tso)))
    worst_days = set(daily_tso.nlargest(worst_n).index)

    name = {"lstm": "LSTM", "xgb": "XGBoost", "tfm": "Transformer"}
    metrics = {}
    for m in MODELS:
        err = (out["y_true"] - out[f"{m}_p50"]).abs()
        cov = ((out["y_true"] >= out[f"{m}_p10"]) & (out["y_true"] <= out[f"{m}_p90"])).mean()
        w = out[out["issue_date"].isin(worst_days)]
        werr = (w["y_true"] - w[f"{m}_p50"]).abs().mean()
        metrics[m] = {"mae": float(err.mean()), "cov": float(cov), "worst": float(werr)}

    print("\n" + "=" * 74)
    print(f"LOAD: Transformer vs XGBoost vs LSTM — {n_days}-day holdout")
    print("=" * 74)
    print(f"\n  P50 MAE (MW)   [TSO baseline = {mae_tso:.1f}]:")
    for m in MODELS:
        skill = (1 - metrics[m]["mae"] / mae_tso) * 100
        print(f"    {name[m]:<12} {metrics[m]['mae']:>6.1f}   skill vs TSO {skill:>+5.1f} %")
    print(f"\n  Worst-10% days MAE (MW)  [TSO = {daily_tso.nlargest(worst_n).mean():.0f}]:")
    for m in MODELS:
        print(f"    {name[m]:<12} {metrics[m]['worst']:>6.1f}")
    print("\n  80% band coverage (target 80%):")
    for m in MODELS:
        print(f"    {name[m]:<12} {metrics[m]['cov']*100:>5.1f} %")

    best_mae = min(MODELS, key=lambda m: metrics[m]["mae"])
    best_worst = min(MODELS, key=lambda m: metrics[m]["worst"])
    print(f"\n  Verdict: best avg MAE = {name[best_mae]}; best worst-10% = {name[best_worst]}")
    print(f"\nWrote {OUT_CSV}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
