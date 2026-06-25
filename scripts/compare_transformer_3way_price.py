"""3-way price comparison: Transformer vs XGBoost (production) vs LSTM
on the 61-day Mar-Apr 2026 holdout. Raw model outputs (no M10 clip, no
conformal) so the comparison is of architectures only.

Reuses the existing predict path: both the LSTM and the Transformer go
through `price_quantile_predict_full` (the transformer via a different
model_dir — the wrapper is architecture-agnostic); XGBoost via the same
engineered-feature helper as compare_lstm_vs_xgboost_price.py. Metrics
(MAE / skill vs naive / 80% coverage / battery dispatch P&L) are computed
generically over the three models.

Output: backtest_results/transformer_3way_price.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import loadforecast.models.transformer_quantile  # noqa: F401  (registers PE layer for load_model)
from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.dispatch import BatterySpec, dispatch_pnl
from loadforecast.features.build import build_target_day_features
from loadforecast.models.predict import price_quantile_predict_full

PARQUET = "smard_merged_15min.parquet"
PRICE_COL = "price__germany_luxembourg"
VRE_FC_COL = "fc_gen__photovoltaics_and_wind"
LSTM_HOLDOUT_CSV = "backtest_results/price_quantile_holdout.csv"
XGB_MODEL_DIR = Path("model_checkpoints/xgboost_price_v1")
TFM_MODEL_DIR = "model_checkpoints/transformer_price_quantile_v1"
OUT_CSV = Path("backtest_results/transformer_3way_price.csv")
QUANTILES = (0.10, 0.50, 0.90)
MODELS = ["lstm", "xgb", "tfm"]  # column prefixes


def load_xgb_models() -> dict:
    out = {}
    for q in QUANTILES:
        reg = xgb.XGBRegressor()
        reg.load_model(XGB_MODEL_DIR / f"xgb_q{int(q*100):02d}.json")
        out[q] = reg
    return out


def _add_engineered_vre(features, df, issue_time):
    out = features.copy()
    vre_fc = out["tso_vre_fc"]
    out["tso_vre_fc_present"] = (~vre_fc.isna()).astype(np.float32)
    out["tso_vre_fc"] = vre_fc.fillna(0.0)
    lf = out["tso_load_fc"]
    out["vre_to_load_ratio"] = (out["tso_vre_fc"] / lf.where(lf > 0, 1.0)).astype(np.float32)
    ref = df[VRE_FC_COL].loc[issue_time - pd.Timedelta(days=90): issue_time].dropna()
    q90 = float(ref.quantile(0.90)) if len(ref) > 100 else 1.0
    out["vre_percentile"] = (out["tso_vre_fc"] / max(q90, 1.0)).astype(np.float32)
    return out


def xgb_predict_for_day(df, issue_time, models):
    feats = _add_engineered_vre(build_target_day_features(df, issue_time), df, issue_time)
    X = feats.to_numpy(dtype=np.float32)
    preds = np.stack([models[q].predict(X) for q in QUANTILES], axis=1)
    return pd.DataFrame({"p10": preds[:, 0], "p50": preds[:, 1], "p90": preds[:, 2]}, index=feats.index)


def main() -> None:
    print("Loading parquet + models...")
    df = load_smard_15min(PARQUET)
    xgb_models = load_xgb_models()

    bt = pd.read_csv(LSTM_HOLDOUT_CSV, parse_dates=["target_ts"])
    holdout_dates = sorted(bt["issue_date"].unique())
    print(f"Scoring {len(holdout_dates)} holdout days...")

    rows = []
    for i, d_str in enumerate(holdout_dates):
        if i % 10 == 0:
            print(f"  day {i+1}/{len(holdout_dates)}: {d_str}")
        issue = issue_time_for(pd.Timestamp(d_str).date())
        fc = {
            "lstm": price_quantile_predict_full(df, issue, apply_extreme_clip=False),
            "tfm": price_quantile_predict_full(df, issue, model_dir=TFM_MODEL_DIR, apply_extreme_clip=False),
            "xgb": xgb_predict_for_day(df, issue, xgb_models),
        }
        if any(f["p50"].isna().any() for f in fc.values()):
            continue
        idx = fc["lstm"].index
        actual = df[PRICE_COL].reindex(idx).to_numpy()
        if np.isnan(actual).any():
            continue
        naive = df[PRICE_COL].reindex(idx - pd.Timedelta(days=1)).set_axis(idx).to_numpy()
        for j, ts in enumerate(idx):
            row = {"issue_date": str(pd.Timestamp(d_str).date()), "target_ts": ts,
                   "y_true": float(actual[j]),
                   "naive": float(naive[j]) if not np.isnan(naive[j]) else np.nan}
            for m in MODELS:
                row[f"{m}_p10"] = float(fc[m]["p10"].iloc[j])
                row[f"{m}_p50"] = float(fc[m]["p50"].iloc[j])
                row[f"{m}_p90"] = float(fc[m]["p90"].iloc[j])
            rows.append(row)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    n_days = out["issue_date"].nunique()
    mae_naive = float((out["y_true"] - out["naive"]).abs().mean(skipna=True))

    # ---- Per-model metrics (generic) ----
    spec = BatterySpec()
    metrics = {}
    for m in MODELS:
        err = (out["y_true"] - out[f"{m}_p50"]).abs()
        cov = ((out["y_true"] >= out[f"{m}_p10"]) & (out["y_true"] <= out[f"{m}_p90"])).mean()
        metrics[m] = {"mae": float(err.mean()), "cov": float(cov)}

    # Battery dispatch P&L per model (P50 greedy), oracle + naive reference.
    pnl = {m: 0.0 for m in MODELS}
    oracle_tot = naive_tot = 0.0
    for _d, day in out.groupby("issue_date"):
        if len(day) != 96 or day["naive"].isna().any():
            continue
        a = day["y_true"].to_numpy()
        n = day["naive"].to_numpy()
        oracle_tot += dispatch_pnl(a, a, a, spec)["net_pnl"]
        naive_tot += dispatch_pnl(n, n, a, spec)["net_pnl"]
        for m in MODELS:
            p50 = day[f"{m}_p50"].to_numpy()
            pnl[m] += dispatch_pnl(p50, p50, a, spec)["net_pnl"]

    name = {"lstm": "LSTM", "xgb": "XGBoost", "tfm": "Transformer"}
    print("\n" + "=" * 74)
    print(f"PRICE: Transformer vs XGBoost vs LSTM — {n_days}-day holdout (raw outputs)")
    print("=" * 74)
    print(f"\n  P50 MAE (EUR/MWh)   [naive yesterday = {mae_naive:.2f}]:")
    for m in MODELS:
        skill = (1 - metrics[m]["mae"] / mae_naive) * 100
        print(f"    {name[m]:<12} {metrics[m]['mae']:>6.2f}   skill vs naive {skill:>+5.1f} %")
    print("\n  80% band coverage (target 80%):")
    for m in MODELS:
        print(f"    {name[m]:<12} {metrics[m]['cov']*100:>5.1f} %")
    print(f"\n  Battery dispatch P&L ({n_days} days, 10MW/20MWh):")
    print(f"    {'perfect-foresight':<20} {oracle_tot:>10,.0f} EUR  (100.0 %)")
    print(f"    {'naive yesterday':<20} {naive_tot:>10,.0f} EUR  ({naive_tot/oracle_tot*100:>5.1f} %)")
    for m in MODELS:
        print(f"    {name[m]+' P50':<20} {pnl[m]:>10,.0f} EUR  ({pnl[m]/oracle_tot*100:>5.1f} %)   "
              f"uplift vs naive {pnl[m]-naive_tot:>+8,.0f}")

    best_mae = min(MODELS, key=lambda m: metrics[m]["mae"])
    best_pnl = max(MODELS, key=lambda m: pnl[m])
    print(f"\n  Verdict: best MAE = {name[best_mae]}; best dispatch P&L = {name[best_pnl]}")
    print(f"\nWrote {OUT_CSV}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
