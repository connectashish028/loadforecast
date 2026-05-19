"""Apples-to-apples comparison: LSTM v4 vs XGBoost on the price model's
61-day Mar-Apr 2026 holdout.

Both models scored RAW (no M10 clip applied — the comparison is of
architectures, not post-processing). The clip is calibrated against
the LSTM specifically, so applying only to LSTM would be unfair;
applying to both requires re-calibration. Easier to compare the raw
outputs first.

Reports: average P50 MAE, spread MAE, worst-10% MAE, naive comparison,
band coverage, training time. Also runs the battery dispatch P&L for
both forecasts so we can see the cost translation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.dispatch import BatterySpec, dispatch_pnl
from loadforecast.features.build import build_target_day_features
from loadforecast.models.predict import price_quantile_predict_full

PARQUET = "smard_merged_15min.parquet"
PRICE_COL = "price__germany_luxembourg"
VRE_FC_COL = "fc_gen__photovoltaics_and_wind"
LSTM_HOLDOUT_CSV = "backtest_results/price_quantile_holdout.csv"
XGB_MODEL_DIR = Path("model_checkpoints/xgboost_price_v1")
OUT_CSV = Path("backtest_results/lstm_vs_xgboost_price.csv")

QUANTILES = (0.10, 0.50, 0.90)


def load_xgb_models() -> dict:
    models = {}
    for q in QUANTILES:
        reg = xgb.XGBRegressor()
        reg.load_model(XGB_MODEL_DIR / f"xgb_q{int(q*100):02d}.json")
        models[q] = reg
    return models


def _add_engineered_vre(features: pd.DataFrame, df: pd.DataFrame,
                       issue_time: pd.Timestamp) -> pd.DataFrame:
    """Mirrors the same engineered features used at training time."""
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


def xgb_predict_for_day(df, issue_time, models) -> pd.DataFrame | None:
    features = build_target_day_features(df, issue_time)
    features = _add_engineered_vre(features, df, issue_time)
    X = features.to_numpy(dtype=np.float32)
    preds = np.stack([models[q].predict(X) for q in QUANTILES], axis=1)
    return pd.DataFrame(
        {"p10": preds[:, 0], "p50": preds[:, 1], "p90": preds[:, 2]},
        index=features.index,
    )


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
        d = pd.Timestamp(d_str).date()
        issue = issue_time_for(d)

        # LSTM forecast — apply_extreme_clip=False to compare RAW outputs.
        lstm_fc = price_quantile_predict_full(df, issue, apply_extreme_clip=False)
        if lstm_fc["p50"].isna().any():
            continue

        # XGBoost forecast.
        xgb_fc = xgb_predict_for_day(df, issue, xgb_models)
        if xgb_fc["p50"].isna().any():
            continue

        actual = df[PRICE_COL].reindex(lstm_fc.index).to_numpy()
        if np.isnan(actual).any():
            continue

        # Naive: yesterday-same-quarter-hour.
        naive = df[PRICE_COL].reindex(
            lstm_fc.index - pd.Timedelta(days=1)
        ).set_axis(lstm_fc.index).to_numpy()

        for ts, lp10, lp50, lp90, xp10, xp50, xp90, a, n in zip(
            lstm_fc.index,
            lstm_fc["p10"], lstm_fc["p50"], lstm_fc["p90"],
            xgb_fc["p10"], xgb_fc["p50"], xgb_fc["p90"],
            actual, naive, strict=True,
        ):
            rows.append({
                "issue_date": str(d),
                "target_ts": ts,
                "y_true": float(a),
                "naive": float(n) if not np.isnan(n) else np.nan,
                "lstm_p10": float(lp10), "lstm_p50": float(lp50), "lstm_p90": float(lp90),
                "xgb_p10":  float(xp10), "xgb_p50":  float(xp50), "xgb_p90":  float(xp90),
            })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    n_days = out["issue_date"].nunique()

    # === Point MAE ===
    out["err_lstm"] = (out["y_true"] - out["lstm_p50"]).abs()
    out["err_xgb"] = (out["y_true"] - out["xgb_p50"]).abs()
    out["err_naive"] = (out["y_true"] - out["naive"]).abs()
    mae_lstm = float(out["err_lstm"].mean())
    mae_xgb = float(out["err_xgb"].mean())
    mae_naive = float(out["err_naive"].mean(skipna=True))

    # === Spread MAE (per day) — what battery dispatch cares about ===
    daily = out.groupby("issue_date").agg(
        spread_true=("y_true", lambda s: s.max() - s.min()),
        spread_lstm=("lstm_p50", lambda s: s.max() - s.min()),
        spread_xgb=("xgb_p50", lambda s: s.max() - s.min()),
        spread_naive=("naive", lambda s: s.max() - s.min()),
        mae_lstm=("err_lstm", "mean"),
        mae_xgb=("err_xgb", "mean"),
        mae_naive=("err_naive", "mean"),
    )
    spread_mae_lstm = float((daily["spread_true"] - daily["spread_lstm"]).abs().mean())
    spread_mae_xgb = float((daily["spread_true"] - daily["spread_xgb"]).abs().mean())
    spread_mae_naive = float((daily["spread_true"] - daily["spread_naive"]).abs().mean())

    # === Worst-10% MAE (by naive — the days where lift matters most) ===
    worst_n = max(1, int(0.1 * len(daily)))
    worst = daily.nlargest(worst_n, "mae_naive")
    w_lstm = float(worst["mae_lstm"].mean())
    w_xgb = float(worst["mae_xgb"].mean())
    w_naive = float(worst["mae_naive"].mean())

    # === 80% band coverage ===
    cov_lstm = float(((out["y_true"] >= out["lstm_p10"]) & (out["y_true"] <= out["lstm_p90"])).mean())
    cov_xgb = float(((out["y_true"] >= out["xgb_p10"]) & (out["y_true"] <= out["xgb_p90"])).mean())

    # === Battery P&L: 10 MW / 20 MWh, both forecasts ===
    spec = BatterySpec()
    pnl_rows = []
    for d_str, day in out.groupby("issue_date"):
        if len(day) != 96:
            continue
        a = day["y_true"].to_numpy()
        n = day["naive"].to_numpy()
        if np.isnan(n).any():
            continue
        oracle = dispatch_pnl(a, a, a, spec)
        naive_pnl = dispatch_pnl(n, n, a, spec)
        lstm_pnl = dispatch_pnl(day["lstm_p50"].to_numpy(), day["lstm_p50"].to_numpy(), a, spec)
        xgb_pnl = dispatch_pnl(day["xgb_p50"].to_numpy(), day["xgb_p50"].to_numpy(), a, spec)
        pnl_rows.append({
            "issue_date": d_str,
            "oracle": oracle["net_pnl"],
            "naive": naive_pnl["net_pnl"],
            "lstm_p50": lstm_pnl["net_pnl"],
            "xgb_p50": xgb_pnl["net_pnl"],
        })
    pnl_df = pd.DataFrame(pnl_rows)
    n_pnl_days = len(pnl_df)
    pct_lstm = pnl_df["lstm_p50"].sum() / pnl_df["oracle"].sum() * 100
    pct_xgb = pnl_df["xgb_p50"].sum() / pnl_df["oracle"].sum() * 100
    pct_naive = pnl_df["naive"].sum() / pnl_df["oracle"].sum() * 100
    uplift_lstm = pnl_df["lstm_p50"].sum() - pnl_df["naive"].sum()
    uplift_xgb = pnl_df["xgb_p50"].sum() - pnl_df["naive"].sum()

    print()
    print("=" * 78)
    print(f"PRICE MODEL: LSTM v4 (no clip) vs XGBoost on {n_days}-day holdout")
    print(f"  ({len(out):,} 15-min slots, raw model outputs, no M10 post-processing)")
    print("=" * 78)
    print()
    print(f"  P50 MAE (EUR/MWh):")
    print(f"    LSTM = {mae_lstm:>6.2f}   XGBoost = {mae_xgb:>6.2f}   Naive yesterday = {mae_naive:>6.2f}")
    print(f"    Skill vs naive: LSTM = {(1-mae_lstm/mae_naive)*100:>+5.1f} %   XGBoost = {(1-mae_xgb/mae_naive)*100:>+5.1f} %")
    print()
    print(f"  Daily spread MAE (EUR/MWh) — what dispatch cares about:")
    print(f"    LSTM = {spread_mae_lstm:>6.2f}   XGBoost = {spread_mae_xgb:>6.2f}   Naive = {spread_mae_naive:>6.2f}")
    print()
    print(f"  Worst-10% days MAE (by naive):")
    print(f"    LSTM = {w_lstm:>6.2f}   XGBoost = {w_xgb:>6.2f}   Naive = {w_naive:>6.2f}")
    print()
    print(f"  80% band coverage:")
    print(f"    LSTM = {cov_lstm*100:>5.1f} %   XGBoost = {cov_xgb*100:>5.1f} %  (target 80%)")
    print()
    print(f"  Battery P&L on {n_pnl_days} days (10 MW / 20 MWh / 90% RTE / 3 cycles/day):")
    print(f"    Perfect-foresight: {pnl_df['oracle'].sum():>10,.0f} EUR  (100.0 %)")
    print(f"    Naive yesterday  : {pnl_df['naive'].sum():>10,.0f} EUR  ({pct_naive:>5.1f} %)")
    print(f"    LSTM P50 (raw)   : {pnl_df['lstm_p50'].sum():>10,.0f} EUR  ({pct_lstm:>5.1f} %)   uplift vs naive: +{uplift_lstm:>8,.0f} EUR")
    print(f"    XGBoost P50      : {pnl_df['xgb_p50'].sum():>10,.0f} EUR  ({pct_xgb:>5.1f} %)   uplift vs naive: +{uplift_xgb:>8,.0f} EUR")
    print()

    # Verdict
    gap_mae = mae_lstm - mae_xgb
    gap_pnl = pct_lstm - pct_xgb
    print(f"  Verdict:")
    if abs(gap_mae) < 1.0:
        print(f"    Average MAE: tied ({'LSTM' if gap_mae < 0 else 'XGBoost'} better by {abs(gap_mae):.2f} EUR/MWh)")
    else:
        print(f"    Average MAE: {'LSTM' if gap_mae < 0 else 'XGBoost'} wins by {abs(gap_mae):.2f} EUR/MWh ({abs(gap_mae)/max(mae_lstm,mae_xgb)*100:.1f}%)")
    print(f"    Worst-10% MAE: {'LSTM' if w_lstm < w_xgb else 'XGBoost'} wins by {abs(w_lstm - w_xgb):.2f} EUR/MWh")
    print(f"    Battery P&L: {'LSTM' if gap_pnl > 0 else 'XGBoost'} wins by {abs(gap_pnl):.2f} pp of perfect-foresight")
    print()
    print(f"Wrote {OUT_CSV}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
