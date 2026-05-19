"""Extended-holdout price comparison + feature-ablation in one script.

Two follow-ups to scripts/compare_lstm_vs_xgboost_price.py:

1. **Extended holdout** Mar 1 -> May 5 2026 (66 days incl. May 1's -500 EUR/MWh
   extreme). Tests whether the M10 clip lets LSTM v4 win on the
   deepest tail, where the original 61-day Mar-Apr comparison didn't
   include that case.

2. **XGBoost without engineered VRE features.** Trains an XGBoost-47
   variant on just the features.build set, removing the 3 v4-specific
   engineered features (tso_vre_fc_present, vre_to_load_ratio,
   vre_percentile). Tests how much of the XGBoost win comes from
   feature engineering vs architecture choice.

Reports three models head-to-head, with a special breakout for May 1
and the worst-tail days.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import holidays as hols
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
XGB50_DIR = Path("model_checkpoints/xgboost_price_v1")           # with engineered VRE
XGB47_DIR = Path("model_checkpoints/xgboost_price_v1_stripped")  # without
OUT_CSV = Path("backtest_results/lstm_vs_xgboost_price_extended.csv")
QUANTILES = (0.10, 0.50, 0.90)


def _drange(start, end):
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


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


def _sample_weight(d, holidays_de):
    w = 1.0
    if d.year in (2022, 2023): w *= 0.5
    if d.weekday() == 6 or d in holidays_de: w *= 3.0
    return w


def build_dataset_stripped(df, dates):
    """Same as train_xgboost_price.build_dataset but NO engineered VRE features.
    Drops tso_vre_fc handling — leaves it as native NaN since XGBoost handles."""
    hd = hols.country_holidays("DE", years=range(2022, 2027))
    X_list, y_list, w_list, cols = [], [], [], None
    for d in dates:
        issue = issue_time_for(d)
        try:
            features = build_target_day_features(df, issue)
        except Exception:
            continue
        target = df[PRICE_COL].reindex(features.index).to_numpy()
        if np.isnan(target).any():
            continue
        if cols is None:
            cols = features.columns.tolist()
        X_list.append(features.to_numpy(dtype=np.float32))
        y_list.append(target.astype(np.float32))
        w_list.append(np.full(96, _sample_weight(d, hd), dtype=np.float32))
    return (np.vstack(X_list), np.concatenate(y_list), np.concatenate(w_list), cols)


def train_stripped_xgboost(df):
    """Train XGBoost-47 (no engineered VRE) and save to XGB47_DIR."""
    print("\nTraining XGBoost-47 (no engineered VRE features)...")
    train_dates = _drange(date(2022, 1, 15), date(2025, 12, 31))
    val_dates = _drange(date(2026, 1, 1), date(2026, 2, 28))
    X_tr, y_tr, w_tr, cols = build_dataset_stripped(df, train_dates)
    X_va, y_va, _, _ = build_dataset_stripped(df, val_dates)
    print(f"  train shape: {X_tr.shape}, features: {len(cols)}")

    models = {}
    t0 = time.time()
    for q in QUANTILES:
        reg = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q,
            learning_rate=0.05, max_depth=6, n_estimators=800,
            tree_method="hist", early_stopping_rounds=30, random_state=42,
        )
        reg.fit(X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_va, y_va)], verbose=False)
        models[q] = reg
    print(f"  trained in {time.time()-t0:.0f}s")

    pred = np.stack([models[q].predict(X_va) for q in QUANTILES], axis=1)
    val_mae = float(np.abs(y_va - pred[:, 1]).mean())
    print(f"  val P50 MAE: {val_mae:.2f} EUR/MWh (XGBoost-50 was 11.19)")

    XGB47_DIR.mkdir(parents=True, exist_ok=True)
    for q, reg in models.items():
        reg.save_model(XGB47_DIR / f"xgb_q{int(q*100):02d}.json")
    return models


def load_xgb(model_dir):
    models = {}
    for q in QUANTILES:
        reg = xgb.XGBRegressor()
        reg.load_model(model_dir / f"xgb_q{int(q*100):02d}.json")
        models[q] = reg
    return models


def xgb_predict_with_eng(df, issue_time, models):
    features = build_target_day_features(df, issue_time)
    features = _add_engineered_vre(features, df, issue_time)
    X = features.to_numpy(dtype=np.float32)
    preds = np.stack([models[q].predict(X) for q in QUANTILES], axis=1)
    return pd.DataFrame(
        {"p10": preds[:, 0], "p50": preds[:, 1], "p90": preds[:, 2]},
        index=features.index,
    )


def xgb_predict_stripped(df, issue_time, models):
    features = build_target_day_features(df, issue_time)
    X = features.to_numpy(dtype=np.float32)
    preds = np.stack([models[q].predict(X) for q in QUANTILES], axis=1)
    return pd.DataFrame(
        {"p10": preds[:, 0], "p50": preds[:, 1], "p90": preds[:, 2]},
        index=features.index,
    )


def main():
    print("Loading parquet + LSTM...")
    df = load_smard_15min(PARQUET)
    xgb50 = load_xgb(XGB50_DIR)
    xgb47 = (load_xgb(XGB47_DIR) if XGB47_DIR.exists()
             else train_stripped_xgboost(df))

    holdout_dates = _drange(date(2026, 3, 1), date(2026, 5, 5))
    de_hols = hols.country_holidays("DE", years=range(2022, 2027))
    print(f"\nScoring {len(holdout_dates)}-day extended holdout (incl. May 1)...")

    rows = []
    for i, d in enumerate(holdout_dates):
        if i % 15 == 0: print(f"  day {i+1}/{len(holdout_dates)}: {d}")
        issue = issue_time_for(d)
        # LSTM v4 WITH M10 clip (production path)
        lstm_fc = price_quantile_predict_full(df, issue, apply_extreme_clip=True)
        if lstm_fc["p50"].isna().any():
            continue
        xgb50_fc = xgb_predict_with_eng(df, issue, xgb50)
        xgb47_fc = xgb_predict_stripped(df, issue, xgb47)
        actual = df[PRICE_COL].reindex(lstm_fc.index).to_numpy()
        if np.isnan(actual).any():
            continue
        naive = df[PRICE_COL].reindex(
            lstm_fc.index - pd.Timedelta(days=1)
        ).set_axis(lstm_fc.index).to_numpy()
        if np.isnan(naive).any():
            continue

        for ts, lp, x50p, x47p, a, n in zip(
            lstm_fc.index,
            lstm_fc["p50"], xgb50_fc["p50"], xgb47_fc["p50"],
            actual, naive, strict=True,
        ):
            rows.append({
                "issue_date": str(d), "target_ts": ts,
                "is_holiday_or_weekend": bool(d.weekday() in (5,6) or d in de_hols),
                "y_true": float(a), "naive": float(n),
                "lstm_clip_p50": float(lp),
                "xgb50_p50": float(x50p),
                "xgb47_p50": float(x47p),
            })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    n_days = out["issue_date"].nunique()

    # === Summary ===
    out["e_lstm"] = (out["y_true"] - out["lstm_clip_p50"]).abs()
    out["e_x50"] = (out["y_true"] - out["xgb50_p50"]).abs()
    out["e_x47"] = (out["y_true"] - out["xgb47_p50"]).abs()
    out["e_naive"] = (out["y_true"] - out["naive"]).abs()

    daily = out.groupby("issue_date").agg(
        mae_lstm=("e_lstm", "mean"),
        mae_x50=("e_x50", "mean"),
        mae_x47=("e_x47", "mean"),
        mae_naive=("e_naive", "mean"),
        min_actual=("y_true", "min"),
    )

    def line(name, mae_lstm, mae_x50, mae_x47, mae_naive):
        print(f"  {name:24s} LSTM+clip: {mae_lstm:>6.2f}   "
              f"XGB-50: {mae_x50:>6.2f}   XGB-47: {mae_x47:>6.2f}   "
              f"Naive: {mae_naive:>6.2f}")

    print()
    print("=" * 90)
    print(f"EXTENDED PRICE COMPARISON ({n_days} days, Mar 1 -> May 5 2026)")
    print("=" * 90)
    print()
    line("All days MAE",
         daily.mae_lstm.mean(), daily.mae_x50.mean(),
         daily.mae_x47.mean(), daily.mae_naive.mean())

    worst10 = daily.nlargest(max(1, int(0.1*len(daily))), "mae_naive")
    line("Worst-10% (by naive)",
         worst10.mae_lstm.mean(), worst10.mae_x50.mean(),
         worst10.mae_x47.mean(), worst10.mae_naive.mean())

    hol = daily[daily.index.isin(
        out[out.is_holiday_or_weekend].issue_date.unique()
    )]
    line(f"Holidays/weekends (n={len(hol)})",
         hol.mae_lstm.mean(), hol.mae_x50.mean(),
         hol.mae_x47.mean(), hol.mae_naive.mean())

    neg = daily[daily.min_actual < -100]
    line(f"Deep-negative (min < -100, n={len(neg)})",
         neg.mae_lstm.mean(), neg.mae_x50.mean(),
         neg.mae_x47.mean(), neg.mae_naive.mean())

    print()
    print("Specific extreme cases:")
    for target_d in [date(2026, 5, 1), date(2026, 4, 26), date(2026, 4, 25), date(2026, 4, 6)]:
        if str(target_d) in daily.index:
            r = daily.loc[str(target_d)]
            print(f"  {target_d}  min_actual={r.min_actual:>+6.0f}   "
                  f"LSTM+clip {r.mae_lstm:>5.1f}   XGB-50 {r.mae_x50:>5.1f}   "
                  f"XGB-47 {r.mae_x47:>5.1f}   Naive {r.mae_naive:>5.1f}")

    # === Battery P&L ===
    print()
    print("Battery dispatch P&L (10 MW / 20 MWh, 90% RTE, 3 cycles/day):")
    spec = BatterySpec()
    pnl_rows = []
    for d_str, day in out.groupby("issue_date"):
        if len(day) != 96: continue
        a = day["y_true"].to_numpy()
        n = day["naive"].to_numpy()
        if np.isnan(n).any(): continue
        pnl_rows.append({
            "oracle": dispatch_pnl(a, a, a, spec)["net_pnl"],
            "naive": dispatch_pnl(n, n, a, spec)["net_pnl"],
            "lstm_clip": dispatch_pnl(day["lstm_clip_p50"].to_numpy(),
                                       day["lstm_clip_p50"].to_numpy(), a, spec)["net_pnl"],
            "xgb50": dispatch_pnl(day["xgb50_p50"].to_numpy(),
                                  day["xgb50_p50"].to_numpy(), a, spec)["net_pnl"],
            "xgb47": dispatch_pnl(day["xgb47_p50"].to_numpy(),
                                  day["xgb47_p50"].to_numpy(), a, spec)["net_pnl"],
        })
    p = pd.DataFrame(pnl_rows)
    print(f"  Perfect-foresight: {p.oracle.sum():>10,.0f} EUR")
    for name, col in [("Naive yesterday", "naive"),
                      ("LSTM v4 + clip", "lstm_clip"),
                      ("XGBoost-50 (eng)", "xgb50"),
                      ("XGBoost-47 (stripped)", "xgb47")]:
        pct = p[col].sum() / p.oracle.sum() * 100
        uplift = p[col].sum() - p.naive.sum()
        print(f"  {name:24s} {p[col].sum():>10,.0f} EUR  ({pct:>5.1f}% of perfect)  "
              f"uplift vs naive: {uplift:>+9,.0f} EUR")

    print(f"\nWrote {OUT_CSV}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
