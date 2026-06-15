"""Per-forecast explainability for the production XGBoost models.

Answers "why did the model say this today?" — the behaviour-explanation
that outcome dashboards (forecast + error + P&L charts) don't provide.

Uses XGBoost's native TreeSHAP (`pred_contribs=True`) — exact, additive
per-feature attributions with no extra dependency. For a delivery day we
compute the SHAP contribution of every feature at each of the 96 quarter-
hours, then aggregate to the mean signed push per feature, so the output
reads as "today's forecast is driven mainly by X (pushing it up €Y) and
Z (pushing it down €W)".

For the LOAD model the prediction is the *residual* (correction on top of
the TSO baseline), so the drivers explain why the model disagrees with the
operator today. For PRICE they explain the raw €/MWh level.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..features.build import build_target_day_features

DEFAULT_XGB_LOAD_DIR = Path("model_checkpoints/xgboost_load_v1")
DEFAULT_XGB_PRICE_DIR = Path("model_checkpoints/xgboost_price_v1")
PRICE_VRE_FC_COL = "fc_gen__photovoltaics_and_wind"

# Human-readable labels for the features a non-technical stakeholder would see.
# Anything not listed falls back to a pattern-based humaniser below.
_LABELS: dict[str, str] = {
    "tso_load_fc": "TSO load forecast",
    "tso_residual_fc": "TSO residual-load forecast (load − renewables)",
    "tso_vre_fc": "Forecast wind + solar generation",
    "tso_residual_share": "Residual-load share",
    "vre_to_load_ratio": "Wind + solar as a share of load",
    "vre_percentile": "Wind + solar vs the recent norm",
    "tso_vre_fc_present": "Renewable forecast availability",
    "is_federal_holiday": "Public holiday",
    "is_bridge_day": "Bridge day (between holiday and weekend)",
    "is_weekend": "Weekend",
    "dom": "Day of month",
    "utc_offset_hours": "Daylight-saving offset",
    "hol_pop_frac": "Population-weighted holiday fraction",
    "week_of_year": "Week of year (season)",
    "month": "Month (season)",
    "load_roll_1d_mean": "Recent load level (1-day average)",
    "load_roll_7d_mean": "Recent load level (7-day average)",
    "load_roll_1d_std": "Recent load volatility (1-day)",
    "load_roll_7d_std": "Recent load volatility (7-day)",
}

_LAG_LABEL = {"192qh": "2 days ago", "672qh": "1 week ago", "1344qh": "2 weeks ago"}
_ZONE_LABEL = {
    "france": "France", "netherlands": "Netherlands", "austria": "Austria",
    "czech_republic": "Czechia", "poland": "Poland", "switzerland": "Switzerland",
}


def humanize_feature(name: str) -> str:
    """Plain-language label for a feature column name."""
    if name in _LABELS:
        return _LABELS[name]
    # cross-border / German price lags: price_<zone>__lag_<X>  /  de_price__lag_<X>
    if name.startswith("price_") and "__lag_" in name:
        zone, lag = name[len("price_"):].split("__lag_")
        return f"{_ZONE_LABEL.get(zone, zone.title())} price ({_LAG_LABEL.get(lag, lag)})"
    if name.startswith("de_price__lag_"):
        return f"German price ({_LAG_LABEL.get(name.split('__lag_')[1], '')})"
    if name.startswith("tso_residual_err__lag_"):
        return f"TSO forecast error ({_LAG_LABEL.get(name.split('__lag_')[1], '')})"
    if name.startswith(("load__lag_", "residual__lag_")):
        kind = "load" if name.startswith("load__lag_") else "residual"
        return f"Recent {kind} ({_LAG_LABEL.get(name.split('__lag_')[1], '')})"
    if name.startswith(("hour", "quarter_of_hour")):
        return "Time of day"
    if name.startswith(("dow", "is_weekend")):
        return "Day of week"
    if name.startswith("month") or name == "week_of_year":
        return "Season / time of year"
    return name


@dataclass(frozen=True)
class Driver:
    feature: str
    label: str
    contribution: float  # mean signed SHAP contribution across the day's 96 slots
    direction: str       # "up" or "down"


@dataclass(frozen=True)
class Explanation:
    target: str               # "load" or "price"
    unit: str                 # "MWh/qh" or "EUR/MWh"
    base_value: float         # model bias (mean prediction before features)
    mean_prediction: float    # mean P50 across the day
    drivers: list[Driver]     # ranked by |contribution|, most influential first
    additivity_error: float   # max |sum(contribs) - prediction|, should be ~0


def _price_features(df: pd.DataFrame, issue_time: pd.Timestamp) -> pd.DataFrame:
    """Replicate the exact 50-feature frame the price model was trained on."""
    f = build_target_day_features(df, issue_time)
    vre = f["tso_vre_fc"]
    f["tso_vre_fc_present"] = (~vre.isna()).astype(np.float32)
    f["tso_vre_fc"] = vre.fillna(0.0)
    lf = f["tso_load_fc"]
    safe = lf.where(lf > 0, 1.0)
    f["vre_to_load_ratio"] = (f["tso_vre_fc"] / safe).astype(np.float32)
    ref = df[PRICE_VRE_FC_COL].loc[issue_time - pd.Timedelta(days=90): issue_time].dropna()
    q90 = float(ref.quantile(0.90)) if len(ref) > 100 else 1.0
    f["vre_percentile"] = (f["tso_vre_fc"] / max(q90, 1.0)).astype(np.float32)
    return f


def explain_xgboost_forecast(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    target: str = "price",
    top_n: int = 6,
    model_dir: Path | str | None = None,
) -> Explanation:
    """Explain the P50 forecast for one delivery day via native TreeSHAP.

    Returns the top `top_n` feature drivers ranked by mean absolute SHAP
    contribution across the 96 quarter-hours, with the signed average push.
    """
    if target == "price":
        model_dir = Path(model_dir or DEFAULT_XGB_PRICE_DIR)
        unit = "EUR/MWh"
        features = _price_features(df, issue_time)
    elif target == "load":
        model_dir = Path(model_dir or DEFAULT_XGB_LOAD_DIR)
        unit = "MWh/qh"
        features = build_target_day_features(df, issue_time)
    else:
        raise ValueError(f"target must be 'load' or 'price', got {target!r}")

    meta = json.loads((model_dir / "meta.json").read_text())
    cols = meta["feature_cols"]
    X = features[cols].to_numpy(dtype=np.float32)

    reg = xgb.XGBRegressor()
    reg.load_model(model_dir / "xgb_q50.json")
    booster = reg.get_booster()
    # Production serves reg.predict, which respects the early-stopping
    # best_iteration. TreeSHAP must use the SAME tree range, or it explains
    # a different model than the one in production.
    best = getattr(reg, "best_iteration", None)
    it_range = (0, best + 1) if best is not None else None
    dm = xgb.DMatrix(X, feature_names=cols)
    contribs = booster.predict(dm, pred_contribs=True, iteration_range=it_range)

    base = float(contribs[:, -1].mean())
    feat_contribs = contribs[:, :-1]           # (96, F)
    mean_signed = feat_contribs.mean(axis=0)   # average push per feature
    pred = reg.predict(X)
    additivity_error = float(np.abs(contribs.sum(axis=1) - pred).max())

    order = np.argsort(-np.abs(mean_signed))[:top_n]
    drivers = [
        Driver(
            feature=cols[i],
            label=humanize_feature(cols[i]),
            contribution=float(mean_signed[i]),
            direction="up" if mean_signed[i] >= 0 else "down",
        )
        for i in order
    ]
    return Explanation(
        target=target,
        unit=unit,
        base_value=base,
        mean_prediction=float(pred.mean()),
        drivers=drivers,
        additivity_error=additivity_error,
    )


def plain_language(exp: Explanation) -> str:
    """One-paragraph, non-technical summary of what drove the forecast."""
    what = ("the model's correction to the TSO load forecast"
            if exp.target == "load" else "today's day-ahead price forecast")
    ups = [d for d in exp.drivers if d.direction == "up"][:2]
    downs = [d for d in exp.drivers if d.direction == "down"][:2]
    parts = []
    if ups:
        parts.append("pushed **up** by " + ", ".join(
            f"{d.label} (+{abs(d.contribution):.1f} {exp.unit})" for d in ups))
    if downs:
        parts.append("pulled **down** by " + ", ".join(
            f"{d.label} (-{abs(d.contribution):.1f} {exp.unit})" for d in downs))
    body = "; ".join(parts) if parts else "near the model's typical level"
    return f"On average across the day, {what} is {body}."


__all__ = ["Driver", "Explanation", "explain_xgboost_forecast", "humanize_feature", "plain_language"]
