"""Windowing for the day-ahead price forecaster.

Mirrors `dataset.py` (load model) but with a price-specific feature set.
The architectural choice is the same — encoder reads the past 7 days,
decoder generates 96 quarter-hour predictions for the delivery day,
issue time T = D-1 12:00 Europe/Berlin (the day-ahead market gate).

Encoder features (what a trader can see at issue time):
  - price            day-ahead spot price for DE-LU (the target's history)
  - load             realised national consumption
  - vre_gen          combined PV + wind generation (renewable supply)
  - hour_sin/cos
  - dow_sin/cos

Decoder features (forward-looking, all published by D-1 12:00):
  - tso_load_fc      TSO's published load forecast for D
  - weather (4)      NWP forecasts (population-weighted national)
  - hour_sin/cos
  - dow_sin/cos
  - is_federal_holiday

Target: raw price for the delivery day (96 quarter-hours).

We forecast the raw price (not a residual) because:
  - log-price breaks for the 4 % of negative-price quarter-hours
  - quantile heads handle the wide range (€-500 → €936) on their own
  - there's no single canonical "naive baseline" to subtract that's
    analogous to TSO_fc for load.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.loader import target_index_for
from ..features.availability import usable_columns
from ..features.calendar import calendar_features
from .dataset import LOOKBACK_QH, WEATHER_COLS

PRICE = "price__germany_luxembourg"
LOAD = "actual_cons__grid_load"
PV = "actual_gen__photovoltaics"
WIND_ON = "actual_gen__wind_onshore"
WIND_OFF = "actual_gen__wind_offshore"
TSO_LOAD_FC = "fc_cons__grid_load"
TSO_VRE_FC = "fc_gen__photovoltaics_and_wind"  # the dominant price driver

PRICE_ENC_FEATURE_NAMES = (
    "price", "load", "vre_gen",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)
PRICE_DEC_FEATURE_NAMES = (
    "tso_load_fc",
    "tso_vre_fc",            # SMARD day-ahead PV+wind forecast (the dominant price driver).
    "tso_vre_fc_present",    # 1 if SMARD has published it for this delivery day, 0 otherwise.
    "vre_to_load_ratio",     # fc_gen / fc_load — the mechanistic price driver.
    "vre_percentile",        # fc_gen / 90d-rolling-q90(fc_gen) — "is this a top-1% PV day?"
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_federal_holiday",
)


@dataclass
class PriceWindow:
    issue_time: pd.Timestamp
    X_enc: np.ndarray  # (672, len(PRICE_ENC_FEATURE_NAMES) [+4 weather])
    X_dec: np.ndarray  # (96, len(PRICE_DEC_FEATURE_NAMES) [+4 weather])
    y_price: np.ndarray  # (96,) raw price target
    target_idx: pd.DatetimeIndex


def _encoder_index(issue_time: pd.Timestamp) -> pd.DatetimeIndex:
    end = issue_time
    start = end - pd.Timedelta(minutes=15 * LOOKBACK_QH)
    return pd.date_range(start=start, end=end, freq="15min", inclusive="left")


def _delivery_target_index(issue_time: pd.Timestamp) -> pd.DatetimeIndex:
    delivery_local = issue_time.tz_convert("Europe/Berlin").normalize() + pd.Timedelta(days=1)
    return target_index_for(delivery_local.date())


def build_price_window(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    include_weather: bool = True,
) -> PriceWindow:
    """Build encoder/decoder/target arrays for one issue time.

    Encoder reads 7 days of price + load + VRE history ending at
    issue_time. Decoder reads forward-known features (TSO load forecast,
    weather, calendar) for the delivery day.
    """
    enc_idx = _encoder_index(issue_time)
    target_idx = _delivery_target_index(issue_time)

    needed = (PRICE, LOAD, PV, WIND_ON, WIND_OFF, TSO_LOAD_FC, TSO_VRE_FC)
    if include_weather:
        needed = (*needed, *WEATHER_COLS)
    masked = usable_columns(df, issue_time, include=needed)

    # --- Encoder ----------------------------------------------------
    cal_enc = calendar_features(enc_idx)
    price_h = masked[PRICE].reindex(enc_idx).to_numpy()
    load_h = masked[LOAD].reindex(enc_idx).to_numpy()
    pv = masked[PV].reindex(enc_idx).to_numpy()
    wind_on = masked[WIND_ON].reindex(enc_idx).to_numpy()
    wind_off = masked[WIND_OFF].reindex(enc_idx).to_numpy()
    vre_gen = pv + wind_on + wind_off

    enc_stack = [
        price_h, load_h, vre_gen,
        cal_enc["hour_sin"].to_numpy(),
        cal_enc["hour_cos"].to_numpy(),
        cal_enc["dow_sin"].to_numpy(),
        cal_enc["dow_cos"].to_numpy(),
    ]
    if include_weather:
        for w in WEATHER_COLS:
            enc_stack.append(masked[w].reindex(enc_idx).to_numpy())
    X_enc = np.column_stack(enc_stack).astype(np.float32)

    # --- Decoder ----------------------------------------------------
    # Industry pattern: real desks can't gate on SMARD's VRE day-ahead
    # publication (lands D-1 ~12:30, sometimes later). We feed the forecast
    # *plus* a binary presence flag, and impute NaN→0 when SMARD hasn't
    # published. With matching feature-dropout augmentation in training,
    # the model learns to fall back to weather + calendar gracefully.
    cal_dec = calendar_features(target_idx)
    tso_d = masked[TSO_LOAD_FC].reindex(target_idx).to_numpy()
    vre_fc_d = masked[TSO_VRE_FC].reindex(target_idx).to_numpy()
    vre_present_d = (~np.isnan(vre_fc_d)).astype(np.float32)
    vre_fc_d = np.nan_to_num(vre_fc_d, nan=0.0)

    # Mechanistic features for price extremes (M9). Both engineered to
    # be safe under VRE-missing: when fc_gen is imputed to 0, the ratio
    # and percentile are 0 too — augmented training teaches the model
    # to ignore them in degraded mode (the present flag is the gate).
    safe_load = np.where(tso_d > 0, tso_d, 1.0)
    vre_to_load_ratio = vre_fc_d / safe_load

    # vre_percentile uses the 90-day rolling q90 of fc_gen evaluated at
    # issue_time T (no leakage — only past data). Single scalar denom
    # per delivery day; "1.0" means a typical-PV day, ">1" means above.
    ref_window = df[TSO_VRE_FC].loc[
        issue_time - pd.Timedelta(days=90): issue_time
    ].dropna()
    q90 = float(ref_window.quantile(0.90)) if len(ref_window) > 100 else 1.0
    vre_percentile = vre_fc_d / max(q90, 1.0)

    dec_stack = [
        tso_d, vre_fc_d, vre_present_d,
        vre_to_load_ratio.astype(np.float32),
        vre_percentile.astype(np.float32),
        cal_dec["hour_sin"].to_numpy(),
        cal_dec["hour_cos"].to_numpy(),
        cal_dec["dow_sin"].to_numpy(),
        cal_dec["dow_cos"].to_numpy(),
        cal_dec["is_federal_holiday"].astype(float).to_numpy(),
    ]
    if include_weather:
        for w in WEATHER_COLS:
            dec_stack.append(masked[w].reindex(target_idx).to_numpy())
    X_dec = np.column_stack(dec_stack).astype(np.float32)

    # --- Target -----------------------------------------------------
    # Raw price; uses the *non-masked* frame because at training time
    # we deliberately have ground truth from the future.
    y_price = df[PRICE].reindex(target_idx).to_numpy().astype(np.float32)

    return PriceWindow(
        issue_time=issue_time,
        X_enc=X_enc, X_dec=X_dec,
        y_price=y_price, target_idx=target_idx,
    )


def build_price_dataset(
    df: pd.DataFrame,
    issue_times: Iterable[pd.Timestamp],
    *,
    drop_incomplete: bool = True,
    include_weather: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Stack many windows into training arrays. Mirrors build_dataset()."""
    Xe, Xd, Y, kept = [], [], [], []
    for t in issue_times:
        w = build_price_window(df, t, include_weather=include_weather)
        if drop_incomplete and (
            np.isnan(w.X_enc).any()
            or np.isnan(w.X_dec).any()
            or np.isnan(w.y_price).any()
        ):
            continue
        Xe.append(w.X_enc)
        Xd.append(w.X_dec)
        Y.append(w.y_price)
        kept.append(t)
    if not Xe:
        raise RuntimeError("No complete price windows — check date range and coverage.")
    return np.stack(Xe), np.stack(Xd), np.stack(Y), kept


__all__ = [
    "PRICE",
    "PRICE_DEC_FEATURE_NAMES",
    "PRICE_ENC_FEATURE_NAMES",
    "PriceWindow",
    "build_price_dataset",
    "build_price_window",
]
