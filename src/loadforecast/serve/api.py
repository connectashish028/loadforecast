"""FastAPI inference service for the day-ahead forecasts.

Runs:
    uvicorn loadforecast.serve.api:app --reload

Endpoints:
    GET  /health          -> liveness + data freshness
    POST /forecast        -> 96 quarter-hour P10/P50/P90 LOAD forecast (MWh / QH)
    POST /forecast/price  -> 96 quarter-hour P10/P50/P90 PRICE forecast (EUR / MWh)

Design:
- Parquet loaded once at startup (lifespan event), not per request. The
  production models are XGBoost quantile regressors (cached on first predict
  in `models.predict`); loading the parquet per request would make the API
  unusable.
- The price endpoint returns split-conformal-calibrated P10/P90 bands when a
  `conformal.json` is present in the price model directory (P50 untouched).
  It also surfaces a `degraded_mode` flag when SMARD's day-ahead VRE forecast
  hasn't published yet for the requested delivery day (XGBoost handles the
  missing input natively, so the forecast still returns).
- All times in responses are tz-aware ISO8601.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ..backtest import issue_time_for, load_smard_15min
from ..models.predict import (
    xgboost_load_predict_full,
    xgboost_price_predict_full,
)

PARQUET_PATH = Path("smard_merged_15min.parquet")
VRE_FC_COL = "fc_gen__photovoltaics_and_wind"


class HorizonPoint(BaseModel):
    """One quarter-hour forecast: timestamp + the three quantiles."""
    timestamp: datetime
    p10: float
    p50: float
    p90: float


class ForecastRequest(BaseModel):
    delivery_date: date = Field(
        ...,
        description="The day to forecast (YYYY-MM-DD). Issue time will be "
                    "the prior day at 12:00 Europe/Berlin (German day-ahead "
                    "gate closure).",
    )


class ForecastResponse(BaseModel):
    delivery_date: date
    issue_time: datetime
    model: str
    n_steps: int
    horizons: list[HorizonPoint]


class PriceForecastResponse(BaseModel):
    delivery_date: date
    issue_time: datetime
    model: str
    n_steps: int
    unit: str = "EUR/MWh"
    degraded_mode: bool = Field(
        ...,
        description="True if SMARD's day-ahead VRE forecast hasn't "
                    "published yet for the delivery day. The model still "
                    "produces a forecast (weather + load + calendar), but "
                    "expect ~+38 % MAE relative to full-feature mode.",
    )
    horizons: list[HorizonPoint]


# Module-level state holds the loaded parquet + model bundle. Populated by
# the lifespan event, cleared on shutdown. Keeps the FastAPI app instance
# free of mutable globals at definition time.
_state: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load the parquet at startup. The XGBoost quantile models are lazy-
    loaded and cached inside the predict module on the first /forecast call;
    this keeps cold-start fast."""
    if not PARQUET_PATH.exists():
        raise RuntimeError(
            f"Parquet not found at {PARQUET_PATH.resolve()}. "
            f"Run `python -m loadforecast.data.refresh` first."
        )
    _state["df"] = load_smard_15min(str(PARQUET_PATH))
    yield
    _state.clear()


app = FastAPI(
    title="German Day-Ahead Forecasts (Load + Price)",
    description=(
        "Production XGBoost quantile forecasters for German grid load "
        "and EPEX day-ahead clearing price. On a 10 MW / 20 MWh battery "
        "dispatched against the price forecast, the system captures "
        "~97 % of perfect-foresight P&L (+€65 k uplift vs naive over 61 "
        "days). LSTM is kept as a comparison baseline; see "
        "/forecast/architecture for the ablation."
    ),
    version="0.3.0",
    lifespan=_lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect to the interactive docs so the bare URL is useful."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    df = _state.get("df")
    if df is None:
        raise HTTPException(status_code=503, detail="data not loaded")
    return {
        "status": "ok",
        "data_rows": int(len(df)),
        "data_through": df.index.max().isoformat(),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    df = _state.get("df")
    if df is None:
        raise HTTPException(status_code=503, detail="data not loaded")

    issue = issue_time_for(req.delivery_date)

    # Hard fail if the issue time is past the parquet's coverage — the model
    # would just hand back the TSO forecast (its NaN-window fallback). That's
    # a misleading silent success; better to 422 explicitly.
    if issue > df.index.max():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Issue time {issue.isoformat()} is past the latest data "
                f"({df.index.max().isoformat()}). Refresh the parquet first."
            ),
        )

    out = xgboost_load_predict_full(df, issue)
    if out["p50"].isna().any():
        raise HTTPException(
            status_code=422,
            detail="Encoder/decoder window has missing values for that "
                   "issue date — try a date with full feature coverage.",
        )

    horizons = [
        HorizonPoint(
            timestamp=ts.to_pydatetime(),
            p10=float(row.p10),
            p50=float(row.p50),
            p90=float(row.p90),
        )
        for ts, row in out.iterrows()
    ]
    return ForecastResponse(
        delivery_date=req.delivery_date,
        issue_time=issue.to_pydatetime(),
        model="LoadCast — XGBoost v1",
        n_steps=len(horizons),
        horizons=horizons,
    )


@app.post("/forecast/price", response_model=PriceForecastResponse)
def forecast_price(req: ForecastRequest) -> PriceForecastResponse:
    """Probabilistic day-ahead price forecast (EUR/MWh).

    Targets the EPEX clearing price directly (no published baseline to
    subtract). P10/P90 are split-conformal-calibrated (P50 untouched).
    Surfaces a `degraded_mode` flag when SMARD's VRE day-ahead forecast
    hasn't published for the delivery day — XGBoost handles the missing
    input natively via tree splits, so the forecast still returns, but
    accuracy is lower without the VRE feature.
    """
    df = _state.get("df")
    if df is None:
        raise HTTPException(status_code=503, detail="data not loaded")

    issue = issue_time_for(req.delivery_date)
    if issue > df.index.max():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Issue time {issue.isoformat()} is past the latest data "
                f"({df.index.max().isoformat()}). Refresh the parquet first."
            ),
        )

    out = xgboost_price_predict_full(df, issue)
    if out["p50"].isna().any():
        raise HTTPException(
            status_code=422,
            detail="Encoder/decoder window has missing values for that "
                   "issue date — try a date with full feature coverage.",
        )

    # Detect degraded mode the same way the dashboard does: VRE forecast
    # column is fully NaN for the delivery day.
    target_idx = out.index
    degraded = bool(
        VRE_FC_COL in df.columns
        and df[VRE_FC_COL].reindex(target_idx).isna().all()
    )

    horizons = [
        HorizonPoint(
            timestamp=ts.to_pydatetime(),
            p10=float(row.p10),
            p50=float(row.p50),
            p90=float(row.p90),
        )
        for ts, row in out.iterrows()
    ]
    return PriceForecastResponse(
        delivery_date=req.delivery_date,
        issue_time=issue.to_pydatetime(),
        model="PriceCast — XGBoost v1",
        n_steps=len(horizons),
        degraded_mode=degraded,
        horizons=horizons,
    )


__all__ = [
    "app",
    "ForecastRequest",
    "ForecastResponse",
    "HorizonPoint",
    "PriceForecastResponse",
]
