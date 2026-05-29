"""Smoke tests for the FastAPI inference service.

We use FastAPI's TestClient, which spins the app up in-process — no
network, no external uvicorn process. Lifespan events fire normally,
so the parquet is loaded the same way it would be in production.

These tests load the real parquet + Keras model. Slow (~5-10s on first
run, model is cached after) but worth it: we want to know the live
prediction path works end-to-end, not a mocked stub of it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from loadforecast.serve.api import app

# Skip if the parquet isn't present (e.g. CI without data).
PARQUET = Path("smard_merged_15min.parquet")
pytestmark = pytest.mark.skipif(
    not PARQUET.exists(),
    reason="smard_merged_15min.parquet not present; run loadforecast.data.refresh first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["data_rows"] > 0
    assert "data_through" in body


def test_forecast_known_good_date(client):
    """A 2025-Q2 delivery day has full feature coverage and should return
    a clean 96-step P10/P50/P90 forecast."""
    r = client.post("/forecast", json={"delivery_date": "2025-06-15"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["delivery_date"] == "2025-06-15"
    assert body["model"] == "LoadCast — XGBoost v1"
    assert body["n_steps"] == 96
    assert len(body["horizons"]) == 96

    # Quantile sanity: most steps should satisfy p10 <= p50 <= p90.
    # We allow a tiny crossing rate; the M6 holdout had 0%, but we don't
    # want to fail on a future numerical hiccup.
    crossings = sum(
        1 for h in body["horizons"]
        if not (h["p10"] <= h["p50"] <= h["p90"])
    )
    assert crossings <= 2, f"unexpected quantile crossings: {crossings}/96"

    # Values are MWh per quarter-hour (project's native unit).
    # German load roughly 7-22 GWh/QH (= 28-88 GW instantaneous).
    for h in body["horizons"]:
        assert 5_000 < h["p50"] < 25_000, f"implausible p50: {h}"


def test_forecast_rejects_invalid_date(client):
    r = client.post("/forecast", json={"delivery_date": "not-a-date"})
    assert r.status_code == 422  # Pydantic validation


def test_forecast_rejects_future_beyond_data(client):
    """A delivery_date past the parquet's coverage should return 422,
    not silently fall back to the TSO baseline."""
    r = client.post("/forecast", json={"delivery_date": "2099-01-01"})
    assert r.status_code == 422
    assert "past the latest data" in r.json()["detail"]


# --- Price endpoint -----------------------------------------------------

def test_price_forecast_known_good_date(client):
    """Mid-holdout day with full feature coverage. Verifies shape, model
    name, and that prices are in a plausible €/MWh range."""
    r = client.post("/forecast/price", json={"delivery_date": "2026-04-15"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["delivery_date"] == "2026-04-15"
    assert body["model"] == "PriceCast — XGBoost v1"
    assert body["unit"] == "EUR/MWh"
    assert body["n_steps"] == 96
    assert len(body["horizons"]) == 96
    assert "degraded_mode" in body

    # Quantile sanity. Pinball-loss heads can occasionally cross by a
    # rounding hair; allow a tiny rate.
    crossings = sum(
        1 for h in body["horizons"]
        if not (h["p10"] <= h["p50"] <= h["p90"])
    )
    assert crossings <= 2, f"unexpected quantile crossings: {crossings}/96"

    # Plausible €/MWh range. German DA prices on the holdout sit in
    # roughly [-200, +400]; we widen the bounds to absorb rare extremes.
    for h in body["horizons"]:
        assert -1000 < h["p50"] < 1500, f"implausible p50: {h}"


def test_price_forecast_rejects_future_beyond_data(client):
    r = client.post("/forecast/price", json={"delivery_date": "2099-01-01"})
    assert r.status_code == 422
    assert "past the latest data" in r.json()["detail"]


def test_price_forecast_rejects_invalid_date(client):
    r = client.post("/forecast/price", json={"delivery_date": "not-a-date"})
    assert r.status_code == 422
