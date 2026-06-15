"""Production-safety test: the production models honour their forecast contract.

Pins the interface of the two production XGBoost models against the
committed parquet on a FIXED in-range historical delivery day (not
"tomorrow", which depends on data freshness and would flake). A model-
file swap, a feature-builder refactor, or a predict-wrapper change that
breaks the contract — wrong shape, NaN on covered data, crossed
quantiles — fails here before it ships.

Contract for each model on a covered day:
  - returns exactly 96 quarter-hour rows
  - columns p10, p50, p90 all finite (no NaN)
  - monotonic per row: p10 <= p50 <= p90 (the sort guard in predict.py)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "smard_merged_15min.parquet"
LOAD_CKPT = ROOT / "model_checkpoints" / "xgboost_load_v1"
PRICE_CKPT = ROOT / "model_checkpoints" / "xgboost_price_v1"

# A delivery day comfortably inside the committed parquet's range, with
# full feature + baseline coverage. Far from the data frontier so it never
# flakes on staleness.
COVERED_DAY = date(2026, 3, 15)

pytestmark = pytest.mark.skipif(
    not (PARQUET.exists() and LOAD_CKPT.exists() and PRICE_CKPT.exists()),
    reason="parquet or production checkpoints not present in this checkout",
)


@pytest.fixture(scope="module")
def df():
    from loadforecast.backtest import load_smard_15min
    return load_smard_15min(str(PARQUET))


def _assert_contract(fc, label: str) -> None:
    assert list(fc.columns) == ["p10", "p50", "p90"], f"{label}: unexpected columns {list(fc.columns)}"
    assert len(fc) == 96, f"{label}: expected 96 rows, got {len(fc)}"
    n_nan = int(fc[["p10", "p50", "p90"]].isna().sum().sum())
    assert n_nan == 0, f"{label}: {n_nan} NaN values on a covered day"
    crossings = int(((fc["p10"] > fc["p50"]) | (fc["p50"] > fc["p90"])).sum())
    assert crossings == 0, f"{label}: {crossings} quantile crossings (sort guard broken)"
    assert np.isfinite(fc.to_numpy()).all(), f"{label}: non-finite values present"


def test_load_model_contract(df) -> None:
    from loadforecast.backtest import issue_time_for
    from loadforecast.models.predict import xgboost_load_predict_full
    fc = xgboost_load_predict_full(df, issue_time_for(COVERED_DAY))
    _assert_contract(fc, "LOAD")
    # Values are MWh per quarter-hour (German load ~35-80 GW => ~9-20k MWh/qh).
    # Band is generous: summer-night trough to winter-evening peak.
    assert fc["p50"].between(5_000, 25_000).all(), "LOAD p50 outside plausible MWh/qh range"


def test_price_model_contract(df) -> None:
    from loadforecast.backtest import issue_time_for
    from loadforecast.models.predict import xgboost_price_predict_full
    fc = xgboost_price_predict_full(df, issue_time_for(COVERED_DAY))
    _assert_contract(fc, "PRICE")
    # Day-ahead price can go negative but stays in a sane band.
    assert fc["p50"].between(-500, 1000).all(), "PRICE p50 outside plausible EUR/MWh range"
