"""Tests for per-forecast TreeSHAP explainability.

The contract: SHAP attributions must be EXACTLY additive against the same
prediction production serves (reg.predict, which respects best_iteration) —
otherwise the 'why this forecast' panel explains a different model than the
one live. Also checks the structure and the plain-language renderer.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "smard_merged_15min.parquet"
LOAD_CKPT = ROOT / "model_checkpoints" / "xgboost_load_v1"
PRICE_CKPT = ROOT / "model_checkpoints" / "xgboost_price_v1"

COVERED_DAY = date(2026, 3, 15)

pytestmark = pytest.mark.skipif(
    not (PARQUET.exists() and LOAD_CKPT.exists() and PRICE_CKPT.exists()),
    reason="parquet or checkpoints not present",
)


@pytest.fixture(scope="module")
def df():
    from loadforecast.backtest import load_smard_15min
    return load_smard_15min(str(PARQUET))


@pytest.mark.parametrize("target", ["load", "price"])
def test_shap_is_additive_and_well_formed(df, target) -> None:
    from loadforecast.backtest import issue_time_for
    from loadforecast.models.explain import explain_xgboost_forecast, plain_language

    exp = explain_xgboost_forecast(df, issue_time_for(COVERED_DAY), target=target)

    # Exact additivity against the production prediction path (best_iteration).
    assert exp.additivity_error < 1e-2, (
        f"{target}: SHAP not additive vs production reg.predict "
        f"(err={exp.additivity_error}) — explains the wrong model"
    )
    # Structure
    assert exp.target == target
    assert exp.drivers, "no drivers returned"
    assert all(d.direction in ("up", "down") for d in exp.drivers)
    # Drivers ranked by |contribution|, descending
    mags = [abs(d.contribution) for d in exp.drivers]
    assert mags == sorted(mags, reverse=True)
    # Labels are humanised (no raw lag suffixes leaking through)
    assert all("__lag_" not in d.label for d in exp.drivers)
    # Plain-language renderer produces a non-empty sentence
    text = plain_language(exp)
    assert isinstance(text, str) and len(text) > 20


def test_humanize_known_features() -> None:
    from loadforecast.models.explain import humanize_feature
    assert humanize_feature("tso_residual_err__lag_192qh") == "TSO forecast error (2 days ago)"
    assert humanize_feature("price_austria__lag_672qh") == "Austria price (1 week ago)"
    assert humanize_feature("vre_to_load_ratio") == "Wind + solar as a share of load"
