"""Tests for the split-conformal band calibration layer.

The contract that matters for production: conformal widens the P10/P90 bands
but leaves P50 EXACTLY untouched (so dispatch P&L is unchanged), preserves
monotonicity, and the fit→apply round-trip hits the target coverage on
exchangeable data.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "smard_merged_15min.parquet"
PRICE_CKPT = ROOT / "model_checkpoints" / "xgboost_price_v1"
CONFORMAL_CFG = PRICE_CKPT / "conformal.json"
COVERED_DAY = date(2026, 3, 15)


def test_fit_apply_hits_target_coverage() -> None:
    """Synthetic check: fit on a calibration sample, coverage on a fresh
    exchangeable sample lands near the 80 % target."""
    from loadforecast import conformal

    rng = np.random.default_rng(0)
    # True y ~ N(0, 1); model bands deliberately too narrow (±1.0 ≈ 68 %).
    def sample(n):
        y = rng.normal(size=n)
        return y, np.full(n, -1.0), np.full(n, 1.0)

    yc, p10c, p90c = sample(4000)
    cal = conformal.fit(yc, p10c, p90c, target_alpha=0.20)
    yt, p10t, p90t = sample(4000)
    p10a, p90a = conformal.apply(p10t, p90t, cal)
    cov = float(((yt >= p10a) & (yt <= p90a)).mean())
    assert 0.77 <= cov <= 0.83, f"calibrated coverage {cov:.3f} not near 0.80"
    assert cal.q_hat > 0, "bands were too narrow; q_hat should widen them"


@pytest.mark.skipif(
    not (PARQUET.exists() and PRICE_CKPT.exists() and CONFORMAL_CFG.exists()),
    reason="parquet / price checkpoint / conformal.json not present",
)
def test_price_wrapper_conformal_keeps_p50_and_widens() -> None:
    """The production wrapper applies conformal: P50 identical, bands wider,
    monotonic preserved."""
    from loadforecast.backtest import issue_time_for, load_smard_15min
    from loadforecast.models.predict import xgboost_price_predict_full

    df = load_smard_15min(str(PARQUET))
    it = issue_time_for(COVERED_DAY)
    raw = xgboost_price_predict_full(df, it, apply_conformal=False)
    cal = xgboost_price_predict_full(df, it, apply_conformal=True)

    assert np.allclose(raw["p50"].to_numpy(), cal["p50"].to_numpy()), \
        "conformal must not touch P50 (dispatch P&L would change)"
    assert (cal["p90"] - cal["p10"]).mean() > (raw["p90"] - raw["p10"]).mean(), \
        "conformal should widen the bands"
    assert ((cal["p10"] <= cal["p50"]) & (cal["p50"] <= cal["p90"])).all(), \
        "monotonicity broken after conformal"
