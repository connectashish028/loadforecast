"""Split-conformal calibration for quantile-regression bands.

Production XGBoost quantile bands have 71 % empirical coverage on the
61-day price holdout vs an 80 % nominal target — under-calibrated by
~9 pp. Split-conformal calibration adjusts the bands with a finite-
sample coverage guarantee under exchangeability.

Two variants:

- **marginal** — single scalar q_hat applied uniformly:
    new_p10 = p10 - q_hat,  new_p90 = p90 + q_hat
  Simple, identical adjustment for every slot.

- **adaptive** — q_hat scales by the per-slot band width (Romano et
  al. 2019, "Conformalized Quantile Regression"). Wider-band slots
  get a larger adjustment, narrower-band slots a smaller one:
    new_p10 = p10 - q_hat * width / median_width
    new_p90 = p90 + q_hat * width / median_width
  Matches the intuition that the model's uncertainty is already
  encoded in the predicted width; we only need to scale it.

Both produce a target_alpha coverage guarantee on test exchangeable
with the calibration set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalCalibration:
    """Result of split-conformal calibration on a quantile-regression model.

    Attributes
    ----------
    target_alpha : float
        Nominal miscoverage rate (0.2 for an 80 % band).
    q_hat : float
        The (1 - target_alpha)-quantile of nonconformity scores on the
        calibration set, with finite-sample correction.
    variant : str
        "marginal" or "adaptive".
    cal_size : int
        Number of calibration observations used.
    cal_coverage_pre : float
        Empirical coverage of the un-calibrated bands on the calibration set.
    """

    target_alpha: float
    q_hat: float
    variant: str
    cal_size: int
    cal_coverage_pre: float


def fit(
    y_cal: np.ndarray,
    p10_cal: np.ndarray,
    p90_cal: np.ndarray,
    target_alpha: float = 0.20,
    variant: str = "marginal",
) -> ConformalCalibration:
    """Compute q_hat from a calibration set.

    Parameters
    ----------
    y_cal, p10_cal, p90_cal :
        Flat arrays of realised values and predicted P10 / P90.
    target_alpha :
        Nominal miscoverage. 0.20 → 80 % bands.
    variant :
        "marginal" or "adaptive".
    """
    if y_cal.shape != p10_cal.shape or y_cal.shape != p90_cal.shape:
        raise ValueError("shape mismatch between y_cal / p10_cal / p90_cal")
    mask = ~(np.isnan(y_cal) | np.isnan(p10_cal) | np.isnan(p90_cal))
    y = y_cal[mask]
    p10 = p10_cal[mask]
    p90 = p90_cal[mask]
    n = int(len(y))
    if n < 20:
        raise ValueError(f"calibration set too small: n={n}")
    coverage_pre = float(((y >= p10) & (y <= p90)).mean())

    # Nonconformity score: how far OUTSIDE the band y fell. Negative when inside.
    raw_scores = np.maximum(p10 - y, y - p90)

    if variant == "marginal":
        scores = raw_scores
    elif variant == "adaptive":
        width = p90 - p10
        # Normalise by width — wider bands get proportionally smaller scores
        eps = 1e-3
        scores = raw_scores / np.maximum(width, eps)
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    # Finite-sample-corrected (1 - alpha) quantile
    k = int(np.ceil((n + 1) * (1 - target_alpha)))
    k = min(k, n)
    q_hat = float(np.partition(scores, k - 1)[k - 1])
    return ConformalCalibration(
        target_alpha=target_alpha,
        q_hat=q_hat,
        variant=variant,
        cal_size=n,
        cal_coverage_pre=coverage_pre,
    )


def apply(
    p10: np.ndarray,
    p90: np.ndarray,
    cal: ConformalCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a calibration to new predictions. Returns (p10_adj, p90_adj)."""
    if cal.variant == "marginal":
        return p10 - cal.q_hat, p90 + cal.q_hat
    width = p90 - p10
    return p10 - cal.q_hat * width, p90 + cal.q_hat * width


__all__ = ["ConformalCalibration", "fit", "apply"]
