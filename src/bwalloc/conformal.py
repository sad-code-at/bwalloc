"""Conformal calibration of allocation levels.

A quantile model that claims 95% coverage frequently delivers something else -- its
quantile estimate inherits whatever bias and miscalibration the fitting procedure had.
For provisioning that gap matters directly: it is the difference between the SLA you
promised and the SLA you deliver.

Split conformal prediction fixes this distribution-free. Given residuals from data the
model did not fit, it adds a single empirical correction so that, under exchangeability,
``P(y <= A) >= tau`` in finite samples regardless of how good the underlying model is.

Three variants live here, in increasing order of what they assume:

:class:`SplitConformal`
    One global correction. Guarantees *marginal* coverage: correct on average over all
    conditions, with no promise about any particular subpopulation.

:class:`LocallyAdaptiveConformal`
    Scales the correction by a fitted uncertainty estimate ``sigma_hat(x)``, so the
    margin widens where the forecast is less reliable. Still uses every residual to
    estimate one quantile, which is what makes it viable on ~900-sample traces.

:class:`MondrianConformal`
    A separate correction per context group, giving *group-conditional* coverage. The
    cleaner guarantee, but it splits the calibration set, and small groups cannot
    support high tau -- see :func:`min_calibration_size`.

Why this matters here
---------------------
On the GP trace, residual sigma is 16.4 Gbps in ordinary conditions and 21.8 during
rainfall -- 33% higher. A marginally calibrated allocator hits its 95% target overall
while under-delivering during exactly the conditions that stress the network. The
locally adaptive and Mondrian variants are the fix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def min_calibration_size(tau: float) -> int:
    """Smallest calibration set that can express a one-sided level ``tau``.

    Split conformal takes the ``ceil((n+1) * tau)``-th order statistic of the
    calibration residuals. That index must not exceed ``n``, which requires
    ``n >= tau / (1 - tau)``.

    So tau=0.95 needs 19 residuals and tau=0.99 needs 99. On the GP trace the
    elevated-risk context group yields roughly 40 calibration residuals, which is why
    context-conditional results in this project are reported at tau <= 0.95 only.
    Requesting more is not conservative -- it silently returns the maximum residual and
    reports a coverage guarantee the data cannot support.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between 0 and 1.")
    # Rounded before the ceiling because binary floating point makes ratios that are
    # exact in decimal come out just above their true value: 0.8 / (1 - 0.8) evaluates
    # to 4.000000000000001, whose ceiling is 5. Without this the guard is off-by-one
    # too strict at several common levels and would refuse a tau the data can support.
    ratio = tau / (1.0 - tau)
    return max(1, int(math.ceil(round(ratio, 9))))


def _conformal_quantile(scores: np.ndarray, tau: float) -> float:
    """The ``ceil((n+1) * tau)``-th order statistic of ``scores``."""
    n = len(scores)
    needed = min_calibration_size(tau)
    if n < needed:
        raise ValueError(
            f"Cannot calibrate tau={tau:g} from {n} residuals; need at least "
            f"{needed}. Lower tau, merge context groups, or pool folds via "
            f"cross-conformal calibration."
        )
    k = int(math.ceil((n + 1) * tau))
    return float(np.sort(scores)[k - 1])


class ConformalCalibrator:
    """Base interface: calibrate on held-out residuals, then adjust a forecast."""

    name = "conformal"

    def calibrate(self, y_calib, pred_calib, **kwargs) -> "ConformalCalibrator":
        raise NotImplementedError  # pragma: no cover

    def allocate(self, pred_test, tau: float, **kwargs) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


@dataclass
class SplitConformal(ConformalCalibrator):
    """Marginal split-conformal calibration.

    Scores are signed residuals ``y - y_hat``; the allocation is
    ``y_hat + q_tau(residuals)``. One-sided by construction, because provisioning only
    cares about the upper edge -- allocating far above demand is wasteful but never an
    SLA breach.
    """

    name: str = "split_conformal"
    residuals_: np.ndarray | None = field(default=None, init=False)

    def calibrate(self, y_calib, pred_calib, **kwargs) -> "SplitConformal":
        y = np.asarray(y_calib, dtype=float).ravel()
        p = np.asarray(pred_calib, dtype=float).ravel()
        if len(y) != len(p):
            raise ValueError("y_calib and pred_calib must have equal length.")
        self.residuals_ = y - p
        return self

    def quantile(self, tau: float) -> float:
        if self.residuals_ is None:
            raise RuntimeError("Call calibrate() first.")
        return _conformal_quantile(self.residuals_, tau)

    def allocate(self, pred_test, tau: float, **kwargs) -> np.ndarray:
        return np.asarray(pred_test, dtype=float).ravel() + self.quantile(tau)


@dataclass
class LocallyAdaptiveConformal(ConformalCalibrator):
    """Normalized ("locally adaptive") conformal calibration.

    Scores are residuals divided by a fitted uncertainty estimate,
    ``(y - y_hat) / sigma_hat(x)``, and the allocation is
    ``y_hat + sigma_hat(x) * q_tau``. The margin therefore widens automatically under
    conditions the uncertainty model flags as volatile, while the quantile ``q_tau``
    is still estimated from the *whole* calibration set.

    That last point is the practical reason this variant leads over Mondrian on these
    traces: it delivers context-adaptive width without paying Mondrian's per-group
    sample cost, so it remains estimable at tau=0.95 where a 40-sample rain group is
    marginal and a 20-sample gathering group is not estimable at all.

    ``sigma_hat`` values are floored at ``sigma_floor`` times their mean to keep the
    normalisation from exploding where the uncertainty model predicts near-zero.
    """

    sigma_floor: float = 0.1
    name: str = "locally_adaptive_conformal"
    scores_: np.ndarray | None = field(default=None, init=False)
    _sigma_mean: float = field(default=1.0, init=False)

    def _clip(self, sigma) -> np.ndarray:
        s = np.asarray(sigma, dtype=float).ravel()
        if np.any(s < 0):
            raise ValueError("sigma_hat must be non-negative.")
        return np.maximum(s, self.sigma_floor * self._sigma_mean)

    def calibrate(self, y_calib, pred_calib, sigma_calib=None, **kwargs
                  ) -> "LocallyAdaptiveConformal":
        if sigma_calib is None:
            raise ValueError("LocallyAdaptiveConformal requires sigma_calib.")
        y = np.asarray(y_calib, dtype=float).ravel()
        p = np.asarray(pred_calib, dtype=float).ravel()
        s = np.asarray(sigma_calib, dtype=float).ravel()
        if not (len(y) == len(p) == len(s)):
            raise ValueError("y_calib, pred_calib and sigma_calib must align.")
        self._sigma_mean = float(np.mean(s)) or 1.0
        self.scores_ = (y - p) / self._clip(s)
        return self

    def quantile(self, tau: float) -> float:
        if self.scores_ is None:
            raise RuntimeError("Call calibrate() first.")
        return _conformal_quantile(self.scores_, tau)

    def allocate(self, pred_test, tau: float, sigma_test=None, **kwargs) -> np.ndarray:
        if sigma_test is None:
            raise ValueError("LocallyAdaptiveConformal requires sigma_test.")
        p = np.asarray(pred_test, dtype=float).ravel()
        return p + self._clip(sigma_test) * self.quantile(tau)


@dataclass
class MondrianConformal(ConformalCalibrator):
    """Group-conditional (Mondrian) conformal calibration.

    Calibrates a separate residual quantile within each context group, giving exact
    finite-sample coverage *per group* rather than only on average. Groups must be
    disjoint, which is why :mod:`bwalloc.context` assigns them by priority rather than
    letting the overlapping raw flags define them.

    Groups whose calibration set is too small for the requested tau fall back to the
    pooled correction, and every fallback is recorded in :attr:`fallbacks_` so the
    results table can report honestly which groups carry a real per-group guarantee
    and which are borrowing the marginal one.
    """

    name: str = "mondrian_conformal"
    min_group: int = 20
    residuals_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    pooled_: np.ndarray | None = field(default=None, init=False)
    fallbacks_: set[str] = field(default_factory=set, init=False)

    def calibrate(self, y_calib, pred_calib, groups_calib=None, **kwargs
                  ) -> "MondrianConformal":
        if groups_calib is None:
            raise ValueError("MondrianConformal requires groups_calib.")
        y = np.asarray(y_calib, dtype=float).ravel()
        p = np.asarray(pred_calib, dtype=float).ravel()
        g = np.asarray(groups_calib).ravel()
        if not (len(y) == len(p) == len(g)):
            raise ValueError("y_calib, pred_calib and groups_calib must align.")
        resid = y - p
        self.pooled_ = resid
        self.residuals_ = {str(k): resid[g == k] for k in np.unique(g)}
        self.fallbacks_ = set()
        return self

    def group_sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.residuals_.items()}

    def quantile(self, tau: float, group: str) -> float:
        if self.pooled_ is None:
            raise RuntimeError("Call calibrate() first.")
        resid = self.residuals_.get(str(group))
        if resid is None or len(resid) < max(self.min_group, min_calibration_size(tau)):
            self.fallbacks_.add(str(group))
            return _conformal_quantile(self.pooled_, tau)
        return _conformal_quantile(resid, tau)

    def allocate(self, pred_test, tau: float, groups_test=None, **kwargs) -> np.ndarray:
        if groups_test is None:
            raise ValueError("MondrianConformal requires groups_test.")
        p = np.asarray(pred_test, dtype=float).ravel()
        g = np.asarray(groups_test).ravel()
        out = np.empty_like(p)
        for key in np.unique(g):
            mask = g == key
            out[mask] = p[mask] + self.quantile(tau, str(key))
        return out


def cross_conformal_residuals(
    per_fold_predictions: pd.DataFrame,
    y_col: str = "y_true",
    pred_col: str = "y_pred",
) -> np.ndarray:
    """Pool calibration residuals across rolling-origin folds.

    With ~900 observations, spending a single calibration split on a rare context
    group leaves too few residuals to calibrate anything. Pooling each fold's
    out-of-sample residuals multiplies the effective calibration set at the cost of
    exact exchangeability -- an acceptable trade here, and one the empirical coverage
    check is there to verify rather than assume.
    """
    return (
        per_fold_predictions[y_col].to_numpy(dtype=float)
        - per_fold_predictions[pred_col].to_numpy(dtype=float)
    )
