"""Forecast accuracy and allocation quality metrics.

Two families live here and they answer different questions.

*Accuracy* metrics (RMSE, MAE, MASE, pinball) score a forecast against what happened.
MASE is included because it is the one number that cannot flatter a model: it is the
model's absolute error divided by a naive forecaster's, so MASE >= 1 means the model
is not worth running. Six of the ten models in the original project would have failed
that test had it been applied.

*Allocation* metrics score a provisioning decision, which is what the project title
actually promises. Under- and over-provisioning are not symmetric failures, so RMSE is
the wrong objective for them; see :mod:`bwalloc.allocation`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _st


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #

def rmse(y_true, y_pred) -> float:
    e = _as_array(y_true) - _as_array(y_pred)
    return float(np.sqrt(np.mean(e**2)))


def mae(y_true, y_pred) -> float:
    e = _as_array(y_true) - _as_array(y_pred)
    return float(np.mean(np.abs(e)))


def mape(y_true, y_pred) -> float:
    yt, yp = _as_array(y_true), _as_array(y_pred)
    nz = yt != 0
    return float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100.0)


def mase(y_true, y_pred, y_insample, season_lag: int = 1) -> float:
    """Mean absolute scaled error.

    Scaled by the in-sample MAE of a seasonal-naive forecaster at ``season_lag``.
    Pass ``season_lag=profile.daily_period`` to scale against daily seasonality, or
    1 for plain persistence.

    A value >= 1 means the model does not beat the naive forecaster it is scaled by.
    """
    ins = _as_array(y_insample)
    if len(ins) <= season_lag:
        raise ValueError("In-sample series too short for the requested season_lag.")
    scale = float(np.mean(np.abs(ins[season_lag:] - ins[:-season_lag])))
    if scale == 0:
        raise ValueError("Naive in-sample error is zero; MASE undefined.")
    return mae(y_true, y_pred) / scale


def pinball_loss(y_true, y_pred, tau: float) -> float:
    """Quantile (pinball) loss at level ``tau``.

    This is the loss whose minimiser is the tau-quantile, and -- via the cost model in
    :mod:`bwalloc.allocation` -- the loss whose minimiser is the cost-optimal
    allocation when ``tau = kappa / (1 + kappa)``.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between 0 and 1.")
    e = _as_array(y_true) - _as_array(y_pred)
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def accuracy_report(y_true, y_pred, y_insample, season_lag: int = 1) -> dict[str, float]:
    """Standard accuracy block, reported for every model on every fold."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_insample, season_lag=season_lag),
    }


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #

def sla_violation_rate(y_true, allocation) -> float:
    """Fraction of intervals where realised demand exceeded the allocation."""
    yt, a = _as_array(y_true), _as_array(allocation)
    return float(np.mean(yt > a))


def overprovision_ratio(y_true, allocation) -> float:
    """Mean fractional excess capacity, over the intervals that were not violated.

    Reported alongside the violation rate because either one alone is trivially
    gameable: allocate infinity and violations go to zero.
    """
    yt, a = _as_array(y_true), _as_array(allocation)
    ok = a >= yt
    if not ok.any():
        return float("nan")
    return float(np.mean((a[ok] - yt[ok]) / yt[ok]))


def mean_allocation(y_true, allocation) -> float:
    """Mean allocated capacity, normalised by mean realised demand.

    The headline efficiency number: 1.20 means the operator provisions 20% above
    average demand across the horizon.
    """
    yt, a = _as_array(y_true), _as_array(allocation)
    return float(np.mean(a) / np.mean(yt))


def asymmetric_cost(y_true, allocation, kappa: float = 10.0,
                    c_over: float = 1.0) -> float:
    """Mean cost per interval under the asymmetric provisioning cost model.

    ``C_t = c_over * max(0, A_t - y_t) + kappa * c_over * max(0, y_t - A_t)``

    ``kappa`` is the ratio of the cost of a unit of unmet demand to the cost of a unit
    of wasted capacity. Operators run this well above 1; the paper sweeps
    {2, 5, 10, 20}.
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    yt, a = _as_array(y_true), _as_array(allocation)
    over = np.maximum(0.0, a - yt)
    under = np.maximum(0.0, yt - a)
    return float(np.mean(c_over * over + kappa * c_over * under))


def coverage(y_true, allocation) -> float:
    """Empirical P(allocation >= demand). The complement of the violation rate."""
    return 1.0 - sla_violation_rate(y_true, allocation)


def coverage_interval(y_true, allocation, confidence: float = 0.95
                      ) -> tuple[float, float, float]:
    """Empirical coverage with a Clopper-Pearson confidence interval.

    Essential when reporting *per-context* coverage: the elevated-risk group on the GP
    trace has on the order of 40 calibration points, and a point estimate of coverage
    from a sample that small is close to meaningless without its interval.

    Returns
    -------
    (point_estimate, lower, upper)
    """
    yt, a = _as_array(y_true), _as_array(allocation)
    n = len(yt)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    k = int(np.sum(a >= yt))
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else _st.beta.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else _st.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return k / n, float(lower), float(upper)


def allocation_report(y_true, allocation, kappas=(2.0, 5.0, 10.0, 20.0)) -> dict[str, float]:
    """Standard allocation block, reported for every policy on every fold."""
    out = {
        "sla_violation_rate": sla_violation_rate(y_true, allocation),
        "overprovision_ratio": overprovision_ratio(y_true, allocation),
        "mean_allocation_ratio": mean_allocation(y_true, allocation),
    }
    for k in kappas:
        out[f"cost_kappa{k:g}"] = asymmetric_cost(y_true, allocation, kappa=k)
    return out


def aggregate_folds(per_fold: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and standard deviation of each metric across folds.

    Every headline number in the paper is reported as ``mean +/- std`` over folds.
    A single-split point estimate on ~300 test points cannot distinguish these models
    and should not be presented as if it can.
    """
    numeric = per_fold.select_dtypes(include="number").columns.difference(["fold"])
    grouped = per_fold.groupby(by)[list(numeric)]
    out = grouped.agg(["mean", "std", "count"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    return out.reset_index()
