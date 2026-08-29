"""Bandwidth allocation policies and the cost model that justifies them.

This module is what the project title promises and the original code never contained:
a decision layer that turns a forecast into an allocated capacity.

The cost argument
-----------------
Provisioning failures are not symmetric. Allocating one Gbps more than needed wastes
capacity; allocating one Gbps less breaches the service level. Writing ``c_over`` for
the unit cost of waste and ``c_under = kappa * c_over`` for the unit cost of unmet
demand, the cost of allocating ``A`` against realised demand ``y`` is

.. math::

    C(A, y) = c_{over}\\,(A - y)^+ + \\kappa\\, c_{over}\\,(y - A)^+

This is the pinball loss up to a positive scale factor. Its minimiser over the
predictive distribution of ``y`` is therefore the **tau-quantile** with

.. math::

    \\tau^* = \\frac{\\kappa}{1 + \\kappa}

So the operator's cost ratio *selects the quantile*: kappa=10 means allocate the
0.909-quantile, kappa=20 the 0.952-quantile. Two consequences follow, and both are
central to the paper:

1. The right model is a **quantile forecaster**, not a point forecaster. An RMSE-
   optimal model targets the conditional mean, which is the cost-optimal allocation
   only in the degenerate case kappa = 1 (waste and outage equally bad) -- a setting
   no operator runs.
2. Reporting RMSE alone cannot rank allocation policies, which is why
   :mod:`bwalloc.metrics` scores violation rate, overprovisioning and cost instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M


def optimal_tau(kappa: float) -> float:
    """Cost-optimal quantile level for an under/over cost ratio of ``kappa``.

    >>> round(optimal_tau(10), 4)
    0.9091
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    return kappa / (1.0 + kappa)


def kappa_for_tau(tau: float) -> float:
    """Inverse of :func:`optimal_tau`: the cost ratio a given quantile is optimal for."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between 0 and 1.")
    return tau / (1.0 - tau)


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #

class AllocationPolicy:
    """Maps a forecast into an allocated capacity."""

    name = "policy"

    def allocate(self, **kwargs) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class QuantileAllocation(AllocationPolicy):
    """Allocate the tau-quantile of the predictive distribution.

    The cost-optimal policy under the model above, provided the quantile is honest.
    Whether it *is* honest is exactly what conformal calibration
    (:mod:`bwalloc.conformal`) enforces and what the coverage checks verify -- an
    uncalibrated quantile model routinely delivers 88% coverage when it claims 95%.
    """

    tau: float = 0.95

    def __post_init__(self):
        self.name = f"quantile_{self.tau:g}"

    def allocate(self, quantile_forecast: pd.DataFrame | np.ndarray = None,
                 **kwargs) -> np.ndarray:
        if quantile_forecast is None:
            raise ValueError("QuantileAllocation requires quantile_forecast.")
        if isinstance(quantile_forecast, pd.DataFrame):
            if self.tau not in quantile_forecast.columns:
                raise KeyError(
                    f"tau={self.tau} not among fitted quantiles "
                    f"{list(quantile_forecast.columns)}"
                )
            return quantile_forecast[self.tau].to_numpy(dtype=float)
        return np.asarray(quantile_forecast, dtype=float).ravel()


@dataclass
class FixedMargin(AllocationPolicy):
    """Point forecast scaled by a fixed safety margin: ``A = (1 + m) * y_hat``.

    This is the incumbent -- what operators actually do, and the policy the paper's
    headline comparison must beat. Its weakness is structural rather than a matter of
    tuning: a single multiplicative margin cannot adapt to conditions where the
    forecast is less reliable, so it necessarily over-provisions in calm periods to
    buy safety in volatile ones.
    """

    margin: float = 0.30

    def __post_init__(self):
        self.name = f"fixed_margin_{self.margin:g}"

    def allocate(self, point_forecast=None, **kwargs) -> np.ndarray:
        if point_forecast is None:
            raise ValueError("FixedMargin requires point_forecast.")
        return (1.0 + self.margin) * np.asarray(point_forecast, dtype=float).ravel()


@dataclass
class StaticPeak(AllocationPolicy):
    """Allocate a constant equal to a high quantile of observed history.

    The crudest and safest incumbent: provision for the peak and stop thinking. Sets
    the upper bound on what a dynamic policy must improve upon.
    """

    history_quantile: float = 1.0

    def __post_init__(self):
        self.name = f"static_peak_{self.history_quantile:g}"

    def allocate(self, history=None, n: int | None = None, **kwargs) -> np.ndarray:
        if history is None or n is None:
            raise ValueError("StaticPeak requires history and n.")
        level = float(np.quantile(np.asarray(history, dtype=float), self.history_quantile))
        return np.full(n, level)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_policy(
    y_true,
    allocation,
    kappas=(2.0, 5.0, 10.0, 20.0),
    label: str = "",
) -> dict[str, float]:
    """Score one allocation against realised demand."""
    report = M.allocation_report(y_true, allocation, kappas=kappas)
    cov, lo, hi = M.coverage_interval(y_true, allocation)
    report.update(coverage=cov, coverage_lo=lo, coverage_hi=hi)
    if label:
        report["policy"] = label
    return report


def pareto_sweep(
    y_true,
    quantile_forecast: pd.DataFrame,
    point_forecast=None,
    margins=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50),
    history=None,
) -> pd.DataFrame:
    """Trace the SLA-violation / over-provisioning frontier for each policy family.

    Produces the paper's headline figure. Each row is one operating point; a policy
    family dominates another when its curve lies below and to the left -- fewer
    violations at the same wasted capacity, or less waste at the same violation rate.

    Returns
    -------
    pandas.DataFrame
        Columns include ``family``, ``setting``, ``sla_violation_rate``,
        ``overprovision_ratio``, ``mean_allocation_ratio`` and per-kappa costs.
    """
    rows: list[dict] = []
    y_true = np.asarray(y_true, dtype=float).ravel()

    for tau in quantile_forecast.columns:
        policy = QuantileAllocation(tau=float(tau))
        alloc = policy.allocate(quantile_forecast=quantile_forecast)
        row = evaluate_policy(y_true, alloc, label=policy.name)
        row.update(family="quantile", setting=float(tau),
                   implied_kappa=kappa_for_tau(float(tau)))
        rows.append(row)

    if point_forecast is not None:
        for m in margins:
            policy = FixedMargin(margin=m)
            alloc = policy.allocate(point_forecast=point_forecast)
            row = evaluate_policy(y_true, alloc, label=policy.name)
            row.update(family="fixed_margin", setting=m)
            rows.append(row)

    if history is not None:
        for q in (0.95, 0.99, 1.0):
            policy = StaticPeak(history_quantile=q)
            alloc = policy.allocate(history=history, n=len(y_true))
            row = evaluate_policy(y_true, alloc, label=policy.name)
            row.update(family="static_peak", setting=q)
            rows.append(row)

    return pd.DataFrame(rows)


def capacity_saving_at_equal_sla(
    pareto: pd.DataFrame,
    family_a: str = "quantile",
    family_b: str = "fixed_margin",
    target_violation: float = 0.01,
) -> dict[str, float]:
    """Capacity saved by ``family_a`` versus ``family_b`` at a common SLA target.

    Answers the sentence the paper is built around: *"at an equal 1% SLA violation
    rate, our allocator provisions X% less capacity than the fixed-margin rule."*

    For each family, picks the operating point with the smallest mean allocation among
    those meeting the target violation rate, then compares. Returns NaNs where a
    family has no operating point that meets the target -- which is itself a result
    worth reporting.
    """
    def _best(family: str) -> pd.Series | None:
        sub = pareto[
            (pareto["family"] == family)
            & (pareto["sla_violation_rate"] <= target_violation)
        ]
        if sub.empty:
            return None
        return sub.loc[sub["mean_allocation_ratio"].idxmin()]

    a, b = _best(family_a), _best(family_b)
    if a is None or b is None:
        return {
            "target_violation": target_violation,
            "alloc_a": float("nan") if a is None else float(a["mean_allocation_ratio"]),
            "alloc_b": float("nan") if b is None else float(b["mean_allocation_ratio"]),
            "capacity_saving": float("nan"),
            "feasible": False,
        }
    saving = (b["mean_allocation_ratio"] - a["mean_allocation_ratio"]) / b["mean_allocation_ratio"]
    return {
        "target_violation": target_violation,
        "setting_a": float(a["setting"]),
        "setting_b": float(b["setting"]),
        "alloc_a": float(a["mean_allocation_ratio"]),
        "alloc_b": float(b["mean_allocation_ratio"]),
        "capacity_saving": float(saving),
        "feasible": True,
    }
