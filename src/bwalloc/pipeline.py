"""End-to-end allocation backtest.

Wires the pieces together: fit a forecaster, estimate its uncertainty, calibrate an
allocation level three ways, and score the resulting decisions -- overall and within
each context group.

The per-fold sequence, and why it is ordered this way:

1. Fit the point forecaster on ``fold.train`` only. The calibration slice is held out
   so its residuals are genuinely out-of-sample; reusing training residuals would
   make the conformal correction optimistic and defeat the entire guarantee.
2. Estimate ``sigma_hat`` from *honest* training residuals, obtained by an internal
   two-fold split of the training window. A tree ensemble's in-sample residuals are
   near zero, so fitting the uncertainty model on them would produce a flat
   ``sigma_hat`` and silently collapse the adaptive method back to the marginal one.
3. Calibrate on ``fold.calib``, allocate on ``fold.test``, score by group.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import metrics as M
from .conformal import LocallyAdaptiveConformal, MondrianConformal, SplitConformal
from .context import UncertaintyModel
from .splits import Fold

ALL_GROUP = "ALL"


def honest_residuals(
    X: pd.DataFrame,
    y: pd.Series,
    idx: np.ndarray,
    model_factory: Callable[[], object],
    n_internal: int = 2,
) -> np.ndarray:
    """Out-of-sample residuals across a training window, via an internal K-fold split.

    Needed because the uncertainty model must learn how large errors *are*, not how
    small they can be made in-sample.
    """
    idx = np.asarray(idx)
    if len(idx) < 2 * n_internal:
        raise ValueError("Training window too small for internal residual estimation.")
    residuals = np.empty(len(idx), dtype=float)
    blocks = np.array_split(np.arange(len(idx)), n_internal)
    for held_out in blocks:
        fit_positions = np.setdiff1d(np.arange(len(idx)), held_out)
        model = model_factory()
        model.fit(X.iloc[idx[fit_positions]], y.iloc[idx[fit_positions]])
        pred = model.predict(X.iloc[idx[held_out]])
        residuals[held_out] = y.iloc[idx[held_out]].to_numpy(dtype=float) - pred
    return residuals


def run_allocation_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[Fold],
    groups: pd.Series,
    model_factory: Callable[[], object],
    taus=(0.8, 0.9, 0.95),
    kappa: float = 10.0,
    methods=("marginal", "adaptive", "mondrian"),
    min_group_eval: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score conformal allocation methods across folds, groups and quantile levels.

    Parameters
    ----------
    groups:
        Context group label per observation, aligned to ``X.index``. Build with
        :func:`bwalloc.context.assign_binary_groups`.
    taus:
        Nominal coverage levels. Each is equivalently a cost ratio via
        :func:`bwalloc.allocation.kappa_for_tau`.
    min_group_eval:
        Skip scoring a group within a fold when it has fewer than this many test
        points, to avoid coverage estimates from a handful of observations.

    Returns
    -------
    (per_fold, allocations)
        ``per_fold`` has one row per (fold, tau, method, group) with coverage,
        allocation ratio and cost. ``allocations`` is long-format with the realised
        allocation for every test point, for plotting and post-hoc analysis.
    """
    y_arr = y.to_numpy(dtype=float)
    groups = groups.reindex(X.index)
    if groups.isna().any():
        raise ValueError("groups must be defined for every row of X.")
    g_arr = groups.to_numpy()

    rows: list[dict] = []
    alloc_rows: list[pd.DataFrame] = []

    for fold in folds:
        if fold.n_calib == 0:
            raise ValueError(
                "Allocation backtest needs a calibration slice; "
                "build folds with calib_frac > 0."
            )

        X_tr, y_tr = X.iloc[fold.train], y.iloc[fold.train]
        X_ca, y_ca = X.iloc[fold.calib], y.iloc[fold.calib]
        X_te = X.iloc[fold.test]
        y_ca_arr = y_ca.to_numpy(dtype=float)
        y_te_arr = y_arr[fold.test]
        g_ca, g_te = g_arr[fold.calib], g_arr[fold.test]

        model = model_factory()
        model.fit(X_tr, y_tr)
        pred_ca = model.predict(X_ca)
        pred_te = model.predict(X_te)

        calibrators: dict[str, tuple[object, dict]] = {}
        if "marginal" in methods:
            calibrators["marginal"] = (
                SplitConformal().calibrate(y_ca_arr, pred_ca), {}
            )
        if "adaptive" in methods:
            resid_tr = honest_residuals(X, y, fold.train, model_factory)
            sigma_model = UncertaintyModel().fit(X_tr, resid_tr)
            calibrators["adaptive"] = (
                LocallyAdaptiveConformal().calibrate(
                    y_ca_arr, pred_ca, sigma_calib=sigma_model.predict(X_ca)
                ),
                {"sigma_test": sigma_model.predict(X_te)},
            )
        if "mondrian" in methods:
            calibrators["mondrian"] = (
                MondrianConformal().calibrate(y_ca_arr, pred_ca, groups_calib=g_ca),
                {"groups_test": g_te},
            )

        for tau in taus:
            for name, (calibrator, kwargs) in calibrators.items():
                try:
                    allocation = calibrator.allocate(pred_te, tau, **kwargs)
                except ValueError as exc:
                    # Raised by the estimability guard when the calibration set is too
                    # small for this tau. Recorded rather than silently skipped.
                    rows.append(
                        {
                            "fold": fold.number, "tau": tau, "method": name,
                            "group": ALL_GROUP, "n": 0, "error": str(exc),
                        }
                    )
                    continue

                alloc_rows.append(
                    pd.DataFrame(
                        {
                            "fold": fold.number, "tau": tau, "method": name,
                            "timestamp": X.index[fold.test],
                            "group": g_te, "y_true": y_te_arr,
                            "allocation": allocation, "point_forecast": pred_te,
                        }
                    )
                )

                for group in [ALL_GROUP, *sorted(pd.unique(g_te))]:
                    mask = (
                        np.ones(len(y_te_arr), dtype=bool)
                        if group == ALL_GROUP
                        else (g_te == group)
                    )
                    if mask.sum() < min_group_eval:
                        continue
                    cov, lo, hi = M.coverage_interval(y_te_arr[mask], allocation[mask])
                    rows.append(
                        {
                            "fold": fold.number,
                            "tau": tau,
                            "method": name,
                            "group": group,
                            "n": int(mask.sum()),
                            "coverage": cov,
                            "coverage_lo": lo,
                            "coverage_hi": hi,
                            "coverage_gap": cov - tau,
                            "sla_violation_rate": M.sla_violation_rate(
                                y_te_arr[mask], allocation[mask]
                            ),
                            "mean_allocation_ratio": M.mean_allocation(
                                y_te_arr[mask], allocation[mask]
                            ),
                            "overprovision_ratio": M.overprovision_ratio(
                                y_te_arr[mask], allocation[mask]
                            ),
                            "cost": M.asymmetric_cost(
                                y_te_arr[mask], allocation[mask], kappa=kappa
                            ),
                            "error": "",
                        }
                    )

    per_fold = pd.DataFrame(rows)
    allocations = (
        pd.concat(alloc_rows, ignore_index=True) if alloc_rows else pd.DataFrame()
    )
    return per_fold, allocations


def coverage_table(per_fold: pd.DataFrame, value: str = "coverage") -> pd.DataFrame:
    """Pivot the backtest into the paper's coverage table: groups by methods, per tau.

    Pooling across folds is a weighted mean by test-block size, not a mean of fold
    means, so small final blocks do not carry disproportionate weight.
    """
    df = per_fold[per_fold["error"] == ""].copy()
    df["_w"] = df["n"] * df[value]
    grouped = df.groupby(["tau", "group", "method"]).agg(
        total_w=("_w", "sum"), total_n=("n", "sum")
    )
    grouped[value] = grouped["total_w"] / grouped["total_n"]
    return (
        grouped.reset_index()
        .pivot(index=["tau", "group"], columns="method", values=value)
        .reset_index()
    )
