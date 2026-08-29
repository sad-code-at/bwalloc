"""Risk-aware and context-conditional bandwidth allocation.

Produces the study's two headline results:

1. The capacity-risk frontier -- how much capacity a conformally calibrated quantile
   allocator saves against the fixed-margin rule at an equal SLA violation rate.
2. Per-context coverage -- whether a marginally calibrated allocator delivers its
   promised service level inside the high-variance context group, and whether
   context-conditional calibration repairs it.

    python experiments/run_allocation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bwalloc as bw  # noqa: E402
from bwalloc.allocation import (  # noqa: E402
    capacity_saving_at_equal_sla,
    evaluate_policy,
    kappa_for_tau,
    pareto_sweep,
)
from bwalloc.conformal import (  # noqa: E402
    LocallyAdaptiveConformal,
    MondrianConformal,
    SplitConformal,
)
from bwalloc.context import UncertaintyModel, assign_binary_groups  # noqa: E402
from bwalloc.data import load, sampling_profile  # noqa: E402
from bwalloc.features import FeatureConfig, build_features  # noqa: E402
from bwalloc.models import QuantileGBM, xgboost_point  # noqa: E402
from bwalloc.pipeline import (  # noqa: E402
    coverage_table,
    honest_residuals,
    run_allocation_backtest,
)
from bwalloc.splits import rolling_origin  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
TAUS = (0.80, 0.90, 0.95)
QUANTILE_TAUS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
#: Levels swept for the conformal families. Capped at 0.95 because the elevated-risk
#: group cannot support a finer level (see conformal.min_calibration_size).
CONFORMAL_TAUS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95)
N_FOLDS = 8


def frontier(operator: str, X, y, folds, history_source, groups) -> pd.DataFrame:
    """Capacity-risk frontier: every policy family on the same folds.

    Includes the **conformally calibrated** families, which are the proposed method.
    Sweeping the raw quantile model alone understates it badly -- an uncalibrated
    QuantileGBM at tau=0.98 delivers only a ~6% violation rate on GP, so it cannot
    reach a 1% SLA target at all, while the calibrated version can.
    """
    parts = []
    for fold in folds:
        # Calibration data is held out from fitting so conformal residuals are honest.
        X_tr, y_tr = X.iloc[fold.train], y.iloc[fold.train]
        X_ca, y_ca = X.iloc[fold.calib], y.iloc[fold.calib]
        X_te, y_te = X.iloc[fold.test], y.iloc[fold.test]
        y_te_arr = y_te.to_numpy()
        g_ca = groups.iloc[fold.calib].to_numpy()
        g_te = groups.iloc[fold.test].to_numpy()

        qmodel = QuantileGBM(taus=QUANTILE_TAUS).fit(X_tr, y_tr)
        quantiles = qmodel.predict_quantiles(X_te)

        pmodel = xgboost_point().fit(X_tr, y_tr)
        point = pmodel.predict(X_te)
        pred_ca = pmodel.predict(X_ca)

        sweep = pareto_sweep(
            y_te_arr, quantiles,
            point_forecast=point,
            history=history_source[: fold.test[0]],
        )

        # -- The proposed method: conformally calibrated allocation ------------
        resid_tr = honest_residuals(X, y, fold.train, xgboost_point)
        sigma_model = UncertaintyModel().fit(X_tr, resid_tr)

        calibrators = {
            "conformal_marginal": (
                SplitConformal().calibrate(y_ca.to_numpy(), pred_ca), {}
            ),
            "conformal_adaptive": (
                LocallyAdaptiveConformal().calibrate(
                    y_ca.to_numpy(), pred_ca, sigma_calib=sigma_model.predict(X_ca)
                ),
                {"sigma_test": sigma_model.predict(X_te)},
            ),
            "conformal_mondrian": (
                MondrianConformal().calibrate(
                    y_ca.to_numpy(), pred_ca, groups_calib=g_ca
                ),
                {"groups_test": g_te},
            ),
        }

        conformal_rows = []
        for family, (calibrator, kwargs) in calibrators.items():
            for tau in CONFORMAL_TAUS:
                try:
                    alloc = calibrator.allocate(point, tau, **kwargs)
                except ValueError:
                    continue  # estimability guard: this tau is not supported here
                row = evaluate_policy(y_te_arr, alloc, label=f"{family}_{tau:g}")
                row.update(family=family, setting=tau)
                conformal_rows.append(row)

        sweep = pd.concat([sweep, pd.DataFrame(conformal_rows)], ignore_index=True)
        sweep["fold"] = fold.number
        parts.append(sweep)

    allp = pd.concat(parts, ignore_index=True)
    # Average each operating point across folds.
    return (
        allp.groupby(["family", "setting"])
        .agg(
            sla_violation_rate=("sla_violation_rate", "mean"),
            mean_allocation_ratio=("mean_allocation_ratio", "mean"),
            overprovision_ratio=("overprovision_ratio", "mean"),
            cost_kappa10=("cost_kappa10", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
    )


def main() -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    for operator in ("gp", "robi"):
        df = load(operator)
        profile = sampling_profile(df)
        X, y = build_features(df, profile, FeatureConfig())
        groups = assign_binary_groups(df).reindex(X.index)
        folds = rolling_origin(len(y), n_folds=N_FOLDS, calib_frac=0.30)

        print("=" * 78)
        print(f"{operator.upper()}  —  allocation study")
        print("=" * 78)
        print(f"  context groups: {groups.value_counts().to_dict()}")

        # -- 1. Capacity-risk frontier ----------------------------------------
        pareto = frontier(operator, X, y, folds, y.to_numpy(), groups)
        pareto.insert(0, "operator", operator)
        pareto.to_csv(RESULTS / f"pareto_{operator}.csv", index=False)
        print("\n  Capacity–risk frontier (mean across folds):")
        print(
            pareto[["family", "setting", "sla_violation_rate",
                    "mean_allocation_ratio", "cost_kappa10"]]
            .to_string(index=False, float_format=lambda v: f"{v:9.4f}")
        )

        # Compare each conformal family against the incumbent fixed-margin rule.
        savings = []
        for family in ("conformal_adaptive", "conformal_marginal", "quantile"):
            for target in (0.01, 0.02, 0.05, 0.10):
                row = capacity_saving_at_equal_sla(
                    pareto, family_a=family, family_b="fixed_margin",
                    target_violation=target,
                )
                row.update(operator=operator, family=family)
                savings.append(row)
                if row["feasible"]:
                    print(
                        f"    {family:>19} at <= {target:.0%} violations: "
                        f"{row['alloc_a']:.3f}x demand vs fixed-margin "
                        f"{row['alloc_b']:.3f}x -> {row['capacity_saving']:+.1%} capacity"
                    )
                else:
                    print(f"    {family:>19} at <= {target:.0%} violations: "
                          "no feasible operating point")
        pd.DataFrame(savings).to_csv(RESULTS / f"savings_{operator}.csv", index=False)

        # -- 2. Context-conditional calibration --------------------------------
        per_fold, allocations = run_allocation_backtest(
            X, y, folds, groups,
            model_factory=xgboost_point,
            taus=TAUS,
        )
        per_fold.insert(0, "operator", operator)
        per_fold.to_csv(RESULTS / f"allocation_{operator}_perfold.csv", index=False)
        allocations.to_csv(RESULTS / f"allocation_{operator}_points.csv", index=False)

        refused = per_fold[per_fold["error"] != ""]
        if not refused.empty:
            print(f"\n  estimability guard refused {len(refused)} "
                  f"(fold, tau, method) combinations — insufficient calibration data")

        print("\n  Achieved coverage by context group:")
        cov = coverage_table(per_fold, "coverage")
        print(cov.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

        print("\n  Mean allocation / mean demand (the cost of that coverage):")
        alloc = coverage_table(per_fold, "mean_allocation_ratio")
        print(alloc.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

        # The headline context-conditional claim, stated numerically.
        for tau in TAUS:
            sub = cov[(cov["tau"] == tau)]
            if "elevated_risk" not in set(sub["group"]):
                continue
            row = sub[sub["group"] == "elevated_risk"].iloc[0]
            base = sub[sub["group"] == "baseline"].iloc[0]
            print(
                f"\n    tau={tau:g}: marginal calibration delivers "
                f"{base['marginal']:.1%} on baseline but {row['marginal']:.1%} on "
                f"elevated-risk; adaptive delivers {row['adaptive']:.1%}, "
                f"mondrian {row['mondrian']:.1%}  "
                f"[implied cost ratio kappa={kappa_for_tau(tau):.0f}]"
            )
        print()

    print(f"Wrote allocation tables to {RESULTS}")


if __name__ == "__main__":
    main()
