"""Multi-horizon forecasting and allocation.

Answers the question the rest of the project ducks: *how far ahead can this actually
be run?* Every other experiment here, and every notebook in the study this extends,
predicts ``y_t`` with ``y_{t-1}`` already in hand -- a nowcast that no operator can
provision from, because the capacity decision has to be made before the previous
measurement arrives.

Three tables come out of this:

``horizon_{op}.csv``
    Accuracy by lead time, with each horizon's own naive baselines. The baseline is
    the important part: an *h*-step model must beat an *h*-step naive forecaster, not
    the one-step one.
``horizon_alloc_{op}.csv``
    The allocation layer re-run at every lead time -- does conformal calibration still
    deliver its promised service level once the forecast has degraded, and at what
    capacity cost?
``horizon_strategy_{op}.csv``
    Direct versus recursive rollout, one model per horizon against one model reused.

    python experiments/run_horizon.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bwalloc as bw  # noqa: E402
from bwalloc.context import assign_binary_groups  # noqa: E402
from bwalloc.data import TARGET, load, sampling_profile  # noqa: E402
from bwalloc.features import FeatureConfig, build_features  # noqa: E402
from bwalloc.forecast import (  # noqa: E402
    direct_design,
    horizon_steps,
    persistence_at_horizon,
    recursive_forecast,
    seasonal_naive_at_horizon,
)
from bwalloc.metrics import accuracy_report  # noqa: E402
from bwalloc.models import default_point_models, xgboost_point  # noqa: E402
from bwalloc.pipeline import coverage_table, run_allocation_backtest  # noqa: E402
from bwalloc.splits import rolling_origin  # noqa: E402

RESULTS = ROOT / "experiments" / "results"

#: Lead times in wall-clock hours, resolved to sample counts per operator. Stated in
#: hours rather than samples for the same reason as everywhere else here: 24 samples
#: is not a day on either trace.
HORIZON_HOURS = (1.5, 3.0, 6.0, 12.0, 24.0)
TAUS = (0.80, 0.90, 0.95)
N_FOLDS = 8


def accuracy_by_horizon(X, y, profile, steps: int) -> pd.DataFrame:
    """Backtest every point model at one lead time, against that lead time's naives."""
    X_h, y_h, _ = direct_design(X, y, steps)
    folds = rolling_origin(
        len(y_h), n_folds=N_FOLDS, calib_frac=0.0, embargo=steps - 1
    )

    # Naive forecasts, computed on the original one-step series and then aligned onto
    # the target timestamps, so each uses only what was observable at the origin.
    naive = {
        "persistence_h": persistence_at_horizon(y, steps).reindex(y_h.index),
        f"seasonal_naive_{profile.daily_period}": seasonal_naive_at_horizon(
            y, steps, profile.daily_period
        ).reindex(y_h.index),
    }

    rows = []
    for fold in folds:
        y_tr = y_h.iloc[fold.train]
        y_te = y_h.iloc[fold.test].to_numpy(dtype=float)
        insample = y_tr.to_numpy(dtype=float)

        for model in default_point_models():
            model.fit(X_h.iloc[fold.train], y_tr)
            pred = model.predict(X_h.iloc[fold.test])
            rows.append(
                {"fold": fold.number, "model": model.name,
                 **accuracy_report(y_te, pred, insample, profile.daily_period)}
            )

        for name, series in naive.items():
            pred = series.iloc[fold.test].to_numpy(dtype=float)
            ok = ~np.isnan(pred)
            if ok.sum() < 3:
                continue
            rows.append(
                {"fold": fold.number, "model": name,
                 **accuracy_report(y_te[ok], pred[ok], insample, profile.daily_period)}
            )

    per_fold = pd.DataFrame(rows)
    return (
        per_fold.groupby("model")
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
             mae_mean=("mae", "mean"), mase_mean=("mase", "mean"),
             folds=("fold", "count"))
        .reset_index()
        .sort_values("rmse_mean")
    )


def strategy_comparison(X, y, df, profile, config, steps: int) -> dict:
    """Direct versus recursive rollout for the same base learner at one lead time."""
    y_full = df[TARGET].astype(float)

    # -- Direct: a model trained to jump the whole gap in one shot.
    X_h, y_h, _ = direct_design(X, y, steps)
    folds_h = rolling_origin(
        len(y_h), n_folds=N_FOLDS, calib_frac=0.0, embargo=steps - 1
    )
    direct_rmse = []
    for fold in folds_h:
        model = xgboost_point().fit(X_h.iloc[fold.train], y_h.iloc[fold.train])
        pred = model.predict(X_h.iloc[fold.test])
        direct_rmse.append(
            float(np.sqrt(np.mean((y_h.iloc[fold.test].to_numpy() - pred) ** 2)))
        )

    # -- Recursive: the one-step model applied `steps` times, on the same fold
    #    schedule and the same target timestamps, so the two are comparable.
    lead = steps - 1
    recursive_rmse = []
    for fold in folds_h:
        # Positions in the one-step frame X; fold indices address the shortened
        # direct frame, whose row i is X row i + lead.
        train_pos = fold.train + lead
        target_pos = fold.test + lead
        model = xgboost_point().fit(X.iloc[train_pos], y.iloc[train_pos])
        pred = recursive_forecast(
            model, X, y_full, profile, config, target_pos, steps
        )
        truth = y.iloc[target_pos].to_numpy(dtype=float)
        ok = ~np.isnan(pred)
        recursive_rmse.append(float(np.sqrt(np.mean((truth[ok] - pred[ok]) ** 2))))

    return {
        "direct_rmse_mean": float(np.mean(direct_rmse)),
        "direct_rmse_std": float(np.std(direct_rmse, ddof=1)),
        "recursive_rmse_mean": float(np.mean(recursive_rmse)),
        "recursive_rmse_std": float(np.std(recursive_rmse, ddof=1)),
    }


def main() -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    config = FeatureConfig()

    for operator in ("gp", "robi"):
        df = load(operator)
        profile = sampling_profile(df)
        X, y = build_features(df, profile, config)
        groups_full = assign_binary_groups(df)

        print("=" * 78)
        print(f"{operator.upper()}  —  multi-horizon study  ({profile.describe()})")
        print("=" * 78)

        acc_rows, alloc_rows, strat_rows = [], [], []

        for hours in HORIZON_HOURS:
            steps = horizon_steps(profile, hours)
            # Lead time from the freshest observation, which sits at origin-1.
            lead_h = profile.hours_for_lag(steps)
            t0 = time.time()

            # -- 1. Accuracy -------------------------------------------------
            acc = accuracy_by_horizon(X, y, profile, steps)
            acc.insert(0, "lead_hours", round(lead_h, 2))
            acc.insert(0, "steps", steps)
            acc.insert(0, "horizon_hours", hours)
            acc_rows.append(acc)

            best = acc.iloc[0]
            naive = acc[acc["model"] == "persistence_h"]
            naive_rmse = float(naive["rmse_mean"].iloc[0]) if not naive.empty else np.nan
            print(
                f"\n  h={hours:>4g}h ({steps:2d} steps, {lead_h:5.2f}h lead): "
                f"best={best['model']:<14} RMSE {best['rmse_mean']:6.3f}"
                f" ± {best['rmse_std']:5.3f}   naive-at-h {naive_rmse:6.3f}"
                f"   ({100 * (best['rmse_mean'] / naive_rmse - 1):+5.1f}%)"
            )

            # -- 2. Allocation at this lead time -----------------------------
            X_h, y_h, _ = direct_design(X, y, steps)
            folds = rolling_origin(
                len(y_h), n_folds=N_FOLDS, calib_frac=0.30, embargo=steps - 1
            )
            groups = groups_full.reindex(X_h.index)
            per_fold, _ = run_allocation_backtest(
                X_h, y_h, folds, groups, model_factory=xgboost_point, taus=TAUS
            )
            per_fold.insert(0, "lead_hours", round(lead_h, 2))
            per_fold.insert(0, "steps", steps)
            per_fold.insert(0, "horizon_hours", hours)
            alloc_rows.append(per_fold)

            cov = coverage_table(per_fold, "coverage")
            ratio = coverage_table(per_fold, "mean_allocation_ratio")
            overall = cov[cov["group"] == "ALL"]
            methods = [c for c in cov.columns if c not in ("tau", "group")]
            for _, r in overall.iterrows():
                rr = ratio[(ratio["tau"] == r["tau"]) & (ratio["group"] == "ALL")]
                achieved = " / ".join(f"{m} {r[m]:.3f}" for m in methods)
                print(
                    f"      tau={r['tau']:.2f}  coverage {achieved}   "
                    f"capacity {float(rr['marginal'].iloc[0]):.3f}x demand"
                )

            # -- 3. Direct vs recursive --------------------------------------
            strat = strategy_comparison(X, y, df, profile, config, steps)
            strat.update(horizon_hours=hours, steps=steps, lead_hours=round(lead_h, 2))
            strat_rows.append(strat)
            print(
                f"      direct {strat['direct_rmse_mean']:6.3f} vs recursive "
                f"{strat['recursive_rmse_mean']:6.3f}"
                f"   [{time.time() - t0:.0f}s]"
            )

        pd.concat(acc_rows, ignore_index=True).assign(operator=operator).to_csv(
            RESULTS / f"horizon_{operator}.csv", index=False
        )
        pd.concat(alloc_rows, ignore_index=True).assign(operator=operator).to_csv(
            RESULTS / f"horizon_alloc_{operator}.csv", index=False
        )
        pd.DataFrame(strat_rows).assign(operator=operator).to_csv(
            RESULTS / f"horizon_strategy_{operator}.csv", index=False
        )
        print()

    print(f"Wrote horizon tables to {RESULTS}")


if __name__ == "__main__":
    main()
