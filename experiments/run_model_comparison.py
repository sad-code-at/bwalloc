"""The final field: every model, default and tuned, on one fold schedule.

The three studies that precede this one each answer a narrower question --
``run_full_features.py`` asks what the feature set is worth, ``run_architectures.py``
what the architecture is worth, ``run_tuning.py`` what tuning is worth. This one puts
the survivors in a single table so the comparison is like-for-like, which is the
standard the corrected protocol has held since notebook 01.

Each model appears three times: at its defaults with covariate channels, at the
same defaults without them, and at the configuration
``run_tuning.py`` selected on the development prefix. Both are scored on the same eight
rolling-origin folds, against the same baselines, with the same Diebold-Mariano
machinery and Benjamini-Hochberg correction over the whole field.

A word on how to read the output, because it matters more than the ranking. With eight
folds and roughly ninety test points per block, most pairs will not separate. The
``winner`` column says ``tie`` when that is the case, and a tie is the honest answer --
this project does not rank on point estimates.

    python experiments/run_model_comparison.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bwalloc as bw  # noqa: E402
from bwalloc.baselines import SeasonalNaive, standard_baselines  # noqa: E402
from bwalloc.data import load, sampling_profile  # noqa: E402
from bwalloc.evaluate import beats_baseline, dm_matrix, run_backtest, summarise  # noqa: E402
from bwalloc.features import build_features  # noqa: E402
from bwalloc.sequence import full_feature_config  # noqa: E402
from bwalloc.splits import rolling_origin  # noqa: E402
from bwalloc.tuning import build_model, feature_config_for, load_best  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from run_tuning import DEFAULTS  # noqa: E402


def without_covariates(params: dict) -> dict:
    """The same defaults with the covariate channels switched off.

    This matters for honesty about what tuning bought. ``run_tuning.DEFAULTS`` mostly
    carries ``covariates=True``, because that was the configuration this round set out
    to test -- so a headline like "tuning improved the CNN by 42%" would be mostly
    reporting that the search turned off covariate channels we had *separately* shown
    are harmful on both traces.

    Scoring both defaults separates the two effects: ``default_cov`` -> ``default``
    is what dropping the covariates is worth, and ``default`` -> ``tuned`` is what the
    hyperparameter search is worth on top of that.
    """
    return {**params, "covariates": False}

RESULTS = ROOT / "experiments" / "results"
N_FOLDS = 8
LOOKBACK = 24


def score(name: str, params: dict, label: str, df, profile) -> tuple:
    """Backtest one configuration, tagged so default and tuned stay distinguishable."""
    X, y = build_features(df, profile, feature_config_for(params))
    folds = rolling_origin(len(y), n_folds=N_FOLDS)
    model = build_model(name, params)
    model.name = f"{name}__{label}"
    per_fold, predictions = run_backtest(
        X, y, folds, models=[model], baselines=[], season_lag=profile.daily_period,
    )
    per_fold["base_model"] = name
    per_fold["variant"] = label
    per_fold["n_parameters"] = getattr(model, "n_parameters", 0)
    return per_fold, predictions


def main() -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    best_path = RESULTS / "tuning_best.csv"
    if not best_path.exists():
        raise SystemExit(
            "tuning_best.csv not found. Run experiments/run_tuning.py first -- this "
            "script reports the tuned field and there is nothing tuned yet."
        )

    for operator in ("gp", "robi"):
        df = load(operator)
        profile = sampling_profile(df)
        tuned = load_best(best_path, operator)

        # Baselines come from the reference design matrix so every model is measured
        # against the same naive floor, whatever window its own configuration uses.
        X_ref, y_ref = build_features(
            df, profile, full_feature_config(LOOKBACK, covariates=True)
        )
        folds = rolling_origin(len(y_ref), n_folds=N_FOLDS)
        baselines = standard_baselines(y_ref.to_numpy(), profile.daily_period)
        baselines.append(SeasonalNaive(y_ref.to_numpy(), period=24))
        base_fold, base_pred = run_backtest(
            X_ref, y_ref, folds, models=[], baselines=baselines,
            season_lag=profile.daily_period,
        )
        base_fold["base_model"] = base_fold["model"]
        base_fold["variant"] = "baseline"
        base_fold["n_parameters"] = 0

        print("=" * 100)
        print(f"{operator.upper()}  —  {len(tuned)} tuned models + defaults + "
              f"{len(baselines)} baselines, {N_FOLDS} folds")
        print("=" * 100)

        frames, preds = [base_fold], [base_pred]
        started = time.time()
        for name, params in tuned.items():
            variants = (
                ("default_cov", DEFAULTS[name]),
                ("default", without_covariates(DEFAULTS[name])),
                ("tuned", params),
            )
            for label, config in variants:
                per_fold, predictions = score(name, config, label, df, profile)
                frames.append(per_fold)
                preds.append(predictions)
            print(f"  {name:14} done")

        per_fold = pd.concat(frames, ignore_index=True)
        predictions = pd.concat(preds, ignore_index=True)
        summary = beats_baseline(summarise(per_fold))
        summary["operator"] = operator

        print(f"\n({time.time() - started:.0f}s)")
        print(summary[["model", "rmse_mean", "rmse_std", "mae_mean", "mase_mean",
                       "vs_persistence", "beats_persistence"]]
              .to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

        per_fold.to_csv(RESULTS / f"comparison_{operator}_perfold.csv", index=False)
        summary.to_csv(RESULTS / f"comparison_{operator}_summary.csv", index=False)

        dm = dm_matrix(predictions, horizon=1)
        dm.to_csv(RESULTS / f"comparison_{operator}_dm.csv", index=False)

        # Separate the two effects rather than letting one stand in for the other.
        split = (per_fold[per_fold["variant"] != "baseline"]
                 .groupby(["base_model", "variant"], as_index=False)["rmse"].mean()
                 .pivot(index="base_model", columns="variant", values="rmse"))
        split["covariates_cost"] = (
            (split["default_cov"] - split["default"]) / split["default_cov"])
        split["tuning_gain"] = (split["default"] - split["tuned"]) / split["default"]
        split.to_csv(RESULTS / f"comparison_{operator}_effects.csv")
        print("\n  Where the improvement came from "
              "(covariates off, then hyperparameters):")
        print(split.sort_values("tuned").round(4).to_string())

        leader = summary["model"].iloc[0]
        pairs = dm[(dm["model_a"] == leader) | (dm["model_b"] == leader)]
        decisive = int(pairs["significant_fdr"].sum())
        print(f"\n  Leader: {leader}. Of {len(pairs)} comparisons against it, "
              f"{decisive} survive BH correction and {len(pairs) - decisive} are ties.")
        print(pairs[["model_a", "model_b", "dm_stat", "p_value",
                     "significant_fdr", "winner"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\nWrote comparison_{gp,robi}_{perfold,summary,dm}.csv")


if __name__ == "__main__":
    main()
