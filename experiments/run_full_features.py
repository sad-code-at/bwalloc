"""Does a sequence model earn anything from the features it was never shown?

`sequence.py` reproduces the original study's architectures faithfully, and the original
read ``input_shape=(lookback, 1)`` -- a bare demand window, no Fourier terms, no
calendar, no context flags. That was a **reproduction constraint**, adopted so the
comparison in notebook 08 would isolate architecture from feature set. It was never a
modelling recommendation, and it has never been lifted.

Meanwhile the tree models in notebook 01 have the opposite problem: they see Fourier
terms and context flags but only four lags, and notebook 08 showed that costs 18% RMSE
on GP. **No model in this project has ever seen the full feature set at full lag depth.**

Five arms, scored on one common row index per operator so the fold schedule cannot
differ between them:

``lags_only``
    24 demand lags and nothing else -- notebook 08's setting, the control.
``full_sparse``
    ``FeatureConfig()`` defaults: four wall-clock lags, rolling statistics, Fourier
    terms, context flags -- notebook 01's setting.
``full_dense``
    The full defaults *plus* demand lags 1-24. Lag depth at full features.
``full_dense_covariates``
    Plus each context flag and daily harmonic at lags 1-24, so a sequence model sees
    their history rather than one contemporaneous value. **The headline arm.**
``dense_no_context``
    Lags 1-24 and Fourier terms with the context flags switched off -- the control that
    separates "more lags" from "context finally reached a model that could use it".

    python experiments/run_full_features.py
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
from bwalloc.baselines import standard_baselines  # noqa: E402
from bwalloc.data import load, sampling_profile  # noqa: E402
from bwalloc.evaluate import beats_baseline, dm_matrix, run_backtest, summarise  # noqa: E402
from bwalloc.features import FeatureConfig, assert_no_leakage, build_features  # noqa: E402
from bwalloc.models import default_point_models  # noqa: E402
from bwalloc.sequence import (covariate_sequence_models, full_feature_config,  # noqa: E402
                              sequence_feature_config, sequence_models)
from bwalloc.splits import rolling_origin  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
N_FOLDS = 8
LOOKBACK = 24
EPOCHS = 30

#: ``dense`` says whether the arm carries a consecutive demand window, and so whether
#: the univariate sequence models can read it at all; ``covariates`` says whether the
#: covariate channels are present for the covariate-aware variants.
ARMS: dict[str, tuple[FeatureConfig, bool, bool]] = {
    "lags_only": (sequence_feature_config(LOOKBACK), True, False),
    "full_sparse": (FeatureConfig(), False, False),
    "full_dense": (full_feature_config(LOOKBACK, covariates=False), True, False),
    "full_dense_covariates": (full_feature_config(LOOKBACK, covariates=True), True, True),
    "dense_no_context": (
        FeatureConfig(lag_samples=tuple(range(1, LOOKBACK + 1)), use_context=False),
        True, False,
    ),
}

HEADLINE = "full_dense_covariates"


def models_for(dense: bool, covariates: bool):
    """Which models can read this arm's design matrix."""
    models = list(default_point_models())
    if dense:
        models += sequence_models(lookback=LOOKBACK, epochs=EPOCHS)
    if covariates:
        models += covariate_sequence_models(lookback=LOOKBACK, epochs=EPOCHS)
    return models


def main() -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    matrix_rows: list[pd.DataFrame] = []

    for operator in ("gp", "robi"):
        df = load(operator)
        profile = sampling_profile(df)

        built = {}
        for arm, (config, dense, covariates) in ARMS.items():
            assert_no_leakage(df, profile, config)
            built[arm] = (build_features(df, profile, config), dense, covariates)

        # One row index shared by every arm. Arms drop different numbers of warm-up
        # rows (four wall-clock lags reach back 17 samples on GP, a 24-lag window
        # reaches back 24), and comparing them on different folds would confound the
        # feature set with the split -- the exact mistake the original study made.
        common = None
        for (Xc, _), _, _ in built.values():
            common = Xc.index if common is None else common.intersection(Xc.index)
        folds = rolling_origin(len(common), n_folds=N_FOLDS)

        print("=" * 96)
        print(f"{operator.upper()}  —  {len(common)} rows common to all "
              f"{len(ARMS)} arms, {N_FOLDS} rolling-origin folds")
        print("=" * 96)

        per_fold_all, headline_predictions = [], None
        for arm, ((Xc, yc), dense, covariates) in built.items():
            Xa, ya = Xc.loc[common], yc.loc[common]
            baselines = standard_baselines(ya.to_numpy(), profile.daily_period)

            started = time.time()
            per_fold, predictions = run_backtest(
                Xa, ya, folds,
                models=models_for(dense, covariates),
                baselines=baselines,
                season_lag=profile.daily_period,
            )
            per_fold["arm"] = arm
            per_fold_all.append(per_fold)
            if arm == HEADLINE:
                headline_predictions = predictions

            summary = beats_baseline(summarise(per_fold))
            print(f"\n{arm}  ({Xa.shape[1]} columns, {time.time() - started:.0f}s)")
            print(summary[["model", "rmse_mean", "rmse_std", "mae_mean",
                           "vs_persistence", "beats_persistence"]]
                  .to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

        per_fold = pd.concat(per_fold_all, ignore_index=True)
        per_fold.to_csv(RESULTS / f"features_{operator}_perfold.csv", index=False)

        summary = (
            per_fold.groupby(["arm", "model"], as_index=False)["rmse"]
            .agg(rmse_mean="mean", rmse_std="std")
        )
        summary["operator"] = operator
        summary.to_csv(RESULTS / f"features_{operator}_summary.csv", index=False)
        matrix_rows.append(summary)

        dm = dm_matrix(headline_predictions, horizon=1)
        dm.to_csv(RESULTS / f"features_{operator}_dm.csv", index=False)

        print(f"\n  Arm against model, RMSE ({operator.upper()}):")
        print(summary.pivot(index="model", columns="arm", values="rmse_mean")
              .to_string(float_format=lambda v: f"{v:9.3f}"))

    pd.concat(matrix_rows, ignore_index=True).to_csv(
        RESULTS / "features_arm_matrix.csv", index=False
    )
    print("\nWrote features_{gp,robi}_{perfold,summary,dm}.csv and "
          "features_arm_matrix.csv")


if __name__ == "__main__":
    main()
