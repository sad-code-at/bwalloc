"""TCN, Transformer, DLinear/NLinear and N-BEATS against the incumbents.

Notebook 08 left a specific question open. It found that **lag depth mattered more than
architecture** -- four lags to twenty-four bought 18% RMSE on GP for any model, while the
best architecture bought 9% on top of that, on one operator only. If *which* past steps
matter is the live question, then the architectures that address it directly had not been
tried: dilated convolutions, attention over lag positions, and basis expansion.

Every model here reads the same design matrix: the full corrected feature set, demand
lags 1-24, and each context flag and daily harmonic at the same 24 lags. Three of these
architectures are univariate as their papers define them; implemented literally they
would have quietly reproduced the very limitation this script exists to remove, so
covariates enter as channels and `NBeats` is retained, labelled, as the univariate
control.

Three tables come out of this beyond the usual accuracy summary:

* ``arch_params.csv`` -- parameter count per model, for "is complexity buying anything?"
* ``arch_attention_*.csv`` -- what the Transformer's last position attends to, by lag,
  checkable against the measured daily period (17 samples on GP, 15 on Robi)
* ``arch_channel_importance_*.csv`` -- permutation importance per input channel, which
  is the direct measurement of whether the covariates are earning their place

    python experiments/run_architectures.py
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
from bwalloc.architectures import (TransformerForecaster, channel_permutation_importance,  # noqa: E402
                                   modern_models)
from bwalloc.baselines import SeasonalNaive, standard_baselines  # noqa: E402
from bwalloc.data import load, sampling_profile  # noqa: E402
from bwalloc.evaluate import beats_baseline, dm_matrix, run_backtest, summarise  # noqa: E402
from bwalloc.features import assert_no_leakage, build_features  # noqa: E402
from bwalloc.models import default_point_models  # noqa: E402
from bwalloc.sequence import channel_window, covariate_sequence_models, full_feature_config  # noqa: E402
from bwalloc.splits import rolling_origin  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
N_FOLDS = 8
LOOKBACK = 24
EPOCHS = 60
#: Models worth asking "what are you actually reading?" of. Cheap enough to permute.
IMPORTANCE_MODELS = ("tcn", "transformer", "nbeatsx", "gru_cov")


def build_models():
    models = list(default_point_models())
    models += covariate_sequence_models(lookback=LOOKBACK, epochs=30)
    models += modern_models(lookback=LOOKBACK, epochs=EPOCHS)
    return models


def main() -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    param_rows: list[dict] = []

    for operator in ("gp", "robi"):
        df = load(operator)
        profile = sampling_profile(df)
        config = full_feature_config(LOOKBACK, covariates=True)
        assert_no_leakage(df, profile, config)

        X, y = build_features(df, profile, config)
        spec = channel_window(X, LOOKBACK)
        folds = rolling_origin(len(y), n_folds=N_FOLDS)
        baselines = standard_baselines(y.to_numpy(), profile.daily_period)
        baselines.append(SeasonalNaive(y.to_numpy(), period=24))

        models = build_models()

        print("=" * 96)
        print(f"{operator.upper()}  —  {len(X)} rows, {spec.n_channels} channels "
              f"x {LOOKBACK} lags + {len(spec.static)} static, {N_FOLDS} folds")
        print("=" * 96)

        started = time.time()
        per_fold, predictions = run_backtest(
            X, y, folds, models=models, baselines=baselines,
            season_lag=profile.daily_period,
        )
        summary = beats_baseline(summarise(per_fold))
        print(f"({time.time() - started:.0f}s)")
        print(summary[["model", "rmse_mean", "rmse_std", "mae_mean", "mase_mean",
                       "vs_persistence", "beats_persistence"]]
              .to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

        per_fold.to_csv(RESULTS / f"arch_{operator}_perfold.csv", index=False)
        summary.to_csv(RESULTS / f"arch_{operator}_summary.csv", index=False)

        dm = dm_matrix(predictions, horizon=1)
        dm.to_csv(RESULTS / f"arch_{operator}_dm.csv", index=False)
        best = summary["model"].iloc[0]
        pairs = dm[(dm["model_a"] == best) | (dm["model_b"] == best)]
        print(f"\n  Diebold-Mariano against the leader ({best}), BH-corrected:")
        print(pairs[["model_a", "model_b", "dm_stat", "p_value",
                     "significant_fdr", "winner"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

        # -- What is each model actually reading? ---------------------------- #
        #
        # Fitted once more on the last fold's training window, because permutation
        # importance needs a fitted model and run_backtest does not keep them.
        last = folds[-1]
        fit_idx = np.concatenate([last.train, last.calib])
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        X_test, y_test = X.iloc[last.test], y.iloc[last.test]

        importance_rows, attention_rows = [], []
        for model in build_models():
            if model.name not in IMPORTANCE_MODELS:
                continue
            model.fit(X_fit, y_fit)
            imp = channel_permutation_importance(model, X_test, y_test, spec)
            imp.insert(0, "model", model.name)
            importance_rows.append(imp)
            param_rows.append({
                "operator": operator, "model": model.name,
                "n_parameters": getattr(model, "n_parameters", 0),
            })
            if isinstance(model, TransformerForecaster):
                model.predict(X_test)      # refresh the stored attention map
                weights = model.attention_by_lag
                if weights is not None:
                    # Oldest first in the window, so position i is lag L-i.
                    attention_rows.append(pd.DataFrame({
                        "operator": operator,
                        "lag": list(range(LOOKBACK, 0, -1)),
                        "attention": weights,
                    }))

        if importance_rows:
            imp = pd.concat(importance_rows, ignore_index=True)
            imp.to_csv(RESULTS / f"arch_channel_importance_{operator}.csv", index=False)
            print(f"\n  Permutation importance by input channel ({operator.upper()}), "
                  "RMSE cost of shuffling each:")
            print(imp.pivot(index="channel", columns="model", values="importance")
                  .sort_values(imp["model"].iloc[0], ascending=False)
                  .to_string(float_format=lambda v: f"{v:8.3f}"))

        if attention_rows:
            att = pd.concat(attention_rows, ignore_index=True)
            att.to_csv(RESULTS / f"arch_attention_{operator}.csv", index=False)
            top = att.nlargest(5, "attention")
            print(f"\n  Transformer attends most to lags: "
                  f"{', '.join(str(int(v)) for v in top['lag'])}  "
                  f"(measured daily period = {profile.daily_period} samples)")

    pd.DataFrame(param_rows).to_csv(RESULTS / "arch_params.csv", index=False)
    print("\nWrote arch_{gp,robi}_{perfold,summary,dm}.csv, arch_params.csv, "
          "arch_attention_*.csv and arch_channel_importance_*.csv")


if __name__ == "__main__":
    main()
