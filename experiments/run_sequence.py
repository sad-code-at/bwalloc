"""Do the sequence models actually win? The original comparison could not say.

The original study reported CNN 9.69 against XGBoost 13.91 on GP and concluded the
sequence architectures were better. They were never compared on equal terms: the
sequence models trained on a ``train_size=0.7`` split (~600 rows) while the tree models
went through the ``lag_336`` -> ``dropna()`` -> date-split path that left 88 training
rows. This script removes that confound.

Two arms, both on one rolling-origin fold schedule:

* **Architecture at equal information.** Every model -- CNN, LSTM, GRU, RNN, ridge,
  random forest, XGBoost -- reads the same univariate window of 24 consecutive sample
  lags, which is the input the original sequence models were given
  (``input_shape=(lookback, 1)``). Any difference here is architecture.
* **Lag depth against feature set.** The univariate window carries 24 consecutive lags
  where the corrected design carries four, which is a confound rather than a fair
  contrast. A ladder of configurations on one common row index -- so the folds cannot
  differ -- separates the two, and the control that settles it is the full corrected
  design *plus* the same 24 lags.

    python experiments/run_sequence.py
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
from bwalloc.features import FeatureConfig, assert_no_leakage, build_features  # noqa: E402
from bwalloc.models import default_point_models  # noqa: E402
from bwalloc.sequence import sequence_feature_config, sequence_models  # noqa: E402
from bwalloc.splits import rolling_origin  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
N_FOLDS = 8
LOOKBACK = 24
EPOCHS = 30

#: What the original reported, for the side-by-side. Reproduced by re-running its
#: notebooks as written; see docs/WALKTHROUGH.md.
ORIGINAL = {
    "gp": {"cnn": 9.69, "rnn": 10.06, "lstm": 10.67, "gru": 12.14,
           "xgboost": 13.91, "random_forest": 14.96},
    "robi": {"cnn": 20.63, "rnn": 21.6, "lstm": 25.0, "gru": 27.5,
             "xgboost": 22.72, "random_forest": 30.10},
}

#: Separates lag depth from feature set. The corrected design carries four lags
#: (1.5, 3, 4.5 and 24 wall-clock hours); the sequence window carries 24 consecutive
#: ones. ``full_plus_lags1_24`` is the control that tells the two apart.
DEPTH_CONFIGS: dict[str, FeatureConfig] = {
    "full_corrected": FeatureConfig(),
    "full_plus_lags1_24": FeatureConfig(lag_samples=tuple(range(1, 25))),
    "lags1_24_only": FeatureConfig(
        lag_hours=(), rolling_hours=(), lag_samples=tuple(range(1, 25)),
        daily_harmonics=0, weekly_harmonics=0, use_context=False),
    "lags1_12_only": FeatureConfig(
        lag_hours=(), rolling_hours=(), lag_samples=tuple(range(1, 13)),
        daily_harmonics=0, weekly_harmonics=0, use_context=False),
    "lags1_8_only": FeatureConfig(
        lag_hours=(), rolling_hours=(), lag_samples=tuple(range(1, 9)),
        daily_harmonics=0, weekly_harmonics=0, use_context=False),
}


def main() -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    all_rows: list[pd.DataFrame] = []

    for operator in ("gp", "robi"):
        df = load(operator)
        profile = sampling_profile(df)

        seq_config = sequence_feature_config(LOOKBACK)
        assert_no_leakage(df, profile, seq_config)

        X, y = build_features(df, profile, seq_config)
        folds = rolling_origin(len(y), n_folds=N_FOLDS)
        baselines = standard_baselines(y.to_numpy(), profile.daily_period)
        baselines.append(SeasonalNaive(y.to_numpy(), period=24))

        models = sequence_models(lookback=LOOKBACK, epochs=EPOCHS)
        models += default_point_models()

        started = time.time()
        per_fold, predictions = run_backtest(
            X, y, folds,
            models=models,
            baselines=baselines,
            season_lag=profile.daily_period,
        )
        elapsed = time.time() - started
        summary = beats_baseline(summarise(per_fold))

        print("=" * 84)
        print(f"{operator.upper()}  —  univariate window of {LOOKBACK} lags, "
              f"{N_FOLDS} rolling-origin folds  ({elapsed:.0f}s)")
        print("=" * 84)
        view = summary[["model", "rmse_mean", "rmse_std", "mae_mean", "mase_mean",
                        "vs_persistence", "beats_persistence"]].copy()
        view["original"] = view["model"].map(ORIGINAL[operator])
        print(view.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

        per_fold.to_csv(RESULTS / f"sequence_{operator}_perfold.csv", index=False)
        summary.to_csv(RESULTS / f"sequence_{operator}_summary.csv", index=False)

        # Significance: is any sequence model distinguishable from the best tree?
        dm = dm_matrix(predictions, horizon=1)
        dm.to_csv(RESULTS / f"sequence_{operator}_dm.csv", index=False)
        interesting = ["cnn", "lstm", "gru", "rnn",
                       "random_forest", "xgboost", "ridge", "persistence"]
        shown = dm[dm["model_a"].isin(interesting) & dm["model_b"].isin(interesting)]
        best_tree = next(m for m in summary["model"]
                         if m in ("random_forest", "xgboost", "ridge"))
        pairs = shown[(shown["model_a"] == best_tree) | (shown["model_b"] == best_tree)]
        print(f"\n  Diebold-Mariano against {best_tree} (BH-corrected):")
        print(pairs[["model_a", "model_b", "dm_stat", "p_value",
                     "significant_fdr", "winner"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

        # -- Is the univariate win about lag depth, or about dropping features? -
        #
        # The univariate window carries 24 consecutive lags where the corrected
        # design carries four. That is a confound, and the answer matters: if the
        # gain is lag depth then the corrected design was simply under-lagged and
        # the sequence models get no architectural credit for it. Every config is
        # scored on one common row index so the fold schedule cannot differ.
        built = {name: build_features(df, profile, cfg)
                 for name, cfg in DEPTH_CONFIGS.items()}
        common = None
        for Xc, _ in built.values():
            common = Xc.index if common is None else common.intersection(Xc.index)
        folds_c = rolling_origin(len(common), n_folds=N_FOLDS)

        depth_rows = []
        for name, (Xc, yc) in built.items():
            per_c, _ = run_backtest(
                Xc.loc[common], yc.loc[common], folds_c,
                models=default_point_models(), baselines=[],
                season_lag=profile.daily_period, keep_predictions=False,
            )
            s = summarise(per_c)[["model", "rmse_mean"]]
            s["config"] = name
            depth_rows.append(s)
        depth = pd.concat(depth_rows, ignore_index=True)
        depth["operator"] = operator

        print(f"\n  Lag depth against feature set, {len(common)} common rows, "
              "identical folds:")
        print(depth.pivot(index="config", columns="model", values="rmse_mean")
              .to_string(float_format=lambda v: f"{v:9.3f}"))
        all_rows.append(depth)

    pd.concat(all_rows, ignore_index=True).to_csv(
        RESULTS / "sequence_lag_depth.csv", index=False
    )
    print("\nWrote sequence_{gp,robi}_{perfold,summary,dm}.csv and "
          "sequence_lag_depth.csv")


if __name__ == "__main__":
    main()
