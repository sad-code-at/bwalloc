"""Corrected forecasting benchmark.

Every model and baseline on one rolling-origin fold schedule, with naive baselines,
fold-level spreads, a feature ablation, and Diebold-Mariano tests under FDR control.

    python experiments/run_benchmark.py
"""

from __future__ import annotations

import sys
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
from bwalloc.splits import rolling_origin  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
N_FOLDS = 8

#: Feature ablation. Each entry isolates one design decision so the corrected
#: protocol's gains can be attributed rather than asserted.
ABLATIONS: dict[str, FeatureConfig] = {
    "full": FeatureConfig(),
    "no_context": FeatureConfig(use_context=False),
    "no_fourier": FeatureConfig(daily_harmonics=0, weekly_harmonics=0),
    "lags_only": FeatureConfig(daily_harmonics=0, weekly_harmonics=0, use_context=False),
    # Faithful reproduction of the original feature design: lags of 1, 2, 3 and 24
    # *samples* (believing 24 meant a day), rolling windows of 3 and 24 samples,
    # calendar integers, context flags, no Fourier terms. The leak is deliberately
    # not reproduced, so this isolates the cost of the sampling-rate error alone.
    "original_style": FeatureConfig(
        lag_hours=(), rolling_hours=(),
        lag_samples=(1, 2, 3, 24), rolling_samples=(3, 24),
        daily_harmonics=0, weekly_harmonics=0,
        calendar_ints=True, use_context=True,
    ),
    # The same design with only the daily lag corrected -- 24 samples replaced by the
    # measured period (17 on GP, 15 on Robi), everything else untouched. Isolates the
    # cost of the sampling-rate error from every other change.
    "original_correct_period": FeatureConfig(
        lag_hours=(24.0,), rolling_hours=(24.0,),
        lag_samples=(1, 2, 3), rolling_samples=(3,),
        daily_harmonics=0, weekly_harmonics=0,
        calendar_ints=True, use_context=True,
    ),
}


def main() -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    for operator in ("gp", "robi"):
        df = load(operator)
        profile = sampling_profile(df)

        # Gate: refuse to benchmark anything if the leak guard does not pass.
        assert_no_leakage(df, profile, FeatureConfig())

        X, y = build_features(df, profile, FeatureConfig())
        folds = rolling_origin(len(y), n_folds=N_FOLDS)
        baselines = standard_baselines(y.to_numpy(), profile.daily_period)
        baselines.append(SeasonalNaive(y.to_numpy(), period=24))  # the original's lag

        per_fold, predictions = run_backtest(
            X, y, folds,
            models=default_point_models(),
            baselines=baselines,
            season_lag=profile.daily_period,
        )
        summary = beats_baseline(summarise(per_fold))

        print("=" * 78)
        print(f"{operator.upper()}  —  corrected benchmark, {N_FOLDS} rolling-origin folds")
        print("=" * 78)
        print(
            summary[
                ["model", "rmse_mean", "rmse_std", "mae_mean", "mase_mean",
                 "vs_persistence", "beats_persistence"]
            ].to_string(index=False, float_format=lambda v: f"{v:9.3f}")
        )

        # Gate: the top-ranked model must actually beat the naive baseline.
        best = summary.iloc[0]
        if not bool(best["beats_persistence"]):
            raise SystemExit(
                f"{operator}: best model {best['model']!r} does not beat persistence. "
                "Do not report this as a result."
            )

        per_fold.to_csv(RESULTS / f"benchmark_{operator}_perfold.csv", index=False)
        summary.to_csv(RESULTS / f"benchmark_{operator}_summary.csv", index=False)
        predictions.to_csv(RESULTS / f"benchmark_{operator}_predictions.csv", index=False)

        # -- Significance ------------------------------------------------------
        dm = dm_matrix(predictions, horizon=1)
        dm.to_csv(RESULTS / f"benchmark_{operator}_dm.csv", index=False)
        top = summary["model"].head(4).tolist()
        shown = dm[dm["model_a"].isin(top) & dm["model_b"].isin(top)]
        print("\n  Diebold-Mariano among the top models (BH-corrected):")
        print(shown[["model_a", "model_b", "dm_stat", "p_value", "significant_fdr", "winner"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

        # -- Ablation ----------------------------------------------------------
        rows = []
        for name, config in ABLATIONS.items():
            try:
                Xa, ya = build_features(df, profile, config)
            except ValueError:
                continue
            folds_a = rolling_origin(len(ya), n_folds=N_FOLDS)
            per_a, _ = run_backtest(
                Xa, ya, folds_a,
                models=default_point_models(),
                baselines=[],
                season_lag=profile.daily_period,
                keep_predictions=False,
            )
            agg = summarise(per_a)
            rows.append(
                {
                    "ablation": name,
                    "n_features": Xa.shape[1],
                    "best_model": agg.iloc[0]["model"],
                    "rmse_mean": agg.iloc[0]["rmse_mean"],
                    "rmse_std": agg.iloc[0]["rmse_std"],
                }
            )
        ablation = pd.DataFrame(rows).sort_values("rmse_mean")
        ablation.insert(0, "operator", operator)
        ablation.to_csv(RESULTS / f"ablation_{operator}.csv", index=False)
        print("\n  Feature ablation (best model within each configuration):")
        print(ablation.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
        print()

    print(f"Wrote benchmark tables to {RESULTS}")


if __name__ == "__main__":
    main()
