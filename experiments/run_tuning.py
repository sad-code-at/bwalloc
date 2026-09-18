"""Tune every model, then ask what tuning was worth.

Until now every hyperparameter in this project was a default: 30 epochs, batch 32 and
lr 1e-3 inherited from the original study's notebooks, ``n_estimators=300,
max_depth=12`` picked by hand. So every ranking the project has produced has been a
ranking of models *at their defaults*, which is a weaker claim than it looks.

Selection happens entirely inside the **development prefix** -- everything before the
first evaluation fold's test block -- on its own inner rolling-origin schedule. The
winning configuration is then frozen and scored on the eight untouched evaluation folds,
against the same model at its defaults. No tuning decision sees a test observation; see
``bwalloc.tuning`` for why the boundary is a timestamp rather than a row count.

The feature set is in the search space alongside learning rate and tree depth --
``lookback``, ``covariates`` and ``context``. This project's own finding is that lag
depth mattered more than architecture, so holding depth fixed while tuning everything
else would bake in the assumption under test.

Long-running (hours). Results are checkpointed per (model, operator), so an interrupted
run resumes where it stopped rather than starting over.

    python experiments/run_tuning.py            # everything
    python experiments/run_tuning.py gp tcn     # one operator, one model
"""

from __future__ import annotations

import json
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
from bwalloc.data import load, sampling_profile  # noqa: E402
from bwalloc.evaluate import run_backtest, summarise  # noqa: E402
from bwalloc.features import build_features  # noqa: E402
from bwalloc.sequence import full_feature_config  # noqa: E402
from bwalloc.splits import rolling_origin  # noqa: E402
from bwalloc.stats import diebold_mariano  # noqa: E402
from bwalloc.tuning import (N_TRIALS, SEARCH_SPACES, best_params, build_model,  # noqa: E402
                            development_cutoff, feature_config_for, random_search)

RESULTS = ROOT / "experiments" / "results"
TRIALS_DIR = RESULTS / "tuning_trials"
N_FOLDS = 8
LOOKBACK = 24

#: What each model looks like untuned, so "what did tuning buy?" has a comparator.
DEFAULTS: dict[str, dict] = {
    "ridge": {"lookback": 24, "covariates": False, "context": "all"},
    "random_forest": {"lookback": 24, "covariates": False, "context": "all"},
    "xgboost": {"lookback": 24, "covariates": False, "context": "all"},
    "cnn_cov": {"lookback": 24, "covariates": True, "context": "all", "epochs": 30},
    "lstm_cov": {"lookback": 24, "covariates": True, "context": "all", "epochs": 30},
    "gru_cov": {"lookback": 24, "covariates": True, "context": "all", "epochs": 30},
    "rnn_cov": {"lookback": 24, "covariates": True, "context": "all", "epochs": 30},
    "dlinear": {"lookback": 24, "covariates": True, "context": "all"},
    "nlinear": {"lookback": 24, "covariates": True, "context": "all"},
    "tcn": {"lookback": 24, "covariates": True, "context": "all"},
    "transformer": {"lookback": 24, "covariates": True, "context": "all"},
    "nbeats": {"lookback": 24, "covariates": False, "context": "all"},
    "nbeatsx": {"lookback": 24, "covariates": True, "context": "all"},
}


def evaluate(name: str, params: dict, df, profile, n_folds: int = N_FOLDS):
    """Score one configuration on the untouched evaluation folds."""
    X, y = build_features(df, profile, feature_config_for(params))
    folds = rolling_origin(len(y), n_folds=n_folds)
    per_fold, predictions = run_backtest(
        X, y, folds, models=[build_model(name, params)], baselines=[],
        season_lag=profile.daily_period,
    )
    summary = summarise(per_fold)
    return float(summary["rmse_mean"].iloc[0]), predictions


def main(operators=("gp", "robi"), only: str | None = None) -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    best_rows, gain_rows = [], []

    for operator in operators:
        df = load(operator)
        profile = sampling_profile(df)

        # The boundary, computed once from the reference design matrix and then held
        # fixed in wall-clock time no matter what lookback a trial asks for.
        X_ref, _ = build_features(df, profile, full_feature_config(LOOKBACK, True))
        cutoff = development_cutoff(pd.DatetimeIndex(X_ref.index), n_folds=N_FOLDS)
        folds_ref = rolling_origin(len(X_ref), n_folds=N_FOLDS)
        first_test = pd.Timestamp(X_ref.index[folds_ref[0].test[0]])
        assert cutoff <= first_test, "development prefix overlaps the first test block"

        print("=" * 96)
        print(f"{operator.upper()}  —  development prefix ends {cutoff}, "
              f"first evaluation test block starts {first_test}")
        print("=" * 96)

        names = [only] if only else list(SEARCH_SPACES)
        for name in names:
            path = TRIALS_DIR / f"{operator}_{name}.csv"
            if path.exists():
                trials = pd.read_csv(path)
                print(f"  {name:14} cached ({len(trials)} trials)")
            else:
                started = time.time()
                trials = random_search(name, df, profile, cutoff, n_trials=N_TRIALS)
                trials["operator"] = operator
                trials.to_csv(path, index=False)
                ok = int(trials["rmse"].notna().sum())
                print(f"  {name:14} {N_TRIALS} trials, {ok} valid, "
                      f"best inner rmse={trials['rmse'].min():.3f}  "
                      f"({time.time() - started:.0f}s)")

            try:
                winner = best_params(trials)
            except ValueError:
                print(f"  {name:14} every trial failed — skipping")
                continue

            tuned_rmse, tuned_pred = evaluate(name, winner, df, profile)
            default_rmse, default_pred = evaluate(name, DEFAULTS[name], df, profile)

            # Is the difference real, or fold noise? Same observations, so the
            # Diebold-Mariano test applies directly.
            common = (
                tuned_pred.set_index("timestamp").index
                .intersection(default_pred.set_index("timestamp").index)
            )
            a = tuned_pred.set_index("timestamp").loc[common]
            b = default_pred.set_index("timestamp").loc[common]
            stat, p = diebold_mariano(
                a["y_true"], a["y_pred"], b["y_pred"], horizon=1
            )

            best_rows.append({
                "operator": operator, "model": name,
                "params": json.dumps(winner, default=str),
                "inner_rmse": float(trials["rmse"].min()),
                "eval_rmse": tuned_rmse,
            })
            gain_rows.append({
                "operator": operator, "model": name,
                "rmse_default": default_rmse, "rmse_tuned": tuned_rmse,
                "gain": (default_rmse - tuned_rmse) / default_rmse,
                "dm_stat": stat, "p_value": p,
                "lookback": winner["lookback"],
                "covariates": winner.get("covariates", False),
                "context": winner.get("context", "all"),
            })
            print(f"  {name:14} default {default_rmse:7.3f} -> tuned "
                  f"{tuned_rmse:7.3f}  ({(default_rmse - tuned_rmse) / default_rmse:+6.1%}"
                  f", DM p={p:.3f})  lookback={winner['lookback']} "
                  f"covariates={winner.get('covariates')} context={winner.get('context')}")

    if best_rows:
        pd.DataFrame(best_rows).to_csv(RESULTS / "tuning_best.csv", index=False)
    if gain_rows:
        gain = pd.DataFrame(gain_rows)
        gain.to_csv(RESULTS / "tuning_gain.csv", index=False)
        print("\nWhat tuning bought:")
        print(gain[["operator", "model", "rmse_default", "rmse_tuned", "gain",
                    "p_value", "lookback", "covariates", "context"]]
              .to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    # One combined trials table, so the search is inspectable without globbing.
    frames = [pd.read_csv(p) for p in sorted(TRIALS_DIR.glob("*.csv"))]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            RESULTS / "tuning_trials.csv", index=False
        )
    print("\nWrote tuning_best.csv, tuning_gain.csv, tuning_trials.csv "
          f"and per-model trials in {TRIALS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    ops = tuple(a for a in args if a in ("gp", "robi")) or ("gp", "robi")
    model = next((a for a in args if a in SEARCH_SPACES), None)
    main(operators=ops, only=model)
