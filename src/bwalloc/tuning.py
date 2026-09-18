"""Hyperparameter search that never sees a test observation.

Every hyperparameter in this project before this module was a default. Thirty epochs,
batch 32 and lr 1e-3 came from the original study's notebooks; ``n_estimators=300,
max_depth=12`` was picked by hand. A ranking of untuned models ranks whose defaults
happen to suit the data, which is a different question from which model is better.

The protocol
------------
Nested search inside all eight evaluation folds is the gold standard and is
unaffordable here -- roughly fourteen models times two operators times eight folds times
thirty trials. What is affordable and still leak-free:

1. Take the **development prefix**: everything before the first evaluation fold's test
   block. That boundary is a *timestamp*, computed once from the reference design
   matrix, so it does not move when a trial changes ``lookback`` and rebuilds the
   features with a different warm-up.
2. Run an inner :func:`bwalloc.splits.rolling_origin` schedule **inside that prefix
   only**, so model selection is itself rolling-origin rather than a single split.
3. Random search over the space, with a fixed seed. Random rather than grid on Bergstra
   & Benito's argument that most hyperparameters do not matter, so at equal budget
   random search covers the ones that do far better than a grid does.
4. Freeze the winner per (model, operator) and score it on the untouched evaluation
   folds.

No tuning decision ever sees a test observation, and
``test_tuning_prefix_ends_before_the_first_test_block`` pins that.

The feature set is in the search space
--------------------------------------
``lookback``, ``covariates`` and ``context`` sit alongside learning rate and tree depth.
That is deliberate: this project's own finding is that lag depth mattered more than
architecture, so leaving depth fixed at 24 while tuning everything else would bake in
exactly the assumption under test. It also means "do the covariates earn anything?" is
settled by measurement -- a model that is better without them will be tuned to drop
them, in a table, rather than by anyone's judgement.

Optuna's TPE would be the obvious alternative and is deliberately not used: a
thirty-trial seeded random search needs no dependency, and adding one would break the
"clone it on Kaggle and run, no installs" property ``docs/KAGGLE.md`` promises.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import SamplingProfile
from .features import FeatureConfig, build_features
from .splits import rolling_origin

#: Fraction of the series before the first evaluation test block. Must match
#: ``splits.rolling_origin``'s ``min_train`` default, or the prefix would overlap the
#: first test fold and the whole protocol would be void.
PREFIX_FRACTION = 0.40

#: Inner folds used for selection, inside the development prefix.
INNER_FOLDS = 3

#: Trials per (model, operator).
N_TRIALS = 30


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #

#: Shared by every model, because the feature set is a hyperparameter here.
FEATURE_SPACE: dict[str, dict] = {
    "lookback": {"type": "choice", "values": [8, 12, 24, 36, 48]},
    "covariates": {"type": "choice", "values": [False, True]},
    "context": {"type": "choice", "values": ["all", "rain_only", "none"]},
}

_DEEP_COMMON: dict[str, dict] = {
    "epochs": {"type": "choice", "values": [20, 40, 60, 90, 120]},
    "lr": {"type": "loguniform", "low": 3e-4, "high": 1e-2},
    "batch_size": {"type": "choice", "values": [16, 32, 64]},
    "dropout": {"type": "choice", "values": [0.0, 0.1, 0.2]},
}

SEARCH_SPACES: dict[str, dict] = {
    "ridge": {"alpha": {"type": "loguniform", "low": 1e-3, "high": 1e3}},
    "random_forest": {
        "n_estimators": {"type": "choice", "values": [200, 300, 500, 800]},
        "max_depth": {"type": "choice", "values": [4, 8, 12, 20, None]},
        "min_samples_leaf": {"type": "choice", "values": [1, 2, 4, 8]},
        "max_features": {"type": "choice", "values": [0.3, 0.5, 0.8, 1.0]},
    },
    "xgboost": {
        "n_estimators": {"type": "choice", "values": [200, 400, 800]},
        "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
        "max_depth": {"type": "choice", "values": [2, 3, 4, 6, 8]},
        "subsample": {"type": "choice", "values": [0.6, 0.8, 0.9, 1.0]},
        "colsample_bytree": {"type": "choice", "values": [0.4, 0.6, 0.9, 1.0]},
        "min_child_weight": {"type": "choice", "values": [1, 3, 6]},
    },
    # The senior's four, now allowed to see covariates.
    "cnn_cov": dict(_DEEP_COMMON),
    "lstm_cov": dict(_DEEP_COMMON),
    "gru_cov": dict(_DEEP_COMMON),
    "rnn_cov": dict(_DEEP_COMMON),
    # The modern architectures.
    "dlinear": {**_DEEP_COMMON,
                "kernel": {"type": "choice", "values": [5, 13, 25]},
                "weight_decay": {"type": "choice", "values": [0.0, 1e-4, 1e-3]}},
    "nlinear": {**_DEEP_COMMON,
                "weight_decay": {"type": "choice", "values": [0.0, 1e-4, 1e-3]}},
    "tcn": {**_DEEP_COMMON,
            "channels": {"type": "choice", "values": [16, 32, 64]},
            "levels": {"type": "choice", "values": [2, 3, 4]},
            "kernel": {"type": "choice", "values": [2, 3, 5]}},
    "transformer": {**_DEEP_COMMON,
                    "d_model": {"type": "choice", "values": [16, 32, 64]},
                    "n_heads": {"type": "choice", "values": [1, 2, 4]},
                    "n_layers": {"type": "choice", "values": [1, 2, 3]}},
    "nbeats": {**_DEEP_COMMON,
               "width": {"type": "choice", "values": [32, 64, 128]},
               "n_blocks": {"type": "choice", "values": [2, 3, 4]},
               "block_layers": {"type": "choice", "values": [2, 3, 4]}},
    "nbeatsx": {**_DEEP_COMMON,
                "width": {"type": "choice", "values": [32, 64, 128]},
                "n_blocks": {"type": "choice", "values": [2, 3, 4]},
                "block_layers": {"type": "choice", "values": [2, 3, 4]},
                "exog_dim": {"type": "choice", "values": [8, 16, 32]}},
}

#: Models whose architecture is univariate by definition; searching the covariate switch
#: for them would waste half the budget exploring a dimension they cannot use.
UNIVARIATE_MODELS = ("nbeats",)


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    """Draw one configuration from a search space."""
    out = {}
    for key, spec in space.items():
        kind = spec["type"]
        if kind == "choice":
            out[key] = spec["values"][int(rng.integers(len(spec["values"])))]
        elif kind == "int":
            out[key] = int(rng.integers(spec["low"], spec["high"] + 1))
        elif kind == "uniform":
            out[key] = float(rng.uniform(spec["low"], spec["high"]))
        elif kind == "loguniform":
            lo, hi = math.log(spec["low"]), math.log(spec["high"])
            out[key] = float(math.exp(rng.uniform(lo, hi)))
        else:
            raise ValueError(f"Unknown space type {kind!r} for {key!r}.")
    return out


# --------------------------------------------------------------------------- #
# Turning a sampled configuration into features and a model
# --------------------------------------------------------------------------- #

def feature_config_for(params: dict) -> FeatureConfig:
    """The design matrix a trial asked for."""
    lookback = int(params["lookback"])
    lags = tuple(range(1, lookback + 1))
    context = params.get("context", "all")
    if context == "none":
        flags, use_context = None, False
    elif context == "rain_only":
        # The one flag the audit found a real effect for: variance ratio 1.330,
        # Levene p < 0.001 on GP. On Robi it is absent, and FeatureConfig drops
        # flags that are not in the frame, so this degenerates to "none" there.
        flags, use_context = ("is_rain",), True
    else:
        flags, use_context = None, True
    return FeatureConfig(
        lag_samples=lags,
        covariate_lag_samples=lags if params.get("covariates") else (),
        context_flags=flags,
        use_context=use_context,
    )


def build_model(name: str, params: dict):
    """Instantiate one candidate. Feature keys are stripped before construction."""
    from . import architectures as A
    from . import models as M
    from .sequence import CovariateSequenceForecaster

    p = {k: v for k, v in params.items() if k not in FEATURE_SPACE}
    lookback = int(params["lookback"])

    # ``.get`` throughout: a DEFAULTS entry carries only the feature keys, so a model
    # hyperparameter it omits must fall back to the library default rather than raise.
    if name == "ridge":
        return M.ridge(alpha=p.get("alpha", 1.0))
    if name == "random_forest":
        return M.random_forest(**p)
    if name == "xgboost":
        return M.xgboost_point(**p)
    if name.endswith("_cov"):
        return CovariateSequenceForecaster(kind=name[:-4], lookback=lookback, **p)

    classes = {
        "dlinear": A.DLinear, "nlinear": A.NLinear, "tcn": A.TCN,
        "transformer": A.TransformerForecaster,
        "nbeats": A.NBeats, "nbeatsx": A.NBeatsX,
    }
    if name not in classes:
        raise ValueError(f"No builder for model {name!r}.")
    return classes[name](lookback=lookback, **p)


# --------------------------------------------------------------------------- #
# The development prefix
# --------------------------------------------------------------------------- #

def development_cutoff(reference_index: pd.DatetimeIndex, n_folds: int = 8,
                       prefix_fraction: float = PREFIX_FRACTION) -> pd.Timestamp:
    """The timestamp at which the first evaluation test block begins.

    Everything strictly before this is fair game for model selection; everything from
    here on is test data and must not influence a single hyperparameter. Returned as a
    timestamp rather than a row count precisely because a trial may change ``lookback``
    and therefore how many warm-up rows its design matrix drops -- a row count would
    silently slide across the boundary, a timestamp cannot.
    """
    n = len(reference_index)
    min_train = max(30, int(prefix_fraction * n))
    if min_train >= n:
        raise ValueError(f"Prefix of {min_train} leaves no evaluation data (n={n}).")
    return pd.Timestamp(reference_index[min_train])


def development_slice(X: pd.DataFrame, y: pd.Series, cutoff: pd.Timestamp):
    """Rows strictly before the cutoff."""
    mask = pd.DatetimeIndex(X.index) < cutoff
    return X.loc[mask], y.loc[mask]


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #

def _score(model, X, y, folds) -> float:
    """Mean RMSE across the inner folds, refitting on each."""
    errors = []
    for fold in folds:
        fit_idx = (
            np.concatenate([fold.train, fold.calib]) if fold.n_calib else fold.train
        )
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        residual = y.iloc[fold.test].to_numpy() - model.predict(X.iloc[fold.test])
        errors.append(float(np.sqrt(np.mean(residual ** 2))))
    return float(np.mean(errors))


def random_search(
    name: str,
    df: pd.DataFrame,
    profile: SamplingProfile,
    cutoff: pd.Timestamp,
    n_trials: int = N_TRIALS,
    inner_folds: int = INNER_FOLDS,
    seed: int = 42,
) -> pd.DataFrame:
    """Search one model's space on the development prefix.

    Returns every trial, ranked, so the search itself is inspectable -- a model whose
    best and median trial are the same distance apart as two different models were not
    really distinguished by the search, and that is worth being able to see.
    """
    rng = np.random.default_rng(seed)
    space = dict(FEATURE_SPACE)
    if name in UNIVARIATE_MODELS:
        space = {k: v for k, v in space.items() if k != "covariates"}
    space.update(SEARCH_SPACES[name])

    rows = []
    for trial in range(n_trials):
        params = sample_params(space, rng)
        params.setdefault("covariates", False)
        try:
            X, y = build_features(df, profile, feature_config_for(params))
            Xd, yd = development_slice(X, y, cutoff)
            folds = rolling_origin(len(yd), n_folds=inner_folds)
            model = build_model(name, params)
            rmse = _score(model, Xd, yd, folds)
            failure = ""
        except Exception as exc:                      # noqa: BLE001
            # A sampled configuration can be genuinely impossible -- lookback 48 with
            # kernel 5 and four dilation levels outruns the window. Recording the
            # failure beats silently narrowing the space, because a space that mostly
            # fails is itself a finding about the search.
            rmse, failure = float("nan"), f"{type(exc).__name__}: {exc}"[:200]
        rows.append({
            "model": name, "trial": trial, "rmse": rmse,
            "params": json.dumps(params, default=str), "failure": failure,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("rmse", na_position="last").reset_index(drop=True)


def best_params(trials: pd.DataFrame) -> dict:
    """The winning configuration from a trials table."""
    valid = trials.dropna(subset=["rmse"])
    if valid.empty:
        raise ValueError("Every trial failed; nothing to select.")
    return json.loads(valid.iloc[0]["params"])


# --------------------------------------------------------------------------- #
# Using the result
# --------------------------------------------------------------------------- #

def load_best(path, operator: str) -> dict[str, dict]:
    """Read ``tuning_best.csv`` into ``{model_name: params}`` for one operator."""
    table = pd.read_csv(path)
    rows = table[table["operator"] == operator]
    return {r["model"]: json.loads(r["params"]) for _, r in rows.iterrows()}


def tuned_models(results_dir, operator: str, warn: bool = True):
    """Every tuned model for an operator, as ``[(FeatureConfig, Forecaster), ...]``.

    Falls back to an empty list with a warning when the tuning results are absent, so
    that nothing already in the repository breaks by importing this module -- the
    existing experiment scripts keep their defaults until they are explicitly pointed
    at a tuned configuration.
    """
    from pathlib import Path
    import warnings as _warnings

    path = Path(results_dir) / "tuning_best.csv"
    if not path.exists():
        if warn:
            _warnings.warn(
                f"{path} not found; run experiments/run_tuning.py first. "
                "Falling back to default hyperparameters.",
                stacklevel=2,
            )
        return []
    out = []
    for name, params in load_best(path, operator).items():
        out.append((feature_config_for(params), build_model(name, params)))
    return out
