"""Forecasters behind one interface.

Every model exposes ``fit(X, y)`` and ``predict(X)``; models that can express a
predictive distribution also expose ``predict_quantiles(X, taus)``. That second method
is what the allocation layer consumes -- a point forecast cannot state how much
headroom a 95%-confident allocation needs, and the whole cost argument in
:mod:`bwalloc.allocation` rests on quantiles rather than means.

Optional dependencies (LightGBM, torch, foundation models) are imported lazily so the
package remains usable when they are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_TAUS = (0.5, 0.8, 0.9, 0.95, 0.98)


class Forecaster:
    """Base interface."""

    name = "forecaster"
    supports_quantiles = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict_quantiles(self, X: pd.DataFrame, taus=DEFAULT_TAUS) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not model quantiles.")


# --------------------------------------------------------------------------- #
# Point forecasters
# --------------------------------------------------------------------------- #

class SklearnForecaster(Forecaster):
    """Adapter for any fitted-by-``fit``/``predict`` scikit-learn estimator."""

    def __init__(self, estimator, name: str):
        self.estimator = estimator
        self.name = name

    def fit(self, X, y):
        self.estimator.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)

    @property
    def feature_importances_(self):
        return getattr(self.estimator, "feature_importances_", None)


def ridge(alpha: float = 1.0, name: str = "ridge") -> SklearnForecaster:
    """Linear baseline. Included because on short traces it is often competitive,
    and a tree ensemble that cannot beat it is not earning its complexity."""
    return SklearnForecaster(make_pipeline(StandardScaler(), Ridge(alpha=alpha)), name)


def random_forest(n_estimators: int = 300, max_depth: int | None = 12,
                  seed: int = 42, name: str = "random_forest",
                  **kwargs) -> SklearnForecaster:
    """The benchmark's workhorse. ``**kwargs`` reaches the estimator directly, which is
    what lets ``bwalloc.tuning`` search ``min_samples_leaf`` and ``max_features``
    without this signature having to enumerate every knob -- the same arrangement
    :func:`xgboost_point` already uses."""
    params = dict(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1,
    )
    params.update(kwargs)
    return SklearnForecaster(RandomForestRegressor(**params), name)


def xgboost_point(seed: int = 42, name: str = "xgboost", **kwargs) -> SklearnForecaster:
    import xgboost as xgb

    params = dict(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=seed, n_jobs=-1,
    )
    params.update(kwargs)
    return SklearnForecaster(xgb.XGBRegressor(**params), name)


def lightgbm_point(seed: int = 42, name: str = "lightgbm", **kwargs) -> SklearnForecaster:
    from lightgbm import LGBMRegressor

    params = dict(
        n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=15,
        subsample=0.9, colsample_bytree=0.9,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    params.update(kwargs)
    return SklearnForecaster(LGBMRegressor(**params), name)


# --------------------------------------------------------------------------- #
# Quantile forecasters
# --------------------------------------------------------------------------- #

class QuantileGBM(Forecaster):
    """Gradient-boosted quantile regression: one booster per requested level.

    Fitting a separate model per tau is the straightforward approach and keeps each
    booster's objective exactly the pinball loss at its own level. The cost is that
    predicted quantiles can cross; :meth:`predict_quantiles` sorts them, which is the
    standard non-crossing repair and is monotone-safe.
    """

    supports_quantiles = True

    def __init__(self, taus=DEFAULT_TAUS, backend: str = "xgboost",
                 seed: int = 42, name: str | None = None, **kwargs):
        self.taus = tuple(sorted(taus))
        self.backend = backend
        self.seed = seed
        self.kwargs = kwargs
        self.name = name or f"quantile_{backend}"
        self.models: dict[float, object] = {}

    def _make(self, tau: float):
        if self.backend == "xgboost":
            import xgboost as xgb

            params = dict(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.9, colsample_bytree=0.9,
                objective="reg:quantileerror", quantile_alpha=tau,
                random_state=self.seed, n_jobs=-1,
            )
            params.update(self.kwargs)
            return xgb.XGBRegressor(**params)

        if self.backend == "lightgbm":
            from lightgbm import LGBMRegressor

            params = dict(
                n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=15,
                subsample=0.9, colsample_bytree=0.9,
                objective="quantile", alpha=tau,
                random_state=self.seed, n_jobs=-1, verbose=-1,
            )
            params.update(self.kwargs)
            return LGBMRegressor(**params)

        if self.backend == "sklearn":
            from sklearn.ensemble import GradientBoostingRegressor

            params = dict(
                n_estimators=300, learning_rate=0.05, max_depth=3,
                loss="quantile", alpha=tau, random_state=self.seed,
            )
            params.update(self.kwargs)
            return GradientBoostingRegressor(**params)

        raise ValueError(f"Unknown backend {self.backend!r}")

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.models = {}
        for tau in self.taus:
            self.models[tau] = self._make(tau).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Median forecast, or the lowest fitted quantile if 0.5 was not requested."""
        tau = 0.5 if 0.5 in self.models else self.taus[0]
        return np.asarray(self.models[tau].predict(X), dtype=float)

    def predict_quantiles(self, X, taus=None) -> pd.DataFrame:
        taus = tuple(sorted(taus)) if taus is not None else self.taus
        missing = [t for t in taus if t not in self.models]
        if missing:
            raise ValueError(f"Model was not fitted for tau in {missing}.")
        preds = np.column_stack([self.models[t].predict(X) for t in taus])
        # Repair quantile crossing by sorting each row.
        preds = np.sort(preds, axis=1)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(preds, columns=list(taus), index=index)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def default_point_models(seed: int = 42) -> list[Forecaster]:
    """The point forecasters compared in the corrected benchmark.

    LightGBM is included only if installed, so the benchmark still runs locally
    without it and picks it up automatically on Colab.
    """
    models: list[Forecaster] = [
        ridge(),
        random_forest(seed=seed),
        xgboost_point(seed=seed),
    ]
    try:
        import lightgbm  # noqa: F401

        models.append(lightgbm_point(seed=seed))
    except ImportError:
        pass
    return models
