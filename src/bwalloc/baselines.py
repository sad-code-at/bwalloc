"""Naive baselines that every model must beat to be worth reporting.

None of these existed in the original project, which is why six of its ten models
were presented as successes while losing to a one-line forecaster. Measured on the
original test windows:

===============  ========  ==========
baseline         GP RMSE   Robi RMSE
===============  ========  ==========
persistence         12.18       31.41
train mean          18.42       47.19
===============  ========  ==========

against which the reported LightGBM (16.00 / 31.10), Random Forest (14.96 / 30.10),
AR (16.27 / 38.15), ARIMA (16.39 / 36.54) and Holt-Winters (60.86 / 49.60) results
are all losses or ties.

Baselines read the raw target series by position rather than a design matrix, because
that is genuinely what they need; the fold indices keep them honest about not seeing
the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .splits import Fold


class Baseline:
    """Common interface: fit on a fold's training slice, predict its test slice."""

    name = "baseline"

    def __init__(self, y: pd.Series | np.ndarray):
        self.y = np.asarray(y, dtype=float)

    def predict(self, fold: Fold) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _fit_slice(self, fold: Fold) -> np.ndarray:
        """Everything the baseline is allowed to look at: train plus calibration."""
        end = fold.test[0]
        return self.y[:end]


class Persistence(Baseline):
    """Last observed value carried forward one step.

    The strongest of the naive baselines on both traces and the one most models fail
    to beat. Note this is a *one-step* forecast with the true previous value
    available -- the same favourable setting the original notebooks evaluated their
    models in.
    """

    name = "persistence"

    def predict(self, fold: Fold) -> np.ndarray:
        idx = np.asarray(fold.test)
        return self.y[idx - 1]


class SeasonalNaive(Baseline):
    """Value from one daily period earlier.

    ``period`` must come from :attr:`bwalloc.data.SamplingProfile.daily_period` (16 on
    GP, 15 on Robi). Instantiating this with 24 -- the value the original code used
    everywhere -- reproduces the original error and scores 30.4 RMSE on GP against
    18.5 at the correct period, because 24 samples spans 34.4 h and lands roughly
    anti-phase to the daily cycle.
    """

    name = "seasonal_naive"

    def __init__(self, y, period: int):
        super().__init__(y)
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = f"seasonal_naive_{period}"

    def predict(self, fold: Fold) -> np.ndarray:
        idx = np.asarray(fold.test)
        src = idx - self.period
        if (src < 0).any():
            raise ValueError("Seasonal lag reaches before the start of the series.")
        return self.y[src]


class TrainMean(Baseline):
    """Constant: the mean of everything observed before the test block."""

    name = "train_mean"

    def predict(self, fold: Fold) -> np.ndarray:
        return np.full(fold.n_test, float(np.mean(self._fit_slice(fold))))


class Drift(Baseline):
    """Last value plus the average per-step change observed so far."""

    name = "drift"

    def predict(self, fold: Fold) -> np.ndarray:
        history = self._fit_slice(fold)
        if len(history) < 2:
            return np.full(fold.n_test, history[-1])
        slope = (history[-1] - history[0]) / (len(history) - 1)
        steps = np.arange(1, fold.n_test + 1, dtype=float)
        return history[-1] + slope * steps


class RollingMean(Baseline):
    """Mean of the most recent ``window`` observations."""

    name = "rolling_mean"

    def __init__(self, y, window: int):
        super().__init__(y)
        self.window = window
        self.name = f"rolling_mean_{window}"

    def predict(self, fold: Fold) -> np.ndarray:
        # Recomputed at each step so the window always ends at t-1.
        idx = np.asarray(fold.test)
        return np.array([self.y[max(0, i - self.window):i].mean() for i in idx])


def standard_baselines(y: pd.Series | np.ndarray, daily_period: int) -> list[Baseline]:
    """The baseline suite reported in every results table."""
    return [
        Persistence(y),
        SeasonalNaive(y, period=daily_period),
        Drift(y),
        RollingMean(y, window=daily_period),
        TrainMean(y),
    ]
