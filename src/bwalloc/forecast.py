"""Multi-horizon forecasting.

Everything else in this repository -- and every notebook in the project it extends --
predicts ``y_t`` with ``y_{t-1}`` already in hand. That is a one-step-ahead nowcast,
and it is not something a network can provision from: by the time the previous
measurement has arrived, the capacity decision for the next interval has already had
to be made. An allocator needs *lead time*.

This module supplies it, and the honest consequence is that accuracy degrades. That
degradation is itself the result worth reporting: it says how far ahead the
allocation layer of :mod:`bwalloc.allocation` can actually be driven before its
margins stop being economical.

Two strategies
--------------
**Direct** (:class:`DirectMultiHorizon`) fits a separate model per horizon, each
trained to map the information available at the forecast origin straight onto the
target *h* steps later. No error feedback, one model per horizon.

**Recursive** (:func:`recursive_forecast`) applies the one-step model repeatedly,
feeding each prediction back in as history. One model for every horizon, but errors
compound.

Which features are known in advance
-----------------------------------
Splitting the design matrix by *role* is what makes direct forecasting honest here:

- **History columns** (lags, rolling means and standard deviations) can only be read
  at the forecast origin. At a 6-hour lead time the most recent observation is 6
  hours old, and the feature vector must say so.
- **Known-future columns** (the Fourier time terms and calendar integers) are
  deterministic functions of wall-clock time, so their values at the *target*
  timestamp are available at the origin. Withholding them would understate the
  method: an operator provisioning for 9 p.m. always knows it will be 9 p.m.

Context flags sit in between and are controlled by ``future_context``. Taking them at
the target timestamp assumes the operator knows at allocation time that rain or a
scheduled offer is coming -- true for a marketing calendar or a holiday, and true for
rain only to the accuracy of a weather forecast. The default is to take them at the
origin instead, which is the conservative reading; set ``future_context=True`` to
model an operator with a reliable context feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import TARGET, SamplingProfile
from .features import FeatureConfig

#: Column-name prefixes produced by the history-dependent blocks of
#: :func:`bwalloc.features.build_features`. Anything matching these can only be
#: evaluated at the forecast origin.
HISTORY_PREFIXES = ("lag_", "rollmean_", "rollstd_", "lagS_", "rollS_")

#: Deterministic functions of wall-clock time: known arbitrarily far in advance.
CALENDAR_PREFIXES = ("day_", "week_", "hour", "dayofweek")


def split_feature_roles(
    columns, future_context: bool = False
) -> tuple[list[str], list[str]]:
    """Partition feature columns into (origin-only, known-at-target).

    Returns
    -------
    (history, future)
        ``history`` must be read at the forecast origin; ``future`` may be read at
        the target timestamp. Context flags land in ``history`` unless
        ``future_context`` is set.
    """
    history, future = [], []
    for col in columns:
        if col.startswith(HISTORY_PREFIXES):
            history.append(col)
        elif col.startswith(CALENDAR_PREFIXES):
            future.append(col)
        else:
            # Context flags and anything unrecognised: conservative by default.
            (future if future_context else history).append(col)
    return history, future


def horizon_steps(profile: SamplingProfile, hours: float) -> int:
    """Number of samples in a lead time of ``hours``, from the measured rate.

    Same discipline as everywhere else in this project: a "6-hour horizon" is 4 steps
    on GP (86 min/sample) and 4 on Robi (99 min), and a "24-hour horizon" is 17 and
    15 -- never a hardcoded 24.
    """
    return max(1, profile.lag_for_hours(hours))


def direct_design(
    X: pd.DataFrame,
    y: pd.Series,
    steps: int,
    future_context: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Re-align a one-step design matrix for direct ``steps``-ahead forecasting.

    Row *i* of ``X`` holds features whose history reaches up to ``i - 1``; predicting
    ``y`` at position ``i + steps - 1`` from it is therefore a genuine ``steps``-ahead
    forecast. History columns are taken from the origin row, known-future columns
    from the target row.

    ``steps=1`` returns ``X`` and ``y`` unchanged, so the horizon study reduces
    exactly to the existing one-step benchmark at its first point.

    Returns
    -------
    (X_h, y_h, origins)
        Indexed by the *target* timestamp, so folds built on it schedule the times
        being predicted. ``origins`` records the matching forecast-origin timestamp.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    n = len(X)
    if steps >= n:
        raise ValueError(f"steps={steps} leaves no rows (n={n}).")

    history, future = split_feature_roles(X.columns, future_context=future_context)
    lead = steps - 1
    origin_pos = np.arange(0, n - lead)
    target_pos = origin_pos + lead

    blocks = []
    if history:
        blocks.append(X[history].to_numpy()[origin_pos])
    if future:
        blocks.append(X[future].to_numpy()[target_pos])

    target_index = X.index[target_pos]
    X_h = pd.DataFrame(
        np.hstack(blocks), columns=history + future, index=target_index
    )
    y_h = y.iloc[target_pos]
    origins = pd.DatetimeIndex(X.index[origin_pos], name="origin")
    return X_h, y_h, origins


def persistence_at_horizon(y: pd.Series, steps: int) -> pd.Series:
    """The naive baseline a ``steps``-ahead forecaster must beat.

    At a lead time of ``steps``, the freshest observation available is ``steps``
    positions back, so the naive forecast is ``y.shift(steps)`` -- not ``y.shift(1)``,
    which is only reachable at a one-step horizon. Comparing an *h*-step model against
    a one-step naive baseline is the flattering comparison, and this avoids it.
    """
    return y.shift(steps)


def seasonal_naive_at_horizon(y: pd.Series, steps: int, period: int) -> pd.Series:
    """Seasonal naive that respects the lead time.

    Forecasts ``y`` from the same phase of the most recent *complete* cycle that has
    already been observed at the origin: ``period * ceil(steps / period)`` positions
    back. At a 24-hour horizon on GP that is exactly one daily cycle; at a 3-hour
    horizon it is also one cycle, because half a cycle back is not yet observable.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    cycles = int(np.ceil(steps / period))
    return y.shift(period * cycles)


@dataclass
class DirectMultiHorizon:
    """One model per horizon, each trained on its own re-aligned design matrix.

    ``model_factory`` is called once per horizon and must return a fresh, unfitted
    :class:`bwalloc.models.Forecaster`.
    """

    model_factory: object
    steps_list: tuple[int, ...] = (1,)
    future_context: bool = False
    models_: dict[int, object] = field(default_factory=dict, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectMultiHorizon":
        for steps in self.steps_list:
            X_h, y_h, _ = direct_design(X, y, steps, self.future_context)
            self.models_[steps] = self.model_factory().fit(X_h, y_h)
        return self

    def predict(self, X: pd.DataFrame, steps: int) -> np.ndarray:
        """Predict at ``steps`` ahead for every row of an already-aligned ``X``."""
        if steps not in self.models_:
            raise KeyError(f"No model fitted for horizon {steps}.")
        return self.models_[steps].predict(X)


# --------------------------------------------------------------------------- #
# Recursive rollout
# --------------------------------------------------------------------------- #

def history_row(
    y_full: np.ndarray,
    pos: int,
    profile: SamplingProfile,
    config: FeatureConfig,
) -> dict[str, float]:
    """Rebuild the history-dependent features at position ``pos`` from a raw series.

    Mirrors the history blocks of :func:`bwalloc.features.build_features`, but reads
    from a mutable array so the recursive rollout can substitute its own predictions
    for observations it does not have yet. ``pos`` indexes ``y_full``, the *undropped*
    target series, because lags reach back into the warm-up rows that
    ``build_features`` discards.

    The duplication of naming and window logic is deliberate but load-bearing:
    ``tests/test_bwalloc.py`` asserts this function reproduces the corresponding row
    of ``build_features`` exactly, so the two cannot drift apart silently.
    """
    out: dict[str, float] = {}

    for hours in config.lag_hours:
        lag = profile.lag_for_hours(hours)
        out[f"lag_{hours:g}h"] = y_full[pos - lag] if pos - lag >= 0 else np.nan

    for hours in config.rolling_hours:
        window = profile.lag_for_hours(hours)
        if window < 2:
            continue
        # Window ends at pos-1: the shift-then-roll ordering of features.py.
        lo = pos - window
        chunk = y_full[lo:pos] if lo >= 0 else np.array([])
        out[f"rollmean_{hours:g}h"] = float(np.mean(chunk)) if len(chunk) == window else np.nan
        out[f"rollstd_{hours:g}h"] = (
            float(np.std(chunk, ddof=1)) if len(chunk) == window else np.nan
        )

    for n in config.lag_samples:
        out[f"lagS_{n}"] = y_full[pos - n] if pos - n >= 0 else np.nan

    for n in config.rolling_samples:
        lo = pos - n
        chunk = y_full[lo:pos] if lo >= 0 else np.array([])
        ok = len(chunk) == n
        out[f"rollS_mean_{n}"] = float(np.mean(chunk)) if ok else np.nan
        out[f"rollS_std_{n}"] = float(np.std(chunk, ddof=1)) if ok and n > 1 else np.nan

    return out


def recursive_forecast(
    model,
    X: pd.DataFrame,
    y_full: pd.Series,
    profile: SamplingProfile,
    config: FeatureConfig,
    target_positions: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Roll a one-step model forward ``steps`` times, feeding predictions back in.

    ``target_positions`` are positions in ``X`` of the timestamps to forecast. For
    each, the rollout restarts from that row's forecast origin (``steps - 1`` rows
    earlier), so every prediction uses only observations available at that origin and
    its own intermediate predictions thereafter.

    The comparison against :class:`DirectMultiHorizon` is the point: recursive reuses
    one model but compounds its errors, and which effect wins is an empirical question
    that this project has never asked.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")

    feature_names = list(X.columns)
    # Map each row of X back to its position in the undropped series, because the
    # lags reach into warm-up rows that build_features dropped.
    full_index = pd.DatetimeIndex(y_full.index)
    x_to_full = full_index.get_indexer(pd.DatetimeIndex(X.index))
    if (x_to_full < 0).any():
        raise ValueError("X.index is not a subset of y_full.index.")

    values = y_full.to_numpy(dtype=float)
    X_values = X.to_numpy(dtype=float)
    lead = steps - 1
    out = np.empty(len(target_positions), dtype=float)

    for k, tpos in enumerate(np.asarray(target_positions, dtype=int)):
        origin = tpos - lead
        if origin < 0:
            out[k] = np.nan
            continue
        # Work on a copy so predictions written back never contaminate the next
        # rollout, which must restart from real observations.
        buf = values.copy()
        pred = np.nan
        for s in range(steps):
            row_x = origin + s
            full_pos = x_to_full[row_x]
            feats = dict(zip(feature_names, X_values[row_x]))
            # Overwrite only the history block; calendar and context terms at this
            # timestamp are already correct in X.
            feats.update(history_row(buf, full_pos, profile, config))
            frame = pd.DataFrame([feats], columns=feature_names, index=[X.index[row_x]])
            pred = float(model.predict(frame)[0])
            # Write the prediction into the buffer as if it had been observed, so the
            # next step's lags and rolling windows see it.
            buf[full_pos] = pred
        out[k] = pred

    return out
