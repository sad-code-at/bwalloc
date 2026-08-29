"""Leak-safe feature construction.

Two rules are enforced here, both of which the original notebooks violated:

1. **No feature at time t may depend on y_t.** Every rolling statistic is computed on
   a series that has already been shifted by one step. The original code wrote
   ``df['Gbps'].rolling(3).mean()`` with no shift, which put y_t inside its own
   feature vector and made the reported RMSE (6.47 on GP) unreachable in practice --
   the same model with the shift restored scores 19.30, worse than the training mean.

2. **Seasonal lags are derived from the measured sampling rate, never hardcoded.**
   ``lag_for_hours(24)`` resolves to 16 samples on GP and 15 on Robi. Preferred over
   integer lags entirely are the Fourier terms below, which are exact under irregular
   sampling because they are computed from wall-clock time rather than sample count.

The public entry point is :func:`build_features`. :class:`FeatureConfig` selects which
blocks are switched on, which is what the ablation study varies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import TARGET, SamplingProfile

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_WEEK = 604_800.0


@dataclass(frozen=True)
class FeatureConfig:
    """Which feature blocks to build.

    The defaults are the recommended configuration. The ablation study in the
    benchmark notebook toggles these one block at a time.
    """

    #: Recent-history lags, in hours. Resolved to integer sample lags via the profile.
    lag_hours: tuple[float, ...] = (1.5, 3.0, 4.5, 24.0)
    #: Rolling mean/std windows, in hours. Always computed on the shifted series.
    rolling_hours: tuple[float, ...] = (4.5, 24.0)
    #: Raw sample-count lags, bypassing the profile. Exists solely to reproduce the
    #: original design faithfully in the ablation -- ``lag_samples=(1, 2, 3, 24)``
    #: is what the original code built while believing 24 meant one day. Do not use
    #: for new work; prefer ``lag_hours``, which is sampling-rate aware.
    lag_samples: tuple[int, ...] = ()
    #: Raw sample-count rolling windows. Same purpose and same caveat.
    rolling_samples: tuple[int, ...] = ()
    #: Number of Fourier harmonics for the daily cycle (0 disables).
    daily_harmonics: int = 3
    #: Number of Fourier harmonics for the weekly cycle (0 disables).
    weekly_harmonics: int = 1
    #: Include raw calendar integers (hour, dayofweek). Kept off by default: with
    #: Fourier terms present they are redundant, and tree models will happily split
    #: on `hour` in ways that do not generalise across the 55-day span.
    calendar_ints: bool = False
    #: Context flags to include, or None for "all flags present in the frame".
    context_flags: tuple[str, ...] | None = None
    #: Include context flags at all.
    use_context: bool = True


def fourier_terms(index: pd.DatetimeIndex, period_seconds: float, n_harmonics: int,
                  prefix: str) -> pd.DataFrame:
    """Sine/cosine harmonics of wall-clock time.

    Unlike an integer lag, these are unaffected by irregular sampling: a sample at
    10:12 and one at 11:38 land at their true phase within the daily cycle. This is
    the principled replacement for ``lag_24``-as-daily-seasonality.
    """
    if n_harmonics <= 0:
        return pd.DataFrame(index=index)
    # Seconds since epoch keeps phase continuous across day and year boundaries.
    t = index.astype("int64").to_numpy() / 1e9
    phase = 2.0 * np.pi * t / period_seconds
    out = {}
    for k in range(1, n_harmonics + 1):
        out[f"{prefix}_sin{k}"] = np.sin(k * phase)
        out[f"{prefix}_cos{k}"] = np.cos(k * phase)
    return pd.DataFrame(out, index=index)


def _lag_block(y: pd.Series, profile: SamplingProfile,
               lag_hours: tuple[float, ...]) -> pd.DataFrame:
    out = {}
    for hours in lag_hours:
        lag = profile.lag_for_hours(hours)
        # Name by wall-clock hours, not sample count, so a reader cannot mistake
        # "lag_24" for "one day" the way the original code did.
        out[f"lag_{hours:g}h"] = y.shift(lag)
    return pd.DataFrame(out, index=y.index)


def _rolling_block(y: pd.Series, profile: SamplingProfile,
                   rolling_hours: tuple[float, ...]) -> pd.DataFrame:
    # Shift first, then roll. This ordering is the whole ballgame: it guarantees the
    # window ends at t-1 and cannot see y_t.
    shifted = y.shift(1)
    out = {}
    for hours in rolling_hours:
        window = profile.lag_for_hours(hours)
        if window < 2:
            continue
        out[f"rollmean_{hours:g}h"] = shifted.rolling(window).mean()
        out[f"rollstd_{hours:g}h"] = shifted.rolling(window).std()
    return pd.DataFrame(out, index=y.index)


def build_features(
    df: pd.DataFrame,
    profile: SamplingProfile,
    config: FeatureConfig | None = None,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the design matrix and target, dropping rows with incomplete history.

    Returns
    -------
    (X, y)
        Aligned on a common index. No column of ``X`` depends on ``y`` at or after
        its own timestamp; :func:`bwalloc.features.assert_no_leakage` verifies this.
    """
    config = config or FeatureConfig()
    y = df[target].astype(float)
    index = pd.DatetimeIndex(df.index)

    blocks: list[pd.DataFrame] = []

    if config.lag_hours:
        blocks.append(_lag_block(y, profile, config.lag_hours))
    if config.rolling_hours:
        blocks.append(_rolling_block(y, profile, config.rolling_hours))
    if config.lag_samples:
        blocks.append(
            pd.DataFrame(
                {f"lagS_{n}": y.shift(n) for n in config.lag_samples}, index=index
            )
        )
    if config.rolling_samples:
        # Shifted first, exactly as in _rolling_block: reproducing the original's
        # choice of window is in scope, reproducing its leak is not.
        shifted = y.shift(1)
        blocks.append(
            pd.DataFrame(
                {
                    f"rollS_{stat}_{n}": getattr(shifted.rolling(n), stat)()
                    for n in config.rolling_samples
                    for stat in ("mean", "std")
                },
                index=index,
            )
        )
    if config.daily_harmonics:
        blocks.append(fourier_terms(index, SECONDS_PER_DAY, config.daily_harmonics, "day"))
    if config.weekly_harmonics:
        blocks.append(fourier_terms(index, SECONDS_PER_WEEK, config.weekly_harmonics, "week"))

    if config.calendar_ints:
        blocks.append(
            pd.DataFrame(
                {"hour": index.hour, "dayofweek": index.dayofweek},
                index=index,
            )
        )

    if config.use_context:
        flags = config.context_flags
        if flags is None:
            flags = tuple(c for c in df.columns if c.startswith("is_") or c == "event")
        present = [f for f in flags if f in df.columns]
        if present:
            blocks.append(df[present].astype(float))

    if not blocks:
        raise ValueError("FeatureConfig produced no features.")

    X = pd.concat(blocks, axis=1)
    # Rows with incomplete lag/rolling history are dropped, never imputed. The
    # original code filled them with 0, which injects a fake "zero bandwidth" regime
    # at the head of the series and biases early splits.
    complete = X.notna().all(axis=1)
    X, y = X.loc[complete], y.loc[complete]
    if X.empty:
        raise ValueError(
            "No complete rows remain. The longest lag/rolling window likely exceeds "
            "the series length."
        )
    return X, y


def assert_no_leakage(
    df: pd.DataFrame,
    profile: SamplingProfile,
    config: FeatureConfig | None = None,
    target: str = TARGET,
    seed: int = 0,
) -> None:
    """Verify no feature column depends on the contemporaneous target.

    Rebuilds the design matrix with the *last* target value perturbed. Any column
    whose value at the final timestamp moves is reading y_t, and this raises.

    This is the regression test for the exact bug that produced the original
    project's headline result.
    """
    config = config or FeatureConfig()
    X_ref, _ = build_features(df, profile, config, target=target)

    perturbed = df.copy()
    rng = np.random.default_rng(seed)
    last = perturbed.index[-1]
    perturbed.loc[last, target] = float(perturbed[target].iloc[-1]) + 1000.0 + rng.random()

    X_alt, _ = build_features(perturbed, profile, config, target=target)

    common = X_ref.index.intersection(X_alt.index)
    diff = (X_ref.loc[common] - X_alt.loc[common]).abs().max()
    offenders = sorted(diff[diff > 1e-9].index)
    if offenders:
        raise AssertionError(
            "Target leakage: perturbing y at the final timestamp changed feature "
            f"column(s) {offenders} at or before that timestamp."
        )
