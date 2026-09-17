"""Dataset loading and sampling-rate diagnostics.

The two operator traces in this project are *not* hourly, despite every notebook in
the original codebase assuming they are. GP is sampled every ~86 min and Robi every
~99 min, which means an integer lag of 24 samples spans ~34 h (GP) or ~40 h (Robi) --
roughly anti-phase to the daily cycle rather than aligned with it.

Every seasonal hyperparameter in this package is derived from `SamplingProfile`
rather than hardcoded, so the mistake cannot be repeated silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Gbps"
TIME_COL = "Timestamp"

#: Context flags present in each trace. GP carries nine hand-labelled flags; Robi
#: only five. The asymmetry is itself a research object (see the transfer study).
CONTEXT_FLAGS: dict[str, tuple[str, ...]] = {
    "gp": (
        "is_weekend",
        "is_holiday",
        "is_offer",
        "is_event",
        "is_political_gathering",
        "is_powercut",
        "is_rain",
        "event",
        "is_drama",
    ),
    "robi": (
        "is_weekend",
        "is_holiday",
        "is_event",
        "is_political_gathering",
        "is_powercut",
    ),
}

#: Columns shipped inside the CSVs that must never be used as features.
#:
#: ``target`` is an exact duplicate of ``Gbps``. The ``lag_*``/``rolling_*`` columns
#: were precomputed by an earlier pipeline and do not reproduce from this file's own
#: ordering (60 of 881 rows disagree with ``Gbps.shift(24)``), so their provenance is
#: unknown. They are dropped and recomputed from scratch by :mod:`bwalloc.features`.
STALE_COLUMNS = (
    "target",
    "hour",
    "dayofweek",
    "dayofyear",
    "month",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "rolling_mean_24",
    "rolling_std_24",
)

DATASETS = {"gp": "gp_dhaka.csv", "robi": "robi_dhaka3.csv"}


def data_dir() -> Path:
    """Locate ``data/`` whether running from the repo, a notebook, or Colab."""
    env = os.environ.get("BWALLOC_DATA_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data"
        if candidate.is_dir() and (candidate / DATASETS["gp"]).exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the data/ directory. Set BWALLOC_DATA_DIR to point at it."
    )


@dataclass(frozen=True)
class SamplingProfile:
    """Empirical sampling geometry of a trace.

    Attributes
    ----------
    median_gap_min:
        Median spacing between consecutive samples, in minutes.
    samples_per_day:
        ``1440 / median_gap_min``. Non-integer in both traces, which is exactly why
        integer-lag seasonal features misfire.
    daily_period:
        ``samples_per_day`` rounded to the nearest integer -- the best available
        integer lag for a seasonal-naive baseline. Use Fourier time features
        (:func:`bwalloc.features.fourier_terms`) instead wherever possible.
    n_samples, span_days, max_gap_min, n_duplicate_timestamps:
        Descriptive diagnostics reported in the data audit.
    """

    median_gap_min: float
    samples_per_day: float
    daily_period: int
    n_samples: int
    span_days: float
    max_gap_min: float
    n_duplicate_timestamps: int

    def lag_for_hours(self, hours: float) -> int:
        """Integer lag (in samples) closest to ``hours`` of wall-clock time.

        ``profile.lag_for_hours(24)`` gives 16 for GP -- not 24. This method exists
        so that no caller ever has to guess.
        """
        return max(1, int(round(hours * 60.0 / self.median_gap_min)))

    def hours_for_lag(self, lag: int) -> float:
        """Wall-clock hours spanned by an integer lag of ``lag`` samples."""
        return lag * self.median_gap_min / 60.0

    def describe(self) -> str:
        return (
            f"n={self.n_samples}, span={self.span_days:.1f} d, "
            f"median gap={self.median_gap_min:.1f} min "
            f"({self.samples_per_day:.2f} samples/day), "
            f"daily period ~= {self.daily_period} samples "
            f"[lag 24 spans {self.hours_for_lag(24):.1f} h]"
        )


def sampling_profile(df: pd.DataFrame) -> SamplingProfile:
    """Measure the sampling geometry of a time-indexed frame."""
    idx = pd.DatetimeIndex(df.index)
    if not idx.is_monotonic_increasing:
        raise ValueError("Index must be sorted before profiling.")
    gaps_min = pd.Series(idx).diff().dt.total_seconds().dropna() / 60.0
    if gaps_min.empty:
        raise ValueError("Need at least two samples to profile sampling rate.")
    median_gap = float(gaps_min.median())
    per_day = 1440.0 / median_gap
    span = (idx.max() - idx.min()).total_seconds() / 86400.0
    return SamplingProfile(
        median_gap_min=median_gap,
        samples_per_day=per_day,
        daily_period=int(round(per_day)),
        n_samples=len(idx),
        span_days=span,
        max_gap_min=float(gaps_min.max()),
        n_duplicate_timestamps=int(idx.duplicated().sum()),
    )


def load(operator: str, drop_stale: bool = True) -> pd.DataFrame:
    """Load one operator trace, sorted and indexed by timestamp.

    Parameters
    ----------
    operator:
        ``"gp"`` or ``"robi"``.
    drop_stale:
        Drop :data:`STALE_COLUMNS` (the duplicated target and the unverifiable
        precomputed lag/rolling columns). Leave ``True`` unless you are specifically
        auditing those columns.
    """
    key = operator.lower()
    if key not in DATASETS:
        raise KeyError(f"Unknown operator {operator!r}; expected one of {list(DATASETS)}")

    df = pd.read_csv(data_dir() / DATASETS[key])
    # The CSVs mix "4/4/2025 10:12" and zero-padded forms; format="mixed" parses both
    # without silently falling back to dayfirst guessing on a per-row basis.
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="mixed")
    df = df.sort_values(TIME_COL).set_index(TIME_COL)

    if drop_stale:
        df = df.drop(columns=[c for c in STALE_COLUMNS if c in df.columns])

    for flag in CONTEXT_FLAGS[key]:
        if flag in df.columns:
            df[flag] = df[flag].astype(int)

    if df[TARGET].isna().any():
        n = int(df[TARGET].isna().sum())
        raise ValueError(f"{operator}: {n} missing values in {TARGET}; handle explicitly.")

    df.attrs["operator"] = key
    return df


def load_all(drop_stale: bool = True) -> dict[str, pd.DataFrame]:
    """Load every operator trace, keyed by operator name."""
    return {name: load(name, drop_stale=drop_stale) for name in DATASETS}


def autocorrelation_by_lag(
    y: pd.Series, max_lag: int = 60, profile: SamplingProfile | None = None
) -> pd.DataFrame:
    """Autocorrelation as a function of lag, annotated with wall-clock hours.

    This is the diagnostic that exposes the core modelling error in the original
    work: for GP the daily peak sits at lag 16 (+0.68) while lag 24 -- the lag every
    original model used as its "1-day" feature -- sits at -0.35.
    """
    if not isinstance(y, pd.Series):
        # A DataFrame reaches pandas' two-argument DataFrame.corr, where the shifted
        # frame is read as the *method* argument and fails deep inside pandas with
        # "truth value of a DataFrame is ambiguous" -- a message that says nothing
        # about the real mistake. Refuse it here instead.
        raise TypeError(
            f"autocorrelation_by_lag expects a Series, got {type(y).__name__}. "
            f"Pass the target column (df[TARGET]), not the whole frame."
        )

    rows = []
    for lag in range(1, max_lag + 1):
        rows.append(
            {
                "lag_samples": lag,
                # `is not None` rather than truthiness: a dataclass that later grows
                # a __len__ or __bool__ would silently start reporting NaN hours.
                "lag_hours": (
                    profile.hours_for_lag(lag) if profile is not None else np.nan
                ),
                "autocorr": float(y.corr(y.shift(lag))),
            }
        )
    return pd.DataFrame(rows)
