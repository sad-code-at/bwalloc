"""Rolling-origin backtesting.

The original notebooks scored every model on a single fixed train/test split, and a
different split per notebook -- so the headline comparison table put numbers from a
70/30 index split next to numbers from a date split and ranked them against each
other. On 250-480 test points the resulting gaps are well inside fold-to-fold noise.

This module produces one fold schedule that every model shares, so differences are
attributable to the model and each metric carries a spread across folds.

A fold optionally carves a calibration slice off the end of its training window.
Conformal calibration needs residuals from data the model did not fit, and taking
them from the most recent pre-test data keeps the exchangeability assumption as close
to satisfied as a time series allows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold, expressed as positional indices."""

    number: int
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_calib(self) -> int:
        return len(self.calib)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Fold({self.number}: train={self.n_train}, "
            f"calib={self.n_calib}, test={self.n_test})"
        )


def rolling_origin(
    n: int,
    n_folds: int = 8,
    min_train: int | None = None,
    calib_frac: float = 0.25,
    horizon: int | None = None,
) -> list[Fold]:
    """Build an expanding-window fold schedule over ``n`` ordered observations.

    Parameters
    ----------
    n:
        Number of observations (after feature construction has dropped warm-up rows).
    n_folds:
        Number of test blocks. The tail of the series is divided into this many
        contiguous blocks; fold *i* trains on everything before block *i*.
    min_train:
        Minimum training observations before the first fold. Defaults to 40% of ``n``,
        which on these traces leaves ~350 training rows -- versus the 88 (GP) and 64
        (Robi) that the original multi-model notebooks actually trained on after
        ``dropna()`` removed the first 336 rows.
    calib_frac:
        Fraction of each training window reserved as a conformal calibration set.
        Set to 0 to disable (the calibration array is then empty).
    horizon:
        Test-block size. Defaults to an even division of the tail.

    Returns
    -------
    list[Fold]
        Folds in chronological order. Training windows expand; test blocks never
        overlap and always follow their training data.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not 0.0 <= calib_frac < 1.0:
        raise ValueError("calib_frac must be in [0, 1)")

    min_train = min_train if min_train is not None else max(30, int(0.40 * n))
    if min_train >= n:
        raise ValueError(f"min_train={min_train} leaves no test data (n={n}).")

    tail = n - min_train
    horizon = horizon if horizon is not None else max(1, tail // n_folds)

    folds: list[Fold] = []
    start = min_train
    number = 0
    while start < n and number < n_folds:
        stop = min(start + horizon, n)
        # The final fold absorbs any remainder rather than leaving a stub block.
        if number == n_folds - 1:
            stop = n

        fit_end = start
        n_calib = int(round(calib_frac * fit_end))
        # Calibration must be large enough to express the quantiles we intend to
        # request; conformal.py enforces the precise bound per requested tau.
        n_calib = min(n_calib, fit_end - 10) if fit_end > 10 else 0
        n_calib = max(n_calib, 0)

        train = np.arange(0, fit_end - n_calib)
        calib = np.arange(fit_end - n_calib, fit_end)
        test = np.arange(start, stop)

        if len(train) > 0 and len(test) > 0:
            folds.append(Fold(number=number, train=train, calib=calib, test=test))

        start = stop
        number += 1

    if not folds:
        raise ValueError("Fold schedule is empty; check n, n_folds and min_train.")
    return folds


def describe_folds(folds: list[Fold], index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Tabulate a fold schedule for reporting."""
    rows = []
    for f in folds:
        row = {
            "fold": f.number,
            "n_train": f.n_train,
            "n_calib": f.n_calib,
            "n_test": f.n_test,
        }
        if index is not None:
            row["train_start"] = index[f.train[0]]
            row["test_start"] = index[f.test[0]]
            row["test_end"] = index[f.test[-1]]
        rows.append(row)
    return pd.DataFrame(rows)
