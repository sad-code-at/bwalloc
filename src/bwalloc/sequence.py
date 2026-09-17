"""Univariate sequence forecasters (CNN, LSTM, GRU, RNN) on the corrected protocol.

The original study reported a CNN at 9.69 RMSE on GP against XGBoost at 13.91 and
concluded the sequence models were better architectures. They were not being compared
on equal terms: the sequence models were fitted on a ``train_size=0.7`` split (~600
rows) while the tree models went through the ``lag_336`` -> ``dropna()`` -> date-split
path that left **88 training rows**. The gap was the split, not the architecture.

This module re-runs those architectures -- same shapes, same widths, same 30 epochs --
under the corrected protocol, so the comparison is finally like-for-like: one
rolling-origin fold schedule, leak-safe features, the same per-fold training windows
and the same naive baselines as every other model in the benchmark.

Design notes
------------
The models are **univariate**, reading a bare window of past demand exactly as the
original did (``input_shape=(lookback, 1)``). They therefore consume a design matrix of
consecutive sample lags (``lagS_1 ... lagS_L``) built by the ordinary leak-safe feature
builder, which keeps them inside the existing backtest harness rather than on a
parallel data path. Because the window is read from the same ``X`` as everything else,
no separate leakage argument is needed: :func:`bwalloc.features.assert_no_leakage`
covers it.

Standardisation statistics are computed on the fitting window only and reused at
predict time. Fitting on the test block's mean would be a second, subtler leak, and it
is the reason a naively written sequence baseline often looks strong.

Torch is imported lazily so the package remains usable without it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import Forecaster

#: Architectures as the original study defined them, so the comparison is of the same
#: models under a different protocol rather than of different models.
ARCHITECTURES = ("cnn", "lstm", "gru", "rnn")

_LAG_COLUMN = re.compile(r"^lagS_(\d+)$")


def lookback_columns(X: pd.DataFrame) -> list[str]:
    """The consecutive sample-lag columns of ``X``, ordered oldest observation first.

    Raises if they are absent or not consecutive from 1, because a sequence model fed
    a design matrix with gaps in its window is silently modelling something other than
    a contiguous history.
    """
    found: dict[int, str] = {}
    for col in X.columns:
        m = _LAG_COLUMN.match(str(col))
        if m:
            found[int(m.group(1))] = str(col)
    if not found:
        raise ValueError(
            "No lagS_* columns found. Sequence models need a univariate lookback "
            "window; build features with FeatureConfig(lag_samples=range(1, L+1))."
        )
    lags = sorted(found)
    if lags != list(range(1, len(lags) + 1)):
        raise ValueError(
            f"Lookback window must be consecutive lags 1..L; got {lags}."
        )
    # Oldest first: lag_L, ..., lag_1 is chronological order.
    return [found[n] for n in reversed(lags)]


def _build_module(kind: str, lookback: int):
    """The original architectures, translated from Keras to torch."""
    import torch.nn as nn

    kind = kind.lower()
    if kind == "cnn":
        # Conv1D(filters=64, kernel_size=3, activation='relu') -> Flatten -> Dense(1)
        return nn.Sequential(
            _Permute(),                       # (N, L, 1) -> (N, 1, L)
            nn.Conv1d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (lookback - 2), 1),
        )
    if kind in ("lstm", "gru", "rnn"):
        width = 50 if kind == "rnn" else 64   # SimpleRNN(50); LSTM(64); GRU(64)
        return _RecurrentHead(kind, width)
    raise ValueError(f"Unknown architecture {kind!r}; expected one of {ARCHITECTURES}.")


def _torch_nn():
    import torch.nn as nn
    return nn


class _Permute:  # replaced at construction time; see _build_module
    """Placeholder so the module list reads in Keras order."""

    def __new__(cls):
        import torch.nn as nn

        class Permute(nn.Module):
            def forward(self, x):
                return x.transpose(1, 2)

        return Permute()


class _RecurrentHead:
    """A recurrent layer over the window, then a dense head on the final state."""

    def __new__(cls, kind: str, width: int):
        import torch
        import torch.nn as nn

        cell = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[kind]

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = cell(input_size=1, hidden_size=width, batch_first=True)
                self.out = nn.Linear(width, 1)

            def forward(self, x):
                h, _ = self.rnn(x)
                return self.out(h[:, -1, :])

        del torch
        return Head()


@dataclass
class SequenceForecaster(Forecaster):
    """A univariate sequence model on a contiguous lookback window.

    Hyperparameters deliberately match the original study (30 epochs, Adam, MSE, batch
    32) so that any difference in the reported numbers is attributable to the
    evaluation protocol rather than to retuning. ``epochs`` is exposed because the
    original used no validation split or early stopping, and a reviewer may reasonably
    ask what happens with less fitting.
    """

    kind: str = "lstm"
    lookback: int = 24
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    name: str = ""

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture {self.kind!r}; expected one of {ARCHITECTURES}."
            )
        if self.lookback < 3:
            # Conv1d with kernel_size=3 needs at least three steps.
            raise ValueError("lookback must be at least 3.")
        self.name = self.name or self.kind
        self._module = None
        self._cols: list[str] | None = None
        self._x_mu = self._x_sd = self._y_mu = self._y_sd = None

    # -- internals ---------------------------------------------------------- #

    def _window(self, X: pd.DataFrame) -> np.ndarray:
        cols = self._cols if self._cols is not None else lookback_columns(X)
        arr = np.asarray(X[cols], dtype=float)
        return arr.reshape(len(arr), len(cols), 1)

    # -- interface ---------------------------------------------------------- #

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SequenceForecaster":
        import torch

        self._cols = lookback_columns(X)
        if len(self._cols) != self.lookback:
            raise ValueError(
                f"{self.name}: lookback={self.lookback} but X carries "
                f"{len(self._cols)} lag columns. Build the design matrix with "
                f"FeatureConfig(lag_samples=tuple(range(1, {self.lookback + 1})))."
            )

        xs = self._window(X)
        ys = np.asarray(y, dtype=float).reshape(-1, 1)

        # Standardise on the fitting window only. Using test-block statistics here
        # would leak the test distribution's location and scale.
        self._x_mu, self._x_sd = xs.mean(), xs.std()
        self._y_mu, self._y_sd = ys.mean(), ys.std()
        self._x_sd = self._x_sd if self._x_sd > 0 else 1.0
        self._y_sd = self._y_sd if self._y_sd > 0 else 1.0

        torch.manual_seed(self.seed)
        self._module = _build_module(self.kind, self.lookback)

        xt = torch.tensor((xs - self._x_mu) / self._x_sd, dtype=torch.float32)
        yt = torch.tensor((ys - self._y_mu) / self._y_sd, dtype=torch.float32)

        opt = torch.optim.Adam(self._module.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        n = len(xt)
        generator = torch.Generator().manual_seed(self.seed)

        self._module.train()
        for _ in range(self.epochs):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self._module(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self._module is None:
            raise RuntimeError(f"{self.name} is not fitted.")
        xs = (self._window(X) - self._x_mu) / self._x_sd
        self._module.eval()
        with torch.no_grad():
            out = self._module(torch.tensor(xs, dtype=torch.float32)).numpy().ravel()
        return out * self._y_sd + self._y_mu


def sequence_models(lookback: int = 24, epochs: int = 30,
                    seed: int = 42) -> list[SequenceForecaster]:
    """The four architectures the original study compared."""
    return [
        SequenceForecaster(kind=k, lookback=lookback, epochs=epochs, seed=seed)
        for k in ARCHITECTURES
    ]


def sequence_feature_config(lookback: int = 24):
    """A univariate design matrix of ``lookback`` consecutive sample lags.

    Deliberately carries no Fourier terms, calendar integers or context flags: the
    original sequence models read ``input_shape=(lookback, 1)``, and reproducing that
    is the point. Running the tree models on this same matrix isolates the effect of
    architecture from the effect of the feature set.
    """
    from .features import FeatureConfig

    return FeatureConfig(
        lag_hours=(),
        rolling_hours=(),
        lag_samples=tuple(range(1, lookback + 1)),
        rolling_samples=(),
        daily_harmonics=0,
        weekly_harmonics=0,
        calendar_ints=False,
        use_context=False,
    )
