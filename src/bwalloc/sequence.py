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


# --------------------------------------------------------------------------- #
# Covariate-aware sequence models
#
# Everything above reproduces the original study, which read a bare demand window
# (``input_shape=(lookback, 1)``). That was a reproduction constraint and nothing more.
# These traces carry nine context flags on GP and five on Robi, plus Fourier terms of
# wall-clock time, and `is_rain` is the one feature the audit found a real effect for
# (variance ratio 1.330, Levene p < 0.001) -- yet no sequence model in this project had
# ever been shown it.
#
# The covariates are *time-varying*: there is a value of `is_rain` at each of the 24
# past timestamps, not one value for the whole window. Flattening them to a single
# static vector throws most of that away. So, following the covariate taxonomy that
# DeepAR and the Temporal Fusion Transformer use, the design matrix is read as three
# blocks: past covariates as extra channels over the window, covariates known at the
# forecast origin, and static summaries.
# --------------------------------------------------------------------------- #

_COVARIATE_LAG_COLUMN = re.compile(r"^(?P<base>.+)__lagS_(?P<lag>\d+)$")

#: Channel 0 is always demand, so a model that ignores every other channel degenerates
#: exactly to the univariate reproduction above. That makes "do the covariates earn
#: anything?" a measurable quantity rather than an assumption.
DEMAND_CHANNEL = "demand"


@dataclass(frozen=True)
class ChannelWindow:
    """A ``(n_rows, lookback, n_channels)`` view of a design matrix, plus the rest.

    ``channels[0]`` is always :data:`DEMAND_CHANNEL`. ``static`` names the columns that
    carry no lookback -- contemporaneous context flags, rolling statistics, and the
    Fourier terms of the *target* timestamp, which are known at the forecast origin
    because the clock is not something we have to predict.
    """

    channels: tuple[str, ...]
    static: tuple[str, ...]
    #: Column names per channel, oldest observation first.
    columns: tuple[tuple[str, ...], ...]

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def lookback(self) -> int:
        return len(self.columns[0])

    def window(self, X: pd.DataFrame) -> np.ndarray:
        """Extract ``(n_rows, lookback, n_channels)`` in the stored column order."""
        stacked = [np.asarray(X[list(cols)], dtype=float) for cols in self.columns]
        return np.stack(stacked, axis=-1)

    def static_block(self, X: pd.DataFrame) -> np.ndarray:
        if not self.static:
            return np.zeros((len(X), 0), dtype=float)
        return np.asarray(X[list(self.static)], dtype=float)


def channel_window(X: pd.DataFrame, lookback: int | None = None) -> ChannelWindow:
    """Group a design matrix into one channel per covariate over the lookback.

    Generalises :func:`lookback_columns`, which is left untouched so the univariate
    reproduction path keeps working byte-for-byte. ``lagS_n`` becomes the demand
    channel; ``{name}__lagS_n`` becomes a channel named ``{name}``; everything else is
    static.

    Raises if the channels disagree on depth or are not consecutive from 1 -- a model
    fed a window with holes in it is silently modelling something other than a
    contiguous history, and that failure is invisible in the RMSE.
    """
    demand: dict[int, str] = {}
    covariates: dict[str, dict[int, str]] = {}
    static: list[str] = []

    for col in X.columns:
        name = str(col)
        m = _LAG_COLUMN.match(name)
        if m:
            demand[int(m.group(1))] = name
            continue
        m = _COVARIATE_LAG_COLUMN.match(name)
        if m:
            covariates.setdefault(m.group("base"), {})[int(m.group("lag"))] = name
            continue
        static.append(name)

    if not demand:
        raise ValueError(
            "No lagS_* columns found. Sequence models need a demand lookback window; "
            "build features with FeatureConfig(lag_samples=range(1, L+1))."
        )

    def ordered(found: dict[int, str], label: str) -> tuple[str, ...]:
        lags = sorted(found)
        if lags != list(range(1, len(lags) + 1)):
            raise ValueError(
                f"Channel {label!r} must carry consecutive lags 1..L; got {lags}."
            )
        # Oldest first: lag_L, ..., lag_1 is chronological order.
        return tuple(found[n] for n in reversed(lags))

    channels = [DEMAND_CHANNEL]
    columns = [ordered(demand, DEMAND_CHANNEL)]
    for base in sorted(covariates):
        channels.append(base)
        columns.append(ordered(covariates[base], base))

    depth = len(columns[0])
    mismatched = [c for c, cols in zip(channels, columns) if len(cols) != depth]
    if mismatched:
        raise ValueError(
            f"Channels disagree on lookback depth: demand has {depth}, "
            f"{mismatched} differ. Build every covariate at the same lags."
        )
    if lookback is not None and depth != lookback:
        raise ValueError(
            f"lookback={lookback} but the design matrix carries {depth} lags per "
            f"channel. Build it with lag_samples=tuple(range(1, {lookback + 1}))."
        )

    return ChannelWindow(
        channels=tuple(channels),
        static=tuple(static),
        columns=tuple(columns),
    )


def _build_covariate_module(kind: str, lookback: int, n_channels: int, n_static: int):
    """The same four architectures, widened to read covariate channels.

    The trunk is unchanged apart from its input width; the static block is concatenated
    with the trunk's output before the dense head, which is how DeepAR admits covariates
    alongside a recurrent state.
    """
    import torch
    import torch.nn as nn

    kind = kind.lower()

    class CovariateNet(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "cnn":
                self.trunk = nn.Sequential(
                    nn.Conv1d(n_channels, 64, kernel_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                trunk_out = 64 * (lookback - 2)
                self.recurrent = False
            elif kind in ("lstm", "gru", "rnn"):
                width = 50 if kind == "rnn" else 64   # SimpleRNN(50); LSTM(64); GRU(64)
                cell = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[kind]
                self.trunk = cell(
                    input_size=n_channels, hidden_size=width, batch_first=True
                )
                trunk_out = width
                self.recurrent = True
            else:
                raise ValueError(
                    f"Unknown architecture {kind!r}; expected one of {ARCHITECTURES}."
                )
            self.out = nn.Linear(trunk_out + n_static, 1)

        def forward(self, window, static):
            if self.recurrent:
                h, _ = self.trunk(window)              # (N, L, C) -> (N, L, W)
                features = h[:, -1, :]
            else:
                features = self.trunk(window.transpose(1, 2))
            if static.shape[1]:
                features = torch.cat([features, static], dim=1)
            return self.out(features)

    return CovariateNet()


def _fit_scaler(arr: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation over ``axis``, with constant columns left alone.

    A context flag can be constant inside a short training window -- `is_rain` is zero
    for most fold-0 windows -- and dividing by its zero standard deviation would put
    NaN through the whole network.
    """
    mu = arr.mean(axis=axis, keepdims=True)
    sd = arr.std(axis=axis, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


@dataclass
class CovariateSequenceForecaster(Forecaster):
    """A sequence model that reads covariate channels, not only demand.

    Identical to :class:`SequenceForecaster` in architecture, width and optimisation --
    the only difference is what it is allowed to see. Running the two side by side on
    one fold schedule therefore isolates the value of the covariates from everything
    else, which is the question notebook 09 exists to answer.
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
            raise ValueError("lookback must be at least 3.")
        self.name = self.name or f"{self.kind}_cov"
        self._module = None
        self._spec: ChannelWindow | None = None

    def _tensors(self, X: pd.DataFrame):
        spec = self._spec
        assert spec is not None
        w = (spec.window(X) - self._w_mu) / self._w_sd
        s = spec.static_block(X)
        if s.shape[1]:
            s = (s - self._s_mu) / self._s_sd
        return w, s

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CovariateSequenceForecaster":
        import torch

        self._spec = channel_window(X, self.lookback)
        w = self._spec.window(X)
        s = self._spec.static_block(X)
        ys = np.asarray(y, dtype=float).reshape(-1, 1)

        # Per-channel scaling on the fitting window only. Per channel rather than
        # globally because demand is ~100 Gbps and a context flag is 0/1; one shared
        # scale would flatten the flags into numerical noise. Using test-block
        # statistics here would leak the test distribution's location and scale, which
        # is the subtler of the two leaks a sequence baseline usually has.
        self._w_mu, self._w_sd = _fit_scaler(w, axis=(0, 1))
        self._s_mu, self._s_sd = (
            _fit_scaler(s, axis=0) if s.shape[1] else (None, None)
        )
        self._y_mu = ys.mean()
        self._y_sd = ys.std() if ys.std() > 0 else 1.0

        torch.manual_seed(self.seed)
        self._module = _build_covariate_module(
            self.kind, self._spec.lookback, self._spec.n_channels, len(self._spec.static)
        )

        wn, sn = self._tensors(X)
        wt = torch.tensor(wn, dtype=torch.float32)
        st = torch.tensor(sn, dtype=torch.float32)
        yt = torch.tensor((ys - self._y_mu) / self._y_sd, dtype=torch.float32)

        opt = torch.optim.Adam(self._module.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        n = len(wt)
        generator = torch.Generator().manual_seed(self.seed)

        self._module.train()
        for _ in range(self.epochs):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self._module(wt[idx], st[idx]), yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self._module is None:
            raise RuntimeError(f"{self.name} is not fitted.")
        wn, sn = self._tensors(X)
        self._module.eval()
        with torch.no_grad():
            out = self._module(
                torch.tensor(wn, dtype=torch.float32),
                torch.tensor(sn, dtype=torch.float32),
            ).numpy().ravel()
        return out * self._y_sd + self._y_mu


def covariate_sequence_models(lookback: int = 24, epochs: int = 30,
                              seed: int = 42) -> list[CovariateSequenceForecaster]:
    """The same four architectures, allowed to see the covariates."""
    return [
        CovariateSequenceForecaster(kind=k, lookback=lookback, epochs=epochs, seed=seed)
        for k in ARCHITECTURES
    ]


def full_feature_config(lookback: int = 24, covariates: bool = True,
                        context_flags: tuple[str, ...] | None = None):
    """The full corrected design at full lag depth -- the opposite of the bare window.

    This is ``FeatureConfig()``'s recommended defaults (wall-clock lags, rolling
    statistics, Fourier terms and context flags) *plus* a dense demand window of
    ``lookback`` consecutive sample lags, which ``run_sequence.py`` established as the
    configuration that actually performs. With ``covariates=True`` each context flag and
    daily harmonic also gets the same lookback, so a sequence model sees their history
    rather than a single contemporaneous value.
    """
    from .features import FeatureConfig

    lags = tuple(range(1, lookback + 1))
    return FeatureConfig(
        lag_samples=lags,
        covariate_lag_samples=lags if covariates else (),
        context_flags=context_flags,
    )
