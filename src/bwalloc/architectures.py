"""Modern forecasting architectures, all reading covariate channels by default.

Notebook 08 found that on these traces **lag depth mattered more than architecture**:
moving a random forest from four lags to twenty-four bought 18% RMSE on GP, while the
best architecture bought 9% on top of that, on one operator only. That finding is what
this module is built to interrogate, with architectures chosen because each tests a
specific reading of it:

``DLinear`` / ``NLinear``
    Zeng et al. (arXiv:2205.13504) showed a one-layer linear model on a decomposed
    series beating every Transformer they tested. With ~350-800 training rows per fold
    this is the model most likely to win outright, and it is the honest control: if a
    Transformer cannot beat a linear layer, it earned nothing.

``TCN``
    Dilated causal convolutions reach 31 steps at four layers (k=3, dilations 1/2/4/8),
    covering the whole window with far fewer parameters than an LSTM. The original
    study's "CNN" is a *single undilated* ``Conv1D``; this is the principled version of
    the architecture that already won on GP (Bai et al., arXiv:1803.01271).

``TransformerForecaster``
    The direct test of the lag-depth reading: attention learns *which* past steps matter
    rather than weighting all 24 alike. The attention map is itself a result, and it is
    checkable against the measured daily period -- 17 samples on GP, 15 on Robi.

``NBeats`` / ``NBeatsX``
    Basis expansion with backward and forward residual links (Oreshkin et al.,
    arXiv:1905.10437). ``NBeats`` is univariate and is kept only as a labelled control;
    ``NBeatsX`` (Olivares et al., arXiv:2104.05522) adds the exogenous block and is the
    reported member of the pair.

**Every model here reads the full feature set by default.** Three of these architectures
are natively univariate as their papers define them, and implemented literally they
would quietly reproduce the very limitation this module exists to remove -- so covariates
enter structurally (as extra channels over the lookback, per the DeepAR / Temporal Fusion
Transformer taxonomy) and a test gate fails any model whose predictions do not move when
a covariate moves.

Honest expectation, recorded before the results: on ~900 points across 55 days the small
models should win. Elsayed et al. (arXiv:2101.02118) found gradient boosting on a
windowed representation matching state-of-the-art deep models on benchmarks two orders of
magnitude larger than this trace. N-BEATS and the Transformer are here because a negative
result at this sample size *is* a result -- it is evidence for the lag-depth reading, not
against the architectures.

Deliberately not implemented, so the omissions are decisions rather than oversights:
PatchTST (patching 24 steps yields ~3 tokens, so its mechanism cannot operate at our
lookback), Informer and Autoformer (built for horizons of 96-720; ours is 1-17), and the
full Temporal Fusion Transformer (it needs many related series; we borrow its covariate
taxonomy, not its architecture).

Torch is imported lazily so the package stays usable without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import Forecaster
from .sequence import ChannelWindow, channel_window, _fit_scaler

#: Everything this module offers, in the order the notebooks report them.
MODERN_ARCHITECTURES = ("dlinear", "nlinear", "tcn", "transformer", "nbeatsx", "nbeats")


# --------------------------------------------------------------------------- #
# Shared training machinery
# --------------------------------------------------------------------------- #

@dataclass
class TorchWindowForecaster(Forecaster):
    """Base class: channel window in, one-step point forecast out.

    Subclasses supply :meth:`build_module` only. Everything else -- reading the design
    matrix through :func:`bwalloc.sequence.channel_window`, per-channel scaling computed
    on the fitting window alone, the Adam/MSE loop, seeding -- is shared, so a
    difference between two rows of the results table is a difference of architecture and
    not of training protocol. That is the same discipline ``evaluate.run_backtest``
    enforces across models.
    """

    lookback: int = 24
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.1
    seed: int = 42
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or self.__class__.__name__.lower()
        self._module = None
        self._spec: ChannelWindow | None = None

    # -- subclass hook ------------------------------------------------------ #

    def build_module(self, n_channels: int, lookback: int, n_static: int):
        raise NotImplementedError

    # -- internals ---------------------------------------------------------- #

    def _tensors(self, X: pd.DataFrame):
        spec = self._spec
        assert spec is not None
        w = (spec.window(X) - self._w_mu) / self._w_sd
        s = spec.static_block(X)
        if s.shape[1]:
            s = (s - self._s_mu) / self._s_sd
        return w, s

    @property
    def n_parameters(self) -> int:
        """Parameter count, for the size-against-accuracy comparison."""
        if self._module is None:
            return 0
        return int(sum(p.numel() for p in self._module.parameters()))

    # -- interface ---------------------------------------------------------- #

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TorchWindowForecaster":
        import torch

        self._spec = channel_window(X, self.lookback)
        w = self._spec.window(X)
        s = self._spec.static_block(X)
        ys = np.asarray(y, dtype=float).reshape(-1, 1)

        # Fit-window statistics only. Standardising on the test block's mean and scale
        # is the subtler of the two leaks a sequence baseline usually carries, and it is
        # the reason a naively written one looks strong.
        self._w_mu, self._w_sd = _fit_scaler(w, axis=(0, 1))
        self._s_mu, self._s_sd = _fit_scaler(s, axis=0) if s.shape[1] else (None, None)
        self._y_mu = ys.mean()
        self._y_sd = ys.std() if ys.std() > 0 else 1.0

        torch.manual_seed(self.seed)
        self._module = self.build_module(
            self._spec.n_channels, self._spec.lookback, len(self._spec.static)
        )

        wn, sn = self._tensors(X)
        wt = torch.tensor(wn, dtype=torch.float32)
        st = torch.tensor(sn, dtype=torch.float32)
        yt = torch.tensor((ys - self._y_mu) / self._y_sd, dtype=torch.float32)

        opt = torch.optim.Adam(
            self._module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
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


# --------------------------------------------------------------------------- #
# DLinear / NLinear  --  Zeng, Chen, Zhang & Xu, arXiv:2205.13504 (AAAI 2023)
# --------------------------------------------------------------------------- #

def _moving_average(x, kernel: int):
    """Centred moving average along time, edges held constant.

    This smooths *within the lookback window*, every step of which is already in the
    past relative to the target, so it is not a causality question -- no value at or
    after the forecast origin is involved. It is the series decomposition DLinear is
    built on, and treating it as a leak would be a misreading.
    """
    import torch
    import torch.nn.functional as F

    pad = (kernel - 1) // 2
    front = x[:, :1, :].repeat(1, pad, 1)
    back = x[:, -1:, :].repeat(1, pad, 1)
    padded = torch.cat([front, x, back], dim=1)
    return F.avg_pool1d(padded.transpose(1, 2), kernel_size=kernel, stride=1).transpose(1, 2)


@dataclass
class DLinear(TorchWindowForecaster):
    """Decomposition into trend and remainder, one linear map each, plus exogenous.

    Roughly ``C * L * 2`` parameters -- a few hundred at our sizes, against ~15k for the
    TCN and ~25k for the Transformer. Zeng et al.'s point is that on real forecasting
    data this is frequently enough, and at ~350-800 training rows per fold it is the
    model whose capacity best matches the evidence available.

    The exogenous term is the standard linear extension: the known-at-origin and static
    block gets its own linear map, summed with the two window terms.
    """

    kernel: int = 25
    name: str = "dlinear"

    def build_module(self, n_channels: int, lookback: int, n_static: int):
        import torch
        import torch.nn as nn

        # The kernel must be odd and no longer than the window it smooths.
        kernel = min(self.kernel, lookback if lookback % 2 else lookback - 1)
        kernel = max(3, kernel if kernel % 2 else kernel - 1)

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.kernel = kernel
                # One weight per (channel, lag) for each component, as in the paper's
                # channel-independent formulation.
                self.w_trend = nn.Parameter(torch.zeros(n_channels, lookback))
                self.w_season = nn.Parameter(torch.zeros(n_channels, lookback))
                nn.init.normal_(self.w_trend, std=1.0 / lookback)
                nn.init.normal_(self.w_season, std=1.0 / lookback)
                self.bias = nn.Parameter(torch.zeros(1))
                self.exog = nn.Linear(n_static, 1) if n_static else None

            def forward(self, window, static):
                trend = _moving_average(window, self.kernel)
                season = window - trend
                out = (
                    torch.einsum("nlc,cl->n", trend, self.w_trend)
                    + torch.einsum("nlc,cl->n", season, self.w_season)
                ).unsqueeze(1) + self.bias
                if self.exog is not None and static.shape[1]:
                    out = out + self.exog(static)
                return out

        return Module()


@dataclass
class NLinear(TorchWindowForecaster):
    """Normalise by the last observation, one linear map, add it back.

    Zeng et al.'s second baseline, and the one that handles distribution shift between
    the fitting window and the test block. On a 55-day trace with a trend -- the same
    drift that makes static conformal calibration under-cover here -- that subtraction
    is doing real work, so it is worth having both variants in the table.
    """

    name: str = "nlinear"

    def build_module(self, n_channels: int, lookback: int, n_static: int):
        import torch
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(n_channels, lookback))
                nn.init.normal_(self.w, std=1.0 / lookback)
                self.bias = nn.Parameter(torch.zeros(1))
                self.exog = nn.Linear(n_static, 1) if n_static else None

            def forward(self, window, static):
                last = window[:, -1:, :]
                # Centre the DEMAND channel only. The paper's normalisation exists to
                # absorb level shift in the series being forecast; applying it to a
                # covariate channel destroys it, because a flag that is constant across
                # the window (which most of them are) centres to exactly zero and the
                # model can never see it. That is not a hypothetical -- it made
                # `is_rain` invisible to this model in testing.
                centred = window.clone()
                centred[:, :, 0] = window[:, :, 0] - last[:, :, 0]
                out = torch.einsum("nlc,cl->n", centred, self.w).unsqueeze(1)
                # Adding the last demand value back is what makes this a correction to
                # persistence rather than a forecast from zero.
                out = out + last[:, 0, 0:1] + self.bias
                if self.exog is not None and static.shape[1]:
                    out = out + self.exog(static)
                return out

        return Module()


# --------------------------------------------------------------------------- #
# TCN  --  Bai, Kolter & Koltun, arXiv:1803.01271
# --------------------------------------------------------------------------- #

@dataclass
class TCN(TorchWindowForecaster):
    """Dilated causal convolutions with residual blocks.

    Receptive field is ``1 + 2 * (k - 1) * (2**levels - 1)``: at k=3 and 4 levels that is
    61 steps, comfortably covering a 24-lag window, for ~15k parameters. Causality is
    enforced by padding on the left and chopping the right-hand overhang -- the
    ``_Chomp`` below. Getting that wrong leaks the future silently and shows up only as
    an implausibly good RMSE, which is exactly the failure mode this project was created
    to fix, so it carries its own test gate.
    """

    channels: int = 32
    levels: int = 4
    kernel: int = 3
    name: str = "tcn"

    def build_module(self, n_channels: int, lookback: int, n_static: int):
        import torch
        import torch.nn as nn

        kernel, levels, width = self.kernel, self.levels, self.channels
        dropout = self.dropout

        class Chomp(nn.Module):
            """Remove the right-hand padding, which is what makes the conv causal."""

            def __init__(self, size):
                super().__init__()
                self.size = size

            def forward(self, x):
                return x[:, :, : -self.size] if self.size else x

        class Block(nn.Module):
            def __init__(self, c_in, c_out, dilation):
                super().__init__()
                pad = (kernel - 1) * dilation
                self.net = nn.Sequential(
                    nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation),
                    Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                    nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation),
                    Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                )
                self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
                self.relu = nn.ReLU()

            def forward(self, x):
                res = x if self.down is None else self.down(x)
                return self.relu(self.net(x) + res)

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                blocks, c_in = [], n_channels
                for level in range(levels):
                    blocks.append(Block(c_in, width, dilation=2 ** level))
                    c_in = width
                self.blocks = nn.Sequential(*blocks)
                self.out = nn.Linear(width + n_static, 1)

            def forward(self, window, static):
                h = self.blocks(window.transpose(1, 2))   # (N, C, L)
                features = h[:, :, -1]                    # last step only
                if static.shape[1]:
                    features = torch.cat([features, static], dim=1)
                return self.out(features)

        return Module()


# --------------------------------------------------------------------------- #
# Transformer  --  Vaswani et al., arXiv:1706.03762
# --------------------------------------------------------------------------- #

@dataclass
class TransformerForecaster(TorchWindowForecaster):
    """A small pre-norm encoder over the lookback, with the attention map retained.

    Each of the ``lookback`` tokens is a linear embedding of **all** channels at that
    timestamp, so a context flag is inside every token rather than bolted on at the end.
    Kept deliberately small (d_model 32, 2 heads, 2 layers, ~25k parameters): the point
    is to test whether attention over lag positions buys anything at this sample size,
    and a model with more parameters than training rows would answer a different
    question.

    :attr:`attention_by_lag` is the mean attention the final layer's last query pays to
    each lag position -- the figure that says which past steps the model decided
    mattered, which is checkable against the measured daily period.
    """

    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    name: str = "transformer"

    def build_module(self, n_channels: int, lookback: int, n_static: int):
        import torch
        import torch.nn as nn

        d_model, n_heads, n_layers, dropout = (
            self.d_model, self.n_heads, self.n_layers, self.dropout
        )

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(d_model)
                self.attn = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )
                self.norm2 = nn.LayerNorm(d_model)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, d_model * 2), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
                )
                self.last_weights = None

            def forward(self, x):
                h = self.norm1(x)
                # No causal mask: every position in the window is already in the past
                # relative to the target, so the tokens may attend to one another
                # freely. A mask here would model a different problem.
                a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
                self.last_weights = w.detach()
                x = x + a
                return x + self.ff(self.norm2(x))

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(n_channels, d_model)
                self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
                nn.init.normal_(self.pos, std=0.02)
                self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])
                self.norm = nn.LayerNorm(d_model)
                self.out = nn.Linear(d_model + n_static, 1)

            def forward(self, window, static):
                h = self.embed(window) + self.pos
                for layer in self.layers:
                    h = layer(h)
                features = self.norm(h)[:, -1, :]      # the most recent token's view
                if static.shape[1]:
                    features = torch.cat([features, static], dim=1)
                return self.out(features)

        return Module()

    @property
    def attention_by_lag(self) -> np.ndarray | None:
        """Mean attention from the most recent position to each lag, oldest first."""
        if self._module is None:
            return None
        w = self._module.layers[-1].last_weights
        if w is None:
            return None
        return w[:, -1, :].mean(axis=0).numpy()


# --------------------------------------------------------------------------- #
# N-BEATS  --  Oreshkin, Carpov, Chapados & Bengio, arXiv:1905.10437
# NBEATSx  --  Olivares, Challu, Marcjasz, Weron & Dubrawski, arXiv:2104.05522
# --------------------------------------------------------------------------- #

@dataclass
class NBeats(TorchWindowForecaster):
    """Generic N-BEATS: stacked blocks with backward and forward residual links.

    Simplifications, stated plainly because they matter to how the result should be
    read. The paper's configuration is millions of parameters trained on thousands of
    M4 series; ours is ~50k on one trace of ~900 points, so the stack is shortened and
    narrowed. The forecast horizon here is a single step, which makes the paper's
    *interpretable* trend and seasonality bases degenerate on the forecast side -- a
    polynomial through one point is a constant. The decomposition that survives and is
    worth plotting is therefore the **backcast** over the 24 lags, exposed as
    :attr:`backcast_by_block`.

    This variant is **univariate by design and is reported as a labelled control only**;
    :class:`NBeatsX` is the member of the pair that sees the covariates.
    """

    width: int = 64
    n_blocks: int = 3
    block_layers: int = 3
    #: Width the exogenous block is projected to before any block reads it.
    exog_dim: int = 32
    name: str = "nbeats"
    #: Kept univariate on purpose -- see the class docstring.
    use_covariates: bool = False

    def build_module(self, n_channels: int, lookback: int, n_static: int):
        import torch
        import torch.nn as nn

        width, n_blocks, layers = self.width, self.n_blocks, self.block_layers
        raw_exog = (n_channels - 1) * lookback + n_static if self.use_covariates else 0
        # NBEATSx passes the exogenous inputs through an encoder before the blocks see
        # them. Ours is a single projection, which matters at this sample size: feeding
        # 385 raw exogenous inputs straight into three blocks costs ~108k parameters
        # against ~350 training rows per fold, and the model is then fitting noise
        # rather than the covariates.
        exog_width = self.exog_dim if raw_exog else 0

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                dims, stack = lookback + exog_width, []
                for _ in range(layers):
                    stack += [nn.Linear(dims, width), nn.ReLU()]
                    dims = width
                self.stack = nn.Sequential(*stack)
                self.backcast = nn.Linear(width, lookback)
                self.forecast = nn.Linear(width, 1)

            def forward(self, x, exog):
                h = self.stack(torch.cat([x, exog], dim=1) if exog.shape[1] else x)
                return self.backcast(h), self.forecast(h)

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
                self.encoder = (
                    nn.Sequential(nn.Linear(raw_exog, exog_width), nn.ReLU())
                    if raw_exog else None
                )
                self.last_backcasts = None

            def forward(self, window, static):
                residual = window[:, :, 0]                 # demand channel
                if self.encoder is not None:
                    exog = self.encoder(torch.cat(
                        [window[:, :, 1:].reshape(len(window), -1), static], dim=1
                    ))
                else:
                    exog = window[:, :0, 0]
                forecast = torch.zeros(len(window), 1, device=window.device)
                traces = []
                for block in self.blocks:
                    back, fore = block(residual, exog)
                    residual = residual - back
                    forecast = forecast + fore
                    traces.append(back.detach())
                self.last_backcasts = traces
                return forecast

        return Module()

    @property
    def backcast_by_block(self) -> np.ndarray | None:
        """Mean backcast per block over the lookback -- the interpretable piece."""
        if self._module is None or self._module.last_backcasts is None:
            return None
        return np.stack([b.mean(axis=0).numpy() for b in self._module.last_backcasts])


@dataclass
class NBeatsX(NBeats):
    """N-BEATS with an exogenous block: the covariate channels enter every block.

    Olivares et al. report roughly a 20% improvement over plain N-BEATS from exogenous
    variables alone on electricity prices. Whether that carries to nine hand-labelled
    context flags on a 55-day cellular trace is exactly the open question, and running
    this against :class:`NBeats` on one fold schedule is how it gets answered.
    """

    name: str = "nbeatsx"
    use_covariates: bool = True


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def modern_models(lookback: int = 24, epochs: int = 60, seed: int = 42,
                  include: tuple[str, ...] = MODERN_ARCHITECTURES) -> list[Forecaster]:
    """One instance of each architecture, at the defaults documented above."""
    factories = {
        "dlinear": lambda: DLinear(lookback=lookback, epochs=epochs, seed=seed),
        "nlinear": lambda: NLinear(lookback=lookback, epochs=epochs, seed=seed),
        "tcn": lambda: TCN(lookback=lookback, epochs=epochs, seed=seed),
        "transformer": lambda: TransformerForecaster(
            lookback=lookback, epochs=epochs, seed=seed),
        "nbeats": lambda: NBeats(lookback=lookback, epochs=epochs, seed=seed),
        "nbeatsx": lambda: NBeatsX(lookback=lookback, epochs=epochs, seed=seed),
    }
    unknown = [k for k in include if k not in factories]
    if unknown:
        raise ValueError(f"Unknown architecture(s) {unknown}; expected "
                         f"{sorted(factories)}.")
    return [factories[k]() for k in include]


def channel_permutation_importance(
    model: Forecaster,
    X: pd.DataFrame,
    y: pd.Series,
    spec: ChannelWindow,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """How much RMSE each input channel is worth, by shuffling it across the window.

    The direct answer to "is this model really using more than the demand lags?".
    A channel whose permutation costs nothing is a channel the model ignored, and on
    these traces that is a live possibility worth measuring rather than assuming --
    the audit found most context flags inert on the *level* of demand.

    Permutation importance follows Breiman (2001); here the unit permuted is a whole
    channel, all lags together, because shuffling one lag of one flag while leaving its
    neighbours intact would leave the information almost entirely recoverable.
    """
    rng = np.random.default_rng(seed)
    truth = np.asarray(y, dtype=float)
    base = float(np.sqrt(np.mean((truth - model.predict(X)) ** 2)))

    rows = []
    for position, channel in enumerate(spec.channels):
        cols = list(spec.columns[position])
        scores = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[cols] = X[cols].to_numpy()[order]
            scores.append(
                float(np.sqrt(np.mean((truth - model.predict(shuffled)) ** 2)))
            )
        rows.append({
            "channel": channel,
            "rmse_base": base,
            "rmse_permuted": float(np.mean(scores)),
            "importance": float(np.mean(scores)) - base,
            "importance_sd": float(np.std(scores)),
        })
    out = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return out.reset_index(drop=True)
