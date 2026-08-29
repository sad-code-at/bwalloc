"""Significance testing for forecast comparisons.

With ~300 test points per fold, RMSE gaps of one or two Gbps between models are not
distinguishable from noise. The original project ranked ten models on exactly such
gaps -- and across inconsistent splits -- without any test. Nothing in this package
reports "model A beats model B" without a p-value attached.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st


def diebold_mariano(
    y_true,
    pred_a,
    pred_b,
    horizon: int = 1,
    loss: str = "squared",
    harvey_correction: bool = True,
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Tests H0: the two forecasts have equal expected loss. A negative statistic favours
    ``pred_a``; a positive one favours ``pred_b``.

    Parameters
    ----------
    horizon:
        Forecast horizon. For h-step forecasts the loss differential is autocorrelated
        up to lag h-1, so the variance uses a Newey-West style truncation at h-1.
    loss:
        ``"squared"`` (compares RMSE) or ``"absolute"`` (compares MAE).
    harvey_correction:
        Apply the Harvey-Leybourne-Newbold small-sample correction and use a
        t-distribution rather than a normal. Strongly recommended at these sample
        sizes -- without it the test is materially over-sized.

    Returns
    -------
    (statistic, p_value)
        Two-sided p-value.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    ea = yt - np.asarray(pred_a, dtype=float).ravel()
    eb = yt - np.asarray(pred_b, dtype=float).ravel()

    if loss == "squared":
        d = ea**2 - eb**2
    elif loss == "absolute":
        d = np.abs(ea) - np.abs(eb)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    n = len(d)
    if n < 8:
        raise ValueError(f"Too few observations for a meaningful DM test (n={n}).")

    d_bar = float(np.mean(d))
    d_dev = d - d_bar

    # Long-run variance: autocovariances up to lag h-1 enter for an h-step forecast.
    gamma0 = float(np.dot(d_dev, d_dev) / n)
    var = gamma0
    for lag in range(1, horizon):
        cov = float(np.dot(d_dev[lag:], d_dev[:-lag]) / n)
        var += 2.0 * cov

    if var <= 0:
        # Can happen when the two forecasts are near-identical; treat as no evidence.
        return 0.0, 1.0

    stat = d_bar / np.sqrt(var / n)

    if harvey_correction:
        k = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        stat *= np.sqrt(max(k, 1e-12))
        p = 2.0 * (1.0 - _st.t.cdf(abs(stat), df=n - 1))
    else:
        p = 2.0 * (1.0 - _st.norm.cdf(abs(stat)))

    return float(stat), float(p)


def paired_bootstrap_rmse(
    y_true, pred_a, pred_b, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap the RMSE difference (a - b) and its two-sided p-value.

    A distribution-free companion to :func:`diebold_mariano`. Resamples test points
    with replacement; use the moving-block variant if serial correlation in the errors
    is severe.

    Returns
    -------
    (mean_rmse_difference, p_value)
        A negative difference favours ``pred_a``.
    """
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true, dtype=float).ravel()
    a = np.asarray(pred_a, dtype=float).ravel()
    b = np.asarray(pred_b, dtype=float).ravel()
    n = len(yt)

    def _rmse(y, p):
        return np.sqrt(np.mean((y - p) ** 2))

    observed = _rmse(yt, a) - _rmse(yt, b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = _rmse(yt[idx], a[idx]) - _rmse(yt[idx], b[idx])

    # p-value for H0: no difference, via the centred bootstrap distribution.
    centred = diffs - diffs.mean()
    p = float(np.mean(np.abs(centred) >= abs(observed)))
    return float(observed), p


def benjamini_hochberg(p_values, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Comparing ten models pairwise is 45 tests; without correction, two or three
    spurious "significant" wins at alpha=0.05 are expected by construction.

    Returns
    -------
    numpy.ndarray of bool
        True where the corresponding hypothesis is rejected at FDR ``alpha``.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    n = len(p)
    order = np.argsort(p)
    thresholds = alpha * np.arange(1, n + 1) / n
    passed = p[order] <= thresholds
    rejected = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))
        rejected[order[: cutoff + 1]] = True
    return rejected
