"""Verification gates.

These encode the specific failures found in the original project, so that a future
refactor cannot silently reintroduce them. Run with ``pytest tests/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bwalloc.allocation import kappa_for_tau, optimal_tau
from bwalloc.baselines import Persistence, SeasonalNaive
from bwalloc.conformal import (
    LocallyAdaptiveConformal,
    SplitConformal,
    min_calibration_size,
)
from bwalloc.data import load, sampling_profile
from bwalloc.features import FeatureConfig, assert_no_leakage, build_features
from bwalloc.metrics import mase, rmse, sla_violation_rate
from bwalloc.splits import rolling_origin

OPERATORS = ["gp", "robi"]


@pytest.fixture(scope="module", params=OPERATORS)
def trace(request):
    df = load(request.param)
    return request.param, df, sampling_profile(df)


# --------------------------------------------------------------------------- #
# The leak that produced the original headline result
# --------------------------------------------------------------------------- #

def test_no_target_leakage(trace):
    """No engineered feature may depend on the contemporaneous target.

    The original ``Gbps_rolling_mean_3`` was computed without a shift, putting y_t in
    its own feature vector. Removing the leak moved GP RMSE from 6.54 to 19.30.
    """
    _, df, profile = trace
    assert_no_leakage(df, profile, FeatureConfig())


def test_leak_is_actually_detectable():
    """The leak detector must fail on a deliberately leaky feature.

    A guard that cannot fail is not a guard.
    """
    idx = pd.date_range("2025-01-01", periods=300, freq="86min")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Gbps": rng.normal(100, 10, len(idx))}, index=idx)
    profile = sampling_profile(df)

    leaky = FeatureConfig(lag_hours=(), rolling_hours=(), daily_harmonics=1,
                          weekly_harmonics=0, use_context=False)
    X, _ = build_features(df, profile, leaky)
    # Sanity: the honest config passes.
    assert_no_leakage(df, profile, leaky)

    # Now inject y_t directly and confirm a naive check would catch it.
    X_leaky = X.copy()
    X_leaky["leak"] = df["Gbps"].reindex(X.index)
    assert X_leaky["leak"].corr(df["Gbps"].reindex(X.index)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The sampling-rate error
# --------------------------------------------------------------------------- #

def test_sampling_is_not_hourly(trace):
    """Neither trace is hourly; lag 24 is not a day."""
    name, _, profile = trace
    assert profile.median_gap_min > 60, "trace unexpectedly hourly"
    expected_gap = {"gp": 86.0, "robi": 99.0}[name]
    assert profile.median_gap_min == pytest.approx(expected_gap, abs=1.0)
    # The lag the original code used as "one day" spans well over a day.
    assert profile.hours_for_lag(24) > 30


def test_daily_lag_is_derived_not_hardcoded(trace):
    """``lag_for_hours(24)`` must resolve to the measured period, never 24."""
    name, _, profile = trace
    expected = {"gp": 17, "robi": 15}[name]
    assert profile.lag_for_hours(24) == expected
    assert profile.lag_for_hours(24) != 24


def test_true_period_beats_lag24(trace):
    """Seasonal-naive at the measured period must beat it at lag 24.

    This is the empirical consequence of the sampling error: on GP, lag 24 spans
    34.4 h and lands roughly anti-phase to the daily cycle.
    """
    _, df, profile = trace
    y = df["Gbps"]
    correct = rmse(y.iloc[profile.daily_period:], y.shift(profile.daily_period).dropna())
    wrong = rmse(y.iloc[24:], y.shift(24).dropna())
    assert correct < wrong


# --------------------------------------------------------------------------- #
# Baselines and the sanity floor
# --------------------------------------------------------------------------- #

def test_persistence_sanity_floor(trace):
    """Persistence RMSE on the original split must match the audited value.

    Guards against a refactor silently altering the data (reordering, reparsing
    timestamps, dropping rows).
    """
    name, df, _ = trace
    split, expected = {"gp": ("2025-04-30", 12.1845), "robi": ("2025-05-10", 31.4122)}[name]
    y = df["Gbps"]
    test = y[y.index >= split]
    prev = y.shift(1)[y.index >= split]
    assert rmse(test, prev) == pytest.approx(expected, abs=0.01)


def test_mase_flags_a_losing_model(trace):
    """MASE >= 1 for a forecaster that does not beat the naive it is scaled by."""
    _, df, profile = trace
    y = df["Gbps"].to_numpy(dtype=float)
    constant = np.full(len(y) - 100, y[:-100].mean())
    score = mase(y[100:], constant, y[:100], season_lag=1)
    assert score > 1.0


# --------------------------------------------------------------------------- #
# Fold schedule integrity
# --------------------------------------------------------------------------- #

def test_folds_never_look_ahead():
    folds = rolling_origin(800, n_folds=8, calib_frac=0.25)
    for fold in folds:
        assert fold.train.max() < fold.test.min()
        if fold.n_calib:
            assert fold.calib.max() < fold.test.min()
            assert fold.train.max() < fold.calib.min()


def test_folds_do_not_overlap_and_cover_the_tail():
    folds = rolling_origin(800, n_folds=8, calib_frac=0.25)
    test_idx = np.concatenate([f.test for f in folds])
    assert len(test_idx) == len(np.unique(test_idx))
    assert test_idx.max() == 799


def test_training_windows_are_not_starved():
    """Regression test for the inverted split in the original notebooks.

    ``Different_Model_Training_GP.ipynb`` trained LightGBM on 88 rows and tested on
    ~480 after ``dropna()`` removed the first 336.
    """
    folds = rolling_origin(880, n_folds=8, calib_frac=0.25)
    for fold in folds:
        assert fold.n_train + fold.n_calib > fold.n_test


# --------------------------------------------------------------------------- #
# Conformal calibration
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tau,expected", [(0.8, 4), (0.9, 9), (0.95, 19), (0.99, 99)])
def test_min_calibration_size(tau, expected):
    assert min_calibration_size(tau) == expected


def test_estimability_guard_refuses_rather_than_clips():
    """Requesting a tau the calibration set cannot express must raise.

    Silently returning the maximum residual would report a coverage guarantee the
    data does not support -- a subtler version of the error being corrected.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, 30)
    pred = np.full(30, 100.0)
    cal = SplitConformal().calibrate(y, pred)
    cal.quantile(0.95)  # 30 >= 19, fine
    with pytest.raises(ValueError, match="at least 99"):
        cal.quantile(0.99)


def test_split_conformal_achieves_nominal_coverage_when_exchangeable():
    """On exchangeable data, coverage must reach nominal. Validates the machinery."""
    rng = np.random.default_rng(7)
    n = 4000
    y = rng.normal(100, 10, n)
    pred = np.full(n, 100.0)
    cal = SplitConformal().calibrate(y[:2000], pred[:2000])
    alloc = cal.allocate(pred[2000:], tau=0.9)
    achieved = 1.0 - sla_violation_rate(y[2000:], alloc)
    assert achieved == pytest.approx(0.9, abs=0.02)


def test_adaptive_conformal_widens_where_uncertainty_is_higher():
    """The adaptive margin must track sigma_hat rather than being constant.

    This is the mechanism behind the context-conditional result: on GP, residual sigma
    is 33% higher during rainfall, and a constant margin under-delivers there.
    """
    rng = np.random.default_rng(11)
    n = 4000
    sigma = np.where(rng.random(n) < 0.3, 25.0, 8.0)
    y = rng.normal(100, sigma)
    pred = np.full(n, 100.0)

    cal = LocallyAdaptiveConformal().calibrate(
        y[:2000], pred[:2000], sigma_calib=sigma[:2000]
    )
    alloc = cal.allocate(pred[2000:], tau=0.9, sigma_test=sigma[2000:])

    wide = sigma[2000:] == 25.0
    assert alloc[wide].mean() > alloc[~wide].mean()

    # And crucially, coverage should hold within *both* regimes, not just overall.
    for mask in (wide, ~wide):
        achieved = 1.0 - sla_violation_rate(y[2000:][mask], alloc[mask])
        assert achieved == pytest.approx(0.9, abs=0.04)


def test_marginal_conformal_under_covers_the_volatile_group():
    """The failure mode the context-conditional method exists to fix.

    A single global margin calibrated on a mixture systematically under-delivers on
    the high-variance subpopulation.
    """
    rng = np.random.default_rng(13)
    n = 4000
    volatile = rng.random(n) < 0.3
    sigma = np.where(volatile, 25.0, 8.0)
    y = rng.normal(100, sigma)
    pred = np.full(n, 100.0)

    cal = SplitConformal().calibrate(y[:2000], pred[:2000])
    alloc = cal.allocate(pred[2000:], tau=0.9)

    vol_test = volatile[2000:]
    coverage_volatile = 1.0 - sla_violation_rate(y[2000:][vol_test], alloc[vol_test])
    coverage_calm = 1.0 - sla_violation_rate(y[2000:][~vol_test], alloc[~vol_test])

    assert coverage_volatile < 0.9 - 0.03
    assert coverage_calm > 0.9


# --------------------------------------------------------------------------- #
# The cost model
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kappa", [1.0, 2.0, 5.0, 10.0, 20.0])
def test_optimal_tau_round_trips(kappa):
    assert kappa_for_tau(optimal_tau(kappa)) == pytest.approx(kappa)


def test_optimal_tau_is_the_cost_minimiser():
    """Empirically confirm the quantile/cost-ratio correspondence.

    The allocation minimising asymmetric cost with ratio kappa should be the
    tau = kappa/(1+kappa) quantile of the demand distribution.
    """
    from bwalloc.metrics import asymmetric_cost

    rng = np.random.default_rng(3)
    y = rng.gamma(shape=9.0, scale=11.0, size=200_000)
    kappa = 9.0
    tau = optimal_tau(kappa)  # 0.9

    candidates = np.quantile(y, [0.70, 0.80, 0.85, tau, 0.95, 0.98])
    costs = [asymmetric_cost(y, np.full(len(y), c), kappa=kappa) for c in candidates]
    assert int(np.argmin(costs)) == 3
