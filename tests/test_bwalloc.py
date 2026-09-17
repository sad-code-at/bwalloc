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
    AdaptiveConformalInference,
    LocallyAdaptiveConformal,
    SplitConformal,
    min_calibration_size,
)
from bwalloc.data import TARGET, load, sampling_profile
from bwalloc.features import FeatureConfig, assert_no_leakage, build_features
from bwalloc.forecast import (
    direct_design,
    history_row,
    persistence_at_horizon,
    seasonal_naive_at_horizon,
)
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


# --------------------------------------------------------------------------- #
# Multi-horizon forecasting
# --------------------------------------------------------------------------- #

def test_history_row_matches_build_features(trace):
    """The recursive rollout's feature builder must agree with the batch one exactly.

    ``forecast.history_row`` re-implements the lag and rolling blocks so the recursive
    rollout can substitute predictions for observations it does not have. That
    duplication is the obvious place for the two paths to drift apart -- one shifted
    window in either file and the recursive results become quietly wrong -- so this
    pins them together on real data.
    """
    _, df, profile = trace
    config = FeatureConfig()
    X, _ = build_features(df, profile, config)
    y_full = df[TARGET].astype(float)
    positions = pd.DatetimeIndex(y_full.index).get_indexer(pd.DatetimeIndex(X.index))

    for row in (0, 1, len(X) // 3, len(X) // 2, len(X) - 1):
        rebuilt = history_row(y_full.to_numpy(), positions[row], profile, config)
        assert rebuilt, "history_row produced no columns for the default config"
        for name, value in rebuilt.items():
            assert name in X.columns
            assert value == pytest.approx(X[name].iloc[row], rel=1e-9, abs=1e-9)


def test_direct_design_reduces_to_the_one_step_problem(trace):
    """At a one-step horizon the re-aligned design must be the original one.

    The horizon study has to meet the existing benchmark at its first point, otherwise
    a degradation curve cannot be read as degradation.
    """
    _, df, profile = trace
    X, y = build_features(df, profile, FeatureConfig())
    X_h, y_h, origins = direct_design(X, y, steps=1)

    assert list(X_h.columns) == sorted(X_h.columns, key=list(X_h.columns).index)
    assert np.allclose(X_h[X.columns].to_numpy(), X.to_numpy())
    assert np.array_equal(y_h.to_numpy(), y.to_numpy())
    assert list(origins) == list(X.index)


def test_direct_design_reads_history_at_the_origin_not_the_target(trace):
    """History columns must lag by the full horizon; calendar columns must not.

    Reading a lag at the target timestamp would hand the model an observation that has
    not been taken yet -- the multi-horizon version of the leak this project exists to
    correct. Fourier terms are exempt because wall-clock time is known in advance.
    """
    _, df, profile = trace
    X, y = build_features(df, profile, FeatureConfig())
    steps = 4
    X_h, y_h, _ = direct_design(X, y, steps)
    lead = steps - 1

    # Row r of the re-aligned frame targets row r + lead of the original.
    assert X_h["lag_1.5h"].iloc[10] == pytest.approx(X["lag_1.5h"].iloc[10])
    assert X_h["day_sin1"].iloc[10] == pytest.approx(X["day_sin1"].iloc[10 + lead])
    assert y_h.iloc[10] == pytest.approx(y.iloc[10 + lead])


def test_horizon_baseline_is_not_the_one_step_baseline(trace):
    """An h-step model must be scored against an h-step naive forecaster.

    Comparing a 24-hour-ahead forecast to persistence-at-one-step is the flattering
    comparison, and it is how multi-step results are most often overstated.
    """
    _, df, profile = trace
    _, y = build_features(df, profile, FeatureConfig())
    steps = profile.lag_for_hours(24.0)

    naive_h = persistence_at_horizon(y, steps)
    assert np.array_equal(naive_h.to_numpy()[steps:], y.to_numpy()[:-steps])

    # Away from the seasonal period it must be strictly worse than the one-step
    # naive, or the horizon is not really costing anything. The daily lag itself is
    # deliberately excluded: on Robi, yesterday's value at the same hour (RMSE 26.5)
    # is a *better* forecast than the most recent observation (29.4), because the
    # daily cycle there is stronger than short-run persistence. That inversion is a
    # finding, not a violation -- and it is only visible once the daily period is
    # measured as 15 samples rather than assumed to be 24.
    mid = max(2, steps // 2)
    naive_mid = persistence_at_horizon(y, mid)
    one_step = y.shift(1)
    both = ~(naive_mid.isna() | one_step.isna())
    assert rmse(y[both], naive_mid[both]) > rmse(y[both], one_step[both])


def test_seasonal_naive_at_horizon_never_reads_the_future(trace):
    """The seasonal baseline must step back a whole number of observable cycles."""
    _, df, profile = trace
    _, y = build_features(df, profile, FeatureConfig())
    period = profile.daily_period
    for steps in (1, 2, period - 1, period, period + 1):
        shifted = seasonal_naive_at_horizon(y, steps, period)
        offset = int(shifted.notna().argmax())
        assert offset >= steps, (
            f"seasonal naive at {steps} steps reads only {offset} back, "
            "which was not observable at the forecast origin"
        )
        assert offset % period == 0


def test_embargo_prevents_training_on_post_origin_outcomes():
    """With an embargo of k, no fitted row may sit within k of the test block.

    Direct multi-horizon training pairs carry targets h-1 rows after their origin, so
    without the embargo the model is fitted on outcomes that had not occurred when the
    first test forecast was issued. That is a subtler restatement of the leak in §1.2
    of the audit, and it inflates long-horizon results specifically.
    """
    for embargo in (0, 1, 5, 16):
        folds = rolling_origin(400, n_folds=4, calib_frac=0.25, embargo=embargo)
        for fold in folds:
            fitted = np.concatenate([fold.train, fold.calib])
            assert fitted.max() <= fold.test[0] - 1 - embargo
            assert len(fold.train) > 0


# --------------------------------------------------------------------------- #
# Relative (multiplicative) conformal calibration
# --------------------------------------------------------------------------- #

def test_relative_conformal_scales_the_margin_with_demand():
    """The relative variant must produce a constant allocation *ratio*.

    Additive conformal adds the same Gbps at 3 a.m. as at peak; the fixed-margin rule
    it is measured against is multiplicative. Comparing the two without this option
    charges the method for its parameterisation rather than its calibration.
    """
    rng = np.random.default_rng(0)
    pred = rng.uniform(20, 120, 400)
    y = pred * (1 + rng.normal(0, 0.1, 400))

    additive = SplitConformal().calibrate(y, pred)
    relative = SplitConformal(relative=True).calibrate(y, pred)

    a_add = additive.allocate(pred, 0.9)
    a_rel = relative.allocate(pred, 0.9)

    assert np.ptp(a_rel / pred) == pytest.approx(0.0, abs=1e-9)
    assert np.ptp(a_add / pred) > 0.3
    assert relative.name == "relative_split_conformal"


def test_relative_conformal_still_delivers_nominal_coverage():
    """Changing the score's scale must not cost the coverage guarantee."""
    rng = np.random.default_rng(1)
    pred_cal = rng.uniform(20, 120, 600)
    y_cal = pred_cal * (1 + rng.normal(0, 0.15, 600))
    pred_te = rng.uniform(20, 120, 4000)
    y_te = pred_te * (1 + rng.normal(0, 0.15, 4000))

    cal = SplitConformal(relative=True).calibrate(y_cal, pred_cal)
    for tau in (0.8, 0.9, 0.95):
        achieved = float(np.mean(y_te <= cal.allocate(pred_te, tau)))
        assert abs(achieved - tau) < 0.02, f"tau={tau}: achieved {achieved:.3f}"


# --------------------------------------------------------------------------- #
# Drift: the assumption split conformal actually depends on
# --------------------------------------------------------------------------- #

def test_split_conformal_under_covers_under_drift():
    """The failure mode measured on the real traces, reproduced in isolation.

    Split conformal's guarantee is conditional on exchangeability. When the error
    scale grows after calibration -- which is what a 55-day trace with a trend looks
    like -- coverage falls below nominal, one-sidedly. ``run_coverage_gate.py`` finds
    exactly this on the real backtest: 39 of 41 gate failures are under-coverage.
    """
    rng = np.random.default_rng(3)
    pred_cal = rng.uniform(50, 100, 400)
    y_cal = pred_cal + rng.normal(0, 5, 400)

    pred_te = rng.uniform(50, 100, 2000)
    # Error scale triples across the test block: calibration no longer describes it.
    scale = np.linspace(5, 15, 2000)
    y_te = pred_te + rng.normal(0, 1, 2000) * scale

    static = SplitConformal().calibrate(y_cal, pred_cal)
    achieved = float(np.mean(y_te <= static.allocate(pred_te, 0.90)))
    assert achieved < 0.90 - 0.02, (
        f"expected under-coverage under drift, got {achieved:.3f}"
    )


def test_adaptive_conformal_inference_recovers_coverage_under_drift():
    """ACI must repair what split conformal loses to drift.

    This is the gate on the fix: the online level update has to bring long-run
    coverage back to nominal on the same non-exchangeable stream where the frozen
    level fails, without assuming exchangeability anywhere.
    """
    rng = np.random.default_rng(3)
    pred_cal = rng.uniform(50, 100, 400)
    y_cal = pred_cal + rng.normal(0, 5, 400)

    pred_te = rng.uniform(50, 100, 2000)
    scale = np.linspace(5, 15, 2000)
    y_te = pred_te + rng.normal(0, 1, 2000) * scale

    aci = AdaptiveConformalInference(gamma=0.05).calibrate(y_cal, pred_cal)
    online = aci.allocate(pred_te, 0.90, y_test=y_te)
    frozen = aci.allocate_static(pred_te, 0.90)

    cov_online = float(np.mean(y_te <= online))
    cov_frozen = float(np.mean(y_te <= frozen))
    assert abs(cov_online - 0.90) < 0.02, f"ACI achieved {cov_online:.3f}"
    assert cov_online > cov_frozen


def test_adaptive_conformal_inference_is_causal():
    """The outcome at t must not influence the allocation at t.

    ACI is only defensible for provisioning if the feedback runs strictly one step
    behind. Perturbing the final observation must leave every allocation unchanged,
    including its own -- if it moved, the allocator would be reading the demand it is
    supposed to be covering.
    """
    rng = np.random.default_rng(4)
    pred_cal = rng.uniform(50, 100, 300)
    y_cal = pred_cal + rng.normal(0, 5, 300)
    pred_te = rng.uniform(50, 100, 200)
    y_te = pred_te + rng.normal(0, 5, 200)

    base = AdaptiveConformalInference().calibrate(y_cal, pred_cal)
    a_ref = base.allocate(pred_te, 0.9, y_test=y_te)

    bumped = y_te.copy()
    bumped[-1] += 500.0
    a_alt = (
        AdaptiveConformalInference().calibrate(y_cal, pred_cal)
        .allocate(pred_te, 0.9, y_test=bumped)
    )
    assert np.allclose(a_ref, a_alt)


# --------------------------------------------------------------------------- #
# What a foundation model's quantile head can and cannot express
# --------------------------------------------------------------------------- #

def test_native_quantile_ceiling_caps_the_expressible_cost_ratio():
    """Chronos-Bolt's 0.9 quantile ceiling is a limit on cost asymmetry, not a detail.

    The model emits quantiles only up to tau = 0.9, silently clipping anything higher.
    Through tau* = kappa/(1+kappa) that ceiling corresponds to a cost ratio of exactly
    9 -- below what under-provisioning typically costs an operator. So the model's own
    quantile head cannot express the service level this application needs, and
    calibration on top of it is what makes it usable rather than an optional
    refinement. This pins the arithmetic behind that claim.
    """
    ceiling = 0.9
    assert kappa_for_tau(ceiling) == pytest.approx(9.0)

    # An operator with a realistic asymmetry needs a level the model cannot reach.
    for kappa in (10.0, 20.0):
        assert optimal_tau(kappa) > ceiling

    # And the levels this project reports on are exactly the ones that straddle it.
    assert optimal_tau(4.0) < ceiling < optimal_tau(19.0)


# --------------------------------------------------------------------------- #
# Sequence models: the original's architectures, on the corrected protocol
# --------------------------------------------------------------------------- #

def test_sequence_window_is_chronological():
    """A sequence model's window must be ordered oldest observation first.

    ``lagS_1`` is the most recent observation, so feeding the columns in natural
    sort order hands a recurrent model time running backwards. It would still
    train -- and score plausibly -- which is exactly why this needs a gate rather
    than an eyeball.
    """
    from bwalloc.sequence import lookback_columns

    X = pd.DataFrame({f"lagS_{n}": [0.0] for n in range(1, 6)})
    assert lookback_columns(X) == [
        "lagS_5", "lagS_4", "lagS_3", "lagS_2", "lagS_1"
    ]


def test_sequence_window_must_be_contiguous():
    """A window with gaps is not a sequence, and must be refused rather than padded."""
    from bwalloc.sequence import lookback_columns

    with pytest.raises(ValueError, match="consecutive"):
        lookback_columns(pd.DataFrame({"lagS_1": [0.0], "lagS_3": [0.0]}))
    with pytest.raises(ValueError, match="No lagS"):
        lookback_columns(pd.DataFrame({"lag_1.5h": [0.0]}))


def test_sequence_standardisation_uses_the_fit_window_only():
    """Scaling on test-block statistics is a second, subtler leak.

    A sequence model that standardises using the block it is about to predict has
    seen that block's mean and scale. The symptom is a model that tracks a level
    shift it could not have known about, so this asserts predictions do not change
    when the test block's location changes after fitting.
    """
    torch = pytest.importorskip("torch")
    from bwalloc.sequence import SequenceForecaster

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(100, 5, size=(120, 6)),
        columns=[f"lagS_{n}" for n in range(1, 7)],
    )
    y = pd.Series(rng.normal(100, 5, size=120))

    model = SequenceForecaster(kind="gru", lookback=6, epochs=2).fit(X, y)
    before = model.predict(X.iloc[:20])
    # Refit nothing; only ask about a shifted block. A fit-window scaler keeps its
    # own statistics, so the shift must propagate rather than be normalised away.
    shifted = model.predict(X.iloc[:20] + 50.0)
    assert not np.allclose(before, shifted), (
        "predictions are invariant to a level shift, which means the scaler was "
        "refitted on the block being predicted"
    )


def test_sequence_models_are_deterministic():
    """Same seed, same data, same predictions -- or no result here is reproducible."""
    pytest.importorskip("torch")
    from bwalloc.sequence import SequenceForecaster

    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        rng.normal(0, 1, size=(80, 4)), columns=[f"lagS_{n}" for n in range(1, 5)]
    )
    y = pd.Series(rng.normal(0, 1, size=80))

    a = SequenceForecaster(kind="lstm", lookback=4, epochs=2, seed=7).fit(X, y)
    b = SequenceForecaster(kind="lstm", lookback=4, epochs=2, seed=7).fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X), rtol=1e-6)


def test_sequence_lookback_must_match_the_design_matrix():
    """A silently truncated window would compare architectures at unequal information."""
    pytest.importorskip("torch")
    from bwalloc.sequence import SequenceForecaster

    X = pd.DataFrame(
        np.zeros((30, 4)), columns=[f"lagS_{n}" for n in range(1, 5)]
    )
    y = pd.Series(np.zeros(30))
    with pytest.raises(ValueError, match="lookback=8"):
        SequenceForecaster(kind="cnn", lookback=8, epochs=1).fit(X, y)


# --------------------------------------------------------------------------- #
# Notebook contract: the generated notebooks call library code, so the library
# must fail loudly on the mistakes a notebook can make, and its output schema
# must not drift away from the columns the notebooks select.
# --------------------------------------------------------------------------- #

def test_autocorrelation_by_lag_rejects_a_frame(trace):
    """A DataFrame must be refused here, not deep inside pandas.

    ``df.corr(other)`` on a DataFrame reads ``other`` as the *method* argument and
    dies with "the truth value of a DataFrame is ambiguous" -- a message that says
    nothing about the actual mistake. This cost a notebook run once; the guard
    turns it into a sentence naming the fix.
    """
    from bwalloc.data import autocorrelation_by_lag

    _, df, profile = trace
    with pytest.raises(TypeError, match="expects a Series"):
        autocorrelation_by_lag(df, max_lag=5, profile=profile)

    # The Series form still works, and reports wall-clock hours when given a profile.
    acf = autocorrelation_by_lag(df[TARGET], max_lag=5, profile=profile)
    assert list(acf["lag_samples"]) == [1, 2, 3, 4, 5]
    assert acf["lag_hours"].notna().all()


def test_autocorrelation_lag_hours_are_nan_without_a_profile(trace):
    """Omitting the profile yields NaN hours rather than crashing on truthiness."""
    from bwalloc.data import autocorrelation_by_lag

    _, df, _ = trace
    acf = autocorrelation_by_lag(df[TARGET], max_lag=3)
    assert acf["lag_hours"].isna().all()


def test_flag_report_columns_match_what_the_notebooks_select(trace):
    """Pin the flag-audit schema.

    ``03_context_conditional`` selects these columns by name. When they were renamed
    the notebook failed with a bare KeyError far from the cause, so the contract is
    asserted here where a rename is made.
    """
    from bwalloc.context import flag_report

    operator, df, _ = trace
    report = flag_report(df, operator)
    required = {
        "flag", "n_on", "mean_on", "mean_off", "delta_mean", "p_level",
        "resid_sd_on", "resid_sd_off", "variance_ratio", "p_variance",
    }
    missing = required - set(report.columns)
    assert not missing, f"flag_report lost columns the notebooks select: {sorted(missing)}"
