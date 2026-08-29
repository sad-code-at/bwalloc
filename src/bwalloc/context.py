"""Context flags: validation, disjoint grouping, and the uncertainty model.

The GP trace carries nine hand-labelled contextual flags (rain, drama broadcast,
promotional offer, political gathering, power cut, ...). Hand-annotated local context
for a Dhaka cell site is the genuinely distinctive asset in this dataset, and the
original code used every flag as a raw binary feature without ever testing one.

Testing them changes what they are for. Measured on GP (Welch t-test on level;
residual spread after removing the time-of-day mean):

============================  =======  =============  ========  ==========
flag                            n=1     delta mean       p       resid sd
============================  =======  =============  ========  ==========
is_weekend                        261          +0.37      0.77       16.86
is_event                          275          -0.23      0.86       17.54
is_drama                          147          -1.52      0.34       17.36
is_political_gathering            101          +2.69      0.17       18.36
is_rain                           131          +4.50     0.026       21.80
is_offer                          235          -5.73   5.1e-06       15.86
(baseline, no flag)                 -              -         -       17.33
============================  =======  =============  ========  ==========

Two findings drive the design of this module:

1. **Most flags do not predict the level.** Six of nine are indistinguishable from
   noise, including ``is_weekend`` -- which both source CSVs are named after.
2. **``is_rain`` predicts the *variance*.** It raises residual sigma from 16.4 to 21.8,
   a 33% increase, while moving the mean only modestly. ``is_political_gathering``
   does the same, weaker.

A point forecaster with a context dummy cannot express finding 2; it can only shift
its mean. Widening the allocation margin under those conditions can. That is what
:class:`UncertaintyModel` estimates and what the conformal layer consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as _st
from sklearn.ensemble import GradientBoostingRegressor

from .data import CONTEXT_FLAGS, TARGET

#: Priority order for assigning disjoint context groups. Mondrian conformal requires
#: non-overlapping groups, and the raw flags do overlap (on GP, rain and gathering
#: co-occur on 33 rows; gathering and offer on 51). Highest-variance conditions take
#: precedence, so a rainy day during a gathering is scored under the wider margin.
DEFAULT_PRIORITY: tuple[str, ...] = (
    "is_rain",
    "is_political_gathering",
    "is_offer",
    "is_powercut",
)

BASELINE_GROUP = "baseline"

#: The two-group split used for Mondrian results. Merging rain and gathering is
#: principled rather than convenient: they are the two elevated-variance conditions,
#: and merged they reach ~199 rows on GP where alone they yield ~40 and ~20
#: calibration residuals -- below what tau=0.95 can be estimated from.
ELEVATED_RISK_FLAGS: tuple[str, ...] = ("is_rain", "is_political_gathering")
ELEVATED_GROUP = "elevated_risk"


def assign_groups(
    df: pd.DataFrame,
    priority: tuple[str, ...] = DEFAULT_PRIORITY,
    baseline: str = BASELINE_GROUP,
) -> pd.Series:
    """Assign each row to exactly one context group by priority order.

    Rows matching no flag fall into ``baseline``. Flags absent from the frame are
    skipped, so the same call works for Robi (five flags) and GP (nine).
    """
    groups = pd.Series(baseline, index=df.index, dtype=object)
    # Assign in reverse priority so higher-priority flags overwrite lower ones.
    for flag in reversed([f for f in priority if f in df.columns]):
        groups[df[flag].astype(bool)] = flag
    return groups


def assign_binary_groups(
    df: pd.DataFrame,
    elevated_flags: tuple[str, ...] = ELEVATED_RISK_FLAGS,
) -> pd.Series:
    """Two-group split: ``elevated_risk`` versus ``baseline``.

    The grouping used for reported Mondrian results, for the sample-size reasons in
    :data:`ELEVATED_RISK_FLAGS`.
    """
    present = [f for f in elevated_flags if f in df.columns]
    if not present:
        return pd.Series(BASELINE_GROUP, index=df.index, dtype=object)
    elevated = df[present].astype(bool).any(axis=1)
    return pd.Series(
        np.where(elevated, ELEVATED_GROUP, BASELINE_GROUP), index=df.index, dtype=object
    )


def flag_report(df: pd.DataFrame, operator: str, target: str = TARGET) -> pd.DataFrame:
    """Test each context flag for a level effect and a variance effect.

    The level test is a Welch t-test of ``target`` with the flag on versus off. The
    variance comparison uses residuals after removing the hour-of-day mean, so a flag
    that merely correlates with time of day is not credited with explaining spread,
    and is tested with Levene's test (robust to non-normality).

    Returns one row per flag, sorted by variance ratio -- the ordering that matters for
    context-conditional allocation, as opposed to the level ordering that a point
    forecaster would care about.
    """
    y = df[target].astype(float)
    resid = y - y.groupby(df.index.hour).transform("mean")

    rows = []
    for flag in CONTEXT_FLAGS[operator.lower()]:
        if flag not in df.columns:
            continue
        mask = df[flag].astype(bool)
        n_on = int(mask.sum())
        if n_on < 5 or n_on > len(df) - 5:
            continue
        t_stat, p_level = _st.ttest_ind(y[mask], y[~mask], equal_var=False)
        _, p_var = _st.levene(resid[mask], resid[~mask])
        sd_on, sd_off = float(resid[mask].std()), float(resid[~mask].std())
        rows.append(
            {
                "flag": flag,
                "n_on": n_on,
                "mean_on": float(y[mask].mean()),
                "mean_off": float(y[~mask].mean()),
                "delta_mean": float(y[mask].mean() - y[~mask].mean()),
                "p_level": float(p_level),
                "resid_sd_on": sd_on,
                "resid_sd_off": sd_off,
                "variance_ratio": sd_on / sd_off if sd_off else np.nan,
                "p_variance": float(p_var),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("variance_ratio", ascending=False).reset_index(drop=True)


def group_sizes(groups: pd.Series) -> pd.DataFrame:
    """Group sizes with the highest tau each could support as a calibration set.

    Reported in the audit so the sample-size limits on group-conditional coverage are
    visible before any claim rests on them.
    """
    from .conformal import min_calibration_size

    counts = groups.value_counts()
    rows = []
    for name, n in counts.items():
        # Roughly a quarter of each fold's training window becomes calibration data.
        approx_calib = int(round(0.25 * n))
        supported = [t for t in (0.8, 0.9, 0.95, 0.98, 0.99)
                     if approx_calib >= min_calibration_size(t)]
        rows.append(
            {
                "group": name,
                "n_rows": int(n),
                "approx_calib": approx_calib,
                "max_supported_tau": max(supported) if supported else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)


@dataclass
class UncertaintyModel:
    """Predicts the scale of forecast error from context, for adaptive margins.

    Fits ``|residual|`` against the same design matrix the point forecaster uses. The
    prediction becomes ``sigma_hat(x)`` in
    :class:`bwalloc.conformal.LocallyAdaptiveConformal`, so the allocation margin
    widens where errors are historically larger.

    Absolute residuals are modelled rather than squared ones because the target is a
    scale for the conformal normaliser, and the absolute scale is far less sensitive
    to the handful of large outliers these traces contain.
    """

    seed: int = 42
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    model_: GradientBoostingRegressor | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, residuals) -> "UncertaintyModel":
        r = np.abs(np.asarray(residuals, dtype=float).ravel())
        if len(r) != len(X):
            raise ValueError("X and residuals must align.")
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
        ).fit(X, r)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        # Clipped at zero: a negative predicted scale is meaningless, and the
        # conformal layer applies its own floor on top of this.
        return np.maximum(np.asarray(self.model_.predict(X), dtype=float), 0.0)

    @property
    def feature_importances_(self):
        return None if self.model_ is None else self.model_.feature_importances_


def permutation_flag_value(
    X: pd.DataFrame,
    y: pd.Series,
    folds,
    model_factory,
    flags: list[str],
    seed: int = 42,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Measure each flag's contribution by permuting it within the test window.

    Complements :func:`flag_report`: that function asks whether a flag correlates with
    demand, this one asks whether a fitted model's accuracy actually degrades when the
    flag is scrambled. A flag can pass the first test and fail this one when the
    information is already carried by lags or time-of-day features.

    Returns mean and standard deviation of the RMSE increase per flag, across folds
    and repeats. Positive means the flag is earning its place.
    """
    from . import metrics as M

    rng = np.random.default_rng(seed)
    present = [f for f in flags if f in X.columns]
    rows = []

    for fold in folds:
        fit_idx = np.concatenate([fold.train, fold.calib]) if fold.n_calib else fold.train
        model = model_factory()
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        X_test, y_test = X.iloc[fold.test], y.iloc[fold.test]
        base = M.rmse(y_test, model.predict(X_test))

        for flag in present:
            deltas = []
            for _ in range(n_repeats):
                shuffled = X_test.copy()
                shuffled[flag] = rng.permutation(shuffled[flag].to_numpy())
                deltas.append(M.rmse(y_test, model.predict(shuffled)) - base)
            rows.append(
                {"fold": fold.number, "flag": flag, "rmse_increase": float(np.mean(deltas))}
            )

    per_fold = pd.DataFrame(rows)
    return (
        per_fold.groupby("flag")["rmse_increase"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
