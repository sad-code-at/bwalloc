"""Cross-operator cold-start transfer.

Does properly what ``Robi_to_GP.ipynb`` in the original project claimed to do. That
notebook is titled as a Robi-to-GP transfer study but loads the GP file for both
halves, so it trains and tests on the same operator; it then adds 20 Gbps to the test
actuals before plotting them against unchanged predictions. Nothing in it can be
salvaged, so this replaces it rather than repairing it.

The question, which is the operationally useful version of "can we transfer?":

    **How much of its own history does a newly deployed cell site need before its
    forecaster is worth using?**

Four arms, all scored on one fixed GP test window so every number is comparable:

``persistence``
    The floor. Needs no history at all.
``gp_only(k)``
    Trained only on the site's own first *k* days. What a new deployment can do alone.
``robi_cold``
    Trained on Robi and applied to GP with no GP training data -- only enough GP
    history to estimate the level and scale, which is unavoidable since the two sites
    differ by ~50 Gbps in mean demand.
``transfer(k)``
    Robi-pretrained, then boosting continued on GP's first *k* days.

Why this is possible at all
---------------------------
The two traces are sampled at different rates (86 min against 99 min), so a design
matrix indexed by *sample count* is not comparable across them -- ``lag_24`` means
34.4 h on one site and 39.6 h on the other. Because the corrected feature builder
defines lags in wall-clock hours and seasonality in Fourier terms of wall-clock time,
the two design matrices measure the same quantities and a model can move between
them. The transfer study is therefore a direct dividend of the sampling-rate
correction, not an independent contribution.

Context availability is itself part of the problem: GP carries nine hand-labelled
flags and Robi five, so transfer is restricted to the five they share.

    python experiments/run_transfer.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bwalloc as bw  # noqa: E402
from bwalloc.data import CONTEXT_FLAGS, TARGET, load, sampling_profile  # noqa: E402
from bwalloc.features import FeatureConfig, build_features  # noqa: E402
from bwalloc.forecast import (  # noqa: E402
    direct_design,
    horizon_steps,
    persistence_at_horizon,
)
from bwalloc.metrics import rmse  # noqa: E402
from bwalloc.models import xgboost_point  # noqa: E402

RESULTS = ROOT / "experiments" / "results"

#: Flags both operators carry. GP's four extra flags -- including `is_rain`, the only
#: one the audit found to carry real variance signal -- cannot be used here.
SHARED_FLAGS = tuple(
    f for f in CONTEXT_FLAGS["gp"] if f in CONTEXT_FLAGS["robi"]
)

#: Days of the target site's own history to fine-tune on.
K_DAYS = (1, 3, 5, 7, 14, 21)

#: Fraction of the target trace held out for testing, taken from the end.
TEST_FRAC = 0.30

#: Lead times to repeat the study at. One step is included because it is what the
#: original notebook attempted, but it is the least informative setting: at a one-step
#: horizon persistence is close to unbeatable on these traces, so every model arm sits
#: near the floor and the transfer question cannot be answered. The 6-hour row is the
#: operationally meaningful one.
HORIZON_HOURS = (1.5, 6.0)

#: Columns carrying the target's units, which must be rescaled when moving between
#: operators whose demand differs by ~50 Gbps in the mean.
LEVEL_PREFIXES = ("lag_", "rollmean_")
SPREAD_PREFIXES = ("rollstd_",)


def standardise(X: pd.DataFrame, y: pd.Series, mu: float, sigma: float):
    """Put an operator's design matrix and target on a common, unit-free scale.

    Level-valued columns are centred and scaled like the target; spread-valued columns
    are scaled but not centred, since a standard deviation has no offset. Fourier
    terms and binary flags are already comparable and are left alone.
    """
    Xs = X.copy()
    for col in Xs.columns:
        if col.startswith(LEVEL_PREFIXES):
            Xs[col] = (Xs[col] - mu) / sigma
        elif col.startswith(SPREAD_PREFIXES):
            Xs[col] = Xs[col] / sigma
    return Xs, (y - mu) / sigma


def prepare(operator: str):
    df = load(operator)
    profile = sampling_profile(df)
    config = FeatureConfig(context_flags=SHARED_FLAGS)
    X, y = build_features(df, profile, config)
    return df, profile, X, y


def fit_xgb(X, y, warm_start=None):
    """Fit a booster, optionally continuing from a pretrained one."""
    model = clone(xgboost_point().estimator)
    if warm_start is None:
        model.fit(X, y)
    else:
        model.fit(X, y, xgb_model=warm_start)
    return model


def run_one(hours: float, src, tgt) -> pd.DataFrame:
    """The four-arm study at one lead time.

    The horizon is resolved to a sample count *per operator*, so "6 hours ahead" is
    4 samples on GP and 4 on Robi rather than the same integer on both -- the same
    discipline that makes the two design matrices comparable in the first place.
    """
    src_profile, X_src_1, y_src_1 = src
    tgt_profile, X_tgt_1, y_tgt_1 = tgt

    X_src, y_src, _ = direct_design(X_src_1, y_src_1, horizon_steps(src_profile, hours))
    steps_t = horizon_steps(tgt_profile, hours)
    X_tgt, y_tgt, _ = direct_design(X_tgt_1, y_tgt_1, steps_t)

    # -- Fixed target test window; everything is scored on exactly these rows. -----
    n = len(y_tgt)
    test_start = int(n * (1 - TEST_FRAC))
    test = slice(test_start, n)
    history_end = X_tgt.index[test_start]
    print(f"\n{'=' * 78}\n{hours:g}-hour horizon — {steps_t} samples on GP "
          f"({tgt_profile.hours_for_lag(steps_t):.2f} h of real lead)\n{'=' * 78}")
    print(f"  GP test window: {X_tgt.index[test_start].date()} to "
          f"{X_tgt.index[-1].date()}  ({n - test_start} points); "
          f"{(history_end - X_tgt.index[0]).days} days of history before it")

    # -- Source model, trained once on all of Robi. --------------------------------
    mu_s, sd_s = float(y_src.mean()), float(y_src.std())
    Xs_src, ys_src = standardise(X_src, y_src, mu_s, sd_s)
    source = fit_xgb(Xs_src, ys_src)

    y_test = y_tgt.iloc[test].to_numpy(dtype=float)
    rows = []

    # -- Floor: persistence at this lead time, which needs no history at all. -----
    persistence = persistence_at_horizon(
        y_tgt_1, steps_t
    ).reindex(y_tgt.index).iloc[test].to_numpy(dtype=float)
    ok = ~np.isnan(persistence)
    rows.append({"arm": "persistence", "k_days": 0, "n_train": 0,
                 "rmse": rmse(y_test[ok], persistence[ok])})

    for k in K_DAYS:
        cutoff = X_tgt.index[0] + pd.Timedelta(days=k)
        train_mask = X_tgt.index < cutoff
        n_train = int(train_mask.sum())
        if n_train < 20 or cutoff >= history_end:
            continue

        X_k, y_k = X_tgt[train_mask], y_tgt[train_mask]
        # Scale from the target site's *own* first k days -- the only statistics a new
        # deployment actually has.
        mu_t, sd_t = float(y_k.mean()), float(y_k.std())
        Xs_k, ys_k = standardise(X_k, y_k, mu_t, sd_t)
        Xs_te, _ = standardise(X_tgt.iloc[test], y_tgt.iloc[test], mu_t, sd_t)

        own = fit_xgb(Xs_k, ys_k)
        rows.append({"arm": "gp_only", "k_days": k, "n_train": n_train,
                     "rmse": rmse(y_test, own.predict(Xs_te) * sd_t + mu_t)})

        tuned = fit_xgb(Xs_k, ys_k, warm_start=source.get_booster())
        rows.append({"arm": "transfer", "k_days": k, "n_train": n_train,
                     "rmse": rmse(y_test, tuned.predict(Xs_te) * sd_t + mu_t)})

        # The cold start: the Robi model applied directly, with only these k days used
        # to fix the level and scale. Repeated per k because that estimate improves.
        rows.append({"arm": "robi_cold", "k_days": k, "n_train": 0,
                     "rmse": rmse(y_test, source.predict(Xs_te) * sd_t + mu_t)})

    table = pd.DataFrame(rows)
    table.insert(0, "lead_hours", round(tgt_profile.hours_for_lag(steps_t), 2))
    table.insert(0, "horizon_hours", hours)

    pivot = table.pivot_table(index="k_days", columns="arm", values="rmse")
    floor = float(table[table["arm"] == "persistence"]["rmse"].iloc[0])
    print("\n  RMSE on the fixed GP test window, by days of GP history available:\n")
    print(pivot.round(3).to_string())
    print(f"\n  persistence at this lead (no history at all): {floor:.3f}")

    if {"transfer", "gp_only"} <= set(pivot.columns):
        gain = (pivot["gp_only"] - pivot["transfer"]) / pivot["gp_only"]
        print("\n  transfer gain over the site's own data alone:")
        for k, g in gain.items():
            print(f"    {k:2d} days: {g:+6.1%}")
        for arm in ("gp_only", "transfer", "robi_cold"):
            useful = pivot[pivot[arm] < floor]
            verdict = (f"beats persistence from k = {useful.index[0]} days"
                       if not useful.empty else "never beats persistence")
            print(f"    {arm:>10}: {verdict}")

    return table


def main() -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    print(f"shared context flags ({len(SHARED_FLAGS)}): {list(SHARED_FLAGS)}")
    print("GP-only flags dropped for transfer:",
          [f for f in CONTEXT_FLAGS["gp"] if f not in SHARED_FLAGS])

    _, src_profile, X_src, y_src = prepare("robi")
    _, tgt_profile, X_tgt, y_tgt = prepare("gp")
    assert list(X_src.columns) == list(X_tgt.columns), "design matrices must align"

    tables = [
        run_one(hours, (src_profile, X_src, y_src), (tgt_profile, X_tgt, y_tgt))
        for hours in HORIZON_HOURS
    ]
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(RESULTS / "transfer_robi_to_gp.csv", index=False)
    print(f"\nWrote {RESULTS / 'transfer_robi_to_gp.csv'}")


if __name__ == "__main__":
    main()
