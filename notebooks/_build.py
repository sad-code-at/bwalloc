"""Generate the analysis notebooks.

The notebooks are kept thin on purpose: every computation lives in ``src/bwalloc``
and is exercised by the test suite, so a notebook is a narrative over library calls
rather than a place where logic hides. The original project's failure mode was seven
near-duplicate copies of the same feature-engineering block that had silently drifted
apart; generating the notebooks from one source keeps that from recurring.

Regenerate after changing the library:

    python notebooks/_build.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

BOOTSTRAP = '''\
# --- Bootstrap: works locally, on Colab and on Kaggle ----------------------
# RUN THIS CELL FIRST, and re-run it after any kernel restart. Every later cell
# depends on it. If it fails, the next cell fails with "No module named bwalloc",
# which looks like a different problem but is not.
#
# On a hosted runtime it clones the repo (or pulls, on a re-run) and installs the
# package into the session, so `import bwalloc` keeps working even from a cell you
# run on its own after restarting. The repository is public; no token is needed.
import os, subprocess, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = "https://github.com/sad-code-at/bwalloc.git"

def _find_root(start: Path):
    node = start
    while not (node / "src" / "bwalloc").exists() and node != node.parent:
        node = node.parent
    return node if (node / "src" / "bwalloc").exists() else None

ROOT = _find_root(Path.cwd())

if ROOT is None:
    # A repo uploaded as a Kaggle Dataset mounts read-only here; prefer it if present.
    for candidate in Path("/kaggle/input").glob("*/src/bwalloc"):
        ROOT = candidate.parent.parent
        break

if ROOT is None:
    on_kaggle = Path("/kaggle/working").exists()
    target = Path("/kaggle/working/bwalloc") if on_kaggle else Path("/content/bwalloc")
    if _find_root(target) is None:
        if os.system("git clone -q " + REPO + " " + str(target)) != 0 or _find_root(target) is None:
            raise RuntimeError(
                "Clone failed. On Kaggle, switch Internet ON in the right sidebar "
                "(Settings > Internet), then re-run this cell. See docs/KAGGLE.md."
            )
        print("cloned", REPO)
    else:
        os.system("git -C " + str(target) + " pull -q --ff-only")
        print("pulled latest into", target)
    os.chdir(target)
    ROOT = target
    # Install into the session so `import bwalloc` survives a kernel restart and
    # does not depend on this cell having set sys.path. --no-deps deliberately:
    # Kaggle and Colab curate their own numpy/pandas and we must not disturb them.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
                   cwd=str(ROOT), capture_output=True)

sys.path.insert(0, str(ROOT / "src"))

try:
    import xgboost  # noqa: F401
except ImportError:
    !pip install -q xgboost

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import bwalloc as bw
from bwalloc.plots import use_paper_style

bw.set_seed()
use_paper_style()
pd.set_option("display.width", 200)
RESULTS = ROOT / "experiments" / "results"
# Scratch output for the exploratory notebooks. Only 07_paper_figures writes into
# paper/figures -- otherwise running notebook 00 or 05 silently overwrites a figure
# the paper cites, which is exactly the kind of drift this project exists to remove.
FIGURES = ROOT / "notebooks" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
_data = sorted((ROOT / "data").glob("*.csv"))
_res = sorted(RESULTS.glob("*.csv"))
print("bwalloc", bw.__version__, "at", ROOT)
print(f"  data/    {len(_data)} csv  ({', '.join(f.name for f in _data) or 'MISSING'})")
print(f"  results/ {len(_res)} csv")
if not _data:
    raise RuntimeError(
        "The trace CSVs are missing, so nothing will run. Re-run this cell to "
        "re-clone, or check that the repository was fetched completely."
    )
'''


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.rstrip().split("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.rstrip().split("\n"),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": [
            # nbformat wants each source line to keep its newline.
            {**c, "source": [ln + "\n" for ln in c["source"][:-1]] + [c["source"][-1]]}
            for c in cells
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #

NOTEBOOKS: dict[str, list[dict]] = {}

NOTEBOOKS["00_data_audit.ipynb"] = [
    md("""
# 00 — Data audit

What the two traces actually are, and the three measurement errors that shaped the
earlier results. Nothing here is modelling; it is all things that can be checked
directly against the CSVs.

The three findings, in order of how much they matter:

1. **Neither trace is hourly.** GP samples every ~86 min and Robi every ~99 min, so
   `lag_24` spans 34.4 h and 39.6 h — roughly anti-phase to the daily cycle.
2. **The rolling features leaked the target.** Windows were computed without a shift,
   putting `y_t` inside its own feature vector.
3. **The context flags act on variance, not level.** Most do not predict how much
   demand there will be; `is_rain` predicts how *wrong* the forecast will be.
"""),
    code(BOOTSTRAP),
    md("## Sampling rate\n\nEvery seasonal hyperparameter in this project is derived "
       "from this table. Nothing is hardcoded to 24."),
    code("""
from bwalloc.data import TARGET, load, sampling_profile, autocorrelation_by_lag

traces = {}
for operator in ("gp", "robi"):
    df = load(operator)
    profile = sampling_profile(df)
    traces[operator] = (df, profile)
    print(f"{operator.upper():5} {profile.describe()}")
    print(f"      lag 24 samples spans {profile.hours_for_lag(24):.1f} h "
          f"— the original code called this 'one day'")
    print(f"      one day is actually {profile.lag_for_hours(24)} samples\\n")
"""),
    md("## Autocorrelation: the lag the original models used is negatively correlated\n\n"
       "This is the whole sampling-rate argument in one figure. The measured daily "
       "period has strong positive autocorrelation; lag 24 has negative."),
    code("""
from bwalloc.plots import plot_autocorrelation_by_lag

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, (operator, (df, profile)) in zip(axes, traces.items()):
    acf = autocorrelation_by_lag(df[TARGET], max_lag=40, profile=profile)
    plot_autocorrelation_by_lag(acf, profile, ax=ax)
    ax.set_title(f"{operator.upper()} — {profile.median_gap_min:.0f} min/sample")
fig.tight_layout()
fig.savefig(FIGURES / "fig1_autocorrelation.png", dpi=200, bbox_inches="tight")
"""),
    md("## The empirical consequence\n\nA seasonal-naive forecaster at the measured "
       "period against the same forecaster at lag 24. Same data, same code, one "
       "hyperparameter."),
    code("""
from bwalloc.metrics import rmse

rows = []
for operator, (df, profile) in traces.items():
    y = df["Gbps"]
    for lag, label in ((profile.daily_period, "measured daily period"), (24, "lag 24")):
        rows.append({
            "operator": operator, "lag": lag, "spans_hours": round(profile.hours_for_lag(lag), 1),
            "label": label, "rmse": rmse(y.iloc[lag:], y.shift(lag).dropna()),
        })
pd.DataFrame(rows)
"""),
    md("## Target leakage\n\n`assert_no_leakage` perturbs the final target value and "
       "asserts no feature column moves. The corrected builder passes; a builder "
       "with an unshifted rolling window would not."),
    code("""
from bwalloc.features import FeatureConfig, assert_no_leakage, build_features

for operator, (df, profile) in traces.items():
    assert_no_leakage(df, profile, FeatureConfig())
    X, y = build_features(df, profile, FeatureConfig())
    print(f"{operator.upper():5} leak-free — {X.shape[1]} features, {len(X)} usable rows")
"""),
    md("## Context flags — the dataset's distinctive asset, tested for the first time\n\n"
       "Welch t-test on the level, Levene's test on the spread of hour-detrended "
       "residuals. The variance column is the one that matters: it is what the "
       "context-conditional method in notebook 03 is built on."),
    code("""
from bwalloc.context import flag_report, group_sizes, assign_binary_groups

for operator, (df, profile) in traces.items():
    print(f"=== {operator.upper()} ===")
    report = flag_report(df, operator)
    display(report)
    print(group_sizes(assign_binary_groups(df)).to_string(index=False), "\\n")
"""),
    md("""
### Reading the table

Two things fall out, and both are used later:

- **`is_weekend` carries no signal at all** on the level (p = 0.77) — yet it is in
  every feature list, and *both source filenames advertise it*. Six of the nine flags
  are indistinguishable from noise on the mean.
- **`is_rain` raises residual σ from 16.4 to 21.8 Gbps — a 33% increase in
  uncertainty** — while moving the mean only modestly.

Context here acts on the **variance**, not the level. A point forecaster with a
context dummy cannot express that. A context-conditional *interval* can, and that is
notebook 03.
"""),
]

NOTEBOOKS["01_corrected_benchmark.ipynb"] = [
    md("""
# 01 — The corrected benchmark

Every model on one rolling-origin fold schedule, with naive baselines, fold-level
spreads, and Diebold–Mariano tests under FDR control.

Three things this fixes relative to the earlier study: models were compared across
*different* train/test splits; no naive baseline was ever computed; and on ~300 test
points the RMSE gaps between models sit inside fold-to-fold noise.

**Read this notebook together with 04.** At one step ahead the learned models beat a
one-line baseline only modestly, and that is the honest result here. The case for
learning is made at the lead times an allocator actually needs, which is notebook 04.
"""),
    code(BOOTSTRAP),
    code("""
from bwalloc.baselines import SeasonalNaive, standard_baselines
from bwalloc.data import load, sampling_profile
from bwalloc.evaluate import beats_baseline, dm_matrix, run_backtest, summarise
from bwalloc.features import FeatureConfig, build_features
from bwalloc.models import default_point_models
from bwalloc.splits import describe_folds, rolling_origin

OPERATOR = "gp"          # switch to "robi" and re-run
N_FOLDS = 8

df = load(OPERATOR)
profile = sampling_profile(df)
X, y = build_features(df, profile, FeatureConfig())
folds = rolling_origin(len(y), n_folds=N_FOLDS)
describe_folds(folds, X.index)
"""),
    md("## Backtest\n\nEvery model and every baseline on identical folds."),
    code("""
baselines = standard_baselines(y.to_numpy(), profile.daily_period)
baselines.append(SeasonalNaive(y.to_numpy(), period=24))   # the lag the original used

per_fold, predictions = run_backtest(
    X, y, folds, models=default_point_models(),
    baselines=baselines, season_lag=profile.daily_period,
)
summary = beats_baseline(summarise(per_fold))
summary[["model", "rmse_mean", "rmse_std", "mae_mean", "mase_mean",
         "vs_persistence", "beats_persistence"]]
"""),
    md("**The gate.** A model that does not beat persistence on the same folds is not "
       "a result. `run_benchmark.py` exits non-zero if the top-ranked model fails this."),
    code("""
best = summary.iloc[0]
assert bool(best["beats_persistence"]), f"{best['model']} does not beat persistence"
print(f"best: {best['model']} — RMSE {best['rmse_mean']:.3f} ± {best['rmse_std']:.3f} "
      f"({best['vs_persistence']:+.1%} vs persistence)")
"""),
    md("## Is the ranking real?\n\nDiebold–Mariano across folds, Benjamini–Hochberg "
       "corrected because ten models is 45 comparisons. Most adjacent pairs are not "
       "separable on this much data — which is itself the finding."),
    code("""
dm = dm_matrix(predictions, horizon=1)
top = summary["model"].head(4).tolist()
dm[dm["model_a"].isin(top) & dm["model_b"].isin(top)][
    ["model_a", "model_b", "dm_stat", "p_value", "significant_fdr", "winner"]
]
"""),
    md("""
## Feature ablation — a negative result, reported

`experiments/run_benchmark.py` runs the full ablation. Its conclusion is worth
stating plainly because it cuts against the correction:

**Correcting the feature design does not improve accuracy on either operator.** Tree
ensembles route around a mis-specified `lag_24` by leaning on `lag_1..3`, so the
sampling-rate error costs almost nothing in RMSE once a flexible model is used.

That does not make the correction pointless — it makes its value *interpretive*
rather than predictive. Every seasonal claim in the earlier study (STL, ACF, FFT,
"peak at f ≈ 0.0417 ⇒ 24-hour cycle") was stated on the wrong time axis, and the
seasonal-naive comparison in notebook 00 shows what that costs a model that cannot
route around it.
"""),
    code("""
ablation = pd.read_csv(RESULTS / f"ablation_{OPERATOR}.csv")
ablation
"""),
]

NOTEBOOKS["02_allocation.ipynb"] = [
    md("""
# 02 — Risk-aware allocation

Forecasting `Gbps` is not provisioning. This notebook builds the decision layer the
project's title promises.

**The cost model.** For an allocation `A` against realised demand `y`:

$$C = c_{over}\\,(A-y)^+ + \\kappa\\,c_{over}\\,(y-A)^+$$

Under-provisioning costs κ times more than over-provisioning. The cost-minimising
allocation is then not the mean but the **τ-quantile** of the predictive
distribution, with

$$\\tau^* = \\frac{\\kappa}{1+\\kappa}$$

So the operator's cost ratio *selects the quantile*. κ = 10 → τ = 0.909.
"""),
    code(BOOTSTRAP),
    code("""
from bwalloc.allocation import kappa_for_tau, optimal_tau

for kappa in (2, 5, 10, 20):
    print(f"kappa={kappa:2d}  ->  tau* = {optimal_tau(kappa):.4f}")
print()
# And the correspondence verified empirically rather than asserted:
rng = np.random.default_rng(0)
y = rng.gamma(shape=9, scale=10, size=200_000)
for kappa in (2, 10):
    grid = np.linspace(np.quantile(y, 0.4), np.quantile(y, 0.995), 400)
    costs = [np.mean(np.maximum(a - y, 0) + kappa * np.maximum(y - a, 0)) for a in grid]
    empirical = grid[int(np.argmin(costs))]
    print(f"kappa={kappa:2d}: cost-minimising allocation {empirical:7.2f}  "
          f"vs quantile at tau* {np.quantile(y, optimal_tau(kappa)):7.2f}")
"""),
    md("## Conformal calibration\n\nA quantile model that claims 95% rarely delivers "
       "95%. Split conformal fixes that distribution-free — which matters precisely "
       "because there are only ~900 points and no parametric error model is credible."),
    code("""
from bwalloc.context import assign_binary_groups
from bwalloc.data import load, sampling_profile
from bwalloc.features import FeatureConfig, build_features
from bwalloc.models import xgboost_point
from bwalloc.pipeline import coverage_table, run_allocation_backtest
from bwalloc.splits import rolling_origin

OPERATOR = "gp"
df = load(OPERATOR)
profile = sampling_profile(df)
X, y = build_features(df, profile, FeatureConfig())
groups = assign_binary_groups(df).reindex(X.index)
folds = rolling_origin(len(y), n_folds=8, calib_frac=0.30)

per_fold, allocations = run_allocation_backtest(
    X, y, folds, groups, model_factory=xgboost_point, taus=(0.80, 0.90, 0.95),
)
coverage_table(per_fold[per_fold["group"] == "ALL"], "coverage")
"""),
    md("""
### Coverage is systematically short — and that is a finding

Achieved coverage lands *below* nominal almost everywhere, and the shortfall is
one-sided. That is the signature of exchangeability failing under temporal drift, not
of a bug: split conformal's finite-sample guarantee is conditional on exchangeability,
and a 55-day trace with a trend does not supply it.

`experiments/run_coverage_gate.py` quantifies it across every configuration.
`conformal.AdaptiveConformalInference` is the remedy — it updates the requested level
online from realised breaches, so long-run coverage converges without assuming
exchangeability at all. The `aci` row above is that method.
"""),
    code("""
gate = pd.read_csv(RESULTS / "coverage_gate.csv")
marginal = gate[(gate["group"] == "ALL") & (gate["operator"] == OPERATOR)]
marginal[["horizon_hours", "tau", "method", "n", "coverage", "gap", "passes", "decisive"]]
"""),
    md("## The capacity–risk frontier\n\nThe headline comparison: the calibrated "
       "allocator against the fixed-margin rule operators actually use "
       "(`A = 1.3 × ŷ`), plus static peak allocation. Read it as *capacity required "
       "at an equal SLA violation rate*."),
    code("""
from bwalloc.plots import plot_pareto

pareto = pd.read_csv(RESULTS / f"pareto_{OPERATOR}.csv")
fig, ax = plt.subplots(figsize=(6.5, 4.2))
plot_pareto(pareto, ax=ax)
ax.set_title(f"{OPERATOR.upper()} — capacity vs SLA risk")
fig.savefig(FIGURES / f"fig3_pareto_{OPERATOR}.png", dpi=200, bbox_inches="tight")

pd.read_csv(RESULTS / f"savings_{OPERATOR}.csv")
"""),
    md("""
### Honest reading of the frontier

**The capacity-saving claim does not hold on these traces.** No conformal family
reaches a 1%, 2% or 5% violation target, and at the targets that are feasible the
saving against fixed-margin is negative or negligible.

The reason is structural and worth stating: the fixed-margin rule is *multiplicative*
(`A = 1.3 ŷ`) while additive conformal adds the same number of Gbps at 3 a.m. as at
peak. On a series whose level swings ~2× within a day that is a real handicap. The
`conformal_relative` family removes the confound by calibrating a percentage margin
instead — that is the like-for-like comparison, and it is the row to read.
"""),
]

NOTEBOOKS["03_context_conditional.ipynb"] = [
    md("""
# 03 — Context-conditional allocation

The contribution that is most identifiably new, and it follows from one property of
the data that nobody had looked at: **the context flags act on the variance of demand,
not its level** (notebook 00).

A marginally calibrated allocator therefore hits its target *on average* while
systematically under-provisioning when the network is under stress — the failures
cluster exactly where they hurt most. The claim tested here:

> Marginal conformal calibration misses its nominal rate inside the elevated-risk
> context group; locally adaptive and Mondrian calibration close that gap at
> near-identical capacity cost.

This is falsifiable, and it is reported in both directions: it **holds on GP** and is
**correctly null on Robi**, which has no flag with elevated variance.
"""),
    code(BOOTSTRAP),
    code("""
from bwalloc.context import assign_binary_groups, flag_report, group_sizes
from bwalloc.data import load, sampling_profile
from bwalloc.features import FeatureConfig, build_features
from bwalloc.models import xgboost_point
from bwalloc.pipeline import coverage_table, run_allocation_backtest
from bwalloc.splits import rolling_origin

OPERATOR = "gp"
df = load(OPERATOR)
profile = sampling_profile(df)

# Which flags mark elevated *uncertainty*? Levene's test on detrended residuals.
report = flag_report(df, OPERATOR)
report[["flag", "n_on", "delta_mean", "p_level", "resid_sd_on",
        "resid_sd_off", "variance_ratio", "p_variance"]]
"""),
    md("""
### Group construction, and why it is a two-group split

The raw flags overlap (`rain ∩ gathering` = 33 rows), and Mondrian calibration needs
*disjoint* groups. A four-group split leaves the gathering group with ~20 calibration
residuals, and a group of *m* residuals cannot express a level finer than `1/(m+1)` —
τ = 0.99 needs 99. So the split is two groups: `elevated_risk = rain ∪ gathering`
against `baseline`. Both merged groups are the elevated-variance ones, so the merge is
principled rather than convenient.

`conformal.min_calibration_size` enforces this and **raises rather than clipping**.
Silently returning the largest residual would report a guarantee the data cannot
support — a subtler version of the error this project exists to correct.
"""),
    code("""
from bwalloc.conformal import min_calibration_size

groups = assign_binary_groups(df).reindex(build_features(df, profile, FeatureConfig())[0].index)
print(group_sizes(groups).to_string(index=False), "\\n")
for tau in (0.80, 0.90, 0.95, 0.99):
    print(f"tau={tau:.2f} needs at least {min_calibration_size(tau):3d} calibration residuals")
"""),
    md("## The result\n\nCoverage by context group, pooled across folds, weighted by "
       "test-block size."),
    code("""
X, y = build_features(df, profile, FeatureConfig())
folds = rolling_origin(len(y), n_folds=8, calib_frac=0.30)
per_fold, allocations = run_allocation_backtest(
    X, y, folds, groups, model_factory=xgboost_point, taus=(0.80, 0.90, 0.95),
)
coverage_table(per_fold, "coverage")
"""),
    code("""
# The capacity cost of closing the gap.
coverage_table(per_fold, "mean_allocation_ratio")
"""),
    code("""
from bwalloc.plots import plot_coverage_by_group

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, tau in zip(axes, (0.80, 0.90, 0.95)):
    plot_coverage_by_group(per_fold, tau=tau, ax=ax)
    ax.set_title(f"tau = {tau:.2f}")
fig.tight_layout()
fig.savefig(FIGURES / f"fig4_coverage_by_group_{OPERATOR}.png", dpi=200, bbox_inches="tight")
"""),
    md("""
## Reading the result, and the null on Robi

On GP the gap is large and consistent: at τ = 0.95, marginal calibration delivers
0.963 on the baseline group but **0.866** inside the elevated-risk group. Locally
adaptive calibration lifts that to 0.911 and Mondrian to 0.922.

Now switch `OPERATOR` to `"robi"` and re-run. On Robi **no flag has elevated
variance** — `is_powercut` and `is_event` actually mark *calmer* periods — and the
method correspondingly does nothing (τ = 0.95 elevated: marginal 0.889, adaptive
0.875, Mondrian 0.903).

That is the right outcome to report. The claim is not "context-conditional
calibration always helps"; it is "context-conditional calibration helps exactly when
context carries variance signal, and can be tested for in advance". The null on Robi
is what makes it a prediction rather than a story.

**Caveat to keep in the paper.** The elevated-risk group carries ~40 calibration
residuals, so per-group coverage estimates are reported with Clopper–Pearson
intervals and those intervals are wide. Robi carries five of the nine flags and no
rain annotation at all, and its group is 89 rows — directional only.
"""),
]

NOTEBOOKS["04_multi_horizon.ipynb"] = [
    md("""
# 04 — Multi-horizon: the case for the whole system

Every other result in this project, and every notebook in the study it extends,
predicts `y_t` with `y_{t-1}` already in hand. That is a nowcast. No network can
provision from it: the capacity decision for an interval has to be made *before* the
previous measurement arrives.

This notebook adds the lead time, and doing so changes the conclusion of notebook 01.
"""),
    code(BOOTSTRAP),
    md("""
## Two things the design has to get right

**Which features are known in advance.** History columns (lags, rolling statistics)
can only be read at the forecast origin — at a 6-hour lead the freshest observation is
6 hours old and the feature vector must say so. Fourier time terms are deterministic
functions of wall-clock time, so their values at the *target* timestamp are legitimately
available: an operator provisioning for 9 p.m. always knows it will be 9 p.m.

**The baseline has to move too.** An *h*-step model must be scored against an *h*-step
naive forecaster. Comparing a 24-hour-ahead model against one-step persistence is the
flattering comparison, and it is how multi-step results are most often overstated.

There is also an embargo: direct training pairs carry targets *h−1* rows after their
origin, so without dropping those rows the model is fitted on outcomes that had not
occurred when the first test forecast was issued.
"""),
    code("""
from bwalloc.data import load, sampling_profile
from bwalloc.features import FeatureConfig, build_features
from bwalloc.forecast import (
    direct_design, horizon_steps, persistence_at_horizon, split_feature_roles,
)

OPERATOR = "gp"
df = load(OPERATOR)
profile = sampling_profile(df)
X, y = build_features(df, profile, FeatureConfig())

history, future = split_feature_roles(X.columns)
print("read at the origin :", history[:6], "...")
print("known at the target:", future)
print()
for hours in (1.5, 3.0, 6.0, 12.0, 24.0):
    steps = horizon_steps(profile, hours)
    print(f"  {hours:>4g} h horizon -> {steps:2d} samples "
          f"({profile.hours_for_lag(steps):5.2f} h of real lead time)")
"""),
    md("## The result\n\nRun `python experiments/run_horizon.py` to regenerate; it "
       "takes a few minutes."),
    code("""
acc = pd.read_csv(RESULTS / f"horizon_{OPERATOR}.csv")
table = acc.pivot_table(index="model", columns="lead_hours", values="rmse_mean")
display(table.round(2))

naive = table.loc["persistence_h"]
best = table.drop(index=[i for i in table.index if "naive" in i or "persistence" in i]).min()
pd.DataFrame({"naive at that lead": naive, "best model": best,
              "advantage": (best / naive - 1)}).round(3)
"""),
    md("""
### Read the advantage column

At one step ahead the learned model beats a one-line baseline by ~14% — the weak
result of notebook 01. At an **11.5-hour lead**, an operationally realistic
provisioning horizon, it beats the naive forecaster *at that same lead* by **~59%**.
The model's own accuracy degraded by 23% across a 17× longer horizon; the baseline's
degraded by 157%.

**This is the argument for the system.** The value of learning is not visible at one
step. It is visible at the lead time an allocator actually needs — and the earlier
study, by only ever evaluating at one step, was measuring in the one regime where its
own models looked worst.
"""),
    code("""
from bwalloc.plots import plot_horizon

fig, ax = plt.subplots(figsize=(6.5, 4.2))
plot_horizon(acc, ax=ax)
ax.set_title(f"{OPERATOR.upper()} — accuracy vs lead time")
fig.savefig(FIGURES / f"fig5_horizon_{OPERATOR}.png", dpi=200, bbox_inches="tight")
"""),
    md("## Direct versus recursive\n\nOne model per horizon, against one model applied "
       "repeatedly with its own predictions fed back."),
    code("""
strategy = pd.read_csv(RESULTS / f"horizon_strategy_{OPERATOR}.csv")
strategy["recursive_penalty"] = (
    strategy["recursive_rmse_mean"] / strategy["direct_rmse_mean"] - 1
)
strategy[["lead_hours", "steps", "direct_rmse_mean", "recursive_rmse_mean",
          "recursive_penalty"]].round(3)
"""),
    md("""
Error compounding dominates: at a 24-hour lead the recursive rollout is ~86% worse.
One model per horizon is the right design.

At one step the two agree to within 0.7%, as they must — that agreement is the
correctness check on the rollout, and `tests/test_bwalloc.py` pins the recursive
feature builder to the batch one so the two paths cannot drift apart.
"""),
    md("## Does the allocation layer survive the lead time?\n\nThe practical question: "
       "conformal calibration still has to deliver its service level once the "
       "forecast has degraded, and the capacity cost of doing so is what the operator "
       "actually pays."),
    code("""
alloc = pd.read_csv(RESULTS / f"horizon_alloc_{OPERATOR}.csv").fillna({"error": ""})
alloc = alloc[(alloc["error"] == "") & (alloc["group"] == "ALL")]
summary = (
    alloc.assign(cov=lambda d: d["n"] * d["coverage"],
                 ratio=lambda d: d["n"] * d["mean_allocation_ratio"])
    .groupby(["lead_hours", "tau", "method"])
    .agg(cov=("cov", "sum"), ratio=("ratio", "sum"), n=("n", "sum"))
    .assign(coverage=lambda d: d["cov"] / d["n"],
            capacity=lambda d: d["ratio"] / d["n"])
    .reset_index()
)
summary.pivot_table(index=["tau", "method"], columns="lead_hours",
                    values=["coverage", "capacity"]).round(3)
"""),
]

NOTEBOOKS["05_transfer.ipynb"] = [
    md("""
# 05 — Cross-operator cold-start transfer

**How much of its own history does a newly deployed cell site need before its
forecaster is worth using?**

This replaces `Robi_to_GP.ipynb` from the original project rather than repairing it.
That notebook is titled as a Robi→GP transfer study but loads the GP file for both
halves, so it trains and tests on the same operator; it then adds 20 Gbps to the test
actuals before plotting them against unchanged predictions. There is nothing in it to
salvage.

Four arms, all scored on one fixed GP test window:

| arm | what it represents |
|---|---|
| `persistence` | the floor — no history at all |
| `gp_only(k)` | the new site's own first *k* days, alone |
| `robi_cold` | trained on the other operator, no fine-tuning at all |
| `transfer(k)` | pretrained on the other operator, then continued on *k* days of GP |
"""),
    code(BOOTSTRAP),
    md("""
## Why this is even possible

The two traces are sampled at different rates — 86 min against 99 min — so a design
matrix indexed by *sample count* is not comparable across them: `lag_24` means 34.4 h
on one site and 39.6 h on the other. Because the corrected feature builder defines
lags in **wall-clock hours** and seasonality in Fourier terms of wall-clock time, the
two design matrices measure the same quantities and a model can move between them.

The transfer study is a dividend of the sampling-rate correction, not an independent
contribution.

Context availability is itself part of the problem: GP carries nine hand-labelled
flags and Robi five, so transfer is restricted to the five they share — and the four
that are dropped include `is_rain`, the only flag the audit found to carry real
variance signal.
"""),
    code("""
from bwalloc.data import CONTEXT_FLAGS

shared = [f for f in CONTEXT_FLAGS["gp"] if f in CONTEXT_FLAGS["robi"]]
dropped = [f for f in CONTEXT_FLAGS["gp"] if f not in shared]
print("shared :", shared)
print("dropped:", dropped)
"""),
    md("## The result\n\nRun `python experiments/run_transfer.py` to regenerate."),
    code("""
from bwalloc.plots import plot_transfer

transfer = pd.read_csv(RESULTS / "transfer_robi_to_gp.csv")
display(
    transfer.pivot_table(index=["horizon_hours", "k_days"], columns="arm", values="rmse")
    .round(3)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, hours in zip(axes, sorted(transfer["horizon_hours"].unique())):
    plot_transfer(transfer, horizon_hours=hours, ax=ax)
    ax.set_title(f"{hours:g}-hour horizon")
fig.tight_layout()
fig.savefig(FIGURES / "fig8_transfer.png", dpi=200, bbox_inches="tight")
"""),
    md("""
## Reading it

**At a one-step horizon the study cannot answer anything.** Persistence scores 12.51
and no arm beats it, so every curve sits above the floor and the transfer question is
invisible. This is the same trap the original notebook fell into, and it is why the
6-hour panel is the one to read.

**At a 6-hour lead the answer is clean:**

- A model trained *entirely on the other operator*, given only enough GP history to
  fix the level and scale, scores **20.4** against persistence's **26.5** — 23% better
  than the floor with **zero** training data from the new site.
- Pretraining helps only while the site is data-poor: **+4.6%** over training on the
  site's own data at 3 days, then nothing by 5–7 days, and **−10%** by 21 days, where
  the pretrained weights are actively holding the model back.
- The crossover is at roughly **5–7 days**. That is the operational answer: a new site
  should borrow a neighbour's model for its first week and switch to its own after.

The scale correction is not incidental. GP averages 92.9 Gbps and Robi 145.5, so
every level-valued feature and the target are z-scored per operator; the new site's
own first *k* days supply those statistics, which is the only information a genuine
cold start has.
"""),
]

NOTEBOOKS["06_foundation_models.ipynb"] = [
    md("""
# 06 — Zero-shot foundation models

The question this dataset's main weakness makes unavoidable:

> **With only ~900 samples per site, is bespoke per-operator training worth it, or does
> a foundation model that has never seen this network do just as well?**

Chronos-Bolt is evaluated strictly zero-shot — no fitting, no fine-tuning, not even a
scaling constant — on exactly the fold schedule, lead times and test blocks that
notebook 04 uses, so the numbers drop into the same table.

Two things make this more than a routine extra baseline.

**It is quantile-native.** Chronos-Bolt emits quantiles directly, so it plugs into the
allocation layer with no quantile-regression step. We can therefore ask whether its
*uncalibrated* quantiles deliver their nominal service level.

**It assumes regular sampling, which these traces violate.** A foundation model
consumes a sequence of values with no timestamps. It cannot know that a step is 86
minutes on one trace and 99 on the other. Everything notebook 00 corrects by *measuring*
the sampling rate, this model class structurally cannot see.

Runs on CPU — no GPU needed. `pip install chronos-forecasting`, then
`python experiments/run_foundation.py` (~7 min).
"""),
    code(BOOTSTRAP),
    md("## Zero-shot against models trained on this operator"),
    code("""
accuracy = pd.read_csv(RESULTS / "foundation_accuracy.csv")
zero = (
    accuracy.groupby(["operator", "model", "lead_hours"])
    .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
    .reset_index()
)

for operator in ("gp", "robi"):
    trained = pd.read_csv(RESULTS / f"horizon_{operator}.csv")
    table = trained.pivot_table(index="model", columns="lead_hours", values="rmse_mean")
    z = zero[zero["operator"] == operator].pivot_table(
        index="model", columns="lead_hours", values="rmse_mean"
    )
    print(operator.upper())
    display(pd.concat([table, z]).round(2))
"""),
    md("## Do its own quantiles deliver their nominal level?"),
    code("""
calibration = pd.read_csv(RESULTS / "foundation_calibration.csv")
calibration["c"] = calibration["n"] * calibration["coverage"]
pooled = (
    calibration.groupby(["operator", "model", "tau", "method"])
    .agg(c=("c", "sum"), n=("n", "sum"),
         natively_expressible=("natively_expressible", "first"))
    .assign(achieved=lambda d: d["c"] / d["n"])
    .reset_index()
)
pooled["gap"] = pooled["achieved"] - pooled["tau"]
pooled.pivot_table(index=["operator", "tau", "natively_expressible"],
                   columns="method", values=["achieved", "gap"]).round(3)
"""),
    md("""
### The ceiling that matters for provisioning

Chronos-Bolt's quantile head was trained on levels **0.1 to 0.9 only**. A request for
τ = 0.95 is silently clipped to τ = 0.90 and returns the same numbers — which is why the
`natively_expressible` column is False there.

That is not a configuration detail, it is a limit on what the model can be used for.
Since τ\\* = κ/(1+κ), a ceiling of 0.9 corresponds to a cost asymmetry of only **κ = 9**.
An operator for whom under-provisioning costs 20× more than over-provisioning needs
τ\\* = 0.952, and cannot get it from the model's own quantile head at all.

So for this application, calibration on top of the foundation model is not an optional
refinement — it is what makes the model usable. The `conformal` rows show split-conformal
calibration on held-out residuals restoring the levels the model cannot express itself.
"""),
]

NOTEBOOKS["07_paper_figures.ipynb"] = [
    md("""
# 05 — Paper figures

Regenerates every figure from `experiments/results/` only. No model is fitted here,
so the figures cannot silently disagree with the tables they are drawn from — and the
paper regenerates on a clean runtime without refitting anything.

Run the experiment scripts first:

```bash
python experiments/run_audit.py
python experiments/run_benchmark.py
python experiments/run_allocation.py
python experiments/run_horizon.py
python experiments/run_coverage_gate.py
```
"""),
    code(BOOTSTRAP),
    code("""
from bwalloc.data import load, sampling_profile
from bwalloc.plots import (
    plot_benchmark, plot_coverage_by_group, plot_flag_report, plot_pareto,
)

# This notebook -- and only this notebook -- writes the figures the paper cites.
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

profiles = {op: sampling_profile(load(op)) for op in ("gp", "robi")}
written = []

def save(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    written.append(name)
    plt.close(fig)
"""),
    md("### Figure 1 — sampling rate and autocorrelation"),
    code("""
from bwalloc.plots import plot_autocorrelation_by_lag

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, op in zip(axes, ("gp", "robi")):
    acf = pd.read_csv(RESULTS / f"audit_acf_{op}.csv")
    plot_autocorrelation_by_lag(acf, profiles[op], ax=ax)
    ax.set_title(f"{op.upper()} — {profiles[op].median_gap_min:.0f} min/sample")
fig.tight_layout()
save(fig, "fig1_autocorrelation.png")
"""),
    md("### Figure 2 — corrected benchmark with baselines"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, op in zip(axes, ("gp", "robi")):
    plot_benchmark(pd.read_csv(RESULTS / f"benchmark_{op}_summary.csv"), ax=ax)
    ax.set_title(op.upper())
fig.tight_layout()
save(fig, "fig2_benchmark.png")
"""),
    md("### Figure 3 — capacity–risk frontier"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, op in zip(axes, ("gp", "robi")):
    plot_pareto(pd.read_csv(RESULTS / f"pareto_{op}.csv"), ax=ax)
    ax.set_title(op.upper())
fig.tight_layout()
save(fig, "fig3_pareto.png")
"""),
    md("### Figure 4 — the context-conditional result (GP) and its null (Robi)"),
    code("""
# One legend for the grid, and one y-range: with per-panel autoscaling the bars are
# not comparable across panels even when the axes are nominally shared.
fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharey=True)
for r, op in enumerate(("gp", "robi")):
    per_fold = pd.read_csv(RESULTS / f"allocation_{op}_perfold.csv").fillna({"error": ""})
    for c, tau in enumerate((0.80, 0.90, 0.95)):
        ax = axes[r][c]
        plot_coverage_by_group(per_fold, tau=tau, ax=ax,
                               legend=(r == 1 and c == 1), ylim=(0.60, 1.0))
        ax.set_title(f"{op.upper()}  tau = {tau:.2f}")
        if c:
            ax.set_ylabel("")
fig.tight_layout()
save(fig, "fig4_coverage_by_group.png")
"""),
    md("### Figure 5 — accuracy versus lead time\n\nThe multi-horizon result: the "
       "learned models stay flat while the naive forecaster collapses, so the gap "
       "between them — the value of learning — widens with lead time."),
    code("""
from bwalloc.plots import plot_horizon

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, op in zip(axes, ("gp", "robi")):
    plot_horizon(pd.read_csv(RESULTS / f"horizon_{op}.csv"), ax=ax)
    ax.set_title(op.upper())
fig.tight_layout()
save(fig, "fig5_horizon.png")
"""),
    md("### Figure 6 — context-flag audit"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, op in zip(axes, ("gp", "robi")):
    plot_flag_report(pd.read_csv(RESULTS / f"audit_flags_{op}.csv"), ax=ax)
    ax.set_title(op.upper())
fig.tight_layout()
save(fig, "fig6_flag_audit.png")
"""),
    md("""
### Figure 7 — provisioning cost, the headline claim

Bars are the cost at the level the theory prescribes from the cost ratio
(τ* = κ/(1+κ)), so no test-set information enters the choice. The dashed line is the
best the fixed-margin heuristic can do *with* hindsight about which margin hit which
violation rate — the comparison is deliberately biased against the bars.

Blue beats that line; red does not. Context-adaptive calibration is blue on both
operators and marginal calibration is red on both, which is the whole argument.
"""),
    code("""
from bwalloc.plots import plot_cost

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, op in zip(axes, ("gp", "robi")):
    plot_cost(pd.read_csv(RESULTS / f"cost_{op}.csv"), ax=ax)
    ax.set_title(op.upper())
fig.tight_layout()
save(fig, "fig7_cost.png")
"""),
    md("### Figure 8 — cold-start transfer"),
    code("""
from bwalloc.plots import plot_transfer

transfer = pd.read_csv(RESULTS / "transfer_robi_to_gp.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, hours in zip(axes, sorted(transfer["horizon_hours"].unique())):
    plot_transfer(transfer, horizon_hours=hours, ax=ax)
    ax.set_title(f"{hours:g}-hour horizon")
fig.tight_layout()
save(fig, "fig8_transfer.png")
"""),
    md("### Figure 9 — zero-shot foundation model against trained and naive"),
    code("""
from bwalloc.plots import plot_foundation

accuracy = pd.read_csv(RESULTS / "foundation_accuracy.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, op in zip(axes, ("gp", "robi")):
    plot_foundation(accuracy, pd.read_csv(RESULTS / f"horizon_{op}.csv"), op,
                    ax=ax, legend=(op == "gp"))
    ax.set_title(op.upper())
fig.tight_layout()
save(fig, "fig9_foundation.png")
"""),
    code("""
print(f"wrote {len(written)} figures to {FIGURES}")
for name in written:
    print("  ", name)
"""),
]


NOTEBOOKS["08_sequence_models.ipynb"] = [
    md("""
# 08 — Sequence models: CNN, RNN, LSTM, GRU

**Do the deep sequence models actually beat the tree ensembles?** The original study
said yes and its evidence could not support the claim. This notebook settles it.

The original reported a CNN at 9.69 RMSE on GP against XGBoost at 13.91 and concluded
the architectures were better. They were never compared on equal terms:

| | training rows it got |
|---|---|
| CNN / LSTM / GRU / RNN | `train_size=0.7` — roughly **600** |
| XGBoost / Random Forest | the `lag_336` → `dropna()` → date-split path — **88** |

A model with 600 training rows beating one with 88 tells you about the split, not the
architecture. Here every model — sequence and tree alike — reads the **same** univariate
window of 24 consecutive sample lags on the **same** rolling-origin folds, which is the
input shape the original gave its sequence models (`input_shape=(lookback, 1)`).

Architectures and hyperparameters are the original's, unchanged: `Conv1D(64, 3)`,
`LSTM(64)`, `GRU(64)`, `SimpleRNN(50)`, 30 epochs, Adam, MSE. Anything that differs in
the results is therefore the protocol, not retuning.
"""),
    code(BOOTSTRAP),
    md("""
## Setup

`sequence_feature_config(24)` builds a design matrix of `lagS_1 … lagS_24` and nothing
else — no Fourier terms, no calendar integers, no context flags. A univariate sequence
model cannot consume those anyway, and giving the trees extra features here would
reintroduce exactly the kind of unequal comparison this notebook exists to remove.

The window is built by the ordinary leak-safe feature builder, so no separate leakage
argument is needed: `assert_no_leakage` covers it.
"""),
    code("""
from bwalloc.baselines import SeasonalNaive, standard_baselines
from bwalloc.data import load, sampling_profile
from bwalloc.evaluate import beats_baseline, dm_matrix, run_backtest, summarise
from bwalloc.features import FeatureConfig, assert_no_leakage, build_features
from bwalloc.models import default_point_models
from bwalloc.sequence import sequence_feature_config, sequence_models
from bwalloc.splits import rolling_origin

OPERATOR = "gp"          # switch to "robi" and re-run
N_FOLDS, LOOKBACK, EPOCHS = 8, 24, 30

df = load(OPERATOR)
profile = sampling_profile(df)
config = sequence_feature_config(LOOKBACK)
assert_no_leakage(df, profile, config)          # gate: refuse to proceed if it leaks

X, y = build_features(df, profile, config)
folds = rolling_origin(len(y), n_folds=N_FOLDS)
print(f"{OPERATOR.upper()}: {X.shape[0]} rows x {X.shape[1]} lag columns, {N_FOLDS} folds")
"""),
    md("""
## The comparison

Four sequence models and three tree/linear models on identical folds, against the full
set of naive baselines. On CPU this takes about two minutes.
"""),
    code("""
# What the original study reported, for the side-by-side.
ORIGINAL = {
    "gp": {"cnn": 9.69, "rnn": 10.06, "lstm": 10.67, "gru": 12.14,
           "xgboost": 13.91, "random_forest": 14.96},
    "robi": {"cnn": 20.63, "rnn": 21.6, "lstm": 25.0, "gru": 27.5,
             "xgboost": 22.72, "random_forest": 30.10},
}

baselines = standard_baselines(y.to_numpy(), profile.daily_period)
baselines.append(SeasonalNaive(y.to_numpy(), period=24))

models = sequence_models(lookback=LOOKBACK, epochs=EPOCHS) + default_point_models()
per_fold, predictions = run_backtest(
    X, y, folds, models=models, baselines=baselines,
    season_lag=profile.daily_period,
)

summary = beats_baseline(summarise(per_fold))
view = summary[["model", "rmse_mean", "rmse_std", "mae_mean", "mase_mean",
                "vs_persistence", "beats_persistence"]].copy()
view["originally_reported"] = view["model"].map(ORIGINAL[OPERATOR])
view
"""),
    md("""
**On GP the CNN wins — 7.88 against random forest's 8.67 — and every model in the table
beats persistence.** Compare the last column: every one of the original's numbers was
worse than what the same architecture achieves here, and the tree models improved most
(14.96 → 8.67) because they were the ones the broken split starved.

On Robi nothing separates: GRU 19.88, ridge 20.14, CNN 20.16, random forest 20.52. The
sequence models' apparent 20–30% margins in the original evaporate.
"""),
    md("""
## Is the CNN's win real, or fold noise?

A 9% gap on eight folds of ~66 test points is not self-evidently a result. Diebold–Mariano
with Benjamini–Hochberg correction answers it.
"""),
    code("""
dm = dm_matrix(predictions, horizon=1)
family = ["cnn", "lstm", "gru", "rnn", "random_forest", "xgboost", "ridge", "persistence"]
best_tree = next(m for m in summary["model"] if m in ("random_forest", "xgboost", "ridge"))

pairs = dm[dm["model_a"].isin(family) & dm["model_b"].isin(family)]
pairs = pairs[(pairs["model_a"] == best_tree) | (pairs["model_b"] == best_tree)]
pairs[["model_a", "model_b", "dm_stat", "p_value", "significant_fdr", "winner"]]
"""),
    md("""
On GP: **`cnn` vs `random_forest`, p = 0.003, significant under BH correction.** Every
other pair is a tie. On Robi no pair is significant at all.

So the original's *conclusion* survives on one operator while its *evidence* does not —
and the corrected protocol is what tells those two cases apart.
"""),
    md("""
## The larger finding: lag depth, not architecture

The window above carries 24 consecutive lags where the corrected design of notebook 01
carries four (1.5, 3, 4.5 and 24 wall-clock hours). That is a confound, and it matters:
if the gain is window depth then the sequence models get no architectural credit for it
and the corrected design was simply under-lagged.

Every configuration below is scored on **one common row index**, so the fold schedule
cannot differ between them. The control that settles it is the full corrected design
*plus* the same 24 lags.
"""),
    code("""
DEPTH = {
    "full_corrected": FeatureConfig(),
    "full_plus_lags1_24": FeatureConfig(lag_samples=tuple(range(1, 25))),
    "lags1_24_only": FeatureConfig(
        lag_hours=(), rolling_hours=(), lag_samples=tuple(range(1, 25)),
        daily_harmonics=0, weekly_harmonics=0, use_context=False),
    "lags1_12_only": FeatureConfig(
        lag_hours=(), rolling_hours=(), lag_samples=tuple(range(1, 13)),
        daily_harmonics=0, weekly_harmonics=0, use_context=False),
    "lags1_8_only": FeatureConfig(
        lag_hours=(), rolling_hours=(), lag_samples=tuple(range(1, 9)),
        daily_harmonics=0, weekly_harmonics=0, use_context=False),
}

built = {name: build_features(df, profile, cfg) for name, cfg in DEPTH.items()}
common = None
for Xc, _ in built.values():
    common = Xc.index if common is None else common.intersection(Xc.index)
folds_c = rolling_origin(len(common), n_folds=N_FOLDS)
print(f"{len(common)} rows common to every configuration")

rows = []
for name, (Xc, yc) in built.items():
    per_c, _ = run_backtest(
        Xc.loc[common], yc.loc[common], folds_c,
        models=default_point_models(), baselines=[],
        season_lag=profile.daily_period, keep_predictions=False,
    )
    r = summarise(per_c)[["model", "rmse_mean"]]
    r["config"] = name
    rows.append(r)

depth = pd.concat(rows, ignore_index=True)
depth.pivot(index="config", columns="model", values="rmse_mean").round(3)
"""),
    md("""
**The control settles it.** On GP, `full_plus_lags1_24` scores 8.65 against
`lags1_24_only` at 8.67 — indistinguishable. Adding the dense window *to* the full
design recovers the entire gain, so the improvement is window depth and not the removal
of Fourier terms or context flags, which stay neutral exactly as notebook 01's ablation
found. Random forest goes 10.56 → 8.65, an **18% improvement from lag depth alone**.

On Robi the same ladder moves 20.72 → 20.53, which is nothing.

This tempers a claim made elsewhere in this project: §5 of the paper reads the value of
learning as invisible at one step because the margin over persistence is 14%. At depth
24 that margin is 29–35% on GP. The lead-time argument still holds in direction — the
naive forecaster decays far faster — but the multi-horizon, allocation and coverage
studies all inherit the sparse window, and re-running them at depth is not done yet.
"""),
    md("## Figure"),
    code("""
order = summary[summary["model"].isin(family)].sort_values("rmse_mean")
is_seq = order["model"].isin(["cnn", "lstm", "gru", "rnn"])

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh(order["model"], order["rmse_mean"],
        xerr=order["rmse_std"], capsize=3,
        color=["#2f6f9f" if s else "#9aa5b1" for s in is_seq])
ax.axvline(summary.loc[summary["model"] == "persistence", "rmse_mean"].iloc[0],
           color="#c0392b", ls="--", lw=1.3, label="persistence")
ax.invert_yaxis()
ax.set_xlabel("RMSE (mean +/- sd over 8 folds)")
ax.set_title(f"{OPERATOR.upper()} - sequence models (blue) vs trees, same 24-lag window")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / f"fig10_sequence_{OPERATOR}.png", dpi=200, bbox_inches="tight")
"""),
    md("""
## What to take away

1. **The original's model ranking was an artefact of an unequal split**, not a finding
   about architectures.
2. **On GP the CNN does win**, by 9% over the best tree, and the margin survives a
   Diebold–Mariano test under FDR correction. On Robi nothing separates.
3. **The bigger lever is the lag window, not the model class.** Four lags to
   twenty-four buys 18% on GP for any model; the architecture buys 9% on top of that,
   on one operator only.
4. A confounded comparison can still reach a true answer. Separating the conclusion
   from the evidence is the whole point of the corrected protocol.

`experiments/run_sequence.py` runs both operators and writes
`sequence_{gp,robi}_{perfold,summary,dm}.csv` and `sequence_lag_depth.csv`.
"""),
]


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        path = HERE / name
        path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(HERE.parent)}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
