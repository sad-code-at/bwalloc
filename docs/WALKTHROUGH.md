# Walkthrough

**Read this first if you are picking the project up.** It explains what was inherited, what
was changed, why each change was needed, and what each one bought — in that order, with the
numbers attached.

It assumes you know regression, RMSE, train/test splits, decision trees, and what
overfitting is. Everything specific to this project — irregular sampling, target leakage,
rolling-origin backtesting, quantile forecasting, conformal calibration, multi-horizon
rollout, foundation models — is explained from scratch. You do not need to read any source
code to follow it.

The other three documents serve different purposes: `README.md` states the results,
`STATUS.md` records every result with its caveat, `paper/paper.md` is the publication draft.
This one teaches.

---

## Contents

- [Part 0 — Run it](#part-0--run-it)
- [Part 1 — What was inherited, and the three things wrong with it](#part-1--what-was-inherited-and-the-three-things-wrong-with-it)
- [Part 2 — The vocabulary](#part-2--the-vocabulary)
- [Part 3 — The five contributions](#part-3--the-five-contributions)
- [Part 4 — Reading the outputs](#part-4--reading-the-outputs)
- [Part 5 — Defending it](#part-5--defending-it)

---

# Part 0 — Run it

## Hardware

**Nothing here needs a GPU.** Every number in the project was produced on CPU on an ordinary
laptop, the foundation model included — Chronos-Bolt-small is 48M parameters and runs a
900-point series in seconds. A free Colab CPU runtime is enough. If you have a GPU it will be
slightly faster and change no result.

## Order

Run them in this order. Later scripts read the CSVs earlier scripts write, so the order is
not cosmetic.

```bash
cd "D:/L4-T-1/EEE 402/project/bwalloc"
pip install -r requirements.txt

PYTHONPATH=src python -m pytest tests/ -q   #  47 gates      ~30 s
python experiments/run_audit.py             #               ~1  min
python experiments/run_benchmark.py         #               ~4  min
python experiments/run_allocation.py        #               ~10 min
python experiments/run_horizon.py           #               ~20 min
python experiments/run_transfer.py          #               ~2  min
python experiments/run_foundation.py        #               ~7  min
python experiments/run_coverage_gate.py     #               instant
python notebooks/_build.py                  #   regenerate the 8 notebooks
```

`run_foundation.py` needs `chronos-forecasting` (in `requirements.txt`); it downloads ~200 MB
of weights on first run. `run_coverage_gate.py` fits nothing — it only reads what
`run_allocation.py` and `run_horizon.py` already wrote, which is why it is instant and why it
must run last.

## What each one produces, and how to tell it worked

| Script | Writes | Sanity check |
|---|---|---|
| `run_audit.py` | `audit_sampling`, `audit_acf_*`, `audit_flags_*`, `audit_groups_*` | `audit_sampling.csv` must say GP `median_gap_min` **86**, Robi **99**. If it says 60, you loaded the wrong file. |
| `run_benchmark.py` | `benchmark_*_summary`, `_perfold`, `_dm`, `ablation_*` | `benchmark_gp_summary.csv`: `random_forest` RMSE ≈ **10.41**, `persistence` ≈ **12.12**, and `beats_persistence` is `True` for exactly three models. |
| `run_allocation.py` | `pareto_*`, `cost_*`, `savings_*`, `allocation_*` | `cost_gp.csv` has 7 rows, one per policy family, and `conformal_adaptive` shows `saving_vs_tuned_baseline` ≈ **0.075**. |
| `run_horizon.py` | `horizon_*`, `horizon_alloc_*`, `horizon_strategy_*` | `horizon_gp.csv` has 5 horizons × 5 models; at 24 h, `persistence_h` and `seasonal_naive_17` are **identical** (17 steps *is* the seasonal period — a good sign the period was measured, not assumed). |
| `run_transfer.py` | `transfer_robi_to_gp.csv` | At `horizon_hours = 6.0`, `persistence` = **26.49** and `robi_cold` at `k_days=3` = **20.39**. |
| `run_foundation.py` | `foundation_accuracy`, `foundation_calibration` | `foundation_accuracy.csv` has 2 operators × 5 horizons × 8 folds = 80 rows. |
| `run_coverage_gate.py` | `coverage_gate.csv` | Filter `group == "ALL"`: `aci` must pass **30 of 30**, `marginal` **2 of 30**. |

Figures land in `paper/figures/` when you run `07_paper_figures.ipynb`; they read only from
`experiments/results/`, never from a live model, so the paper rebuilds without refitting
anything.

## In Colab

Each notebook opens with a bootstrap cell that pip-installs, finds the repo in both Colab and
local, and sets seeds. The only thing to get right is that `src/` is on the path
(`PYTHONPATH=src`, or the bootstrap cell's `sys.path.insert`). Notebooks are *generated* from
`notebooks/_build.py` — edit the builder, not the `.ipynb`, or your change is lost on the next
rebuild.

---

# Part 1 — What was inherited, and the three things wrong with it

The senior's project is nine Colab notebooks comparing about ten forecasting models on two
Bangladeshi operator traces: Grameenphone Dhaka (905 rows) and Robi Dhaka-3 (888 rows), each
about two months of bandwidth demand in Gbps. The original code is untouched at
`../ML-Based-Dynamic-Bandwidth-Allocation-for-Mobile-Data-Usage-main/`.

Three errors were found and reproduced. Each one is now a section of the paper's methodology
— **not an accusation**. The right sentence in a viva is *"the earlier study's protocol had a
sampling assumption we corrected, and here is what the correction costs"*, said in the same
tone you would use about any published baseline.

### 1. The data is not hourly, but every model assumed it was

Measure the gap between consecutive timestamps and you get a median of **86 minutes** on GP
and **99 minutes** on Robi. So one day is 16.74 samples on GP and 14.55 on Robi — not 24.

Every notebook used `lag_24` as "the same time yesterday". At 86 min/sample, 24 samples is
**34.4 hours**. That is not a day; it is a day plus ten hours, which lands roughly opposite in
the daily cycle:

| GP | lag in samples | real time span | correlation with today |
|---|---|---|---|
| what was used | 24 | 34.4 h | **−0.35** |
| the true daily lag | 17 | 24.4 h | **+0.45** |

The feature intended to say "yesterday was busy, so today will be busy" was in fact saying the
opposite. `lag_168` and `lag_336`, labelled 7-day and 14-day, are really 10-day and 20-day.

**What it cost.** A seasonal-naive forecaster (predict today = one period ago) scores RMSE
**18.2** at the true period versus **28.4** at lag 24 on GP, and **26.6** versus **74.1** on
Robi — the wrong period is **179% worse**. Every STL decomposition, ACF plot and FFT
conclusion in the original work was stated on a mis-scaled time axis.

**The fix:** measure the sampling interval per operator and derive every seasonal
hyperparameter from it. Never hardcode 24.

### 2. Target leakage

The best-reported result computed a rolling mean like this:

```python
df['Gbps_rolling_mean_3'] = df['Gbps'].rolling(3).mean()   # no .shift(1)
```

`rolling(3)` at row *t* averages rows *t−2, t−1, t*. So `y_t` is inside its own feature
vector, divided by three. The model does not need to forecast; it can partly read the answer.

**What it cost.** Shift the window by one and nothing else:

| | as written | leak removed | change |
|---|---|---|---|
| GP RMSE | 6.54 | **19.30** | **+195%** |
| Robi RMSE | 16.32 | **27.37** | **+68%** |

The corrected GP number is *worse than predicting the training mean* (18.87). The headline
result was almost entirely the leak.

**The fix:** one feature builder used by every model, in which all rolling statistics are
shifted before they are rolled, and features are chosen from an explicit whitelist rather than
"every column except the target".

### 3. No baseline was ever computed

Persistence — `ŷ_t = y_{t−1}`, one line of code — scores RMSE **12.18** on GP and **31.41** on
Robi. Six of the ten originally reported models lose to it or tie. A model comparison without a
naive baseline cannot tell you whether *any* of the models learned anything.

**The fix:** four baselines (persistence, seasonal-naive at the measured period, drift, train
mean) reported in every table, plus MASE, which is RMSE-like but scaled so that **1.0 = as good
as naive** and you can read the verdict off the number.

### And one notebook that had to be deleted

`Robi_to_GP.ipynb` is titled as a Robi→GP transfer study. It loads the GP file for *both*
halves, so it trains and tests on the same operator. It then does:

```python
y_test_offset = [e + 20 for e in y_test]
```

— adds 20 Gbps to the test actuals and plots the *shifted actuals* against the unchanged
predictions, which makes the fit look better than it is. Nothing in it is recoverable. It is
deleted and replaced by the cold-start study in Part 3.

---

# Part 2 — The vocabulary

These are the terms you need to be able to use. Each entry says what it means, why *this*
project needs it, and where it shows up.

### Sampling and features

**Irregular sampling.** The gap between consecutive readings is not constant. Here it is *near*
constant (median 86 min on GP) but not a round number of hours, which is worse than obviously
irregular — it looks regular enough that nobody checked. Everything in Part 1 flows from this.

**Sampling profile.** The measured description of a trace: median gap, samples per day, and
the integer lag closest to one day. GP: 86 min, 16.744 samples/day, daily period 17. Robi: 99
min, 14.545, period 15. Computed once in `data.py` and passed to everything downstream.

**Fourier time features.** Instead of "the value 17 samples ago", give the model
`sin(2πt/86400)` and `cos(2πt/86400)` computed from the *wall-clock timestamp* — plus the same
at 2× and 3× frequency, and a weekly pair. These say "it is 3 p.m." directly, in a way that
does not care how often you sampled. This is the principled fix for irregular sampling:
integer lags are a proxy for time-of-day, Fourier terms *are* time-of-day.

**Target leakage.** Any feature at time *t* that contains information from time *t* or later.
The leak in Part 1 is the classic form. The defence here is a test that shuffles `y` and
asserts every feature column is unchanged — if a feature moves when you scramble the target,
it was reading the target.

### Evaluation

**Rolling-origin backtest** (also: expanding-window, walk-forward). Instead of one train/test
split, make several: train on the first 40%, test on the next chunk; then train on the first
50%, test on the next; and so on, eight times. Every model sees identical folds. You get eight
numbers per model instead of one, so you can report a mean *and a spread* — which matters
enormously when the test set is only ~66 points and the gaps between models are small.

**Lead time and forecast origin.** The *origin* is the moment the forecast is issued; the
*lead time* is how far ahead the target sits. Predicting `y_t` when you already have `y_{t−1}`
is a lead time of one sample — a **nowcast**. It is not useful for provisioning, because you
cannot buy capacity for a moment that has already begun.

**Direct vs recursive multi-horizon.** To forecast 8 steps ahead you can either train a
dedicated model on (features at *t*, target at *t+8*) — **direct** — or run a one-step model
forward eight times, feeding its own predictions back in — **recursive**. Direct needs one
model per horizon; recursive needs one model but compounds its own errors.

**The embargo.** When training a direct model for horizon *h*, the last *h−1* training rows
have targets that had not yet happened when the first test forecast was issued. Including them
leaks the future. Dropping them is the embargo. Without it, long-horizon results are silently
optimistic.

**MASE** (mean absolute scaled error). MAE divided by the MAE of the naive forecaster on the
training data. **Below 1.0 means you beat naive**; above means you did not. Scale-free, so GP
and Robi numbers are comparable.

**Diebold–Mariano test.** Asks whether model A's errors are *significantly* smaller than model
B's, given that they are measured on the same data and so are correlated. Without it, "A beat
B by 0.3 RMSE" on 66 test points is a coin flip. Applied here across folds, with
Benjamini–Hochberg correction because many pairs are tested at once.

**Clopper–Pearson interval.** An exact confidence interval for a proportion. Coverage measured
on 179 points is itself an estimate; quoting "88.6%" without saying "[80.7%, 91.2%]" is the
same class of error this project exists to correct. Used everywhere a coverage number appears.

### Allocation

**The cost model.** If you allocate capacity `A` and demand turns out to be `y`:

```
cost = c_over · (A − y)⁺  +  κ · c_over · (y − A)⁺
```

In words: you pay for capacity you allocated and did not use, and you pay **κ times more** for
every Gbps you were short. κ = 10 means one unit of SLA breach hurts as much as ten units of
idle capacity. κ is the operator's number, not a tuned hyperparameter.

**Quantile regression and pinball loss.** A normal regressor trained on squared error predicts
the *mean*. Train it on **pinball loss** instead — which penalises being under by τ and over by
(1−τ) — and it predicts the **τ-quantile**: the level demand stays below τ of the time.

**The result the whole allocation layer rests on.** The allocation that minimises the cost
above is not the predicted mean. It is the τ-quantile with

```
τ* = κ / (1 + κ)          κ = 10  →  τ* = 0.909
```

In words: **the operator's cost ratio chooses the quantile for you.** You never tune τ on test
data. This is the project's one clean theoretical hook, and it is the equation to be able to
write on a board.

**Conformal calibration.** A quantile model's 90% level is only nominally 90% — nothing forces
it to be right. Split conformal fixes this without assuming anything about the error
distribution: hold out a calibration slice, collect residuals `r = y − ŷ`, take their
(1−α) quantile `q̂`, and allocate

```
A = ŷ + q̂            (or  A = ŷ · (1 + q̂)  in the relative variant)
```

The guarantee is *distribution-free* and finite-sample, which is exactly why it suits 900
points — you cannot verify a parametric error model on this much data.

**Coverage.** The fraction of test points where `A ≥ y`, i.e. where you did not under-provision.
If you asked for τ = 0.90 you want to measure ≈ 0.90.

**Marginal vs conditional coverage.** *Marginal* is coverage over everything pooled.
*Conditional* is coverage within a subgroup — rainy days, say. A method can hit 90% overall
while delivering 77% during rain and 95% otherwise. The failures then cluster exactly where
the network is most stressed, which is the worst possible place for them.

**Locally-adaptive (normalised) conformal.** Fit a small second model `σ̂(x)` that predicts how
*large the error will be* from the context flags and time features. Conformalise the normalised
residual `r/σ̂(x)`. The margin becomes `σ̂(x)·q̂` — it widens automatically under rain. Crucially,
every calibration residual still contributes to the single `q̂`, which is what makes it usable
at n = 905.

**Mondrian (group-conditional) conformal.** Split residuals into disjoint groups and calibrate
a separate `q̂` per group. Cleaner theory — exact per-group coverage — but data-hungry, because
each group needs its own residuals.

**The estimability bound.** A group with *n* calibration residuals cannot express a quantile
finer than `1/(n+1)`:

```
n ≥ ceil( τ / (1 − τ) )        τ = 0.95 → 19 residuals;  τ = 0.99 → 99
```

The elevated-risk group here has ~199 rows, so τ = 0.95 is fine and τ = 0.99 is not. The code
**refuses** rather than silently clipping — claiming 99% per-group coverage off 39 residuals
would be a subtler version of the lag-24 error.

**Online / adaptive conformal inference (ACI).** Split conformal assumes *exchangeability* —
loosely, that calibration and test residuals are drawn alike. A 55-day trace with a trend does
not satisfy that, and the failure is one-sided (you always under-cover). ACI drops the
assumption: after each step, if you were breached, ask for a slightly higher level; if not,
relax. `α ← α + γ(α_target − err)`. Long-run coverage converges without any exchangeability
assumption. It is strictly causal — `y_t` only affects steps after *t* — and there is a test
pinning that.

### Transfer and foundation models

**Cold start.** A newly deployed cell site has no history. How long until its own forecaster is
worth using?

**Warm start / fine-tuning.** Train on operator A, then continue training the same booster on
operator B's first *k* days rather than starting from scratch. Requires per-operator z-scoring
first, because the two traces have different scales.

**Zero-shot foundation model.** A large model pretrained on a corpus of unrelated time series
(Chronos-Bolt here), applied to your data **with no training on it at all**. You hand it the
last *N* values and it returns a forecast. The live question in the 2025–26 literature is
whether, at 900 points per site, bespoke training is even worth it.

**Timestamp-blindness.** Chronos-Bolt reads a bare sequence of numbers. It has no clock. It can
learn "there is a cycle of about 15 steps" from the values, but it cannot know that the cycle
is a *day* — and if the day is 14.545 samples, no integer stride holds phase. This turns out
to be the whole story of Part 3's foundation-model result.

---

# Part 3 — The five contributions

Each section is the same five beats: **the question → the idea in words → what was done → the
number → where it lives**.

---

## C1 — A corrected evaluation protocol

**The question.** Are any of the reported comparisons meaningful?

**The idea.** Before adding anything, make the measuring instrument trustworthy. Derive
seasonality from the *measured* sampling rate; make time-of-day a continuous Fourier feature
instead of an integer lag; shift every rolling statistic before rolling it; score everything on
identical rolling-origin folds against naive baselines; test the differences for significance.

**What was done.** `data.py` measures the sampling profile. `features.py` is the single
leak-safe builder. `splits.py` generates the eight folds. `baselines.py` supplies the four
naives. `stats.py` runs Diebold–Mariano under FDR control. All of it enforced by
`tests/test_bwalloc.py`.

**The number — and this is the honest part.** The corrected benchmark, GP, eight folds:

| model | RMSE | MASE | beats persistence |
|---|---|---|---|
| random_forest | **10.41 ± 1.19** | 0.51 | yes |
| xgboost | 10.80 ± 1.52 | 0.55 | yes |
| ridge | 11.26 ± 0.88 | 0.57 | yes |
| persistence | 12.12 ± 0.92 | 0.46 | — |
| rolling_mean_17 | 16.16 | 0.90 | no |
| train_mean | 18.23 | 1.06 | no |
| seasonal_naive_17 | 18.31 | 1.04 | no |

And the ablation, which is the result people miss:

| GP feature set | features | RMSE |
|---|---|---|
| lags_only | 8 | **9.98** |
| no_fourier | 17 | 10.00 |
| original_style | 19 | 10.11 |
| original_correct_period | 19 | 10.12 |
| full (everything) | 25 | 10.41 |

**Correcting the features does not improve RMSE.** The simplest feature set is marginally the
best, and every difference is inside the fold-to-fold spread (±1.19). Say this out loud rather
than hiding it: the correction is not an accuracy trick, it is what makes every *later*
comparison mean something. The lag-24 error damages the *seasonal-naive baseline* enormously
(+179% on Robi) and the tree models barely at all, because a tree given 19 correlated lags will
route around a bad one. That asymmetry is precisely why the original comparison was
untrustworthy — the errors did not hit all models equally.

**Where.** `run_audit.py`, `run_benchmark.py` → `audit_*.csv`, `benchmark_*.csv`,
`ablation_*.csv`. Figures **1** (autocorrelation by lag) and **2** (benchmark with error bars).
Notebooks `00`, `01`.

---

## C2 — Forecasting at a real lead time

**The question.** The original setup predicts `y_t` holding `y_{t−1}`. How far ahead can this
actually be run, and does the ranking survive?

**The idea.** Provisioning needs lead time. Train direct models at 1.5, 3, 6, 12 and 24 hours,
with three things made explicit that multi-step evaluations usually get wrong:

1. **History features are read at the forecast origin; calendar features at the target.** You
   do not know demand at *t+8*, but you *do* know that *t+8* is a Friday at 9 p.m. Splitting
   features by role is what makes that legitimate.
2. **The naive baseline is re-derived per horizon.** An 8-step model must be compared to
   `y.shift(8)`, never to `y.shift(1)`. Comparing against one-step persistence flatters every
   long-horizon model, and it is a common error.
3. **The embargo.** Drop the last *h−1* training rows.

**The number — this reframes the whole project.** Best learned model versus persistence *at
that same lead*:

| lead | GP model | GP naive | advantage | Robi model | Robi naive | advantage |
|---|---|---|---|---|---|---|
| 1.4 h | 10.40 | 12.12 | **14%** | 20.77 | 29.75 | **30%** |
| 2.9 h | 11.85 | 17.41 | 32% | 24.04 | 45.07 | 47% |
| 5.7 h | 12.36 | 25.11 | 51% | 27.23 | 65.97 | 59% |
| 11.5 h | 12.79 | 31.16 | **59%** | 27.74 | 75.95 | **64%** |
| 24.4 h | 12.82 | 18.26 | 30% | 22.69 | 26.49 | 14% |

Read the *columns*, not the rows. Across a 17× longer horizon the learned model degrades by
23% (10.40 → 12.82) while the naive baseline degrades by 157%. **The original one-step protocol
measured these models in the single regime where they look worst.** The 24-hour column
recovers because at 24 h persistence *becomes* seasonal-naive — one day back — and the daily
cycle carries it.

Direct beats recursive by **86%** at a 24-hour lead on GP. On Robi recursive is marginally
ahead at intermediate horizons. Reported as it is, not smoothed into a clean win.

**Where.** `run_horizon.py` → `horizon_*.csv`. `forecast.py` holds the role split, the
per-horizon baselines and both rollout strategies; `splits.py` holds the embargo. Figure **5**.
Notebook `04`.

---

## C3 — The allocation layer the title promises

**The question.** Nothing in the original repo allocates bandwidth. Every notebook stops at
predicting Gbps and scoring RMSE. So what does a forecast actually buy an operator?

**The idea.** RMSE is the wrong objective. Under-provisioning breaks the SLA;
over-provisioning wastes capacity; the two cost wildly different amounts. A squared-error loss
optimises for neither. Write the asymmetric cost, and the optimal allocation falls out as a
quantile — with κ selecting which one (see Part 2). Then use conformal calibration to make the
achieved service level match the promised one.

**What was done.** `allocation.py` (cost model, quantile policy, fixed-margin and static-peak
baselines), `conformal.py` (split, locally-adaptive, Mondrian, relative, ACI), `pipeline.py`
(the end-to-end backtest). Compared against what operators actually do: **point forecast × a
fixed margin** (ŷ × 1.15, 1.30, 1.50) and **static peak allocation**.

**The number.** Cost at κ = 10, evaluated at the a-priori τ\* = 0.909 — no test-set
information — against the fixed-margin rule given its **best** margin chosen with hindsight.
That handicap is deliberate and biases the comparison *against* the proposed method:

| κ = 10 | cost at τ\* | vs hindsight-tuned fixed margin |
|---|---|---|
| GP, context-adaptive conformal | **19.27** | **−7.5%** |
| GP, marginal conformal | 21.35 | +2.4% (worse) |
| GP, fixed margin (best = 1.15) | 20.84 | — |
| Robi, context-adaptive conformal | **49.34** | **−14.8%** |
| Robi, marginal conformal | 59.09 | +2.0% (worse) |
| Robi, fixed margin (best = 1.30) | 57.93 | — |

Read the *pattern*: conditioning the margin on predicted uncertainty beats the tuned heuristic
on both operators; calibrating one margin for all conditions loses to it on both. That is a
statement about *conditioning*, and it is more interesting than a win.

**What is not claimed.** The original plan wanted "at an equal 1% SLA violation rate we
provision X% less capacity". That comparison **does not hold** and is not made. The reason is
structural, not a modelling failure: the fixed-margin frontier is drawn by sweeping the margin
and reading off the point that happens to hit your SLA — which requires knowing the test-set
outcome. Comparing an honest a-priori method to a hindsight-optimal frontier at a matched
operating point is a rigged race. Cost at τ\* is the comparison that can actually be made
before the fact, so cost is what is reported.

**Where.** `run_allocation.py` → `pareto_*.csv`, `cost_*.csv`, `savings_*.csv`. Figures **3**
(the risk/capacity frontier) and **7** (cost). Notebook `02`.

---

## C4 — Making the allocation context-conditional

**The question.** The GP trace carries nine hand-labelled contextual flags — `is_rain`,
`is_drama`, `is_offer`, `is_political_gathering`, `is_powercut`, `is_holiday`, `is_weekend`,
`is_event`, `event`. They are the genuinely distinctive thing about this dataset; you cannot
download hand-annotated local context for a Dhaka cell site. They were fed to every model as
raw binary features and never once tested. Are they worth anything?

**The idea, and it is the sharpest thing in the project.** Test them, and they mostly fail as
*mean* predictors. But that is the wrong test. Look at what they do to the *spread*:

| flag | n | Δ mean | p (level) | resid σ on | resid σ off | p (variance) |
|---|---|---|---|---|---|---|
| `is_rain` | 131 | +4.50 | 0.026 | **21.80** | 16.38 | **4.7e−07** |
| `is_political_gathering` | 101 | +2.69 | 0.166 | 18.36 | 17.19 | 0.26 |
| `is_offer` | 235 | −5.73 | 5e−06 | 15.86 | 17.59 | 0.013 |
| `is_powercut` | 118 | −2.63 | 0.130 | 17.67 | 17.27 | 0.83 |
| `is_event` | 275 | −0.23 | 0.856 | 17.54 | 17.25 | 0.57 |
| `is_drama` | 147 | −1.52 | 0.337 | 17.36 | 17.33 | 0.77 |
| **`is_weekend`** | 261 | +0.37 | **0.770** | 16.86 | 17.53 | 0.94 |
| `is_holiday` | 49 | −3.99 | 0.111 | 16.36 | 17.37 | 0.39 |

Two things fall out. First, a small honest negative result: **`is_weekend` carries no signal at
all** (p = 0.77) — and *both source CSV filenames advertise it*
(`..._with_weekend_encoded_...`, `..._with_weekend_flag`). Six of nine flags are
indistinguishable from noise on the mean.

Second, the contribution: **`is_rain` raises residual σ from 16.4 to 21.8 Gbps — a 33% increase
in uncertainty — while barely moving the level.** *Context here acts on the variance, not the
mean.* A point forecaster with a rain dummy cannot express that; it can only shift its
prediction up or down. A context-conditional *interval* is exactly the object that can.

This converts the dataset's weak flags from mediocre demand predictors into useful **risk**
signals. They do not need to predict demand well — they only need to mark when the forecast is
less trustworthy, and `is_rain` demonstrably does.

**What was done.** Group the flags into two disjoint groups by priority —
`elevated_risk = rain ∪ gathering` (~199 rows) and `baseline` — because four groups would put
`gathering` at ~20 calibration residuals and below the estimability bound. Then calibrate three
ways: marginal (one margin), locally-adaptive (margin scaled by `σ̂(x)`), Mondrian (a margin per
group). Report per-group coverage **with Clopper–Pearson intervals and the group's calibration
size stated.**

**The number.** GP, elevated-risk group, 1.5 h lead, 179 points:

| τ | method | coverage | 95% CI | decisive? |
|---|---|---|---|---|
| 0.95 | marginal | 0.866 | [0.807, 0.912] | **yes — excludes 0.95** |
| 0.95 | adaptive | 0.899 | [0.846, 0.939] | no |
| 0.95 | mondrian | 0.933 | [0.886, 0.965] | no |
| 0.90 | marginal | 0.771 | [0.702, 0.830] | **yes** |
| 0.90 | adaptive | 0.816 | [0.751, 0.870] | yes |
| 0.90 | ACI | **0.911** | [0.859, 0.948] | **passes** |

**Quote this carefully, and the care is the point.** The *failure* is decisive: marginal
calibration at τ = 0.95 delivers 86.6% inside the elevated-risk group and its interval excludes
the nominal level. The *repair* is directional but not decisive: adaptive's interval
[0.846, 0.939] overlaps marginal's heavily, so on 179 points you cannot claim the two are
distinguishable. Saying "context-conditional calibration restores coverage" without that
caveat would be overclaiming, and an examiner who knows binomial intervals will find it.

An unexpected result: the only **decisive** per-group improvement comes from ACI (0.911 vs
0.771 at τ = 0.90, non-overlapping intervals) — a method that is not context-conditional at
all. It adapts in *time* rather than by *group*, and on this data that turns out to matter
more. This is reported as found. It is also what makes a per-group *online* calibration the
clearest open question the work raises.

**The falsifiability test.** On Robi, no flag shows elevated variance — so the method should do
nothing. It does nothing. The claim is **correctly null on Robi**, which is much stronger
evidence that C4 is measuring something real than a second win would be.

**Where.** `context.py` (grouping, `σ̂` model, permutation flag value), `conformal.py`,
`run_allocation.py` → `allocation_*.csv`; `run_coverage_gate.py` → `coverage_gate.csv`.
Figures **4** (coverage by group) and **6** (flag audit). Notebook `03`.

---

## C5 — Cold start and zero-shot foundation models

Two studies that share one explanation.

### Cold start: how much history does a new cell site need?

**The question.** A newly deployed site has no data. Do you wait, or borrow a model from
elsewhere?

**What was done.** Four arms at each of *k* = 1, 3, 5, 7, 14, 21 days of GP history:
`persistence`; `gp_only(k)` trained on *k* days of GP alone; `robi_cold` trained on Robi and
applied to GP with **zero** GP training data; `transfer(k)` warm-started from Robi and
fine-tuned on *k* days. Per-operator z-scoring makes the scales commensurate.

**The number** (6-hour lead, RMSE; persistence = 26.49):

| k days | gp_only | transfer | robi_cold |
|---|---|---|---|
| 3 | 20.61 | **19.65** | 20.39 |
| 5 | 19.81 | 19.87 | 20.01 |
| 7 | 18.98 | 19.91 | 19.74 |
| 14 | **14.82** | 15.63 | 19.06 |
| 21 | 15.86 | 17.44 | 19.04 |

A model that has **never seen GP** scores 20.39 against persistence's 26.49 — **23% better with
zero local training data**. Transfer helps at 3 days (+4.6% over own-data) and hurts by 21 days
(−10%). **The crossover is about one week.** That is a directly operational answer: for the
first week, borrow; after that, train locally.

Note what makes this possible: because lags are defined in *wall-clock hours* rather than
sample counts, a model trained at 99 min/sample applies to a trace at 86 min/sample at all.
The C1 correction is what buys the transfer result.

### Zero-shot: is bespoke training worth it at 900 points?

**What was done.** Chronos-Bolt-small, zero-shot, on the same folds and horizons as everything
else. It gets the last ≤2048 observations and returns quantiles directly.

**The number, and the answer depends entirely on the operator:**

| lead | GP zero-shot | vs trained | vs persistence | Robi zero-shot | vs trained | vs persistence |
|---|---|---|---|---|---|---|
| 1.4 h | 11.18 | +7.5% | −7.8% ✓ | 30.54 | +47.0% | +2.6% ✗ |
| 2.9 h | 13.53 | +14.2% | −22.3% ✓ | 38.84 | +61.5% | −13.8% ✓ |
| 5.7 h | 13.50 | +9.2% | −46.2% ✓ | 40.34 | +48.1% | −38.9% ✓ |
| 11.5 h | 14.41 | +12.7% | −53.8% ✓ | 42.56 | +53.4% | −44.0% ✓ |
| 24.4 h | 15.20 | +18.6% | −16.8% ✓ | 43.63 | **+92.3%** | **+64.7% ✗** |

On GP it trails per-operator training by 7.5–18.6% and beats persistence at *every* lead — a
genuinely usable model with no training. On Robi it trails by 47–92% and **loses to a one-line
baseline** at the shortest and longest leads.

**Why — and this is the part worth writing up.** Chronos-Bolt reads a bare sequence with no
timestamps. It can infer "there is a cycle of roughly *N* steps" but only at integer strides.
The daily cycle occupies:

| | samples/day | nearest integer | misregistration |
|---|---|---|---|
| GP | 16.744 | 17 | **0.256** |
| Robi | 14.545 | 15 | **0.455** |

Robi's daily cycle is almost exactly halfway between two integers. A timestamp-blind model
cannot hold phase against it, and the drift accumulates with horizon — which is why the worst
result in the whole table is Robi at 24 hours, *exactly* where the daily cycle is the entire
signal. **Measuring the sampling interval tells you in advance whether a zero-shot model will
work.** That is a cheap diagnostic for a decision operators now actually face, and it is C1
paying off a second time.

**And its quantiles are not usable as they come.** Chronos-Bolt was trained on levels 0.1–0.9,
so anything above 0.9 is silently clipped — via τ\* = κ/(1+κ) that caps the expressible cost
ratio at **κ = 9**, below the 10–20 the provisioning literature uses. There is a test pinning
this. Averaged over folds:

| operator | τ | zero-shot coverage | after conformal |
|---|---|---|---|
| GP | 0.80 | 0.790 | 0.815 |
| GP | 0.95 | 0.886 (clipped) | 0.941 |
| Robi | 0.80 | **0.662** | 0.813 |
| Robi | 0.95 | 0.785 (clipped) | 0.950 |

Robi's nominal 80% interval delivers 66%. Conformal calibration repairs every level on both
operators — so the right way to use a foundation model here is **as a forecaster inside the C3
allocation pipeline**, not as an allocator on its own.

**Where.** `run_transfer.py` → `transfer_robi_to_gp.csv`; `run_foundation.py` →
`foundation_accuracy.csv`, `foundation_calibration.csv`. Figures **8** and **9**. Notebooks
`05` and `06`.

---

## The coverage gate — the project checking its own claim

Worth its own note, because it is the part most likely to impress and most likely to be
misread.

The verification plan demanded a direct check: on real test folds, measured coverage must land
within ±2 percentage points of nominal. Enforced across 30 configurations (2 operators × 5 lead
times × 3 levels):

| method | marginal configurations passed |
|---|---|
| split conformal (marginal) | **2 / 30** |
| locally-adaptive | 6 / 30 |
| Mondrian | 5 / 30 |
| **ACI (online)** | **30 / 30** |

Split conformal *fails on real data*, and essentially every miss is **under**-coverage — never
over. That one-sidedness is diagnostic rather than random noise: it is the exchangeability
assumption breaking, exactly as the theory says it should on a 55-day trace with a trend. ACI,
which makes no exchangeability assumption, passes everything at every lead time from 1.4 to 24
hours on both operators.

Running a gate that your own headline method fails, and reporting it, is the strongest single
signal of honesty in the project. Do not soften it.

---

# Part 4 — Reading the outputs

Where to look, and what to be careful about.

| Question | File | Column to read |
|---|---|---|
| How fast is the data sampled? | `audit_sampling.csv` | `median_gap_min`, `samples_per_day`, `daily_period` |
| Do the context flags matter? | `audit_flags_gp.csv` | `p_level` for the mean, **`p_variance` for the real story** |
| Which model wins, and does it beat naive? | `benchmark_*_summary.csv` | `rmse_mean` ± `rmse_std`, `beats_persistence` |
| Is the difference significant? | `benchmark_*_dm.csv` | Diebold–Mariano p-values, FDR-adjusted |
| Do the corrected features help? | `ablation_*.csv` | compare `rmse_mean` — they don't, and that's the finding |
| How far ahead can it forecast? | `horizon_*.csv` | model rows vs `persistence_h` **at the same horizon** |
| Does the allocator save money? | `cost_*.csv` | **`cost_at_tau_star`** and `saving_vs_tuned_baseline` |
| Is calibration actually working? | `coverage_gate.csv` | `coverage`, `ci_lo`, `ci_hi`, `passes`, `decisive` |

### Four traps

1. **`cost_*.csv` has two saving columns.** `saving_vs_tuned_baseline` is the a-priori number
   at τ\* — **quote this one**. `saving_best_vs_tuned_baseline` uses the best setting chosen
   with hindsight; it is there for the ablation, not for the abstract.
2. **`passes` and `decisive` are different questions** in `coverage_gate.csv`. `passes` = the
   point estimate is within ±2 points of nominal. `decisive` = the Clopper–Pearson interval
   *excludes* nominal, so the result is not attributable to sampling noise. A method can fail
   without being decisive (too few points to tell) and that is a different, weaker statement.
3. **Compare against the naive at the same horizon.** In `horizon_*.csv`, `persistence_h` is
   `y.shift(steps)`, not `y.shift(1)`. Comparing a 12-hour model against 1-step persistence
   inflates the advantage enormously — it is the specific error C2 exists to avoid.
4. **Against *seasonal*-naive the long-horizon picture is more modest**, and you should know
   this before someone points it out. On Robi at 6–12 h the learned models are actually a few
   percent *behind* seasonal-naive-at-period-15 (27.23 vs 26.27), even while beating
   persistence by 59%. The daily cycle is doing most of the work at those leads. The honest
   framing is "learning adds most where the naive baseline degrades fastest", not "the models
   dominate everywhere".

### Per-group coverage

Never quote a per-group coverage number without its interval and its *n*. On 179 points, an
86.6% coverage has a 95% interval of roughly [80.7%, 91.2%] — nearly five points wide either
side. This is not pedantry; it is the same species of error as calling 34.4 hours "one day".

---

# Part 5 — Defending it

**"Your corrected features didn't improve accuracy. So what was the point?"**
Correctness of measurement, not accuracy. The lag-24 error damaged the seasonal-naive baseline
by 179% and the tree models by almost nothing — so it corrupted the *comparison* while leaving
the winners' numbers roughly intact. Until the instrument is right, no ranking from it means
anything. Also, the correction is what makes cross-operator transfer possible at all: lags in
wall-clock hours travel between an 86-minute trace and a 99-minute one; lags in sample counts
do not.

**"You only have 900 points from two cell sites."**
Stated in the limitations section, first paragraph. That is why the method is distribution-free
conformal rather than a parametric error model; why everything is reported with fold spreads
and binomial intervals; why the estimability bound is enforced in code rather than assumed; and
why the contribution is framed as a *method* for sparse, irregularly sampled traces rather than
a claim about Bangladeshi mobile traffic. Small *n* is part of the problem statement, not
something being hidden.

**"Your own coverage check failed — 2 of 30."**
Yes, and it is in the abstract. Split conformal assumes exchangeability; a 55-day trace with a
trend does not provide it; the failures are one-sided under-coverage, exactly as the theory
predicts. ACI drops the assumption and passes 30 of 30. A negative result that is explained and
then repaired is a stronger contribution than a positive result that was never stress-tested.

**"Why is there no capacity-saving number? The plan asked for one."**
Because it cannot be measured honestly here. The fixed-margin frontier can only be drawn with
hindsight — you sweep the margin and read off the point that hit your SLA on the test set.
Matching operating points against a hindsight-optimal frontier is a rigged comparison. Cost at
the a-priori τ\* = κ/(1+κ) is the comparison that exists before the fact, and on that the
method wins on both operators (−7.5%, −14.8%) while the heuristic is given its *best* margin.

**"Isn't 47 tests excessive for a student project?"**
Each one pins a specific failure this project corrected: leakage, the sampling assumption, the
inverted split, the estimability bound, the multi-horizon embargo, ACI's causality, the
foundation model's quantile ceiling. They exist so a refactor cannot silently reintroduce an
error the paper claims to have fixed. Given that the original headline result was a leak, that
is proportionate.

**"Why should the flags matter if most of them are statistically null?"**
Because they were tested on the wrong quantity. They are weak predictors of the *level* of
demand and that is what everyone checked. `is_rain` raises residual σ by 33% (p = 4.7e−07) —
it predicts *when the forecast is unreliable*. That is a different and, for provisioning, more
useful thing: an interval can act on it, a point forecast cannot.

**"Your context-conditional repair isn't statistically decisive."**
Correct, and it is reported that way. The *failure* of marginal calibration is decisive — its
interval excludes nominal. The *repair* is directional; adaptive's interval overlaps marginal's
on 179 points. The decisive improvement comes from ACI, which adapts in time rather than by
group. The honest conclusion is that temporal adaptation matters more than group conditioning
on this data, and that combining the two is the open question.

**"Why is the foundation model worse on Robi?"**
Not bad luck — predictable in advance from the sampling rate. Chronos-Bolt reads a bare
sequence with no clock, so it can only lock onto integer-length cycles. GP's day is 16.744
samples (0.256 from an integer); Robi's is 14.545 (0.455 — almost exactly halfway). Phase drift
accumulates with horizon, which is why the worst cell in the table is Robi at 24 hours, where
the daily cycle *is* the signal. Measuring the sampling interval is a one-line diagnostic that
tells an operator beforehand whether a zero-shot model is worth trying.

---

## Where everything is

```
src/bwalloc/       data · features · splits · baselines · models · forecast
                   context · conformal · allocation · metrics · stats
                   evaluate · pipeline · plots
experiments/       run_audit · run_benchmark · run_allocation · run_horizon
                   run_transfer · run_foundation · run_coverage_gate
   results/        every CSV; figures read from here and nowhere else
notebooks/         00 audit · 01 benchmark · 02 allocation · 03 context
                   04 horizon · 05 transfer · 06 foundation · 07 figures
                   (generated by _build.py — edit the builder, not the .ipynb)
tests/             47 gates
paper/             paper.md + figures/
```

`git log --oneline` is the chronology; each commit message says *why*, not what.
`STATUS.md` has every result with its caveat and what remains to be done.
