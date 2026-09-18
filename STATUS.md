# Project status

Last updated: 2026-09-18. Working notes so this can be picked up cold in a new session.

`docs/WALKTHROUGH.md` is the teaching document — run order, vocabulary, what each change
bought, and how to defend it. This file is the working record; that one is the explanation.

`docs/PROVENANCE.md` maps every method to the paper it came from, separates borrowed
machinery from this project's own contributions, and marks which citations have actually
been verified. **Add a row to it in the same commit as any new method or finding** — it
exists so that "where did this come from?" always has a citable answer.

Plan file: `C:\Users\User\.claude\plans\nested-plotting-bengio.md` (approved).
Original senior's code, untouched, is at
`D:\L4-T-1\EEE 402\project\ML-Based-Dynamic-Bandwidth-Allocation-for-Mobile-Data-Usage-main\`.
All new work is in `D:\L4-T-1\EEE 402\project\bwalloc\`.

**The repository is under git and pushed to GitHub** — `main`, remote `origin` at
<https://github.com/sad-code-at/bwalloc> (**public**, by the user's decision). Every session's work is
committed, so `git log` is the record of what changed and `git show <sha>` recovers any
earlier state. Commit *and push* before stopping: `git push`. The remote is the offsite
backup, so a lost laptop no longer loses the project.

---

## Done and verified

**Scaffold + library** — complete and tested. `pytest tests/` → **100 passed**.

| module | status |
|---|---|
| `data.py` | done — loading, `SamplingProfile`, ACF-by-lag |
| `features.py` | done — leak-safe builder, Fourier terms, `lag_samples` for faithful reproduction |
| `splits.py` | done — rolling-origin folds, calibration slice, **embargo** for multi-horizon |
| `baselines.py` | done — persistence, seasonal-naive, drift, rolling mean, train mean |
| `models.py` | done — Ridge / RF / XGBoost point + `QuantileGBM` |
| `architectures.py` | done — TCN, Transformer, DLinear/NLinear, N-BEATS/NBEATSx |
| `tuning.py` | done — leak-free random search on a development prefix |
| `evaluate.py` | done — backtest harness, DM tests + BH correction |
| `forecast.py` | done — direct + recursive multi-horizon, per-horizon naive baselines |
| `context.py` | done — flag validation, disjoint groups, `UncertaintyModel` |
| `conformal.py` | done — marginal / relative / locally adaptive / Mondrian / **ACI** + estimability guard |
| `allocation.py` | done — cost model, policies, Pareto sweep |
| `pipeline.py` | done — end-to-end allocation backtest |
| `metrics.py`, `stats.py`, `plots.py` | done, and now exercised — every notebook runs clean |
| `tests/test_bwalloc.py` | done — 100 gates, all passing |

**Experiments run:** `run_audit.py`, `run_benchmark.py`, `run_allocation.py`,
`run_horizon.py`, `run_coverage_gate.py`. Every table is in `experiments/results/`.

---

## Multi-horizon — this reframes the whole project (NEW, verified)

The single most important result so far, and it was invisible while everything was
one-step-ahead. GP, RMSE by lead time, 8 rolling-origin folds with an embargo:

| lead | 1.43 h | 2.87 h | 5.73 h | 11.47 h | 24.37 h |
|---|---|---|---|---|---|
| random_forest | **10.40** | **11.85** | **12.48** | **12.83** | **12.82** |
| xgboost | 10.73 | 12.18 | 12.36 | 12.79 | 13.01 |
| ridge | 11.26 | 14.45 | 15.87 | 14.50 | 15.08 |
| seasonal_naive_17 | 18.31 | 18.31 | 18.35 | 18.38 | 18.26 |
| persistence at that lead | 12.12 | 17.41 | 25.11 | 31.16 | 18.26 |
| **RF advantage over naive** | **−14%** | **−32%** | **−50%** | **−59%** | **−30%** |

Read the last row. At one step ahead the learned model beats a one-line baseline by
14%, which is the weak result that made the corrected benchmark look unexciting. At
an 11.5-hour lead — an operationally realistic provisioning horizon — it beats the
naive forecaster by **59%**, and its own accuracy has degraded by only 23% while the
baseline's has degraded by 157%.

**This is the argument for the entire system**, and it is the sentence to lead the
paper with: the value of learning is not visible at one step, it is visible at the
lead time an allocator actually needs. It also disposes of the "six of ten models
lose to persistence" problem in the audit — that comparison was only ever damning at
h=1.

**Robi replicates it**, which makes this a two-operator finding rather than one
trace's quirk:

| lead | 1.65 h | 3.30 h | 6.60 h | 11.55 h | 24.75 h |
|---|---|---|---|---|---|
| random_forest | **20.77** | **25.23** | **27.27** | **28.01** | **22.69** |
| xgboost | 21.11 | 24.04 | 27.23 | 27.74 | 24.25 |
| seasonal_naive_15 | 26.26 | 26.26 | 26.27 | 26.24 | 26.49 |
| persistence at that lead | 29.75 | 45.07 | 65.97 | 75.95 | 26.49 |
| **RF advantage over naive** | **−30%** | **−44%** | **−59%** | **−63%** | **−14%** |

Same shape: advantage grows from −30% to **−63%** as lead time grows, then narrows at
24 h where the seasonal baseline becomes strong again.

### Direct vs recursive — the answer differs by operator

GP: direct dominates and the gap widens with horizon.

| steps | 1 | 2 | 4 | 8 | 17 |
|---|---|---|---|---|---|
| direct | 10.73 | 12.19 | 12.37 | 12.79 | 13.01 |
| recursive | 10.80 | 14.33 | 19.19 | 20.48 | 24.18 |

At 17 steps recursive is **86% worse**. (At 1 step the two agree to 0.7%, as they
must — that agreement is the correctness check on the rollout.)

Robi: **the two are within noise at intermediate horizons and recursive is slightly
ahead**, with direct winning only at the daily horizon.

| steps | 1 | 2 | 4 | 7 | 15 |
|---|---|---|---|---|---|
| direct | 21.11 | 24.04 | 27.23 | 27.74 | **24.25** |
| recursive | 21.32 | **23.76** | **26.82** | **26.62** | 28.10 |

Do not overstate "direct beats recursive" — it is operator-dependent, and the honest
statement is that direct is the safer default and is decisively better at long
horizons and on the trace with strong short-run autocorrelation.

Tables: `horizon_gp.csv`, `horizon_alloc_gp.csv`, `horizon_strategy_gp.csv`
(and `_robi` equivalents).

### A second finding that only the corrected sampling rate makes visible

On **Robi**, forecasting 24 hours ahead with yesterday's value at the same hour
(RMSE 26.49) is *better* than forecasting 1.6 hours ahead with the most recent
observation (29.40). The daily cycle there is stronger than short-run persistence.
This inverts the usual "accuracy decays with horizon" intuition, and it is only
visible once the daily period is measured as 15 samples rather than assumed to be 24.
Pinned in `test_horizon_baseline_is_not_the_one_step_baseline`.

---

## Coverage gate — now enforced on real data, and the failure is fixed (NEW)

`experiments/run_coverage_gate.py` runs the plan's ±2% check on actual backtest
output, with Clopper-Pearson intervals, and writes `coverage_gate.csv`.

**First result (one-step allocation only): 41 of 54 configurations fail; 39 of the 41
failures are under-coverage.** The full sweep over lead times is below and is worse
still — 249 of 360 — but the conclusion is identical.
GP is close (marginal 0.931 at τ=0.95); Robi under-covers everywhere, worst −5.1 pp
at τ=0.90.

The one-sidedness is diagnostic: split conformal's guarantee is conditional on
exchangeability, and a 55-day trace with a trend does not supply it. So this is drift,
not a bug — and drift has a standard fix, now implemented:

`conformal.AdaptiveConformalInference` (Gibbs & Candès) updates the requested
miscoverage online, `α ← α + γ(α_target − err)`, so a run of breaches widens the next
margin and a quiet stretch reclaims the capacity. Long-run coverage converges without
assuming exchangeability at all. Verified in `tests/`: on a synthetic stream whose
error scale triples, frozen split conformal under-covers and ACI recovers nominal to
within 2 pp; a separate gate proves the feedback is strictly causal.

**Still to do:** re-run `run_allocation.py` (ACI is wired into `pipeline.py` as the
`aci` method but the one-step results on disk predate it) and re-run the gate to
confirm the failure count drops on the real traces.

---

## Established results (reproduced, safe to cite)

### Audit — the three original errors

| | GP | Robi |
|---|---|---|
| Sampling interval | 86 min | 99 min |
| True daily period | 17 samples | 15 samples |
| `lag 24` actually spans | 34.4 h | 39.6 h |
| ACF at true period | **+0.45** | **+0.82** |
| ACF at lag 24 | **−0.35** | **−0.38** |
| Seasonal-naive RMSE, true period | 18.16 | 26.58 |
| Seasonal-naive RMSE, lag 24 | 28.42 | 74.12 |
| **Cost of the error** | **+56%** | **+179%** |

**Leak:** notebook-as-written GP 6.54 / Robi 16.32 → shift restored GP **19.30** /
Robi **27.37** (+195% / +68%). Corrected GP is worse than the training mean (18.87).

**Persistence baseline (sanity floor, locked in a test):** GP **12.1845**, Robi **31.4122**.

**Context flags (GP)** — most are inert on the *level*, one matters on the *variance*:
`is_rain` variance ratio **1.330**, Levene **p < 0.001** (the only significant variance
increase). `is_weekend` p = 0.77 on the level — and both CSVs are named after it.
On Robi, `is_powercut` and `is_event` have variance ratio **< 1** (p < 0.001) — they mark
*calmer* periods. Direction differs by operator; worth a sentence in the paper.

### Corrected benchmark (8 rolling-origin folds, mean ± sd)

GP — persistence 12.13 ± 0.92; **random_forest 10.41 ± 1.19 (−14.1%)**, xgboost 10.80,
ridge 11.26. DM: RF beats persistence p<0.001, beats ridge p=0.025 and xgboost p=0.026
(BH-significant). ridge vs xgboost = tie.

Robi — persistence 29.75 ± 4.72; **random_forest 20.75 ± 4.42 (−30.3%)**, xgboost 21.32,
ridge 22.72, seasonal_naive_15 26.26. DM: RF vs xgboost = tie; both beat ridge and
seasonal-naive.

Compare to the original: GP XGBoost was 13.91 (a *loss* to persistence); corrected it is
10.80, an 11% win. That reframing is a paper result on its own.

### Ablation — a genuine negative result, on both operators

GP: `lags_only` 9.98 < `no_fourier` 10.00 < **`original_style` 10.11** <
`original_correct_period` 10.12 < `no_context` 10.41 < `full` 10.41.

Robi (re-run, now trustworthy): **`original_style` 20.53** < `full` 20.75 <
`no_context` 20.82 < `original_correct_period` 21.39 < `lags_only` 21.78 <
`no_fourier` 21.80.

**The corrected feature design does not improve accuracy on either operator.** Once the
leak is removed, the original feature set is competitive or slightly better. Fourier
terms and context flags cost ~0.4 RMSE on GP and are roughly neutral on Robi.

Why, and this is the point to make in the paper: the lag-24 error devastates a *naive*
forecaster (seasonal-naive is 56% / 179% worse at lag 24) but a tree ensemble that also
sees `lag_1`, `lag_2` and `lag_3` simply routes around the bad feature — short-term
persistence carries most of the signal on these traces. So the sampling error is real
and matters for interpretation, seasonality claims and any seasonal-naive or ARIMA-style
model, but it is not where accuracy was being lost.

**C1's contribution is therefore correctness and evaluation rigor, not accuracy gains.**
State that plainly. The accuracy claim that does survive is a different one: the original
XGBoost result (13.91, a loss to persistence) becomes 10.80 (an 11% win) under the
corrected protocol — that gain comes from proper training windows and honest folds, not
from the features.

### C2b — context-conditional coverage — **holds on GP, correctly null on Robi**

Achieved coverage, pooled across folds, weighted by test-block size:

| nominal τ | group | marginal | adaptive | mondrian |
|---|---|---|---|---|
| 0.80 | baseline | 0.831 | 0.836 | 0.808 |
| 0.80 | **elevated_risk** | **0.698** | **0.732** | 0.726 |
| 0.90 | baseline | 0.915 | 0.904 | 0.898 |
| 0.90 | **elevated_risk** | **0.771** | **0.816** | 0.816 |
| 0.95 | baseline | 0.963 | 0.941 | 0.949 |
| 0.95 | **elevated_risk** | **0.866** | **0.911** | 0.922 |

Consistent at every level: marginal calibration under-delivers on the elevated-risk
group by 9–14 points, and both context-conditional methods recover roughly half to two
thirds of that gap at near-identical capacity cost (~1.18× demand at τ=0.95).

### IMPORTANT correction — attach the confidence intervals before quoting this

I previously called this "the strongest result in the project". With Clopper–Pearson
95% intervals attached (n = 354 baseline, n = 179 elevated risk) the claim has to be
split in two, because its halves have different evidential status:

| τ | group | marginal | adaptive | mondrian | aci |
|---|---|---|---|---|---|
| 0.90 | elevated | 0.771 [.702, .830] | 0.816 [.751, .870] | 0.788 [.720, .845] | **0.911 [.859, .948]** |
| 0.95 | elevated | 0.866 [.807, .912] | 0.899 [.846, .939] | 0.933 [.886, .965] | 0.944 [.900, .973] |

- **The failure IS decisive.** At τ=0.90 and τ=0.95 the marginal interval on the
  elevated-risk group excludes the nominal level. That marginal calibration fails where
  the network is stressed is established, and is safe to claim.
- **The context-conditional repair is NOT decisive at this sample size.** Adaptive moves
  τ=0.95 from 0.866 to 0.899, but [.807, .912] and [.846, .939] overlap heavily, and
  the adaptive interval still excludes 0.95. Mondrian reaches 0.933 [.886, .965], which
  does contain the nominal level — the best of the three, but on the same 179 points.

179 elevated-risk test points cannot resolve a 3–7 point coverage difference. Quote the
point estimates *with* the intervals, and describe the repair as directional.

**Unexpected, and worth leading with:** the largest and only clearly decisive
improvement on the elevated-risk group comes from **ACI**, which is not
context-conditional at all — at τ=0.90, 0.911 [.859, .948] against marginal's 0.771
[.702, .830], non-overlapping. Elevated-risk periods cluster in time, so an online level
update partly absorbs them without ever being told the context. The temporal mechanism
is better evidenced than the contextual one on this dataset. A per-group online update
is the obvious combination and is untried.

**Robi — the method correctly does nothing, as predicted.**

| nominal τ | group | marginal | adaptive | mondrian |
|---|---|---|---|---|
| 0.90 | elevated_risk | 0.847 | 0.792 | 0.847 |
| 0.95 | elevated_risk | 0.889 | 0.875 | 0.903 |

No consistent gain — adaptive is sometimes worse. This is **not a failure, it is a
confirmed prediction.** The audit found no flag on Robi with elevated residual variance
(`is_political_gathering` variance ratio 0.996; Robi has no rain annotation), so there
is no context-variance signal to exploit and the method should do nothing. It does
nothing.

Frame it exactly that way in the paper: the method helps where the audit says the
variance signal exists (GP, `is_rain`, ratio 1.33, p<0.001) and is inert where the audit
says it does not (Robi). A falsifiable prediction, tested on a second operator, and
confirmed in both directions. That is considerably stronger than "it works everywhere."

---

## The capacity-saving claim fails — but the COST claim holds, on both operators

Two things were wrong with how this was being measured, and fixing the second one
turned a negative result into the paper's headline.

### 1. Capacity-at-equal-SLA: still negative, and now understood

Even with the conformal families swept properly and the multiplicative
`conformal_relative` variant added, no family reaches a 1%, 2% or 5% violation target
on either trace, and at 10% the saving against fixed-margin is −2 to −3%.

**Do not write the sentence from the plan** ("at an equal 1% SLA violation rate our
allocator provisions X% less capacity"). The data does not support it.

The relative variant was implemented as planned and did not rescue it (GP −3.1% at
≤10% violations, against −2.7% for additive). So the additive/multiplicative mismatch
was *not* the explanation. The actual explanation is worse for the comparison itself:

**Drawing the fixed-margin frontier requires hindsight.** Picking "the margin that
lands at a 1% violation rate" presupposes knowing the violation rate each margin
achieved *on the test data*. No operator can make that choice in advance. Conformal
picks its level a priori from the cost ratio via `τ* = κ/(1+κ)` and is then held to
whatever it delivers. Scoring an a-priori method against a hindsight-tuned heuristic,
on the heuristic's home metric, is not a fair test — and capacity-at-equal-SLA is
exactly that metric.

### 2. Cost is the right metric, and on cost the method wins — twice

Cost is what the entire allocation argument is built on: the asymmetric cost model is
what makes a quantile the correct allocation in the first place. It is well defined
for both families, and `allocation.cost_comparison` reports it.

Setup, biased *against* the proposed method: conformal is evaluated at the level the
theory prescribes (τ* = 0.909 for κ = 10, no test-set information), while
fixed-margin is given its **best** margin chosen with hindsight.

| operator | family | cost at τ* | vs tuned fixed-margin |
|---|---|---|---|
| **GP** | **conformal_adaptive** | **19.27** | **−7.5%** |
| GP | conformal_mondrian | 20.11 | −3.5% |
| GP | conformal_relative | 20.81 | −0.1% |
| GP | conformal_marginal | 21.35 | **+2.4% (worse)** |
| GP | fixed_margin (tuned) | 20.84 | — |
| GP | static_peak | 29.52 | +41.6% (worse) |
| **Robi** | **conformal_adaptive** | **49.34** | **−14.8%** |
| Robi | conformal_marginal | 59.09 | +2.0% (worse) |
| Robi | conformal_mondrian | 59.20 | +2.2% (worse) |
| Robi | conformal_relative | 60.34 | +4.2% (worse) |
| Robi | fixed_margin (tuned) | 57.93 | — |

Read the pattern, not just the winner. **Locally adaptive calibration beats the tuned
heuristic on both operators; marginal calibration loses to it on both.** The saving
comes specifically from making the margin *condition on predicted uncertainty* — which
is the C2b idea, now attached to money rather than only to coverage.

The two sentences the project was trying to earn, in their supportable form:

> *"At the cost ratio the operator specifies, context-adaptive conformal allocation
> provisions 7.5% (GP) and 14.8% (Robi) more cheaply than a fixed-margin rule tuned
> with hindsight, while marginally calibrated conformal allocation is slightly worse
> than that rule on both traces."*

> *"Marginal calibration meets its 95% target overall but delivers only 86.6% during
> elevated-risk periods on GP; context-conditional calibration restores it to
> 91–92% at near-identical capacity cost."*

**Caveat to state in the paper.** On Robi the adaptive gain cannot be credited to the
context flags — the audit found no Robi flag with elevated variance, and the C2b
coverage result is correctly null there. `UncertaintyModel` learns σ̂(x) from the whole
feature vector, so on Robi the adaptivity is coming from time and lag features. The
honest claim is "conditioning the margin on predicted uncertainty pays"; only on GP is
that uncertainty demonstrably *contextual*.

Tables: `cost_gp.csv`, `cost_robi.csv`, `pareto_*.csv`, `savings_*.csv`.

---

## ACI fixes the coverage gate — verified on real data

Re-running `run_coverage_gate.py` after adding the `aci` method to the one-step
allocation backtest:

| method | configurations passing ±2% | mean coverage gap |
|---|---|---|
| **aci** | **14 / 18** | **−0.9 pp** |
| mondrian | 5 / 18 | −4.6 pp |
| adaptive | 3 / 18 | −4.4 pp |
| marginal | 3 / 18 | −4.6 pp |

(That table is the one-step allocation slice, including per-group rows. The
lead-time-wide numbers are below.)

Marginal coverage (group = ALL), nominal versus achieved:

| operator | τ | aci | adaptive | marginal | mondrian |
|---|---|---|---|---|---|
| GP | 0.80 | **0.805** | 0.784 | 0.784 | 0.777 |
| GP | 0.90 | **0.902** | 0.863 | 0.867 | 0.854 |
| GP | 0.95 | **0.953** | 0.919 | 0.927 | 0.940 |
| Robi | 0.80 | **0.781** | 0.754 | 0.752 | 0.752 |
| Robi | 0.90 | **0.884** | 0.859 | 0.847 | 0.847 |
| Robi | 0.95 | **0.945** | 0.916 | 0.906 | 0.903 |

On GP, ACI is essentially exact at all three levels. This closes the plan's Part 5
coverage check, which had never been enforced on real output.

### And it holds as the forecast degrades

`run_horizon.py` was re-run with `aci` wired in. GP, achieved coverage by lead time:

| τ | 1.4 h | 2.9 h | 5.7 h | 11.5 h | 24.4 h |
|---|---|---|---|---|---|
| 0.80, **aci** | **0.803** | **0.803** | **0.806** | **0.817** | **0.798** |
| 0.80, marginal | 0.784 | 0.767 | 0.727 | 0.754 | 0.775 |
| 0.90, **aci** | **0.904** | **0.902** | **0.906** | **0.907** | **0.901** |
| 0.90, marginal | 0.867 | 0.878 | 0.851 | 0.839 | 0.868 |
| 0.95, **aci** | **0.947** | **0.946** | **0.949** | **0.945** | **0.950** |
| 0.95, marginal | 0.927 | 0.957 | 0.915 | 0.909 | 0.924 |

### Final gate result, after re-running everything

Marginal coverage (group = ALL) across 5 lead times x 3 levels x 2 operators —
**30 configurations**:

| method | passing ±2% | GP mean gap | Robi mean gap |
|---|---|---|---|
| **online (ACI)** | **30 / 30** | **+0.2 pp** | **−0.9 pp** |
| locally adaptive | 6 / 30 | −2.8 pp | −4.2 pp |
| Mondrian | 5 / 30 | −3.3 pp | −5.5 pp |
| marginal | 2 / 30 | −3.4 pp | −4.9 pp |

**ACI meets its target in every single one**, at every lead time from 1.4 to 24 hours,
on both operators, while static calibration drifts as far as 7 points below nominal
(GP τ=0.80 at a 5.7 h lead) as the forecast degrades.

This closes the plan's Part 5 coverage check. Quote it as *"online calibration holds
its nominal service level in all 30 (operator, lead time, level) configurations, while
static calibration meets it in 2 to 6."*

**Do not overstate it.** This is *marginal* coverage. Counting the per-group rows too,
ACI passes 61 of 90 against 12–20 for the static methods — much better, but not
complete, because ACI is not group-conditional. Per-group coverage is what the
context-conditional methods address, and the two fixes are complementary rather than
substitutes. Combining them (a per-group online update) is untried and is the obvious
next methodological step.

---

## C4 — cold-start transfer, done, and the answer is about a week (NEW, verified)

`experiments/run_transfer.py` replaces the broken `Robi_to_GP.ipynb`. Four arms on
one fixed GP test window (last 30%, 267 points), Robi as the source operator.

**This experiment is only possible because of the C1 correction.** The two traces are
sampled at 86 and 99 min, so a design matrix indexed by sample count is not comparable
across them — `lag_24` means 34.4 h on one site and 39.6 h on the other. Because lags
are defined in wall-clock hours and seasonality in Fourier terms of wall-clock time,
the two design matrices measure the same quantities and a model can move between them.
Transfer is a dividend of the sampling-rate correction.

Only the five flags both operators carry can be used; the four dropped include
`is_rain`, the one flag with real variance signal.

**At a 6-hour lead** (RMSE on the fixed GP test window; persistence = **26.49**):

| days of GP history | 3 | 5 | 7 | 14 | 21 |
|---|---|---|---|---|---|
| gp_only (own data alone) | 20.61 | 19.81 | 18.98 | **14.82** | 15.86 |
| transfer (pretrained + fine-tuned) | **19.65** | 19.87 | 19.91 | 15.63 | 17.45 |
| robi_cold (pretrained, no fine-tuning) | 20.39 | 20.01 | 19.74 | 19.06 | 19.04 |
| transfer gain over own data | **+4.6%** | −0.3% | −4.9% | −5.5% | −10.0% |

Three things to say about this:

1. **A model trained entirely on a different operator, given only enough of the new
   site's history to fix its level and scale, scores 20.4 against persistence's 26.5 —
   23% better than the floor with zero training data from the new site.**
2. Pretraining helps only while the site is data-poor: +4.6% at 3 days, nothing by
   5–7 days, and **−10% by 21 days**, where the borrowed weights actively hold the
   model back.
3. **The crossover is at roughly 5–7 days.** That is the operational answer: borrow a
   neighbour's model for the first week, then switch to the site's own.

**At a 1.5-hour lead the study answers nothing** — persistence scores 12.51 and no arm
beats it, so every curve sits above the floor. That is precisely the trap the original
notebook fell into, and it is a second demonstration of the multi-horizon point.

Scale handling: GP averages 92.9 Gbps against Robi's 145.5, so the target and every
level-valued feature are z-scored per operator, with the new site's own first *k* days
supplying the statistics — the only information a genuine cold start has.

Table: `transfer_robi_to_gp.csv`. Figure: `fig8_transfer.png`.

---

## C3 — zero-shot foundation models, done, and it validates C1 (NEW, verified)

`experiments/run_foundation.py`, Chronos-Bolt Small, strictly zero-shot on the same
folds/leads/test blocks as `run_horizon.py`. CPU, ~7 minutes, no GPU.

**The answer depends entirely on the operator**, which is the finding.

| lead | GP zero-shot | best trained | vs trained | vs persistence |
|---|---|---|---|---|
| 1.4 h | 11.18 | 10.40 | +7.5% | −7.8% |
| 5.7 h | 13.50 | 12.36 | +9.2% | −46.2% |
| 11.5 h | 14.41 | 12.79 | +12.7% | −53.8% |
| 24.4 h | 15.20 | 12.82 | +18.6% | −16.8% |

| lead | Robi zero-shot | best trained | vs trained | vs persistence |
|---|---|---|---|---|
| 1.7 h | 30.54 | 20.77 | +47.1% | **+2.7%** |
| 6.6 h | 40.34 | 27.23 | +48.2% | −38.8% |
| 11.6 h | 42.56 | 27.74 | +53.4% | −44.0% |
| 24.8 h | 43.63 | 22.69 | **+92.3%** | **+64.7%** |

GP: a credible zero-shot system — beats persistence at every lead, trails bespoke
training by only 7.5–18.6%. Robi: trails by 47–92% and loses to persistence at the
shortest and longest leads.

### Why — and this is the part worth writing up

A foundation model reads a bare sequence with **no timestamps**. It cannot know a step
is 86 min on one trace and 99 on the other, so it must infer the cycle from the
sequence — and under irregular sampling the daily cycle is not a whole number of steps:

| | gap | exact daily period | misregistration |
|---|---|---|---|
| GP | 86 min | 16.744 samples | 0.256 samples/day |
| Robi | 99 min | **14.545** | **0.455 samples/day** |

Robi slips ~2× faster: ~25 samples (1.7 cycles) of drift over 55 days against GP's ~14.
And the worst single result in the experiment is **Robi at 24 h**, the one setting where
the daily cycle *is* the signal. The naive baseline wins there by stepping back one
**measured** cycle — precisely what the model cannot do.

**So C1 is not just a repair of the senior's work; it is a diagnostic.** Measuring the
sampling interval tells you in advance whether a timestamp-blind foundation model will
work on your trace. Lead with this — it is the most novel thing C3 produced.

### Its quantiles are not usable for provisioning as they come

Chronos-Bolt's quantile head was trained on 0.1–0.9 **only**; τ=0.95 is silently
clipped to τ=0.90 and returns identical numbers. Via τ* = κ/(1+κ) that ceiling is a
cost ratio of just **κ = 9**. Pinned in `test_native_quantile_ceiling_caps_the_expressible_cost_ratio`.

| operator | τ | zero-shot | conformalised |
|---|---|---|---|
| GP | 0.80 | 0.791 | 0.815 |
| GP | 0.90 | 0.886 | 0.903 |
| GP | 0.95 | 0.886 *(clipped)* | 0.941 |
| Robi | 0.80 | **0.662** | 0.813 |
| Robi | 0.90 | **0.786** | 0.909 |
| Robi | 0.95 | 0.786 *(clipped)* | 0.950 |

Robi's zero-shot 80% interval delivers 66%. Conformal repairs every level on both
traces. The synthesis for the paper: **the foundation model supplies a cheap point
forecast; conformal calibration supplies the service-level guarantee it cannot.**

Tables: `foundation_accuracy.csv`, `foundation_calibration.csv`. Figure:
`fig9_foundation.png`. Notebook: `06_foundation_models.ipynb`.

---

## Sequence models — done, and the answer changes a paper claim (NEW, verified)

`experiments/run_sequence.py`, `src/bwalloc/sequence.py` (torch, CPU, ~4 min total).
The original's four architectures (`Conv1D(64,3)`, `LSTM(64)`, `GRU(64)`,
`SimpleRNN(50)`), same 30 epochs, on the corrected fold schedule, every model reading
the identical univariate window of 24 consecutive sample lags.

**Why the original's ranking was meaningless:** its sequence models trained on
`train_size=0.7` (~600 rows) while its tree models got the broken date split's **88
rows**. CNN 9.69 vs XGBoost 13.91 compared 600 training rows to 88.

**GP — the CNN genuinely wins, and significantly:**

| model | RMSE | vs persistence | originally reported |
|---|---|---|---|
| **cnn** | **7.875 ± 1.03** | **−35.3%** | 9.69 |
| gru | 8.559 | −29.6% | 12.14 |
| rnn | 8.625 | −29.1% | 10.06 |
| random_forest | 8.673 | −28.7% | 14.96 |
| lstm | 8.816 | −27.5% | 10.67 |
| xgboost | 8.846 | −27.3% | 13.91 |
| ridge | 8.879 | −27.0% | — |
| persistence | 12.164 | — | — |

DM vs random_forest: **cnn p = 0.003, BH-significant.** Every other pair ties.

**Robi — nothing separates.** gru 19.876, ridge 20.143, cnn 20.160, rnn 20.217,
lstm 20.293, random_forest 20.519, xgboost 21.239. No significant pair.

### The real finding is lag depth, and it tempers §5

On one common row index (881 GP / 864 Robi rows, identical folds), random forest:

| configuration | GP | Robi |
|---|---|---|
| full corrected design | 10.562 | 20.722 |
| lags 1–8 only | 9.136 | 21.210 |
| lags 1–12 only | 8.863 | 20.929 |
| **lags 1–24 only** | **8.673** | **20.519** |
| **full corrected + lags 1–24** | **8.651** | 20.529 |

The control (`full + lags1_24` = 8.651 ≈ `lags1_24_only` = 8.673) proves the gain is
**window depth**, not the removal of Fourier/context, which stay neutral as §4.4 found.
**The corrected design was under-lagged; on GP that cost 18% RMSE.**

**This affects the paper's §5 claim.** §5 says a learned model beats persistence by
only 14% at one step and reads the value of learning as living at longer leads. At
depth 24 that one-step margin is 29–35% on GP. The *direction* of the lead-time
argument is unaffected (the naive forecaster still degrades far faster), but the
one-step regime is less unfavourable than stated. §4.5 of the paper flags this
explicitly rather than letting it stand.

**Not done:** `run_horizon.py`, `run_allocation.py` and `run_coverage_gate.py` all
still use the sparse four-lag design. Re-running them at depth 24 is the obvious next
step and would change the headline multi-horizon numbers. Decide before submitting.

Tables: `sequence_{gp,robi}_{perfold,summary,dm}.csv`, `sequence_lag_depth.csv`.
Notebook: `08_sequence_models.ipynb` — runs the four architectures live and
reproduces the lag-depth ladder; committed executed.
Gates: 5 new (52 total; 58 after the notebook-contract gates) — window chronology, contiguity, fit-window-only scaling,
determinism, lookback/design-matrix agreement.

---

## Covariates, modern architectures and tuning (NEW — code landed, runs in progress)

Three things were true of every number this project had produced, and this round
addresses all three.

**1. The sequence models had never seen the features.** `sequence_feature_config()`
builds a bare demand window because the original used `input_shape=(lookback, 1)` and
notebook 08's contrast depends on isolating architecture from feature set. That is a
**reproduction constraint, not a recommendation** — and it had never been lifted.
Meanwhile the trees had the opposite problem: full features at four lags. **No model
here had seen the full feature set at full lag depth.**

Covariates are time-varying — there is an `is_rain` value at each of the 24 past
timestamps, not one per window — so they enter as **channels over the lookback**, per the
covariate taxonomy in DeepAR and the Temporal Fusion Transformer, not as a flat vector.

- `features.FeatureConfig.covariate_lag_samples` emits `{column}__lagS_{n}` through the
  same `.shift()` path as the demand lags, so `assert_no_leakage` covers the new block
  with **no exemption** — deliberate, given what this project exists to fix.
- `sequence.channel_window` groups them into one channel per covariate;
  `sequence.CovariateSequenceForecaster` is the senior's four architectures widened to
  read them. GP builds **16 channels x 24 lags + 25 static**; Robi **12 + 21**.
- `lookback_columns` is untouched, so notebook 08 and `run_sequence.py` still reproduce
  byte-for-byte.

**2. Only four architectures had been tried.** `src/bwalloc/architectures.py` adds TCN,
a small Transformer, DLinear/NLinear and N-BEATS/NBEATSx. Each tests a specific reading
of the lag-depth finding rather than being added for novelty — the linear models are the
honest control (if a Transformer cannot beat a linear layer it earned nothing), and plain
N-BEATS is retained **labelled as the univariate control**. PatchTST, Informer/Autoformer
and the full TFT are recorded as deliberate omissions with reasons in `docs/PROVENANCE.md`.

Because three of these are univariate as their papers define them, "reads the full
feature set" is **enforced, not intended**: a gate fails any model whose prediction does
not move when a covariate channel does, and `channel_permutation_importance` reports what
each channel is actually worth.

Two real bugs this caught, both found by the gates rather than by inspection:

- **NLinear annihilated the context flags.** Its normalisation subtracts the window's
  last value; a flag constant across the window centres to exactly zero, so `is_rain` was
  invisible. Fixed to centre the **demand channel only**. Do not "simplify" it back.
- **NBEATSx was 108k parameters against ~350 training rows.** Fixed with the exogenous
  encoder the paper actually specifies; now ~53k.

**3. Nothing had ever been tuned.** Every hyperparameter was a default — 30 epochs,
batch 32, lr 1e-3 from the original notebooks; `n_estimators=300, max_depth=12` by hand.
So every ranking, notebook 08's included, ranked *whose defaults suited the data*.

`src/bwalloc/tuning.py` does 30-trial seeded random search on a **development prefix**
that ends before the first evaluation test block, with its own inner rolling-origin
schedule. **The boundary is a timestamp, not a row count** — a trial may ask for
`lookback=48`, drop more warm-up rows, and a row-count boundary would slide into
evaluation data. Two gates pin it. The **feature set is in the search space**
(`lookback`, `covariates`, `context`), so "do the covariates earn anything?" is settled
by measurement rather than judgement.

No new dependency: Optuna's TPE would be the obvious alternative and is cited as the road
not taken, because adding it breaks the zero-install promise in `docs/KAGGLE.md`.

**Gates: 58 -> 100, all passing.**

### Runtimes, measured

Per fold at the largest fit window (814 rows) on GP: ridge 0.01 s, random forest 0.91 s,
xgboost 1.07 s, and the sequence models 5-13 s each. That is why the arm study is ~45 min
and the tuning run is hours — budget accordingly, and note `run_tuning.py` checkpoints
per (model, operator) so an interrupted run resumes.

### The answer: the covariates do not help. They hurt, on both operators.

`run_full_features.py`, five arms on one common row index (GP 881 rows, Robi 864),
8 folds. RMSE:

**GP**

| model | lags_only | full_sparse | full_dense | dense_no_context | **full_dense_covariates** |
|---|---|---|---|---|---|
| cnn | 7.875 | — | 7.875 | 7.875 | 7.875 |
| **cnn_cov** | — | — | — | — | **14.311** |
| gru | 8.559 | — | 8.559 | 8.559 | 8.559 |
| **gru_cov** | — | — | — | — | **10.643** |
| rnn | 8.625 | — | 8.625 | 8.625 | **10.007** (`rnn_cov`) |
| lstm | 8.816 | — | 8.816 | 8.816 | **11.938** (`lstm_cov`) |
| random_forest | 8.673 | 10.562 | 8.651 | **8.619** | 9.014 |
| xgboost | 8.846 | 10.787 | **8.733** | 8.811 | 8.798 |
| ridge | 8.879 | 11.300 | 9.007 | 8.799 | **14.146** |
| persistence | 12.164 | | | | |

**Robi** — same direction throughout: cnn 20.160 against `cnn_cov` 26.190, gru 19.876
against `gru_cov` 21.717, random_forest 20.519 against 20.931, ridge 20.143 against
25.582. Persistence 29.864.

**This is a negative result, and it is the honest answer to "use all the features, not
just the lags".** Adding roughly 385 covariate columns to ~350-800 training rows costs
every model on both traces. It is a capacity result, not a wiring fault: the gates prove
each architecture consumes its covariate channels (flipping `is_rain` moves the
prediction), so the channels are read and then not paid for.

**The tuning search reached the same conclusion independently**, which is the strongest
form of this evidence because nothing coordinated the two. With `covariates` in the
search space, the selected configuration on GP was:

| model | default | tuned | DM p | what it chose |
|---|---|---|---|---|
| cnn_cov | 14.311 | **8.279** | <0.001 | `covariates=False`, lookback 12 |
| lstm_cov | 11.938 | **8.370** | <0.001 | `covariates=False`, lookback 8 |
| ridge | 9.007 | **8.688** | 0.002 | covariates=True, context=none |
| random_forest | 8.651 | 8.744 (**worse**) | 0.568 | lookback 12 |
| xgboost | 8.733 | 8.753 (worse) | 0.832 | lookback 12 |

Putting the feature set in the search space is what made that measurable rather than a
matter of judgement. **Report the trees honestly**: tuning made random forest and XGBoost
slightly worse on held-out folds and neither change is significant, so the search
overfitted its inner split and the trees were already near their best at defaults.

### The other two findings survive, and one is reinforced

- **Context flags stay neutral.** `dense_no_context` (8.619 GP) is indistinguishable from
  `full_dense` (8.651), exactly as notebook 01's ablation found at four lags. Adding them
  at depth 24 changes nothing.
- **Lag depth is still the lever, and still only on GP.** `full_sparse` 10.562 ->
  `full_dense` 8.651 is the same 18% notebook 08 reported. On Robi `full_sparse` 20.722 ->
  `lags_only` 20.519 is nothing. Two operators, one effect.

### Architectures: the TCN wins, on both operators — a new best on GP

`run_architectures.py`, both feature sets, 8 folds. RMSE on the `dense` arm (full
corrected features, demand lags 1-24, no covariate channels), and what the covariate
channels cost each model:

| | GP `dense` | cov. cost | Robi `dense` | cov. cost |
|---|---|---|---|---|
| **tcn** | **7.474** | +49% | **19.879** | +29% |
| nbeats *(univariate control)* | 7.833 | **0%** | 22.559 | **0%** |
| cnn_cov | 8.019 | +79% | 20.300 | +29% |
| random_forest | 8.651 | +4% | 20.529 | +2% |
| xgboost | 8.733 | +1% | 20.773 | +1% |
| dlinear | 8.945 | +67% | 20.953 | +19% |
| ridge | 9.007 | +57% | 21.011 | +22% |
| nbeatsx | 9.078 | +10% | 22.685 | +3% |
| nlinear | 9.138 | +34% | 20.641 | +12% |
| transformer | 9.198 | +48% | 21.885 | +12% |
| persistence | 12.164 | | 29.864 | |

**The TCN is the best model this project has produced on GP** — 7.474 against notebook
08's CNN at 7.875 and the corrected benchmark's random forest at 10.41. Diebold-Mariano
with BH correction: it wins **17 of 18** comparisons on GP and **9 of 18** on Robi, so
unlike almost every other finding here it replicates on the second operator. That is
what the architecture was picked to test: the original's "CNN" is a single undilated
`Conv1D`, and the dilated causal version is the principled form of the model that had
already won.

**The `nbeats` row is an internal control worth reading twice.** It is the one model that
ignores covariate channels by construction, and it is the one model whose covariate cost
is exactly zero. Every model that *can* read them is made worse by them. Nothing
coordinated that.

### What each input channel is actually worth

`channel_permutation_importance`, RMSE cost of shuffling one channel across the whole
window, averaged over tcn / transformer / nbeatsx / gru_cov:

| channel | GP | Robi |
|---|---|---|
| **demand** | **11.04** | **28.47** |
| day_cos1 | 0.35 | 4.37 |
| day_sin1 | 0.18 | 1.92 |
| is_event | 0.05 | — |

**Demand outweighs the best covariate by 30x on GP and 6.5x on Robi, and the context
flags are worth essentially nothing.** This is the direct measurement behind the negative
result: the models do read the channels — the gates prove it — and the channels have
almost nothing in them. On Robi the daily Fourier channels carry modest signal, which is
consistent with Robi's much stronger daily autocorrelation (ACF +0.82 against GP's +0.45).

### Attention does not concentrate at the daily period — on GP

Peak attention 0.071 against 0.042 uniform, so it is barely peaked at all.

- **GP** top lags 5, 6, 24, 7, 23. The measured daily period is 17, and it is not there.
- **Robi** top lags 12, 14, 13, 18, 1. The measured daily period is 15, and lags 13-14
  sit right against it.

Read alongside the ACF (GP +0.45, Robi +0.82 at the true period) this is coherent: the
operator with the strong daily cycle is the one whose attention finds it. It also means
**the lag-depth gain on GP is not seasonality** — a deep window there buys general
short-run smoothing, not a daily echo. That is a more specific claim than notebook 08
could make.

### Still to fill in

`run_model_comparison.py`, once the tuning search lands. **Everything in this section is one-step-ahead**:
`run_horizon.py`, `run_allocation.py` and `run_coverage_gate.py` still use the sparse
four-lag design and are unaffected by any of it.

---

## DeepCog is now cited (was a must-fix, resolved)

Bega et al., *DeepCog*, IEEE INFOCOM 2019 (pp. 280–288) and IEEE JSAC 38(2):361–376,
2020, does cost-aware *capacity* forecasting under asymmetric over/under-provisioning
costs — the same problem as §6, in a top-tier venue. It is now cited in §2 as the
closest prior work, and in §6.3, where its `MAE-post-best` baseline turns out to be
the same hindsight-tuned fixed margin we construct independently.

The positioning, as written: DeepCog trains the asymmetry into the network via a
custom loss, so the cost ratio is fixed at training time and a different κ needs a
different model; we keep an ordinary forecaster and move the asymmetry into a
calibrated quantile, so κ is an inference-time dial and one model serves every cost
ratio, plus a distribution-free coverage guarantee they do not have. The contributions
list no longer implies the allocation idea is new — the *mechanism* is.

Numbers and the full comparison are in `docs/BENCHMARKS.md`.

---

## What is still not done

- **TimesFM** alongside Chronos-Bolt in C3. Optional; Chronos-Bolt already answers
  the question and TimesFM is the heavier install.
- **A per-group online calibration**, combining the context-conditional idea with ACI.
  This is the clearest open methodological question the work raises — see the ACI
  section above for why.
- ~~**The paper.**~~ **Done.** `paper/paper.md` is complete: abstract, introduction,
  related work (§2, new), all seven results sections, limitations, conclusion and a
  16-entry reference list. `paper/tex/paper.tex` is the IEEEtran conference build and
  compiles clean (MiKTeX 24.1: 0 errors, 0 undefined refs, 0 missing figures, 0 overfull
  boxes >20pt), producing a 9-page `paper.pdf`. **Three things remain before submission,
  listed in `paper/tex/README.md`:** it is 9 pages against a typical 6-page limit, the
  author block is a placeholder, and reference [1]'s author list is incomplete because
  the publisher blocks automated access. The seven arXiv references were verified against
  the arXiv API; the classical ones were not machine-checked.
- **Re-run the horizon / allocation / coverage studies at lag depth 24.** This is now
  the most consequential open item: the sequence study showed the four-lag design costs
  18% RMSE on GP, and every downstream result inherits it.
- Quantile-LSTM not implemented (the point forecasters now are — see `sequence.py`);
  `pinball_loss` implemented but never used in a reported table; LightGBM/sklearn
  `QuantileGBM` backends untested locally.

---

## Things to be careful about

- `run_benchmark.py` **exits non-zero** if the top model does not beat persistence. That
  gate is deliberate; do not remove it to make a run pass.
- `conformal.min_calibration_size` had a floating-point off-by-one (0.8/(1−0.8) =
  4.000000000000001 → ceil 5). Fixed with a `round(ratio, 9)`. Do not "simplify" it back.
- Context-conditional results are valid at **τ ≤ 0.95 only** on these traces. The guard
  raises rather than clipping; that is intended.
- `RandomForest` wins on both operators but is slow — the full benchmark with ablation
  takes ~4 min locally. Colab with a GPU will not help (it is CPU-bound sklearn).
- The GP ablation negative result is real. Do not bury it.

## Reproduce everything

```bash
cd "D:/L4-T-1/EEE 402/project/bwalloc"
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q      # 58 gates, ~30 s
python experiments/run_audit.py                # ~1 min
python experiments/run_benchmark.py            # ~4 min
python experiments/run_allocation.py           # ~10 min
python experiments/run_horizon.py              # ~20 min
python experiments/run_sequence.py              # ~4 min, needs torch
python experiments/run_transfer.py              # ~2 min
python experiments/run_foundation.py           # ~7 min, needs chronos-forecasting
python experiments/run_coverage_gate.py        # instant, reads CSVs only
python notebooks/_build.py                     # regenerate the notebooks (strips outputs)
python -m nbconvert --to notebook --execute --inplace notebooks/0*.ipynb  # ~25 min
```

Git is the record of what changed: `git log --oneline`, `git show <sha>`. Commit
before stopping, and update this file in the same commit as the work it describes.
