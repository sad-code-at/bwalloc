# Project status

Last updated: 2026-08-29. Working notes so this can be picked up cold in a new session.

Plan file: `C:\Users\User\.claude\plans\nested-plotting-bengio.md` (approved).
Original senior's code, untouched, is at
`D:\L4-T-1\EEE 402\project\ML-Based-Dynamic-Bandwidth-Allocation-for-Mobile-Data-Usage-main\`.
All new work is in `D:\L4-T-1\EEE 402\project\bwalloc\`.

**The repository is now under git** (`main`, local only, no remote). Every session's
work is committed, so `git log` is the record of what changed and `git show <sha>`
recovers any earlier state. Commit before stopping.

---

## Done and verified

**Scaffold + library** — complete and tested. `pytest tests/` → **46 passed**.

| module | status |
|---|---|
| `data.py` | done — loading, `SamplingProfile`, ACF-by-lag |
| `features.py` | done — leak-safe builder, Fourier terms, `lag_samples` for faithful reproduction |
| `splits.py` | done — rolling-origin folds, calibration slice, **embargo** for multi-horizon |
| `baselines.py` | done — persistence, seasonal-naive, drift, rolling mean, train mean |
| `models.py` | done — Ridge / RF / XGBoost point + `QuantileGBM` |
| `evaluate.py` | done — backtest harness, DM tests + BH correction |
| `forecast.py` | done — direct + recursive multi-horizon, per-horizon naive baselines |
| `context.py` | done — flag validation, disjoint groups, `UncertaintyModel` |
| `conformal.py` | done — marginal / relative / locally adaptive / Mondrian / **ACI** + estimability guard |
| `allocation.py` | done — cost model, policies, Pareto sweep |
| `pipeline.py` | done — end-to-end allocation backtest |
| `metrics.py`, `stats.py`, `plots.py` | done (`plots.py` still never executed) |
| `tests/test_bwalloc.py` | done — 46 gates, all passing |

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
**This is the strongest result in the project and is the paper's core contribution.**

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

## What is still not done

- **C3 — zero-shot foundation models** (Chronos-Bolt, TimesFM). Not started, and the
  only remaining contribution from the plan. A GPU makes it faster but is not required:
  Chronos-Bolt Small is ~48M parameters and these are 900-point series, so CPU
  inference is minutes, not hours. Colab's free T4 is the easy route; a local CPU run
  is also viable. TimesFM is the heavier of the two — try Chronos-Bolt first.
- **The paper itself.** No draft exists. Everything it needs is now in
  `experiments/results/` and `paper/figures/`.
- Quantile-LSTM not implemented; `pinball_loss` implemented but never used in a
  reported table; LightGBM/sklearn `QuantileGBM` backends untested locally.

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
PYTHONPATH=src python -m pytest tests/ -q      # 46 gates, ~30 s
python experiments/run_audit.py                # ~1 min
python experiments/run_benchmark.py            # ~4 min
python experiments/run_allocation.py           # ~10 min
python experiments/run_horizon.py              # ~20 min
python experiments/run_transfer.py              # ~2 min
python experiments/run_coverage_gate.py        # instant, reads CSVs only
python notebooks/_build.py                     # regenerate the notebooks
```

Git is the record of what changed: `git log --oneline`, `git show <sha>`. Commit
before stopping, and update this file in the same commit as the work it describes.
