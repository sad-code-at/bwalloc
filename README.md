# Context-Conditional Bandwidth Allocation

Cost-aware, conformally calibrated bandwidth allocation from sparse, irregularly
sampled mobile operator traces.

This extends an earlier forecasting study on two Bangladeshi operator traces
(Grameenphone Dhaka and Robi Dhaka-3) in four directions: a corrected evaluation
protocol, forecasting at an actual lead time, an allocation layer, and
context-conditional calibration of that layer.

**New to this project? Read [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) first.** It teaches
the whole thing end to end — how to run every experiment, what was changed and why, what each
term means, and what each change bought. This README states the results; the walkthrough
explains them.

---

## At a glance

Two Bangladeshi cell-site traces, 905 and 888 observations over ~2 months. Seven studies,
every number reproduced from `experiments/results/` and gated by `pytest tests/`.

| # | Study | Headline | Where |
|---|---|---|---|
| 1 | **Audit** | The traces are sampled every 86 / 99 min, not hourly; rolling features leaked the target; no baseline was ever computed. Six of ten originally reported models lose to persistence. | `run_audit.py` |
| 2 | **Corrected benchmark** | Rolling-origin folds, real baselines, Diebold-Mariano with FDR control. The correction is a *negative* result on accuracy — and it is reported as one. | `run_benchmark.py` |
| 3 | **Lead time** | The value of learning is invisible at one step and decisive at eleven hours: −14% vs naive at a 1.4 h lead, **−59%** at 11.5 h. | `run_horizon.py` |
| 4 | **Allocation** | Cost asymmetry κ selects a quantile via τ\* = κ/(1+κ). Beats a *hindsight-tuned* fixed margin by 7.5% (GP) and 14.8% (Robi). | `run_allocation.py` |
| 5 | **Context-conditional** | Context acts on *variance*, not level: `is_rain` raises residual σ by 33%. Holds on GP, correctly null on Robi. | `run_allocation.py` |
| 6 | **Cold start** | A model trained on the *other* operator beats persistence with zero local training data. The crossover is **about a week**. | `run_transfer.py` |
| 7 | **Zero-shot** | Chronos-Bolt works on GP and fails on Robi — and the sampling interval predicts which, in advance. | `run_foundation.py` |
| 8 | **Sequence models** | The senior's CNN/LSTM ranking was confounded by an unequal split. Re-run fairly, the CNN *does* win on GP (7.88, p = 0.003) and everything ties on Robi — and the real lesson is that the corrected design was under-lagged by 18%. | `run_sequence.py` |
| — | **Coverage gate** | Static conformal meets its ±2% target in 2 of 30 configurations. Online (ACI) meets it in **30 of 30**. | `run_coverage_gate.py` |

### Where to read what

| If you want | Read |
|---|---|
| To learn the project end to end | [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — run order, vocabulary, what each change bought |
| The results, argued | [`paper/paper.md`](paper/paper.md) + [`paper/figures/`](paper/figures) (9 figures) |
| The current state and open questions | [`STATUS.md`](STATUS.md) — including what is *not* done |
| Where every method came from | [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — one row per decision, with its source paper |
| How our numbers compare to published work | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — external results, on comparable metrics |
| To run the notebooks on Colab or Kaggle | [`docs/COLAB.md`](docs/COLAB.md) and [`docs/KAGGLE.md`](docs/KAGGLE.md) — token setup, runtimes, what goes wrong |
| To run it yourself | [Quick start](#quick-start) below, or `notebooks/` in Colab |

---

## The result to read first

Every result in the earlier study, and every notebook in it, predicts `Gbps` at time
*t* with the observation at *t−1* already in hand. That is a nowcast, and no network
can provision from it — the capacity decision has to be made before the previous
measurement arrives.

Adding the lead time changes the conclusion. GP, RMSE across 8 rolling-origin folds:

| forecast lead | 1.4 h | 2.9 h | 5.7 h | 11.5 h | 24.4 h |
|---|---|---|---|---|---|
| random forest | **10.40** | **11.85** | **12.48** | **12.83** | **12.82** |
| naive forecast at that same lead | 12.12 | 17.41 | 25.11 | 31.16 | 18.26 |
| **advantage of learning** | **−14%** | **−32%** | **−50%** | **−59%** | **−30%** |

Robi replicates it — the advantage grows from −30% at a 1.7-hour lead to **−63%** at
11.6 hours, while the naive baseline degrades by 155%.

At one step ahead a learned model beats a one-line baseline by 14%, which is why the
corrected benchmark below looks unexciting. At an 11.5-hour lead — a realistic
provisioning horizon — it beats the naive forecaster *at that same lead* by 59%: its
own accuracy degraded 23% across a 17× longer horizon while the baseline's degraded
157%.

The value of learning here is not visible at one step. It is visible at the lead time
an allocator actually needs, and evaluating only at one step measured these models in
the single regime where they look worst.

---

## What this adds

The earlier work compared ten forecasting models by RMSE. This repository does four
things it did not.

**1. Corrects the evaluation protocol.** Three errors materially affected the earlier
results, all reproduced and quantified in `experiments/run_audit.py`:

- *The traces are not hourly.* GP is sampled every ~86 min and Robi every ~99 min, so
  a lag of 24 samples spans 34.4 h and 39.6 h respectively — roughly anti-phase to the
  daily cycle rather than aligned with it. Used as a "1-day" feature, lag 24 correlates
  **−0.35** with demand on GP where the true daily lag (17 samples) correlates **+0.45**.
  Seasonal-naive RMSE: 18.2 at the true period versus 28.4 at lag 24 on GP, and 26.6
  versus 74.1 on Robi (**+179%**).
- *Target leakage.* Rolling means were computed without shifting, placing `y_t` inside
  its own feature vector. Removing the leak moves the best reported GP result from
  6.5 to 19.3 RMSE — worse than predicting the training mean.
- *No baselines.* Against a one-line persistence forecaster (GP 12.18, Robi 31.41),
  six of the ten originally reported models are losses or ties.

**2. Forecasts at a lead time, and scores it honestly.** Direct multi-horizon models
at 1.5–24 h, with three things the multi-step literature most often gets wrong made
explicit: history features are read at the forecast *origin* while deterministic
calendar terms are read at the target; the naive baseline is re-derived per horizon,
so an *h*-step model is never flattered by comparison with one-step persistence; and
training pairs within *h−1* rows of the test block are embargoed, because otherwise
the model is fitted on outcomes that had not occurred when the first test forecast was
issued. Direct rollout beats recursive by 86% at a 24-hour lead on GP.

**3. Adds the allocation layer the title promises.** Forecasting `Gbps` is not
provisioning. Under- and over-provisioning have asymmetric costs, so the cost-optimal
allocation is a **quantile** of the predictive distribution, not its mean — with the
operator's cost ratio κ selecting the level via `τ* = κ/(1+κ)`. Split-conformal
calibration then makes the achieved service level match the promised one.

Enforcing that claim on real output rather than on synthetic data
(`run_coverage_gate.py`) shows it does **not** hold out of the box: of the 30
(operator, lead time, level) configurations of marginal coverage, only 2 meet the ±2%
target, and essentially every miss is an *under*-coverage. The
one-sidedness is diagnostic — split conformal assumes exchangeability, and a 55-day
trace with a trend does not supply it. `AdaptiveConformalInference` is the remedy,
updating the requested level online from realised breaches so long-run coverage
converges without any exchangeability assumption. It passes **30 of 30**, at every
lead time from 1.4 to 24 hours on both operators, against 2 to 6 for the static
methods.

The payoff is measured in cost, not in RMSE. Conformal is evaluated at the level the
theory prescribes from the cost ratio — no test-set information — while the
fixed-margin rule is given its *best* margin chosen with hindsight, which biases the
comparison against the proposed method:

| κ = 10 | cost at τ* = 0.909 | vs hindsight-tuned fixed margin |
|---|---|---|
| GP, context-adaptive conformal | **19.27** | **−7.5%** |
| GP, marginal conformal | 21.35 | +2.4% (worse) |
| Robi, context-adaptive conformal | **49.34** | **−14.8%** |
| Robi, marginal conformal | 59.09 | +2.0% (worse) |

Read the pattern rather than the winner: conditioning the margin on *predicted
uncertainty* beats the tuned heuristic on both operators; calibrating one margin for
all conditions loses to it on both.

The capacity-at-equal-SLA comparison from the original plan is **not** supported and
is not claimed — see `STATUS.md` for why that metric structurally favours a heuristic
whose frontier can only be drawn with hindsight.

**4. Makes that allocation context-conditional.** The GP trace carries nine
hand-labelled contextual flags. Testing them shows most do not predict the *level* of
demand — including `is_weekend`, which both source CSVs are named after (p = 0.77).
But `is_rain` raises residual σ from 16.4 to 21.8 Gbps, a **33% increase in
uncertainty** (Levene p < 0.001). Context here acts on the *variance*, not the mean.

A point forecaster with a context dummy cannot use that; a context-conditional
prediction interval can. A marginally calibrated allocator hits its 95% target overall
while delivering only ~88% inside the elevated-risk group — the failures cluster
exactly where the network is most stressed. Locally adaptive and Mondrian conformal
calibration repair it at near-identical capacity cost.

The claim is falsifiable and is reported in both directions: it holds on GP, and is
**correctly null on Robi**, where no flag has elevated variance and the method
accordingly does nothing.

---

## Cold start: how much history does a new cell site need?

`run_transfer.py` replaces `Robi_to_GP.ipynb` from the original project — a notebook
titled as a Robi→GP transfer study that loads the GP file for both halves and then adds
20 Gbps to the test actuals before plotting them.

This is possible only because of the correction above. At 86 and 99 min per sample, a
design matrix indexed by sample count is not comparable across the two sites; because
lags are defined in wall-clock hours and seasonality in Fourier terms of wall-clock
time, a model can move between them.

At a 6-hour lead, against a persistence floor of 26.5 RMSE:

| days of the new site's own history | 3 | 5 | 7 | 14 | 21 |
|---|---|---|---|---|---|
| its own data alone | 20.61 | 19.81 | 18.98 | **14.82** | 15.86 |
| pretrained elsewhere, then fine-tuned | **19.65** | 19.87 | 19.91 | 15.63 | 17.45 |
| pretrained elsewhere, no fine-tuning | 20.39 | 20.01 | 19.74 | 19.06 | 19.04 |

A model trained entirely on the *other* operator, given only enough history to fix the
new site's level and scale, is already 23% better than persistence with zero training
data of its own. Pretraining then helps only while the site is data-poor — +4.6% at 3
days, nothing by 5–7, and −10% by 21, where the borrowed weights hold the model back.
**The crossover is about a week.**

---

## Zero-shot foundation models, and a use for the audit

`run_foundation.py` runs Chronos-Bolt Small strictly zero-shot — no fitting, no fine-tuning —
on the same folds, leads and test blocks as the trained models. It takes ~7 minutes on CPU.

**The answer depends entirely on the operator, and that is the finding.**

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

On GP this is a credible drop-in system: it beats persistence at every lead and trails
bespoke training by only 7.5–18.6%. On Robi it trails by 47–92% and *loses* to persistence
at both the shortest and the longest lead.

The reason turns the audit into a diagnostic. A foundation model reads a bare sequence with
**no timestamps**. It cannot know that a step is 86 min on one trace and 99 on the other, so
it must infer the daily cycle from the sequence — and under irregular sampling that cycle is
not a whole number of steps:

| | gap | exact daily period | misregistration |
|---|---|---|---|
| GP | 86 min | 16.744 samples | 0.256 samples/day |
| Robi | 99 min | **14.545 samples** | **0.455 samples/day** |

Robi's cycle slips roughly twice as fast — about 25 samples, or 1.7 full cycles, of drift
across the 55-day trace against GP's 14. And the single worst result in the experiment is
Robi at a 24-hour lead: the one setting where the daily cycle *is* the signal, and where the
naive baseline wins by stepping back exactly one **measured** cycle — precisely the operation
a timestamp-blind model cannot perform.

So measuring the sampling interval does not merely repair the earlier study. It tells you in
advance whether a zero-shot foundation model will work on your trace at all.

**Its quantiles are not usable for provisioning as they come.** Chronos-Bolt's quantile head
was trained on 0.1–0.9 only, so τ = 0.95 is silently clipped to 0.90 and returns identical
numbers — which through τ\* = κ/(1+κ) caps the expressible cost ratio at κ = 9. Robi's
nominal 80% interval delivers 66% coverage. Conformal calibration repairs every level on both
traces:

| operator | τ | zero-shot | conformalised |
|---|---|---|---|
| GP | 0.80 | 0.791 | 0.815 |
| GP | 0.90 | 0.886 | 0.903 |
| GP | 0.95 | 0.886 *(clipped)* | 0.941 |
| Robi | 0.80 | **0.662** | 0.813 |
| Robi | 0.90 | **0.786** | 0.909 |
| Robi | 0.95 | 0.786 *(clipped)* | 0.950 |

The synthesis: the foundation model supplies a cheap point forecast; conformal calibration
supplies the service-level guarantee it cannot.

---

## Quick start

### Colab and Kaggle

The repository is public, so no credentials are needed. Open any notebook in
`notebooks/` and run the first cell — it clones the repo, or pulls if it is already
there, and puts `src/` on the path.

- **Colab** — File → Open notebook → GitHub → `https://github.com/sad-code-at/bwalloc`.
- **Kaggle** — File → Import Notebook → GitHub, and switch **Internet on** in Settings.

After pushing a change from your laptop, re-run that first cell to pull it; restart the
kernel too if you changed anything under `src/`.

Colab and Kaggle both ship everything notebooks 00–05, 07 and 08 need. Only notebook 06
wants an extra install:

```python
!pip install -q chronos-forecasting
```

Nothing here needs a GPU on either platform. Full walkthroughs, the push-from-notebook
recipe and the common failure modes are in [`docs/COLAB.md`](docs/COLAB.md) and
[`docs/KAGGLE.md`](docs/KAGGLE.md).

### Local

```bash
pip install -r requirements.txt
python experiments/run_audit.py         # data audit, seasonality, flag validation
python experiments/run_benchmark.py     # corrected benchmark + ablation + DM tests
python experiments/run_allocation.py    # capacity frontier + context-conditional coverage
python experiments/run_horizon.py       # accuracy and allocation vs lead time (~15 min)
python experiments/run_sequence.py      # CNN/LSTM/GRU/RNN vs trees (~4 min, torch)
python experiments/run_transfer.py      # cross-operator cold start (~2 min)
python experiments/run_foundation.py    # zero-shot Chronos-Bolt (~7 min, CPU)
python experiments/run_coverage_gate.py # the +/-2% coverage check, on real output
pytest tests/                           # 58 verification gates
```

All tables land in `experiments/results/` as CSV; figures read only from there, so the
paper regenerates without refitting anything.

---

## Layout

```
data/                    gp_dhaka.csv, robi_dhaka3.csv
src/bwalloc/
  data.py                loading; SamplingProfile — every seasonal parameter derives from it
  features.py            leak-safe design matrices; Fourier terms for irregular sampling
  splits.py              rolling-origin folds; calibration slice; multi-horizon embargo
  baselines.py           persistence, seasonal-naive, drift, rolling mean, train mean
  models.py              point and quantile forecasters behind one interface
  forecast.py            direct and recursive multi-horizon; per-horizon naive baselines
  evaluate.py            backtest harness; Diebold-Mariano with FDR control
  context.py             flag validation, disjoint groups, the uncertainty model
  conformal.py           marginal, relative, locally adaptive, Mondrian, and online (ACI)
  sequence.py            univariate CNN / LSTM / GRU / RNN on the corrected protocol
  allocation.py          cost model, allocation policies, capacity-risk frontier
  pipeline.py            end-to-end allocation backtest
  metrics.py             RMSE/MAE/MASE/pinball; SLA rate, overprovisioning, cost, coverage
  plots.py               shared figure style
experiments/             runnable studies; results/ holds every generated table
tests/                   verification gates (see below)
notebooks/               Colab-first analysis notebooks, generated by notebooks/_build.py
  00_data_audit          sampling truth, corrected ACF, leak check, context-flag audit
  01_corrected_benchmark identical folds, baselines, DM tests, feature ablation
  02_allocation          cost model, tau* = kappa/(1+kappa), conformal, capacity frontier
  03_context_conditional the context-conditional result on GP and its null on Robi
  04_multi_horizon       lead time, direct vs recursive, allocation at h > 1
  05_transfer            cold start: how much history a new cell site needs
  06_foundation_models   zero-shot Chronos-Bolt, and when it fails
  07_paper_figures       regenerates every figure from experiments/results/ alone
  08_sequence_models     CNN/RNN/LSTM/GRU vs trees on one window; and lag depth
paper/                   paper.md (full draft) and figures/ (9 rendered figures)
docs/WALKTHROUGH.md      the teaching document: read this first
docs/PROVENANCE.md       every method mapped to the paper it came from
docs/BENCHMARKS.md       published RMSE/MAPE figures, and why raw RMSE does not compare
docs/COLAB.md            running the notebooks on Colab from a private repo
docs/KAGGLE.md           the same for Kaggle, plus the no-token Dataset route
STATUS.md                working record — what is verified, what is open, what to be careful of
```

The notebooks are generated from `notebooks/_build.py`, stay thin over library calls,
and are committed **with their outputs** so they read on GitHub without being run.
Regenerating strips those outputs, so the full recipe is `python notebooks/_build.py`
then `python -m nbconvert --to notebook --execute --inplace notebooks/0*.ipynb`.
Only `07_paper_figures` writes into `paper/figures/`; the rest write to
`notebooks/figures/`, which is gitignored. The earlier project kept ~7 near-duplicate copies of the same feature block
which had silently drifted apart; generating them from one source makes that
impossible rather than merely discouraged.

---

## Verification gates

`pytest tests/` (58 gates) encodes the specific failures found in the earlier work so a refactor
cannot silently reintroduce them:

| gate | what it protects |
|---|---|
| `test_no_target_leakage` | no feature depends on the contemporaneous target |
| `test_sampling_is_not_hourly` | the sampling-rate fact stays measured, not assumed |
| `test_daily_lag_is_derived_not_hardcoded` | `lag_for_hours(24)` resolves to 17/15, never 24 |
| `test_true_period_beats_lag24` | the empirical consequence of the above |
| `test_persistence_sanity_floor` | GP 12.1845 / Robi 31.4122 — data integrity after any refactor |
| `test_folds_never_look_ahead` | no fold trains on data after its test block |
| `test_training_windows_are_not_starved` | guards the inverted split (88 train / 480 test) |
| `test_estimability_guard_refuses_rather_than_clips` | a τ the calibration set cannot express raises |
| `test_marginal_conformal_under_covers_the_volatile_group` | the failure the context-conditional method fixes |
| `test_optimal_tau_is_the_cost_minimiser` | the κ ↔ τ correspondence holds empirically |
| `test_history_row_matches_build_features` | the recursive and batch feature paths cannot drift apart |
| `test_direct_design_reads_history_at_the_origin_not_the_target` | the multi-horizon form of the leak above |
| `test_horizon_baseline_is_not_the_one_step_baseline` | an *h*-step model is scored against an *h*-step naive |
| `test_embargo_prevents_training_on_post_origin_outcomes` | no training target postdates the first test origin |
| `test_split_conformal_under_covers_under_drift` | the measured failure, reproduced in isolation |
| `test_adaptive_conformal_inference_recovers_coverage_under_drift` | and the fix for it |
| `test_adaptive_conformal_inference_is_causal` | the online feedback runs strictly one step behind |

---

## Reporting conventions

- Every metric is `mean ± sd` across rolling-origin folds. A single-split point
  estimate on ~300 test points cannot separate these models.
- Every table includes the naive baselines. `run_benchmark.py` exits non-zero if the
  top-ranked model does not beat persistence.
- Model rankings carry Diebold-Mariano p-values with Benjamini-Hochberg FDR control
  (ten models is 45 comparisons).
- Per-group coverage is reported with Clopper-Pearson intervals and the group's
  calibration size, because the elevated-risk group carries ~50 calibration points and
  a point estimate from that sample is not meaningful alone.
- Context-conditional results are reported at **τ ≤ 0.95**. A calibration set of *m*
  residuals cannot express a level finer than `1/(m+1)`; `conformal.py` raises rather
  than silently clipping.

---

## Limitations

905 and 888 observations from two cell sites over ~2 months. The sample size does not
support strong claims about model superiority, and the two-month span cannot speak to
seasonal or annual effects. The context flags are hand-labelled by a single annotator
without an inter-rater check. Robi carries five of the nine flags and no rain
annotation, so its context-conditional results rest on a 89-row group and should be
read as directional only.

Two further limits are measured rather than asserted, and both are reported:

- **Correcting the feature design does not improve accuracy.** The ablation is a
  negative result on both operators: tree ensembles route around a mis-specified
  `lag_24` via the short lags, so the sampling-rate error costs almost nothing in RMSE.
  Its cost is interpretive — every seasonal claim in the earlier study was stated on
  the wrong time axis — not predictive.
- **Static conformal coverage does not meet its ±2% target on these traces.** Only 2
  of 30 marginal configurations do, essentially all misses being under-coverage,
  because exchangeability fails under drift. Absolute coverage guarantees should
  therefore not be claimed from the split-conformal results; the online variant passes
  30 of 30 and is what should be deployed. The context-conditional comparison is
  unaffected either way: it is a *relative* comparison between methods calibrated on
  identical data.

The methods here are chosen to be *robust to* those limits rather than to hide them:
conformal calibration is distribution-free and finite-sample valid, the estimability
guard makes the sample-size ceiling explicit rather than implicit, and the drift that
breaks exchangeability is addressed by an online method that does not assume it.
