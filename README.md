# Context-Conditional Bandwidth Allocation

Cost-aware, conformally calibrated bandwidth allocation from sparse, irregularly
sampled mobile operator traces.

This extends an earlier forecasting study on two Bangladeshi operator traces
(Grameenphone Dhaka and Robi Dhaka-3) in three directions: a corrected evaluation
protocol, an actual allocation layer, and context-conditional calibration of that
layer.

---

## What this adds

The earlier work compared ten forecasting models by RMSE. This repository does three
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

**2. Adds the allocation layer the title promises.** Forecasting `Gbps` is not
provisioning. Under- and over-provisioning have asymmetric costs, so the cost-optimal
allocation is a **quantile** of the predictive distribution, not its mean — with the
operator's cost ratio κ selecting the level via `τ* = κ/(1+κ)`. Split-conformal
calibration then makes the achieved service level match the promised one.

**3. Makes that allocation context-conditional.** The GP trace carries nine
hand-labelled contextual flags. Testing them shows most do not predict the *level* of
demand — including `is_weekend`, which both source CSVs are named after (p = 0.77).
But `is_rain` raises residual σ from 16.4 to 21.8 Gbps, a **33% increase in
uncertainty** (Levene p < 0.001). Context here acts on the *variance*, not the mean.

A point forecaster with a context dummy cannot use that; a context-conditional
prediction interval can. A marginally calibrated allocator hits its 95% target overall
while delivering only ~88% inside the elevated-risk group — the failures cluster
exactly where the network is most stressed. Locally adaptive and Mondrian conformal
calibration repair it at near-identical capacity cost.

---

## Quick start

### Colab

```python
!git clone https://github.com/<you>/bwalloc.git
%cd bwalloc
!pip install -q -r requirements.txt
import sys; sys.path.insert(0, "src")

import bwalloc as bw
bw.set_seed()
```

Then open any notebook in `notebooks/`.

### Local

```bash
pip install -r requirements.txt
python experiments/run_audit.py        # data audit, seasonality, flag validation
python experiments/run_benchmark.py    # corrected benchmark + ablation + DM tests
python experiments/run_allocation.py   # capacity frontier + context-conditional coverage
pytest tests/                          # verification gates
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
  splits.py              rolling-origin folds with a conformal calibration slice
  baselines.py           persistence, seasonal-naive, drift, rolling mean, train mean
  models.py              point and quantile forecasters behind one interface
  evaluate.py            backtest harness; Diebold-Mariano with FDR control
  context.py             flag validation, disjoint groups, the uncertainty model
  conformal.py           marginal, locally adaptive, and Mondrian calibration
  allocation.py          cost model, allocation policies, capacity-risk frontier
  pipeline.py            end-to-end allocation backtest
  metrics.py             RMSE/MAE/MASE/pinball; SLA rate, overprovisioning, cost, coverage
  plots.py               shared figure style
experiments/             runnable studies; results/ holds every generated table
tests/                   verification gates (see below)
notebooks/               Colab-first analysis notebooks
```

---

## Verification gates

`pytest tests/` encodes the specific failures found in the earlier work so a refactor
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

The methods here are chosen to be *robust to* those limits rather than to hide them:
conformal calibration is distribution-free and finite-sample valid, and the
estimability guard makes the sample-size ceiling explicit rather than implicit.
