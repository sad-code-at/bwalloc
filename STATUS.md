# Project status

Last updated: 2026-08-29. Working notes so this can be picked up cold in a new session.

Plan file: `C:\Users\User\.claude\plans\nested-plotting-bengio.md` (approved).
Original senior's code, untouched, is at
`D:\L4-T-1\EEE 402\project\ML-Based-Dynamic-Bandwidth-Allocation-for-Mobile-Data-Usage-main\`.
All new work is in `D:\L4-T-1\EEE 402\project\bwalloc\`.

---

## Done and verified

**Scaffold + library** — complete and tested. `pytest tests/` → **30 passed**.

| module | status |
|---|---|
| `data.py` | done — loading, `SamplingProfile`, ACF-by-lag |
| `features.py` | done — leak-safe builder, Fourier terms, `lag_samples` for faithful reproduction |
| `splits.py` | done — rolling-origin folds with calibration slice |
| `baselines.py` | done — persistence, seasonal-naive, drift, rolling mean, train mean |
| `models.py` | done — Ridge / RF / XGBoost point + `QuantileGBM` |
| `evaluate.py` | done — backtest harness, DM tests + BH correction |
| `context.py` | done — flag validation, disjoint groups, `UncertaintyModel` |
| `conformal.py` | done — marginal / locally adaptive / Mondrian + estimability guard |
| `allocation.py` | done — cost model, policies, Pareto sweep |
| `pipeline.py` | done — end-to-end allocation backtest |
| `metrics.py`, `stats.py`, `plots.py` | done |
| `tests/test_bwalloc.py` | done — 30 gates, all passing |

**Experiments run:** `run_audit.py`, `run_benchmark.py`, `run_allocation.py` — all
complete for both operators. Every table is in `experiments/results/`.

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

## The capacity-saving claim does NOT hold — this changes the paper's framing

`frontier()` now sweeps the conformal families properly (fixed 2026-08-29). With the
comparison done correctly, the result is still negative:

| operator | family | at ≤10% violations | vs fixed-margin |
|---|---|---|---|
| GP | conformal_adaptive | 1.175× demand | **−2.0%** |
| GP | conformal_marginal | 1.183× | −2.7% |
| Robi | conformal_adaptive | 1.328× | −0.3% |
| Robi | conformal_marginal | 1.298× | **+2.0%** |

And **no family reaches a 1%, 2% or 5% violation target** on either trace.

**Do not write the sentence from the plan** ("at an equal 1% SLA violation rate our
allocator provisions X% less capacity"). The data does not support it. Conformal
quantile allocation is roughly tied with the fixed-margin heuristic on aggregate
capacity efficiency here.

### Why, and the one fix worth trying

The comparison is not apples-to-apples. The fixed-margin rule is **multiplicative**
(`A = 1.3 × ŷ`), so its headroom scales with the forecast level. The conformal
correction as implemented is **additive** (`A = ŷ + q̂`), so it applies the same absolute
headroom at 60 Gbps as at 130 Gbps. On a series whose level varies ~2×, that is a real
handicap.

**Next action:** calibrate on *relative* residuals, `(y − ŷ)/ŷ`, giving
`A = ŷ · (1 + q̂)`. That makes the conformal method multiplicative too and is the fair
comparison. Implement as a `relative=True` option on `SplitConformal` /
`LocallyAdaptiveConformal`. This is the single highest-value experiment remaining.

The τ ceiling is the other binding constraint: `CONFORMAL_TAUS` stops at 0.95 because of
the elevated-risk group's calibration size, and achieved coverage at τ=0.95 is ~93%, so
~7% violations is the floor. Reaching a 1% target needs τ≈0.99, which needs ≥99
calibration residuals per group — not available. Report that as a data limitation, not a
method failure.

### What this means for the paper

The contribution narrows, and arguably improves. Drop the efficiency claim; lead with
the **reliability-across-contexts** claim, which is solidly supported:

> Marginal calibration meets its aggregate service target while silently under-serving
> the highest-variance operating conditions (96.3% vs 86.6% at τ=0.95 on GP).
> Context-conditional calibration closes most of that gap at no capacity cost, and is
> correctly inert on an operator whose flags carry no variance signal.

That is a statement about *who bears the risk*, not about saving money, and it is the
result this data can actually carry.

---

## Audit against the approved plan — what is missing

Roughly half the plan is delivered. Phases 0–3 are done in *code and results*; the
deliverable layer and Phases 4–5 are not. Honest inventory:

### Gaps that undermine current claims — fix these first

**1. Everything is still one-step-ahead.** `forecast.py` was never written. There is no
recursive or direct multi-horizon forecasting anywhere, so every result in this repo
predicts `y_t` with the true `y_{t-1}` in hand.

This is the sharpest gap, because §1.8 of the audit criticises the original project for
exactly this — "not forecasting, and useless for allocation, which needs a lead time" —
and this repo currently has the same limitation. An allocator that needs the previous
observation cannot provision ahead. **Until H>1 results exist, the allocation study
describes a system that could not be deployed.**
Fix: `forecast.py` with direct multi-horizon models for H ∈ {1, 3, 6, ~1 day}, then
re-run the allocation study at H>1. Expect all numbers to degrade substantially.

**2. The ±2% coverage gate from the plan was never enforced on real data.**
`test_split_conformal_achieves_nominal_coverage_when_exchangeable` only checks synthetic
exchangeable data, where it passes trivially. Checking the actual results:
**11 of 18 (operator, τ, method) configurations fail the ±2% criterion.**

| | τ=0.8 | τ=0.9 | τ=0.95 |
|---|---|---|---|
| GP | ok (−0.014 to −0.020) | **fails (−0.026 to −0.033)** | ok (−0.010 to −0.019) |
| Robi | **fails (−0.037 to −0.046)** | **fails (−0.041 to −0.051)** | **fails (−0.032 to −0.049)** |

Every gap is negative — conformal systematically **under-covers**, worse on Robi. That
is the exchangeability assumption failing under temporal drift, which is expected for
time series and is a legitimate finding, but it must be reported rather than discovered
by an ad-hoc check. Fix: add a real gate over `experiments/results/`, and consider
adaptive conformal methods designed for distribution shift (ACI / online conformal).

Note this does not invalidate the C2b result, which is a *relative* comparison between
methods on the same group — but it does mean no absolute coverage guarantee should be
claimed.

### Not started at all

- **All 7 notebooks.** `notebooks/` is empty. This is the primary deliverable format
  (the work runs in Colab), and the library + runners exist, so these are thin wrappers.
- **No figure has ever been rendered.** `plots.py` is written but never executed;
  `paper/figures/` is empty.
- **C3** — zero-shot foundation models (Chronos-Bolt, TimesFM).
- **C4** — cross-operator transfer study.
- **Paper** — no writeup.

### Smaller deviations from the plan

- Quantile-LSTM not implemented (`QuantileGBM` only).
- `pinball_loss` is implemented and tested but never used in any reported table.
- `QuantileGBM`'s lightgbm and sklearn backends are untested locally (lightgbm absent).
- The plan said to delete `Robi_to_GP.ipynb`; it was left in place, since the senior's
  original folder was kept untouched as a reference. Deliberate, but a deviation.

### Extras delivered beyond the plan

`evaluate.py` and `pipeline.py` (harness layers the plan did not name), the
`lag_samples`/`rolling_samples` escape hatch for faithful reproduction of the original
design, and the ablation study that produced the negative result in §Ablation.

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
PYTHONPATH=src python -m pytest tests/ -q     # 30 gates
python experiments/run_audit.py
python experiments/run_benchmark.py
python experiments/run_allocation.py
```
