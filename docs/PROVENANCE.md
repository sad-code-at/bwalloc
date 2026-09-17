# Provenance: where every idea in this project came from

One row per methodological decision, naming the paper it came from, where it is used in
this repository, and what we took from it. Written so that "why did you do it that way?"
has an answer with a citation attached, and so that the project's **own** contributions
are clearly separated from borrowed machinery.

**Keep this file current.** When a new method, test or design decision enters the
project, add its row in the same commit as the code. A source added later is a source
nobody can check.

## How to read the status column

| | meaning |
|---|---|
| **verified** | Bibliographic details machine-checked against the arXiv API or the publisher, on the date noted. Safe to cite as written. |
| **standard** | A classical reference we are confident of but have not machine-checked. Correct to the best of our knowledge; **verify against the original before submission**. |
| **ours** | No source paper. This is the project's own contribution or an original finding. |
| **practice** | Established practice with no single canonical paper to cite. |

Papers marked **[cited]** appear in the paper's reference list. The rest are background
that shaped a decision without being cited, which is still worth being able to name.

---

## 1. Problem framing

| Idea | Where used | Source | Status |
|---|---|---|---|
| Cellular traffic forecasting is dominated by "rank models by RMSE"; this is the framing we depart from | Paper §1, §2 | Wang et al., *A Survey on Deep Learning for Cellular Traffic Prediction*, Intelligent Computing 3:0054, 2024 **[cited]** | verified (DOI); author list incomplete |
| Predicting under explicit SLA-violation constraints rather than squared error | Paper §2, §6 framing | Tuna & Soysal, arXiv:2304.11156, 2023 **[cited]** | verified 2026-09-17 |
| Multi-step, spatiotemporal traffic prediction under SLA constraints | Paper §2 | Tuna & Soysal, arXiv:2309.03898, 2023 **[cited]** | verified 2026-09-17 |
| Optimising slice resources against a learned SLA-satisfaction model — the closest prior work to our allocation layer | Paper §2 | Sulaiman, Ahmadi, Sun, Saha, Salahuddin, Boutaba & Saleh, *MicroOpt*, arXiv:2407.18342, 2024 **[cited]** | verified 2026-09-17 |
| That data quality and preprocessing, not architecture, dominate cellular-traffic prediction results | Supports our §4 audit framing | Vesselinova, Harjula & Ilmonen, *Data Matters: The Case of Predicting Mobile Cellular Traffic*, arXiv:2411.02418, 2024 (IEEE VNC 2025) | verified 2026-09-18 |
| **That the allocation decision, not the forecast, is the deliverable — and that the original repository contained no allocation at all** | The entire `allocation.py` / `pipeline.py` layer | — | **ours** |

## 2. The audit

| Idea | Where used | Source | Status |
|---|---|---|---|
| Target leakage: a feature computed from data at or after the target's timestamp inflates results invisibly | `features.assert_no_leakage`, paper §4.2, gate `test_no_target_leakage` | Kaufman, Rosset & Perlich, *Leakage in Data Mining: Formulation, Detection, and Avoidance*, KDD 2011; ACM TKDD 6(4), 2012 | standard |
| **That these two traces are sampled at 86 and 99 min, not hourly, and that `lag_24` therefore spans 34.4 h and anti-correlates with demand** | `data.SamplingProfile`, paper §4.1 | — | **ours** |
| **That the sampling interval predicts whether a timestamp-blind foundation model will work on a trace** | Paper §9.2 | — | **ours** |
| **That the original's sequence-vs-tree ranking was confounded by an unequal split (600 rows vs 88)** | Paper §4.5, `run_sequence.py` | — | **ours** |
| **That the corrected feature design was under-lagged, costing 18% RMSE on GP** | Paper §4.5, `sequence_lag_depth.csv` | — | **ours** |

## 3. Evaluation protocol

| Idea | Where used | Source | Status |
|---|---|---|---|
| Rolling-origin evaluation instead of a single split | `splits.rolling_origin`, paper §4.3 | Tashman, *Out-of-sample tests of forecasting accuracy*, Int. J. Forecasting 16(4):437–450, 2000 **[cited]** | standard |
| Cross-validation is usable for time series if the temporal order is respected | `splits.py` design | Bergmeir & Benítez, *On the use of cross-validation for time series predictor evaluation*, Information Sciences 191:192–213, 2012 | standard |
| MASE — scale-free error against a naive baseline | `metrics.mase`, every results table | Hyndman & Koehler, *Another look at measures of forecast accuracy*, Int. J. Forecasting 22(4):679–688, 2006 **[cited]** | standard |
| Diebold–Mariano test for equal predictive accuracy | `evaluate.pairwise_dm`, `dm_matrix` | Diebold & Mariano, *Comparing Predictive Accuracy*, J. Business & Economic Statistics 13(3):253–263, 1995 **[cited]** | standard |
| Small-sample correction to the DM statistic | `evaluate.py` | Harvey, Leybourne & Newbold, *Testing the equality of prediction mean squared errors*, Int. J. Forecasting 13(2):281–291, 1997 | standard |
| Benjamini–Hochberg FDR control across many pairwise comparisons | `evaluate.dm_matrix` (45 comparisons) | Benjamini & Hochberg, *Controlling the False Discovery Rate*, JRSS-B 57(1):289–300, 1995 **[cited]** | standard |
| Embargo/purge between training and test to stop training on outcomes that postdate the forecast origin | `splits.py` embargo, gate `test_embargo_prevents_training_on_post_origin_outcomes` | López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018 (purged K-fold with embargo) | standard |
| **A benchmark that exits non-zero if the top model does not beat persistence** | `run_benchmark.py` | — | **ours** (practice, enforced) |

## 4. Multi-horizon forecasting

| Idea | Where used | Source | Status |
|---|---|---|---|
| Direct vs recursive strategies for multi-step forecasting, and that the choice matters | `forecast.py`, paper §5 | Ben Taieb, Bontempi, Atiya & Sorjamaa, *A review and comparison of strategies for multi-step ahead time series forecasting...*, arXiv:1108.3259, 2011; Expert Systems with Applications 39(8), 2012 | verified 2026-09-18 |
| Direct multi-step estimation as an econometric strategy | `forecast.py` | Chevillon, *Direct Multi-step Estimation and Forecasting*, Journal of Economic Surveys 21(4):746–785, 2007 | standard |
| **That an *h*-step model must be scored against an *h*-step naive baseline, not one-step persistence** | `forecast.persistence_at_horizon`, gate `test_horizon_baseline_is_not_the_one_step_baseline` | — | **ours** (a correctness point, not a novel method) |
| **That history features read at the origin and calendar features read at the target is the correct split** | `forecast.direct_design`, gate `test_direct_design_reads_history_at_the_origin_not_the_target` | — | **ours** |

## 5. Allocation and cost asymmetry

| Idea | Where used | Source | Status |
|---|---|---|---|
| Newsvendor / critical-fractile result: under asymmetric over- and under-supply costs the optimal order quantity is a quantile, τ\* = κ/(1+κ) | `allocation.optimal_tau`, paper §6.1, gate `test_optimal_tau_is_the_cost_minimiser` | Arrow, Harris & Marschak, *Optimal Inventory Policy*, Econometrica 19(3):250–272, 1951 | standard |
| Quantile regression and the pinball loss | `models.QuantileGBM`, `metrics.pinball_loss` | Koenker & Bassett, *Regression Quantiles*, Econometrica 46(1):33–50, 1978 **[in refs.bib]** | standard |
| **Applying the critical fractile to bandwidth provisioning, so the operator's cost ratio selects the service level with no retraining** | `allocation.py`, paper §6 | — | **ours** |
| **That capacity-at-equal-SLA is not a well-posed comparison, because the fixed-margin frontier can only be drawn with hindsight** | Paper §6.3 | — | **ours** (a negative result) |

## 6. Conformal prediction

| Idea | Where used | Source | Status |
|---|---|---|---|
| Conformal prediction: distribution-free, finite-sample valid coverage from exchangeable residuals | All of `conformal.py` | Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*, Springer, 2005 **[cited]** | standard |
| Inductive / split conformal — calibrate on a held-out slice instead of refitting | `conformal.SplitConformal` | Papadopoulos, Proedrou, Vovk & Gammerman, *Inductive Confidence Machines for Regression*, ECML 2002 | standard |
| Split-conformal regression, its guarantees and its practical form | `conformal.py` | Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, *Distribution-Free Predictive Inference For Regression*, arXiv:1604.04173, 2016; JASA 113(523), 2018 | verified 2026-09-18 |
| Modern textbook treatment | Background for §6.2, §10 | Angelopoulos, Barber & Bates, *Theoretical Foundations of Conformal Prediction*, arXiv:2411.11824, 2024 **[cited]** | verified 2026-09-17 |
| Locally adaptive / normalised nonconformity — interval width scales with predicted difficulty | `conformal.LocallyAdaptiveConformal` | Papadopoulos, Gammerman & Vovk, normalised nonconformity measures (ECML 2002 line of work); see also Lei et al. 2016 | standard |
| Conformalized quantile regression | `models.QuantileGBM` + conformal layer, paper §2 | Romano, Patterson & Candès, *Conformalized Quantile Regression*, NeurIPS 32, 2019 **[cited]** | standard |
| Mondrian (group-conditional) conformal regression | `conformal.MondrianConformal`, paper §7 | Boström & Johansson, *Mondrian conformal regressors*, COPA 2020, PMLR 128:114–133 | verified 2026-09-18 |
| Small calibration sets in Mondrian conformal regressors — directly relevant, our elevated-risk group carries ~50 calibration residuals | `conformal.min_calibration_size`, the estimability guard | Linusson, Johansson, Boström & Löfström, *Handling Small Calibration Sets in Mondrian Inductive Conformal Regressors*, SLDS 2015, LNCS 9047 | standard |
| Exact conditional coverage is impossible distribution-free — why we do not claim it | Paper §7.2, §11 | Foygel Barber, Candès, Ramdas & Tibshirani, *The limits of distribution-free conditional predictive inference*, arXiv:1903.04684, 2019 | verified 2026-09-18 |
| What conditional coverage *is* achievable, against a specified class of shifts | Paper §2, §7 | Gibbs, Cherian & Candès, *Conformal Prediction With Conditional Guarantees*, arXiv:2305.12616, 2023 **[cited]** | verified 2026-09-17 |
| Adaptive conformal inference — update the level online from realised breaches, α ← α + γ(α_target − err) | `conformal.AdaptiveConformalInference`, paper §10 | Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift*, arXiv:2106.00170, 2021; NeurIPS 34 **[cited]** | verified 2026-09-18 |
| **That hand-labelled operating context acts on the *variance* of demand, not its level, and is therefore usable as a calibration group rather than a mean predictor** | Paper §7.1, `context.py` | — | **ours** — the paper's central claim |
| **That a marginally valid allocator is systematically invalid inside the high-variance group, established with Clopper–Pearson intervals** | Paper §7.2, gate `test_marginal_conformal_under_covers_the_volatile_group` | — | **ours** |
| **That an estimability guard should refuse a τ the calibration set cannot express, rather than silently clipping** | `conformal.min_calibration_size`, gate `test_estimability_guard_refuses_rather_than_clips` | Bound follows from Linusson et al. 2015; the refuse-don't-clip behaviour is ours | **ours** (implementation) |

## 7. Sequence and foundation models

| Idea | Where used | Source | Status |
|---|---|---|---|
| Chronos — tokenised time-series values through a language-model architecture, marketed on zero-shot transfer | `run_foundation.py`, paper §9 | Ansari et al. (18 authors), *Chronos: Learning the Language of Time Series*, arXiv:2403.07815, 2024 **[cited]** | verified 2026-09-17 |
| TimesFM — decoder-only pretrained forecasting model | Paper §2; not run | Das, Kong, Sen & Zhou, *A decoder-only foundation model for time-series forecasting*, arXiv:2310.10688, 2023 **[cited]** | verified 2026-09-17 |
| CNN / LSTM / GRU / RNN architectures for traffic forecasting | `sequence.py` | Architectures taken directly from the original study's notebooks, which follow standard practice; see Wang et al. 2024 survey for the family | practice |
| **That the original's sequence-model win was a split artefact, and that re-run fairly the CNN still wins on GP (p = 0.003) while everything ties on Robi** | Paper §4.5 | — | **ours** |

## 8. Statistical tests

| Idea | Where used | Source | Status |
|---|---|---|---|
| Welch's unequal-variance *t*-test — flag effect on the level | `context.py` flag audit, paper §7.1 | Welch, *The generalization of Student's problem when several different population variances are involved*, Biometrika 34(1/2):28–35, 1947 | standard |
| Levene / Brown–Forsythe test for equality of variances — flag effect on the spread | `context.py` flag audit, paper §7.1 | Levene, 1960, in *Contributions to Probability and Statistics*; Brown & Forsythe, JASA 69(346):364–367, 1974 | standard |
| Clopper–Pearson exact binomial interval — used instead of a normal approximation at n = 179 | Paper §7.2 coverage tables | Clopper & Pearson, *The use of confidence or fiducial limits illustrated in the case of the binomial*, Biometrika 26(4):404–413, 1934 | standard |

---

## What is ours

Collected for convenience — these are the rows marked **ours** above, and are what a
viva question about contribution should be answered with:

1. **The sampling-rate audit** and the demonstration that it invalidates every seasonal
   claim in the original study.
2. **The allocation layer itself.** The original repository predicts `Gbps` and stops;
   there is no cost model, no service level, no allocation.
3. **Context acts on variance, not level** — and is therefore useful as a calibration
   group even though it is a poor mean predictor. This is the paper's central claim.
4. **The failure of marginal calibration inside the high-variance group**, established
   with exact binomial intervals rather than point estimates.
5. **Sampling interval as a pre-deployment diagnostic** for timestamp-blind foundation
   models.
6. **The confound in the original's model ranking** (600 training rows against 88), and
   the corrected comparison that confirms its conclusion on GP and refutes it on Robi.
7. **That the corrected design was under-lagged**, costing 18% RMSE on GP.
8. **Three negative results reported rather than buried**: correcting the features does
   not improve RMSE; capacity-at-equal-SLA is not a well-posed comparison; static
   conformal coverage fails its ±2% target almost everywhere.

## What we borrowed without modification

Conformal prediction, the critical-fractile result, Diebold–Mariano, Benjamini–Hochberg,
MASE, rolling-origin evaluation, ACI, and the Chronos model. None of these are ours and
the paper does not imply otherwise.

---

*Last updated 2026-09-18. arXiv entries verified against the arXiv API on the dates
shown. Entries marked "standard" are classical references stated from bibliographic
knowledge and should be spot-checked against the originals before submission.*
