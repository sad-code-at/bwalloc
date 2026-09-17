# External benchmarks: how our numbers compare to published work

A place to check our models against what the literature reports, and to keep checking as
models change. **Add a row whenever a model is added or a published comparison is found.**

---

## Read this first: raw RMSE is not comparable across papers

This is the single most important thing on this page, and it is the thing a supervisor is
most likely to probe.

RMSE carries the units and the scale of the series it was computed on. Our traces average
**92.9 Gbps (GP)** and **145.5 Gbps (Robi)**. The most-used public dataset in this field,
Telecom Italia Milan, is measured in *Call Detail Record activity units* with no physical
unit at all. A published "RMSE 57.18" on Milan and our "RMSE 7.88" on GP are not on the
same axis, and neither number is better than the other. Quoting them side by side would be
a serious error.

Three things **are** comparable across datasets, and the tables below use them:

| Metric | Why it transfers | Our best (GP / Robi) |
|---|---|---|
| **MAPE** — mean absolute percentage error | Unit-free by construction | **6.88% / 13.07%** |
| **NRMSE** — RMSE ÷ mean demand | Expresses error as a fraction of typical load | **8.5% / 13.7%** |
| **Improvement over a naive baseline** | Measures skill, not scale | **−35.3% / −33.4%** vs persistence |
| **MASE** — error ÷ in-sample naive error | Unit-free, <1 means better than naive | **0.399 / 0.820** |

A caveat that cuts the other way: these are still not strictly comparable, because
difficulty differs by dataset. A trace aggregated over 470 base stations is smoother and
easier than a single cell site, and 900 samples over 55 days is a harder setting than
4,320 samples over a month. Treat the tables as *orientation*, not as a league table.

---

## Our current results

Full corrected protocol, 8 rolling-origin folds. Regenerate with `run_benchmark.py` and
`run_sequence.py`.

### Grameenphone Dhaka — mean demand 92.9 Gbps

| Model | RMSE | NRMSE | MAPE | MASE | vs persistence |
|---|---|---|---|---|---|
| **CNN** (24-lag window) | **7.88 ± 1.03** | **8.5%** | **6.88%** | **0.399** | **−35.3%** |
| GRU (24-lag) | 8.56 ± 1.03 | 9.2% | 7.71% | 0.448 | −29.6% |
| RNN (24-lag) | 8.63 ± 0.87 | 9.3% | 7.85% | 0.457 | −29.1% |
| Random forest (24-lag) | 8.67 ± 1.43 | 9.3% | 7.33% | 0.417 | −28.7% |
| XGBoost (24-lag) | 8.85 ± 1.66 | 9.5% | 7.42% | 0.424 | −27.3% |
| Random forest (4-lag design) | 10.41 ± 1.19 | 11.2% | 8.91% | 0.514 | −14.1% |
| *Persistence* | *12.12 ± 0.92* | *13.1%* | *8.16%* | *0.464* | — |

### Robi Dhaka-3 — mean demand 145.5 Gbps

| Model | RMSE | NRMSE | MAPE | MASE | vs persistence |
|---|---|---|---|---|---|
| **GRU** (24-lag) | **19.88 ± 4.66** | **13.7%** | 13.53% | **0.820** | **−33.4%** |
| Ridge (24-lag) | 20.14 ± 4.22 | 13.8% | **13.07%** | 0.824 | −32.6% |
| CNN (24-lag) | 20.16 ± 3.92 | 13.9% | 13.75% | 0.832 | −32.5% |
| Random forest (4-lag design) | 20.75 ± 4.42 | 14.3% | 13.82% | 0.888 | −30.3% |
| *Persistence* | *29.75 ± 4.72* | *20.4%* | *17.79%* | *1.032* | — |

**Note the MAPE inversion on GP.** Persistence has a *better* MAPE (8.16%) than random
forest on the four-lag design (8.91%) while having a much worse RMSE. That is not an
error: MAPE weights proportional error, so it rewards a model that tracks the low-demand
troughs, while RMSE punishes the large absolute misses at peak. For provisioning, the
peaks are what matter — which is part of why this project scores on cost rather than on
either metric. Worth being able to explain if asked.

---

## Published results

Ordered by how directly they bear on this work.

### Cost-aware capacity forecasting — the closest prior work

| Paper | Venue | Dataset | What they report |
|---|---|---|---|
| Bega, Gramaglia, Fiore, Banchs & Costa-Pérez, **DeepCog** | **IEEE INFOCOM 2019**; extended in **IEEE JSAC 38(2):361–376, 2020** | Operational network, ~100 km² metro region, 470 4G eNodeBs, GTP gateway, bytes; 5-min steps; 2 months train + 2 weeks test; horizon 5 min | **Normalised monetary cost**, split into overprovisioning and SLA violations. Gains over the best competing benchmark of **273–381%** at cost ratio α = 2, and **up to 87%** at α = 0.5. Baselines: naive (same time last week), Infocom17, MobiHoc18, MAE-loss deep predictor. |

**This paper matters to us more than any other on this page, for three reasons.**

1. It does what we do — forecast *capacity* under asymmetric costs, not traffic under
   squared error — and it is in a top-tier venue. **It is prior work we must cite**, and
   currently the paper does not.
2. It reports **cost, not RMSE**, which independently validates our §6.3 argument that
   accuracy is the wrong scoreboard for this problem.
3. Its `MAE-post-best` baseline is a fixed overprovisioning offset chosen by exhaustive
   search over the test data — **exactly the hindsight-tuned fixed margin our §6.3 compares
   against**. Two independent groups converged on the same comparison.

**How we differ**, and this is the positioning to state explicitly: DeepCog trains the
asymmetry *into the network* via a custom loss, so the cost ratio is baked into the
weights and changing it means retraining. We keep an ordinary forecaster and move the
asymmetry into a calibrated quantile, so κ is a dial at inference time. We also give a
distribution-free finite-sample coverage guarantee, which DeepCog does not; they have far
more data and do not need one.

### Accuracy benchmarks on public cellular datasets

Numbers are in each dataset's own units — **do not compare them to ours directly**, use the
MAPE/NRMSE column where available.

| Paper | Venue / year | Dataset | Model | MAE | RMSE |
|---|---|---|---|---|---|
| Wang et al., AHSTGNN (arXiv:2303.00498) | 2023 | Jiangsu, 1,051 nodes, 15-min, Jan–Mar 2021 | HA | 196.76 | 334.74 |
| " | " | " | LSTM | 168.46 | 313.21 |
| " | " | " | Graph WaveNet | 130.83 | 254.28 |
| " | " | " | MVSTGN | 164.14 | 319.57 |
| " | " | " | MTGNN | 130.61 | 253.26 |
| " | " | " | AGCRN | 129.01 | 251.19 |
| " | " | " | AMF-STGCN | 129.78 | 252.71 |
| " | " | " | **AHSTGNN** | **124.81** | **243.91** |
| " | " | Milan, 900 nodes, 10-min, Nov 2013 | HA | 61.28 | 120.73 |
| " | " | " | LSTM | 43.28 | 79.77 |
| " | " | " | Graph WaveNet | 32.71 | 65.35 |
| " | " | " | MVSTGN | 35.03 | 70.29 |
| " | " | " | MTGNN | 29.13 | 57.63 |
| " | " | " | AGCRN | 30.27 | 59.81 |
| " | " | " | AMF-STGCN | 30.59 | 57.89 |
| " | " | " | **AHSTGNN** | **27.96** | **57.18** |

> Row alignment above was recovered from a PDF layout extraction; **check it against the
> published Table II before citing any single row.** The best/worst ordering is consistent
> with the paper's prose, but individual rows could be off by one.

**What this table is actually useful for.** Not the absolute numbers — the *spread*. Across
eight models spanning simple LSTM to state-of-the-art spatio-temporal graph networks, Milan
RMSE moves from 120.73 to 57.18. The historical-average baseline is roughly 2× worse than
the best model, and the gap between a plain LSTM and a 2023 graph network is about 28%. Our
own spread — persistence 12.12 down to CNN 7.88 on GP, a 35% improvement — sits in a
comparable band. That is the honest comparison available: *how much does modelling buy over
a naive baseline*, not whose RMSE is smaller.

### Single-cell prediction, and an independent confirmation of our §4.1

| Paper | Venue / year | Dataset | Model | MSE | implied RMSE |
|---|---|---|---|---|---|
| Mehri, Chen & Mehrpouyan, *Cellular Traffic Prediction Using Online Prediction Algorithms* (arXiv:2405.05239) | 2024 | Telecom Italia Milan CDR, single cell, 10-min, Dec 2013 | ARIMA(3,0,4) | 513.53 | ≈22.7 |
| " | " | " | **SARIMA(1,0,1)(1,0,1,144)** | **61.78** | **≈7.9** |
| " | " | " | LSTM + FLSP (online) | 34.71 | ≈5.9 |
| " | " | " | LSTM + rolling | 62.67 | ≈7.9 |
| " | " | " | ConvLSTM + FLSP | 66.85 | ≈8.2 |
| " | " | " | CNN-LSTM + FLSP | 134.72 | ≈11.6 |

**Read the first two rows together — they are our §4.1 finding, independently.** ARIMA
without a seasonal term scores MSE 513.53. The *same family* with the seasonal period set
correctly — 144 steps, which is exactly 24 hours at 10-minute sampling — scores 61.78. An
**8.3× improvement from getting the daily period right**, on a different dataset, by a
different group.

That is strong external support for our audit: we showed seasonal-naive RMSE degrades 56%
(GP) and 179% (Robi) when the daily lag is 24 samples instead of the measured 17 and 15.
Same mechanism, same direction. **Cite this alongside §4.1** — a reviewer is far more likely
to accept the point when it is corroborated on someone else's data.

Note also that their online (FLSP) variants beat their rolling counterparts throughout,
which is consistent with our §10 finding that online calibration beats static.

### Domain framing

| Paper | Venue / year | Relevance |
|---|---|---|
| Wang et al., *A Survey on Deep Learning for Cellular Traffic Prediction* | Intelligent Computing 3:0054, 2024 | Surveys the "rank models by RMSE" framing we depart from; collects state-of-the-art performance per dataset |
| Vesselinova, Harjula & Ilmonen, *Data Matters: The Case of Predicting Mobile Cellular Traffic* (arXiv:2411.02418) | IEEE VNC 2025 | Argues data quality and preprocessing dominate architecture — direct support for the audit being the contribution |
| Tuna & Soysal (arXiv:2304.11156, arXiv:2309.03898) | 2023 | SLA-violation-constrained prediction; already cited in §2 |
| Sulaiman et al., *MicroOpt* (arXiv:2407.18342) | 2024 | Slice resource optimisation against a learned SLA model; already cited in §2 |

---

## What we cannot compare, and should say so

- **No published paper uses our traces.** Grameenphone Dhaka and Robi Dhaka-3 are not public
  datasets. There is no external number for *our* data, and there never will be. Every
  comparison here is across datasets and therefore indicative only.
- **Nobody else reports our headline metric.** Our central results are provisioning *cost*
  at a given cost ratio and *coverage* against a nominal service level. DeepCog reports cost
  but on a normalised scale with a different cost model; no one reports conformal coverage
  on cellular traffic that we have found. On the contributions that matter most, we have no
  benchmark — which is a statement about novelty, not a gap to apologise for.
- **Sample size is not comparable.** Milan gives 4,320 samples over a month across 900–10,000
  cells. We have 905 and 888 samples from two sites. Our error bars are wider for a reason,
  and a model that wins here on 8 folds of ~66 test points would need far more data to be
  called better in general.

---

## How to use this file when models change

1. Re-run `run_benchmark.py` and `run_sequence.py`; both write to `experiments/results/`.
2. Update the **Our current results** tables — RMSE, and NRMSE = RMSE ÷ mean demand
   (92.88 Gbps GP, 145.54 Gbps Robi), plus MAPE, MASE and the persistence comparison.
3. Compare on **NRMSE / MAPE / improvement-over-naive**, never on raw RMSE.
4. If a new published comparison is found, add it with its dataset, units, granularity and
   horizon. A number without its units is not evidence.
5. Add the source to [`PROVENANCE.md`](PROVENANCE.md) in the same commit.

---

*Last updated 2026-09-18. DeepCog figures read from the INFOCOM 2019 PDF (IMDEA open
archive); AHSTGNN and Mehri et al. figures extracted from the arXiv PDFs. The AHSTGNN row
alignment needs checking against the published table. Our own figures are regenerated from
`experiments/results/` and are current as of commit `a483afe`.*
