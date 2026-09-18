# Running these notebooks on Google Colab

The repository is public, so there is nothing to authenticate. Open a notebook and run
the first cell.

## Step by step

1. **File → Open notebook → GitHub**, paste
   `https://github.com/sad-code-at/bwalloc`, pick a notebook from `notebooks/`.
2. Run the first cell. It clones into `/content/bwalloc`, changes into it, and puts
   `src/` on the path. On a re-run it pulls instead of cloning, so changes you have
   pushed since arrive without restarting the session.
3. That is all. Runtime → Run all.

From a blank notebook, the same thing by hand:

```python
!git clone -q https://github.com/sad-code-at/bwalloc.git
%cd bwalloc
import sys; sys.path.insert(0, "src")

import bwalloc as bw
bw.set_seed()
```

## One notebook at a time

Colab, like Kaggle, opens a **single notebook** — the GitHub browser copies one
`.ipynb` onto Colab's servers, so there is no way to open the whole repository at once.
The first cell then clones the project for the code and data.

This means `git pull` refreshes `src/`, the data and the results, but **not the notebook
in your tab**, which is Colab's own copy. Re-open it from the GitHub tab to pick up a
change to a notebook itself.

## The loop: change something, see it here

Push from your laptop, then **re-run the first cell** — it runs `git pull --ff-only`
when the repository is already present.

⚠️ **Restart the runtime after pulling if you changed anything under `src/`.** Python
caches imported modules, so a pulled change will not take effect in a kernel that
already imported it. *Runtime → Restart session*, then Run all. Or put
`%load_ext autoreload` and `%autoreload 2` at the top of your session and skip the
restart.

**Pushing from Colab** needs a token with *Contents: Read and write*, stored in Secrets
(the key icon) as `GH_TOKEN` and read with `google.colab.userdata.get("GH_TOKEN")` —
never typed into a cell, because it would be saved into the `.ipynb`. Reading needs no
token at all.

---

## What Colab already has, and what it does not

Colab ships with `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `statsmodels`,
`xgboost`, `lightgbm` and `torch`, which covers notebooks 00–05 and 07–12 with no
installation at all.

Only **notebook 06 (zero-shot foundation models)** needs something extra:

```python
!pip install -q chronos-forecasting
```

It downloads Chronos-Bolt Small (~48M parameters) on first use. It runs on CPU in about
seven minutes — a GPU runtime is not required and will not help much at this series
length.

Nothing here needs a GPU. The heaviest work is scikit-learn's random forest, which is
CPU-bound, and the sequence models in notebooks 08-10 are small enough that a CPU
runtime finishes them in a few minutes.

## Which notebook to open first

| Notebook | What it shows | Runtime |
|---|---|---|
| `00_data_audit` | The three measurement errors, checked against the CSVs | seconds |
| `01_corrected_benchmark` | Corrected benchmark, baselines, DM tests, ablation | ~4 min |
| `02_allocation` | Cost model, τ\* = κ/(1+κ), conformal, capacity frontier | ~10 min |
| `03_context_conditional` | Context acts on variance; the coverage failure and repair | ~5 min |
| `04_multi_horizon` | Lead time, direct vs recursive | ~20 min |
| `05_transfer` | Cold start: how much history a new site needs | ~2 min |
| `06_foundation_models` | Zero-shot Chronos-Bolt, and when it fails | ~7 min + install |
| `07_paper_figures` | Regenerates every paper figure from stored results | seconds |
| `08_sequence_models` | CNN / RNN / LSTM / GRU against the trees; lag depth | ~4 min |
| `09_full_feature_models` | Lifting the univariate constraint: covariates as channels | ~12 min |
| `10_modern_architectures` | TCN, Transformer, DLinear/NLinear, N-BEATS/NBEATSx | ~15 min |
| `11_hyperparameter_tuning` | Reads the search results; what tuning was worth | seconds |
| `12_final_comparison` | Every model, default and tuned, on one fold schedule | seconds |

`00` and `07` are the two to open if you only want to look at something. `07` reads
only from `experiments/results/`, so it draws every figure in the paper without
refitting a single model.

Note that the notebooks are committed **with their outputs**, so you can read all of
them on GitHub without running anything.

---

## Things that go wrong

**`Could not find the repository, and no GH_TOKEN is available.`**
The secret is not set, or *Notebook access* is off for this notebook. Both live behind
the key icon.

**`ModuleNotFoundError: No module named 'bwalloc'`**
The first cell did not complete. This is the most common error here and it is almost
never what it looks like — the import cell is fine, it is the cell above it that failed,
usually because Internet was off so the clone could not run. Scroll up, fix whatever the
first cell reported, run it again, then continue.

It also appears if you restarted the kernel and ran a later cell without re-running the
first one. The first cell now installs the package into the session, so `import bwalloc`
survives a restart — but `ROOT`, `RESULTS` and `FIGURES` are defined there too, so a
later cell will still fail on those. **After any restart, run the first cell.**

The first cell prints what it found, which is the quickest way to confirm it worked:

```
bwalloc 0.1.0 at /kaggle/working/bwalloc
  data/    2 csv  (gp_dhaka.csv, robi_dhaka3.csv)
  results/ 46 csv
```

**`git clone failed.`**
The token is expired, or it does not grant *Contents: Read* on this repository.
Fine-grained tokens are per-repository — a token made for a different repo will not
work here.

**A cell fails with `NameError` or an old traceback after you changed something.**
Colab keeps the kernel between runs. **Runtime → Restart session**, then run all cells.
The same applies locally, and it is the single most common cause of "I fixed it but it
still fails".

**Runtime disconnected, and the clone is gone.**
Colab wipes `/content` when a session ends. Re-running the first cell clones again;
nothing is lost, because results and figures are regenerated from the repository.

**You want your changes to survive.**
Edits made inside `/content/bwalloc` on Colab disappear with the session. Either commit
and push from Colab (`!git -C /content/bwalloc commit …`, using the same token), or do
development locally and use Colab only to run things.

---

---

For Kaggle, see [`KAGGLE.md`](KAGGLE.md) — same shape, but Internet must be switched on
in the notebook settings first.
