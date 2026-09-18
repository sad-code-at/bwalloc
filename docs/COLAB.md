# Running these notebooks on Google Colab

The notebooks are written to run unchanged in Colab and locally. The one complication
is that **this repository is private**, so Colab cannot clone it without a credential.
That is what most of this page is about; once it is set up, opening a notebook is two
clicks.

## The short version

1. Make a GitHub token with read access to this repository.
2. Put it in Colab **Secrets** under the name `GH_TOKEN`.
3. Open a notebook from the GitHub tab in Colab and run it.

The first cell of every notebook finds the repository, clones it if it is not there,
and puts `src/` on the path. Nothing else is needed.

---

## 1. Make a token

On GitHub: **Settings → Developer settings → Personal access tokens → Fine-grained
tokens → Generate new token**.

| Field | Value |
|---|---|
| Repository access | *Only select repositories* → `sad-code-at/bwalloc` |
| Permissions | **Contents: Read-only** |
| Expiration | 90 days is plenty for a semester |

Read-only on one repository is the whole point: if the token leaks, it exposes nothing
else and can write nothing. Copy it when shown — GitHub will not display it again.

## 2. Put it in Colab Secrets — not in a cell

In Colab, click the **key icon** in the left sidebar, **Add new secret**:

- Name: `GH_TOKEN`
- Value: the token
- Toggle **Notebook access** on for the notebook you are running

**Do not paste the token into a code cell.** A token in a cell gets saved into the
`.ipynb`, and if that notebook is ever committed or shared the token goes with it.
Secrets are stored against your Google account and never appear in the file.

## 3. Open a notebook

**File → Open notebook → GitHub tab.** Sign in to GitHub when prompted and tick
*Include private repos*. Pick `sad-code-at/bwalloc` and choose a notebook.

If the GitHub tab will not show private repositories, the fallback is to download the
`.ipynb` from GitHub and use **File → Upload notebook**. The bootstrap cell still
clones the rest of the project, so the notebook works either way.

## 4. Run it

Run the first cell. It will:

- look for the repository by walking up from the working directory (this is what
  happens locally, and it finds nothing on a fresh Colab VM);
- read `GH_TOKEN` from Colab Secrets and clone into `/content/bwalloc`;
- `chdir` there and add `src/` to `sys.path`;
- print the `bwalloc` version and the results directory.

If the token is missing or wrong you get a plain message saying so rather than an
import error forty lines later.

---

## What Colab already has, and what it does not

Colab ships with `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `statsmodels`,
`xgboost`, `lightgbm` and `torch`, which covers notebooks 00–05, 07 and 08 with no
installation at all.

Only **notebook 06 (zero-shot foundation models)** needs something extra:

```python
!pip install -q chronos-forecasting
```

It downloads Chronos-Bolt Small (~48M parameters) on first use. It runs on CPU in about
seven minutes — a GPU runtime is not required and will not help much at this series
length.

Nothing here needs a GPU. The heaviest work is scikit-learn's random forest, which is
CPU-bound, and the sequence models in notebook 08 are small enough that a CPU runtime
finishes them in a couple of minutes.

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

## If you would rather not use a token

Two alternatives, both worse but both workable.

**Mount Google Drive.** Upload the repository folder to Drive once, then:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/bwalloc
```

The bootstrap then finds the repository by walking up and never tries to clone. The
cost is that Drive is slow for many small files, and you must re-upload by hand to get
changes.

**Make the repository public.** Then a plain `git clone` works with no credential.
Before doing that, note that the repository contains the two operator traces, which
were inherited rather than collected, and an unpublished paper draft — see the
visibility discussion in `STATUS.md`.
