# Running these notebooks on Kaggle

Kaggle works the same way Colab does, with two differences worth knowing before you
start: **Internet is off by default** and must be switched on, and secrets live under
**Add-ons → Secrets** rather than a sidebar icon.

You do **not** need to make this repository public. Kaggle can clone a private repo with
a token, and there is a second route (upload it as a private Kaggle Dataset) that needs
no token and no internet at all.

## Route A — clone with a token (recommended)

### 1. Make a token

GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens**:

| Field | Value |
|---|---|
| Repository access | *Only select repositories* → `sad-code-at/bwalloc` |
| Permissions | **Contents: Read-only** |
| Expiration | 90 days |

Read-only on one repository means a leak exposes nothing else and can write nothing.

### 2. Add it to Kaggle Secrets

In a Kaggle notebook: **Add-ons → Secrets → Add a new secret**.

- Label: `GH_TOKEN`
- Value: the token
- Attach it to the notebook (the toggle beside it)

**Do not paste the token into a cell.** Kaggle notebooks are versioned and can be made
public later; a token in a cell travels with the file.

### 3. Switch Internet on

Right-hand sidebar → **Settings → Internet → On**. Kaggle requires a phone-verified
account for this. Without it the clone cannot reach GitHub, and the bootstrap will say
so.

### 4. Run

Open a notebook and run the first cell. It detects Kaggle, reads `GH_TOKEN` from
`kaggle_secrets`, clones into `/kaggle/working/bwalloc`, changes into it and puts `src/`
on the path.

To get the notebooks themselves onto Kaggle: **File → Import Notebook → GitHub** (sign
in, tick private repos), or download the `.ipynb` from GitHub and use **File → Import
Notebook → File**. Either works; the bootstrap fetches the rest of the project.

## Route B — upload as a private Kaggle Dataset (no token, no internet)

Useful if you cannot phone-verify, or if you want the notebooks to run with Internet
off, which is faster to start and works under Kaggle's offline competition rules.

1. Download the repository as a ZIP from GitHub (**Code → Download ZIP**).
2. Kaggle → **Datasets → New Dataset** → upload the ZIP → set visibility **Private**.
3. In your notebook: **Add Data → Your Datasets →** the one you just made.

It mounts read-only at `/kaggle/input/<dataset-name>/`. The bootstrap looks there before
attempting any clone, so it is picked up automatically with no further changes.

The trade-off: it is a snapshot. When the repository changes you must re-upload, which
is why Route A is the better default if you can use it.

## What Kaggle already has

Kaggle's Python image ships `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`,
`statsmodels`, `xgboost`, `lightgbm` and `torch`. That covers notebooks 00–05, 07 and 08
with nothing to install.

Only **notebook 06 (zero-shot foundation models)** needs an extra package, and this one
does require Internet on:

```python
!pip install -q chronos-forecasting
```

**No GPU is needed.** Leave the accelerator on *None*. The heaviest step is
scikit-learn's random forest, which is CPU-bound and gains nothing from a GPU, and the
sequence models in notebook 08 are small enough to finish in a couple of minutes on CPU.
A GPU session only burns your weekly quota.

## Runtimes

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

All nine together are well inside Kaggle's 12-hour session limit. `04_multi_horizon` is
the only one long enough to be worth starting and leaving.

Note that every notebook is committed **with its outputs**, so you can read all of them
on GitHub without running anything.

## Things that go wrong

**`Could not find the repository and no GH_TOKEN is available.`**
The secret is not attached to this notebook, or it is named something other than
`GH_TOKEN`. Check Add-ons → Secrets and the toggle beside the entry.

**`git clone failed.`**
Usually Internet is still off — check Settings in the right sidebar. Otherwise the token
is expired or lacks *Contents: Read* on this repository.

**Everything vanished when I came back.**
`/kaggle/working` persists only while the session lives. Re-running the first cell
re-clones. Nothing is lost, because all results and figures regenerate from the
repository.

**Edits I made are gone.**
Changes inside `/kaggle/working/bwalloc` disappear with the session. Develop locally and
use Kaggle to run, or commit and push from Kaggle using the same token.

**A cell fails with an error that should already be fixed.**
Kaggle keeps the kernel between runs. **Run → Restart & Clear Cell Outputs**, then run
all. This is the most common cause of "I fixed it but it still fails".

## Saving results off the session

Anything written to `/kaggle/working` is captured when you **Save Version**, and appears
under the notebook's Output tab afterwards. So to keep a figure or a table:

```python
import shutil
shutil.copytree(ROOT / "notebooks" / "figures", "/kaggle/working/figures",
                dirs_exist_ok=True)
```

Then Save Version, and download from the Output tab.

---

For Colab, see [`COLAB.md`](COLAB.md) — the setup is the same shape, with Secrets behind
the key icon instead.
