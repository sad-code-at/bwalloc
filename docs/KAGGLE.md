# Running these notebooks on Kaggle

The repository is public, so there is nothing to authenticate for reading. Getting a
notebook running is four steps, and re-running it after you push a change is one.

## Step by step

### 1. Create the notebook

Kaggle → **Create → New Notebook**. Or, to start from one of ours:
**File → Import Notebook → GitHub**, paste
`https://github.com/sad-code-at/bwalloc`, and pick a notebook from `notebooks/`.

### 2. Switch Internet on

Right-hand sidebar → **Settings → Internet → On**.

This is the step people miss. Without it the clone cannot reach GitHub and the first
cell stops with a message saying so. Kaggle requires a phone-verified account to enable
internet; if you cannot, see [No internet?](#no-internet) below.

### 3. Leave the accelerator on None

Right-hand sidebar → **Settings → Accelerator → None**.

Nothing here benefits from a GPU. The heaviest step is scikit-learn's random forest,
which is CPU-bound, and the sequence models in notebook 08 are small. A GPU session only
spends your weekly quota.

### 4. Run the first cell

If you imported one of our notebooks, just run it — the bootstrap does everything. If
you started a blank notebook, paste this:

```python
!git clone -q https://github.com/sad-code-at/bwalloc.git /kaggle/working/bwalloc
%cd /kaggle/working/bwalloc
import sys; sys.path.insert(0, "src")

import bwalloc as bw
bw.set_seed()
print(bw.__version__)
```

Then run whatever you like. Everything is importable from `bwalloc`, and the stored
results are under `experiments/results/`.

---

## The loop: change something, see it on Kaggle

This is the workflow you asked about.

**On your laptop** — edit, then push:

```bash
cd "D:/L4-T-1/EEE 402/project/bwalloc"
git add -A
git commit -m "what changed and why"
git push
```

**On Kaggle** — re-run the first cell. That is all.

The bootstrap checks whether the repository is already in `/kaggle/working/bwalloc`. If
it is, it runs `git pull --ff-only` instead of cloning again, so your pushed changes
arrive and it prints `pulled latest into /kaggle/working/bwalloc`. If the session has
been restarted and the directory is gone, it clones fresh. Either way the first cell is
the only thing you touch.

If you are not using one of our notebooks, the same thing by hand:

```python
!git -C /kaggle/working/bwalloc pull --ff-only
```

⚠️ **Restart the kernel after pulling if you changed anything under `src/`.** Python
caches imported modules, so a pulled change to `bwalloc/*.py` will not take effect in a
kernel that already imported it. **Run → Restart & Clear Cell Outputs**, then run all.
This is the single most common cause of "I pulled but it still does the old thing".

Or avoid the restart entirely by putting this at the top of your session:

```python
%load_ext autoreload
%autoreload 2
```

### Pushing *from* Kaggle

Reading is anonymous; writing is not. To push from a Kaggle notebook you need a token
with **Contents: Read and write** on the repository — note that is a wider permission
than reading needs, so prefer editing locally where you can.

1. GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained
   tokens**. Repository access: only `sad-code-at/bwalloc`. Permissions:
   **Contents: Read and write**.
2. Kaggle → **Add-ons → Secrets** → new secret named `GH_TOKEN`, attached to the
   notebook.
3. In a cell:

```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GH_TOKEN")

!git -C /kaggle/working/bwalloc config user.email "2106110@eee.buet.ac.bd"
!git -C /kaggle/working/bwalloc config user.name "your name"
!git -C /kaggle/working/bwalloc add -A
!git -C /kaggle/working/bwalloc commit -m "from kaggle"
!git -C /kaggle/working/bwalloc push https://{tok}@github.com/sad-code-at/bwalloc.git main
```

**Never type the token into a cell literally** — read it from Secrets as above. A token
written into an `.ipynb` is saved with the notebook, and Kaggle notebooks can be shared
or made public later.

---

## What Kaggle already has

Kaggle's Python image ships `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`,
`statsmodels`, `xgboost`, `lightgbm` and `torch`. That covers notebooks 00–05, 07 and 08
with nothing to install.

Only **notebook 06 (zero-shot foundation models)** needs an extra package:

```python
!pip install -q chronos-forecasting
```

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

All nine together sit well inside Kaggle's 12-hour session limit. `04_multi_horizon` is
the only one long enough to be worth starting and leaving.

Every notebook is committed **with its outputs**, so you can read all of them on GitHub
without running anything.

## Keeping results after the session ends

`/kaggle/working` is captured when you **Save Version**, and appears under the
notebook's Output tab. So to keep figures:

```python
import shutil
shutil.copytree(ROOT / "notebooks" / "figures", "/kaggle/working/figures",
                dirs_exist_ok=True)
```

Then Save Version and download from Output. Anything *not* copied into
`/kaggle/working` disappears with the session — including edits made to the cloned
repository, which is why the loop above goes laptop → GitHub → Kaggle rather than the
other way.

<a name="no-internet"></a>
## No internet?

If you cannot phone-verify, upload the repository as a Kaggle Dataset instead:

1. GitHub → **Code → Download ZIP**.
2. Kaggle → **Datasets → New Dataset** → upload the ZIP.
3. In the notebook: **Add Data →** your dataset.

It mounts read-only under `/kaggle/input/…`, and the bootstrap looks there before trying
to clone, so it is found automatically. The trade-off is that it is a snapshot: to pick
up changes you must re-upload, which is exactly what the pull workflow avoids.

## Things that go wrong

**`git clone failed.`** Internet is off. Right sidebar → Settings → Internet → On.

**I pulled but the old behaviour is still there.** The kernel cached the modules.
Restart it, or use `%autoreload 2`.

**`git pull` refuses with a conflict.** You edited files inside `/kaggle/working/bwalloc`
and those edits clash with what you pushed. Since Kaggle edits are disposable, the fix
is usually `!git -C /kaggle/working/bwalloc reset --hard origin/main`, which **discards
local changes** — check you do not want them first.

**Everything vanished.** `/kaggle/working` does not survive a session ending. Re-running
the first cell clones again; nothing is lost, because results and figures regenerate
from the repository.

---

For Colab, see [`COLAB.md`](COLAB.md) — same shape, `/content/bwalloc` instead of
`/kaggle/working/bwalloc`.
