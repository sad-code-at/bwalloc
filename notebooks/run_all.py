"""Execute every notebook in one pass, and write the results somewhere you can read.

Why this exists: a Kaggle or Colab notebook is a single document, not a workspace.
Importing this repository does not give you nine notebooks -- the GitHub import copies
*one* `.ipynb` onto the platform's servers, and its first cell clones the project to get
the code and data. So running all nine there means importing all nine, one at a time.

This script is the alternative. From a single hosted notebook it executes every
notebook from the clone and writes executed copies (plus rendered HTML) to an output
directory, which on Kaggle is captured by Save Version and downloadable from the Output
tab afterwards.

    # in one Kaggle cell, after the bootstrap has cloned the repo
    !python notebooks/run_all.py --out /kaggle/working/executed

    # locally, to refresh the committed outputs in place
    python notebooks/run_all.py --inplace

`06_foundation_models` is skipped unless `chronos-forecasting` is installed, because it
is the only notebook with a dependency outside the default Kaggle and Colab images.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: Roughly how long each takes on a CPU runtime, so the log is readable as progress
#: rather than as a stall. Measured locally; Kaggle is broadly comparable.
EXPECTED_MIN = {
    "00_data_audit": 0.2, "01_corrected_benchmark": 4, "02_allocation": 10,
    "03_context_conditional": 5, "04_multi_horizon": 20, "05_transfer": 2,
    "06_foundation_models": 7, "07_paper_figures": 0.2, "08_sequence_models": 4,
}


def _has_chronos() -> bool:
    try:
        import chronos  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for executed copies (default: alongside, --inplace)")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite the notebooks in notebooks/ with executed versions")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these (substring match, e.g. --only 08 02)")
    ap.add_argument("--html", action="store_true",
                    help="also write rendered .html next to each executed notebook")
    ap.add_argument("--timeout", type=int, default=2400,
                    help="per-notebook timeout in seconds (default 2400)")
    args = ap.parse_args()

    if not args.inplace and args.out is None:
        ap.error("pass --inplace, or --out DIR")

    books = sorted(HERE.glob("0*.ipynb"))
    if args.only:
        books = [b for b in books if any(k in b.name for k in args.only)]
    if not books:
        print("no notebooks matched", file=sys.stderr)
        return 1

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    skip_chronos = not _has_chronos()
    failures: list[str] = []
    started_all = time.time()

    for book in books:
        stem = book.stem
        if stem.startswith("06_") and skip_chronos:
            print(f"SKIP {stem}  (chronos-forecasting not installed; "
                  f"pip install chronos-forecasting to include it)")
            continue

        expect = EXPECTED_MIN.get(stem)
        note = f"  (~{expect:g} min)" if expect and expect >= 1 else ""
        print(f"RUN  {stem}{note}", flush=True)

        cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
               f"--ExecutePreprocessor.timeout={args.timeout}", str(book)]
        cmd += ["--inplace"] if args.inplace else ["--output-dir", str(args.out)]

        started = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        elapsed = (time.time() - started) / 60

        if proc.returncode != 0:
            failures.append(stem)
            print(f"FAIL {stem}  after {elapsed:.1f} min")
            # The useful line is the exception, not the nbconvert stack above it.
            tail = [ln for ln in proc.stderr.splitlines()
                    if ln.strip() and ("Error" in ln or "Exception" in ln)]
            for ln in tail[-3:]:
                print("       " + ln.strip())
        else:
            print(f"OK   {stem}  in {elapsed:.1f} min")

        if args.html and proc.returncode == 0:
            target = book if args.inplace else args.out / book.name
            subprocess.run(
                [sys.executable, "-m", "nbconvert", "--to", "html", str(target),
                 "--output-dir", str(args.out or HERE)],
                cwd=ROOT, capture_output=True, text=True,
            )

    total = (time.time() - started_all) / 60
    print(f"\n{len(books) - len(failures)}/{len(books)} succeeded in {total:.1f} min")

    if args.out:
        # Figures are written by the notebooks themselves; carry them along so a
        # single Save Version captures everything worth looking at.
        figs = ROOT / "notebooks" / "figures"
        if figs.exists():
            shutil.copytree(figs, args.out / "figures", dirs_exist_ok=True)
            print(f"figures copied to {args.out / 'figures'}")
        print(f"executed notebooks in {args.out}")

    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
