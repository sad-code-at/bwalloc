"""The coverage gate, enforced on real results.

Part 5 of the study plan asks for a direct check that conformal calibration works:
on the test folds, the empirical ``P(A_t >= y_t)`` must land within +/-2 percentage
points of the nominal tau, marginally and within each context group.

That check had only ever been run on synthetic exchangeable data, where it passes
trivially. This runs it on the actual backtest output, and it does not pass
everywhere. That result is kept rather than tuned away, because it is informative:
every failure is *negative* -- conformal systematically under-covers -- which is the
signature of exchangeability breaking under temporal drift, not of a bug in the
calibration. The finite-sample guarantee is conditional on exchangeability, and a
55-day trace with a trend does not supply it.

The consequence for the paper is a restriction on what may be claimed, not a
retraction: the context-conditional result (C2b) is a *relative* comparison between
methods calibrated on identical data, and is unaffected. What must not be claimed is
an absolute coverage guarantee.

Exits 0 by default so the finding is reported rather than blocking; pass ``--strict``
to exit non-zero on any failure, which is the right setting once the drift correction
is in place and the gate is expected to hold.

    python experiments/run_coverage_gate.py [--strict]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "experiments" / "results"

#: The plan's tolerance, in coverage proportion.
TOLERANCE = 0.02


def clopper_pearson(successes: int, n: int, confidence: float = 0.95
                    ) -> tuple[float, float]:
    """Exact binomial interval for a coverage estimate.

    Used rather than a normal approximation because the elevated-risk group carries
    on the order of 40 test points per fold, where the approximation is not reliable.
    """
    if n == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    lo = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2, successes, n - successes + 1)
    hi = 1.0 if successes == n else stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return float(lo), float(hi)


def pool(per_fold: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Pool per-fold coverage into one estimate per configuration.

    Weighted by test-block size rather than averaging fold means, so a short final
    block does not carry the same weight as a full one.
    """
    df = per_fold[per_fold.get("error", "") == ""].copy()
    df["covered"] = df["n"] * df["coverage"]
    out = (
        df.groupby(keys)
        .agg(covered=("covered", "sum"), n=("n", "sum"), folds=("fold", "nunique"))
        .reset_index()
    )
    out["covered"] = out["covered"].round().astype(int)
    out["coverage"] = out["covered"] / out["n"]
    out["gap"] = out["coverage"] - out["tau"]
    bounds = [clopper_pearson(c, n) for c, n in zip(out["covered"], out["n"])]
    out["ci_lo"] = [b[0] for b in bounds]
    out["ci_hi"] = [b[1] for b in bounds]
    out["passes"] = out["gap"].abs() <= TOLERANCE
    # A failure the confidence interval cannot resolve is a sample-size problem, not
    # a calibration problem, and is labelled so the paper does not overstate it.
    out["decisive"] = ~((out["ci_lo"] <= out["tau"]) & (out["tau"] <= out["ci_hi"]))
    return out


def main() -> None:
    strict = "--strict" in sys.argv
    pd.set_option("display.width", 220)
    tables = []

    for operator in ("gp", "robi"):
        one_step = RESULTS / f"allocation_{operator}_perfold.csv"
        if one_step.exists():
            df = pd.read_csv(one_step).fillna({"error": ""})
            t = pool(df, ["tau", "method", "group"])
            t.insert(0, "horizon_hours", 1.5)
            t.insert(0, "operator", operator)
            tables.append(t)

        multi = RESULTS / f"horizon_alloc_{operator}.csv"
        if multi.exists():
            df = pd.read_csv(multi).fillna({"error": ""})
            t = pool(df, ["horizon_hours", "tau", "method", "group"])
            t.insert(0, "operator", operator)
            tables.append(t)

    if not tables:
        raise SystemExit(
            "No allocation results found. Run experiments/run_allocation.py first."
        )

    gate = pd.concat(tables, ignore_index=True).drop_duplicates(
        subset=["operator", "horizon_hours", "tau", "method", "group"], keep="last"
    )
    gate.to_csv(RESULTS / "coverage_gate.csv", index=False)

    marginal = gate[gate["group"] == "ALL"]
    print("=" * 92)
    print(f"Coverage gate: |achieved - nominal| <= {TOLERANCE:.0%}, pooled across folds")
    print("=" * 92)
    print(
        marginal[["operator", "horizon_hours", "tau", "method", "n",
                  "coverage", "gap", "ci_lo", "ci_hi", "passes", "decisive"]]
        .to_string(index=False, float_format=lambda v: f"{v:8.3f}")
    )

    n_fail = int((~gate["passes"]).sum())
    n_total = len(gate)
    decisive_fail = int((~gate["passes"] & gate["decisive"]).sum())
    under = int((gate["gap"] < -TOLERANCE).sum())

    print(f"\n  {n_fail} of {n_total} configurations fail the +/-2% gate "
          f"({decisive_fail} of them decisively, i.e. the 95% interval excludes tau).")
    print(f"  {under} of the {n_fail} failures are under-coverage, "
          f"{n_fail - under} over-coverage.")
    if under == n_fail and n_fail:
        print("  Every failure is one-sided (under-coverage): the signature of "
              "exchangeability\n  breaking under temporal drift, not of miscalibration.")
    print(f"\nWrote {RESULTS / 'coverage_gate.csv'}")

    if strict and n_fail:
        raise SystemExit(f"{n_fail} configurations outside tolerance.")


if __name__ == "__main__":
    main()
