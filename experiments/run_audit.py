"""Data audit: sampling geometry, seasonality, leakage, and context-flag validation.

Establishes the facts the rest of the study depends on, and quantifies the three
methodological errors in the original codebase.

    python experiments/run_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bwalloc as bw  # noqa: E402
from bwalloc.context import assign_binary_groups, flag_report, group_sizes  # noqa: E402
from bwalloc.data import CONTEXT_FLAGS, autocorrelation_by_lag, load, sampling_profile  # noqa: E402
from bwalloc.metrics import rmse  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def main() -> None:
    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    sampling_rows = []
    for operator in ("gp", "robi"):
        df = load(operator)
        profile = sampling_profile(df)
        y = df["Gbps"]

        print("=" * 78)
        print(f"{operator.upper()}  —  {profile.describe()}")
        print("=" * 78)

        sampling_rows.append(
            {
                "operator": operator,
                "n": profile.n_samples,
                "span_days": profile.span_days,
                "median_gap_min": profile.median_gap_min,
                "samples_per_day": profile.samples_per_day,
                "daily_period": profile.daily_period,
                "hours_spanned_by_lag24": profile.hours_for_lag(24),
            }
        )

        # -- 1. Seasonality: the true daily period versus the lag 24 the original used.
        acf = autocorrelation_by_lag(y, max_lag=60, profile=profile)
        acf.insert(0, "operator", operator)
        acf.to_csv(RESULTS / f"audit_acf_{operator}.csv", index=False)

        true_lag = profile.daily_period
        r_true = float(acf.loc[acf["lag_samples"] == true_lag, "autocorr"].iloc[0])
        r_24 = float(acf.loc[acf["lag_samples"] == 24, "autocorr"].iloc[0])
        sn_true = rmse(y.iloc[true_lag:], y.shift(true_lag).dropna())
        sn_24 = rmse(y.iloc[24:], y.shift(24).dropna())

        print(f"\n  Seasonality")
        print(f"    lag {true_lag:>2} = {profile.hours_for_lag(true_lag):5.1f} h   "
              f"r = {r_true:+.3f}   seasonal-naive RMSE = {sn_true:6.2f}   <- true period")
        print(f"    lag 24 = {profile.hours_for_lag(24):5.1f} h   "
              f"r = {r_24:+.3f}   seasonal-naive RMSE = {sn_24:6.2f}   <- original code")
        print(f"    cost of the error: {100 * (sn_24 - sn_true) / sn_true:+.0f}% RMSE")

        # -- 2. Context flags: level effect versus variance effect.
        report = flag_report(df, operator)
        report.insert(0, "operator", operator)
        report.to_csv(RESULTS / f"audit_flags_{operator}.csv", index=False)
        print(f"\n  Context flags (sorted by effect on error variance)")
        print(
            report[["flag", "n_on", "delta_mean", "p_level", "variance_ratio", "p_variance"]]
            .to_string(index=False, float_format=lambda v: f"{v:9.3f}")
        )
        inert = report[report["p_level"] > 0.05]["flag"].tolist()
        if inert:
            print(f"    no significant level effect: {', '.join(inert)}")

        # -- 3. Context groups and what coverage level each can support.
        groups = assign_binary_groups(df)
        sizes = group_sizes(groups)
        sizes.insert(0, "operator", operator)
        sizes.to_csv(RESULTS / f"audit_groups_{operator}.csv", index=False)
        print(f"\n  Context groups (two-group split)")
        print(sizes.to_string(index=False))
        print()

    pd.DataFrame(sampling_rows).to_csv(RESULTS / "audit_sampling.csv", index=False)
    print(f"Wrote audit tables to {RESULTS}")


if __name__ == "__main__":
    main()
