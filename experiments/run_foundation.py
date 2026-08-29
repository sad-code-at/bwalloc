"""Zero-shot time-series foundation models against per-operator training.

The question this answers is the one the dataset's main weakness makes unavoidable:

    **With only ~900 samples per site, is bespoke per-operator training worth it, or
    does a foundation model that has never seen this network do just as well?**

Chronos-Bolt is evaluated strictly zero-shot -- no fitting, no fine-tuning, not even a
scaling constant -- on exactly the fold schedule, lead times and test blocks that
``run_horizon.py`` uses, so the numbers drop straight into the same table as the
trained models.

Two things make this comparison interesting rather than routine:

**It is quantile-native.** Chronos-Bolt emits quantiles directly, so it plugs into the
allocation layer of :mod:`bwalloc.allocation` with no quantile-regression step. We
therefore report not just its accuracy but whether its *uncalibrated* quantiles deliver
their nominal service level -- and whether split-conformal calibration repairs them, as
it must for the trained models.

**It assumes regular sampling, which these traces violate.** A foundation model
consumes a sequence of values with no timestamps; it cannot know that a step is 86
minutes on one trace and 99 on the other, nor that "24 steps" is not a day. Everything
§3 of the paper corrects by measuring the sampling rate, this model class simply cannot
see. Whether that costs it anything is an empirical question worth reporting either
way, and it is the sharpest reason to include this experiment in *this* paper rather
than treating it as an unrelated baseline.

    pip install chronos-forecasting
    python experiments/run_foundation.py [--models chronos-bolt-small,chronos-bolt-base]

Runs on CPU. Chronos-Bolt Small is ~48M parameters against 900-point series, so a full
sweep is minutes, not hours; no GPU is required.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bwalloc as bw  # noqa: E402
from bwalloc.conformal import SplitConformal  # noqa: E402
from bwalloc.data import TARGET, load, sampling_profile  # noqa: E402
from bwalloc.features import FeatureConfig, build_features  # noqa: E402
from bwalloc.forecast import direct_design, horizon_steps  # noqa: E402
from bwalloc.metrics import coverage, mean_allocation, rmse  # noqa: E402
from bwalloc.splits import rolling_origin  # noqa: E402

RESULTS = ROOT / "experiments" / "results"

#: Same lead times as run_horizon.py, so the tables merge.
HORIZON_HOURS = (1.5, 3.0, 6.0, 12.0, 24.0)
N_FOLDS = 8
TAUS = (0.80, 0.90, 0.95)

#: Quantile levels requested from the model. Includes TAUS plus the median, which is
#: used as the point forecast.
QUANTILE_LEVELS = (0.1, 0.5, 0.8, 0.9, 0.95)

#: The highest quantile Chronos-Bolt was trained to emit. Anything above this is
#: silently clipped to it, so a tau=0.95 request returns the tau=0.90 prediction.
#:
#: This is a real limitation rather than a configuration detail, and it is one of the
#: findings of this experiment: an operator whose cost ratio implies a 95% or 99%
#: service level cannot obtain that level from the model's own quantile head at all.
#: Since tau* = kappa/(1+kappa), the ceiling of 0.9 corresponds to a cost asymmetry of
#: only kappa = 9 -- below what under-provisioning typically costs. Calibration on top
#: of the model is therefore not an optional refinement here; it is what makes the
#: model usable for provisioning.
NATIVE_QUANTILE_CEILING = 0.9

DEFAULT_MODELS = ("amazon/chronos-bolt-small",)

#: Longest context handed to the model. The whole trace is ~900 points and Chronos-Bolt
#: accepts far more, so in practice every forecast sees all available history.
MAX_CONTEXT = 2048


def load_pipeline(model_id: str):
    """Load a Chronos pipeline on CPU, tolerating both package layouts."""
    import torch

    try:
        from chronos import BaseChronosPipeline as Pipeline
    except ImportError:  # older releases exposed only the T5 pipeline
        from chronos import ChronosPipeline as Pipeline  # type: ignore

    return Pipeline.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float32
    )


def predict_block(pipeline, contexts: list[np.ndarray], steps: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Forecast ``steps`` ahead from each context; return the final step only.

    Returns
    -------
    (quantiles, median)
        ``quantiles`` has shape (n_contexts, len(QUANTILE_LEVELS)).
    """
    import torch

    batch = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    kwargs = dict(prediction_length=steps, quantile_levels=list(QUANTILE_LEVELS))
    try:
        # chronos-forecasting >= 2.x names the series argument `inputs`.
        q, _mean = pipeline.predict_quantiles(inputs=batch, **kwargs)
    except TypeError:
        # 1.x called it `context`. Colab may install either.
        q, _mean = pipeline.predict_quantiles(context=batch, **kwargs)
    # q: (batch, prediction_length, n_quantiles). Only the final step is the
    # steps-ahead forecast we asked for; the intermediate ones are discarded.
    final = q[:, -1, :].cpu().numpy()
    median = final[:, QUANTILE_LEVELS.index(0.5)]
    return final, median


def evaluate(pipeline, model_name: str, operator: str, df, profile,
             X, y, hours: float) -> tuple[list[dict], list[dict]]:
    """Zero-shot accuracy and quantile calibration at one lead time."""
    steps = horizon_steps(profile, hours)
    lead_h = profile.hours_for_lag(steps)
    lead = steps - 1

    X_h, y_h, _ = direct_design(X, y, steps)
    folds = rolling_origin(
        len(y_h), n_folds=N_FOLDS, calib_frac=0.30, embargo=lead
    )

    # Map the feature frame onto the undropped series, so each context can include the
    # warm-up rows that build_features discarded -- the model has no lag features to
    # construct, so there is no reason to withhold that history from it.
    y_full = df[TARGET].astype(float)
    full_index = pd.DatetimeIndex(y_full.index)
    origin_full = full_index.get_indexer(pd.DatetimeIndex(X.index))
    values = y_full.to_numpy(dtype=float)

    def contexts_for(positions: np.ndarray) -> list[np.ndarray]:
        """History strictly before each forecast origin."""
        out = []
        for p in positions:
            # Row p of the re-aligned frame has its origin at row p of X, whose last
            # observed value sits one position earlier in the raw series.
            cut = origin_full[p]
            out.append(values[max(0, cut - MAX_CONTEXT):cut])
        return out

    acc_rows, cal_rows = [], []
    for fold in folds:
        y_te = y_h.iloc[fold.test].to_numpy(dtype=float)
        q_te, med_te = predict_block(pipeline, contexts_for(fold.test), steps)

        acc_rows.append({
            "operator": operator, "model": model_name,
            "horizon_hours": hours, "steps": steps, "lead_hours": round(lead_h, 2),
            "fold": fold.number, "n": len(y_te),
            "rmse": rmse(y_te, med_te),
            "mae": float(np.mean(np.abs(y_te - med_te))),
        })

        # -- Do its own quantiles deliver their nominal level? -------------------
        y_ca = y_h.iloc[fold.calib].to_numpy(dtype=float)
        q_ca, med_ca = predict_block(pipeline, contexts_for(fold.calib), steps)

        for tau in TAUS:
            j = QUANTILE_LEVELS.index(tau)
            raw = q_te[:, j]
            # And does conformal calibration on held-out residuals repair them?
            calibrated = (
                SplitConformal()
                .calibrate(y_ca, q_ca[:, j])
                .allocate(raw, tau)
            )
            for method, alloc in (("zero_shot", raw), ("conformal", calibrated)):
                cal_rows.append({
                    "operator": operator, "model": model_name,
                    "horizon_hours": hours, "lead_hours": round(lead_h, 2),
                    "fold": fold.number, "tau": tau, "method": method,
                    "n": len(y_te),
                    "coverage": coverage(y_te, alloc),
                    "mean_allocation_ratio": mean_allocation(y_te, alloc),
                    # False where the model's quantile head cannot express this level
                    # and has silently returned its ceiling instead.
                    "natively_expressible": tau <= NATIVE_QUANTILE_CEILING,
                })

    return acc_rows, cal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help="comma-separated Hugging Face model ids",
    )
    args = parser.parse_args()
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]

    bw.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    acc_all, cal_all = [], []

    for model_id in model_ids:
        name = model_id.split("/")[-1]
        print(f"loading {model_id} ...", flush=True)
        t0 = time.time()
        pipeline = load_pipeline(model_id)
        print(f"  loaded in {time.time() - t0:.0f}s (CPU)\n", flush=True)

        for operator in ("gp", "robi"):
            df = load(operator)
            profile = sampling_profile(df)
            X, y = build_features(df, profile, FeatureConfig())

            print("=" * 78)
            print(f"{operator.upper()}  —  {name}, zero-shot")
            print("=" * 78, flush=True)

            for hours in HORIZON_HOURS:
                t1 = time.time()
                acc, cal = evaluate(
                    pipeline, name, operator, df, profile, X, y, hours
                )
                acc_all.extend(acc)
                cal_all.extend(cal)
                mean_rmse = float(np.mean([r["rmse"] for r in acc]))
                print(f"  h={hours:>4g}h  RMSE {mean_rmse:7.3f}"
                      f"   [{time.time() - t1:.0f}s]", flush=True)
            print()

    accuracy = pd.DataFrame(acc_all)
    calibration = pd.DataFrame(cal_all)
    accuracy.to_csv(RESULTS / "foundation_accuracy.csv", index=False)
    calibration.to_csv(RESULTS / "foundation_calibration.csv", index=False)

    # -- The comparison table: zero-shot against models trained on this operator ---
    summary = (
        accuracy.groupby(["operator", "model", "lead_hours"])
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
        .reset_index()
    )

    print("=" * 78)
    print("Zero-shot versus trained, RMSE by lead time")
    print("=" * 78)
    for operator in ("gp", "robi"):
        trained_path = RESULTS / f"horizon_{operator}.csv"
        if not trained_path.exists():
            print(f"  {operator}: run experiments/run_horizon.py first for the "
                  "trained-model comparison")
            continue
        trained = pd.read_csv(trained_path)
        table = trained.pivot_table(
            index="model", columns="lead_hours", values="rmse_mean"
        )
        zero = summary[summary["operator"] == operator].pivot_table(
            index="model", columns="lead_hours", values="rmse_mean"
        )
        combined = pd.concat([table, zero]).round(2)
        print(f"\n{operator.upper()}:")
        print(combined.to_string())

        if not zero.empty:
            best_trained = table.drop(
                index=[i for i in table.index if "naive" in i or "persistence" in i],
                errors="ignore",
            ).min()
            for model in zero.index:
                gap = (zero.loc[model] / best_trained - 1)
                print(f"\n  {model} vs the best per-operator trained model:")
                print("   " + "  ".join(
                    f"{h:g}h {g:+.1%}" for h, g in gap.items()
                ))

    print("\n" + "=" * 78)
    print("Do its own quantiles deliver their nominal level?")
    print(f"(the model's quantile head stops at tau={NATIVE_QUANTILE_CEILING}; "
          "higher levels are clipped to it)")
    print("=" * 78)
    cal = calibration.copy()
    cal["c"] = cal["n"] * cal["coverage"]
    pooled = (
        cal.groupby(["operator", "model", "tau", "method"])
        .agg(c=("c", "sum"), n=("n", "sum"))
        .assign(achieved=lambda d: d["c"] / d["n"])
        .reset_index()
    )
    pooled["gap"] = pooled["achieved"] - pooled["tau"]
    print(
        pooled.pivot_table(
            index=["operator", "model", "tau"], columns="method",
            values=["achieved", "gap"],
        ).round(3).to_string()
    )

    print(f"\nWrote foundation tables to {RESULTS}")


if __name__ == "__main__":
    main()
