"""Backtest harness: one fold schedule, every model, identical treatment.

This is the piece that makes the results table trustworthy. Models and baselines are
scored on the *same* folds with the *same* metrics, so a difference between two rows
is attributable to the model rather than to a different split -- which was not true of
the original project, where a 70/30 index split and two different date splits were
compared side by side in one table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import Baseline
from .models import Forecaster
from .splits import Fold
from . import metrics as M


def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[Fold],
    models: list[Forecaster] | None = None,
    baselines: list[Baseline] | None = None,
    season_lag: int = 1,
    keep_predictions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model and baseline on every fold.

    Models are refit from scratch on each fold's training window (train + calibration
    slice, since a point forecaster has no use for a calibration set).

    Returns
    -------
    (per_fold, predictions)
        ``per_fold`` has one row per (model, fold) with accuracy metrics.
        ``predictions`` is long-format with columns
        ``[model, fold, timestamp, y_true, y_pred]`` -- retained so significance tests
        and allocation policies can be run afterwards without refitting.
    """
    models = models or []
    baselines = baselines or []
    yv = np.asarray(y, dtype=float)
    index = X.index

    rows: list[dict] = []
    preds: list[pd.DataFrame] = []

    def _record(name: str, fold: Fold, y_hat: np.ndarray) -> None:
        y_true = yv[fold.test]
        report = M.accuracy_report(
            y_true, y_hat, yv[: fold.test[0]], season_lag=season_lag
        )
        report.update(model=name, fold=fold.number, n_test=fold.n_test)
        rows.append(report)
        if keep_predictions:
            preds.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold.number,
                        "timestamp": index[fold.test],
                        "y_true": y_true,
                        "y_pred": y_hat,
                    }
                )
            )

    for fold in folds:
        # A point forecaster may use the calibration slice for fitting; only conformal
        # calibration needs it held out.
        fit_idx = np.concatenate([fold.train, fold.calib]) if fold.n_calib else fold.train
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        X_test = X.iloc[fold.test]

        for baseline in baselines:
            _record(baseline.name, fold, baseline.predict(fold))

        for model in models:
            model.fit(X_fit, y_fit)
            _record(model.name, fold, model.predict(X_test))

    per_fold = pd.DataFrame(rows)
    predictions = (
        pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    )
    return per_fold, predictions


def summarise(per_fold: pd.DataFrame, sort_by: str = "rmse_mean") -> pd.DataFrame:
    """Aggregate per-fold metrics to the reporting table, ranked."""
    agg = M.aggregate_folds(per_fold, ["model"])
    return agg.sort_values(sort_by).reset_index(drop=True)


def beats_baseline(summary: pd.DataFrame, baseline_name: str = "persistence",
                   metric: str = "rmse_mean") -> pd.DataFrame:
    """Annotate a summary table with each model's margin over a baseline.

    Adds ``vs_<baseline>`` (relative improvement, positive = better) and a boolean
    ``beats_<baseline>``. The benchmark notebook fails loudly if a model is presented
    as best while this column is False.
    """
    if baseline_name not in set(summary["model"]):
        raise KeyError(f"Baseline {baseline_name!r} not present in summary.")
    ref = float(summary.loc[summary["model"] == baseline_name, metric].iloc[0])
    out = summary.copy()
    out[f"vs_{baseline_name}"] = (ref - out[metric]) / ref
    out[f"beats_{baseline_name}"] = out[metric] < ref
    return out


def pairwise_dm(
    predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
    horizon: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test between two models, pooled across folds.

    Predictions are aligned on timestamp so both models are compared on exactly the
    same observations.
    """
    from .stats import diebold_mariano

    a = predictions[predictions["model"] == model_a].set_index("timestamp")
    b = predictions[predictions["model"] == model_b].set_index("timestamp")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError(f"No overlapping predictions for {model_a} and {model_b}.")
    return diebold_mariano(
        a.loc[common, "y_true"], a.loc[common, "y_pred"], b.loc[common, "y_pred"],
        horizon=horizon,
    )


def dm_matrix(predictions: pd.DataFrame, models: list[str] | None = None,
              horizon: int = 1, alpha: float = 0.05) -> pd.DataFrame:
    """All pairwise Diebold-Mariano tests with Benjamini-Hochberg FDR control.

    Ten models means 45 comparisons; at alpha=0.05 uncorrected, two or three spurious
    wins are expected by construction.
    """
    from .stats import benjamini_hochberg

    models = models or sorted(predictions["model"].unique())
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            try:
                stat, p = pairwise_dm(predictions, a, b, horizon=horizon)
            except ValueError:
                continue
            rows.append({"model_a": a, "model_b": b, "dm_stat": stat, "p_value": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["significant_fdr"] = benjamini_hochberg(out["p_value"].to_numpy(), alpha=alpha)
    out["winner"] = np.where(
        ~out["significant_fdr"], "tie",
        np.where(out["dm_stat"] < 0, out["model_a"], out["model_b"]),
    )
    return out
