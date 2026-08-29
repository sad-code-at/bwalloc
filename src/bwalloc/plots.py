"""Figures for the paper.

One shared style so every figure in the writeup matches, and each function returns its
Axes so notebooks can annotate further. Colours are chosen to stay distinguishable in
greyscale print and for the most common colour-vision deficiencies.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#c8553d",
    "muted": "#7d8ca3",
    "ok": "#2a7f62",
    "warn": "#e0a458",
    "grid": "#d9dee5",
}

SERIES_COLORS = ["#1f4e79", "#c8553d", "#2a7f62", "#e0a458", "#7d5ba6", "#7d8ca3"]


def use_paper_style() -> None:
    """Apply the shared figure style. Call once at the top of a notebook."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def plot_autocorrelation_by_lag(acf: pd.DataFrame, profile, ax=None):
    """Autocorrelation against lag, marking the true daily period and lag 24.

    The figure that makes the sampling error legible at a glance: on GP the daily peak
    sits at lag 16-17 while lag 24 -- used as the "1-day" feature throughout the
    original work -- sits in negative territory.
    """
    ax = ax or plt.subplots(figsize=(8, 3.4))[1]
    ax.axhline(0, color=PALETTE["muted"], lw=0.9)
    ax.plot(acf["lag_samples"], acf["autocorr"], color=PALETTE["primary"], marker="o",
            markersize=2.5)

    true_lag = profile.daily_period
    ax.axvline(true_lag, color=PALETTE["ok"], ls="--", lw=1.4)
    ax.annotate(
        f"true daily period\nlag {true_lag} = {profile.hours_for_lag(true_lag):.1f} h",
        xy=(true_lag, acf.loc[acf["lag_samples"] == true_lag, "autocorr"].iloc[0]),
        xytext=(true_lag + 3, 0.55), color=PALETTE["ok"], fontsize=9,
        arrowprops=dict(arrowstyle="->", color=PALETTE["ok"], lw=1.1),
    )

    if (acf["lag_samples"] == 24).any():
        v24 = acf.loc[acf["lag_samples"] == 24, "autocorr"].iloc[0]
        ax.axvline(24, color=PALETTE["accent"], ls="--", lw=1.4)
        ax.annotate(
            f"lag 24 used as \"1 day\"\n= {profile.hours_for_lag(24):.1f} h, r = {v24:+.2f}",
            xy=(24, v24), xytext=(28, -0.55), color=PALETTE["accent"], fontsize=9,
            arrowprops=dict(arrowstyle="->", color=PALETTE["accent"], lw=1.1),
        )

    ax.set_xlabel("lag (samples)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Daily seasonality sits at the measured period, not at lag 24")
    return ax


def plot_benchmark(summary: pd.DataFrame, baseline: str = "persistence",
                   metric: str = "rmse", ax=None):
    """Horizontal bars of mean metric with fold-to-fold standard deviation."""
    ax = ax or plt.subplots(figsize=(7, 0.42 * len(summary) + 1.2))[1]
    df = summary.sort_values(f"{metric}_mean", ascending=False)
    ref = float(df.loc[df["model"] == baseline, f"{metric}_mean"].iloc[0])

    colors = [
        PALETTE["ok"] if v < ref else (PALETTE["muted"] if m == baseline else PALETTE["accent"])
        for m, v in zip(df["model"], df[f"{metric}_mean"])
    ]
    ax.barh(df["model"], df[f"{metric}_mean"], xerr=df[f"{metric}_std"],
            color=colors, error_kw=dict(ecolor=PALETTE["muted"], lw=1, capsize=3))
    ax.axvline(ref, color=PALETTE["muted"], ls="--", lw=1.3)
    ax.text(ref, -0.75, f" {baseline} = {ref:.2f}", color=PALETTE["muted"], fontsize=9,
            va="top")
    ax.set_xlabel(f"{metric.upper()} (mean ± sd across folds)")
    ax.set_title(f"Corrected benchmark — green beats {baseline}, red does not")
    return ax


def plot_pareto(pareto: pd.DataFrame, ax=None, target_violation: float | None = 0.01):
    """SLA violation rate against provisioned capacity, by policy family.

    The paper's headline figure. A family dominates when its curve lies lower and to
    the left: fewer violations for the same capacity, or less capacity for the same
    violation rate.
    """
    ax = ax or plt.subplots(figsize=(6.4, 4.4))[1]
    for color, (family, grp) in zip(SERIES_COLORS, pareto.groupby("family")):
        grp = grp.sort_values("mean_allocation_ratio")
        ax.plot(grp["mean_allocation_ratio"], grp["sla_violation_rate"],
                marker="o", markersize=4.5, color=color, label=family.replace("_", " "))

    if target_violation is not None:
        ax.axhline(target_violation, color=PALETTE["muted"], ls="--", lw=1.2)
        ax.text(ax.get_xlim()[1], target_violation, f" {target_violation:.0%} SLA target",
                fontsize=9, color=PALETTE["muted"], va="bottom", ha="right")

    ax.set_xlabel("mean allocated capacity / mean demand")
    ax.set_ylabel("SLA violation rate")
    ax.set_title("Capacity–risk frontier by allocation policy")
    ax.legend(title="policy family")
    return ax


def plot_coverage_by_group(per_fold: pd.DataFrame, tau: float, ax=None):
    """Achieved coverage per context group and method, against the nominal level.

    The context-conditional result: a marginally calibrated allocator meets its target
    overall while under-delivering inside the high-variance group.
    """
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    df = per_fold[(per_fold["tau"] == tau) & (per_fold["error"] == "")].copy()
    df["_w"] = df["n"] * df["coverage"]
    agg = df.groupby(["group", "method"]).agg(w=("_w", "sum"), n=("n", "sum"))
    agg["coverage"] = agg["w"] / agg["n"]
    table = agg.reset_index().pivot(index="group", columns="method", values="coverage")

    groups = list(table.index)
    methods = list(table.columns)
    width = 0.8 / len(methods)
    x = np.arange(len(groups))

    for i, (method, color) in enumerate(zip(methods, SERIES_COLORS)):
        ax.bar(x + i * width - 0.4 + width / 2, table[method], width * 0.92,
               label=method, color=color)

    ax.axhline(tau, color=PALETTE["accent"], ls="--", lw=1.4)
    ax.text(len(groups) - 0.4, tau, f" nominal {tau:.0%}", color=PALETTE["accent"],
            fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ") for g in groups])
    ax.set_ylabel("achieved coverage")
    ax.set_ylim(min(0.6, table.min().min() - 0.05), 1.0)
    ax.set_title(f"Per-context coverage at nominal τ = {tau:g}")
    ax.legend(title="calibration")
    return ax


def plot_forecast(timestamps, y_true, y_pred, allocation=None, ax=None,
                  title: str = "Forecast and allocation"):
    """Realised demand against forecast, optionally with the allocated capacity band."""
    ax = ax or plt.subplots(figsize=(11, 3.8))[1]
    ax.plot(timestamps, y_true, color=PALETTE["primary"], label="actual")
    ax.plot(timestamps, y_pred, color=PALETTE["accent"], lw=1.3, ls="--", label="forecast")
    if allocation is not None:
        ax.plot(timestamps, allocation, color=PALETTE["ok"], lw=1.2, label="allocation")
        ax.fill_between(timestamps, y_true, allocation,
                        where=np.asarray(allocation) >= np.asarray(y_true),
                        color=PALETTE["ok"], alpha=0.12, label="headroom")
        breach = np.asarray(allocation) < np.asarray(y_true)
        if breach.any():
            ax.scatter(np.asarray(timestamps)[breach], np.asarray(y_true)[breach],
                       color=PALETTE["accent"], s=18, zorder=5, label="SLA breach")
    ax.set_ylabel("Gbps")
    ax.set_title(title)
    ax.legend(ncol=5)
    return ax


def plot_flag_report(report: pd.DataFrame, ax=None):
    """Context flags ranked by their effect on error variance, not on the mean.

    Flags to the right of 1.0 mark conditions where the forecast is less reliable --
    the signal the context-conditional allocator exploits.
    """
    ax = ax or plt.subplots(figsize=(6.6, 0.42 * len(report) + 1.2))[1]
    df = report.sort_values("variance_ratio")
    colors = [
        PALETTE["accent"] if p < 0.05 else PALETTE["muted"]
        for p in df["p_variance"]
    ]
    ax.barh(df["flag"], df["variance_ratio"], color=colors)
    ax.axvline(1.0, color=PALETTE["primary"], ls="--", lw=1.3)
    ax.set_xlabel("residual sd (flag on) / residual sd (flag off)")
    ax.set_title("Context acts on variance — red = significant (Levene p < 0.05)")
    return ax


def plot_horizon(accuracy: pd.DataFrame, ax=None, models=("random_forest", "xgboost", "ridge")):
    """Accuracy against forecast lead time, with the naive baselines at each lead.

    The figure the multi-horizon argument rests on: the learned models stay roughly
    flat while the naive forecaster degrades steeply, so the gap between them -- the
    actual value of learning -- widens with lead time. The naive lines are drawn in
    distinct styles because they behave very differently: persistence collapses, while
    seasonal-naive is flat by construction (it always steps back a whole cycle).

    Parameters
    ----------
    accuracy:
        A ``horizon_{operator}.csv`` table.
    """
    ax = ax or plt.gca()
    table = accuracy.pivot_table(index="model", columns="lead_hours", values="rmse_mean")

    for model in models:
        if model in table.index:
            ax.plot(table.columns, table.loc[model], marker="o", lw=1.8, label=model)

    naive_styles = {"persistence": ("--", "0.25"), "naive": (":", "0.45")}
    for model in table.index:
        for key, (style, colour) in naive_styles.items():
            if key in model:
                ax.plot(table.columns, table.loc[model], style, color=colour,
                        lw=1.8, marker="x", ms=4, label=model)
                break

    ax.set_xlabel("forecast lead time (hours)")
    ax.set_ylabel("RMSE (Gbps)")
    ax.legend(fontsize=7, framealpha=0.9)
    return ax


def plot_cost(cost: pd.DataFrame, ax=None):
    """Provisioning cost per policy family against the tuned fixed-margin rule.

    Bars are the cost at the level the theory prescribes from the cost ratio, so no
    test-set information enters the choice. The dashed line is the *best* the
    fixed-margin heuristic can do with hindsight about which margin hit which
    violation rate -- the comparison is deliberately biased against the bars.

    Blue beats that line, red does not.

    Parameters
    ----------
    cost:
        A ``cost_{operator}.csv`` table from :func:`bwalloc.allocation.cost_comparison`.
    """
    ax = ax or plt.gca()
    reference = cost[cost["family"] == "fixed_margin"]["cost_best"]
    ranked = cost.dropna(subset=["cost_at_tau_star"]).sort_values("cost_at_tau_star")

    colours = ["#2a7ab0" if s > 0 else "#b03a2a"
               for s in ranked["saving_vs_tuned_baseline"]]
    ax.barh(ranked["family"], ranked["cost_at_tau_star"], color=colours)
    if not reference.empty:
        ax.axvline(float(reference.iloc[0]), color="0.15", ls="--", lw=1.3,
                   label="fixed margin, tuned with hindsight")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    tau_star = float(cost["tau_star"].iloc[0])
    kappa = float(cost["kappa"].iloc[0])
    # Raw string: without it Python turns the "\t" of "\tau" into a tab and mathtext
    # renders the label as "au".
    ax.set_xlabel(
        rf"provisioning cost at $\kappa$={kappa:g}, $\tau^*$={tau_star:.3f}"
    )
    ax.invert_yaxis()
    return ax
