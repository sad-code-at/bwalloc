"""Context-conditional, cost-aware bandwidth allocation from sparse operator traces.

The package is organised around the observation that provisioning is a *decision*
problem with asymmetric costs, not a point-forecasting problem scored by RMSE.

Modules
-------
data
    Loading, plus the sampling-rate diagnostics that every seasonal hyperparameter is
    derived from.
features
    Leak-safe design matrices; Fourier time features that survive irregular sampling.
splits
    Rolling-origin backtesting with an optional conformal calibration slice.
baselines
    Naive forecasters that any reported model must beat.
models
    Point and quantile forecasters behind one interface.
evaluate
    Backtest harness; every model scored on identical folds.
metrics
    Accuracy (RMSE/MAE/MASE/pinball) and allocation (SLA rate, overprovisioning, cost).
context
    Context-flag validation, disjoint grouping, and the uncertainty model that drives
    context-adaptive margins.
conformal
    Distribution-free calibration: marginal, locally adaptive, and group-conditional.
allocation
    The decision layer -- cost model, allocation policies, and the SLA/overprovisioning
    frontier.
stats
    Diebold-Mariano and friends, so model rankings carry p-values.
"""

from __future__ import annotations

__version__ = "0.1.0"

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG this package can reach.

    Called at the top of each notebook so results regenerate exactly.
    """
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


from . import (  # noqa: E402
    allocation,
    baselines,
    conformal,
    context,
    data,
    evaluate,
    features,
    metrics,
    models,
    splits,
    stats,
)

__all__ = [
    "allocation",
    "baselines",
    "conformal",
    "context",
    "data",
    "evaluate",
    "features",
    "metrics",
    "models",
    "splits",
    "stats",
    "set_seed",
    "SEED",
    "__version__",
]
