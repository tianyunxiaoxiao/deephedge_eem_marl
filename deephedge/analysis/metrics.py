"""Performance metrics for hedging experiments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from deephedge.envs import HedgingResult, evaluate_hedge
from deephedge.models import HurstSVJJPaths
from deephedge.pricing import EEMPathTargets


def performance_summary(result: HedgingResult, initial_premium: float, dt: float) -> dict[str, float]:
    """Summarize one strategy in the paper's table vocabulary."""

    horizon_years = result.tracking_error.shape[1] * dt
    returns = result.pnl / max(initial_premium, 1e-8)
    ann_return = float(np.mean(returns) / max(horizon_years, 1e-8) * 100.0)
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(1.0 / max(horizon_years, 1e-8)))
    mean_equity = np.mean(result.equity, axis=0)
    running_peak = np.maximum.accumulate(mean_equity)
    drawdown = mean_equity - running_peak
    max_drawdown = float(np.min(drawdown) / max(initial_premium, 1e-8) * 100.0)
    return {
        "annual_return_pct": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown,
        "hedging_mse": result.mse,
        "win_rate_pct": float(np.mean(result.pnl > 0.0) * 100.0),
        "turnover": result.mean_turnover,
        "transaction_cost": float(np.mean(np.sum(result.transaction_costs, axis=1))),
        "mean_pnl": result.mean_pnl,
    }


def compute_breakeven_costs(
    paths: HurstSVJJPaths,
    targets: EEMPathTargets,
    strike: float,
    baseline_agent,
    challenger_agents: list,
    cost_grid_bps: np.ndarray | None = None,
) -> dict[str, float]:
    """Find the largest one-sided cost where each challenger beats BS PnL."""

    grid = cost_grid_bps if cost_grid_bps is not None else np.arange(0.0, 101.0, 5.0)
    output: dict[str, float] = {}
    baseline_hedge = baseline_agent.hedge(paths, targets)
    for challenger in challenger_agents:
        challenger_hedge = challenger.hedge(paths, targets)
        best = 0.0
        for bps in grid:
            bs = evaluate_hedge(baseline_agent.name, paths, targets, baseline_hedge, strike, float(bps))
            ch = evaluate_hedge(challenger.name, paths, targets, challenger_hedge, strike, float(bps))
            if np.mean(ch.pnl) >= np.mean(bs.pnl):
                best = float(bps)
        output[challenger.name] = best
    return output


def masked_mean_metric(
    result_factory: Callable[[float], HedgingResult],
    mask: np.ndarray,
    metric: str = "pnl",
) -> float:
    """Evaluate a result metric under a state mask."""

    result = result_factory(10.0)
    path_mask = np.any(mask[:, : result.hedge.shape[1]], axis=1)
    values = getattr(result, metric)
    return float(np.mean(values[path_mask])) if np.any(path_mask) else float("nan")
