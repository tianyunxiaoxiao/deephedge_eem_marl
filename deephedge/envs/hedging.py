"""Hedging MDP accounting and EEM reward evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deephedge.models import HurstSVJJPaths
from deephedge.pricing import EEMPathTargets


@dataclass(frozen=True)
class HedgingResult:
    """Outputs from replaying a hedge policy over simulated paths."""

    name: str
    hedge: np.ndarray
    pnl: np.ndarray
    equity: np.ndarray
    tracking_error: np.ndarray
    transaction_costs: np.ndarray
    turnover: np.ndarray
    rewards: np.ndarray

    @property
    def mean_pnl(self) -> float:
        return float(np.mean(self.pnl))

    @property
    def mse(self) -> float:
        return float(np.mean(self.tracking_error**2))

    @property
    def mean_turnover(self) -> float:
        return float(np.mean(self.turnover))


def evaluate_hedge(
    name: str,
    paths: HurstSVJJPaths,
    targets: EEMPathTargets,
    hedge: np.ndarray,
    strike: float,
    transaction_cost_bps: float,
    eem_weight: float = 1.0,
    action_penalty: float = 0.001,
) -> HedgingResult:
    """Replay hedge positions and compute EEM tracking rewards."""

    spot = paths.spot
    n_paths, n_cols = spot.shape
    n_steps = n_cols - 1
    if hedge.shape != spot.shape:
        raise ValueError(f"hedge shape {hedge.shape} must match spot shape {spot.shape}")

    hedge = np.clip(hedge, -1.0, 1.0)
    cost_rate = transaction_cost_bps / 10_000.0
    payoff = np.maximum(spot[:, -1] - strike, 0.0)

    cash = targets.call_value[:, 0] - hedge[:, 0] * spot[:, 0]
    initial_cost = cost_rate * np.abs(hedge[:, 0]) * spot[:, 0]
    cash -= initial_cost

    equity = np.zeros_like(spot)
    equity[:, 0] = cash + hedge[:, 0] * spot[:, 0] - targets.call_value[:, 0]
    costs = np.zeros((n_paths, n_steps), dtype=float)
    tracking_error = np.zeros((n_paths, n_steps), dtype=float)
    rewards = np.zeros((n_paths, n_steps), dtype=float)

    for t in range(n_steps):
        prev_portfolio = cash + hedge[:, t] * spot[:, t]
        trade = hedge[:, t + 1] - hedge[:, t]
        costs[:, t] = cost_rate * np.abs(trade) * spot[:, t + 1]
        cash -= trade * spot[:, t + 1] + costs[:, t]
        new_portfolio = cash + hedge[:, t + 1] * spot[:, t + 1]
        portfolio_increment = new_portfolio - prev_portfolio
        tracking_error[:, t] = portfolio_increment - targets.increments[:, t]
        rewards[:, t] = (
            -eem_weight * tracking_error[:, t] ** 2
            - costs[:, t]
            - action_penalty * trade**2 * spot[:, t + 1]
        )
        equity[:, t + 1] = new_portfolio - targets.call_value[:, t + 1]

    terminal_value = cash + hedge[:, -1] * spot[:, -1]
    pnl = terminal_value - payoff
    turnover = np.sum(np.abs(np.diff(hedge, axis=1)), axis=1)

    return HedgingResult(
        name=name,
        hedge=hedge,
        pnl=pnl,
        equity=equity,
        tracking_error=tracking_error,
        transaction_costs=costs,
        turnover=turnover,
        rewards=rewards,
    )
