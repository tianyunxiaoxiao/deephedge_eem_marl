"""Configuration objects for the synthetic Hurst-SVJJ study."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketConfig:
    """Parameters for the Hurst-SVJJ synthetic market."""

    spot0: float = 100.0
    rate: float = 0.02
    drift: float = 0.06
    initial_variance: float = 0.04
    long_variance: float = 0.04
    vol_of_vol: float = 0.55
    hurst: float = 0.72
    memory_decay: float = 1.35
    price_jump_intensity: float = 0.10
    vol_jump_intensity: float = 0.06
    price_jump_mean: float = -0.06
    price_jump_std: float = 0.09
    vol_jump_mean: float = 0.18
    vol_jump_std: float = 0.08
    corr: float = -0.55
    trading_days: int = 252


@dataclass(frozen=True)
class StudyConfig:
    """Default experiment dimensions."""

    seed: int = 7
    n_paths: int = 256
    n_steps: int = 64
    maturity_days: int = 64
    strike: float = 100.0
    transaction_cost_bps: float = 10.0
    window: int = 20
    alpha_factor_count: int = 191
