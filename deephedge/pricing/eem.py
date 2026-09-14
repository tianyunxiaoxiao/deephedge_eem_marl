"""Equivalent Expected Measure pricing targets.

The paper uses EEM to prevent the learner from exploiting predictable
fractional drift. In this implementation the EEM target is a
quasi-risk-neutral anchor: physical Hurst-SVJJ paths supply the state, while
option values are marked with risk-neutral drift and local stochastic
variance. This creates a stable benchmark increment for the RL reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deephedge.models import HurstSVJJPaths
from deephedge.pricing.black_scholes import bs_call_delta, bs_call_price


@dataclass(frozen=True)
class EEMPathTargets:
    """Per-path EEM values and hedge deltas."""

    call_value: np.ndarray
    delta: np.ndarray
    tau: np.ndarray
    increments: np.ndarray


def compute_eem_targets(paths: HurstSVJJPaths, strike: float, rate: float) -> EEMPathTargets:
    """Compute EEM call prices and deltas at every simulated time."""

    n_steps = paths.spot.shape[1] - 1
    tau = np.arange(n_steps, -1, -1, dtype=float) * paths.dt
    volatility = np.sqrt(np.maximum(paths.variance, 1e-10))
    values = bs_call_price(paths.spot, strike=strike, tau=tau[None, :], rate=rate, volatility=volatility)
    deltas = bs_call_delta(paths.spot, strike=strike, tau=tau[None, :], rate=rate, volatility=volatility)
    return EEMPathTargets(
        call_value=values,
        delta=np.clip(deltas, 0.0, 1.0),
        tau=tau,
        increments=np.diff(values, axis=1),
    )
