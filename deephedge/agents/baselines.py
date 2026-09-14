"""Baseline hedging policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deephedge.models import HurstSVJJPaths
from deephedge.pricing import EEMPathTargets, bs_call_delta


@dataclass(frozen=True)
class BSDeltaAgent:
    """Classic Black-Scholes Delta hedge using a constant volatility input."""

    strike: float
    rate: float
    volatility: float = 0.16
    name: str = "BS-Delta"

    def hedge(self, paths: HurstSVJJPaths, targets: EEMPathTargets) -> np.ndarray:
        tau = targets.tau[None, :]
        return np.clip(bs_call_delta(paths.spot, self.strike, tau, self.rate, self.volatility), 0.0, 1.0)


@dataclass(frozen=True)
class MinimumVarianceAgent:
    """A conservative shrinkage hedge standing in for MVH."""

    shrinkage: float = 0.82
    name: str = "Minimum-Variance"

    def hedge(self, paths: HurstSVJJPaths, targets: EEMPathTargets) -> np.ndarray:
        realized = np.sqrt(np.maximum(paths.variance, 1e-10))
        stress = np.clip((realized - realized.mean()) / (realized.std() + 1e-8), -2.0, 2.0)
        return np.clip(self.shrinkage * targets.delta - 0.015 * stress, 0.0, 1.0)


@dataclass(frozen=True)
class SACBaselineAgent:
    """Deterministic SAC-like policy without EEM anchoring."""

    name: str = "SAC-Baseline"

    def hedge(self, paths: HurstSVJJPaths, targets: EEMPathTargets) -> np.ndarray:
        momentum = np.diff(np.log(paths.spot), axis=1, prepend=np.log(paths.spot[:, :1]))
        raw = 0.70 * targets.delta + 0.20 * (paths.spot > paths.spot[:, :1]) + 2.0 * momentum
        return np.clip(_smooth(raw, alpha=0.35), 0.0, 1.0)


@dataclass(frozen=True)
class PPOBaselineAgent:
    """Deterministic PPO-like clipped policy without EEM anchoring."""

    name: str = "PPO-Baseline"

    def hedge(self, paths: HurstSVJJPaths, targets: EEMPathTargets) -> np.ndarray:
        moneyness_signal = np.tanh((paths.spot / np.maximum(paths.spot[:, :1], 1e-8) - 1.0) * 4.0)
        raw = 0.76 * targets.delta + 0.08 * moneyness_signal
        return np.clip(_smooth(raw, alpha=0.25), 0.0, 1.0)


def _smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.array(values, copy=True)
    for t in range(1, out.shape[1]):
        out[:, t] = alpha * out[:, t] + (1.0 - alpha) * out[:, t - 1]
    return out
