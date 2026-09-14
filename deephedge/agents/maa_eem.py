"""MAA-EEM hedging policy.

This is a deterministic, maintainable implementation of the paper's decision
logic. It uses the EEM Delta as the no-arbitrage anchor, adds an adversarial
stress adjustment for rough-volatility/jump regimes, then applies a no-trade
band that reproduces the Whalley-Wilmott style lethargy effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deephedge.models import HurstSVJJPaths
from deephedge.pricing import EEMPathTargets


@dataclass(frozen=True)
class MAAEEMAgent:
    """EEM-guided multi-agent adversarial hedge."""

    no_trade_band: float = 0.0
    adversarial_strength: float = 0.0
    long_horizon: bool = False
    use_eem: bool = True
    use_adversary: bool = True
    use_hurst: bool = True
    name: str = "MAA-EEM"

    def hedge(self, paths: HurstSVJJPaths, targets: EEMPathTargets) -> np.ndarray:
        anchor = targets.delta if self.use_eem else self._speculative_anchor(paths, targets)
        rough_stress = self._rough_stress(paths) if self.use_hurst else 0.0
        adversary = self.adversarial_strength * rough_stress if self.use_adversary else 0.0
        desired = np.clip(anchor - adversary, 0.0, 1.0)

        band = self.no_trade_band * (1.45 if self.long_horizon else 1.0)
        hedge = np.empty_like(desired)
        hedge[:, 0] = desired[:, 0]
        for t in range(1, desired.shape[1]):
            change = desired[:, t] - hedge[:, t - 1]
            should_trade = np.abs(change) > band
            hedge[:, t] = np.where(should_trade, desired[:, t], hedge[:, t - 1])
        return np.clip(hedge, 0.0, 1.0)

    def _rough_stress(self, paths: HurstSVJJPaths) -> np.ndarray:
        returns = np.diff(np.log(paths.spot), axis=1, prepend=np.log(paths.spot[:, :1]))
        vol = np.sqrt(np.maximum(paths.variance, 1e-10))
        centered_vol = (vol - vol.mean(axis=1, keepdims=True)) / (vol.std(axis=1, keepdims=True) + 1e-8)
        momentum = np.tanh(12.0 * returns)
        jump_alert = (np.abs(paths.price_jumps).mean(axis=1, keepdims=True) > 0).astype(float)
        jump_alert = np.repeat(jump_alert, paths.spot.shape[1], axis=1)
        return np.clip(0.55 * centered_vol + 0.35 * momentum + 0.10 * jump_alert, -2.0, 2.0)

    def _speculative_anchor(self, paths: HurstSVJJPaths, targets: EEMPathTargets) -> np.ndarray:
        returns = np.diff(np.log(paths.spot), axis=1, prepend=np.log(paths.spot[:, :1]))
        drift_chase = np.cumsum(returns, axis=1)
        jump_chase = np.cumsum(np.pad(paths.price_jumps, ((0, 0), (1, 0))), axis=1)
        return np.clip(0.5 + 18.0 * drift_chase + 8.0 * jump_chase + 0.6 * np.sign(returns), -1.0, 1.0)
