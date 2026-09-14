"""Hurst-SVJJ simulation with Gamma-BSS volatility memory.

The implementation follows the paper's model ingredients while staying
lightweight enough for repeatable unit tests:

- Gamma-BSS kernel: g(x) = x^(H - 1/2) exp(-lambda x).
- Positive variance via a log-variance process.
- Price jumps and volatility jumps.
- Correlated price/volatility Brownian shocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deephedge.config import MarketConfig


@dataclass(frozen=True)
class HurstSVJJPaths:
    """Container for simulated market paths."""

    spot: np.ndarray
    variance: np.ndarray
    log_spot: np.ndarray
    price_jumps: np.ndarray
    vol_jumps: np.ndarray
    dt: float
    config: MarketConfig

    @property
    def returns(self) -> np.ndarray:
        return np.diff(np.log(self.spot), axis=1)

    @property
    def realized_volatility(self) -> np.ndarray:
        scale = np.sqrt(1.0 / self.dt)
        return np.std(self.returns, axis=1) * scale


def gamma_bss_kernel(max_lag: int, hurst: float, memory_decay: float, dt: float) -> np.ndarray:
    """Return a normalized Gamma-BSS kernel over positive lags."""

    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    if not 0.0 < hurst < 1.0:
        raise ValueError("hurst must be in (0, 1)")
    if memory_decay <= 0.0:
        raise ValueError("memory_decay must be positive")

    lags = np.arange(1, max_lag + 1, dtype=float) * dt
    kernel = np.power(lags, hurst - 0.5) * np.exp(-memory_decay * lags)
    norm = np.sqrt(np.sum(kernel * kernel))
    return kernel / norm


def _correlated_normals(rng: np.random.Generator, shape: tuple[int, int], corr: float) -> tuple[np.ndarray, np.ndarray]:
    z1 = rng.standard_normal(shape)
    z2 = rng.standard_normal(shape)
    corr = float(np.clip(corr, -0.999, 0.999))
    return z1, corr * z1 + np.sqrt(1.0 - corr * corr) * z2


def simulate_hurst_svjj(
    n_paths: int,
    n_steps: int,
    config: MarketConfig | None = None,
    seed: int | None = None,
) -> HurstSVJJPaths:
    """Simulate Hurst-SVJJ paths under the physical measure."""

    if n_paths <= 0 or n_steps <= 1:
        raise ValueError("n_paths must be positive and n_steps must exceed 1")

    cfg = config or MarketConfig()
    rng = np.random.default_rng(seed)
    dt = 1.0 / cfg.trading_days

    spot = np.empty((n_paths, n_steps + 1), dtype=float)
    variance = np.empty_like(spot)
    log_spot = np.empty_like(spot)
    price_jumps = np.zeros((n_paths, n_steps), dtype=float)
    vol_jumps = np.zeros((n_paths, n_steps), dtype=float)

    spot[:, 0] = cfg.spot0
    log_spot[:, 0] = np.log(cfg.spot0)
    variance[:, 0] = cfg.initial_variance

    z_price, z_vol = _correlated_normals(rng, (n_paths, n_steps), cfg.corr)
    kernel = gamma_bss_kernel(n_steps, cfg.hurst, cfg.memory_decay, dt)
    vol_memory = np.zeros((n_paths, n_steps), dtype=float)

    price_jump_prob = cfg.price_jump_intensity * dt
    vol_jump_prob = cfg.vol_jump_intensity * dt

    price_jump_flags = rng.random((n_paths, n_steps)) < price_jump_prob
    vol_jump_flags = rng.random((n_paths, n_steps)) < vol_jump_prob
    price_jumps[price_jump_flags] = rng.normal(
        cfg.price_jump_mean, cfg.price_jump_std, size=int(price_jump_flags.sum())
    )
    vol_jumps[vol_jump_flags] = np.maximum(
        0.0,
        rng.normal(cfg.vol_jump_mean, cfg.vol_jump_std, size=int(vol_jump_flags.sum())),
    )

    log_long_var = np.log(cfg.long_variance)
    log_var = np.full(n_paths, np.log(cfg.initial_variance), dtype=float)
    compensated_jump = cfg.price_jump_intensity * (np.exp(cfg.price_jump_mean + 0.5 * cfg.price_jump_std**2) - 1.0)

    for t in range(n_steps):
        shock = cfg.vol_of_vol * np.sqrt(dt) * z_vol[:, t]
        for lag in range(t + 1):
            vol_memory[:, t] += kernel[lag] * cfg.vol_of_vol * np.sqrt(dt) * z_vol[:, t - lag]

        mean_reversion = 1.25 * (log_long_var - log_var) * dt
        log_var = log_long_var + vol_memory[:, t] + mean_reversion + vol_jumps[:, t]
        variance[:, t + 1] = np.clip(np.exp(log_var), 1e-5, 4.0)

        var_t = variance[:, t]
        diffusion = np.sqrt(var_t * dt) * z_price[:, t]
        drift = (cfg.drift - compensated_jump - 0.5 * var_t) * dt
        log_spot[:, t + 1] = log_spot[:, t] + drift + diffusion + price_jumps[:, t]
        spot[:, t + 1] = np.exp(log_spot[:, t + 1])

    return HurstSVJJPaths(
        spot=spot,
        variance=variance,
        log_spot=log_spot,
        price_jumps=price_jumps,
        vol_jumps=vol_jumps,
        dt=dt,
        config=cfg,
    )
