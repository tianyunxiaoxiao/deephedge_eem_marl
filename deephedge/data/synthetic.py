"""Synthetic SPX option and factor data.

The paper references licensed SPX, CBOE option, IV surface, and
WorldQuant-style Alpha_001..Alpha_191 inputs. This module creates a
deterministic substitute with the same schema and stylized facts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deephedge.config import StudyConfig
from deephedge.models import HurstSVJJPaths, simulate_hurst_svjj


@dataclass(frozen=True)
class SyntheticMarketData:
    """Market panel used by experiments and tests."""

    paths: HurstSVJJPaths
    close: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    realized_vol: np.ndarray
    implied_vol: np.ndarray
    moneyness: np.ndarray
    maturity_days: np.ndarray
    alpha_factors: np.ndarray
    feature_names: list[str]
    regimes: dict[str, np.ndarray]

    def describe(self) -> dict[str, dict[str, float]]:
        """Return descriptive statistics mirroring the paper's Table 1."""

        returns = np.diff(np.log(self.close), axis=1).reshape(-1)
        rv = self.realized_vol.reshape(-1)
        iv = self.implied_vol.reshape(-1)
        return {
            "log_returns_daily": _summary(returns),
            "realized_volatility_ann": _summary(rv),
            "implied_volatility_ann": _summary(iv),
        }


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x)
    for t in range(x.shape[1]):
        start = max(0, t - window + 1)
        out[:, t] = np.std(x[:, start : t + 1], axis=1)
    return out


def _summary(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-12)
    centered = x - mean
    skew = float(np.mean(centered**3) / std**3)
    kurt = float(np.mean(centered**4) / std**4)
    jb = len(x) / 6.0 * (skew**2 + 0.25 * (kurt - 3.0) ** 2)
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skewness": skew,
        "kurtosis": kurt,
        "jarque_bera": float(jb),
        "hurst_rs": float(estimate_hurst_rs(x)),
    }


def estimate_hurst_rs(series: np.ndarray, min_chunk: int = 8) -> float:
    """Estimate Hurst exponent with a robust rescaled-range slope."""

    x = np.asarray(series, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) < min_chunk * 4:
        return 0.5

    max_chunk = max(min_chunk + 1, len(x) // 4)
    sizes = np.unique(np.geomspace(min_chunk, max_chunk, num=6).astype(int))
    rs_values: list[float] = []
    used_sizes: list[int] = []
    for size in sizes:
        n_blocks = len(x) // size
        if n_blocks < 2:
            continue
        blocks = x[: n_blocks * size].reshape(n_blocks, size)
        adjusted = blocks - blocks.mean(axis=1, keepdims=True)
        cumulative = np.cumsum(adjusted, axis=1)
        r = cumulative.max(axis=1) - cumulative.min(axis=1)
        s = blocks.std(axis=1) + 1e-12
        rs = np.mean(r / s)
        if rs > 0:
            rs_values.append(float(rs))
            used_sizes.append(int(size))
    if len(rs_values) < 2:
        return 0.5
    slope = np.polyfit(np.log(used_sizes), np.log(rs_values), 1)[0]
    return float(np.clip(slope, 0.05, 0.95))


def generate_synthetic_market_data(config: StudyConfig | None = None) -> SyntheticMarketData:
    """Generate the complete synthetic data panel."""

    cfg = config or StudyConfig()
    paths = simulate_hurst_svjj(cfg.n_paths, cfg.n_steps, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed + 101)

    close = paths.spot
    daily_noise = rng.normal(0.0, 0.0018, size=close.shape)
    open_ = np.maximum(1e-6, close * np.exp(daily_noise))
    spread = np.abs(rng.normal(0.0, 0.006, size=close.shape)) * close
    high = np.maximum(open_, close) + spread
    low = np.maximum(1e-6, np.minimum(open_, close) - spread)

    returns = np.diff(np.log(close), axis=1, prepend=np.log(close[:, :1]))
    realized = _rolling_std(returns, window=max(3, min(cfg.window, cfg.n_steps))) * np.sqrt(252.0)
    realized = np.maximum(realized, 0.03)
    implied = np.clip(
        0.05 + 0.86 * realized + 0.03 * rng.standard_t(df=5, size=realized.shape),
        0.05,
        1.20,
    )

    volume = np.maximum(
        1.0,
        1_000_000.0 * (1.0 + 5.0 * realized) * np.exp(rng.normal(0.0, 0.25, close.shape)),
    )

    strikes = np.array([0.92, 0.97, 1.00, 1.03, 1.08]) * cfg.strike
    strike_grid = rng.choice(strikes, size=close.shape)
    moneyness = close / strike_grid
    maturity_choices = np.array([10, 21, 45, 64, 90, 126])
    maturity_days = rng.choice(maturity_choices, size=close.shape)

    base_features = [
        returns,
        realized,
        implied,
        moneyness - 1.0,
        np.log(volume),
        np.tanh(10.0 * returns),
    ]
    alpha_factors = []
    for idx in range(cfg.alpha_factor_count):
        source = base_features[idx % len(base_features)]
        lag = idx % 7
        shifted = np.roll(source, lag, axis=1)
        if lag:
            shifted[:, :lag] = shifted[:, lag : lag + 1]
        nonlinear = np.tanh((idx % 11 + 1) * shifted)
        alpha_factors.append(nonlinear + 0.01 * rng.standard_normal(close.shape))
    alpha = np.stack(alpha_factors, axis=-1)

    high_vol = realized > np.quantile(realized, 0.75)
    bull = returns > np.quantile(returns, 0.60)
    bear = returns < np.quantile(returns, 0.40)
    regimes = {
        "low_volatility": ~high_vol,
        "high_volatility": high_vol,
        "bull_market": bull,
        "bear_market": bear,
    }

    feature_names = [f"ALPHA_{i:03d}" for i in range(1, cfg.alpha_factor_count + 1)]
    return SyntheticMarketData(
        paths=paths,
        close=close,
        open=open_,
        high=high,
        low=low,
        volume=volume,
        realized_vol=realized,
        implied_vol=implied,
        moneyness=moneyness,
        maturity_days=maturity_days,
        alpha_factors=alpha,
        feature_names=feature_names,
        regimes=regimes,
    )
