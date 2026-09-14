"""Dependency-light Black-Scholes helpers."""

from __future__ import annotations

from math import erf

import numpy as np


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def bs_call_price(
    spot: np.ndarray | float,
    strike: float,
    tau: np.ndarray | float,
    rate: float,
    volatility: np.ndarray | float,
) -> np.ndarray:
    """Return European call values with stable expiry handling."""

    s = np.asarray(spot, dtype=float)
    t = np.asarray(tau, dtype=float)
    vol = np.asarray(volatility, dtype=float)
    t = np.maximum(t, 0.0)
    intrinsic = np.maximum(s - strike, 0.0)

    safe_t = np.maximum(t, 1e-10)
    safe_vol = np.maximum(vol, 1e-8)
    denom = safe_vol * np.sqrt(safe_t)
    d1 = (np.log(np.maximum(s, 1e-12) / strike) + (rate + 0.5 * safe_vol**2) * safe_t) / denom
    d2 = d1 - denom
    price = s * _norm_cdf(d1) - strike * np.exp(-rate * safe_t) * _norm_cdf(d2)
    return np.where(t <= 1e-9, intrinsic, np.maximum(price, intrinsic * 0.999))


def bs_call_delta(
    spot: np.ndarray | float,
    strike: float,
    tau: np.ndarray | float,
    rate: float,
    volatility: np.ndarray | float,
) -> np.ndarray:
    """Return Black-Scholes call Delta."""

    s = np.asarray(spot, dtype=float)
    t = np.asarray(tau, dtype=float)
    vol = np.asarray(volatility, dtype=float)
    safe_t = np.maximum(t, 1e-10)
    safe_vol = np.maximum(vol, 1e-8)
    denom = safe_vol * np.sqrt(safe_t)
    d1 = (np.log(np.maximum(s, 1e-12) / strike) + (rate + 0.5 * safe_vol**2) * safe_t) / denom
    delta = _norm_cdf(d1)
    return np.where(t <= 1e-9, (s > strike).astype(float), delta)
