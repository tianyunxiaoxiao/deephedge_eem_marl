"""Feature importance helpers."""

from __future__ import annotations

import numpy as np


def permutation_style_importance(features: np.ndarray, target: np.ndarray, names: list[str], top_k: int = 10) -> list[dict]:
    """Rank features by absolute correlation with a target decision surface."""

    x = features.reshape(-1, features.shape[-1])
    y = target.reshape(-1)
    y = (y - y.mean()) / (y.std() + 1e-12)
    scores = []
    for idx, name in enumerate(names):
        col = x[:, idx]
        col = (col - col.mean()) / (col.std() + 1e-12)
        score = float(abs(np.mean(col * y)))
        scores.append({"feature": name, "importance": score})
    scores.sort(key=lambda item: item["importance"], reverse=True)
    return scores[:top_k]
