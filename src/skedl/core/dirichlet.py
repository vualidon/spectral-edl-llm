from __future__ import annotations

import numpy as np
from scipy.special import digamma


def _validate_alpha(alpha: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(alpha, dtype=float)
    if arr.ndim != 1:
        raise ValueError("alpha must be a 1D vector")
    if arr.size == 0:
        raise ValueError("alpha must be non-empty")
    if np.any(arr <= eps):
        raise ValueError("alpha entries must be positive")
    return arr


def dirichlet_mean(alpha: np.ndarray) -> np.ndarray:
    a = _validate_alpha(alpha)
    return a / np.sum(a)


def dirichlet_epistemic_uncertainty(alpha: np.ndarray) -> float:
    a = _validate_alpha(alpha)
    k = a.size
    return float(k / np.sum(a))


def dirichlet_epistemic_uncertainty_with_plus_one(alpha: np.ndarray) -> float:
    a = _validate_alpha(alpha)
    k = a.size
    return float(k / np.sum(a + 1.0))


def dirichlet_aleatoric_uncertainty(alpha: np.ndarray, eps: float = 1e-12) -> float:
    a = _validate_alpha(alpha, eps=eps)
    a0 = float(np.sum(a))
    probs = a / a0
    term = digamma(a + 1.0) - digamma(a0 + 1.0)
    au = -float(np.sum(probs * term))
    return max(0.0, au)


def edl_confidence(alpha: np.ndarray) -> float:
    a = _validate_alpha(alpha)
    s = float(np.sum(a))
    probs = a / s
    u_d = float(a.size / s)
    conf = (1.0 - u_d) * float(np.max(probs))
    return float(np.clip(conf, 0.0, 1.0))


def evidence_to_alpha(evidence: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    e = np.asarray(evidence, dtype=float)
    if e.ndim != 1:
        raise ValueError("evidence must be a 1D vector")
    if np.any(e < 0):
        raise ValueError("evidence must be non-negative")
    return e + (1.0 + eps)
