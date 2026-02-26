from __future__ import annotations

import math

import numpy as np


def normalized_symmetric_laplacian(graph: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(graph, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("graph must be square")
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    degrees = np.sum(w, axis=1)
    inv_sqrt_deg = np.zeros_like(degrees)
    valid = degrees > eps
    inv_sqrt_deg[valid] = 1.0 / np.sqrt(degrees[valid])
    d_inv_sqrt = np.diag(inv_sqrt_deg)
    identity = np.eye(w.shape[0], dtype=float)
    lap = identity - d_inv_sqrt @ w @ d_inv_sqrt
    return 0.5 * (lap + lap.T)


def laplacian_eigenvalues(graph: np.ndarray) -> np.ndarray:
    lap = normalized_symmetric_laplacian(graph)
    eigvals = np.linalg.eigvalsh(lap)
    eigvals = np.clip(eigvals, 0.0, None)
    return np.sort(eigvals)


def lambda2_normalized_laplacian(graph: np.ndarray) -> float:
    eigvals = laplacian_eigenvalues(graph)
    if eigvals.size < 2:
        return 0.0
    return float(eigvals[1])


def connectivity_risk(lambda2_value: float, tau: float = 1.0) -> float:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return float(math.exp(-tau * float(lambda2_value)))
