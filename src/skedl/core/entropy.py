from __future__ import annotations

import numpy as np


def trace_normalize_psd(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    mat = 0.5 * (mat + mat.T)
    trace = float(np.trace(mat))
    if trace <= eps:
        raise ValueError("matrix trace must be positive")
    return mat / trace


def kernel_von_neumann_entropy(kernel: np.ndarray, eps: float = 1e-12) -> float:
    rho = trace_normalize_psd(kernel, eps=eps)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(np.sum(eigvals))
    if total <= eps:
        return 0.0
    probs = eigvals / total
    probs = np.clip(probs, eps, 1.0)
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy
