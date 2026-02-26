from __future__ import annotations

import numpy as np


def _as_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if arr.shape[0] == 0:
        raise ValueError("embeddings must contain at least one row")
    return arr


def cosine_kernel(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    z = _as_2d_embeddings(embeddings)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    z_norm = z / norms
    sim = z_norm @ z_norm.T
    kernel = 0.5 * (1.0 + sim)
    kernel = np.clip(kernel, 0.0, 1.0)
    return 0.5 * (kernel + kernel.T)


def rbf_kernel(
    embeddings: np.ndarray,
    gamma: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    z = _as_2d_embeddings(embeddings)
    diff = z[:, None, :] - z[None, :, :]
    sq_dists = np.sum(diff * diff, axis=-1)
    if gamma is None:
        positive = sq_dists[sq_dists > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        gamma = 1.0 / max(scale, eps)
    kernel = np.exp(-gamma * sq_dists)
    return 0.5 * (kernel + kernel.T)
