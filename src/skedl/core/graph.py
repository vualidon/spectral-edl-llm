from __future__ import annotations

import numpy as np


def _validate_square(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    return 0.5 * (arr + arr.T)


def symmetrized_knn_graph(
    kernel: np.ndarray,
    *,
    k: int,
    include_self: bool = False,
) -> np.ndarray:
    k_mat = _validate_square(kernel)
    n = k_mat.shape[0]
    if n == 0:
        raise ValueError("kernel must be non-empty")
    if n == 1:
        return k_mat.copy() if include_self else np.zeros_like(k_mat)

    if k < 1:
        raise ValueError("k must be >= 1")
    effective_k = min(int(k), n - 1)

    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = k_mat[i].copy()
        row[i] = -np.inf
        neighbor_idx = np.argpartition(row, -effective_k)[-effective_k:]
        mask[i, neighbor_idx] = True

    if include_self:
        np.fill_diagonal(mask, True)

    mask = mask | mask.T
    graph = np.where(mask, k_mat, 0.0)
    if not include_self:
        np.fill_diagonal(graph, 0.0)
    return 0.5 * (graph + graph.T)
