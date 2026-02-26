from __future__ import annotations

import math

import numpy as np


def _validate_binary_inputs(confidences: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(confidences, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if c.shape[0] != y.shape[0]:
        raise ValueError("confidences and labels must have same length")
    if c.size == 0:
        raise ValueError("inputs must be non-empty")
    if np.any((c < 0.0) | (c > 1.0)):
        raise ValueError("confidences must be in [0, 1]")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must be binary (0/1)")
    return c, y


def brier_score(confidences: np.ndarray, labels: np.ndarray) -> float:
    c, y = _validate_binary_inputs(confidences, labels)
    return float(np.mean((c - y) ** 2))


def negative_log_likelihood(confidences: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    c, y = _validate_binary_inputs(confidences, labels)
    p = np.clip(c, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(
    confidences: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    c, y = _validate_binary_inputs(confidences, labels)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (c >= left) & (c <= right)
        else:
            mask = (c >= left) & (c < right)
        if not np.any(mask):
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(c[mask]))
        ece += (np.sum(mask) / c.size) * abs(acc - conf)
    return float(ece)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    n = values.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def auroc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    s, y = _validate_binary_inputs(scores, labels)
    pos = y == 1
    neg = y == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _rankdata(s)
    rank_sum_pos = float(np.sum(ranks[pos]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    s, y = _validate_binary_inputs(scores, labels)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return math.nan

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    ap = float(np.sum(precision[y_sorted == 1]) / n_pos)
    return ap


def risk_coverage_curve(confidences: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, y = _validate_binary_inputs(confidences, labels)
    order = np.argsort(-c, kind="mergesort")
    y_sorted = y[order]
    n = y_sorted.size
    coverage = np.arange(1, n + 1, dtype=float) / float(n)
    errors = 1 - y_sorted
    cum_errors = np.cumsum(errors, dtype=float)
    risk = cum_errors / np.arange(1, n + 1, dtype=float)
    return coverage, risk


def aurc_binary(confidences: np.ndarray, labels: np.ndarray) -> float:
    _, risk = risk_coverage_curve(confidences, labels)
    return float(np.mean(risk))


def reliability_bins(
    confidences: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> list[dict[str, float]]:
    c, y = _validate_binary_inputs(confidences, labels)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        left, right = float(bins[i]), float(bins[i + 1])
        if i == n_bins - 1:
            mask = (c >= left) & (c <= right)
        else:
            mask = (c >= left) & (c < right)
        count = int(np.sum(mask))
        if count:
            acc = float(np.mean(y[mask]))
            conf = float(np.mean(c[mask]))
        else:
            acc = 0.0
            conf = 0.0
        rows.append(
            {
                "bin_index": float(i),
                "left": left,
                "right": right,
                "count": float(count),
                "accuracy": acc,
                "mean_confidence": conf,
                "gap": float(abs(acc - conf)) if count else 0.0,
            }
        )
    return rows


def ece_sensitivity(
    confidences: np.ndarray,
    labels: np.ndarray,
    *,
    bin_counts: list[int] | tuple[int, ...] = (10, 15, 25, 35),
) -> dict[str, float]:
    c, y = _validate_binary_inputs(confidences, labels)
    out: dict[str, float] = {}
    for n_bins in bin_counts:
        out[f"ece@{int(n_bins)}"] = expected_calibration_error(c, y, n_bins=int(n_bins))
    return out
