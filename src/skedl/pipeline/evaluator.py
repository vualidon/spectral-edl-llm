from __future__ import annotations

import numpy as np

from skedl.core.metrics import (
    aurc_binary,
    auroc_binary,
    average_precision_binary,
    brier_score,
    ece_sensitivity,
    expected_calibration_error,
    negative_log_likelihood,
)


def evaluate_binary_confidence(confidences: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    correct_auroc = auroc_binary(confidences, labels)
    error_scores = 1.0 - np.asarray(confidences, dtype=float)
    error_labels = 1 - np.asarray(labels, dtype=int)
    error_auroc = auroc_binary(error_scores, error_labels)
    error_auprc = average_precision_binary(error_scores, error_labels)
    ece_bins = ece_sensitivity(confidences, labels, bin_counts=[5, 10, 15, 20, 25, 35])
    return {
        "brier": brier_score(confidences, labels),
        "nll": negative_log_likelihood(confidences, labels),
        "ece": expected_calibration_error(confidences, labels),
        "auroc": correct_auroc,
        "auroc_correct": correct_auroc,
        "auroc_error": error_auroc,
        "auprc_error": error_auprc,
        "aurc": aurc_binary(confidences, labels),
        **ece_bins,
    }
