from __future__ import annotations

import numpy as np


def test_brier_and_ece_are_computed_for_binary_confidence():
    from skedl.core.metrics import brier_score, expected_calibration_error

    confidences = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
    labels = np.array([1, 1, 0, 0], dtype=int)

    brier = brier_score(confidences, labels)
    ece = expected_calibration_error(confidences, labels, n_bins=2)

    assert brier >= 0.0
    assert 0.0 <= ece <= 1.0


def test_average_precision_binary_is_one_for_perfect_ranking():
    from skedl.core.metrics import average_precision_binary

    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
    labels = np.array([1, 1, 0, 0], dtype=int)

    ap = average_precision_binary(scores, labels)

    assert ap == 1.0


def test_risk_coverage_curve_and_aurc_return_valid_ranges():
    from skedl.core.metrics import aurc_binary, risk_coverage_curve

    confidences = np.array([0.95, 0.8, 0.6, 0.4, 0.2], dtype=float)
    labels = np.array([1, 1, 0, 1, 0], dtype=int)

    coverage, risk = risk_coverage_curve(confidences, labels)
    aurc = aurc_binary(confidences, labels)

    assert coverage.shape == risk.shape
    assert coverage.shape[0] == 5
    assert np.all(np.diff(coverage) > 0)
    assert np.isclose(coverage[0], 1.0 / 5.0)
    assert np.isclose(coverage[-1], 1.0)
    assert np.all((risk >= 0.0) & (risk <= 1.0))
    assert 0.0 <= aurc <= 1.0


def test_reliability_bins_and_ece_sensitivity_cover_all_examples():
    from skedl.core.metrics import ece_sensitivity, reliability_bins

    confidences = np.array([0.95, 0.8, 0.7, 0.45, 0.2, 0.1], dtype=float)
    labels = np.array([1, 1, 0, 1, 0, 0], dtype=int)

    bins = reliability_bins(confidences, labels, n_bins=3)
    ece_by_bins = ece_sensitivity(confidences, labels, bin_counts=[5, 10, 15, 20, 25, 35])

    assert len(bins) == 3
    assert sum(int(b["count"]) for b in bins) == 6
    assert set(ece_by_bins.keys()) == {"ece@5", "ece@10", "ece@15", "ece@20", "ece@25", "ece@35"}
    assert all(0.0 <= float(v) <= 1.0 for v in ece_by_bins.values())
