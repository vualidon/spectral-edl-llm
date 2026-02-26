from __future__ import annotations

import numpy as np


def test_edl_confidence_increases_with_concentrated_alpha():
    from skedl.core.dirichlet import edl_confidence

    diffuse = np.array([2.0, 2.0, 2.0])
    concentrated = np.array([10.0, 2.0, 2.0])

    assert edl_confidence(concentrated) > edl_confidence(diffuse)


def test_dirichlet_uncertainties_are_non_negative():
    from skedl.core.dirichlet import dirichlet_aleatoric_uncertainty, dirichlet_epistemic_uncertainty

    alpha = np.array([4.0, 2.0, 1.5], dtype=float)

    assert dirichlet_aleatoric_uncertainty(alpha) >= 0.0
    assert dirichlet_epistemic_uncertainty(alpha) >= 0.0
