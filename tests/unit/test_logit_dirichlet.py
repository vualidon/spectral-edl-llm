from __future__ import annotations

import numpy as np


def test_logit_dirichlet_extracts_au_eu_from_scores():
    from skedl.models.logit_dirichlet import LogitDirichletExtractor

    extractor = LogitDirichletExtractor(top_k=3, evidence_transform="shift")

    logits_steps = [
        np.array([4.2, 2.5, 0.3, -1.0], dtype=float),
        np.array([3.8, 1.9, 0.1, -0.8], dtype=float),
        np.array([5.1, 2.7, 0.2, -0.4], dtype=float),
    ]

    features = extractor.extract_from_logits_steps(logits_steps)

    assert "au_mean" in features
    assert "eu_mean" in features
    assert "c_d_logit" in features
    assert features["au_mean"] >= 0.0
    assert features["eu_mean"] >= 0.0
    assert 0.0 <= features["c_d_logit"] <= 1.0


def test_sharper_logits_reduce_logit_dirichlet_uncertainty():
    from skedl.models.logit_dirichlet import LogitDirichletExtractor

    extractor = LogitDirichletExtractor(top_k=3, evidence_transform="shift")

    diffuse = [np.array([2.0, 1.9, 1.8, 0.5], dtype=float)]
    sharp = [np.array([5.0, 2.0, 0.2, -1.0], dtype=float)]

    f_diffuse = extractor.extract_from_logits_steps(diffuse)
    f_sharp = extractor.extract_from_logits_steps(sharp)

    assert f_sharp["eu_mean"] < f_diffuse["eu_mean"]


def test_logit_dirichlet_handles_non_finite_logits():
    from skedl.models.logit_dirichlet import LogitDirichletExtractor

    extractor = LogitDirichletExtractor(top_k=3, evidence_transform="shift")

    logits_steps = [
        np.array([5.0, 2.0, -np.inf, -np.inf], dtype=float),
        np.array([4.0, -np.inf, 1.0, -np.inf], dtype=float),
    ]

    features = extractor.extract_from_logits_steps(logits_steps)

    assert np.isfinite(features["au_mean"])
    assert np.isfinite(features["eu_mean"])
    assert np.isfinite(features["c_d_logit"])


def test_logit_dirichlet_default_top_k_is_ten():
    from skedl.models.logit_dirichlet import LogitDirichletExtractor
    from skedl.pipeline.features import SKEDLFeatureExtractor

    assert LogitDirichletExtractor().top_k == 10
    assert SKEDLFeatureExtractor().top_k_dirichlet == 10
