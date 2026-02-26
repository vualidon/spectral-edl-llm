from __future__ import annotations

import numpy as np

from skedl.schemas import CoTSample, GenerationStep


def test_feature_pipeline_computes_sk_edl_features_from_mock_cots():
    from skedl.pipeline.features import SKEDLFeatureExtractor

    samples = []
    embeddings = [
        np.array([1.0, 0.0]),
        np.array([0.95, 0.05]),
        np.array([0.9, 0.1]),
        np.array([0.1, 0.9]),
        np.array([0.05, 0.95]),
        np.array([0.0, 1.0]),
    ]
    for i, emb in enumerate(embeddings):
        logits = [
            np.array([4.0 - 0.1 * i, 2.0, 0.2, -1.0], dtype=float),
            np.array([3.8 - 0.1 * i, 1.9, 0.1, -0.8], dtype=float),
        ]
        steps = [GenerationStep(token_id=j, token_text=str(j), logits=step) for j, step in enumerate(logits)]
        samples.append(CoTSample(cot_text=f"cot-{i}", answer_text="42", embedding=emb, steps=steps))

    extractor = SKEDLFeatureExtractor(k=2, tau=5.0, top_k_dirichlet=3)
    result = extractor.extract(samples)

    assert "kernel_entropy" in result.features
    assert "lambda2" in result.features
    assert "connectivity_risk" in result.features
    assert "au_mean" in result.features
    assert "eu_mean" in result.features
    assert "num_cots" in result.features
    assert result.features["num_cots"] == 6.0
    assert result.features["kernel_entropy"] >= 0.0
    assert 0.0 <= result.features["connectivity_risk"] <= 1.0
