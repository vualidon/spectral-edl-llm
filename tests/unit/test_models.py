from __future__ import annotations

import torch


def test_edl_head_outputs_positive_evidence():
    from skedl.models.edl_head import EDLHead

    head = EDLHead(input_dim=4, num_classes=3)
    x = torch.randn(2, 4)
    out = head(x)

    assert out["evidence"].shape == (2, 3)
    assert out["alpha"].shape == (2, 3)
    assert torch.all(out["evidence"] >= 0)
    assert torch.all(out["alpha"] > 1.0)


def test_hybrid_fusion_accepts_missing_branch_masks():
    from skedl.models.hybrid_model import HybridConfidenceModel

    model = HybridConfidenceModel(
        handcrafted_dim=4,
        hidden_dim=8,
        num_answer_classes=3,
        use_edl_head=True,
    )

    handcrafted = torch.tensor([[0.2, 0.4, 0.1, 0.9]], dtype=torch.float32)
    hidden_summary = torch.randn(1, 8)
    branch_mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    out = model(
        handcrafted_features=handcrafted,
        branch_mask=branch_mask,
        hidden_summary=hidden_summary,
    )

    assert "confidence_logit" in out
    assert "confidence" in out
    assert out["confidence"].shape == (1, 1)
    assert torch.all(out["confidence"] >= 0.0)
    assert torch.all(out["confidence"] <= 1.0)
