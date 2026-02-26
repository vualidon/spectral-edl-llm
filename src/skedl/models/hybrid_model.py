from __future__ import annotations

import torch
from torch import nn

from skedl.core.fusion import LogisticFusion, edl_confidence_from_alpha_torch
from skedl.models.edl_head import EDLHead


class HybridConfidenceModel(nn.Module):
    def __init__(
        self,
        *,
        handcrafted_dim: int,
        hidden_dim: int,
        num_answer_classes: int,
        use_edl_head: bool = True,
    ) -> None:
        super().__init__()
        self.handcrafted_dim = handcrafted_dim
        self.hidden_dim = hidden_dim
        self.num_answer_classes = num_answer_classes
        self.use_edl_head = use_edl_head
        self.edl_head = (
            EDLHead(input_dim=hidden_dim, num_classes=num_answer_classes)
            if use_edl_head
            else None
        )
        self.fusion = LogisticFusion()

    def forward(
        self,
        *,
        handcrafted_features: torch.Tensor,
        branch_mask: torch.Tensor,
        hidden_summary: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if handcrafted_features.ndim != 2:
            raise ValueError("handcrafted_features must be [batch, dim]")
        if branch_mask.ndim != 2:
            raise ValueError("branch_mask must be [batch, mask_dim]")
        if handcrafted_features.shape[0] != branch_mask.shape[0]:
            raise ValueError("batch sizes must match")

        components = [handcrafted_features, branch_mask]
        outputs: dict[str, torch.Tensor] = {}

        if self.use_edl_head and self.edl_head is not None and hidden_summary is not None:
            edl_out = self.edl_head(hidden_summary)
            outputs.update(edl_out)
            c_d_head = edl_confidence_from_alpha_torch(edl_out["alpha"])
            outputs["c_d_head"] = c_d_head
            components.append(c_d_head)

        fused_in = torch.cat(components, dim=1)
        fusion_out = self.fusion(fused_in)
        outputs.update(fusion_out)
        return outputs
