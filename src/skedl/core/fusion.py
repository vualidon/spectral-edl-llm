from __future__ import annotations

import torch
from torch import nn


def edl_confidence_from_alpha_torch(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.ndim != 2:
        raise ValueError("alpha must be [batch, classes]")
    s = alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
    probs = alpha / s
    k = alpha.shape[1]
    u_d = float(k) / s
    conf = (1.0 - u_d) * probs.max(dim=1, keepdim=True).values
    return conf.clamp(0.0, 1.0)


class LogisticFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.LazyLinear(1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logit = self.linear(features)
        conf = torch.sigmoid(logit)
        return {"confidence_logit": logit, "confidence": conf}
