from __future__ import annotations

import torch
from torch import nn


class EDLHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = self.softplus(self.linear(x))
        alpha = evidence + 1.0
        return {"evidence": evidence, "alpha": alpha}
