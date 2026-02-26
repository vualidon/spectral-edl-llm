from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def select_torch_device(device: str = "auto") -> torch.device:
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _LogisticRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass(slots=True)
class FusionTrainer:
    device: str = "auto"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3

    def fit_logistic(self, features: np.ndarray, labels: np.ndarray) -> dict[str, float | nn.Module]:
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(labels), dtype=torch.float32).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("features must be 2D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and labels must have same length")

        torch_device = select_torch_device(self.device)
        model = _LogisticRegressor(input_dim=x.shape[1]).to(torch_device)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        last_loss = 0.0
        model.train()
        for _ in range(max(1, self.epochs)):
            for xb, yb in loader:
                xb = xb.to(torch_device)
                yb = yb.to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

        return {"train_loss": last_loss, "model": model, "device": torch_device.type}
