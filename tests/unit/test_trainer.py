from __future__ import annotations

import numpy as np
import torch


def test_device_selector_prefers_mps_when_available(monkeypatch):
    from skedl.pipeline.trainer import select_torch_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device = select_torch_device("auto")

    assert device.type == "mps"


def test_fusion_trainer_runs_single_epoch():
    from skedl.pipeline.trainer import FusionTrainer

    x = np.array(
        [
            [0.1, 0.2, 0.8],
            [0.2, 0.1, 0.7],
            [0.8, 0.7, 0.2],
            [0.9, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    y = np.array([1, 1, 0, 0], dtype=np.float32)

    trainer = FusionTrainer(device="cpu", epochs=1, batch_size=2, lr=1e-2)
    result = trainer.fit_logistic(x, y)

    assert "train_loss" in result
    assert result["train_loss"] >= 0.0
