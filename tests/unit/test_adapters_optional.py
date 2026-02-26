from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_transformers_adapter_raises_clear_error_when_dependency_missing(monkeypatch):
    from skedl.adapters.llm.transformers_local import TransformersLocalLLM

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="pip install .*hf"):
        TransformersLocalLLM(model_name="dummy-local-model")


def test_sentence_transformer_embedder_initializes_with_slots(monkeypatch):
    import types

    import skedl.adapters.emb.sentence_transformers as emb_mod

    class FakeModel:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.model_name = model_name
            self.device = device

        def encode(self, texts, normalize_embeddings=False):
            assert normalize_embeddings is False
            return [[1.0, 0.0] for _ in texts]

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeModel)

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "sentence_transformers":
            return fake_module
        return real_import_module(name, package)

    monkeypatch.setattr(emb_mod.importlib, "import_module", fake_import_module)

    embedder = emb_mod.SentenceTransformerEmbedder(model_name="fake-embedder", device="cpu")
    vectors = embedder.encode(["a", "b"])

    assert vectors.shape == (2, 2)


def test_transformers_logits_capture_helper_keeps_topk_only():
    import torch

    from skedl.adapters.llm.transformers_local import _capture_logits_snapshot

    row = torch.tensor([0.1, -2.0, 3.5, 1.2, 7.0, -0.4], dtype=torch.float32)

    top3 = _capture_logits_snapshot(row, top_k_capture=3)
    full = _capture_logits_snapshot(row, top_k_capture=None)

    assert top3.shape == (3,)
    assert full.shape == (6,)
    assert np.allclose(np.sort(top3), np.sort(np.array([7.0, 3.5, 1.2], dtype=np.float32)))


def test_transformers_resolve_device_supports_cuda(monkeypatch):
    import skedl.adapters.llm.transformers_local as llm_mod

    monkeypatch.setattr(llm_mod.torch.cuda, "is_available", lambda: True)
    device = llm_mod._resolve_device("cuda")

    assert device.type == "cuda"
