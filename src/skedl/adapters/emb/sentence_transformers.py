from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            st = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is required for local embeddings. Install optional deps: pip install .[hf]"
            ) from exc
        self._model = st.SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(texts, normalize_embeddings=False)
        return np.asarray(embeddings, dtype=float)
