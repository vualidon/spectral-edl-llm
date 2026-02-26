from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingAdapter(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...
