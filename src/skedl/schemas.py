from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ArrayLike = np.ndarray


@dataclass(slots=True)
class GenerationStep:
    token_id: int
    token_text: str
    logits: ArrayLike | None = None


@dataclass(slots=True)
class CoTSample:
    cot_text: str
    answer_text: str | None = None
    embedding: ArrayLike | None = None
    steps: list[GenerationStep] = field(default_factory=list)
    hidden_states: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TemperatureScheduleConfig:
    num_cots: int = 6
    mode: str = "fixed"
    temperature: float = 0.7
    delta: float = 0.2
    min_temperature: float = 0.0
    max_temperature: float = 2.0


@dataclass(slots=True)
class SKEDLFeatureResult:
    features: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)
