from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from skedl.schemas import CoTSample


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class SampleCoTRequest:
    prompt: str
    num_cots: int = 6
    cot_batch_size: int = 1
    logit_capture_top_k: int | None = 10
    max_new_tokens: int = 1024
    temperatures: list[float] | None = None
    top_p: float = 0.95
    capture_hidden_states: bool = False
    progress_callback: ProgressCallback | None = None


class LocalLLMAdapter(Protocol):
    def sample_cots(self, request: SampleCoTRequest) -> list[CoTSample]:
        ...
