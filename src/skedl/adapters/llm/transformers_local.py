from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np
import torch

from skedl.adapters.llm.base import LocalLLMAdapter, SampleCoTRequest
from skedl.pipeline.sampling import build_temperature_schedule
from skedl.schemas import CoTSample, GenerationStep


def _resolve_device(preferred: str = "auto") -> torch.device:
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but unavailable")
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but unavailable")
    if preferred == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _contiguous_temperature_groups(temperatures: list[float]) -> list[tuple[int, int, float]]:
    if not temperatures:
        return []
    groups: list[tuple[int, int, float]] = []
    start = 0
    current = float(temperatures[0])
    for idx in range(1, len(temperatures)):
        temp = float(temperatures[idx])
        if temp != current:
            groups.append((start, idx, current))
            start = idx
            current = temp
    groups.append((start, len(temperatures), current))
    return groups


def _capture_logits_snapshot(logits_row: torch.Tensor, top_k_capture: int | None) -> np.ndarray:
    row = logits_row.detach().float()
    if row.ndim != 1:
        raise ValueError("logits_row must be 1D")
    if top_k_capture is None or int(top_k_capture) <= 0:
        return row.cpu().numpy()
    k = min(int(top_k_capture), int(row.shape[-1]))
    if k <= 0:
        return row.cpu().numpy()
    values = torch.topk(row, k=k, dim=-1, largest=True, sorted=True).values
    return values.cpu().numpy()


@dataclass(slots=True)
class TransformersLocalLLM(LocalLLMAdapter):
    model_name: str
    device: str = "auto"
    dtype: str | None = None
    _transformers: Any = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _tokenizer: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is required for local inference. Install optional deps: pip install .[hf]"
            ) from exc

        self._transformers = transformers
        self._device = _resolve_device(self.device)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        torch_dtype = None
        if self.dtype is not None:
            torch_dtype = getattr(torch, self.dtype)
        model_kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        ).to(self._device)
        self._model.eval()

    def _decode_new_tokens(self, input_ids: torch.Tensor, full_sequence: torch.Tensor) -> str:
        new_ids = full_sequence[input_ids.shape[-1] :]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def _truncate_generated_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        ids = generated_ids
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id
        cutoff = len(ids)
        for pos, token_id in enumerate(ids.tolist()):
            if eos_id is not None and int(token_id) == int(eos_id):
                cutoff = pos + 1
                break
            if pad_id is not None and int(token_id) == int(pad_id):
                cutoff = pos
                break
        return ids[:cutoff]

    @torch.inference_mode()
    def sample_cots(self, request: SampleCoTRequest) -> list[CoTSample]:
        progress_cb = request.progress_callback

        def _emit_progress(event: dict[str, Any]) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(event)
            except Exception:
                # Progress reporting must never break inference.
                return

        if request.temperatures is None:
            temperatures = build_temperature_schedule(num_cots=request.num_cots)
        else:
            temperatures = request.temperatures
        if len(temperatures) != request.num_cots:
            raise ValueError("temperatures length must equal num_cots")
        cot_batch_size = max(1, int(request.cot_batch_size))

        encoded = self._tokenizer(request.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        samples: list[CoTSample | None] = [None] * request.num_cots
        sampling_t0 = time()
        _emit_progress(
            {
                "event": "sampling_start",
                "num_cots": request.num_cots,
                "device": str(self._device),
                "cot_batch_size": cot_batch_size,
            }
        )
        groups = _contiguous_temperature_groups([float(t) for t in temperatures])
        for group_start, group_end, group_temp in groups:
            for batch_start in range(group_start, group_end, cot_batch_size):
                batch_end = min(batch_start + cot_batch_size, group_end)
                batch_indices = list(range(batch_start, batch_end))
                batch_t0 = time()
                for cot_index in batch_indices:
                    _emit_progress(
                        {
                            "event": "cot_start",
                            "cot_index": cot_index,
                            "num_cots": request.num_cots,
                            "temperature": float(group_temp),
                        }
                    )

                batch_n = len(batch_indices)
                if batch_n == 1:
                    batch_input_ids = input_ids
                    batch_attention_mask = attention_mask
                else:
                    batch_input_ids = input_ids.repeat(batch_n, 1)
                    batch_attention_mask = None if attention_mask is None else attention_mask.repeat(batch_n, 1)

                generate_out = self._model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    do_sample=True,
                    temperature=max(float(group_temp), 1e-4),
                    top_p=request.top_p,
                    max_new_tokens=request.max_new_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                scores_list = list(generate_out.scores)
                for row_idx, cot_index in enumerate(batch_indices):
                    cot_t0 = batch_t0
                    sequence = generate_out.sequences[row_idx]
                    full_generated = sequence[input_ids.shape[-1] :]
                    generated_ids = self._truncate_generated_ids(full_generated)
                    cot_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

                    steps: list[GenerationStep] = []
                    step_count = min(int(generated_ids.shape[0]), len(scores_list))
                    for step_pos in range(step_count):
                        token_id = int(generated_ids[step_pos].item())
                        logits = _capture_logits_snapshot(
                            scores_list[step_pos][row_idx],
                            request.logit_capture_top_k,
                        )
                        token_text = self._tokenizer.decode([token_id], skip_special_tokens=False)
                        steps.append(GenerationStep(token_id=token_id, token_text=token_text, logits=logits))

                    hidden_summary = None
                    if request.capture_hidden_states:
                        full = sequence.unsqueeze(0).to(self._device)
                        forward_out = self._model(full, output_hidden_states=True, use_cache=False)
                        last_hidden = forward_out.hidden_states[-1][0]  # [seq, dim]
                        generated_hidden = last_hidden[input_ids.shape[-1] :, :]
                        if generated_hidden.numel() > 0:
                            hidden_summary = generated_hidden.mean(dim=0).detach().float().cpu().numpy()

                    _emit_progress(
                        {
                            "event": "cot_done",
                            "cot_index": cot_index,
                            "num_cots": request.num_cots,
                            "temperature": float(group_temp),
                            "generated_tokens": int(generated_ids.shape[0]),
                            "elapsed_sec": float(time() - cot_t0),
                            "batch_size": batch_n,
                        }
                    )
                    samples[cot_index] = CoTSample(
                        cot_text=cot_text,
                        steps=steps,
                        hidden_states=hidden_summary,
                        metadata={"temperature": float(group_temp)},
                    )
        _emit_progress(
            {
                "event": "sampling_done",
                "num_cots": request.num_cots,
                "elapsed_sec": float(time() - sampling_t0),
            }
        )
        if any(sample is None for sample in samples):
            raise RuntimeError("internal error: missing generated CoT sample")
        return [sample for sample in samples if sample is not None]
