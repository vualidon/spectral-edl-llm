from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from skedl.adapters.emb.sentence_transformers import SentenceTransformerEmbedder
from skedl.adapters.llm.base import SampleCoTRequest
from skedl.adapters.llm.transformers_local import TransformersLocalLLM
from skedl.benchmarks.tasks import (
    arc_is_correct,
    build_arc_prompt,
    build_boolq_prompt,
    build_commonsenseqa_prompt,
    build_gsm8k_prompt,
    build_piqa_prompt,
    build_race_prompt,
    build_sciq_choice_map,
    build_sciq_prompt,
    build_truthfulqa_prompt,
    boolq_is_correct,
    csqa_is_correct,
    extract_arc_answer,
    extract_boolq_answer,
    extract_commonsenseqa_answer,
    extract_gsm8k_answer,
    extract_gsm8k_gold_answer,
    extract_piqa_answer,
    extract_truthfulqa_answer,
    gsm8k_is_correct,
    majority_vote,
    normalize_mcq_label,
    piqa_is_correct,
)
from skedl.benchmarks.truthfulqa import make_truthfulqa_scorer, score_truthfulqa_prediction
from skedl.pipeline.evaluator import evaluate_binary_confidence
from skedl.pipeline.features import SKEDLFeatureExtractor
from skedl.pipeline.sampling import build_temperature_schedule


@dataclass(slots=True)
class BenchmarkRunConfig:
    dataset: str
    split: str
    limit: int | None
    llm_model: str
    embedding_model: str
    device: str = "auto"
    embedding_device: str = "cpu"
    dtype: str | None = None
    num_cots: int = 6
    cot_batch_size: int = 1
    temp_mode: str = "mixed"
    temperature: float = 0.7
    temperature_delta: float = 0.2
    max_new_tokens: int = 1024
    top_p: float = 0.95
    k: int = 3
    tau: float = 5.0
    output_dir: str | None = None
    save_cots: bool = False
    truthfulqa_scorer: str = "bleurt"
    truthfulqa_bleurt_threshold: float = 0.5
    truthfulqa_reference_mode: str = "max_correct"
    show_progress: bool = True
    progress_every: int = 1
    incremental_write: bool = True


def _load_dataset_rows(dataset: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    split_spec = split if limit is None else f"{split}[:{limit}]"
    if dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split_spec)
    elif dataset == "commonsense_qa":
        ds = load_dataset("commonsense_qa", split=split_spec)
    elif dataset == "piqa":
        ds = load_dataset("piqa", split=split_spec)
    elif dataset == "boolq":
        ds = load_dataset("super_glue", "boolq", split=split_spec)
    elif dataset == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split=split_spec)
    elif dataset == "arc_easy":
        ds = load_dataset("ai2_arc", "ARC-Easy", split=split_spec)
    elif dataset == "openbookqa":
        ds = load_dataset("allenai/openbookqa", "main", split=split_spec)
    elif dataset == "sciq":
        ds = load_dataset("sciq", split=split_spec)
    elif dataset == "race":
        ds = load_dataset("race", "all", split=split_spec)
    elif dataset == "truthfulqa_generation":
        ds = load_dataset("truthful_qa", "generation", split=split_spec)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return [dict(row) for row in ds]


def _build_prompt_and_gold(dataset: str, row: dict[str, Any]) -> tuple[str, str | None]:
    if dataset == "gsm8k":
        return build_gsm8k_prompt(row["question"]), extract_gsm8k_gold_answer(row["answer"])
    if dataset == "commonsense_qa":
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choice_map = {label: text for label, text in zip(labels, texts)}
        return build_commonsenseqa_prompt(row["question"], choice_map), normalize_mcq_label(row.get("answerKey"))
    if dataset == "piqa":
        gold = row.get("label")
        gold_label = "A" if str(gold) == "0" else "B" if str(gold) == "1" else None
        return build_piqa_prompt(row["goal"], row["sol1"], row["sol2"]), gold_label
    if dataset == "boolq":
        gold = row.get("label")
        gold_text = None if gold is None else ("yes" if bool(gold) else "no")
        return build_boolq_prompt(row["question"], row["passage"]), gold_text
    if dataset == "arc_challenge":
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choice_map = {normalize_mcq_label(label) or str(label): text for label, text in zip(labels, texts)}
        # keep A-E ordering for prompt readability
        choice_map = {k: choice_map[k] for k in ["A", "B", "C", "D", "E"] if k in choice_map}
        return build_arc_prompt(row["question"], choice_map), normalize_mcq_label(row.get("answerKey"))
    if dataset == "arc_easy":
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choice_map = {normalize_mcq_label(label) or str(label): text for label, text in zip(labels, texts)}
        choice_map = {k: choice_map[k] for k in ["A", "B", "C", "D", "E"] if k in choice_map}
        return build_arc_prompt(row["question"], choice_map), normalize_mcq_label(row.get("answerKey"))
    if dataset == "openbookqa":
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choice_map = {normalize_mcq_label(label) or str(label): text for label, text in zip(labels, texts)}
        choice_map = {k: choice_map[k] for k in ["A", "B", "C", "D", "E"] if k in choice_map}
        return build_arc_prompt(row["question_stem"], choice_map), normalize_mcq_label(row.get("answerKey"))
    if dataset == "sciq":
        choice_map, gold_label = build_sciq_choice_map(
            question=row["question"],
            correct_answer=row["correct_answer"],
            distractors=[row["distractor1"], row["distractor2"], row["distractor3"]],
        )
        return build_sciq_prompt(row["question"], str(row.get("support", "")), choice_map), gold_label
    if dataset == "race":
        options = list(row.get("options") or [])
        choice_map = {label: str(text) for label, text in zip(["A", "B", "C", "D"], options)}
        return build_race_prompt(row["article"], row["question"], choice_map), normalize_mcq_label(row.get("answer"))
    if dataset == "truthfulqa_generation":
        return build_truthfulqa_prompt(row["question"]), str(row.get("best_answer") or "").strip() or None
    raise ValueError(f"unsupported dataset: {dataset}")


def _extract_pred_answer(dataset: str, cot_text: str) -> str | None:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(cot_text)
    if dataset == "commonsense_qa":
        return extract_commonsenseqa_answer(cot_text)
    if dataset == "piqa":
        return extract_piqa_answer(cot_text)
    if dataset == "boolq":
        return extract_boolq_answer(cot_text)
    if dataset == "arc_challenge":
        return extract_arc_answer(cot_text)
    if dataset == "arc_easy":
        return extract_arc_answer(cot_text)
    if dataset == "openbookqa":
        return extract_arc_answer(cot_text)
    if dataset == "sciq":
        return extract_arc_answer(cot_text)
    if dataset == "race":
        return extract_arc_answer(cot_text)
    if dataset == "truthfulqa_generation":
        return extract_truthfulqa_answer(cot_text)
    raise ValueError(f"unsupported dataset: {dataset}")


def _is_correct(dataset: str, pred: str | None, gold: str | None) -> bool:
    if dataset == "gsm8k":
        return gsm8k_is_correct(pred, gold)
    if dataset == "commonsense_qa":
        return csqa_is_correct(pred, gold)
    if dataset == "piqa":
        return piqa_is_correct(pred, gold)
    if dataset == "boolq":
        return boolq_is_correct(pred, gold)
    if dataset == "arc_challenge":
        return arc_is_correct(pred, gold)
    if dataset == "arc_easy":
        return arc_is_correct(pred, gold)
    if dataset == "openbookqa":
        return arc_is_correct(pred, gold)
    if dataset == "sciq":
        return arc_is_correct(pred, gold)
    if dataset == "race":
        return arc_is_correct(pred, gold)
    raise ValueError(f"unsupported dataset: {dataset}")


def _proxy_confidence(features: dict[str, float]) -> float:
    return float(
        np.clip(
            np.exp(
                -(
                    0.5 * float(features.get("connectivity_risk", 0.0))
                    + 0.25 * float(features.get("kernel_entropy", 0.0))
                    + 0.25 * float(features.get("eu_mean", 0.0))
                )
            ),
            0.0,
            1.0,
        )
    )


def _mps_memory_snapshot() -> dict[str, float] | None:
    try:
        import torch
    except Exception:
        return None
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        return None
    snap: dict[str, float] = {}
    for attr, key in [
        ("current_allocated_memory", "mps_current_allocated_gb"),
        ("driver_allocated_memory", "mps_driver_allocated_gb"),
        ("recommended_max_memory", "mps_recommended_max_gb"),
    ]:
        fn = getattr(torch.mps, attr, None)
        if callable(fn):
            try:
                snap[key] = float(fn()) / (1024.0**3)
            except Exception:
                continue
    return snap or None


def _progress_paths(out_dir: Path, dataset: str, split: str) -> tuple[Path, Path]:
    base = f"{dataset}_{split}"
    return out_dir / f"{base}.partial.records.jsonl", out_dir / f"{base}.progress.json"


def _format_benchmark_progress_line(
    *,
    dataset: str,
    split: str,
    completed: int,
    total: int,
    elapsed_sec: float,
    mean_sec_per_example: float,
    eta_sec: float | None,
    accuracy_so_far: float,
    mean_proxy_confidence: float,
) -> str:
    pct = (100.0 * completed / total) if total > 0 else 0.0
    eta_str = "?" if eta_sec is None else f"{eta_sec:.1f}s"
    return (
        f"[benchmark-progress] dataset={dataset} split={split} "
        f"{completed}/{total} ({pct:.1f}%) elapsed={elapsed_sec:.1f}s "
        f"avg={mean_sec_per_example:.2f}s/ex eta={eta_str} "
        f"acc_so_far={accuracy_so_far:.4f} mean_conf={mean_proxy_confidence:.4f}"
    )


def run_real_benchmark(config: BenchmarkRunConfig) -> dict[str, Any]:
    rows = _load_dataset_rows(config.dataset, config.split, config.limit)
    total_rows = len(rows)
    llm = TransformersLocalLLM(model_name=config.llm_model, device=config.device, dtype=config.dtype)
    embedder = SentenceTransformerEmbedder(model_name=config.embedding_model, device=config.embedding_device)
    feature_extractor = SKEDLFeatureExtractor(k=config.k, tau=config.tau)
    truthfulqa_scorer = None
    if config.dataset == "truthfulqa_generation":
        truthfulqa_scorer = make_truthfulqa_scorer(config.truthfulqa_scorer)
    temperatures = build_temperature_schedule(
        num_cots=config.num_cots,
        mode=config.temp_mode,
        temperature=config.temperature,
        delta=config.temperature_delta,
    )

    out_dir: Path | None = None
    partial_records_path: Path | None = None
    progress_json_path: Path | None = None
    partial_records_fp = None
    if config.output_dir:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if config.incremental_write:
            partial_records_path, progress_json_path = _progress_paths(out_dir, config.dataset, config.split)
            partial_records_path.write_text("", encoding="utf-8")
            partial_records_fp = partial_records_path.open("a", encoding="utf-8")

    if config.show_progress:
        init_line = (
            f"[benchmark-start] dataset={config.dataset} split={config.split} "
            f"n={total_rows} llm={config.llm_model} emb={config.embedding_model} "
            f"num_cots={config.num_cots} cot_batch_size={config.cot_batch_size} temp_mode={config.temp_mode}"
        )
        print(init_line, flush=True)
        mps_snap = _mps_memory_snapshot()
        if mps_snap:
            parts = " ".join(f"{k}={v:.2f}" for k, v in sorted(mps_snap.items()))
            print(f"[benchmark-start] {parts}", flush=True)

    records: list[dict[str, Any]] = []
    t0 = time()
    try:
        for idx, row in enumerate(rows):
            row_t0 = time()
            prompt, gold_answer = _build_prompt_and_gold(config.dataset, row)
            example_index = idx + 1
            if config.show_progress:
                print(
                    (
                        f"[benchmark-example-start] dataset={config.dataset} split={config.split} "
                        f"example={example_index}/{total_rows} row_index={idx}"
                    ),
                    flush=True,
                )

            if progress_json_path is not None:
                in_flight_payload: dict[str, Any] = {
                    "dataset": config.dataset,
                    "split": config.split,
                    "completed": idx,
                    "total": total_rows,
                    "stage": "sampling_cots",
                    "current_example_index": example_index,
                    "current_row_index": idx,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                mps_snap_live = _mps_memory_snapshot()
                if mps_snap_live:
                    in_flight_payload["runtime"] = mps_snap_live
                progress_json_path.write_text(
                    json.dumps(in_flight_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            def _llm_progress(event: dict[str, Any]) -> None:
                event_name = str(event.get("event", "unknown"))
                cot_num = event.get("cot_index")
                cot_total = event.get("num_cots", config.num_cots)
                if config.show_progress:
                    if event_name == "sampling_start":
                        print(
                            (
                                f"[cot-progress] dataset={config.dataset} split={config.split} "
                                f"example={example_index}/{total_rows} sampling_start num_cots={cot_total}"
                            ),
                            flush=True,
                        )
                    elif event_name == "cot_start":
                        cot_disp = "?" if cot_num is None else int(cot_num) + 1
                        temp = event.get("temperature")
                        temp_part = "" if temp is None else f" temp={float(temp):.2f}"
                        print(
                            (
                                f"[cot-progress] dataset={config.dataset} split={config.split} "
                                f"example={example_index}/{total_rows} cot={cot_disp}/{cot_total} start{temp_part}"
                            ),
                            flush=True,
                        )
                    elif event_name == "cot_done":
                        cot_disp = "?" if cot_num is None else int(cot_num) + 1
                        toks = event.get("generated_tokens")
                        elapsed_sec = event.get("elapsed_sec")
                        temp = event.get("temperature")
                        parts = [
                            f"[cot-progress] dataset={config.dataset} split={config.split}",
                            f"example={example_index}/{total_rows}",
                            f"cot={cot_disp}/{cot_total}",
                            "done",
                        ]
                        if temp is not None:
                            parts.append(f"temp={float(temp):.2f}")
                        if toks is not None:
                            parts.append(f"tokens={int(toks)}")
                        if elapsed_sec is not None:
                            parts.append(f"elapsed={float(elapsed_sec):.2f}s")
                        print(" ".join(parts), flush=True)
                    elif event_name == "sampling_done":
                        elapsed_sec = event.get("elapsed_sec")
                        elapsed_part = "" if elapsed_sec is None else f" elapsed={float(elapsed_sec):.2f}s"
                        print(
                            (
                                f"[cot-progress] dataset={config.dataset} split={config.split} "
                                f"example={example_index}/{total_rows} sampling_done{elapsed_part}"
                            ),
                            flush=True,
                        )

                if progress_json_path is not None:
                    progress_event_payload: dict[str, Any] = {
                        "dataset": config.dataset,
                        "split": config.split,
                        "completed": idx,
                        "total": total_rows,
                        "stage": "sampling_cots",
                        "current_example_index": example_index,
                        "current_row_index": idx,
                        "sampling_event": event_name,
                        "sampling_event_payload": {
                            k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                            for k, v in event.items()
                            if k != "event"
                        },
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    mps_snap_live = _mps_memory_snapshot()
                    if mps_snap_live:
                        progress_event_payload["runtime"] = mps_snap_live
                    progress_json_path.write_text(
                        json.dumps(progress_event_payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

            request = SampleCoTRequest(
                prompt=prompt,
                num_cots=config.num_cots,
                cot_batch_size=config.cot_batch_size,
                max_new_tokens=config.max_new_tokens,
                temperatures=temperatures,
                top_p=config.top_p,
                capture_hidden_states=False,
                progress_callback=_llm_progress,
            )
            samples = llm.sample_cots(request)
            cot_texts = [s.cot_text for s in samples]
            embeddings = embedder.encode(cot_texts)
            for sample, emb, temp in zip(samples, embeddings, temperatures):
                sample.embedding = np.asarray(emb, dtype=float)
                sample.metadata["temperature"] = float(temp)

            parsed_answers = [_extract_pred_answer(config.dataset, s.cot_text) for s in samples]
            pred_answer = majority_vote(parsed_answers)
            truth_eval: dict[str, Any] | None = None
            if config.dataset == "truthfulqa_generation":
                if pred_answer is None:
                    pred_answer = next((a for a in parsed_answers if a), None)
                truth_eval = score_truthfulqa_prediction(
                    prediction=pred_answer or "",
                    row=row,
                    scorer=truthfulqa_scorer,
                    threshold=config.truthfulqa_bleurt_threshold,
                    reference_mode=config.truthfulqa_reference_mode,
                )
                is_correct = bool(truth_eval["label"])
            else:
                is_correct = _is_correct(config.dataset, pred_answer, gold_answer)
            feature_result = feature_extractor.extract(samples)
            proxy_conf = _proxy_confidence(feature_result.features)

            record: dict[str, Any] = {
                "index": idx,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "is_correct": bool(is_correct),
                "proxy_confidence": proxy_conf,
                "features": {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in feature_result.features.items()
                },
            }
            if config.dataset == "commonsense_qa":
                record["question_id"] = row.get("id")
            elif config.dataset == "arc_challenge":
                record["question_id"] = row.get("id")
            elif config.dataset == "arc_easy":
                record["question_id"] = row.get("id")
            elif config.dataset == "openbookqa":
                record["question_id"] = row.get("id")
            elif config.dataset == "race":
                record["question_id"] = row.get("example_id")
            elif config.dataset == "truthfulqa_generation":
                record["question"] = row.get("question")
                if truth_eval is not None:
                    record["truthfulqa_score"] = float(truth_eval["score"])
                    record["truthfulqa_label_threshold"] = float(truth_eval["threshold"])
                    record["truthfulqa_reference_mode"] = str(truth_eval["reference_mode"])
                    record["truthfulqa_scorer"] = str(truth_eval["scorer"])
            if config.save_cots:
                record["cots"] = cot_texts
                record["cot_answers"] = parsed_answers
            records.append(record)

            if partial_records_fp is not None:
                partial_records_fp.write(json.dumps(record) + "\n")
                partial_records_fp.flush()
                try:
                    os.fsync(partial_records_fp.fileno())
                except OSError:
                    pass

            completed = idx + 1
            elapsed = time() - t0
            mean_sec = elapsed / completed if completed > 0 else 0.0
            eta = mean_sec * (total_rows - completed) if total_rows > completed else 0.0
            labels_so_far = np.asarray([1 if r["is_correct"] else 0 for r in records], dtype=float)
            confs_so_far = np.asarray([float(r["proxy_confidence"]) for r in records], dtype=float)
            accuracy_so_far = float(labels_so_far.mean()) if labels_so_far.size else 0.0
            mean_conf_so_far = float(confs_so_far.mean()) if confs_so_far.size else 0.0

            if progress_json_path is not None:
                progress_payload: dict[str, Any] = {
                    "dataset": config.dataset,
                    "split": config.split,
                    "completed": completed,
                    "total": total_rows,
                    "elapsed_sec": elapsed,
                    "mean_sec_per_example": mean_sec,
                    "eta_sec": None if completed >= total_rows else eta,
                    "accuracy_so_far": accuracy_so_far,
                    "mean_proxy_confidence": mean_conf_so_far,
                    "last_index": idx,
                    "last_example_sec": time() - row_t0,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                mps_snap_live = _mps_memory_snapshot()
                if mps_snap_live:
                    progress_payload["runtime"] = mps_snap_live
                progress_json_path.write_text(
                    json.dumps(progress_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            should_print = (
                config.show_progress
                and (
                    completed == 1
                    or completed == total_rows
                    or (int(config.progress_every) > 0 and completed % int(config.progress_every) == 0)
                )
            )
            if should_print:
                print(
                    _format_benchmark_progress_line(
                        dataset=config.dataset,
                        split=config.split,
                        completed=completed,
                        total=total_rows,
                        elapsed_sec=elapsed,
                        mean_sec_per_example=mean_sec,
                        eta_sec=None if completed >= total_rows else eta,
                        accuracy_so_far=accuracy_so_far,
                        mean_proxy_confidence=mean_conf_so_far,
                    ),
                    flush=True,
                )
    finally:
        if partial_records_fp is not None:
            partial_records_fp.close()

    elapsed = time() - t0

    labels = np.asarray([1 if r["is_correct"] else 0 for r in records], dtype=int)
    confs = np.asarray([r["proxy_confidence"] for r in records], dtype=float)
    summary: dict[str, Any] = {
        "dataset": config.dataset,
        "split": config.split,
        "n": len(records),
        "accuracy": float(np.mean(labels)) if len(labels) else 0.0,
        "mean_proxy_confidence": float(np.mean(confs)) if len(confs) else 0.0,
        "elapsed_sec": elapsed,
        "llm_model": config.llm_model,
        "embedding_model": config.embedding_model,
        "num_cots": config.num_cots,
        "temp_mode": config.temp_mode,
    }
    if config.dataset == "truthfulqa_generation":
        summary["truthfulqa_scorer"] = config.truthfulqa_scorer
        summary["truthfulqa_bleurt_threshold"] = float(config.truthfulqa_bleurt_threshold)
        summary["truthfulqa_reference_mode"] = config.truthfulqa_reference_mode
    if len(records) > 0 and len(np.unique(labels)) > 1:
        summary["confidence_metrics"] = evaluate_binary_confidence(confs, labels)
    else:
        summary["confidence_metrics"] = None

    if out_dir is not None:
        tag = f"{config.dataset}_{config.split}_n{len(records)}"
        (out_dir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (out_dir / f"{tag}.records.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    return {"summary": summary, "records": records}
