from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skedl.benchmarks.compare_published import compare_published_strict
from skedl.benchmarks.published_baselines import published_baseline_rows
from skedl.benchmarks.runner import BenchmarkRunConfig, run_real_benchmark


SUPPORTED_TASK_MAP: dict[str, tuple[str, str]] = {
    "truthfulqa": ("truthfulqa_generation", "validation"),
    "gsm8k": ("gsm8k", "test"),
    "commonsense_qa": ("commonsense_qa", "validation"),
    "arc_challenge": ("arc_challenge", "validation"),
    "arc_easy": ("arc_easy", "validation"),
    "openbookqa": ("openbookqa", "validation"),
    "sciq": ("sciq", "validation"),
    "race": ("race", "validation"),
}


UNSUPPORTED_TASK_REASONS: dict[str, str] = {
    "aggregate_4bench": "EDTR table reports a four-benchmark aggregate (AIME/GSM8K/CommonsenseQA/S&P500); aggregate reproduction is not implemented yet.",
    "sciq_vs_mmlu": "IB-EDL OOD AUROC setting (SciQ vs MMLU) is not implemented in the current OOD evaluation pipeline.",
    "winogrande_vs_mmlu": "IB-EDL OOD AUROC setting (WinoGrande vs MMLU) is not implemented in the current OOD evaluation pipeline.",
    "hellaswag_vs_mmlu": "IB-EDL OOD AUROC setting (HellaSwag vs MMLU) is not implemented in the current OOD evaluation pipeline.",
}


def _slug(text: str) -> str:
    chars = []
    for ch in text.strip():
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_metric_rows_from_summary(
    *,
    paper_id: str,
    model_paper: str,
    hf_model: str,
    dataset: str,
    split: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "paper_id": paper_id,
            "model_paper": model_paper,
            "hf_model": hf_model,
            "task_id": dataset,
            "split": split,
            "metric": "accuracy",
            "method": "SK-EDL",
            "value": float(summary.get("accuracy", 0.0)),
        }
    )

    conf_metrics = summary.get("confidence_metrics") or {}
    if isinstance(conf_metrics, dict):
        for metric in ["ece", "ece@5", "ece@10", "ece@15", "ece@20", "ece@25", "ece@35", "brier", "nll", "auroc", "auroc_error", "auprc_error", "aurc"]:
            if metric in conf_metrics and conf_metrics[metric] is not None:
                try:
                    value = float(conf_metrics[metric])
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "paper_id": paper_id,
                        "model_paper": model_paper,
                        "hf_model": hf_model,
                        "task_id": dataset,
                        "split": split,
                        "metric": metric,
                        "method": "SK-EDL",
                        "value": value,
                    }
                )
    return rows


def build_paper_protocol_plan(
    *,
    paper_id: str,
    model_paper: str,
    hf_model: str,
    embedding_model: str,
    output_dir: str,
    device: str,
    embedding_device: str,
    num_cots: int,
    cot_batch_size: int,
    temp_mode: str,
    temperature: float,
    temperature_delta: float,
    max_new_tokens: int,
    top_p: float,
    k: int,
    tau: float,
    truthfulqa_scorer: str = "bleurt",
    truthfulqa_bleurt_threshold: float = 0.5,
    truthfulqa_reference_mode: str = "max_correct",
    dtype: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    pub_rows = [r for r in published_baseline_rows() if r["paper_id"] == paper_id and r["model_paper"] == model_paper]
    if not pub_rows:
        raise ValueError(f"no published rows found for paper={paper_id!r}, model_paper={model_paper!r}")

    runs: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in pub_rows:
        by_task.setdefault(str(row.get("task_id")), []).append(row)

    for task_id, task_rows in sorted(by_task.items()):
        metrics = sorted({str(r["metric"]) for r in task_rows})
        table_ids = sorted({str(r["table_id"]) for r in task_rows})
        if task_id in SUPPORTED_TASK_MAP:
            dataset, default_split = SUPPORTED_TASK_MAP[task_id]
            runs.append(
                {
                    "task_id": task_id,
                    "dataset": dataset,
                    "split": default_split,
                    "metrics_required": metrics,
                    "table_ids": table_ids,
                    "supported": True,
                    "reason": None,
                    "llm_model": hf_model,
                    "embedding_model": embedding_model,
                    "device": device,
                    "embedding_device": embedding_device,
                    "dtype": dtype,
                    "num_cots": int(num_cots),
                    "cot_batch_size": int(cot_batch_size),
                    "temp_mode": temp_mode,
                    "temperature": float(temperature),
                    "temperature_delta": float(temperature_delta),
                    "max_new_tokens": int(max_new_tokens),
                    "top_p": float(top_p),
                    "k": int(k),
                    "tau": float(tau),
                    "truthfulqa_scorer": truthfulqa_scorer,
                    "truthfulqa_bleurt_threshold": float(truthfulqa_bleurt_threshold),
                    "truthfulqa_reference_mode": truthfulqa_reference_mode,
                    "limit": None if limit is None else int(limit),
                }
            )
        else:
            runs.append(
                {
                    "task_id": task_id,
                    "dataset": None,
                    "split": None,
                    "metrics_required": metrics,
                    "table_ids": table_ids,
                    "supported": False,
                    "reason": UNSUPPORTED_TASK_REASONS.get(task_id, "Task/protocol not implemented in current pipeline."),
                }
            )

    return {
        "paper_id": paper_id,
        "model_paper": model_paper,
        "hf_model": hf_model,
        "embedding_model": embedding_model,
        "output_dir": output_dir,
        "device": device,
        "embedding_device": embedding_device,
        "num_cots": int(num_cots),
        "cot_batch_size": int(cot_batch_size),
        "temp_mode": temp_mode,
        "temperature": float(temperature),
        "temperature_delta": float(temperature_delta),
        "truthfulqa_scorer": truthfulqa_scorer,
        "truthfulqa_bleurt_threshold": float(truthfulqa_bleurt_threshold),
        "truthfulqa_reference_mode": truthfulqa_reference_mode,
        "runs": runs,
        "strict_compare_default": True,
    }


def run_paper_protocol(
    *,
    paper_id: str,
    model_paper: str,
    hf_model: str,
    embedding_model: str,
    output_dir: str,
    device: str = "auto",
    embedding_device: str = "cpu",
    dtype: str | None = None,
    num_cots: int = 6,
    cot_batch_size: int = 1,
    temp_mode: str = "mixed",
    temperature: float = 0.7,
    temperature_delta: float = 0.2,
    max_new_tokens: int = 1024,
    top_p: float = 0.95,
    k: int = 3,
    tau: float = 5.0,
    truthfulqa_scorer: str = "bleurt",
    truthfulqa_bleurt_threshold: float = 0.5,
    truthfulqa_reference_mode: str = "max_correct",
    limit: int | None = None,
    dry_run: bool = False,
    compare_after_run: bool = True,
) -> dict[str, Any]:
    plan = build_paper_protocol_plan(
        paper_id=paper_id,
        model_paper=model_paper,
        hf_model=hf_model,
        embedding_model=embedding_model,
        output_dir=output_dir,
        device=device,
        embedding_device=embedding_device,
        num_cots=num_cots,
        cot_batch_size=cot_batch_size,
        temp_mode=temp_mode,
        temperature=temperature,
        temperature_delta=temperature_delta,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        k=k,
        tau=tau,
        truthfulqa_scorer=truthfulqa_scorer,
        truthfulqa_bleurt_threshold=truthfulqa_bleurt_threshold,
        truthfulqa_reference_mode=truthfulqa_reference_mode,
        dtype=dtype,
        limit=limit,
    )

    run_dir = Path(output_dir) / paper_id / f"{_slug(model_paper)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "protocol_plan.json", plan)

    if dry_run:
        return {
            "status": "dry_run",
            "run_dir": str(run_dir),
            "plan": plan,
        }

    benchmark_dir = run_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    our_rows: list[dict[str, Any]] = []
    executed_runs: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []

    for item in plan["runs"]:
        if not item.get("supported"):
            skipped_runs.append(item)
            continue
        cfg = BenchmarkRunConfig(
            dataset=str(item["dataset"]),
            split=str(item["split"]),
            limit=item.get("limit"),
            llm_model=hf_model,
            embedding_model=embedding_model,
            device=device,
            embedding_device=embedding_device,
            dtype=dtype,
            num_cots=num_cots,
            cot_batch_size=cot_batch_size,
            temp_mode=temp_mode,
            temperature=temperature,
            temperature_delta=temperature_delta,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            k=k,
            tau=tau,
            output_dir=str(benchmark_dir),
            save_cots=False,
            truthfulqa_scorer=truthfulqa_scorer,
            truthfulqa_bleurt_threshold=float(truthfulqa_bleurt_threshold),
            truthfulqa_reference_mode=truthfulqa_reference_mode,
        )
        out = run_real_benchmark(cfg)
        summary = dict(out["summary"])
        summary["paper_id"] = paper_id
        summary["model_paper"] = model_paper
        summary["task_id"] = str(item["task_id"])
        summaries.append(summary)
        executed_runs.append(
            {
                "task_id": item["task_id"],
                "dataset": item["dataset"],
                "split": item["split"],
                "n": summary.get("n"),
                "summary_file": None,  # benchmark runner already writes canonical summary filename in benchmark_dir
            }
        )
        our_rows.extend(
            _extract_metric_rows_from_summary(
                paper_id=paper_id,
                model_paper=model_paper,
                hf_model=hf_model,
                dataset=str(item["task_id"]),
                split=str(item["split"]),
                summary=summary,
            )
        )

    _write_json(run_dir / "our_rows.json", our_rows)
    _write_json(
        run_dir / "protocol_run.json",
        {
            "paper_id": paper_id,
            "model_paper": model_paper,
            "hf_model": hf_model,
            "run_dir": str(run_dir),
            "executed_runs": executed_runs,
            "skipped_runs": skipped_runs,
            "summaries": summaries,
            "our_rows_path": str(run_dir / "our_rows.json"),
        },
    )

    compare_out = None
    if compare_after_run:
        compare_out = compare_published_strict(
            paper_id=paper_id,
            published_rows=published_baseline_rows(),
            our_rows=our_rows,
            output_dir=str(run_dir),
            bundle_name=f"{paper_id}_{_slug(model_paper)}",
            model_paper=model_paper,
            write_csv=True,
        )

    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "plan": plan,
        "executed_runs": executed_runs,
        "skipped_runs": skipped_runs,
        "our_rows_path": str(run_dir / "our_rows.json"),
        "compare": None if compare_out is None else {"json_path": compare_out["json_path"], "matched": len(compare_out["matched"])},
    }
