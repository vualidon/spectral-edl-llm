from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


LOGU_URL = "https://arxiv.org/pdf/2502.00290v2"
IB_EDL_URL = "https://arxiv.org/pdf/2502.06351"
EDTR_URL = "https://arxiv.org/pdf/2511.06437"


def _metric_direction(metric: str) -> str:
    m = metric.lower()
    if m.startswith("ece@"):
        return "lower"
    if m in {"accuracy", "acc", "auroc", "auroc_correct", "auroc_error", "auprc_error"}:
        return "higher"
    if m in {"ece", "ece@10", "ece@15", "ece@25", "ece@35", "brier", "nll", "aurc", "reliability_score"}:
        return "lower"
    return "unknown"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _base_row(
    *,
    paper_id: str,
    paper_label: str,
    source_url: str,
    table_id: str,
    method: str,
    model_paper: str,
    metric: str,
    value: float,
    value_std: float | None = None,
    task_id: str | None = None,
    task_label: str | None = None,
    split: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "paper_label": paper_label,
        "source_url": source_url,
        "table_id": table_id,
        "method": method,
        "model_paper": model_paper,
        "task_id": task_id,
        "task_label": task_label,
        "split": split,
        "metric": metric,
        "metric_direction": _metric_direction(metric),
        "value": float(value),
        "value_std": None if value_std is None else float(value_std),
        "notes": notes,
    }


def _rows_logu() -> list[dict[str, Any]]:
    models = ["LLaMA2-7B", "LLaMA2-13B", "LLaMA2-70B", "LLaMA3-8B", "LLaMA3-70B"]
    methods = {
        "Average": [(0.720, 0.022), (0.749, 0.015), (0.763, 0.005), (0.731, 0.018), (0.834, 0.006)],
        "Verbalized": [(0.553, 0.017), (0.518, 0.026), (0.568, 0.010), (0.522, 0.022), (0.535, 0.014)],
        "P(ik)": [(0.599, 0.024), (0.672, 0.019), (0.690, 0.006), (0.623, 0.023), (0.747, 0.006)],
        "LeS": [(0.670, 0.027), (0.668, 0.030), (0.697, 0.010), (0.607, 0.034), (0.649, 0.023)],
        "DSE": [(0.709, 0.018), (0.726, 0.025), (0.747, 0.007), (0.685, 0.021), (0.814, 0.007)],
        "SE": [(0.714, 0.020), (0.705, 0.015), (0.732, 0.010), (0.732, 0.023), (0.819, 0.007)],
        "LogU": [(0.801, 0.016), (0.811, 0.014), (0.820, 0.006), (0.839, 0.015), (0.888, 0.006)],
    }
    rows: list[dict[str, Any]] = []
    for method, values in methods.items():
        for model_paper, (val, std) in zip(models, values):
            rows.append(
                _base_row(
                    paper_id="logu",
                    paper_label="LogU",
                    source_url=LOGU_URL,
                    table_id="table2_truthfulqa_auroc",
                    method=method,
                    model_paper=model_paper,
                    task_id="truthfulqa",
                    task_label="TruthfulQA (Reliability test)",
                    split="validation",  # standard TruthfulQA generation split; paper table does not restate split in table
                    metric="auroc",
                    value=val,
                    value_std=std,
                    notes="Table 2 reliability AUROC on TruthfulQA.",
                )
            )
    return rows


def _rows_ib_edl_calibration_table(
    *,
    table_id: str,
    model_paper: str,
    source_url: str,
    acc_rows: dict[str, list[tuple[float, float | None]]],
    ece_rows: dict[str, list[float]],
    nll_rows: dict[str, list[float]],
) -> list[dict[str, Any]]:
    tasks = [
        ("arc_challenge", "ARC-C", "validation"),
        ("arc_easy", "ARC-E", "validation"),
        ("openbookqa", "OpenBookQA", "validation"),
        ("commonsense_qa", "CommonsenseQA", "validation"),
        ("sciq", "SciQ", "validation"),
        ("race", "RACE", "test"),
    ]
    rows: list[dict[str, Any]] = []
    for method, vals in acc_rows.items():
        for (task_id, task_label, split), (v, s) in zip(tasks, vals):
            rows.append(
                _base_row(
                    paper_id="ib_edl",
                    paper_label="IB-EDL",
                    source_url=source_url,
                    table_id=table_id,
                    method=method,
                    model_paper=model_paper,
                    task_id=task_id,
                    task_label=task_label,
                    split=split,
                    metric="accuracy",
                    value=(v / 100.0),
                    value_std=s,
                    notes="Accuracy reported as percentage in paper tables; normalized to [0,1] here.",
                )
            )
    for method, vals in ece_rows.items():
        for (task_id, task_label, split), v in zip(tasks, vals):
            rows.append(
                _base_row(
                    paper_id="ib_edl",
                    paper_label="IB-EDL",
                    source_url=source_url,
                    table_id=table_id,
                    method=method,
                    model_paper=model_paper,
                    task_id=task_id,
                    task_label=task_label,
                    split=split,
                    metric="ece@15",
                    value=v,
                    notes="ECE with 15 bins (paper default; sensitivity shown separately in Table 14).",
                )
            )
    for method, vals in nll_rows.items():
        for (task_id, task_label, split), v in zip(tasks, vals):
            rows.append(
                _base_row(
                    paper_id="ib_edl",
                    paper_label="IB-EDL",
                    source_url=source_url,
                    table_id=table_id,
                    method=method,
                    model_paper=model_paper,
                    task_id=task_id,
                    task_label=task_label,
                    split=split,
                    metric="nll",
                    value=v,
                    notes="Negative log-likelihood.",
                )
            )
    return rows


def _rows_ib_edl() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        _rows_ib_edl_calibration_table(
            table_id="table1_llama2_7b_calibration",
            model_paper="Llama2-7B",
            source_url=IB_EDL_URL,
            acc_rows={
                "Base model": [(53.67, 0.00), (80.26, 0.00), (57.97, 0.00), (64.31, 0.00), (93.98, 0.00), (43.66, 0.00)],
                "MCDropout": [(54.59, 0.01), (79.09, 0.02), (57.42, 0.02), (65.18, 0.01), (94.06, 0.00), (45.13, 0.03)],
                "DensitySoftmax": [(53.67, 0.00), (80.26, 0.00), (57.97, 0.00), (64.31, 0.00), (93.98, 0.00), (43.66, 0.00)],
                "I-EDL": [(53.85, 0.01), (79.17, 0.00), (59.96, 0.02), (63.93, 0.03), (94.17, 0.01), (45.53, 0.01)],
                "IB-EDL": [(54.10, 0.01), (79.80, 0.01), (59.67, 0.01), (64.37, 0.01), (94.36, 0.00), (45.65, 0.04)],
            },
            ece_rows={
                "Base model": [0.1583, 0.0719, 0.1612, 0.2636, 0.0744, 0.2689],
                "MCDropout": [0.1537, 0.0621, 0.1505, 0.2558, 0.0634, 0.2692],
                "DensitySoftmax": [0.1433, 0.0500, 0.1452, 0.2498, 0.0578, 0.2496],
                "I-EDL": [0.1544, 0.0513, 0.1370, 0.2828, 0.0543, 0.2909],
                "IB-EDL": [0.1263, 0.0339, 0.1239, 0.2197, 0.0449, 0.2058],
            },
            nll_rows={
                "Base model": [1.2916, 0.5280, 1.0959, 1.1093, 0.2615, 1.4659],
                "MCDropout": [1.2502, 0.5166, 1.0645, 1.0789, 0.2604, 1.3888],
                "DensitySoftmax": [1.2916, 0.5280, 1.0959, 1.1093, 0.2615, 1.4659],
                "I-EDL": [1.2862, 0.5278, 1.0802, 1.1450, 0.2626, 1.4380],
                "IB-EDL": [1.1810, 0.4808, 1.0180, 0.9979, 0.2400, 1.3150],
            },
        )
    )
    rows.extend(
        _rows_ib_edl_calibration_table(
            table_id="table2_llama3_8b_calibration",
            model_paper="Llama3-8B",
            source_url=IB_EDL_URL,
            acc_rows={
                "Base model": [(68.77, 0.00), (86.89, 0.00), (74.44, 0.00), (68.55, 0.00), (96.27, 0.00), (50.41, 0.00)],
                "MCDropout": [(68.89, 0.01), (86.75, 0.01), (74.07, 0.01), (69.23, 0.01), (96.49, 0.00), (50.98, 0.00)],
                "DensitySoftmax": [(68.77, 0.00), (86.89, 0.00), (74.44, 0.00), (68.55, 0.00), (96.27, 0.00), (50.41, 0.00)],
                "I-EDL": [(68.41, 0.03), (87.20, 0.00), (74.57, 0.04), (69.82, 0.02), (96.36, 0.00), (52.14, 0.03)],
                "IB-EDL": [(68.97, 0.03), (87.13, 0.00), (75.36, 0.01), (69.56, 0.01), (96.53, 0.01), (52.45, 0.01)],
            },
            ece_rows={
                "Base model": [0.1188, 0.0313, 0.1038, 0.2126, 0.0337, 0.2427],
                "MCDropout": [0.1255, 0.0291, 0.0989, 0.2188, 0.0278, 0.2465],
                "DensitySoftmax": [0.0945, 0.0284, 0.0796, 0.1843, 0.0235, 0.1963],
                "I-EDL": [0.1028, 0.0276, 0.0688, 0.1860, 0.0238, 0.2023],
                "IB-EDL": [0.0776, 0.0170, 0.0584, 0.1454, 0.0198, 0.1424],
            },
            nll_rows={
                "Base model": [0.8618, 0.3228, 0.7674, 0.9856, 0.1484, 1.2162],
                "MCDropout": [0.8555, 0.3289, 0.7598, 0.9708, 0.1507, 1.1610],
                "DensitySoftmax": [0.8618, 0.3228, 0.7674, 0.9856, 0.1484, 1.2162],
                "I-EDL": [0.8530, 0.3200, 0.7537, 0.9692, 0.1485, 1.1304],
                "IB-EDL": [0.7797, 0.2919, 0.6894, 0.8606, 0.1366, 0.9810],
            },
        )
    )

    # Table 3 OOD detection AUROC with OOD data MMLU and different in-domain datasets.
    for model_paper, values in [
        (
            "Llama2-7B",
            {
                "Base model": [0.7101, 0.8551, 0.8508],
                "MCDropout": [0.7095, 0.8260, 0.8480],
                "DensitySoftmax": [0.6981, 0.7873, 0.7972],
                "I-EDL": [0.7014, 0.8606, 0.8574],
                "IB-EDL": [0.7280, 0.8947, 0.8928],
            },
        ),
        (
            "Llama3-8B",
            {
                "Base model": [0.7725, 0.8314, 0.8836],
                "MCDropout": [0.7880, 0.8354, 0.8862],
                "DensitySoftmax": [0.7504, 0.7851, 0.8243],
                "I-EDL": [0.7919, 0.8374, 0.8886],
                "IB-EDL": [0.8252, 0.8829, 0.9204],
            },
        ),
    ]:
        task_cols = [("sciq_vs_mmlu", "SciQ (ID) vs MMLU (OOD)"), ("winogrande_vs_mmlu", "WinoGrande (ID) vs MMLU (OOD)"), ("hellaswag_vs_mmlu", "HellaSwag (ID) vs MMLU (OOD)")]
        for method, vals in values.items():
            for (task_id, task_label), v in zip(task_cols, vals):
                rows.append(
                    _base_row(
                        paper_id="ib_edl",
                        paper_label="IB-EDL",
                        source_url=IB_EDL_URL,
                        table_id="table3_ood_auroc_mmlu",
                        method=method,
                        model_paper=model_paper,
                        task_id=task_id,
                        task_label=task_label,
                        split="validation",
                        metric="auroc",
                        value=v,
                        notes="OOD detection AUROC where MMLU is OOD (paper Table 3).",
                    )
                )

    # Table 14 (ECE sensitivity) for IB-EDL only, Llama2-7B.
    for bins, values in [
        (5, [0.1741, 0.0561, 0.1571, 0.2707, 0.0641, 0.2315]),
        (10, [0.1498, 0.0443, 0.1316, 0.2254, 0.0518, 0.2184]),
        (15, [0.1263, 0.0339, 0.1239, 0.2197, 0.0449, 0.2058]),
        (20, [0.1190, 0.0401, 0.1163, 0.2250, 0.0511, 0.2062]),
    ]:
        for (task_id, task_label), v in zip(
            [
                ("arc_challenge", "ARC-C"),
                ("arc_easy", "ARC-E"),
                ("openbookqa", "OpenBookQA"),
                ("commonsense_qa", "CommonsenseQA"),
                ("sciq", "SciQ"),
                ("race", "RACE"),
            ],
            values,
        ):
            rows.append(
                _base_row(
                    paper_id="ib_edl",
                    paper_label="IB-EDL",
                    source_url=IB_EDL_URL,
                    table_id="table14_ece_sensitivity_ib_edl_l2_7b",
                    method="IB-EDL",
                    model_paper="Llama2-7B",
                    task_id=task_id,
                    task_label=task_label,
                    split="validation" if task_id != "race" else "test",
                    metric=f"ece@{bins}",
                    value=v,
                    notes="Table 14 ECE sensitivity for IB-EDL (Llama2-7B).",
                )
            )
    return rows


def _rows_edtr() -> list[dict[str, Any]]:
    methods = {
        "Probability": {"GPT-OSS-20B": (0.183, 0.387, 0.210), "Llama3.1-8B": (0.198, 0.396, 0.215), "Qwen2.5-14B": (0.211, 0.405, 0.228)},
        "Vote": {"GPT-OSS-20B": (0.167, 0.379, 0.195), "Llama3.1-8B": (0.179, 0.386, 0.202), "Qwen2.5-14B": (0.191, 0.394, 0.217)},
        "Degree": {"GPT-OSS-20B": (0.160, 0.371, 0.187), "Llama3.1-8B": (0.172, 0.379, 0.194), "Qwen2.5-14B": (0.183, 0.387, 0.206)},
        "Density": {"GPT-OSS-20B": (0.158, 0.367, 0.184), "Llama3.1-8B": (0.169, 0.374, 0.190), "Qwen2.5-14B": (0.181, 0.382, 0.201)},
        "Component": {"GPT-OSS-20B": (0.154, 0.364, 0.182), "Llama3.1-8B": (0.165, 0.372, 0.188), "Qwen2.5-14B": (0.179, 0.380, 0.199)},
        "Dirichlet": {"GPT-OSS-20B": (0.149, 0.359, 0.177), "Llama3.1-8B": (0.161, 0.366, 0.183), "Qwen2.5-14B": (0.174, 0.375, 0.194)},
        "Fusion": {"GPT-OSS-20B": (0.142, 0.351, 0.170), "Llama3.1-8B": (0.153, 0.359, 0.177), "Qwen2.5-14B": (0.166, 0.368, 0.188)},
    }
    rows: list[dict[str, Any]] = []
    for method, by_model in methods.items():
        for model_paper, (ece, brier, rs) in by_model.items():
            rows.append(
                _base_row(
                    paper_id="edtr",
                    paper_label="EDTR",
                    source_url=EDTR_URL,
                    table_id="table1_4benchmark_average",
                    method=method,
                    model_paper=model_paper,
                    task_id="aggregate_4bench",
                    task_label="Average over AIME, GSM8K, CommonsenseQA, S&P500",
                    split="paper_mixed",
                    metric="ece",
                    value=ece,
                    notes="Table 1 average over four benchmarks.",
                )
            )
            rows.append(
                _base_row(
                    paper_id="edtr",
                    paper_label="EDTR",
                    source_url=EDTR_URL,
                    table_id="table1_4benchmark_average",
                    method=method,
                    model_paper=model_paper,
                    task_id="aggregate_4bench",
                    task_label="Average over AIME, GSM8K, CommonsenseQA, S&P500",
                    split="paper_mixed",
                    metric="brier",
                    value=brier,
                    notes="Table 1 average over four benchmarks.",
                )
            )
            rows.append(
                _base_row(
                    paper_id="edtr",
                    paper_label="EDTR",
                    source_url=EDTR_URL,
                    table_id="table1_4benchmark_average",
                    method=method,
                    model_paper=model_paper,
                    task_id="aggregate_4bench",
                    task_label="Average over AIME, GSM8K, CommonsenseQA, S&P500",
                    split="paper_mixed",
                    metric="reliability_score",
                    value=rs,
                    notes="Table 1 reliability score (paper-defined composite) average over four benchmarks.",
                )
            )
    return rows


@lru_cache(maxsize=1)
def published_baseline_rows() -> list[dict[str, Any]]:
    rows = []
    rows.extend(_rows_logu())
    rows.extend(_rows_ib_edl())
    rows.extend(_rows_edtr())
    return rows


def export_published_baselines(*, output_dir: str, bundle_name: str = "published") -> dict[str, Any]:
    rows = published_baseline_rows()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{bundle_name}.published_baselines.json"
    csv_path = out_dir / f"{bundle_name}.published_baselines.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    return {
        "count": len(rows),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
