from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from skedl.benchmarks.reliability_report import (
    _dataset_tag_from_file,
    _read_records,
    summarize_binary_reliability,
)


Record = dict[str, Any]
Scorer = Callable[[Record], float | None]


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _feature(record: Record, key: str) -> float | None:
    features = record.get("features")
    if not isinstance(features, dict):
        return None
    value = features.get(key)
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _score_proxy(record: Record) -> float | None:
    try:
        return _clip01(float(record["proxy_confidence"]))
    except Exception:
        return None


def _score_dirichlet_cd(record: Record) -> float | None:
    x = _feature(record, "c_d_logit")
    return None if x is None else _clip01(x)


def _score_dirichlet_eu(record: Record) -> float | None:
    x = _feature(record, "eu_mean")
    return None if x is None else float(np.exp(-max(0.0, x)))


def _score_dirichlet_au(record: Record) -> float | None:
    x = _feature(record, "au_mean")
    return None if x is None else float(np.exp(-max(0.0, x)))


def _score_connectivity_only(record: Record) -> float | None:
    x = _feature(record, "connectivity_risk")
    return None if x is None else _clip01(1.0 - x)


def _score_kernel_entropy_only(record: Record) -> float | None:
    x = _feature(record, "kernel_entropy")
    return None if x is None else float(np.exp(-max(0.0, x)))


def _score_spectral_kernel(record: Record) -> float | None:
    r = _feature(record, "connectivity_risk")
    h = _feature(record, "kernel_entropy")
    if r is None or h is None:
        return None
    return float(np.exp(-(0.5 * max(0.0, r) + 0.5 * max(0.0, h))))


SCORERS: dict[str, Scorer] = {
    "skedl_proxy": _score_proxy,
    "dirichlet_cd": _score_dirichlet_cd,
    "dirichlet_eu": _score_dirichlet_eu,
    "dirichlet_au": _score_dirichlet_au,
    "connectivity_only": _score_connectivity_only,
    "kernel_entropy_only": _score_kernel_entropy_only,
    "spectral_kernel": _score_spectral_kernel,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _scored_arrays(records: list[Record], scorer: Scorer) -> tuple[np.ndarray, np.ndarray]:
    confs: list[float] = []
    labels: list[int] = []
    for record in records:
        score = scorer(record)
        if score is None:
            continue
        if not math.isfinite(score):
            continue
        confs.append(_clip01(float(score)))
        labels.append(1 if bool(record.get("is_correct")) else 0)
    if not confs:
        return np.asarray([], dtype=float), np.asarray([], dtype=int)
    return np.asarray(confs, dtype=float), np.asarray(labels, dtype=int)


def generate_compare_reliability_report(
    *,
    record_files: list[str],
    output_dir: str,
    dataset_name: str | None = None,
    methods: list[str] | None = None,
    include_aggregate: bool = True,
    write_csv: bool = False,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if not record_files:
        raise ValueError("record_files must be non-empty")

    method_names = methods or ["skedl_proxy", "dirichlet_cd", "dirichlet_eu", "connectivity_only", "kernel_entropy_only", "spectral_kernel"]
    unknown = [m for m in method_names if m not in SCORERS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, Any]] = []
    by_method_all: dict[str, dict[str, list[np.ndarray]]] = {
        m: {"confs": [], "labels": []} for m in method_names
    }
    skipped: list[dict[str, Any]] = []

    for record_file in record_files:
        records = _read_records(record_file)
        dataset = _dataset_tag_from_file(record_file)
        for method in method_names:
            confs, labels = _scored_arrays(records, SCORERS[method])
            if confs.size == 0:
                skipped.append({"dataset": dataset, "method": method, "reason": "no_scored_examples"})
                continue
            row, _bins = summarize_binary_reliability(
                dataset=dataset,
                confidences=confs,
                labels=labels,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            row["method"] = method
            row["n_scored"] = int(confs.size)
            table_rows.append(row)
            by_method_all[method]["confs"].append(confs)
            by_method_all[method]["labels"].append(labels)

    if include_aggregate and len(record_files) > 1:
        for method in method_names:
            conf_parts = by_method_all[method]["confs"]
            label_parts = by_method_all[method]["labels"]
            if not conf_parts:
                continue
            confs = np.concatenate(conf_parts, axis=0)
            labels = np.concatenate(label_parts, axis=0)
            row, _bins = summarize_binary_reliability(
                dataset="aggregate",
                confidences=confs,
                labels=labels,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            row["method"] = method
            row["n_scored"] = int(confs.size)
            table_rows.append(row)

    bundle_name = dataset_name or "compare"
    _write_json(out_dir / f"{bundle_name}.compare.table.json", table_rows)
    _write_json(out_dir / f"{bundle_name}.compare.skipped.json", skipped)
    if write_csv:
        _write_csv(out_dir / f"{bundle_name}.compare.table.csv", table_rows)

    return {
        "tables": table_rows,
        "skipped": skipped,
        "output_dir": str(out_dir),
        "methods": method_names,
    }
