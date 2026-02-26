from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from skedl.core.metrics import reliability_bins
from skedl.pipeline.evaluator import evaluate_binary_confidence


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"records file is empty: {path}")
    return rows


def _dataset_tag_from_file(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".records.jsonl"):
        return name[: -len(".records.jsonl")]
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")]
    return Path(path).stem


def _extract_binary_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([1 if bool(r.get("is_correct")) else 0 for r in records], dtype=int)
    confidences = np.asarray([float(r["proxy_confidence"]) for r in records], dtype=float)
    return confidences, labels


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _bootstrap_ci_columns(
    confidences: np.ndarray,
    labels: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    ci_level: float = 95.0,
) -> dict[str, float | None]:
    if bootstrap_samples <= 0:
        return {}
    n = int(labels.size)
    if n <= 0:
        return {}

    rng = np.random.default_rng(int(bootstrap_seed))
    alpha = max(0.0, min(100.0, float(ci_level)))
    lo_q = 0.5 * (100.0 - alpha)
    hi_q = 100.0 - lo_q
    suffix = int(round(alpha))

    tracked_keys = [
        "accuracy",
        "mean_confidence",
        "brier",
        "nll",
        "ece",
        "ece@5",
        "ece@10",
        "ece@15",
        "ece@20",
        "ece@25",
        "ece@35",
        "auroc",
        "auroc_correct",
        "auroc_error",
        "auprc_error",
        "aurc",
    ]
    samples_by_metric: dict[str, list[float]] = {key: [] for key in tracked_keys}

    for _ in range(int(bootstrap_samples)):
        idx = rng.integers(0, n, size=n)
        c_b = confidences[idx]
        y_b = labels[idx]
        metrics = evaluate_binary_confidence(c_b, y_b)
        bootstrap_row: dict[str, Any] = {
            "accuracy": float(np.mean(y_b)),
            "mean_confidence": float(np.mean(c_b)),
            **metrics,
        }
        for key in tracked_keys:
            x = _float_or_none(bootstrap_row.get(key))
            if x is not None:
                samples_by_metric[key].append(x)

    out: dict[str, float | None] = {}
    for key, vals in samples_by_metric.items():
        if not vals:
            out[f"{key}_ci{suffix}_lo"] = None
            out[f"{key}_ci{suffix}_hi"] = None
            continue
        arr = np.asarray(vals, dtype=float)
        out[f"{key}_ci{suffix}_lo"] = float(np.percentile(arr, lo_q))
        out[f"{key}_ci{suffix}_hi"] = float(np.percentile(arr, hi_q))
    return out


def summarize_binary_reliability(
    *,
    dataset: str,
    confidences: np.ndarray,
    labels: np.ndarray,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    confs = np.asarray(confidences, dtype=float).reshape(-1)
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    metrics = evaluate_binary_confidence(confs, labels)
    bins_15 = reliability_bins(confs, labels, n_bins=15)

    row: dict[str, Any] = {
        "dataset": dataset,
        "n": int(labels_arr.size),
        "accuracy": float(np.mean(labels_arr)) if labels_arr.size else 0.0,
        "mean_confidence": float(np.mean(confs)) if confs.size else 0.0,
    }
    row.update({k: _float_or_none(v) for k, v in metrics.items()})
    if bootstrap_samples > 0:
        row.update(
            _bootstrap_ci_columns(
                confs,
                labels_arr,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                ci_level=95.0,
            )
        )
    bins_payload = {
        "dataset": dataset,
        "n": int(labels_arr.size),
        "bins_15": bins_15,
    }
    return row, bins_payload


def _metrics_row(
    dataset: str,
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    confs, labels = _extract_binary_arrays(records)
    return summarize_binary_reliability(
        dataset=dataset,
        confidences=confs,
        labels=labels,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


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


def _plot_reliability_diagram(path: Path, bins_payload: dict[str, Any]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    bins = bins_payload["bins_15"]
    counts = np.asarray([float(b["count"]) for b in bins], dtype=float)
    accs = np.asarray([float(b["accuracy"]) for b in bins], dtype=float)
    confs = np.asarray([float(b["mean_confidence"]) for b in bins], dtype=float)
    centers = np.asarray([(float(b["left"]) + float(b["right"])) * 0.5 for b in bins], dtype=float)
    widths = np.asarray([float(b["right"]) - float(b["left"]) for b in bins], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, label="Perfect calibration")
    ax.bar(centers, accs, width=widths * 0.95, alpha=0.6, color="#2f6fba", edgecolor="#1b3e6b", label="Accuracy")
    ax.plot(centers, confs, color="#d94841", marker="o", linewidth=1.2, markersize=3, label="Mean confidence")

    # annotate coverage mass visually on a secondary axis
    ax2 = ax.twinx()
    ax2.plot(centers, counts / max(np.sum(counts), 1.0), color="#6a994e", linewidth=1.0, alpha=0.8, label="Bin mass")
    ax2.set_ylabel("Bin mass")
    ax2.set_ylim(0.0, 1.0)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram: {bins_payload['dataset']}")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower right", fontsize=8)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def generate_reliability_report(
    *,
    record_files: list[str],
    output_dir: str,
    dataset_name: str | None = None,
    include_aggregate: bool = True,
    write_csv: bool = False,
    make_plots: bool = True,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if not record_files:
        raise ValueError("record_files must be non-empty")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, Any]] = []
    bins_payloads: list[dict[str, Any]] = []
    plot_paths: list[str] = []
    all_records: list[dict[str, Any]] = []

    for record_file in record_files:
        records = _read_records(record_file)
        all_records.extend(records)
        dataset = _dataset_tag_from_file(record_file)
        row, bins_payload = _metrics_row(
            dataset,
            records,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        table_rows.append(row)
        bins_payloads.append(bins_payload)

        metrics_path = out_dir / f"{dataset}.reliability.metrics.json"
        bins_path = out_dir / f"{dataset}.reliability.bins.json"
        _write_json(metrics_path, row)
        _write_json(bins_path, bins_payload)
        if make_plots:
            png_path = out_dir / f"{dataset}.reliability.png"
            written = _plot_reliability_diagram(png_path, bins_payload)
            if written is not None:
                plot_paths.append(written)

    if include_aggregate and len(record_files) > 1:
        agg_row, agg_bins = _metrics_row(
            "aggregate",
            all_records,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        table_rows.append(agg_row)
        bins_payloads.append(agg_bins)
        _write_json(out_dir / "aggregate.reliability.metrics.json", agg_row)
        _write_json(out_dir / "aggregate.reliability.bins.json", agg_bins)
        if make_plots:
            written = _plot_reliability_diagram(out_dir / "aggregate.reliability.png", agg_bins)
            if written is not None:
                plot_paths.append(written)

    bundle_name = dataset_name or "report"
    _write_json(out_dir / f"{bundle_name}.reliability.table.json", table_rows)
    _write_json(out_dir / f"{bundle_name}.reliability.bins.bundle.json", bins_payloads)
    if write_csv:
        _write_csv(out_dir / f"{bundle_name}.reliability.table.csv", table_rows)

    return {
        "tables": table_rows,
        "bins": bins_payloads,
        "plots": plot_paths,
        "output_dir": str(out_dir),
    }
