from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from skedl.benchmarks.published_baselines import published_baseline_rows


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


def _compare_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("paper_id"),
        row.get("model_paper"),
        row.get("task_id"),
        row.get("split"),
        row.get("metric"),
    )


def _metric_better(metric_direction: str | None, our_value: float, published_value: float) -> bool | None:
    if metric_direction == "higher":
        return bool(our_value > published_value)
    if metric_direction == "lower":
        return bool(our_value < published_value)
    return None


def load_our_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(x) for x in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("our_rows"), list):
            return [dict(x) for x in payload["our_rows"]]
        if isinstance(payload.get("rows"), list):
            return [dict(x) for x in payload["rows"]]
    raise ValueError(f"unsupported our rows payload format: {path}")


def compare_published_strict(
    *,
    paper_id: str,
    published_rows: list[dict[str, Any]],
    our_rows: list[dict[str, Any]],
    output_dir: str,
    bundle_name: str = "compare",
    model_paper: str | None = None,
    write_csv: bool = True,
) -> dict[str, Any]:
    pub = [dict(r) for r in published_rows if str(r.get("paper_id")) == paper_id]
    ours = [dict(r) for r in our_rows if str(r.get("paper_id")) == paper_id]
    if model_paper is not None:
        pub = [r for r in pub if r.get("model_paper") == model_paper]
        ours = [r for r in ours if r.get("model_paper") == model_paper]

    our_index: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in ours:
        our_index.setdefault(_compare_key(row), []).append(row)

    matched: list[dict[str, Any]] = []
    unmatched_published: list[dict[str, Any]] = []
    duplicate_our_rows: list[dict[str, Any]] = []

    used_keys: set[tuple[Any, ...]] = set()
    for p in pub:
        key = _compare_key(p)
        candidates = our_index.get(key, [])
        if not candidates:
            unmatched_published.append(p)
            continue
        if len(candidates) > 1:
            duplicate_our_rows.extend(candidates[1:])
        o = candidates[0]
        used_keys.add(key)
        try:
            published_value = float(p["value"])
            our_value = float(o["value"])
        except Exception:
            unmatched_published.append(p)
            continue
        metric_direction = p.get("metric_direction")
        matched.append(
            {
                "paper_id": paper_id,
                "table_id": p.get("table_id"),
                "model_paper": p.get("model_paper"),
                "task_id": p.get("task_id"),
                "split": p.get("split"),
                "metric": p.get("metric"),
                "metric_direction": metric_direction,
                "published_method": p.get("method"),
                "published_value": published_value,
                "published_value_std": p.get("value_std"),
                "our_method": o.get("method", "SK-EDL"),
                "our_value": our_value,
                "delta_our_minus_published": our_value - published_value,
                "better": _metric_better(str(metric_direction) if metric_direction is not None else None, our_value, published_value),
                "strict_match": True,
            }
        )

    unmatched_our = []
    for key, rows in our_index.items():
        if key in used_keys:
            continue
        unmatched_our.extend(rows)

    out = {
        "paper_id": paper_id,
        "model_paper": model_paper,
        "match_policy": "strict_exact_model_task_split_metric",
        "matched": matched,
        "unmatched_published": unmatched_published,
        "unmatched_our": unmatched_our,
        "duplicate_our_rows": duplicate_our_rows,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{bundle_name}.compare_published.strict.json"
    _write_json(json_path, out)
    if write_csv:
        _write_csv(out_dir / f"{bundle_name}.compare_published.strict.matched.csv", matched)
        _write_csv(out_dir / f"{bundle_name}.compare_published.strict.unmatched_published.csv", unmatched_published)
        _write_csv(out_dir / f"{bundle_name}.compare_published.strict.unmatched_our.csv", unmatched_our)
    out["json_path"] = str(json_path)
    return out


def compare_published_strict_from_files(
    *,
    paper_id: str,
    our_rows_json: str,
    output_dir: str,
    bundle_name: str = "compare",
    model_paper: str | None = None,
    published_json: str | None = None,
    write_csv: bool = True,
) -> dict[str, Any]:
    if published_json:
        published_rows = json.loads(Path(published_json).read_text(encoding="utf-8"))
    else:
        published_rows = published_baseline_rows()
    our_rows = load_our_rows(our_rows_json)
    return compare_published_strict(
        paper_id=paper_id,
        published_rows=published_rows,
        our_rows=our_rows,
        output_dir=output_dir,
        bundle_name=bundle_name,
        model_paper=model_paper,
        write_csv=write_csv,
    )
