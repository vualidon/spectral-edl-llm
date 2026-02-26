from __future__ import annotations

import json


def test_generate_reliability_report_writes_metrics_and_bins(tmp_path):
    from skedl.benchmarks.reliability_report import generate_reliability_report

    records_path = tmp_path / "toy.records.jsonl"
    rows = [
        {"is_correct": True, "proxy_confidence": 0.95},
        {"is_correct": True, "proxy_confidence": 0.80},
        {"is_correct": False, "proxy_confidence": 0.70},
        {"is_correct": False, "proxy_confidence": 0.20},
    ]
    with records_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    out_dir = tmp_path / "report"
    report = generate_reliability_report(
        record_files=[str(records_path)],
        output_dir=str(out_dir),
        dataset_name="toy",
        bootstrap_samples=32,
        bootstrap_seed=123,
    )

    assert "tables" in report
    assert len(report["tables"]) == 1
    metrics = report["tables"][0]
    assert "accuracy" in metrics
    assert "auprc_error" in metrics
    assert "aurc" in metrics
    assert "ece@15" in metrics
    assert "accuracy_ci95_lo" in metrics
    assert "ece@15_ci95_hi" in metrics

    metrics_json = out_dir / "toy.reliability.metrics.json"
    bins_json = out_dir / "toy.reliability.bins.json"
    assert metrics_json.exists()
    assert bins_json.exists()


def test_generate_reliability_report_supports_aggregate_row(tmp_path):
    from skedl.benchmarks.reliability_report import generate_reliability_report

    file_a = tmp_path / "a.records.jsonl"
    file_b = tmp_path / "b.records.jsonl"
    rows_a = [
        {"is_correct": True, "proxy_confidence": 0.9},
        {"is_correct": False, "proxy_confidence": 0.1},
    ]
    rows_b = [
        {"is_correct": True, "proxy_confidence": 0.7},
        {"is_correct": False, "proxy_confidence": 0.8},
    ]
    for path, rows in [(file_a, rows_a), (file_b, rows_b)]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    report = generate_reliability_report(
        record_files=[str(file_a), str(file_b)],
        output_dir=str(tmp_path / "out"),
        dataset_name=None,
        include_aggregate=True,
        bootstrap_samples=16,
        bootstrap_seed=7,
    )

    names = [row["dataset"] for row in report["tables"]]
    assert "aggregate" in names


def test_generate_reliability_report_can_disable_bootstrap(tmp_path):
    from skedl.benchmarks.reliability_report import generate_reliability_report

    records_path = tmp_path / "toy.records.jsonl"
    rows = [
        {"is_correct": True, "proxy_confidence": 0.9},
        {"is_correct": False, "proxy_confidence": 0.2},
    ]
    with records_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report = generate_reliability_report(
        record_files=[str(records_path)],
        output_dir=str(tmp_path / "out"),
        bootstrap_samples=0,
    )
    row = report["tables"][0]
    assert "accuracy_ci95_lo" not in row
