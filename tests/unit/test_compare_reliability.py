from __future__ import annotations

import json


def _write_records(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_generate_compare_reliability_produces_method_rows(tmp_path):
    from skedl.benchmarks.compare_reliability import generate_compare_reliability_report

    records = [
        {
            "is_correct": True,
            "proxy_confidence": 0.9,
            "features": {
                "c_d_logit": 0.85,
                "eu_mean": 0.1,
                "au_mean": 0.1,
                "connectivity_risk": 0.05,
                "kernel_entropy": 0.2,
            },
        },
        {
            "is_correct": False,
            "proxy_confidence": 0.3,
            "features": {
                "c_d_logit": 0.35,
                "eu_mean": 0.8,
                "au_mean": 0.6,
                "connectivity_risk": 0.7,
                "kernel_entropy": 0.9,
            },
        },
        {
            "is_correct": True,
            "proxy_confidence": 0.8,
            "features": {
                "c_d_logit": 0.75,
                "eu_mean": 0.2,
                "au_mean": 0.2,
                "connectivity_risk": 0.1,
                "kernel_entropy": 0.25,
            },
        },
    ]
    rec_path = tmp_path / "toy.records.jsonl"
    _write_records(rec_path, records)

    out = generate_compare_reliability_report(
        record_files=[str(rec_path)],
        output_dir=str(tmp_path / "out"),
        dataset_name="toy",
        bootstrap_samples=16,
        bootstrap_seed=0,
    )

    rows = out["tables"]
    methods = {row["method"] for row in rows}
    assert "skedl_proxy" in methods
    assert "dirichlet_cd" in methods
    assert "spectral_kernel" in methods
    assert all("auprc_error" in row for row in rows)
    assert all("accuracy_ci95_lo" in row for row in rows)


def test_generate_compare_reliability_aggregate_rows(tmp_path):
    from skedl.benchmarks.compare_reliability import generate_compare_reliability_report

    a = tmp_path / "a.records.jsonl"
    b = tmp_path / "b.records.jsonl"
    base_features = {
        "c_d_logit": 0.6,
        "eu_mean": 0.4,
        "au_mean": 0.3,
        "connectivity_risk": 0.3,
        "kernel_entropy": 0.5,
    }
    _write_records(
        a,
        [
            {"is_correct": True, "proxy_confidence": 0.8, "features": base_features},
            {"is_correct": False, "proxy_confidence": 0.2, "features": base_features},
        ],
    )
    _write_records(
        b,
        [
            {"is_correct": True, "proxy_confidence": 0.7, "features": base_features},
            {"is_correct": False, "proxy_confidence": 0.3, "features": base_features},
        ],
    )

    out = generate_compare_reliability_report(
        record_files=[str(a), str(b)],
        output_dir=str(tmp_path / "out"),
        include_aggregate=True,
        bootstrap_samples=0,
    )
    assert any(row["dataset"] == "aggregate" for row in out["tables"])
