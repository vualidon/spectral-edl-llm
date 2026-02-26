from __future__ import annotations

import json


def test_compare_published_strict_matches_exact_model_task_metric_only(tmp_path):
    from skedl.benchmarks.compare_published import compare_published_strict

    published_rows = [
        {
            "paper_id": "ib_edl",
            "table_id": "table1_llama2_7b_calibration",
            "model_paper": "Llama2-7B",
            "task_id": "commonsense_qa",
            "split": "validation",
            "metric": "accuracy",
            "metric_direction": "higher",
            "method": "IB-EDL",
            "value": 0.68,
        },
        {
            "paper_id": "ib_edl",
            "table_id": "table1_llama2_7b_calibration",
            "model_paper": "Llama2-7B",
            "task_id": "commonsense_qa",
            "split": "validation",
            "metric": "ece@15",
            "metric_direction": "lower",
            "method": "IB-EDL",
            "value": 0.12,
        },
    ]
    our_rows = [
        {
            "paper_id": "ib_edl",
            "model_paper": "Llama2-7B",
            "task_id": "commonsense_qa",
            "split": "validation",
            "metric": "accuracy",
            "method": "SK-EDL",
            "value": 0.70,
        },
        {
            # Wrong model; must not match in strict mode.
            "paper_id": "ib_edl",
            "model_paper": "Llama3-8B",
            "task_id": "commonsense_qa",
            "split": "validation",
            "metric": "ece@15",
            "method": "SK-EDL",
            "value": 0.10,
        },
    ]

    out = compare_published_strict(
        paper_id="ib_edl",
        published_rows=published_rows,
        our_rows=our_rows,
        output_dir=str(tmp_path),
        bundle_name="ib-edl-l2",
    )

    assert len(out["matched"]) == 1
    assert out["matched"][0]["metric"] == "accuracy"
    assert out["matched"][0]["published_value"] == 0.68
    assert out["matched"][0]["our_value"] == 0.70
    assert out["matched"][0]["better"] is True

    # ECE row remains unmatched because our row uses a different model.
    unmatched_metrics = {(r["task_id"], r["metric"]) for r in out["unmatched_published"]}
    assert ("commonsense_qa", "ece@15") in unmatched_metrics

    payload = json.loads((tmp_path / "ib-edl-l2.compare_published.strict.json").read_text(encoding="utf-8"))
    assert "matched" in payload and "unmatched_published" in payload
