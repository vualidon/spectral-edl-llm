from __future__ import annotations

import json


def test_published_baselines_include_expected_papers_and_metrics():
    from skedl.benchmarks.published_baselines import published_baseline_rows

    rows = published_baseline_rows()
    assert rows, "expected curated published baseline rows"

    papers = {row["paper_id"] for row in rows}
    assert {"logu", "ib_edl", "edtr"}.issubset(papers)

    # Sanity-check a few exact values we extracted from the papers.
    row = next(
        r
        for r in rows
        if r["paper_id"] == "logu"
        and r["table_id"] == "table2_truthfulqa_auroc"
        and r["model_paper"] == "LLaMA3-8B"
        and r["method"] == "LogU"
        and r["metric"] == "auroc"
    )
    assert row["value"] == 0.839

    row = next(
        r
        for r in rows
        if r["paper_id"] == "ib_edl"
        and r["table_id"] == "table1_llama2_7b_calibration"
        and r["task_id"] == "commonsense_qa"
        and r["method"] == "IB-EDL"
        and r["metric"] == "ece@15"
    )
    assert row["value"] == 0.2197


def test_export_published_baselines_writes_json_and_csv(tmp_path):
    from skedl.benchmarks.published_baselines import export_published_baselines

    out = export_published_baselines(output_dir=str(tmp_path), bundle_name="paper-baselines")
    assert out["json_path"].endswith("paper-baselines.published_baselines.json")
    assert out["csv_path"].endswith("paper-baselines.published_baselines.csv")

    rows = json.loads((tmp_path / "paper-baselines.published_baselines.json").read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert any(r["paper_id"] == "edtr" for r in rows)
