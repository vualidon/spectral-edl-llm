from __future__ import annotations


def test_build_paper_protocol_plan_marks_unsupported_tasks_for_ib_edl():
    from skedl.benchmarks.paper_protocols import build_paper_protocol_plan

    plan = build_paper_protocol_plan(
        paper_id="ib_edl",
        model_paper="Llama2-7B",
        hf_model="meta-llama/Llama-2-7b-chat-hf",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        output_dir="results/paper_runs",
        device="mps",
        embedding_device="cpu",
        num_cots=6,
        cot_batch_size=1,
        temp_mode="mixed",
        temperature=0.7,
        temperature_delta=0.2,
        max_new_tokens=64,
        top_p=0.95,
        k=3,
        tau=5.0,
    )

    assert plan["paper_id"] == "ib_edl"
    assert plan["model_paper"] == "Llama2-7B"
    assert any(item["supported"] for item in plan["runs"])
    assert any((not item["supported"]) for item in plan["runs"])
    unsupported_tasks = {item["task_id"] for item in plan["runs"] if not item["supported"]}
    assert "race" not in unsupported_tasks
    assert "arc_easy" not in unsupported_tasks
    assert "sciq" not in unsupported_tasks
    assert "sciq_vs_mmlu" in unsupported_tasks


def test_build_paper_protocol_plan_for_logu_supports_truthfulqa_task():
    from skedl.benchmarks.paper_protocols import build_paper_protocol_plan

    plan = build_paper_protocol_plan(
        paper_id="logu",
        model_paper="LLaMA3-8B",
        hf_model="meta-llama/Meta-Llama-3-8B-Instruct",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        output_dir="results/paper_runs",
        device="mps",
        embedding_device="cpu",
        num_cots=6,
        cot_batch_size=1,
        temp_mode="mixed",
        temperature=0.7,
        temperature_delta=0.2,
        max_new_tokens=64,
        top_p=0.95,
        k=3,
        tau=5.0,
    )

    assert plan["paper_id"] == "logu"
    task_ids = {item["task_id"] for item in plan["runs"]}
    assert task_ids == {"truthfulqa"}
    assert all(item["supported"] for item in plan["runs"])
