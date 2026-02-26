from __future__ import annotations


def test_cli_train_fusion_parses_mps_and_temperature_options():
    from skedl.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "train-fusion",
            "--device",
            "mps",
            "--num-cots",
            "6",
            "--temp-mode",
            "mixed",
            "--temperature",
            "0.7",
            "--temperature-delta",
            "0.2",
        ]
    )

    assert args.command == "train-fusion"
    assert args.device == "mps"
    assert args.num_cots == 6
    assert args.temp_mode == "mixed"
    assert args.temperature == 0.7
    assert args.temperature_delta == 0.2


def test_cli_extract_features_real_local_path_uses_adapters(monkeypatch, capsys):
    import json
    import numpy as np

    import skedl.cli as cli
    from skedl.schemas import CoTSample, GenerationStep

    class FakeLLM:
        def __init__(self, model_name: str, device: str = "auto", dtype: str | None = None):
            self.model_name = model_name
            self.device = device
            self.dtype = dtype

        def sample_cots(self, request):
            assert request.num_cots == 6
            assert request.temperatures is not None
            samples = []
            for i in range(6):
                steps = [
                    GenerationStep(token_id=1, token_text="1", logits=np.array([3.0, 1.0, 0.0, -1.0])),
                    GenerationStep(token_id=2, token_text="2", logits=np.array([2.5, 1.2, 0.1, -1.0])),
                ]
                samples.append(CoTSample(cot_text=f"cot {i}", answer_text="42", steps=steps))
            return samples

    class FakeEmbedder:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.model_name = model_name
            self.device = device

        def encode(self, texts):
            assert len(texts) == 6
            return np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3, dtype=float)

    import skedl.adapters.llm.transformers_local as llm_mod
    import skedl.adapters.emb.sentence_transformers as emb_mod

    monkeypatch.setattr(llm_mod, "TransformersLocalLLM", FakeLLM)
    monkeypatch.setattr(emb_mod, "SentenceTransformerEmbedder", FakeEmbedder)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "extract-features",
            "--backend",
            "transformers-local",
            "--prompt",
            "Solve 2+2 and explain briefly.",
            "--llm-model",
            "Qwen/Qwen2.5-3B-Instruct",
            "--embedding-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--device",
            "mps",
            "--embedding-device",
            "cpu",
            "--dtype",
            "float16",
            "--num-cots",
            "6",
            "--temp-mode",
            "mixed",
        ]
    )

    rc = cli._cmd_extract_features(args)
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["features"]["num_cots"] == 6.0
    assert payload["features"]["logit_dirichlet_available"] == 1.0


def test_cli_benchmark_run_parses_dataset_and_local_model_options():
    from skedl.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-run",
            "--dataset",
            "gsm8k",
            "--split",
            "test",
            "--limit",
            "5",
            "--llm-model",
            "Qwen/Qwen2.5-3B-Instruct",
            "--embedding-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--device",
            "mps",
            "--embedding-device",
            "cpu",
            "--num-cots",
            "6",
            "--temp-mode",
            "mixed",
        ]
    )

    assert args.command == "benchmark-run"
    assert args.dataset == "gsm8k"
    assert args.split == "test"
    assert args.limit == 5
    assert args.device == "mps"
    assert args.embedding_device == "cpu"


def test_cli_benchmark_run_parses_truthfulqa_options():
    from skedl.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-run",
            "--dataset",
            "truthfulqa_generation",
            "--split",
            "validation",
            "--limit",
            "3",
            "--llm-model",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "--embedding-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--truthfulqa-scorer",
            "bleurt",
            "--truthfulqa-bleurt-threshold",
            "0.5",
        ]
    )

    assert args.command == "benchmark-run"
    assert args.dataset == "truthfulqa_generation"
    assert args.truthfulqa_scorer == "bleurt"
    assert args.truthfulqa_bleurt_threshold == 0.5


def test_cli_reliability_report_parses_record_files_and_output_options():
    from skedl.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "reliability-report",
            "--records",
            "results/benchmarks/commonsense_qa_validation_n10.records.jsonl",
            "results/benchmarks/gsm8k_test_n5.records.jsonl",
            "--output-dir",
            "results/reports",
            "--dataset-name",
            "pilot-suite",
            "--write-csv",
        ]
    )

    assert args.command == "reliability-report"
    assert len(args.records) == 2
    assert args.output_dir == "results/reports"
    assert args.dataset_name == "pilot-suite"
    assert args.write_csv is True


def test_cli_compare_reliability_parses_options():
    from skedl.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-reliability",
            "--records",
            "results/benchmarks/commonsense_qa_validation_n10.records.jsonl",
            "results/benchmarks/gsm8k_test_n5.records.jsonl",
            "--output-dir",
            "results/reports",
            "--dataset-name",
            "pilot-suite",
            "--methods",
            "skedl_proxy",
            "dirichlet_cd",
            "spectral_kernel",
            "--bootstrap-samples",
            "100",
            "--write-csv",
        ]
    )

    assert args.command == "compare-reliability"
    assert len(args.records) == 2
    assert args.methods == ["skedl_proxy", "dirichlet_cd", "spectral_kernel"]
    assert args.bootstrap_samples == 100


def test_cli_paper_baselines_export_parses_options():
    from skedl.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "paper-baselines-export",
            "--output-dir",
            "results/paper_baselines",
            "--bundle-name",
            "v1",
        ]
    )
    assert args.command == "paper-baselines-export"
    assert args.output_dir == "results/paper_baselines"
    assert args.bundle_name == "v1"


def test_cli_paper_protocol_run_parses_options():
    from skedl.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "paper-protocol-run",
            "--paper",
            "ib_edl",
            "--model-paper",
            "Llama2-7B",
            "--hf-model",
            "meta-llama/Llama-2-7b-chat-hf",
            "--embedding-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--device",
            "mps",
            "--embedding-device",
            "cpu",
            "--output-dir",
            "results/paper_runs",
            "--dry-run",
        ]
    )
    assert args.command == "paper-protocol-run"
    assert args.paper == "ib_edl"
    assert args.model_paper == "Llama2-7B"
    assert args.hf_model == "meta-llama/Llama-2-7b-chat-hf"
    assert args.dry_run is True


def test_cli_paper_protocol_run_supports_cuda_and_new_generation_defaults():
    from skedl.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "paper-protocol-run",
            "--paper",
            "logu",
            "--model-paper",
            "LLaMA3-8B",
            "--hf-model",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "--device",
            "cuda",
            "--embedding-device",
            "cuda",
        ]
    )
    assert args.device == "cuda"
    assert args.embedding_device == "cuda"
    assert args.max_new_tokens == 1024


def test_cli_compare_published_parses_options():
    from skedl.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-published",
            "--paper",
            "ib_edl",
            "--our-rows-json",
            "results/paper_runs/ib_edl_llama2_7b/our_rows.json",
            "--output-dir",
            "results/reports",
            "--bundle-name",
            "ib-edl-l2",
        ]
    )
    assert args.command == "compare-published"
    assert args.paper == "ib_edl"
    assert args.our_rows_json.endswith("our_rows.json")
