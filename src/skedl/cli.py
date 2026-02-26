from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skedl.adapters.llm.base import SampleCoTRequest
from skedl.benchmarks.runner import BenchmarkRunConfig, run_real_benchmark
from skedl.pipeline.features import SKEDLFeatureExtractor
from skedl.pipeline.sampling import build_temperature_schedule
from skedl.pipeline.trainer import FusionTrainer
from skedl.schemas import CoTSample, GenerationStep


DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu"]


def _add_common_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-cots", type=int, default=6)
    parser.add_argument("--cot-batch-size", type=int, default=1, help="How many CoTs to generate in parallel per temperature group")
    parser.add_argument("--temp-mode", choices=["fixed", "mixed"], default="fixed")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--temperature-delta", type=float, default=0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skedl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-fusion", help="Train a fusion calibrator")
    train.add_argument("--features-jsonl", type=str, default=None)
    train.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    _add_common_sampling_args(train)

    extract = subparsers.add_parser("extract-features", help="Extract SK-EDL features from local traces")
    extract.add_argument("--backend", choices=["mock", "transformers-local"], default="mock")
    extract.add_argument("--input-jsonl", type=str, required=False)
    extract.add_argument("--output-jsonl", type=str, required=False)
    extract.add_argument("--prompt", type=str, default=None)
    extract.add_argument("--llm-model", type=str, default=None)
    extract.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    extract.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    extract.add_argument("--embedding-device", choices=DEVICE_CHOICES, default="cpu")
    extract.add_argument("--dtype", type=str, default=None)
    extract.add_argument("--max-new-tokens", type=int, default=1024)
    extract.add_argument("--top-p", type=float, default=0.95)
    extract.add_argument("--capture-hidden-states", action="store_true")
    extract.add_argument("--k", type=int, default=3)
    extract.add_argument("--tau", type=float, default=5.0)
    _add_common_sampling_args(extract)

    demo = subparsers.add_parser("demo", help="Run a local mock SK-EDL demo")
    demo.add_argument("--backend", choices=["mock"], default="mock")
    demo.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    demo.add_argument("--k", type=int, default=2)
    demo.add_argument("--tau", type=float, default=5.0)
    _add_common_sampling_args(demo)

    bench = subparsers.add_parser("benchmark-run", help="Run SK-EDL on a real dataset with local inference")
    bench.add_argument(
        "--dataset",
        choices=[
            "gsm8k",
            "commonsense_qa",
            "piqa",
            "boolq",
            "arc_challenge",
            "arc_easy",
            "openbookqa",
            "sciq",
            "race",
            "truthfulqa_generation",
        ],
        required=True,
    )
    bench.add_argument("--split", type=str, default=None)
    bench.add_argument("--limit", type=int, default=5)
    bench.add_argument("--llm-model", type=str, required=True)
    bench.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    bench.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    bench.add_argument("--embedding-device", choices=DEVICE_CHOICES, default="cpu")
    bench.add_argument("--dtype", type=str, default=None)
    bench.add_argument("--max-new-tokens", type=int, default=1024)
    bench.add_argument("--top-p", type=float, default=0.95)
    bench.add_argument("--k", type=int, default=3)
    bench.add_argument("--tau", type=float, default=5.0)
    bench.add_argument("--output-dir", type=str, default="results/benchmarks")
    bench.add_argument("--save-cots", action="store_true")
    bench.add_argument("--truthfulqa-scorer", choices=["bleurt", "lexical"], default="bleurt")
    bench.add_argument("--truthfulqa-bleurt-threshold", type=float, default=0.5)
    bench.add_argument("--truthfulqa-reference-mode", choices=["best", "max_correct"], default="max_correct")
    _add_common_sampling_args(bench)

    report = subparsers.add_parser("reliability-report", help="Aggregate reliability metrics from benchmark records")
    report.add_argument("--records", nargs="+", required=True, help="One or more *.records.jsonl files")
    report.add_argument("--output-dir", type=str, default="results/reports")
    report.add_argument("--dataset-name", type=str, default=None)
    report.add_argument("--no-aggregate", action="store_true")
    report.add_argument("--no-plot", action="store_true")
    report.add_argument("--write-csv", action="store_true")
    report.add_argument("--bootstrap-samples", type=int, default=0)
    report.add_argument("--bootstrap-seed", type=int, default=0)

    compare = subparsers.add_parser("compare-reliability", help="Compare offline confidence scorers on cached benchmark records")
    compare.add_argument("--records", nargs="+", required=True, help="One or more *.records.jsonl files")
    compare.add_argument("--output-dir", type=str, default="results/reports")
    compare.add_argument("--dataset-name", type=str, default=None)
    compare.add_argument("--methods", nargs="+", default=None)
    compare.add_argument("--no-aggregate", action="store_true")
    compare.add_argument("--write-csv", action="store_true")
    compare.add_argument("--bootstrap-samples", type=int, default=0)
    compare.add_argument("--bootstrap-seed", type=int, default=0)

    pub_export = subparsers.add_parser("paper-baselines-export", help="Export curated published baseline results from related papers")
    pub_export.add_argument("--output-dir", type=str, default="results/paper_baselines")
    pub_export.add_argument("--bundle-name", type=str, default="published")

    paper_run = subparsers.add_parser("paper-protocol-run", help="Run SK-EDL under a paper-like protocol and compare strictly to curated published rows")
    paper_run.add_argument("--paper", choices=["logu", "ib_edl", "edtr"], required=True)
    paper_run.add_argument("--model-paper", type=str, required=True, help="Exact model name as used in the paper table (strict comparison key)")
    paper_run.add_argument("--hf-model", type=str, required=True, help="Local HuggingFace model id to run")
    paper_run.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    paper_run.add_argument("--output-dir", type=str, default="results/paper_runs")
    paper_run.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    paper_run.add_argument("--embedding-device", choices=DEVICE_CHOICES, default="cpu")
    paper_run.add_argument("--dtype", type=str, default=None)
    paper_run.add_argument("--limit", type=int, default=None, help="Optional per-dataset cap for pilot runs; omit for full split")
    paper_run.add_argument("--max-new-tokens", type=int, default=1024)
    paper_run.add_argument("--top-p", type=float, default=0.95)
    paper_run.add_argument("--k", type=int, default=3)
    paper_run.add_argument("--tau", type=float, default=5.0)
    paper_run.add_argument("--dry-run", action="store_true", help="Only write the protocol plan and print what would run")
    paper_run.add_argument("--no-compare", action="store_true", help="Skip strict published comparison after execution")
    paper_run.add_argument("--truthfulqa-scorer", choices=["bleurt", "lexical"], default="bleurt")
    paper_run.add_argument("--truthfulqa-bleurt-threshold", type=float, default=0.5)
    paper_run.add_argument("--truthfulqa-reference-mode", choices=["best", "max_correct"], default="max_correct")
    _add_common_sampling_args(paper_run)

    cmp_pub = subparsers.add_parser("compare-published", help="Strictly compare our results rows to curated published baselines")
    cmp_pub.add_argument("--paper", choices=["logu", "ib_edl", "edtr"], required=True)
    cmp_pub.add_argument("--our-rows-json", type=str, required=True)
    cmp_pub.add_argument("--output-dir", type=str, default="results/reports")
    cmp_pub.add_argument("--bundle-name", type=str, default="compare")
    cmp_pub.add_argument("--model-paper", type=str, default=None, help="Optional exact paper model filter")
    cmp_pub.add_argument("--published-json", type=str, default=None, help="Optional override for curated published rows JSON")
    cmp_pub.add_argument("--no-csv", action="store_true")

    return parser


def _mock_samples(num_cots: int) -> list[CoTSample]:
    # Two semantic modes to exercise spectral connectivity and kernel entropy.
    embs = []
    for i in range(num_cots):
        if i < num_cots // 2:
            embs.append(np.array([1.0 - 0.05 * i, 0.05 * i], dtype=float))
        else:
            j = i - (num_cots // 2)
            embs.append(np.array([0.05 * j, 1.0 - 0.05 * j], dtype=float))

    samples: list[CoTSample] = []
    for i, emb in enumerate(embs):
        logits_steps = [
            np.array([4.0 - 0.1 * i, 2.1, 0.2, -1.0], dtype=float),
            np.array([3.8 - 0.1 * i, 1.7, 0.1, -0.9], dtype=float),
        ]
        steps = [GenerationStep(token_id=t, token_text=str(t), logits=logit) for t, logit in enumerate(logits_steps)]
        samples.append(CoTSample(cot_text=f"cot-{i}", answer_text="42", embedding=emb, steps=steps))
    return samples


def _cmd_demo(args: argparse.Namespace) -> int:
    temps = build_temperature_schedule(
        num_cots=args.num_cots,
        mode=args.temp_mode,
        temperature=args.temperature,
        delta=args.temperature_delta,
    )
    samples = _mock_samples(args.num_cots)
    for sample, temp in zip(samples, temps):
        sample.metadata["temperature"] = temp

    extractor = SKEDLFeatureExtractor(k=args.k, tau=args.tau)
    result = extractor.extract(samples)
    confidence = float(
        np.clip(
            np.exp(
                -(
                    0.5 * result.features.get("connectivity_risk", 0.0)
                    + 0.25 * result.features.get("kernel_entropy", 0.0)
                    + 0.25 * result.features.get("eu_mean", 0.0)
                )
            ),
            0.0,
            1.0,
        )
    )
    payload = {
        "answer": "42",
        "confidence": confidence,
        "features": {k: float(v) for k, v in result.features.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_train_fusion(args: argparse.Namespace) -> int:
    # Minimal runnable path: train a logistic calibrator on synthetic features if no file is supplied.
    if args.features_jsonl is None:
        x = np.array(
            [
                [0.2, 0.8, 0.1],
                [0.1, 0.7, 0.2],
                [0.8, 0.2, 0.8],
                [0.9, 0.1, 0.9],
            ],
            dtype=np.float32,
        )
        y = np.array([1, 1, 0, 0], dtype=np.float32)
    else:
        path = Path(args.features_jsonl)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError("features-jsonl is empty")
        x = np.asarray([row["features"] for row in rows], dtype=np.float32)
        y = np.asarray([row["label"] for row in rows], dtype=np.float32)

    trainer = FusionTrainer(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    result = trainer.fit_logistic(x, y)
    print(json.dumps({"train_loss": float(result["train_loss"]), "device": result["device"]}))
    return 0


def _cmd_extract_features(args: argparse.Namespace) -> int:
    temps = build_temperature_schedule(
        num_cots=args.num_cots,
        mode=args.temp_mode,
        temperature=args.temperature,
        delta=args.temperature_delta,
    )

    if args.backend == "mock":
        samples = _mock_samples(args.num_cots)
        for sample, temp in zip(samples, temps):
            sample.metadata["temperature"] = temp
    elif args.backend == "transformers-local":
        if not args.prompt:
            raise ValueError("--prompt is required for --backend transformers-local")
        if not args.llm_model:
            raise ValueError("--llm-model is required for --backend transformers-local")

        from skedl.adapters.emb.sentence_transformers import SentenceTransformerEmbedder
        from skedl.adapters.llm.transformers_local import TransformersLocalLLM

        llm = TransformersLocalLLM(
            model_name=args.llm_model,
            device=args.device,
            dtype=args.dtype,
        )
        request = SampleCoTRequest(
            prompt=args.prompt,
            num_cots=args.num_cots,
            cot_batch_size=args.cot_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperatures=temps,
            top_p=args.top_p,
            capture_hidden_states=bool(args.capture_hidden_states),
        )
        samples = llm.sample_cots(request)

        embedder = SentenceTransformerEmbedder(
            model_name=args.embedding_model,
            device=args.embedding_device,
        )
        texts = [sample.cot_text for sample in samples]
        embeddings = embedder.encode(texts)
        if len(embeddings) != len(samples):
            raise ValueError("embedding count does not match sampled CoTs")
        for sample, emb, temp in zip(samples, embeddings, temps):
            sample.embedding = np.asarray(emb, dtype=float)
            sample.metadata["temperature"] = float(temp)
    else:
        raise ValueError(f"unsupported extract-features backend: {args.backend}")

    extractor = SKEDLFeatureExtractor(k=args.k, tau=args.tau)
    result = extractor.extract(samples)
    payload = {
        "features": result.features,
        "meta": {
            "backend": args.backend,
            "num_cots": args.num_cots,
            "cot_batch_size": args.cot_batch_size,
            "temp_mode": args.temp_mode,
        },
    }
    if args.output_jsonl:
        Path(args.output_jsonl).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _default_split_for_dataset(dataset: str) -> str:
    if dataset == "gsm8k":
        return "test"
    if dataset == "commonsense_qa":
        return "validation"
    if dataset == "piqa":
        return "validation"
    if dataset == "boolq":
        return "validation"
    if dataset == "arc_challenge":
        return "validation"
    if dataset == "openbookqa":
        return "validation"
    if dataset == "arc_easy":
        return "validation"
    if dataset == "sciq":
        return "validation"
    if dataset == "race":
        return "validation"
    if dataset == "truthfulqa_generation":
        return "validation"
    raise ValueError(f"unsupported dataset: {dataset}")


def _cmd_benchmark_run(args: argparse.Namespace) -> int:
    split = args.split or _default_split_for_dataset(args.dataset)
    config = BenchmarkRunConfig(
        dataset=args.dataset,
        split=split,
        limit=args.limit,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        device=args.device,
        embedding_device=args.embedding_device,
        dtype=args.dtype,
        num_cots=args.num_cots,
        cot_batch_size=args.cot_batch_size,
        temp_mode=args.temp_mode,
        temperature=args.temperature,
        temperature_delta=args.temperature_delta,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        k=args.k,
        tau=args.tau,
        output_dir=args.output_dir,
        save_cots=bool(args.save_cots),
        truthfulqa_scorer=args.truthfulqa_scorer,
        truthfulqa_bleurt_threshold=float(args.truthfulqa_bleurt_threshold),
        truthfulqa_reference_mode=args.truthfulqa_reference_mode,
    )
    out = run_real_benchmark(config)
    print(json.dumps(out["summary"], indent=2, sort_keys=True))
    return 0


def _cmd_reliability_report(args: argparse.Namespace) -> int:
    from skedl.benchmarks.reliability_report import generate_reliability_report

    out = generate_reliability_report(
        record_files=list(args.records),
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        include_aggregate=not bool(args.no_aggregate),
        write_csv=bool(args.write_csv),
        make_plots=not bool(args.no_plot),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(json.dumps({"output_dir": out["output_dir"], "tables": out["tables"], "plots": out["plots"]}, indent=2, sort_keys=True))
    return 0


def _cmd_compare_reliability(args: argparse.Namespace) -> int:
    from skedl.benchmarks.compare_reliability import generate_compare_reliability_report

    out = generate_compare_reliability_report(
        record_files=list(args.records),
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        methods=list(args.methods) if args.methods else None,
        include_aggregate=not bool(args.no_aggregate),
        write_csv=bool(args.write_csv),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(json.dumps({"output_dir": out["output_dir"], "methods": out["methods"], "tables": out["tables"]}, indent=2, sort_keys=True))
    return 0


def _cmd_paper_baselines_export(args: argparse.Namespace) -> int:
    from skedl.benchmarks.published_baselines import export_published_baselines

    out = export_published_baselines(output_dir=args.output_dir, bundle_name=args.bundle_name)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_paper_protocol_run(args: argparse.Namespace) -> int:
    from skedl.benchmarks.paper_protocols import run_paper_protocol

    out = run_paper_protocol(
        paper_id=args.paper,
        model_paper=args.model_paper,
        hf_model=args.hf_model,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
        device=args.device,
        embedding_device=args.embedding_device,
        dtype=args.dtype,
        num_cots=args.num_cots,
        cot_batch_size=args.cot_batch_size,
        temp_mode=args.temp_mode,
        temperature=args.temperature,
        temperature_delta=args.temperature_delta,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        k=args.k,
        tau=args.tau,
        limit=args.limit,
        truthfulqa_scorer=args.truthfulqa_scorer,
        truthfulqa_bleurt_threshold=float(args.truthfulqa_bleurt_threshold),
        truthfulqa_reference_mode=args.truthfulqa_reference_mode,
        dry_run=bool(args.dry_run),
        compare_after_run=not bool(args.no_compare),
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_compare_published(args: argparse.Namespace) -> int:
    from skedl.benchmarks.compare_published import compare_published_strict_from_files

    out = compare_published_strict_from_files(
        paper_id=args.paper,
        our_rows_json=args.our_rows_json,
        output_dir=args.output_dir,
        bundle_name=args.bundle_name,
        model_paper=args.model_paper,
        published_json=args.published_json,
        write_csv=not bool(args.no_csv),
    )
    print(
        json.dumps(
            {
                "paper_id": out["paper_id"],
                "model_paper": out["model_paper"],
                "match_policy": out["match_policy"],
                "matched": len(out["matched"]),
                "unmatched_published": len(out["unmatched_published"]),
                "unmatched_our": len(out["unmatched_our"]),
                "json_path": out["json_path"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "train-fusion":
        return _cmd_train_fusion(args)
    if args.command == "extract-features":
        return _cmd_extract_features(args)
    if args.command == "benchmark-run":
        return _cmd_benchmark_run(args)
    if args.command == "reliability-report":
        return _cmd_reliability_report(args)
    if args.command == "compare-reliability":
        return _cmd_compare_reliability(args)
    if args.command == "paper-baselines-export":
        return _cmd_paper_baselines_export(args)
    if args.command == "paper-protocol-run":
        return _cmd_paper_protocol_run(args)
    if args.command == "compare-published":
        return _cmd_compare_published(args)

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
