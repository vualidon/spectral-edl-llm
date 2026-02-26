#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from skedl.benchmarks.paper_protocols import run_paper_protocol


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run SK-EDL under a paper protocol and compare strictly against curated published rows."
    )
    p.add_argument("--paper", choices=["logu", "ib_edl", "edtr"], required=True)
    p.add_argument("--model-paper", required=True, help="Exact model name used in the paper table")
    p.add_argument("--hf-model", required=True, help="Local HuggingFace model id to execute")
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--output-dir", default="results/paper_runs")
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--embedding-device", choices=["auto", "mps", "cpu"], default="cpu")
    p.add_argument("--dtype", default=None)
    p.add_argument("--limit", type=int, default=None, help="Optional per-dataset cap for pilot runs")
    p.add_argument("--num-cots", type=int, default=6)
    p.add_argument("--temp-mode", choices=["fixed", "mixed"], default="mixed")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--temperature-delta", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--truthfulqa-scorer", choices=["bleurt", "lexical"], default="bleurt")
    p.add_argument("--truthfulqa-bleurt-threshold", type=float, default=0.5)
    p.add_argument("--truthfulqa-reference-mode", choices=["best", "max_correct"], default="max_correct")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-compare", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        temp_mode=args.temp_mode,
        temperature=args.temperature,
        temperature_delta=args.temperature_delta,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        k=args.k,
        tau=args.tau,
        truthfulqa_scorer=args.truthfulqa_scorer,
        truthfulqa_bleurt_threshold=args.truthfulqa_bleurt_threshold,
        truthfulqa_reference_mode=args.truthfulqa_reference_mode,
        limit=args.limit,
        dry_run=bool(args.dry_run),
        compare_after_run=not bool(args.no_compare),
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
