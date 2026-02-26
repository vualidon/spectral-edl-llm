#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/vualidon/project/spectral-edl-llm"
cd "$ROOT"

mkdir -p results/benchmarks/logs

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(ts)] full benchmark suite start" | tee -a results/benchmarks/logs/full_suite_launcher.log

set -a
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=src

run_dataset() {
  local dataset="$1"
  local split="$2"
  local limit="$3"
  local max_new_tokens="$4"
  local log_file="$5"

  echo "[$(ts)] start dataset=$dataset split=$split limit=$limit" | tee -a results/benchmarks/logs/full_suite_launcher.log
  python -u -m skedl.cli benchmark-run \
    --dataset "$dataset" \
    --split "$split" \
    --limit "$limit" \
    --llm-model "Qwen/Qwen3-1.7B" \
    --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
    --device mps \
    --embedding-device cpu \
    --dtype float16 \
    --num-cots 6 \
    --temp-mode mixed \
    --temperature 0.7 \
    --temperature-delta 0.2 \
    --max-new-tokens "$max_new_tokens" \
    --top-p 0.95 \
    --k 3 \
    --tau 5.0 \
    --output-dir results/benchmarks \
    > "$log_file" 2>&1
  local rc=$?
  echo "[$(ts)] done dataset=$dataset rc=$rc" | tee -a results/benchmarks/logs/full_suite_launcher.log
  return $rc
}

run_dataset "commonsense_qa" "validation" "1221" "32" "results/benchmarks/logs/commonsense_qa_validation_full.log"
run_dataset "gsm8k" "test" "1319" "48" "results/benchmarks/logs/gsm8k_test_full.log"

echo "[$(ts)] full benchmark suite end" | tee -a results/benchmarks/logs/full_suite_launcher.log
