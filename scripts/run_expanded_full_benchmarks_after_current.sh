#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/vualidon/project/spectral-edl-llm"
cd "$ROOT"

mkdir -p results/benchmarks/logs
mkdir -p results/reports

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

LAUNCH_LOG="results/benchmarks/logs/expanded_full_suite_launcher.log"

echo "[$(ts)] expanded suite queued" | tee -a "$LAUNCH_LOG"

set -a
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=src

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    echo "[$(ts)] waiting for prerequisite file: $path" | tee -a "$LAUNCH_LOG"
    sleep 300
  done
  echo "[$(ts)] found prerequisite file: $path" | tee -a "$LAUNCH_LOG"
}

run_dataset() {
  local dataset="$1"
  local split="$2"
  local limit="$3"
  local max_new_tokens="$4"
  local log_file="$5"
  local summary_file="$6"

  if [[ -f "$summary_file" ]]; then
    echo "[$(ts)] skip dataset=$dataset (summary exists: $summary_file)" | tee -a "$LAUNCH_LOG"
    return 0
  fi

  echo "[$(ts)] start dataset=$dataset split=$split limit=$limit" | tee -a "$LAUNCH_LOG"
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
  echo "[$(ts)] done dataset=$dataset rc=$rc" | tee -a "$LAUNCH_LOG"
  return $rc
}

CURRENT_CSQA_SUMMARY="results/benchmarks/commonsense_qa_validation_n1221.summary.json"
CURRENT_GSM8K_SUMMARY="results/benchmarks/gsm8k_test_n1319.summary.json"

echo "[$(ts)] waiting for current full suite to finish (CSQA + GSM8K)" | tee -a "$LAUNCH_LOG"
wait_for_file "$CURRENT_CSQA_SUMMARY"
wait_for_file "$CURRENT_GSM8K_SUMMARY"

run_dataset "boolq" "validation" "3270" "20" \
  "results/benchmarks/logs/boolq_validation_full.log" \
  "results/benchmarks/boolq_validation_n3270.summary.json"

run_dataset "arc_challenge" "validation" "299" "24" \
  "results/benchmarks/logs/arc_challenge_validation_full.log" \
  "results/benchmarks/arc_challenge_validation_n299.summary.json"

run_dataset "openbookqa" "validation" "500" "20" \
  "results/benchmarks/logs/openbookqa_validation_full.log" \
  "results/benchmarks/openbookqa_validation_n500.summary.json"

echo "[$(ts)] generating full reliability report with bootstrap CIs" | tee -a "$LAUNCH_LOG"
python -u -m skedl.cli reliability-report \
  --records \
  results/benchmarks/commonsense_qa_validation_n1221.records.jsonl \
  results/benchmarks/gsm8k_test_n1319.records.jsonl \
  results/benchmarks/boolq_validation_n3270.records.jsonl \
  results/benchmarks/arc_challenge_validation_n299.records.jsonl \
  results/benchmarks/openbookqa_validation_n500.records.jsonl \
  --output-dir results/reports \
  --dataset-name full-5ds \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42 \
  --write-csv \
  > results/benchmarks/logs/full_5ds_reliability_report.log 2>&1

echo "[$(ts)] generating offline compare-reliability report with bootstrap CIs" | tee -a "$LAUNCH_LOG"
python -u -m skedl.cli compare-reliability \
  --records \
  results/benchmarks/commonsense_qa_validation_n1221.records.jsonl \
  results/benchmarks/gsm8k_test_n1319.records.jsonl \
  results/benchmarks/boolq_validation_n3270.records.jsonl \
  results/benchmarks/arc_challenge_validation_n299.records.jsonl \
  results/benchmarks/openbookqa_validation_n500.records.jsonl \
  --output-dir results/reports \
  --dataset-name full-5ds \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42 \
  --write-csv \
  > results/benchmarks/logs/full_5ds_compare_reliability.log 2>&1

echo "[$(ts)] expanded suite complete" | tee -a "$LAUNCH_LOG"
