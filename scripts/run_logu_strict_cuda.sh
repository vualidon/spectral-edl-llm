#!/usr/bin/env bash
set -euo pipefail

# Strict LogU-style TruthfulQA reliability run on CUDA.
# Current code defaults (set in Python):
# - logit capture top-k = 10
# - Dirichlet top-k = 10

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAIN_VENV="${MAIN_VENV:-.venv}"
BLEURT_VENV="${BLEURT_VENV:-.venv-bleurt}"
PAPER_MODEL_NAME="${PAPER_MODEL_NAME:-LLaMA3-8B}"
HF_MODEL="${HF_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_runs}"
DEVICE="${DEVICE:-cuda}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
NUM_COTS="${NUM_COTS:-6}"
COT_BATCH_SIZE="${COT_BATCH_SIZE:-6}"
TEMP_MODE="${TEMP_MODE:-fixed}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TEMPERATURE_DELTA="${TEMPERATURE_DELTA:-0.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TOP_P="${TOP_P:-0.95}"
TRUTHFULQA_SCORER="${TRUTHFULQA_SCORER:-bleurt}"
TRUTHFULQA_BLEURT_THRESHOLD="${TRUTHFULQA_BLEURT_THRESHOLD:-0.5}"
TRUTHFULQA_REFERENCE_MODE="${TRUTHFULQA_REFERENCE_MODE:-max_correct}"

if [[ ! -d "$MAIN_VENV" ]]; then
  echo "[run] missing main venv: $MAIN_VENV (run scripts/setup_cuda_server.sh first)" >&2
  exit 1
fi
if [[ ! -x "$BLEURT_VENV/bin/python" ]]; then
  echo "[run] missing BLEURT sidecar python: $BLEURT_VENV/bin/python (run scripts/setup_cuda_server.sh first)" >&2
  exit 1
fi

BLEURT_SIDECAR_PY=""
if [[ "$BLEURT_VENV" = /* ]]; then
  BLEURT_SIDECAR_PY="$BLEURT_VENV/bin/python"
else
  BLEURT_SIDECAR_PY="$ROOT_DIR/$BLEURT_VENV/bin/python"
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

mkdir -p "$OUTPUT_DIR/logu/llama3_8b"
LOG_PATH="$OUTPUT_DIR/logu/llama3_8b/full_run.strict.log"

echo "[run] log=$LOG_PATH"
echo "[run] model=$HF_MODEL device=$DEVICE embedding_device=$EMBEDDING_DEVICE num_cots=$NUM_COTS cot_batch_size=$COT_BATCH_SIZE max_new_tokens=$MAX_NEW_TOKENS"
if [[ ${#LIMIT_ARGS[@]} -gt 0 ]]; then
  echo "[run] smoke mode limit=${LIMIT}"
fi

source "$MAIN_VENV/bin/activate"
export SKEDL_BLEURT_SIDECAR_PYTHON="$BLEURT_SIDECAR_PY"

PYTHONPATH=src python -m skedl.cli paper-protocol-run \
  --paper logu \
  --model-paper "$PAPER_MODEL_NAME" \
  --hf-model "$HF_MODEL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --device "$DEVICE" \
  --embedding-device "$EMBEDDING_DEVICE" \
  --dtype "$DTYPE" \
  --num-cots "$NUM_COTS" \
  --cot-batch-size "$COT_BATCH_SIZE" \
  --temp-mode "$TEMP_MODE" \
  --temperature "$TEMPERATURE" \
  --temperature-delta "$TEMPERATURE_DELTA" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --top-p "$TOP_P" \
  --output-dir "$OUTPUT_DIR" \
  --truthfulqa-scorer "$TRUTHFULQA_SCORER" \
  --truthfulqa-bleurt-threshold "$TRUTHFULQA_BLEURT_THRESHOLD" \
  --truthfulqa-reference-mode "$TRUTHFULQA_REFERENCE_MODE" \
  "${LIMIT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"
