#!/usr/bin/env bash
set -euo pipefail

# Server setup helper for CUDA-based SK-EDL / LogU runs.
# Notes:
# - Main model inference uses CUDA PyTorch in .venv
# - BLEURT runs in a separate CPU-sidecar venv for isolation/stability
# - Edit TORCH_INDEX_URL if your server uses a different CUDA wheel index (e.g., cu124)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Allow override, otherwise auto-detect a usable Python interpreter.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[setup] ERROR: no Python interpreter found (tried python3.11, 3.10, 3.9, python3)." >&2
  echo "[setup] Install Python 3.9+ and rerun, or set PYTHON_BIN explicitly." >&2
  exit 1
fi

MAIN_VENV="${MAIN_VENV:-.venv}"
BLEURT_VENV="${BLEURT_VENV:-.venv-bleurt}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "[setup] root=$ROOT_DIR"
echo "[setup] python=$PYTHON_BIN"
echo "[setup] main venv=$MAIN_VENV"
echo "[setup] bleurt venv=$BLEURT_VENV"
echo "[setup] torch index=$TORCH_INDEX_URL"

"$PYTHON_BIN" -m venv "$MAIN_VENV"
source "$MAIN_VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Install CUDA-enabled torch first so editable install reuses it.
pip install --index-url "$TORCH_INDEX_URL" "torch>=2.2"

# Core project + HF extras + common runtime deps used by paper protocol runs.
pip install -e ".[hf,dev]"
pip install "datasets>=2.18" "sentencepiece>=0.1.99" "protobuf>=4" "accelerate>=0.27"

deactivate

"$PYTHON_BIN" -m venv "$BLEURT_VENV"
source "$BLEURT_VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install "tensorflow-cpu>=2.15,<2.16" "evaluate>=0.4.6" "sentencepiece>=0.1.99"
pip install "git+https://github.com/google-research/bleurt.git"
deactivate

cat <<'EOF'
[setup] complete

Next:
  1. Put your Hugging Face token in .env:
       HF_TOKEN=...
  2. (Optional) warm up model access:
       source .venv/bin/activate
       python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')"
  3. Run strict LogU on CUDA:
       bash scripts/run_logu_strict_cuda.sh

Smoke run first (recommended):
       LIMIT=1 bash scripts/run_logu_strict_cuda.sh
EOF
