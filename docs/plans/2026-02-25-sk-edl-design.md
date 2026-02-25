# SK-EDL Local Hybrid (MPS) Design

## Goal

Implement a training-ready local-only research package for Spectral-Kernel EDL (SK-EDL) confidence estimation on chain-of-thought (CoT) generation, with:

- spectral connectivity risk from Laplacian `lambda_2`
- kernel von Neumann entropy over CoT embeddings
- two Dirichlet branches:
  - direct-from-prelogits (LogU-style) uncertainty features
  - trainable EDL Dirichlet head
- hybrid fusion/calibration for final confidence

The first version targets Apple Silicon (`torch` MPS), not CUDA, and uses local inference only.

## Scope Decisions (approved)

- Delivery target: training-ready package (`v1`)
- Support mode: hybrid (availability-aware)
- Inference integration: local only (no OpenAI API)
- Device target: Apple Silicon MPS
- Default CoT sample count: `N=6`
- Sampling strategies:
  - `fixed` (unchanged temperature)
  - `mixed` (temperature schedule around base `T`)

## Architecture

### Package layout

- `pyproject.toml`
- `src/skedl/`
  - `__init__.py`
  - `schemas.py`
  - `cli.py`
  - `core/`
    - `kernel.py`
    - `entropy.py`
    - `graph.py`
    - `spectral.py`
    - `dirichlet.py`
    - `fusion.py`
    - `metrics.py`
  - `models/`
    - `logit_dirichlet.py`
    - `edl_head.py`
    - `dirichlet_router.py`
    - `hybrid_model.py`
  - `adapters/`
    - `llm/base.py`
    - `llm/transformers_local.py`
    - `emb/base.py`
    - `emb/sentence_transformers.py`
  - `pipeline/`
    - `sampling.py`
    - `features.py`
    - `datasets.py`
    - `trainer.py`
    - `evaluator.py`
- `tests/`
  - `unit/`
  - `integration/`
- `examples/`

### Design principles

- Keep spectral/kernel math deterministic and testable with `numpy`/`scipy`.
- Keep trainable modules in `torch`.
- Make local adapters optional at import time (clear runtime error if dependency missing).
- Hybrid model is mask-aware so missing branches are explicit features.

## Core Method (SK-EDL)

Given a prompt `x`, generate `N=6` CoTs and embeddings `z_i`.

Compute:

1. Dirichlet-derived confidence/uncertainty signals
2. Connectivity risk from `lambda_2` of a kNN graph Laplacian
3. Kernel entropy from a dense PSD semantic kernel

Fuse with a calibrator to produce final confidence.

### Kernel entropy branch

- Build dense semantic kernel `K` from CoT embeddings:
  - RBF (optional local scaling), or
  - cosine kernel after L2 normalization
- Normalize to unit trace:
  - `rho = K / trace(K)`
- Compute kernel entropy:
  - `H_ker = -sum(mu_tilde * log(mu_tilde + eps))`

### Spectral connectivity branch

- Build sparse symmetrized kNN graph `W` from kernel similarities
- Compute normalized symmetric Laplacian:
  - `L_sym = I - D^{-1/2} W D^{-1/2}`
- Compute second smallest eigenvalue `lambda_2`
- Convert to risk:
  - `R_lambda = exp(-tau * lambda_2)`

Small `lambda_2` implies weak connectivity and increases risk.

## Dirichlet Branches (local-only)

### A. Direct-from-prelogits branch (LogU-style)

This branch derives token-level uncertainty directly from local generation pre-softmax logits (no extra training required).

Per generated token step:

- Take top-`K_dir` logits as main candidates
- Treat top logits as Dirichlet evidence:
  - `alpha_k = topk_logit_k` (configurable evidence transform policy supported)
  - `alpha_0 = sum(alpha_k)`

Token-level uncertainty features:

- Aleatoric uncertainty (AU), expected entropy under Dirichlet:
  - `AU_t = -sum_k (alpha_k/alpha_0) * (digamma(alpha_k + 1) - digamma(alpha_0 + 1))`
- Epistemic / model uncertainty (EU):
  - `EU_t = K_dir / sum_k(alpha_k + 1)`

CoT-level aggregation (configurable):

- `mean`, `max`, `last_m`, and optional answer-span aggregation

Outputs for fusion:

- aggregated `AU`, `EU`
- optional derived confidence proxy `C_D_logit`

### B. Trainable EDL head

- Input: hidden-state summary vectors from local model generation
- Head outputs non-negative evidence `e >= 0`
- Dirichlet params:
  - `alpha = e + 1`
- Derived confidence:
  - `S = sum(alpha)`
  - `pi = alpha / S`
  - `U_D = K / S`
  - `C_D_head = (1 - U_D) * max(pi)`

This branch is used in white-box or hybrid settings when hidden states are available.

### Dirichlet router / hybrid usage

- If only logits are present: use direct-logit branch
- If logits + hidden states are present: use both branches
- If hidden states missing: disable trainable EDL head and set availability mask

## Fusion Model

Base fusion is a logistic calibrator / small MLP over:

- `R_lambda`
- `H_ker`
- logit-Dirichlet features (`AU`, `EU`, and/or `C_D_logit`)
- head-Dirichlet confidence (`C_D_head`, optional)
- branch availability masks

Default v1 starts with logistic calibration for interpretability and easy ablation.

## Local Inference and Sampling

### Backend

- Primary backend: local `transformers`
- Optional local embedding backend: `sentence-transformers`
- No remote APIs in v1

### Device behavior

- Default training/inference device: MPS when available
- Fallback to CPU if MPS unavailable
- Spectral/kernel eigendecompositions run on CPU (`numpy`/`scipy`)

### CoT sampling (`N=6`)

Default `N=6` for all examples and experiments.

Supported temperature modes:

- `fixed`:
  - all six CoTs use same temperature `T`
- `mixed`:
  - a temperature schedule around base `T`, e.g.
  - `[T, T, T, clamp(T-delta), clamp(T+delta), clamp(T+2*delta)]`

Both modes share the same pipeline so ablations are easy.

## Data Flow

### Inference (single prompt)

1. Sample `N=6` CoTs from local LLM with `fixed` or `mixed` temperature strategy
2. Extract final answers
3. Embed CoTs (or final answers) locally
4. Compute dense kernel `K` and `H_ker`
5. Compute sparse kNN graph `W`, `lambda_2`, `R_lambda`
6. Compute direct-logit Dirichlet features (AU/EU aggregates)
7. If hidden states available, compute trainable EDL-head features
8. Fuse into final confidence score `C`
9. Return selected answer + confidence + diagnostic features

### Training modes

- `train-fusion`
  - learns calibrator on precomputed features + correctness labels
- `train-whitebox-edl`
  - trains EDL head on hidden-state summaries and class labels
- `train-hybrid`
  - trains EDL head + fusion jointly (mask-aware)

## Error Handling and Failure Modes

### Numerical stability

- Add `eps` in entropy/log/normalization paths
- Symmetrize matrices before eigendecomposition
- Clip tiny negative eigenvalues caused by floating point noise

### Graph edge cases

- `N < 2`: return fallback spectral features with warning flag
- Invalid `k`: clamp to `[1, N-1]`
- Disconnected graph: allow multiple zeros and compute sorted spectrum safely

### Branch availability

- Missing hidden states disables trainable EDL branch with explicit metadata flag
- Missing generation scores/logits disables direct-logit branch with explicit error/warning
- Tokenizer mismatch raises configuration error

### Known scientific limitations

- Wrong-but-consistent CoTs can still appear confident
- Embedding mismatch can degrade spectral/kernel signals
- Low diversity sampling can create artificially overconfident graph/kernel features

## Testing and Verification Strategy

### Unit tests

- kernel PSD normalization and entropy sanity checks
- kNN graph + normalized Laplacian properties
- `lambda_2` sign/risk behavior on synthetic clustered graphs
- Dirichlet utility correctness
- LogU-style AU/EU computation from synthetic logits
- sampling schedules (`fixed`, `mixed`) with `N=6`
- MPS/CPU device selection helpers

### Integration tests

- mocked local generation trace -> feature extraction -> confidence
- hybrid feature masking when hidden states absent
- CLI command parsing and ablation flags
- optional slow `transformers` tiny-model smoke test

### Verification gates

- TDD cycle for each module (fail -> pass -> refactor)
- full unit suite before completion claim
- end-to-end local demo run (or explicit note if skipped due environment limits)

## Evaluation Plan (v1-ready scaffold)

- Metrics:
  - ECE
  - NLL
  - Brier
  - AUROC / AUPRC
  - risk-coverage
- Ablations:
  - spectral only
  - kernel only
  - logit-Dirichlet only
  - EDL-head only
  - full hybrid
  - `fixed` vs `mixed` temperature modes

## Notes for Implementation

- Start with a strong core and mocked integration tests.
- Implement `transformers` local adapter as optional dependency.
- Prioritize correctness and observability (diagnostic outputs) over benchmark throughput in v1.
