# SK-EDL Local Hybrid (MPS) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a training-ready local-only SK-EDL package for Apple MPS with spectral/kernel confidence features, dual Dirichlet branches (direct-from-prelogits and trainable EDL head), hybrid fusion, and a runnable CLI scaffold.

**Architecture:** Implement a `src/skedl` package with deterministic spectral/kernel math in `numpy/scipy`, trainable modules in `torch`, and optional local `transformers`/`sentence-transformers` adapters. Use a TDD-first workflow with unit tests for each mathematical component, then integrate through feature extraction, training/evaluation utilities, and CLI commands.

**Tech Stack:** Python 3.11+, pytest, numpy, scipy, torch (MPS), transformers (optional), sentence-transformers (optional)

---

### Task 1: Package scaffold and test infrastructure

**Files:**
- Create: `pyproject.toml`
- Create: `src/skedl/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/test_imports.py`

**Step 1: Write the failing test**

```python
def test_package_imports():
    import skedl
    assert hasattr(skedl, "__version__")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_imports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'skedl'`

**Step 3: Write minimal implementation**

```python
__version__ = "0.1.0"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_imports.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml src/skedl/__init__.py tests/unit/test_imports.py tests/conftest.py
git commit -m "build: scaffold skedl package and tests"
```

### Task 2: Sampling schedules (`N=6`, fixed/mixed) and schemas

**Files:**
- Create: `src/skedl/schemas.py`
- Create: `src/skedl/pipeline/sampling.py`
- Create: `tests/unit/test_sampling.py`

**Step 1: Write the failing tests**

```python
def test_fixed_temperature_schedule_has_six_equal_values():
    ...

def test_mixed_temperature_schedule_uses_offsets_and_clamps():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sampling.py -v`
Expected: FAIL because module/functions do not exist

**Step 3: Write minimal implementation**

```python
def build_temperature_schedule(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sampling.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/skedl/schemas.py src/skedl/pipeline/sampling.py tests/unit/test_sampling.py
git commit -m "feat: add sampling schedules and core schemas"
```

### Task 3: Kernel construction and entropy branch

**Files:**
- Create: `src/skedl/core/kernel.py`
- Create: `src/skedl/core/entropy.py`
- Create: `tests/unit/test_kernel_entropy.py`

**Step 1: Write the failing tests**

```python
def test_cosine_kernel_is_symmetric_and_bounded():
    ...

def test_trace_normalized_kernel_entropy_is_low_for_identical_embeddings():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_kernel_entropy.py -v`
Expected: FAIL due to missing functions

**Step 3: Write minimal implementation**

```python
def cosine_kernel(...): ...
def kernel_entropy(...): ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_kernel_entropy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/skedl/core/kernel.py src/skedl/core/entropy.py tests/unit/test_kernel_entropy.py
git commit -m "feat: implement kernel entropy branch"
```

### Task 4: kNN graph, Laplacian, `lambda_2` risk

**Files:**
- Create: `src/skedl/core/graph.py`
- Create: `src/skedl/core/spectral.py`
- Create: `tests/unit/test_spectral.py`

**Step 1: Write the failing tests**

```python
def test_knn_graph_is_symmetrized():
    ...

def test_lambda2_risk_increases_for_weaker_connectivity():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_spectral.py -v`
Expected: FAIL due to missing graph/spectral functions

**Step 3: Write minimal implementation**

```python
def symmetrized_knn_graph(...): ...
def normalized_laplacian(...): ...
def lambda2(...): ...
def connectivity_risk(...): ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_spectral.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/skedl/core/graph.py src/skedl/core/spectral.py tests/unit/test_spectral.py
git commit -m "feat: add spectral connectivity risk features"
```

### Task 5: Dirichlet utilities and direct-from-prelogits (LogU-style) branch

**Files:**
- Create: `src/skedl/core/dirichlet.py`
- Create: `src/skedl/models/logit_dirichlet.py`
- Create: `tests/unit/test_dirichlet.py`
- Create: `tests/unit/test_logit_dirichlet.py`

**Step 1: Write the failing tests**

```python
def test_edl_confidence_increases_with_concentrated_alpha():
    ...

def test_logit_dirichlet_extracts_au_eu_from_scores():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_dirichlet.py tests/unit/test_logit_dirichlet.py -v`
Expected: FAIL due to missing implementations

**Step 3: Write minimal implementation**

```python
def edl_confidence(...): ...
class LogitDirichletExtractor: ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_dirichlet.py tests/unit/test_logit_dirichlet.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/skedl/core/dirichlet.py src/skedl/models/logit_dirichlet.py tests/unit/test_dirichlet.py tests/unit/test_logit_dirichlet.py
git commit -m "feat: add dirichlet utilities and logit-based uncertainty branch"
```

### Task 6: Trainable EDL head, hybrid fusion, and routing

**Files:**
- Create: `src/skedl/core/fusion.py`
- Create: `src/skedl/models/edl_head.py`
- Create: `src/skedl/models/dirichlet_router.py`
- Create: `src/skedl/models/hybrid_model.py`
- Create: `tests/unit/test_models.py`

**Step 1: Write the failing tests**

```python
def test_edl_head_outputs_positive_evidence():
    ...

def test_hybrid_fusion_accepts_missing_branch_masks():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_models.py -v`
Expected: FAIL due to missing torch modules

**Step 3: Write minimal implementation**

```python
class EDLHead(nn.Module): ...
class LogisticFusion(nn.Module): ...
class HybridConfidenceModel(nn.Module): ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/skedl/core/fusion.py src/skedl/models/edl_head.py src/skedl/models/dirichlet_router.py src/skedl/models/hybrid_model.py tests/unit/test_models.py
git commit -m "feat: add trainable EDL and hybrid fusion modules"
```

### Task 7: Feature extraction pipeline (spectral + kernel + Dirichlet)

**Files:**
- Create: `src/skedl/pipeline/features.py`
- Create: `tests/integration/test_feature_pipeline.py`

**Step 1: Write the failing test**

```python
def test_feature_pipeline_computes_sk_edl_features_from_mock_cots():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_feature_pipeline.py -v`
Expected: FAIL due to missing feature pipeline

**Step 3: Write minimal implementation**

```python
class SKEDLFeatureExtractor:
    def extract(...): ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_feature_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/skedl/pipeline/features.py tests/integration/test_feature_pipeline.py
git commit -m "feat: implement SK-EDL feature extraction pipeline"
```

### Task 8: Local adapters (transformers + embeddings) with optional deps

**Files:**
- Create: `src/skedl/adapters/llm/base.py`
- Create: `src/skedl/adapters/llm/transformers_local.py`
- Create: `src/skedl/adapters/emb/base.py`
- Create: `src/skedl/adapters/emb/sentence_transformers.py`
- Create: `tests/unit/test_adapters_optional.py`

**Step 1: Write the failing tests**

```python
def test_transformers_adapter_raises_clear_error_when_dependency_missing():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_adapters_optional.py -v`
Expected: FAIL due to missing adapter modules

**Step 3: Write minimal implementation**

```python
class TransformersLocalLLM(...): ...
class SentenceTransformerEmbedder(...): ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_adapters_optional.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/skedl/adapters/llm/base.py src/skedl/adapters/llm/transformers_local.py src/skedl/adapters/emb/base.py src/skedl/adapters/emb/sentence_transformers.py tests/unit/test_adapters_optional.py
git commit -m "feat: add local adapter interfaces and optional dependency guards"
```

### Task 9: Training/evaluation utilities and metrics (MPS aware)

**Files:**
- Create: `src/skedl/core/metrics.py`
- Create: `src/skedl/pipeline/trainer.py`
- Create: `src/skedl/pipeline/evaluator.py`
- Create: `src/skedl/pipeline/datasets.py`
- Create: `tests/unit/test_metrics.py`
- Create: `tests/unit/test_trainer.py`

**Step 1: Write the failing tests**

```python
def test_device_selector_prefers_mps_when_available():
    ...

def test_brier_and_ece_are_computed_for_binary_confidence():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_metrics.py tests/unit/test_trainer.py -v`
Expected: FAIL due to missing utilities

**Step 3: Write minimal implementation**

```python
def select_torch_device(...): ...
def brier_score(...): ...
class FusionTrainer: ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_metrics.py tests/unit/test_trainer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/skedl/core/metrics.py src/skedl/pipeline/trainer.py src/skedl/pipeline/evaluator.py src/skedl/pipeline/datasets.py tests/unit/test_metrics.py tests/unit/test_trainer.py
git commit -m "feat: add MPS-aware training and evaluation utilities"
```

### Task 10: CLI and end-to-end local demo path

**Files:**
- Create: `src/skedl/cli.py`
- Create: `examples/local_mock_demo.py`
- Create: `tests/integration/test_cli.py`

**Step 1: Write the failing tests**

```python
def test_cli_train_fusion_parses_mps_and_temperature_options():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_cli.py -v`
Expected: FAIL due to missing CLI

**Step 3: Write minimal implementation**

```python
def main(argv=None): ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/skedl/cli.py examples/local_mock_demo.py tests/integration/test_cli.py
git commit -m "feat: add SK-EDL CLI and local demo"
```

### Task 11: Full verification and polish

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-02-25-sk-edl-design.md` (if implementation notes need updates)

**Step 1: Run targeted test suites**

Run: `pytest tests/unit -v`
Expected: PASS

**Step 2: Run integration tests**

Run: `pytest tests/integration -v`
Expected: PASS (slow/live tests may be skipped)

**Step 3: Run full suite**

Run: `pytest -v`
Expected: PASS

**Step 4: Run local demo**

Run: `python -m skedl.cli demo --backend mock --num-cots 6 --temp-mode mixed`
Expected: prints answer, confidence, and diagnostic SK-EDL features

**Step 5: Commit**

```bash
git add README.md src tests examples
git commit -m "feat: ship local SK-EDL hybrid confidence scaffold"
```

## Execution Notes

- Follow strict TDD: no production code before the corresponding failing test.
- Use MPS-aware tests that monkeypatch availability checks instead of requiring hardware access.
- Keep optional dependencies guarded so test suite passes in minimal environments.
- Prefer mocked local inference traces for deterministic integration tests; add tiny-model smoke tests as optional/slow.
