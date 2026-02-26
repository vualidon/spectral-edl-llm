from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skedl.core.entropy import kernel_von_neumann_entropy
from skedl.core.graph import symmetrized_knn_graph
from skedl.core.kernel import cosine_kernel
from skedl.core.spectral import connectivity_risk, lambda2_normalized_laplacian
from skedl.models.logit_dirichlet import LogitDirichletExtractor
from skedl.schemas import CoTSample, SKEDLFeatureResult


@dataclass(slots=True)
class SKEDLFeatureExtractor:
    k: int = 3
    tau: float = 5.0
    top_k_dirichlet: int = 10
    kernel: str = "cosine"

    def _kernel(self, embeddings: np.ndarray) -> np.ndarray:
        if self.kernel != "cosine":
            raise ValueError(f"unsupported kernel: {self.kernel}")
        return cosine_kernel(embeddings)

    def extract(self, samples: list[CoTSample]) -> SKEDLFeatureResult:
        if not samples:
            raise ValueError("samples must be non-empty")

        embeddings = []
        per_cot_logit_features: list[dict[str, float]] = []
        logit_extractor = LogitDirichletExtractor(top_k=self.top_k_dirichlet)

        for sample in samples:
            if sample.embedding is None:
                raise ValueError("each sample must contain an embedding")
            embeddings.append(np.asarray(sample.embedding, dtype=float))

            logits_steps = []
            for step in sample.steps:
                if step.logits is not None:
                    logits_steps.append(np.asarray(step.logits, dtype=float))
            if logits_steps:
                per_cot_logit_features.append(logit_extractor.extract_from_logits_steps(logits_steps))

        emb_mat = np.vstack(embeddings)
        kernel = self._kernel(emb_mat)
        h_ker = kernel_von_neumann_entropy(kernel)

        graph = symmetrized_knn_graph(kernel, k=min(self.k, len(samples) - 1), include_self=False)
        lam2 = lambda2_normalized_laplacian(graph)
        r_lambda = connectivity_risk(lam2, tau=self.tau)

        features: dict[str, float] = {
            "num_cots": float(len(samples)),
            "kernel_entropy": float(h_ker),
            "lambda2": float(lam2),
            "connectivity_risk": float(r_lambda),
        }

        if per_cot_logit_features:
            keys = sorted(per_cot_logit_features[0].keys())
            for key in keys:
                vals = [f[key] for f in per_cot_logit_features if key in f]
                if vals:
                    features[key] = float(np.mean(vals))
            features["logit_dirichlet_available"] = 1.0
        else:
            features["logit_dirichlet_available"] = 0.0

        diagnostics = {
            "kernel_matrix": kernel,
            "graph_matrix": graph,
            "logit_feature_count": len(per_cot_logit_features),
        }
        return SKEDLFeatureResult(features=features, diagnostics=diagnostics)
