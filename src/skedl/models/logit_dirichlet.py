from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skedl.core.dirichlet import (
    dirichlet_aleatoric_uncertainty,
    dirichlet_epistemic_uncertainty_with_plus_one,
)


EvidenceTransform = str


@dataclass(slots=True)
class LogitDirichletExtractor:
    top_k: int = 10
    evidence_transform: EvidenceTransform = "shift"
    aggregate_last_m: int = 3
    eps: float = 1e-8

    def _to_evidence(self, topk_logits: np.ndarray) -> np.ndarray:
        x = np.asarray(topk_logits, dtype=float)
        if x.ndim != 1:
            raise ValueError("topk logits must be 1D")
        if self.evidence_transform == "none":
            evidence = x
        elif self.evidence_transform == "relu":
            evidence = np.maximum(x, 0.0)
        elif self.evidence_transform == "shift":
            evidence = x - np.min(x)
        elif self.evidence_transform == "softplus":
            evidence = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        else:
            raise ValueError(f"unsupported evidence_transform: {self.evidence_transform}")

        evidence = np.asarray(evidence, dtype=float)
        evidence = np.clip(evidence, 0.0, None)
        if float(np.sum(evidence)) <= self.eps:
            evidence = np.ones_like(evidence)
        return evidence + self.eps

    def _token_au_eu(self, logits: np.ndarray) -> tuple[float, float]:
        scores = np.asarray(logits, dtype=float)
        if scores.ndim != 1:
            raise ValueError("logits must be a 1D vector")
        finite_mask = np.isfinite(scores)
        finite_scores = scores[finite_mask]
        if finite_scores.size == 0:
            finite_scores = np.zeros(1, dtype=float)
        k = min(self.top_k, finite_scores.size)
        if k <= 0:
            raise ValueError("top_k must be positive")
        idx = np.argpartition(finite_scores, -k)[-k:]
        topk = np.sort(finite_scores[idx])[::-1]
        alpha = self._to_evidence(topk)
        au = dirichlet_aleatoric_uncertainty(alpha)
        eu = dirichlet_epistemic_uncertainty_with_plus_one(alpha)
        return au, eu

    def extract_from_logits_steps(self, logits_steps: list[np.ndarray]) -> dict[str, float]:
        if not logits_steps:
            raise ValueError("logits_steps must be non-empty")

        token_au = []
        token_eu = []
        for step_logits in logits_steps:
            au, eu = self._token_au_eu(step_logits)
            token_au.append(au)
            token_eu.append(eu)

        au_arr = np.asarray(token_au, dtype=float)
        eu_arr = np.asarray(token_eu, dtype=float)

        last_m = min(self.aggregate_last_m, au_arr.size)
        au_last = float(np.mean(au_arr[-last_m:]))
        eu_last = float(np.mean(eu_arr[-last_m:]))

        # Simple bounded confidence proxy for fusion; raw AU/EU are also exposed.
        raw_risk = 0.5 * (float(np.mean(au_arr)) + float(np.mean(eu_arr)))
        c_d_logit = float(np.clip(np.exp(-raw_risk), 0.0, 1.0))

        return {
            "au_mean": float(np.mean(au_arr)),
            "au_max": float(np.max(au_arr)),
            "au_last_m": au_last,
            "eu_mean": float(np.mean(eu_arr)),
            "eu_max": float(np.max(eu_arr)),
            "eu_last_m": eu_last,
            "c_d_logit": c_d_logit,
            "num_token_steps": float(au_arr.size),
        }
