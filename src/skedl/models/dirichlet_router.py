from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DirichletBranchOutput:
    features: dict[str, float]
    available: bool


def merge_dirichlet_features(
    logit_branch: DirichletBranchOutput | None,
    head_branch: DirichletBranchOutput | None,
) -> tuple[dict[str, float], list[float]]:
    features: dict[str, float] = {}
    mask: list[float] = []

    if logit_branch is None:
        mask.append(0.0)
    else:
        features.update(logit_branch.features)
        mask.append(1.0 if logit_branch.available else 0.0)

    if head_branch is None:
        mask.append(0.0)
    else:
        features.update(head_branch.features)
        mask.append(1.0 if head_branch.available else 0.0)

    return features, mask
