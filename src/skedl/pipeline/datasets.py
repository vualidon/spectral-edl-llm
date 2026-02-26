from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class FeatureDataset:
    features: np.ndarray
    labels: np.ndarray


def load_feature_jsonl(path: str | Path) -> FeatureDataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("empty dataset")
    features = np.asarray([row["features"] for row in rows], dtype=np.float32)
    labels = np.asarray([row["label"] for row in rows], dtype=np.float32)
    return FeatureDataset(features=features, labels=labels)
