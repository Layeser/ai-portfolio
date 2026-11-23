"""Model wrapper shared by training, the scoring API and the investigation agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraudguard.features import FEATURES

MODEL_FILE = "model.txt"
META_FILE = "model_meta.json"


@dataclass
class FraudModel:
    booster: lgb.Booster
    meta: dict

    @property
    def review_threshold(self) -> float:
        return self.meta["thresholds"]["review"]

    @property
    def block_threshold(self) -> float:
        return self.meta["thresholds"]["block"]

    @property
    def version(self) -> str:
        return self.meta["version"]

    @classmethod
    def load(cls, artifacts_dir: Path) -> FraudModel:
        booster = lgb.Booster(model_file=str(artifacts_dir / MODEL_FILE))
        meta = json.loads((artifacts_dir / META_FILE).read_text())
        return cls(booster, meta)

    def save(self, artifacts_dir: Path) -> None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(artifacts_dir / MODEL_FILE))
        (artifacts_dir / META_FILE).write_text(json.dumps(self.meta, indent=2))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(features[FEATURES].astype(float))

    def decide(self, score: float) -> str:
        if score >= self.block_threshold:
            return "block"
        if score >= self.review_threshold:
            return "review"
        return "allow"

    def explain(self, features: pd.DataFrame, top_k: int = 3) -> list[list[dict]]:
        """Top-k SHAP contributions (log-odds) pushing each prediction towards fraud."""
        contrib = self.booster.predict(features[FEATURES].astype(float), pred_contrib=True)[:, :-1]
        values = features[FEATURES].to_numpy()
        reasons = []
        for row_contrib, row_values in zip(contrib, values, strict=True):
            order = np.argsort(-row_contrib)[:top_k]
            reasons.append(
                [
                    {
                        "feature": FEATURES[i],
                        "value": round(float(row_values[i]), 3),
                        "contribution": round(float(row_contrib[i]), 3),
                    }
                    for i in order
                    if row_contrib[i] > 0
                ]
            )
        return reasons
