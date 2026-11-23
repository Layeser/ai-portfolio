"""Categorisation service.

POST /classify returns the predicted leaf with its full path, the top-3 alternatives and a routing
decision: `auto` when confidence reaches the threshold calibrated for 95 % accuracy, otherwise
`review`, together with the deepest category that *is* confident (partial categorisation).
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from prodcat.data import ROOT, TAXONOMY_PATH, build_text
from prodcat.models import load_model
from prodcat.taxonomy import Taxonomy

ARTIFACTS = Path(os.getenv("PRODCAT_ARTIFACTS_DIR", ROOT / "artifacts"))


class Product(BaseModel):
    title: str = Field(min_length=2, max_length=1000, examples=["Caso WineComfort 38 Cave à vin 2 zones"])
    description: str = Field("", max_length=20_000)
    brand: str = ""
    color: str = ""


class Candidate(BaseModel):
    category_id: str
    path: list[str]
    probability: float


class Classification(BaseModel):
    category_id: str
    path: list[str]
    confidence: float
    decision: Literal["auto", "review"]
    deepest_confident_category: str
    deepest_confident_probability: float
    top3: list[Candidate]
    model: str


class Service:
    def __init__(self, model, taxonomy: Taxonomy, threshold: float):
        self.model, self.taxonomy, self.threshold = model, taxonomy, threshold

    @classmethod
    def from_artifacts(cls, artifacts: Path = ARTIFACTS) -> Service:
        model = load_model(artifacts)
        threshold = json.loads((artifacts / model.name / "threshold.json").read_text())["auto_threshold"]
        return cls(model, Taxonomy.load(TAXONOMY_PATH), threshold)

    def classify(self, products: list[Product]) -> list[Classification]:
        frame = pd.DataFrame([p.model_dump() for p in products])
        probs = self.model.predict_proba(build_text(frame))
        classes = self.model.classes_
        results = []
        for row in probs:
            order = np.argsort(-row)[:3]
            leaf, conf = classes[order[0]], float(row[order[0]])
            node, node_p = self.taxonomy.deepest_confident_node(row, classes, self.threshold)
            results.append(Classification(
                category_id=leaf, path=self.taxonomy.path(leaf), confidence=round(conf, 4),
                decision="auto" if conf >= self.threshold else "review",
                deepest_confident_category=node, deepest_confident_probability=round(node_p, 4),
                top3=[Candidate(category_id=classes[i], path=self.taxonomy.path(classes[i]),
                                probability=round(float(row[i]), 4)) for i in order],
                model=self.model.name,
            ))
        return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is None:  # tests inject a service
        app.state.service = Service.from_artifacts()
    yield


app = FastAPI(title="Product categorisation API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    service: Service | None = getattr(request.app.state, "service", None)
    return {"status": "ok" if service else "loading", "model": service.model.name if service else None,
            "auto_threshold": service.threshold if service else None}


@app.post("/classify", response_model=Classification)
def classify(product: Product, request: Request) -> Classification:
    return _service(request).classify([product])[0]


@app.post("/classify/batch", response_model=list[Classification])
def classify_batch(products: list[Product], request: Request) -> list[Classification]:
    if not 1 <= len(products) <= 256:
        raise HTTPException(422, "batch size must be between 1 and 256")
    return _service(request).classify(products)


def _service(request: Request) -> Service:
    service: Service | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "model not loaded")
    return service
