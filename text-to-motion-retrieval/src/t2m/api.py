"""Motion search service: describe a movement in English, get the closest motions of the library.

The motion library is embedded once at start-up; a query costs one text-encoder pass and a
matrix product, so search over ~6k motions takes a few milliseconds.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from t2m.data import DATA_DIR, MotionTextDataset, Normalizer, motion_ids
from t2m.model import DualEncoder
from t2m.train import ARTIFACTS, embed


class SearchRequest(BaseModel):
    text: str = Field(min_length=3, max_length=500, examples=["two people shake hands and then walk away"])
    k: int = Field(5, ge=1, le=50)


class Match(BaseModel):
    motion_id: str
    score: float
    reference_description: str
    in_training_set: bool


class SearchResponse(BaseModel):
    matches: list[Match]
    latency_ms: float


class MotionSearch:
    def __init__(self, model: DualEncoder, motion_emb: torch.Tensor, ids: list[str], texts: dict, train_ids: set,
                 device: str):
        self.model, self.motion_emb, self.ids, self.texts = model, motion_emb, ids, texts
        self.train_ids, self.device = train_ids, device

    @classmethod
    def from_artifacts(cls) -> MotionSearch:
        import json

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DualEncoder().to(device)
        model.load_state_dict(torch.load(ARTIFACTS / "dual_encoder.pt", map_location=device))
        model.eval()
        split = json.loads((ARTIFACTS / "split.json").read_text())
        dataset = MotionTextDataset(motion_ids(), Normalizer.load(ARTIFACTS / "normalizer.npz"), train=False,
                                    data_dir=DATA_DIR)
        emb, ids = embed(model, dataset, device)
        return cls(model, emb, ids, dataset.texts, set(split["train"]), device)

    @torch.no_grad()
    def search(self, text: str, k: int) -> list[Match]:
        query = self.model.encode_text([text]).float().cpu()
        scores = (query @ self.motion_emb.T)[0].numpy()
        top = np.argsort(-scores)[:k]
        return [Match(motion_id=self.ids[i], score=round(float(scores[i]), 4),
                      reference_description=self.texts[self.ids[i]][0],
                      in_training_set=self.ids[i] in self.train_ids) for i in top]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "search", None) is None:
        app.state.search = MotionSearch.from_artifacts()
    yield


app = FastAPI(title="Text-to-motion search", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    search = getattr(request.app.state, "search", None)
    return {"status": "ok" if search else "loading", "motions_indexed": len(search.ids) if search else 0}


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    engine: MotionSearch | None = getattr(request.app.state, "search", None)
    if engine is None:
        raise HTTPException(503, "index not loaded")
    start = time.perf_counter()
    matches = engine.search(body.text, body.k)
    return SearchResponse(matches=matches, latency_ms=round((time.perf_counter() - start) * 1000, 2))
