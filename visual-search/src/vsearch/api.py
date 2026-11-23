"""Image similarity search service.

    POST /search?k=8   (multipart image)  -> nearest gallery images with class and cosine score
    GET  /images/{id}.png                  -> a gallery image (to display results)

The encoder (default: the best one in reports/results.json, override with VSEARCH_ENCODER) embeds
the uploaded image; a FAISS inner-product index over the precomputed gallery returns neighbours.
"""

from __future__ import annotations

import io
import json
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from vsearch.data import CLASSES, ROOT, load_cifar10
from vsearch.encoders import ENCODERS
from vsearch.evaluate import ARTIFACTS, build_index, normalise


class Match(BaseModel):
    image_id: int
    label: str
    score: float


class SearchResponse(BaseModel):
    encoder: str
    matches: list[Match]
    latency_ms: float


class Engine:
    def __init__(self, name: str, encoder, gallery: np.ndarray, images: np.ndarray, labels: np.ndarray):
        self.name, self.encoder, self.images, self.labels = name, encoder, images, labels
        self.index = build_index(gallery)

    @classmethod
    def from_artifacts(cls) -> Engine:
        results = json.loads((ROOT / "reports" / "results.json").read_text())
        name = os.getenv("VSEARCH_ENCODER") or max(results, key=lambda n: results[n]["P@10"])
        encoder = ENCODERS[name]()
        if isinstance(encoder, torch.nn.Module) and (ARTIFACTS / f"{name}.pt").exists():
            encoder.load_state_dict(torch.load(ARTIFACTS / f"{name}.pt", map_location="cpu"))
        elif name == "pixels_pca":
            raise RuntimeError("pixels_pca has no saved model; choose a neural encoder with VSEARCH_ENCODER")
        train_images, train_labels, _, _ = load_cifar10()
        return cls(name, encoder, np.load(ARTIFACTS / f"gallery_{name}.npy"), train_images, train_labels)

    def search(self, image: Image.Image, k: int) -> list[Match]:
        x = np.asarray(image.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.uint8)[None]
        scores, ids = self.index.search(normalise(self.encoder.embed(x)), k)
        return [
            Match(image_id=int(i), label=CLASSES[self.labels[i]], score=round(float(s), 4))
            for s, i in zip(scores[0], ids[0], strict=True)
        ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "engine", None) is None:
        app.state.engine = Engine.from_artifacts()
    yield


app = FastAPI(title="Visual similarity search", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    engine: Engine = request.app.state.engine
    return {"status": "ok", "encoder": engine.name, "gallery_size": engine.index.ntotal}


@app.post("/search", response_model=SearchResponse)
async def search(request: Request, file: UploadFile = File(...), k: int = Query(8, ge=1, le=50)) -> SearchResponse:
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(422, "not a readable image") from exc
    engine: Engine = request.app.state.engine
    start = time.perf_counter()
    matches = engine.search(image, k)
    return SearchResponse(
        encoder=engine.name, matches=matches, latency_ms=round((time.perf_counter() - start) * 1000, 1)
    )


@app.get("/images/{image_id}.png")
def image(image_id: int, request: Request) -> Response:
    engine: Engine = request.app.state.engine
    if not 0 <= image_id < len(engine.images):
        raise HTTPException(404, "unknown image")
    buffer = io.BytesIO()
    Image.fromarray(engine.images[image_id]).resize((96, 96), Image.NEAREST).save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")
