#!/usr/bin/env python3
"""Minimal FastAPI inference service for the trained sklearn model."""

from __future__ import annotations

import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

app = FastAPI(
    title="MLOps FastAPI CI Demo",
    description="Toy regression API: train with MLflow, serve with FastAPI, package with Docker.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    x: float = Field(..., description="Scalar input feature")


class PredictResponse(BaseModel):
    result: float


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run `python train.py` before starting the API."
        )
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


@app.on_event("startup")
def startup() -> None:
    app.state.model = _load_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    pred = float(model.predict([[payload.x]])[0])
    return PredictResponse(result=pred)
