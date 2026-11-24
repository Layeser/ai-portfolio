"""Extraction service: POST /extract {"text": "<receipt OCR text>"} -> {"data": {...}, "valid_json": bool}.

The LoRA adapter is merged into the base weights at start-up, so inference costs the same as the base model.
    RECEIPT_ADAPTER=artifacts/lora uvicorn receiptllm.api:app
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from receiptllm.metrics import parse_json

state: dict = {}


class Receipt(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class Extraction(BaseModel):
    data: dict | None
    valid_json: bool
    raw: str


def load_model() -> None:
    from receiptllm.data import ROOT
    from receiptllm.model import BASE, load

    adapter = os.environ.get("RECEIPT_ADAPTER", "artifacts/lora")
    state["tok"], state["model"] = load(os.environ.get("RECEIPT_BASE", BASE), str(ROOT / adapter))


def run(text: str) -> str:
    from receiptllm.model import generate

    return generate(state["tok"], state["model"], [text])[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "extract" not in state:  # tests inject a fake extractor
        await asyncio.to_thread(load_model)
        state["extract"] = run
    state["lock"] = asyncio.Lock()
    yield


app = FastAPI(title="Receipt extraction (fine-tuned LLM)", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/extract", response_model=Extraction)
async def extract(receipt: Receipt) -> Extraction:
    # one GPU: generation runs in a worker thread so the event loop stays free, one request at a time
    async with state["lock"]:
        raw = await asyncio.to_thread(state["extract"], receipt.text)
    data = parse_json(raw)
    return Extraction(data=data, valid_json=data is not None, raw=raw)
