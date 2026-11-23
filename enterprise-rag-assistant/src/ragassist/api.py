"""REST API: POST /ask returns a cited answer (or an explicit refusal)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ragassist.pipeline import RAGPipeline, build_pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["Quel est le plafond d'un repas à l'étranger ?"])
    department: Literal["RH", "finance", "IT", "juridique"] | None = None


class CitationOut(BaseModel):
    n: int
    doc: str
    section: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    timings_ms: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "pipeline", None) is None:  # tests inject their own pipeline
        app.state.pipeline = build_pipeline()
    yield


app = FastAPI(title="Atelier Nord policy assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    return {"status": "ok" if pipeline else "loading",
            "chunks_indexed": len(pipeline.retriever.chunks) if pipeline else 0}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "index not loaded")
    result = pipeline.ask(body.question, body.department)
    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        citations=[CitationOut(**{k: getattr(c, k) for k in ("n", "doc", "section", "page", "snippet")})
                   for c in result.citations],
        timings_ms=result.timings_ms,
    )
