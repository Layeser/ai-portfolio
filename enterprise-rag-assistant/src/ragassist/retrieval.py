"""Hybrid retrieval: BM25 + dense embeddings (Chroma) fused with RRF, then cross-encoder reranking.

Why hybrid? Policy questions mix exact tokens ("CPF", "4242", "Kelvin")
where BM25 shines, and paraphrases ("my laptop was stolen" vs "loss or theft
of equipment") where embeddings shine. Reciprocal Rank Fusion merges the two
rankings without having to calibrate their scores. A cross-encoder then reads
(question, chunk) pairs jointly, which is slower but much more precise, so it
is only applied to the fused top candidates.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from ragassist.config import settings
from ragassist.ingest import Chunk, load_corpus
from ragassist.text import tokenize

Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float  # mode-specific: BM25 score, cosine, RRF score or cross-encoder logit
    rerank_score: float | None = None


def reciprocal_rank_fusion(rankings: list[list[str]], weights: list[float] | None = None,
                           k: int = settings.rrf_k) -> list[tuple[str, float]]:
    """score(d) = sum_i w_i / (k + rank_i(d)), rank starting at 1 (weights default to 1)."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder=None, reranker=None, collection=None):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])
        self.embedder = embedder
        self.reranker = reranker
        self.collection = collection

    # ---- construction ---------------------------------------------------------------------

    @classmethod
    def build(cls, index_dir: Path = settings.index_dir, with_models: bool = True) -> HybridRetriever:
        """Chunk the corpus, embed it into a persistent Chroma collection and save the chunks."""
        chunks = load_corpus()
        if index_dir.exists():
            shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True)
        (index_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8"
        )
        if not with_models:
            return cls(chunks)
        embedder, reranker = _load_models()
        collection = _chroma(index_dir, reset=True)
        passage_prefix = _prefixes(settings.embedding_model)[1]
        vectors = embedder.encode([passage_prefix + c.indexed_text for c in chunks], normalize_embeddings=True,
                                  batch_size=16)
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"doc": c.doc, "department": c.department, "section": c.section, "page": c.page}
                       for c in chunks],
        )
        return cls(chunks, embedder, reranker, collection)

    @classmethod
    def load(cls, index_dir: Path = settings.index_dir, with_models: bool = True) -> HybridRetriever:
        lines = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk(**json.loads(line)) for line in lines if line]
        if not with_models:
            return cls(chunks)
        embedder, reranker = _load_models()
        return cls(chunks, embedder, reranker, _chroma(index_dir))

    # ---- search ---------------------------------------------------------------------------

    def bm25_search(self, query: str, n: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:n]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def dense_search(self, query: str, n: int) -> list[Hit]:
        query_prefix = _prefixes(settings.embedding_model)[0]
        vector = self.embedder.encode([query_prefix + query], normalize_embeddings=True)[0]
        res = self.collection.query(query_embeddings=[vector.tolist()], n_results=min(n, len(self.chunks)))
        return [Hit(self.by_id[i], 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0], strict=True)]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self.reranker.predict([(query, h.chunk.indexed_text) for h in hits])
        for h, s in zip(hits, scores, strict=True):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)

    def search(self, query: str, k: int = settings.top_k, mode: Mode = "hybrid_rerank",
               department: str | None = None) -> list[Hit]:
        n = settings.candidates
        if mode == "bm25":
            hits = self.bm25_search(query, n)
        elif mode == "dense":
            hits = self.dense_search(query, n)
        else:
            fused = reciprocal_rank_fusion(
                [[h.chunk.chunk_id for h in self.bm25_search(query, n)],
                 [h.chunk.chunk_id for h in self.dense_search(query, n)]],
                weights=[1 - settings.dense_weight, settings.dense_weight],
            )
            hits = [Hit(self.by_id[cid], score) for cid, score in fused[:n]]
            if mode == "hybrid_rerank":
                hits = self.rerank(query, hits[: settings.rerank_candidates])
        if department:
            hits = [h for h in hits if h.chunk.department == department]
        return hits[:k]


def _prefixes(model_name: str) -> tuple[str, str]:
    """E5 models are trained with asymmetric 'query:' / 'passage:' prefixes."""
    return ("query: ", "passage: ") if "e5" in model_name.lower() else ("", "")


def _load_models():
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = _device()
    return (SentenceTransformer(settings.embedding_model, device=device),
            CrossEncoder(settings.reranker_model, device=device, max_length=512))


def _device() -> str:
    """GPU when available (12x faster reranking measured), CPU otherwise; RAG_DEVICE overrides."""
    import os

    import torch

    return os.getenv("RAG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _chroma(index_dir: Path, reset: bool = False):
    import chromadb

    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    if reset:
        try:
            client.delete_collection("policies")
        except Exception:  # collection did not exist
            pass
    return client.get_or_create_collection("policies", metadata={"hnsw:space": "cosine"})


if __name__ == "__main__":
    retriever = HybridRetriever.build()
    print(f"indexed {len(retriever.chunks)} chunks into {settings.index_dir}")
