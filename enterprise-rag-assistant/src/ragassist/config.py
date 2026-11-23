"""Central settings (overridable with environment variables)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = Path(os.getenv("RAG_CORPUS_DIR", ROOT / "corpus" / "pdf"))
    index_dir: Path = Path(os.getenv("RAG_INDEX_DIR", ROOT / "index"))
    reports_dir: Path = Path(os.getenv("RAG_REPORTS_DIR", ROOT / "reports"))

    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "openai/gpt-oss-20b")
    judge_model: str = os.getenv("RAG_JUDGE_MODEL", "qwen/qwen3.8-27b")  # other model family: less self-preference
    guard_model: str = os.getenv("RAG_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")

    chunk_size: int = 600  # characters
    chunk_overlap_sentences: int = 1
    candidates: int = 20  # retrieved by each retriever before fusion
    rerank_candidates: int = 12  # fused candidates scored by the cross-encoder
    top_k: int = 4  # chunks given to the LLM
    rrf_k: int = 60
    dense_weight: float = float(os.getenv("RAG_DENSE_WEIGHT", "0.9"))  # weighted RRF (0.5 = classic RRF)
    # Context gate: below this cross-encoder score (sigmoid, 0-1) the system refuses *without
    # calling the LLM*. Calibration (reports/eval_results.json) showed the reranker score does not
    # separate answerable from unanswerable questions well enough (best threshold would wrongly
    # refuse 15 % of answerable ones), so the gate is off by default and refusal is left to the LLM.
    min_rerank_score: float = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.0"))
    injection_threshold: float = 0.5

    refusal_message: str = "Je ne trouve pas cette information dans les documents disponibles."
    departments: tuple[str, ...] = field(default=("RH", "finance", "IT", "juridique"))


settings = Settings()
