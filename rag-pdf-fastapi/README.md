# RAG PDF Assistant (FastAPI)

Retrieval-Augmented Generation API: ingest a PDF → embed chunks (MiniLM) → FAISS index → grounded LLM answers via OpenRouter-compatible API.

## Features

- FastAPI endpoints: `/api/health`, `/api/docs`, `/api/chat`
- PDF chunking + SentenceTransformers embeddings + FAISS
- Config via environment (`.env`) — **no keys in code**
- Optional CLI under `cli/` (`build_index.py`, `chat.py`)

## Stack

Python · FastAPI · FAISS · SentenceTransformers · OpenAI SDK (OpenRouter) · pypdf

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set API_KEY

# Build vector store (from project root)
python -m backend.rag.build_vs   # or follow scripts/Makefile if present

uvicorn backend.api.main:app --reload
# Docs: http://127.0.0.1:8000/docs
```

Place your PDF at `data/sample.pdf` (a sample document is included when available).

## Architecture

```
PDF → chunking → MiniLM embeddings → FAISS
                                      ↓
User query → retrieve top-k → LLM generate answer
```

## Security

- Never commit `.env`
- Rotate any key that was previously hardcoded in older notebooks elsewhere
