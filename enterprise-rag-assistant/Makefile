PY ?= python
export PYTHONPATH := src
.PHONY: install corpus index serve ui test eval eval-retrieval lint docker

install:        ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
corpus:           ## render the Markdown policies to PDF
	$(PY) scripts/build_corpus.py
index:            ## chunk the PDFs, embed them into Chroma, build BM25
	$(PY) -m ragassist.retrieval
serve:            ## REST API on http://127.0.0.1:8000/docs
	uvicorn ragassist.api:app --reload
ui:               ## Gradio chat on http://127.0.0.1:7860
	$(PY) -m ragassist.ui
test:             ## fast tests (no API key, no model download)
	$(PY) -m pytest -q
eval-retrieval:   ## retrieval ablation + refusal-gate calibration (no API key)
	$(PY) -m ragassist.evaluate --retrieval-only
eval:             ## full evaluation: answers, citations, refusals, faithfulness, guardrails
	$(PY) -m ragassist.evaluate
lint:
	ruff check src tests scripts
docker:
	docker compose up --build
