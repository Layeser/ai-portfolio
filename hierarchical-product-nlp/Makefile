PY ?= python
export PYTHONPATH := src
.PHONY: install train-tfidf train-xlmr report serve test lint docker

install:        ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
train-tfidf:    ## baseline, a few minutes on CPU (needs data/trainset.csv and data/testset.csv)
	$(PY) -m prodcat.train --model tfidf
train-xlmr:     ## fine-tune XLM-RoBERTa on GPU
	$(PY) -m prodcat.train --model xlmr --epochs 3
report:         ## comparison figures + reports/metrics.json
	$(PY) -m prodcat.train --report
serve:          ## API on http://127.0.0.1:8000/docs (best available model)
	uvicorn prodcat.api:app --reload
test:
	$(PY) -m pytest -q
lint:
	ruff check src tests
docker:
	docker build -t prodcat . && docker run --rm -p 8000:8000 -v $$PWD/artifacts:/app/artifacts prodcat

export-weights:  ## gather trained files into release/ + weights.sha256
	$(PY) scripts/weights.py export
upload-weights:  ## publish release/ (GitHub Release via gh CLI, or Hugging Face Hub with HF_TOKEN)
	$(PY) scripts/weights.py upload
download-weights: ## fetch + verify the published weights (no training needed)
	$(PY) scripts/weights.py download
