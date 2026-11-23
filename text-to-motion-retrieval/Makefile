PY ?= python
export PYTHONPATH := src
.PHONY: install train report serve test
install:        ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
train:
	$(PY) -m t2m.train --epochs 40
report:
	$(PY) -m t2m.report
serve:
	uvicorn t2m.api:app --reload
test:
	$(PY) -m pytest -q

export-weights:  ## gather trained files into release/ + weights.sha256
	$(PY) scripts/weights.py export
upload-weights:  ## publish release/ (GitHub Release via gh CLI, or Hugging Face Hub with HF_TOKEN)
	$(PY) scripts/weights.py upload
download-weights: ## fetch + verify the published weights (no training needed)
	$(PY) scripts/weights.py download
