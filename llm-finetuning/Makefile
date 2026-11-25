PY ?= python
export PYTHONPATH := src
.PHONY: install data baseline train evaluate all figures serve test export-weights upload-weights download-weights

install:          ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
data:             ## CORD v2 annotations only (~1.5 MB, no images) into data/raw
	$(PY) -m receiptllm.data
baseline:         ## Qwen2.5-1.5B-Instruct zero-shot on the test split
	$(PY) -m receiptllm.evaluate --split test --name base --batch 16
train:            ## LoRA, epoch chosen on the validation split -> artifacts/lora
	$(PY) -m receiptllm.train --epochs 3
evaluate:         ## fine-tuned model on the test split
	$(PY) -m receiptllm.evaluate --split test --name lora --adapter artifacts/lora --batch 16
all: baseline train evaluate figures
figures:
	$(PY) scripts/make_figures.py
export-weights:   ## adapter + test scores -> release/, checksums in weights.sha256
	$(PY) scripts/weights.py export
upload-weights:   ## attach release/ to the GitHub Release (gh CLI)
	$(PY) scripts/weights.py upload
download-weights: ## fetch + verify the released adapter (no training needed)
	$(PY) scripts/weights.py download
serve:            ## POST /extract {"text": "..."}
	uvicorn receiptllm.api:app
test:
	$(PY) -m pytest -q
