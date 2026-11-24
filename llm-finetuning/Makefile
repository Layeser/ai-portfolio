PY ?= python
export PYTHONPATH := src
.PHONY: install data baseline train evaluate all figures serve test

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
serve:            ## POST /extract {"text": "..."}
	uvicorn receiptllm.api:app
test:
	$(PY) -m pytest -q
