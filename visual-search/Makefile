PY ?= python
export PYTHONPATH := src
.PHONY: install evaluate quick serve test export-weights upload-weights download-weights
install:        ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
evaluate:        ## train + evaluate the 4 encoders (GPU: ~45 min), figures + reports/results.json
	$(PY) -m vsearch.evaluate
quick:           ## pixels+PCA and a 2-epoch autoencoder, CPU-friendly
	$(PY) -m vsearch.evaluate --encoders pixels_pca autoencoder --ae-epochs 2
serve:           ## search API (best encoder by P@10)
	uvicorn vsearch.api:app --reload
test:
	$(PY) -m pytest -q
export-weights:  ## release/ + weights.sha256 (SimCLR encoder, gallery embeddings)
	$(PY) scripts/weights.py export
upload-weights:  ## attach release/ to the GitHub Release (gh CLI)
	$(PY) scripts/weights.py upload
download-weights: ## fetch + verify the released files (no training needed)
	$(PY) scripts/weights.py download
