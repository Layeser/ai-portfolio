PY ?= python
export PYTHONPATH := src
.PHONY: install proposals train-faster train-fast train-rcnn evaluate figures test export-weights upload-weights download-weights

install:          ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
proposals:        ## Selective Search boxes for VOC07 trainval + test (CPU, ~1 h with 8 workers)
	$(PY) scripts/selective_search.py --workers 8
train-faster:     ## Faster R-CNN, 12 epochs (~18 min/epoch on an 8 GB laptop GPU)
	$(PY) -m twostage.train --model faster_rcnn
train-fast:       ## Fast R-CNN on Selective Search proposals
	$(PY) -m twostage.train --model fast_rcnn
train-rcnn:       ## R-CNN: one CNN pass per warped proposal
	$(PY) -m twostage.train --model rcnn
evaluate:         ## VOC 2007 test mAP of the three best checkpoints
	for m in rcnn fast_rcnn faster_rcnn; do $(PY) -m twostage.evaluate --model $$m --weights checkpoints/$$m/best.pt; done
export-weights:   ## float16 inference checkpoints + test results -> release/, checksums in weights.sha256
	$(PY) scripts/export_fp16.py
	$(PY) scripts/weights.py export
upload-weights:   ## attach release/ to the GitHub Release (gh CLI)
	$(PY) scripts/weights.py upload
download-weights: ## fetch + verify the released weights (no training needed)
	$(PY) scripts/weights.py download
figures:
	$(PY) scripts/make_figures.py
test:             ## CPU tests on synthetic data
	$(PY) -m pytest -q
