PY ?= python
.PHONY: install prepare train resume evaluate export-weights upload-weights download-weights serve test

install:          ## dependencies into the active (shared) venv
	$(PY) -m pip install -r requirements.txt
prepare:          ## download + convert VOC 2007/2012, carve the 1,000-image holdout
	$(PY) prepare_voc.py && $(PY) scripts/make_holdout.py
train:            ## fresh training run (GPU, ~7.5 min/epoch on an 8 GB card)
	$(PY) train.py --config config/train.yaml --fresh
resume:
	$(PY) train.py --config config/train.yaml --resume-from checkpoints/yolov1/last.pt
evaluate:         ## VOC 2007 test mAP@0.5 of the checkpoint selected on the holdout
	$(PY) evaluate.py --weights checkpoints/yolov1/best.pt
export-weights:   ## light fp16 inference checkpoint + release/ + weights.sha256
	$(PY) scripts/export_fp16.py checkpoints/yolov1/best.pt checkpoints/yolov1/yolov1_voc2007_fp16.pt
	$(PY) scripts/weights.py export
upload-weights:   ## attach release/ to the GitHub Release (gh CLI)
	$(PY) scripts/weights.py upload
download-weights: ## fetch + verify the released weights (no training needed)
	$(PY) scripts/weights.py download
serve:            ## detection API with the released weights
	YOLO_WEIGHTS=checkpoints/yolov1/yolov1_voc2007_fp16.pt uvicorn api:app
test:
	$(PY) -m pytest -q
