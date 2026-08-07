# YOLOv1 from Scratch

End-to-end **YOLOv1** object detection pipeline in PyTorch — model, loss, VOC preparation, training, inference, and tests.

Built as a systems / Computer Vision exercise (not a Ultralytics wrapper): you can read every module from backbone to NMS.

## Features

- YOLOv1 head on **ResNet-18 / ResNet-50** ImageNet backbones
- Classic YOLO loss (\(\lambda_{coord}\), \(\lambda_{noobj}\))
- PASCAL VOC download + conversion (`prepare_voc.py`)
- Checkpointing / resume, cosine LR + warmup
- Inference with confidence threshold + NMS
- Unit tests (pytest) for boxes, resume logic, training stability helpers

## Tech stack

PyTorch · torchvision · PyYAML · Pillow · NumPy · tqdm · pytest

## Layout

```
yolo-from-scratch/
├── src/                 # backbone, neck, head, loss, model
├── utils/               # dataset, VOC, boxes, helpers
├── config/train.yaml
├── train.py / detect.py / infer.py / prepare_voc.py
├── tests/
└── images/              # tiny sample images for demo inference
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train on PASCAL VOC

```bash
# Download + convert VOC 2007 (~1 GB)
python prepare_voc.py

# Train (auto-prepares data if missing when configured)
python train.py --config config/train.yaml

# Resume
python train.py --config config/train.yaml --resume-from checkpoints/yolov1/last.pt
```

Tip: start with `max_train_samples: 500` in `config/train.yaml` for a quick smoke run.

## Inference

```bash
python detect.py --weights checkpoints/yolov1/best.pt --source images/000009.jpg
# or
python infer.py --weights checkpoints/yolov1/best.pt --source images/
```

## Tests

```bash
pytest -q
```

## Results

Training checkpoints are **not** shipped (too large). Reproduce locally; report mAP on VOC if you evaluate with an external mAP script.

## Why publish this

Shows low-level CV engineering: custom loss, data pipeline, training loop — complementary to fine-tuning modern YOLO APIs.
