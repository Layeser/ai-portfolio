#!/usr/bin/env python3
"""Run YOLOv1 inference on a single image or folder.

Usage:
    python infer.py --weights checkpoints/yolov1/best.pt --source path/to/image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import YOLOv1
from utils.boxes import decode_predictions, non_max_suppression
from utils.helpers import get_device, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv1 inference")
    parser.add_argument("--config", type=str, default="config/train.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--source", type=str, required=True, help="Image or directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["custom", "pretrained_resnet18", "pretrained_resnet50"],
        help="Backbone variant to use",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Use pretrained torchvision weights for the selected backbone",
    )
    return parser.parse_args()


def load_image(path: Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)
    device = get_device(cfg["train"]["device"])

    backbone_type = args.backbone or cfg["model"].get("backbone", "custom")
    pretrained_backbone = args.pretrained_backbone or bool(cfg["model"].get("pretrained_backbone", False))

    model = YOLOv1(
        num_classes=cfg["model"]["num_classes"],
        num_boxes=cfg["model"]["num_boxes"],
        grid_size=cfg["model"]["grid_size"],
        backbone_type=backbone_type,
        pretrained_backbone=pretrained_backbone,
    ).to(device)
    load_checkpoint(PROJECT_ROOT / args.weights, model, device=device)
    model.eval()

    source = Path(args.source)
    if source.is_dir():
        image_paths = list(source.glob("*.jpg")) + list(source.glob("*.png"))
    else:
        image_paths = [source]

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg["data"]["image_size"]
    conf_threshold = cfg["inference"]["conf_threshold"]
    nms_threshold = cfg["inference"]["nms_iou_threshold"]

    for img_path in image_paths:
        tensor = load_image(img_path, image_size).to(device)
        with torch.no_grad():
            preds = model(tensor)

        detections = decode_predictions(
            preds,
            grid_size=cfg["model"]["grid_size"],
            num_boxes=cfg["model"]["num_boxes"],
            num_classes=cfg["model"]["num_classes"],
            conf_threshold=conf_threshold,
            image_size=image_size,
        )[0]

        if detections["boxes"].numel() > 0:
            keep = non_max_suppression(
                detections["boxes"],
                detections["scores"],
                iou_threshold=nms_threshold,
            )
            boxes = detections["boxes"][keep]
            scores = detections["scores"][keep]
            labels = detections["labels"][keep]
            print(f"{img_path.name}: {len(boxes)} detections")
            for box, score, label in zip(boxes, scores, labels, strict=False):
                print(f"  class={label.item()} score={score.item():.3f} box={box.tolist()}")
        else:
            print(f"{img_path.name}: no detections")


if __name__ == "__main__":
    main()
