#!/usr/bin/env python3
"""Run YOLOv1 detection on one image or a folder of images and save annotated outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import YOLOv1
from utils.boxes import decode_predictions, non_max_suppression
from utils.helpers import get_device, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv1 detection on images")
    parser.add_argument("--config", type=str, default="config/train.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--source", type=str, required=True, help="Image file or folder")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--conf-threshold", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=None, help="Override NMS IoU threshold")
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


def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size

    scale = min(image_size / image.width, image_size / image.height)
    new_width = max(1, int(round(image.width * scale)))
    new_height = max(1, int(round(image.height * scale)))
    resized = image.resize((new_width, new_height), Image.BILINEAR)

    padded = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    offset_x = (image_size - new_width) // 2
    offset_y = (image_size - new_height) // 2
    padded.paste(resized, (offset_x, offset_y))

    # Must match training preprocessing (ImageNet normalization) exactly.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    return transform(padded).unsqueeze(0), original_size


def scale_boxes_to_image(
    boxes: torch.Tensor,
    original_size: tuple[int, int],
    model_size: int,
) -> torch.Tensor:
    """Map boxes from the model's padded square input back to the original image size."""
    if boxes.numel() == 0:
        return boxes

    scaled = boxes.clone()
    original_width, original_height = original_size
    scale = min(model_size / original_width, model_size / original_height)
    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))
    pad_x = (model_size - new_width) // 2
    pad_y = (model_size - new_height) // 2

    scaled[:, 0] = (scaled[:, 0] - pad_x) / scale
    scaled[:, 1] = (scaled[:, 1] - pad_y) / scale
    scaled[:, 2] = scaled[:, 2] / scale
    scaled[:, 3] = scaled[:, 3] / scale

    scaled[:, 0] = scaled[:, 0].clamp(0, original_width)
    scaled[:, 1] = scaled[:, 1].clamp(0, original_height)
    scaled[:, 2] = scaled[:, 2].clamp(0, original_width)
    scaled[:, 3] = scaled[:, 3].clamp(0, original_height)
    return scaled


def draw_boxes(image: Image.Image, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box, score, label in zip(boxes, scores, labels):
        x_center, y_center, w, h = box.tolist()
        x1 = max(0, x_center - w / 2)
        y1 = max(0, y_center - h / 2)
        x2 = min(image.width, x_center + w / 2)
        y2 = min(image.height, y_center + h / 2)

        label_name = class_names[label.item()] if 0 <= label.item() < len(class_names) else str(label.item())
        text = f"{label_name} {score.item():.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if font is not None:
            draw.text((x1 + 3, max(0, y1 - 16)), text, fill="red", font=font)
        else:
            draw.text((x1 + 3, max(0, y1 - 16)), text, fill="red")

    return image


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
        image_paths = sorted(list(source.glob("*.jpg")) + list(source.glob("*.png")) + list(source.glob("*.jpeg")))
    else:
        image_paths = [source]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {source}")

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg["data"]["image_size"]
    conf_threshold = args.conf_threshold if args.conf_threshold is not None else cfg["inference"]["conf_threshold"]
    nms_threshold = args.nms_threshold if args.nms_threshold is not None else cfg["inference"]["nms_iou_threshold"]

    # PASCAL VOC class names (matches the default dataset)
    class_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        tensor, original_size = load_image(img_path, image_size)
        tensor = tensor.to(device)

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
        else:
            boxes = detections["boxes"]
            scores = detections["scores"]
            labels = detections["labels"]

        boxes = scale_boxes_to_image(boxes, original_size, image_size)
        annotated = draw_boxes(image.copy(), boxes, scores, labels, class_names)
        out_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}"
        annotated.save(out_path)
        print(f"{img_path.name}: {len(boxes)} detections -> {out_path}")


if __name__ == "__main__":
    main()
