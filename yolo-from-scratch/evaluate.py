"""PASCAL VOC mAP@0.5 evaluation for the YOLOv1 model.

    python evaluate.py --weights checkpoints/yolov1/best.pt          # VOC 2007 test set

* Detections are kept down to a 0.001 score (standard for mAP: the PR curve needs the low-score tail).
* NMS is applied **per class** (the inference default in detect.py is class-agnostic).
* AP uses all-point interpolation (VOC 2010+ protocol); ground truth comes from every annotated
  object, not from the YOLO grid target (which can only hold one object per cell).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import YOLOv1  # noqa: E402
from utils.boxes import box_iou, decode_predictions, non_max_suppression  # noqa: E402
from utils.dataset import YOLODataset  # noqa: E402
from utils.helpers import get_device, load_checkpoint, load_config  # noqa: E402

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou: float) -> torch.Tensor:
    """Per-class NMS: boxes of different classes never suppress each other."""
    keep = []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        keep.append(idx[non_max_suppression(boxes[idx], scores[idx], iou)])
    return torch.cat(keep) if keep else torch.zeros(0, dtype=torch.long)


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision envelope (all-point interpolation)."""
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    changes = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[changes + 1] - r[changes]) * p[changes + 1]))


def average_precision(detections: list[tuple[int, float, torch.Tensor]],
                      ground_truth: dict[int, torch.Tensor], iou_threshold: float = 0.5) -> float:
    """AP for one class. detections: (image_id, score, box); ground_truth: image_id -> (N, 4) boxes."""
    n_gt = sum(len(b) for b in ground_truth.values())
    if n_gt == 0:
        return float("nan")
    detections = sorted(detections, key=lambda d: -d[1])
    matched = {img: torch.zeros(len(b), dtype=torch.bool) for img, b in ground_truth.items()}
    tp = np.zeros(len(detections))
    for i, (img, _, box) in enumerate(detections):
        gt = ground_truth.get(img)
        if gt is None or len(gt) == 0:
            continue
        ious = box_iou(box[None], gt)[0]
        best = int(ious.argmax())
        if ious[best] >= iou_threshold and not matched[img][best]:
            tp[i], matched[img][best] = 1, True
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(1 - tp)
    return voc_ap(tp_cum / n_gt, tp_cum / np.maximum(tp_cum + fp_cum, 1e-9))


@torch.no_grad()
def evaluate(model, dataset: YOLODataset, cfg: dict, device, batch_size: int = 32) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    dets: dict[int, list] = {c: [] for c in range(cfg["model"]["num_classes"])}
    image_id = 0
    for images, _ in tqdm(loader, desc="eval", leave=False):
        preds = model(images.to(device)).float().cpu()
        for det in decode_predictions(preds, grid_size=cfg["model"]["grid_size"], num_boxes=cfg["model"]["num_boxes"],
                                      num_classes=cfg["model"]["num_classes"], conf_threshold=0.001,
                                      image_size=cfg["data"]["image_size"]):
            if len(det["boxes"]):
                keep = batched_nms(det["boxes"], det["scores"], det["labels"], cfg["inference"]["nms_iou_threshold"])
                for k in keep.tolist():
                    dets[int(det["labels"][k])].append((image_id, float(det["scores"][k]), det["boxes"][k]))
            image_id += 1

    gts: dict[int, dict[int, torch.Tensor]] = {c: {} for c in dets}
    for i in range(len(dataset)):
        boxes, labels = dataset.annotations(i)
        for c in labels.unique().tolist():
            gts[c][i] = boxes[labels == c]
    aps = {VOC_CLASSES[c]: round(average_precision(dets[c], gts[c]), 4) for c in dets}
    valid = [v for v in aps.values() if not np.isnan(v)]
    return {"mAP@0.5": round(float(np.mean(valid)), 4), "per_class_AP": aps, "n_images": len(dataset)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="reports/eval_voc2007_test.json")
    parser.add_argument("--max-images", type=int, default=0, help="evaluate on the first N test images (0 = all)")
    parser.add_argument("--device", default=None, help="override device (e.g. cpu while the GPU is training)")
    parser.add_argument("--split", choices=["test", "holdout"], default="test",
                        help="holdout = monitoring during training; test = final number only")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    device = get_device(args.device or cfg["train"]["device"])
    model = YOLOv1(num_classes=cfg["model"]["num_classes"], num_boxes=cfg["model"]["num_boxes"],
                   grid_size=cfg["model"]["grid_size"], backbone_type=cfg["model"]["backbone"],
                   pretrained_backbone=False).to(device)
    load_checkpoint(PROJECT_ROOT / args.weights, model, device=device)
    test_root = cfg["data"].get("test_root", cfg["data"]["val_root"])  # VOC 2007 test, never used for selection
    if args.split == "holdout":
        test_root = cfg["data"]["val_root"]
    dataset = YOLODataset(PROJECT_ROOT / test_root, image_size=cfg["data"]["image_size"],
                          grid_size=cfg["model"]["grid_size"], num_classes=cfg["model"]["num_classes"])
    if args.max_images:
        dataset.image_paths = dataset.image_paths[: args.max_images]
    result = evaluate(model, dataset, cfg, device)
    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
