"""PASCAL VOC mAP@0.5 (official protocol: difficult objects are ignored, one match per ground truth).

Both AP definitions are reported: the VOC 2007 11-point metric (used in the R-CNN, Fast and Faster
R-CNN papers, so the numbers are comparable) and the all-point metric used since VOC 2010.

    python -m twostage.evaluate --model faster_rcnn --weights checkpoints/faster_rcnn/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from twostage.voc import CLASSES, ROOT, VOCDetection, collate, load_annotation, split_ids


def average_precision(recall: np.ndarray, precision: np.ndarray, use_07_metric: bool) -> float:
    if use_07_metric:
        return float(np.mean([precision[recall >= t].max() if (recall >= t).any() else 0.0
                              for t in np.arange(0, 1.1, 0.1)]))
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def voc_map(detections: dict, image_ids: list[str], iou_threshold: float = 0.5) -> dict:
    """detections[class_index] = list of (image_id, score, x1, y1, x2, y2) in original pixels."""
    results = {"AP07": {}, "AP": {}}
    for c, name in enumerate(CLASSES, start=1):
        gts, n_pos = {}, 0
        for image_id in image_ids:
            ann = load_annotation(image_id)
            m = ann["labels"] == c
            gts[image_id] = {"boxes": ann["boxes"][m], "difficult": ann["difficult"][m],
                             "used": np.zeros(m.sum(), dtype=bool)}
            n_pos += int((~ann["difficult"][m]).sum())
        dets = sorted(detections.get(c, []), key=lambda d: -d[1])
        tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
        for k, (image_id, _, *box) in enumerate(dets):
            g = gts[image_id]
            if len(g["boxes"]) == 0:
                fp[k] = 1
                continue
            b = np.array(box)
            ix1, iy1 = np.maximum(g["boxes"][:, 0], b[0]), np.maximum(g["boxes"][:, 1], b[1])
            ix2, iy2 = np.minimum(g["boxes"][:, 2], b[2]), np.minimum(g["boxes"][:, 3], b[3])
            inter = np.clip(ix2 - ix1 + 1, 0, None) * np.clip(iy2 - iy1 + 1, 0, None)  # VOC devkit: +1 pixels
            union = ((b[2] - b[0] + 1) * (b[3] - b[1] + 1)
                     + (g["boxes"][:, 2] - g["boxes"][:, 0] + 1) * (g["boxes"][:, 3] - g["boxes"][:, 1] + 1) - inter)
            iou = inter / union
            j = int(iou.argmax())
            if iou[j] >= iou_threshold:
                if g["difficult"][j]:
                    continue  # neither TP nor FP
                if not g["used"][j]:
                    tp[k], g["used"][j] = 1, True
                else:
                    fp[k] = 1
            else:
                fp[k] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recall = tp / max(n_pos, 1)
        precision = tp / np.maximum(tp + fp, np.finfo(float).eps)
        results["AP07"][name] = round(average_precision(recall, precision, True), 4)
        results["AP"][name] = round(average_precision(recall, precision, False), 4)
    return {
        "mAP07": round(float(np.mean(list(results["AP07"].values()))), 4),
        "mAP": round(float(np.mean(list(results["AP"].values()))), 4),
        "per_class_AP07": results["AP07"],
        "n_images": len(image_ids),
    }


@torch.no_grad()
def evaluate_model(model, split: str = "holdout", max_images: int = 0, device: str | None = None,
                   workers: int = 4) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ids = split_ids(split)[: max_images or None]
    loader = DataLoader(VOCDetection(ids, train=False, proposals=model.needs_proposals), batch_size=1,
                        collate_fn=collate, num_workers=workers)
    model.eval()
    detections, start = defaultdict(list), time.time()
    for images, targets in loader:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model([im.to(device) for im in images], [_to(t, device) for t in targets])
        for det, t in zip(out, targets, strict=True):
            boxes = (det["boxes"].float() / t["scale"]).cpu().numpy()
            for box, score, label in zip(boxes, det["scores"].float().cpu().numpy(), det["labels"].cpu().numpy(),
                                         strict=True):
                detections[int(label)].append((t["image_id"], float(score), *box.tolist()))
    seconds = (time.time() - start) / len(ids)
    result = voc_map(detections, ids)
    result["seconds_per_image"] = round(seconds, 3)
    result["split"] = split
    return result


def _to(target: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()}


def main() -> None:
    from twostage.models import build

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rcnn", "fast_rcnn", "faster_rcnn"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="test", choices=["test", "holdout"])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # CPU works, about 10-30x slower
    model = build(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device)["model"])
    result = evaluate_model(model, args.split, args.max_images, device)
    print(json.dumps({k: v for k, v in result.items() if k != "per_class_AP07"}, indent=2))
    output = args.output or ROOT / "reports" / f"eval_{args.model}_{args.split}.json"
    with open(output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
