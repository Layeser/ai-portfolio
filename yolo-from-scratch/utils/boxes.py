"""Bounding-box utilities: encoding, decoding, IoU, and non-maximum suppression."""

from __future__ import annotations

import torch


def encode_boxes(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    grid_size: int = 7,
    num_classes: int = 20,
    image_size: int = 448,
) -> torch.Tensor:
    """Encode ground-truth boxes into YOLOv1 target tensor.

    Args:
        boxes:  (N, 4) absolute [x_center, y_center, w, h] in pixels
        labels: (N,)   class indices (0-based)
        grid_size: S
        num_classes: C
        image_size: input image side length

    Returns:
        target (S, S, 5 + C) — one object per cell: when two centres fall in the same cell, the last one
        overwrites the earlier ones (13.5 % of VOC 2007 test objects, see scripts/explainer_figures.py)
    """
    target = torch.zeros(grid_size, grid_size, 5 + num_classes)
    cell_size = image_size / grid_size

    for box, label in zip(boxes, labels, strict=False):
        x, y, w, h = box.tolist()
        grid_x = min(int(x / cell_size), grid_size - 1)
        grid_y = min(int(y / cell_size), grid_size - 1)

        # Normalized offsets relative to the cell
        target[grid_y, grid_x, 0] = 1.0  # object present
        target[grid_y, grid_x, 1] = (x / cell_size) - grid_x
        target[grid_y, grid_x, 2] = (y / cell_size) - grid_y
        target[grid_y, grid_x, 3] = w / image_size
        target[grid_y, grid_x, 4] = h / image_size
        target[grid_y, grid_x, 5 + int(label)] = 1.0

    return target


def decode_predictions(
    predictions: torch.Tensor,
    grid_size: int = 7,
    num_boxes: int = 2,
    num_classes: int = 20,
    conf_threshold: float = 0.3,
    image_size: int = 448,
) -> list[dict[str, torch.Tensor]]:
    """Decode raw network output into a list of detections per batch item.

    Returns list of dicts with keys: boxes (N,4), scores (N,), labels (N,)
    in absolute pixel coordinates.
    """
    batch_size = predictions.size(0)
    results: list[dict[str, torch.Tensor]] = []

    for b in range(batch_size):
        pred = predictions[b]  # (S, S, B*5 + C)
        boxes_list: list[torch.Tensor] = []
        scores_list: list[float] = []
        labels_list: list[int] = []

        for i in range(grid_size):
            for j in range(grid_size):
                cell = pred[i, j]
                box_preds = cell[: num_boxes * 5].view(num_boxes, 5)
                class_scores = torch.softmax(cell[num_boxes * 5 :], dim=-1)  # head returns logits
                class_id = class_scores.argmax().item()
                class_prob = class_scores[class_id].item()

                for box in box_preds:
                    conf = box[4].item()
                    score = conf * class_prob
                    if score < conf_threshold:
                        continue

                    # Convert cell-relative coords to absolute pixels
                    cx = (j + box[0].item()) * (image_size / grid_size)
                    cy = (i + box[1].item()) * (image_size / grid_size)
                    w = box[2].item() * image_size
                    h = box[3].item() * image_size
                    boxes_list.append(torch.tensor([cx, cy, w, h]))
                    scores_list.append(score)
                    labels_list.append(class_id)

        if boxes_list:
            results.append(
                {
                    "boxes": torch.stack(boxes_list),
                    "scores": torch.tensor(scores_list),
                    "labels": torch.tensor(labels_list, dtype=torch.long),
                }
            )
        else:
            results.append(
                {
                    "boxes": torch.zeros(0, 4),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                }
            )

    return results


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU. boxes are (N,4) and (M,4) in [cx,cy,w,h] format."""
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2

    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    return inter / (union + 1e-6)


def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Greedy NMS returning indices of kept boxes."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long)
