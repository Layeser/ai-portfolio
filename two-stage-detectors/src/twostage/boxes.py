"""Box geometry shared by the three detectors: IoU, regression targets, sampling."""

from __future__ import annotations

import math

import torch

BBOX_CLIP = math.log(1000.0 / 16)


def box_area(b: torch.Tensor) -> torch.Tensor:
    return (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, 4) x (M, 4) -> (N, M) intersection over union, boxes as (x1, y1, x2, y2)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    return inter / (box_area(a)[:, None] + box_area(b)[None] - inter).clamp(min=1e-6)


def encode(gt: torch.Tensor, ref: torch.Tensor, weights=(1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    """Regression targets (dx, dy, dw, dh) that turn `ref` boxes into `gt` boxes (R-CNN parameterisation)."""
    wx, wy, ww, wh = weights
    rw, rh = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    rx, ry = ref[:, 0] + 0.5 * rw, ref[:, 1] + 0.5 * rh
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    return torch.stack([wx * (gx - rx) / rw, wy * (gy - ry) / rh, ww * torch.log(gw / rw), wh * torch.log(gh / rh)], 1)


def decode(deltas: torch.Tensor, ref: torch.Tensor, weights=(1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    """Inverse of `encode`; deltas may hold several boxes per reference: (N, 4k)."""
    wx, wy, ww, wh = weights
    rw, rh = (ref[:, 2] - ref[:, 0])[:, None], (ref[:, 3] - ref[:, 1])[:, None]
    rx, ry = ref[:, 0][:, None] + 0.5 * rw, ref[:, 1][:, None] + 0.5 * rh
    dx, dy = deltas[:, 0::4] / wx, deltas[:, 1::4] / wy
    dw, dh = (deltas[:, 2::4] / ww).clamp(max=BBOX_CLIP), (deltas[:, 3::4] / wh).clamp(max=BBOX_CLIP)
    cx, cy, w, h = dx * rw + rx, dy * rh + ry, torch.exp(dw) * rw, torch.exp(dh) * rh
    out = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], 2)
    return out.reshape(deltas.shape)


def clip(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp(0, width - 1)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(0, height - 1)
    return boxes


def sample(labels: torch.Tensor, num: int, positive_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Random subset of positives (label > 0) and negatives (label == 0); label -1 is ignored."""
    pos = torch.nonzero(labels > 0).flatten()
    neg = torch.nonzero(labels == 0).flatten()
    n_pos = min(len(pos), int(num * positive_fraction))
    n_neg = min(len(neg), num - n_pos)
    pos = pos[torch.randperm(len(pos), device=pos.device)[:n_pos]]
    neg = neg[torch.randperm(len(neg), device=neg.device)[:n_neg]]
    return pos, neg


def match(proposals: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, fg_iou: float, bg_low: float):
    """Label each proposal with its best ground truth: class if IoU >= fg_iou, background if in
    [bg_low, fg_iou), ignored (-1) below bg_low. Returns (labels, matched gt boxes)."""
    if len(gt_boxes) == 0:
        return torch.zeros(len(proposals), dtype=torch.long, device=proposals.device), torch.zeros_like(proposals)
    iou, idx = box_iou(proposals, gt_boxes).max(1)
    labels = gt_labels[idx].clone()
    labels[iou < fg_iou] = 0
    labels[iou < bg_low] = -1
    return labels, gt_boxes[idx]
