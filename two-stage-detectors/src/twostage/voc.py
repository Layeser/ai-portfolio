"""PASCAL VOC loading: annotations, splits, resizing and flipping shared by the three detectors.

Images are resized so that the shorter side is 600 px (longer side at most 1000 px), as in the
Fast / Faster R-CNN papers. Boxes are (x1, y1, x2, y2) in pixels, 0-based.
"""

from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
VOC_ROOT = Path(os.getenv("TWOSTAGE_VOC", ROOT / "data" / "VOCdevkit"))
PROPOSALS_DIR = ROOT / "data" / "proposals"
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
HOLDOUT_SIZE = 300  # VOC07 trainval images kept out of training, used to monitor and select checkpoints


def split_ids(split: str, year: str = "2007") -> list[str]:
    """'trainval' | 'test' | 'train' (trainval minus holdout) | 'holdout'."""
    if split in ("train", "holdout"):
        ids = split_ids("trainval", year)
        held = set(random.Random(0).sample(ids, HOLDOUT_SIZE))
        return [i for i in ids if (i in held) == (split == "holdout")]
    path = VOC_ROOT / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
    return path.read_text().split()


@cache
def load_annotation(image_id: str, year: str = "2007") -> dict:
    tree = ET.parse(VOC_ROOT / f"VOC{year}" / "Annotations" / f"{image_id}.xml")
    boxes, labels, difficult = [], [], []
    for obj in tree.findall("object"):
        bb = obj.find("bndbox")
        boxes.append([float(bb.find(k).text) - 1 for k in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(CLASSES.index(obj.find("name").text.strip()) + 1)  # 0 = background
        difficult.append(int(obj.find("difficult").text or 0) == 1)
    return {
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(labels, dtype=np.int64),
        "difficult": np.array(difficult, dtype=bool),
    }


def load_proposals(image_id: str, top: int | None = None) -> np.ndarray:
    """Selective Search boxes cached by scripts/selective_search.py (original pixel coordinates)."""
    boxes = np.load(PROPOSALS_DIR / f"{image_id}.npy").astype(np.float32)
    return boxes[:top] if top else boxes


def resize_scale(width: int, height: int, short: int = 600, max_size: int = 1000) -> float:
    return min(short / min(width, height), max_size / max(width, height))


class VOCDetection(Dataset):
    """Returns (image float tensor in [0, 1], target dict). Difficult objects are dropped for training
    (standard practice) but kept in evaluation, where they count as "ignore"."""

    def __init__(self, ids: list[str], train: bool, proposals: bool = False, short: int = 600, year: str = "2007"):
        self.ids, self.train, self.proposals, self.short, self.year = ids, train, proposals, short, year

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        image_id = self.ids[i]
        image = Image.open(VOC_ROOT / f"VOC{self.year}" / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
        w, h = image.size
        scale = resize_scale(w, h, self.short)
        image = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0

        ann = load_annotation(image_id, self.year)
        keep = ~ann["difficult"] if self.train else np.ones(len(ann["labels"]), dtype=bool)
        target = {
            "boxes": torch.from_numpy(ann["boxes"][keep] * scale),
            "labels": torch.from_numpy(ann["labels"][keep]),
            "image_id": image_id,
            "scale": scale,
            "orig_size": (h, w),
        }
        if self.proposals:
            target["proposals"] = torch.from_numpy(load_proposals(image_id) * scale)
        if self.train and random.random() < 0.5:
            x = x.flip(-1)
            width = x.shape[-1]
            for key in ("boxes", "proposals"):
                if key in target and len(target[key]):
                    b = target[key]
                    target[key] = torch.stack([width - 1 - b[:, 2], b[:, 1], width - 1 - b[:, 0], b[:, 3]], 1)
        return x, target


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]
