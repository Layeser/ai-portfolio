"""Dataset utilities for YOLOv1 training (PASCAL VOC / custom YOLO-format)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.boxes import encode_boxes


class YOLODataset(Dataset):
    """Generic detection dataset with YOLO-format labels.

    Expected layout:
        root/
          images/   *.jpg
          labels/   *.txt   (class cx cy w h — normalized 0-1)

    Each label file contains one object per line.
    """

    # ImageNet statistics — required so pretrained backbones see the input
    # distribution they were trained on.
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str | Path,
        image_size: int = 448,
        grid_size: int = 7,
        num_classes: int = 20,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_classes = num_classes

        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        self.image_paths = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))

        # Color-only augmentation keeps geometry (and therefore box coords) intact.
        self.color_jitter = (
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            if augment
            else None
        )
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _letterbox(self, image: Image.Image) -> tuple[Image.Image, float, int, int]:
        """Resize preserving aspect ratio and pad to a square (matches detect.py)."""
        orig_w, orig_h = image.size
        scale = min(self.image_size / orig_w, self.image_size / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = image.resize((new_w, new_h), Image.BILINEAR)

        padded = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        pad_x = (self.image_size - new_w) // 2
        pad_y = (self.image_size - new_h) // 2
        padded.paste(resized, (pad_x, pad_y))
        return padded, scale, pad_x, pad_y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        image, scale, pad_x, pad_y = self._letterbox(image)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        image = self.to_tensor(image)

        boxes: list[list[float]] = []
        labels: list[int] = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    # Normalized (original image) -> absolute pixels in the letterboxed square
                    boxes.append(
                        [
                            cx * orig_w * scale + pad_x,
                            cy * orig_h * scale + pad_y,
                            w * orig_w * scale,
                            h * orig_h * scale,
                        ]
                    )
                    labels.append(cls_id)

        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            box_tensor = torch.zeros(0, 4)
            label_tensor = torch.zeros(0, dtype=torch.long)

        target = encode_boxes(
            box_tensor,
            label_tensor,
            grid_size=self.grid_size,
            num_classes=self.num_classes,
            image_size=self.image_size,
        )
        return image, target
