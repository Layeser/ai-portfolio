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

    def annotations(self, idx: int, image_size_wh: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Ground-truth boxes (cx, cy, w, h in letterboxed pixels) and labels for one image.

        Kept separate from the YOLO grid target, which can only hold one object per cell:
        evaluation must use *all* annotated objects.
        """
        img_path = self.image_paths[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")
        if image_size_wh is None:
            with Image.open(img_path) as im:
                image_size_wh = im.size
        orig_w, orig_h = image_size_wh
        scale = min(self.image_size / orig_w, self.image_size / orig_h)
        pad_x = (self.image_size - max(1, int(round(orig_w * scale)))) // 2
        pad_y = (self.image_size - max(1, int(round(orig_h * scale)))) // 2

        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.exists():
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cx, cy, w, h = map(float, parts[1:5])
                    # Normalized (original image) -> absolute pixels in the letterboxed square
                    boxes.append([cx * orig_w * scale + pad_x, cy * orig_h * scale + pad_y,
                                  w * orig_w * scale, h * orig_h * scale])
                    labels.append(int(parts[0]))
        if not boxes:
            return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        size = image.size
        image, _, _, _ = self._letterbox(image)
        box_tensor, label_tensor = self.annotations(idx, size)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
            if torch.rand(1).item() < 0.5:  # horizontal flip: mirror the image and the box centres
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if len(box_tensor):
                    box_tensor = box_tensor.clone()
                    box_tensor[:, 0] = self.image_size - box_tensor[:, 0]
            image, box_tensor, label_tensor = random_scale_translate(image, box_tensor, label_tensor)
        image = self.to_tensor(image)

        target = encode_boxes(
            box_tensor,
            label_tensor,
            grid_size=self.grid_size,
            num_classes=self.num_classes,
            image_size=self.image_size,
        )
        return image, target


def random_scale_translate(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scale: tuple[float, float] = (0.8, 1.2),
    translate: float = 0.2,
    min_visible: float = 0.35,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """YOLO-paper style augmentation: random scaling and translation of up to 20 % of the image size.

    Boxes (cx, cy, w, h in pixels) follow the same affine map, are clipped to the image, and are
    dropped when less than `min_visible` of their area remains inside.
    """
    size = image.size[0]
    s = float(torch.empty(1).uniform_(*scale))
    tx, ty = (float(v) * translate * size for v in torch.empty(2).uniform_(-1, 1))
    c = size / 2
    # PIL's affine takes the inverse map (output pixel -> input pixel): x_in = (x_out - c - tx) / s + c
    inverse = (1 / s, 0, c - (c + tx) / s, 0, 1 / s, c - (c + ty) / s)
    out = image.transform(image.size, Image.AFFINE, inverse, resample=Image.BILINEAR, fillcolor=(114, 114, 114))
    if not len(boxes):
        return out, boxes, labels

    x1 = (boxes[:, 0] - boxes[:, 2] / 2 - c) * s + c + tx
    y1 = (boxes[:, 1] - boxes[:, 3] / 2 - c) * s + c + ty
    x2 = (boxes[:, 0] + boxes[:, 2] / 2 - c) * s + c + tx
    y2 = (boxes[:, 1] + boxes[:, 3] / 2 - c) * s + c + ty
    area = (x2 - x1) * (y2 - y1)
    x1, y1, x2, y2 = (v.clamp(0, size) for v in (x1, y1, x2, y2))
    keep = ((x2 - x1) * (y2 - y1) >= min_visible * area) & (x2 - x1 > 2) & (y2 - y1 > 2)
    new_boxes = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)[keep]
    return out, new_boxes, labels[keep]
