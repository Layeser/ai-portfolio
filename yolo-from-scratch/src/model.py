"""Full YOLOv1 model assembling backbone, neck, and detection header."""

import torch
import torch.nn as nn

from .backbone import YOLOv1Backbone
from .header import YOLOv1Header
from .neck import YOLOv1Neck


class YOLOv1(nn.Module):
    """YOLOv1 unified real-time object detector.

    Architecture:
        Input (B, 3, 448, 448)
          -> Backbone  : GoogLeNet-style feature extraction
          -> Neck      : 1024-dim feature refinement to S×S grid
          -> Header    : per-cell box + class predictions

    Output: (B, S, S, B*5 + C)  default S=7, B=2, C=20 (PASCAL VOC).
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_boxes: int = 2,
        grid_size: int = 7,
        backbone_type: str = "custom",
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.grid_size = grid_size
        self.backbone_type = backbone_type
        self.pretrained_backbone = pretrained_backbone

        self.backbone = YOLOv1Backbone(
            backbone_type=backbone_type,
            pretrained=pretrained_backbone,
        )
        self.neck = YOLOv1Neck(grid_size=grid_size)
        self.header = YOLOv1Header(
            in_channels=1024,
            num_boxes=num_boxes,
            num_classes=num_classes,
            grid_size=grid_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.neck(features)
        predictions = self.header(features)
        return predictions

    @property
    def num_outputs(self) -> int:
        return self.num_boxes * 5 + self.num_classes
