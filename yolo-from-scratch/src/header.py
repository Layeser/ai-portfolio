"""YOLOv1 detection head: predicts bounding boxes and class probabilities per grid cell."""

import torch
import torch.nn as nn

from .blocks import ConvBlock


class YOLOv1Header(nn.Module):
    """Detection head producing S×S×(B·5 + C) predictions.

    Each grid cell predicts:
      - B bounding boxes: (x, y, w, h, confidence)  — 5 values each
      - C class probabilities (shared across boxes in the cell)

    Output tensor layout (per cell, flattened):
      [box1_x, box1_y, box1_w, box1_h, box1_conf,
       box2_x, box2_y, box2_w, box2_h, box2_conf,
       class_1, ..., class_C]
    """

    def __init__(
        self,
        in_channels: int = 1024,
        num_boxes: int = 2,
        num_classes: int = 20,
        grid_size: int = 7,
    ) -> None:
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_outputs = num_boxes * 5 + num_classes

        # Original YOLOv1 used two FC layers; we use 1×1 conv for efficiency
        self.head = nn.Sequential(
            ConvBlock(in_channels, 1024, kernel_size=3),
            nn.Conv2d(1024, self.num_outputs, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: neck features (B, 1024, S, S)

        Returns:
            predictions (B, S, S, num_outputs)
        """
        x = self.head(x)
        # (B, num_outputs, S, S) -> (B, S, S, num_outputs)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Keep outputs in a bounded range for stable training. The target tensor
        # uses normalized box coordinates and class/object confidences, so sigmoid
        # is a natural activation for the head outputs. Build the output
        # functionally (no in-place writes on views) to keep autograd correct.
        box_raw = x[..., : self.num_boxes * 5].reshape(*x.shape[:-1], self.num_boxes, 5)
        box_coords = torch.sigmoid(box_raw[..., :4])       # x, y, w, h in [0, 1]
        box_conf = torch.sigmoid(box_raw[..., 4:5])         # objectness in [0, 1]
        box_predictions = torch.cat([box_coords, box_conf], dim=-1).reshape(
            *x.shape[:-1], self.num_boxes * 5
        )

        class_predictions = torch.sigmoid(x[..., self.num_boxes * 5 :])
        return torch.cat([box_predictions, class_predictions], dim=-1)
