"""YOLOv1 neck: high-level feature refinement before detection.

In the original paper the last conv blocks (before the FC/conv head) act as
a feature refinement stage. We group them here as the "neck" to match modern
detector layouts (backbone -> neck -> head).
"""

import torch
import torch.nn as nn

from .blocks import ConvBlock


class YOLOv1Neck(nn.Module):
    """Refines backbone features with additional 1024-channel convolutions.

    Input : (B, 1024, H/32, W/32)
    Output: (B, 1024, S, S)  where S = grid_size (7 for PASCAL VOC setup).
    """

    def __init__(self, grid_size: int = 7) -> None:
        super().__init__()
        self.grid_size = grid_size

        # High-capacity conv stack that reduces spatial dims to the detection grid
        self.neck = nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=3),
            ConvBlock(1024, 1024, kernel_size=3, stride=2),  # downsample toward S×S
            ConvBlock(1024, 1024, kernel_size=3),
            ConvBlock(1024, 1024, kernel_size=3),  # stride 1: the paper downsamples only once (14 -> 7)
        )
        # Bug fixed in 2026: a second stride-2 conv here produced a 4x4 map that adaptive pooling
        # stretched to 7x7, so the "7x7" grid had 4x4 real positions and the detector could not localise.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.neck(x)

        # Ensure exact grid size via adaptive pooling if spatial dims differ
        if x.shape[-1] != self.grid_size or x.shape[-2] != self.grid_size:
            x = nn.functional.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))

        return x
