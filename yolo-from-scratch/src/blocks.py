"""Reusable convolutional building blocks for YOLOv1."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Standard conv block: Conv2d -> BatchNorm -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaxPoolBlock(nn.Module):
    """Max pooling with optional stride override (YOLOv1 uses stride=1 on some pools)."""

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
