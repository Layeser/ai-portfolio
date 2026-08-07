"""YOLOv1 backbone: either the original custom CNN or a pretrained ResNet18."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as torchvision_models

from .blocks import ConvBlock, MaxPoolBlock


class YOLOv1Backbone(nn.Module):
    """Feature extractor for YOLOv1.

    Supports:
      - custom: the original lightweight CNN used by this project
      - pretrained_resnet18: a more stable pretrained ResNet18 backbone
      - pretrained_resnet50: a stronger pretrained ResNet50 backbone
    """

    def __init__(self, backbone_type: str = "custom", pretrained: bool = False) -> None:
        super().__init__()
        self.backbone_type = backbone_type
        self.pretrained = pretrained

        if backbone_type in {"pretrained_resnet18", "pretrained_resnet50"}:
            model_name = "resnet18" if backbone_type == "pretrained_resnet18" else "resnet50"
            weights_cls = getattr(torchvision_models, f"{model_name.title()}_Weights", None)
            weights = weights_cls.DEFAULT if pretrained and weights_cls is not None else None
            try:
                resnet = getattr(torchvision_models, model_name)(weights=weights)
            except Exception:
                resnet = getattr(torchvision_models, model_name)(weights=None)
                if pretrained:
                    print(f"Warning: could not load pretrained {model_name} weights; falling back to random init.")

            self.features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
            feature_channels = 512 if backbone_type == "pretrained_resnet18" else 2048
            self.projection = nn.Conv2d(feature_channels, 1024, kernel_size=1, bias=False)
            return

        # --- Custom YOLOv1-inspired backbone ---
        self.stage1 = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            MaxPoolBlock(2, 2),
            ConvBlock(64, 192, kernel_size=3),
            MaxPoolBlock(2, 2),
        )

        self.stage2 = nn.Sequential(
            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            MaxPoolBlock(2, 2),
        )

        self.stage3 = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 512, kernel_size=3),
            ConvBlock(512, 1024, kernel_size=3),
            MaxPoolBlock(2, 2),
        )

        self.stage4 = nn.Sequential(
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_type in {"pretrained_resnet18", "pretrained_resnet50"}:
            x = self.features(x)
            x = self.projection(x)
            return x

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
