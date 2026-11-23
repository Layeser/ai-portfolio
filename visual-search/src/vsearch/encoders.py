"""Four ways to turn an image into a search vector, from no learning to transfer learning.

| encoder      | learns from                         | labels used |
|--------------|-------------------------------------|-------------|
| pixels_pca   | CIFAR-10 train pixels (PCA)         | none        |
| autoencoder  | CIFAR-10 train, reconstruction      | none        |
| simclr       | CIFAR-10 train, contrastive views   | none        |
| imagenet     | ImageNet-1k classification          | ImageNet's, never CIFAR's |
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 (N, 32, 32, 3) -> float (N, 3, 32, 32), CIFAR-normalised."""
    x = torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def batched(fn, images: np.ndarray, batch: int = 1024) -> np.ndarray:
    return np.concatenate([fn(images[i : i + batch]) for i in range(0, len(images), batch)])


# ---- 1. pixels + PCA -------------------------------------------------------------------------


class PixelsPCA:
    name = "pixels_pca"

    def __init__(self, dim: int = 128):
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=dim, random_state=0)

    def fit(self, images: np.ndarray) -> PixelsPCA:
        self.pca.fit(to_tensor(images).flatten(1).numpy())
        return self

    def embed(self, images: np.ndarray) -> np.ndarray:
        return self.pca.transform(to_tensor(images).flatten(1).numpy()).astype(np.float32)


# ---- 2. convolutional autoencoder -----------------------------------------------------------


class ConvAutoencoder(nn.Module):
    name = "autoencoder"

    def __init__(self, dim: int = 256):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(
            block(3, 64), block(64, 128), block(128, 256), nn.Flatten(), nn.Linear(256 * 4 * 4, dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            up(256, 128),
            up(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def fit(self, images: np.ndarray, epochs: int = 30, batch: int = 256) -> ConvAutoencoder:
        dev = device()
        self.to(dev).train()
        data = to_tensor(images)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(epochs):
            perm, total = torch.randperm(len(data)), 0.0
            for i in range(0, len(data), batch):
                x = data[perm[i : i + batch]].to(dev)
                loss = F.mse_loss(self(x), x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(x)
            print(f"[autoencoder] epoch {epoch + 1}/{epochs} mse {total / len(data):.4f}", flush=True)
        return self

    @torch.no_grad()
    def embed(self, images: np.ndarray) -> np.ndarray:
        dev = device()
        self.to(dev).eval()
        return batched(lambda b: self.encoder(to_tensor(b).to(dev)).cpu().numpy(), images)


# ---- 3. SimCLR ------------------------------------------------------------------------------


def cifar_resnet18() -> nn.Module:
    """ResNet-18 adapted to 32 px images: 3x3 stem, no max-pool (standard for CIFAR SSL)."""
    net = torchvision.models.resnet18(num_classes=10)
    net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net


def nt_xent(z: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """SimCLR loss: each view must find its twin among the 2N-1 other views of the batch."""
    z = F.normalize(z, dim=1)
    n = len(z) // 2
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def simclr_views(x: torch.Tensor) -> torch.Tensor:
    """Random view of every image in a uint8 batch (B, 3, 32, 32), with **independent** parameters per image.

    torchvision's batch transforms draw one set of random parameters for the whole batch, which would
    give every image the same crop and colours and starve the contrastive task. Everything here is
    vectorised on the GPU: random resized crop (roi_align), flip, colour jitter (p=0.8), grayscale (p=0.2).
    """
    from torchvision.ops import roi_align

    b, dev = len(x), x.device
    x = x.float() / 255.0
    scale = torch.empty(b, device=dev).uniform_(0.2, 1.0)
    ratio = torch.exp(torch.empty(b, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3)))
    w = (torch.sqrt(scale * ratio) * 32).clamp(4, 32)
    h = (torch.sqrt(scale / ratio) * 32).clamp(4, 32)
    x0 = torch.rand(b, device=dev) * (32 - w)
    y0 = torch.rand(b, device=dev) * (32 - h)
    boxes = torch.stack([torch.arange(b, device=dev, dtype=torch.float), x0, y0, x0 + w, y0 + h], dim=1)
    x = roi_align(x, boxes, output_size=32, spatial_scale=1.0, sampling_ratio=2, aligned=True)

    flip = torch.rand(b, 1, 1, 1, device=dev) < 0.5
    x = torch.where(flip, x.flip(-1), x)

    def factor(low, high):
        return torch.empty(b, 1, 1, 1, device=dev).uniform_(low, high)

    jitter = torch.rand(b, 1, 1, 1, device=dev) < 0.8
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    jittered = x * factor(0.6, 1.4)  # brightness
    jittered = (jittered - gray.mean((2, 3), keepdim=True)) * factor(0.6, 1.4) + gray.mean((2, 3), keepdim=True)
    gray_j = 0.299 * jittered[:, 0:1] + 0.587 * jittered[:, 1:2] + 0.114 * jittered[:, 2:3]
    jittered = gray_j + (jittered - gray_j) * factor(0.6, 1.4)  # saturation
    x = torch.where(jitter, jittered.clamp(0, 1), x)
    to_gray = torch.rand(b, 1, 1, 1, device=dev) < 0.2
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return torch.where(to_gray, gray.expand_as(x), x)


class SimCLR(nn.Module):
    name = "simclr"

    def __init__(self, proj_dim: int = 128):
        super().__init__()
        self.backbone = cifar_resnet18()
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def fit(self, images: np.ndarray, epochs: int = 60, batch: int = 512) -> SimCLR:
        """Views are generated on the GPU, batch-wise, so data loading never bottlenecks."""
        dev = device()
        self.to(dev).train()
        raw = torch.from_numpy(images).permute(0, 3, 1, 2)  # uint8
        opt = torch.optim.SGD(self.parameters(), lr=0.06 * batch / 256, momentum=0.9, weight_decay=5e-4)
        steps = epochs * (len(raw) // batch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
        mean, std = MEAN.to(dev), STD.to(dev)
        for epoch in range(epochs):
            perm, total, start = torch.randperm(len(raw)), 0.0, time.time()
            for i in range(0, len(raw) - batch + 1, batch):
                x = raw[perm[i : i + batch]].to(dev)
                views = torch.cat([simclr_views(x), simclr_views(x)])
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    loss = nt_xent(self.head(self.backbone((views - mean) / std)).float())
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item()
            print(
                f"[simclr] epoch {epoch + 1}/{epochs} loss {total / (len(raw) // batch):.3f} "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )
        return self

    @torch.no_grad()
    def embed(self, images: np.ndarray) -> np.ndarray:
        """Backbone features (before the projection head), as recommended in the SimCLR paper."""
        dev = device()
        self.to(dev).eval()
        return batched(lambda b: self.backbone(to_tensor(b).to(dev)).float().cpu().numpy(), images)


# ---- 4. ImageNet transfer -------------------------------------------------------------------


class ImageNetResNet(nn.Module):
    name = "imagenet"

    def __init__(self):
        super().__init__()
        self.net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Identity()

    def fit(self, images: np.ndarray) -> ImageNetResNet:
        return self  # no training on CIFAR: pure transfer

    @torch.no_grad()
    def embed(self, images: np.ndarray) -> np.ndarray:
        dev = device()
        self.to(dev).eval()
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

        def run(b):
            x = torch.from_numpy(b).permute(0, 3, 1, 2).float().to(dev) / 255.0
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            return self.net((x - mean) / std).cpu().numpy()

        return batched(run, images, batch=256)


ENCODERS = {"pixels_pca": PixelsPCA, "autoencoder": ConvAutoencoder, "simclr": SimCLR, "imagenet": ImageNetResNet}
