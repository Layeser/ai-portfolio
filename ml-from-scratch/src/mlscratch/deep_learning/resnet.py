"""
ResNet-18 (CIFAR variant) implemented from scratch — no torchvision.models.

BasicBlock with skip connections; trained on CIFAR-10 with random-crop and flip augmentation, a one-cycle learning-rate
schedule and mixed precision. `residual=False` removes the skip connections (the "plain" network of the paper).
Reference: He et al., Deep Residual Learning for Image Recognition (2015).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlscratch.deep_learning.autoencoder import batches
from mlscratch.utils.common import set_seed

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if residual and (stride != 1 or in_planes != planes):
            # 1x1 projection when the shape changes, so the addition is defined
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_blocks: list[int], num_classes: int = 10, residual: bool = True):
        super().__init__()
        self.in_planes = 64
        self.residual = residual
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, residual=self.residual))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def ResNet18(num_classes: int = 10, residual: bool = True) -> ResNet:
    return ResNet([2, 2, 2, 2], num_classes=num_classes, residual=residual)


def ResNet34(num_classes: int = 10, residual: bool = True) -> ResNet:
    return ResNet([3, 4, 6, 3], num_classes=num_classes, residual=residual)


def load_cifar10(root: str = "./data", device: str | torch.device = "cpu") -> tuple[torch.Tensor, ...]:
    """CIFAR-10 as normalised float tensors (N, 3, 32, 32) on `device`, plus labels: 50 000 train, 10 000 test."""
    from torchvision import datasets

    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    out = []
    for train in (True, False):
        ds = datasets.CIFAR10(root, train=train, download=True)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255
        out += [((x - mean) / std).to(device), torch.tensor(ds.targets).to(device)]
    return tuple(out)


def augment(x: torch.Tensor, pad: int = 4, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random crop after zero padding, and random horizontal flip, for a whole batch at once on its device.

    The standard CIFAR augmentation (He et al., 2015). Each image gets its own offset: the crop is gathered with
    advanced indexing instead of a Python loop over images.
    """
    B, _, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (B,), generator=generator).to(x.device)
    dx = torch.randint(0, 2 * pad + 1, (B,), generator=generator).to(x.device)
    rows = (torch.arange(H, device=x.device)[None] + dy[:, None])[:, :, None]  # (B, H, 1)
    cols = (torch.arange(W, device=x.device)[None] + dx[:, None])[:, None, :]  # (B, 1, W)
    out = xp[torch.arange(B, device=x.device)[:, None, None], :, rows, cols]  # (B, H, W, C)
    out = out.permute(0, 3, 1, 2)
    flip = (torch.rand(B, generator=generator) < 0.5).to(x.device)
    return torch.where(flip[:, None, None, None], out.flip(3), out)


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 1000) -> float:
    model.eval()
    correct = 0
    for xb, yb in zip(X.split(batch_size), y.split(batch_size), strict=True):
        with torch.autocast(xb.device.type, enabled=xb.device.type == "cuda"):
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Accuracy over a DataLoader (kept for callers that use one)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def train_resnet(
    model: nn.Module, X_tr, y_tr, X_te, y_te, epochs: int = 30, batch_size: int = 128, max_lr: float = 0.1,
    weight_decay: float = 5e-4, use_augment: bool = True, seed: int = 0, verbose: bool = True,
) -> dict[str, list[float]]:
    """SGD with momentum and a one-cycle schedule (warm-up then cosine decay); mixed precision on GPU."""
    set_seed(seed)
    device = X_tr.device
    opt = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    steps = epochs * ((len(X_tr) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    g = torch.Generator().manual_seed(seed)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        total = correct = 0.0
        for idx in batches(len(X_tr), batch_size, g, device):
            x, y = X_tr[idx], y_tr[idx]
            if use_augment:
                x = augment(x, generator=g)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        history["loss"].append(total / len(X_tr))
        history["train_acc"].append(correct / len(X_tr))
        history["test_acc"].append(accuracy(model, X_te, y_te))
        if verbose:
            print(f"epoch {ep:2d}  loss={history['loss'][-1]:.3f}  train_acc={history['train_acc'][-1]:.3f}"
                  f"  test_acc={history['test_acc'][-1]:.3f}", flush=True)
    return history


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr, y_tr, X_te, y_te = load_cifar10(device=device)
    if device.type == "cpu":  # laptop-friendly demo without a GPU
        X_tr, y_tr, X_te, y_te = X_tr[:8000], y_tr[:8000], X_te[:2000], y_te[:2000]
    model = ResNet18().to(device).to(memory_format=torch.channels_last)
    train_resnet(model, X_tr, y_tr, X_te, y_te, epochs=30 if device.type == "cuda" else 5)


if __name__ == "__main__":
    main()
