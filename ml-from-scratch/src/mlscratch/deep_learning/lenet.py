"""
LeNet-5 style CNN from scratch (PyTorch) on MNIST.

Classic architecture (modernized activations):
  Conv(1→6,5) → ReLU → MaxPool
  Conv(6→16,5) → ReLU → MaxPool
  FC 400→120→84→10
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlscratch.deep_learning.autoencoder import batches, load_mnist
from mlscratch.utils.common import set_seed

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081  # training-set statistics


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Activation maps after each convolution block: (B, 6, 14, 14) and (B, 16, 5, 5)."""
        a1 = F.max_pool2d(F.relu(self.conv1(x)), 2)
        a2 = F.max_pool2d(F.relu(self.conv2(a1)), 2)
        return a1, a2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)[1].flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Centre and scale pixels with the training-set statistics: inputs around 0 make the first layer easier to fit."""
    return (x - MNIST_MEAN) / MNIST_STD


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    model.eval()
    return torch.cat([model(xb).argmax(1) for xb in X.split(batch_size)])


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


def train_lenet(
    X_tr, y_tr, X_te, y_te, epochs: int = 10, batch_size: int = 128, lr: float = 1e-3, seed: int = 0,
    cosine: bool = True,
) -> tuple[LeNet, dict[str, list[float]]]:
    """Trains on normalised tensors; records train loss and test accuracy after every epoch.

    The learning rate follows a cosine from `lr` to 0: large steps early, small ones at the end, so the last epoch is
    not a random draw among nearby solutions.
    """
    set_seed(seed)
    model = LeNet().to(X_tr.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    g = torch.Generator().manual_seed(seed)
    history = {"loss": [], "test_acc": []}
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in batches(len(X_tr), batch_size, g, X_tr.device):
            loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if sched is not None:
            sched.step()
        history["loss"].append(total / len(X_tr))
        history["test_acc"].append((predict(model, X_te) == y_te).float().mean().item())
    return model, history


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr, y_tr, X_te, y_te = load_mnist(device=device)
    _, history = train_lenet(normalize(X_tr), y_tr, normalize(X_te), y_te)
    for ep, (loss, acc) in enumerate(zip(history["loss"], history["test_acc"], strict=True), 1):
        print(f"epoch {ep:2d}  loss={loss:.4f}  test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
