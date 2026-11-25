"""
Autoencoder from scratch (PyTorch) on MNIST.

Encoder: 784 → 256 → latent_dim
Decoder: latent_dim → 256 → 784
Loss: MSE reconstruction, reported on the test set.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlscratch.utils.common import ensure_outputs, set_seed


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 28 * 28),
            nn.Sigmoid(),  # pixels are in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def load_mnist(root: str = "./data", device: str | torch.device = "cpu") -> tuple[torch.Tensor, ...]:
    """MNIST as four tensors on `device`: images in [0, 1], shape (N, 1, 28, 28), and labels.

    60 000 images take 188 MB as float32: small enough to keep on the GPU and skip the DataLoader.
    """
    from torchvision import datasets

    out = []
    for train in (True, False):
        ds = datasets.MNIST(root, train=train, download=True)
        out += [(ds.data.float() / 255).unsqueeze(1).to(device), ds.targets.to(device)]
    return tuple(out)


def batches(n: int, batch_size: int, generator: torch.Generator | None = None, device="cpu"):
    """Index batches over a random permutation (`generator` fixes the order)."""
    perm = torch.randperm(n, generator=generator).to(device)
    return perm.split(batch_size)


@torch.no_grad()
def reconstruction_mse(model: nn.Module, X: torch.Tensor, batch_size: int = 1000) -> float:
    """Mean squared error per pixel, the metric PCA minimises too: the two are directly comparable."""
    model.eval()
    total = sum(F.mse_loss(model(xb), xb, reduction="sum").item() for xb in X.split(batch_size))
    return total / X.numel()


def train_autoencoder(
    X_train: torch.Tensor, latent_dim: int = 64, epochs: int = 10, batch_size: int = 128, lr: float = 1e-3, seed: int = 0
) -> tuple[Autoencoder, list[float]]:
    set_seed(seed)
    model = Autoencoder(latent_dim).to(X_train.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in batches(len(X_train), batch_size, g, X_train.device):
            x = X_train[idx]
            loss = F.mse_loss(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        losses.append(total / len(X_train))
    return model, losses


def main() -> None:
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr, _, X_te, _ = load_mnist(device=device)
    model, losses = train_autoencoder(X_tr, latent_dim=64, epochs=10)
    for ep, loss in enumerate(losses, 1):
        print(f"epoch {ep:2d}  train mse={loss:.4f}")
    print(f"test mse={reconstruction_mse(model, X_te):.4f}")

    model.eval()
    x = X_te[:8]
    with torch.no_grad():
        recon = model(x).cpu()
    out = ensure_outputs("autoencoder")
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for i in range(8):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray")
        axes[1, i].imshow(recon[i, 0], cmap="gray")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Autoencoder reconstructions (test set): original above, reconstruction below")
    fig.tight_layout()
    fig.savefig(out / "recon.png", dpi=140)
    print(f"saved {out / 'recon.png'}")


if __name__ == "__main__":
    main()
