"""
Autoencoder from scratch (PyTorch) on MNIST.

Encoder: 784 → 256 → 64
Decoder: 64 → 256 → 784
Loss: MSE reconstruction
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import ensure_outputs, set_seed


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z).view(-1, 1, 28, 28)


def main() -> None:
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    loader = DataLoader(train, batch_size=128, shuffle=True)

    model = Autoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    model.train()
    for epoch in range(3):
        total = 0.0
        for x, _ in tqdm(loader, desc=f"epoch {epoch+1}", leave=False):
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = crit(recon, x)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        print(f"epoch {epoch+1}  mse={total / len(train):.4f}")

    # visualize reconstructions
    model.eval()
    x, _ = next(iter(DataLoader(train, batch_size=8)))
    with torch.no_grad():
        recon = model(x.to(device)).cpu()
    out = ensure_outputs("autoencoder")
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for i in range(8):
        axes[0, i].imshow(x[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("orig")
    axes[1, 0].set_ylabel("recon")
    fig.suptitle("Autoencoder reconstructions")
    fig.tight_layout()
    fig.savefig(out / "recon.png", dpi=140)
    print(f"saved {out / 'recon.png'}")


if __name__ == "__main__":
    main()
