"""
Variational Autoencoder from scratch (PyTorch) on MNIST.

Encoder outputs (mu, logvar); reparameterize z = mu + eps * sigma.
Loss = reconstruction BCE + KL(q(z|x) || N(0,I))
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import ensure_outputs, set_seed


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 28 * 28)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.fc1(x.view(-1, 28 * 28)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h)).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    # KL divergence closed form
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl


def main() -> None:
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(train, batch_size=128, shuffle=True)

    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(3):
        total = 0.0
        for x, _ in tqdm(loader, desc=f"epoch {epoch+1}", leave=False):
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch+1}  elbo_like={total / len(train):.1f}")

    # sample from prior
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, 16, device=device)
        samples = model.decode(z).cpu()
    out = ensure_outputs("vae")
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap="gray")
        ax.axis("off")
    fig.suptitle("VAE samples from N(0,I)")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=140)
    print(f"saved {out / 'samples.png'}")


if __name__ == "__main__":
    main()
