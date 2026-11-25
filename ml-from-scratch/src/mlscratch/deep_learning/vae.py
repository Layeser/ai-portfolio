"""
Variational Autoencoder from scratch (PyTorch) on MNIST.

Encoder outputs (mu, logvar); reparameterize z = mu + eps * sigma.
Loss = reconstruction BCE + beta * KL(q(z|x) || N(0,I))   (beta = 1: the negative ELBO)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlscratch.deep_learning.autoencoder import batches, load_mnist
from mlscratch.utils.common import ensure_outputs, set_seed


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden)
        self.fc4 = nn.Linear(hidden, 28 * 28)

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

    @torch.no_grad()
    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Decode n draws from the prior N(0, I): new digits, not reconstructions."""
        device = self.fc3.weight.device
        z = torch.randn(n, self.latent_dim, generator=generator).to(device)
        return self.decode(z)


def vae_terms(recon, x, mu, logvar) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruction term and KL term, both summed over the batch."""
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    # KL(N(mu, sigma^2) || N(0, 1)) in closed form, summed over latent dimensions
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce, kl


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    """Negative ELBO for beta = 1; beta > 1 weighs the prior more (smoother latent space, blurrier digits)."""
    bce, kl = vae_terms(recon, x, mu, logvar)
    return bce + beta * kl


@torch.no_grad()
def evaluate(model: VAE, X: torch.Tensor, batch_size: int = 1000) -> dict[str, float]:
    """Per-image reconstruction and KL terms on X, and their sum (the negative ELBO, in nats)."""
    model.eval()
    rec = kl = 0.0
    for xb in X.split(batch_size):
        recon, mu, logvar = model(xb)
        b, k = vae_terms(recon, xb, mu, logvar)
        rec, kl = rec + b.item(), kl + k.item()
    n = len(X)
    return {"recon": rec / n, "kl": kl / n, "neg_elbo": (rec + kl) / n}


def train_vae(
    X_train: torch.Tensor, latent_dim: int = 16, beta: float = 1.0, epochs: int = 10, batch_size: int = 128,
    lr: float = 1e-3, seed: int = 0,
) -> tuple[VAE, list[float]]:
    set_seed(seed)
    model = VAE(latent_dim).to(X_train.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in batches(len(X_train), batch_size, g, X_train.device):
            x = X_train[idx]
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta) / len(x)
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
    model, losses = train_vae(X_tr, latent_dim=16, epochs=10)
    for ep, loss in enumerate(losses, 1):
        print(f"epoch {ep:2d}  train loss per image={loss:.1f}")
    m = evaluate(model, X_te)
    print(f"test: recon={m['recon']:.1f}  kl={m['kl']:.1f}  -ELBO={m['neg_elbo']:.1f} nats per image")

    samples = model.sample(16, torch.Generator().manual_seed(0)).cpu()
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
