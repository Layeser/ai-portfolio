"""Shape and learning-signal tests for the PyTorch models (CPU, a few seconds)."""


import pytest

torch = pytest.importorskip("torch")

from mlscratch.deep_learning.autoencoder import Autoencoder  # noqa: E402
from mlscratch.deep_learning.lenet import LeNet  # noqa: E402
from mlscratch.deep_learning.resnet import ResNet18  # noqa: E402
from mlscratch.deep_learning.vae import VAE, vae_loss  # noqa: E402


def test_output_shapes():
    torch.manual_seed(0)
    assert LeNet()(torch.randn(4, 1, 28, 28)).shape == (4, 10)
    assert ResNet18()(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    recon, mu, logvar = VAE(latent_dim=8)(torch.rand(3, 1, 28, 28))
    assert recon.shape == (3, 1, 28, 28) and mu.shape == logvar.shape == (3, 8)


def test_vae_kl_term_is_zero_for_standard_normal_posterior():
    x = torch.full((1, 1, 28, 28), 0.5)
    mu, logvar = torch.zeros(1, 4), torch.zeros(1, 4)
    bce_only = torch.nn.functional.binary_cross_entropy(x, x, reduction="sum")
    assert vae_loss(x, x, mu, logvar).item() == pytest.approx(bce_only.item(), rel=1e-6)


@pytest.mark.parametrize("make_model", [lambda: Autoencoder(latent_dim=16), lambda: VAE(latent_dim=8)])
def test_models_can_overfit_a_tiny_batch(make_model):
    """A model that cannot overfit 16 images has a bug (classic sanity check)."""
    torch.manual_seed(0)
    model = make_model()
    # Binary pixels: with continuous random targets the BCE has an irreducible floor (their entropy).
    x = torch.bernoulli(torch.full((16, 1, 28, 28), 0.3))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn():
        out = model(x)
        if isinstance(out, tuple):
            return vae_loss(out[0], x, out[1], out[2]) / len(x)
        return torch.nn.functional.mse_loss(out.view_as(x), x)

    first = loss_fn().item()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    assert loss.item() < 0.7 * first


def test_lenet_overfits_a_tiny_labelled_batch():
    torch.manual_seed(0)
    model, x, y = LeNet(), torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert (model(x).argmax(1) == y).float().mean() > 0.9
