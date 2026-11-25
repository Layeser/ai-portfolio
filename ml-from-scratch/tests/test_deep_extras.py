"""Tests for the deep-learning additions: augmentation, plain networks, beta-VAE, attention maps, data splits."""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from mlscratch.deep_learning.autoencoder import Autoencoder, reconstruction_mse  # noqa: E402
from mlscratch.deep_learning.lenet import LeNet  # noqa: E402
from mlscratch.deep_learning.resnet import ResNet18, ResNet34, augment  # noqa: E402
from mlscratch.deep_learning.vae import VAE, vae_loss, vae_terms  # noqa: E402
from mlscratch.nlp.en_fr_translation.data import (  # noqa: E402
    PAD,
    build_vocab,
    encode,
    template_pairs,
    train_test_split,
)
from mlscratch.nlp.en_fr_translation.model import Seq2SeqTransformer  # noqa: E402
from mlscratch.nlp.en_fr_translation.train import exact_match, train  # noqa: E402


# ------------------------------------------------------------------------------------------- vision
def test_augment_returns_a_shifted_and_possibly_flipped_crop():
    torch.manual_seed(0)
    x = torch.randn(16, 3, 8, 8)
    out = augment(x, pad=2, generator=torch.Generator().manual_seed(1))
    assert out.shape == x.shape
    xp = F.pad(x, (2, 2, 2, 2))
    for b in range(len(x)):
        candidates = [xp[b, :, i:i + 8, j:j + 8] for i in range(5) for j in range(5)]
        candidates += [c.flip(2) for c in candidates]
        assert any(torch.equal(out[b], c) for c in candidates)


def test_augment_flips_about_half_the_images():
    x = torch.arange(64.0).view(1, 1, 8, 8).repeat(2000, 1, 1, 1)
    out = augment(x, pad=0, generator=torch.Generator().manual_seed(0))  # pad 0: only the flip remains
    flipped = (out == x.flip(3)).flatten(1).all(1).float().mean().item()
    assert 0.45 < flipped < 0.55


def test_plain_network_has_no_shortcut_and_runs():
    torch.manual_seed(0)
    plain, res = ResNet18(residual=False), ResNet18()
    n_plain = sum(p.numel() for p in plain.parameters())
    n_res = sum(p.numel() for p in res.parameters())
    assert n_plain < n_res  # only the 1x1 projection shortcuts differ
    assert plain(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert ResNet34()(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_lenet_feature_maps_have_the_textbook_shapes():
    a1, a2 = LeNet().features(torch.randn(2, 1, 28, 28))
    assert a1.shape == (2, 6, 14, 14) and a2.shape == (2, 16, 5, 5)


def test_reconstruction_mse_is_the_mean_squared_error_per_pixel():
    torch.manual_seed(0)
    model, x = Autoencoder(latent_dim=4), torch.rand(50, 1, 28, 28)
    with torch.no_grad():
        expected = F.mse_loss(model(x), x).item()
    assert reconstruction_mse(model, x, batch_size=7) == pytest.approx(expected, rel=1e-5)


def test_beta_weighs_only_the_kl_term():
    torch.manual_seed(0)
    x = torch.rand(4, 1, 28, 28)
    recon, mu, logvar = VAE(latent_dim=3)(x)
    bce, kl = vae_terms(recon, x, mu, logvar)
    assert vae_loss(recon, x, mu, logvar, beta=0).item() == pytest.approx(bce.item(), rel=1e-6)
    assert vae_loss(recon, x, mu, logvar, beta=4).item() == pytest.approx((bce + 4 * kl).item(), rel=1e-6)
    assert kl.item() >= 0


def test_vae_samples_are_images_in_unit_range():
    s = VAE(latent_dim=5).sample(6, torch.Generator().manual_seed(0))
    assert s.shape == (6, 1, 28, 28) and s.min() >= 0 and s.max() <= 1


# -------------------------------------------------------------------------------------- translation
def test_template_grid_and_split_are_consistent():
    pairs = template_pairs()
    assert len(pairs) == len({e for e, _ in pairs}) == 205
    assert ("the car is white", "la voiture est blanche") in pairs
    assert ("we like the dog", "nous aimons le chien") in pairs
    tr, te = train_test_split(test_fraction=0.5, seed=3)
    assert not {e for e, _ in tr} & {e for e, _ in te}
    seen = {w for e, f in tr for w in (e + " " + f).split()}
    assert all(set((e + " " + f).split()) <= seen for e, f in te)  # held-out words were all seen in training


def test_systematic_split_holds_out_exactly_the_predicate():
    tr, te = train_test_split(hold_out=lambda p: p[0].startswith("the window is"))
    assert te and all(e.startswith("the window is") for e, _ in te)
    assert not any(e.startswith("the window is") for e, _ in tr)


def test_cross_attention_rows_sum_to_one_and_ignore_padding():
    torch.manual_seed(0)
    model = Seq2SeqTransformer(src_vocab=12, tgt_vocab=10, d_model=32, nhead=4, dim_ff=64, pad_id=0)
    src = torch.tensor([[4, 5, 6, 0, 0], [7, 8, 9, 10, 11]])  # first sentence padded
    tgt = torch.tensor([[1, 4, 5], [1, 6, 7]])
    att = model.cross_attention(src, tgt)
    assert att.shape == (2, 3, 5)
    torch.testing.assert_close(att.sum(-1), torch.ones(2, 3))
    assert att[0, :, 3:].abs().max() == 0  # padded source positions receive no attention


def test_transformer_learns_a_small_corpus():
    """Classic bug detector: a few short pairs must be learned perfectly."""
    pairs = [("the cat is black", "le chat est noir"), ("the car is red", "la voiture est rouge"),
             ("i see the dog", "je vois le chien"), ("we like the house", "nous aimons la maison")]
    t = train(pairs, epochs=60, eval_every=60, device="cpu", verbose=False)
    assert exact_match(t, pairs) == 1.0


def test_vocabulary_reserves_the_special_tokens():
    stoi, itos = build_vocab(["a b"])
    assert stoi[PAD] == 0 and itos[0] == PAD
    assert encode("a zzz", stoi)[2] == stoi["<unk>"]
