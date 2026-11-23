"""Fast tests on synthetic images (no CIFAR download, CPU only)."""

import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from vsearch.encoders import ConvAutoencoder, SimCLR, nt_xent, simclr_views
from vsearch.evaluate import build_index, normalise, retrieval_metrics


def coloured_images(n_per_class=20, seed=0):
    """Class c = mostly colour c: an easy, controlled retrieval problem."""
    rng = np.random.default_rng(seed)
    colours = np.array([[230, 30, 30], [30, 230, 30], [30, 30, 230]])
    images, labels = [], []
    for c, colour in enumerate(colours):
        for _ in range(n_per_class):
            img = np.clip(colour + rng.normal(0, 25, (32, 32, 3)), 0, 255).astype(np.uint8)
            images.append(img)
            labels.append(c)
    return np.stack(images), np.array(labels)


def test_retrieval_metrics_on_known_neighbours():
    gallery_labels = np.array([0, 0, 1, 1])
    neighbours = np.array([[0, 1, 2, 3] * 5, [2, 0, 3, 1] * 5])  # (2 queries, 20 neighbours)
    m = retrieval_metrics(neighbours, gallery_labels, np.array([0, 1]), k=2)
    assert m["P@2"] == pytest.approx(0.75)  # query 0: 2/2 relevant; query 1: 1/2
    assert m["mAP@2"] == pytest.approx((1.0 + 1.0) / 2)  # relevant results ranked first in both


def test_index_returns_cosine_nearest():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    scores, ids = build_index(gallery).search(normalise(np.array([[1.0, 0.1]])), 2)
    assert ids[0].tolist() == [0, 2] and scores[0][0] > scores[0][1]


def test_nt_xent_is_low_when_views_agree():
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    assert nt_xent(torch.cat([z, z])) < nt_xent(torch.cat([z, torch.randn(8, 16)]))


def test_views_are_random_per_image_not_per_batch():
    x = torch.full((64, 3, 32, 32), 128, dtype=torch.uint8)
    x[:, :, :, :16] = 255  # left half white: crops / flips change the mean differently per image
    torch.manual_seed(0)
    v = simclr_views(x)
    assert v.shape == (64, 3, 32, 32) and v.min() >= 0 and v.max() <= 1
    assert v.flatten(1).std(0).mean() > 0.05  # images of the batch received different transforms


def test_neural_encoders_embed_and_autoencoder_learns_colours():
    images, labels = coloured_images()
    ae = ConvAutoencoder(dim=16).fit(images, epochs=3, batch=30)
    emb = ae.embed(images)
    assert emb.shape == (60, 16)
    _, neighbours = build_index(emb).search(normalise(emb), 20)
    assert retrieval_metrics(neighbours, labels, labels, k=5)["P@5"] > 0.9
    assert SimCLR().embed(images[:4]).shape == (4, 512)


def test_api_contract():
    from vsearch import api

    images, labels = coloured_images()
    ae = ConvAutoencoder(dim=16).fit(images, epochs=3, batch=30)
    api.app.state.engine = api.Engine("autoencoder", ae, ae.embed(images), images, labels)
    with TestClient(api.app) as client:
        assert client.get("/health").json()["gallery_size"] == 60
        buffer = io.BytesIO()
        Image.fromarray(images[0]).save(buffer, format="PNG")
        body = client.post("/search?k=5", files={"file": ("q.png", buffer.getvalue(), "image/png")}).json()
        assert len(body["matches"]) == 5
        assert client.get("/images/0.png").headers["content-type"] == "image/png"
        assert client.get("/images/999.png").status_code == 404
        assert client.post("/search", files={"file": ("x.txt", b"no", "text/plain")}).status_code == 422
