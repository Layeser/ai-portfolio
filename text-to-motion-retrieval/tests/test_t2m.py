"""Tests on a tiny synthetic dataset written to a temp dir (no challenge data needed)."""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from t2m.data import MotionTextDataset, Normalizer, collate, crop, query_groups, split_ids
from t2m.model import clip_loss
from t2m.train import challenge_score, recall_at_k

ACTIONS = ["two people shake hands", "one person pushes the other", "they dance together", "a person waves"]


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("t2m")
    (root / "motions").mkdir()
    (root / "texts").mkdir()
    rng = np.random.default_rng(0)
    for i in range(40):
        pattern = np.zeros(384, dtype=np.float32)
        pattern[(i % 4) * 96:(i % 4 + 1) * 96] = 1.0  # each action lights a different feature block
        frames = rng.integers(20, 260)
        motion = pattern + rng.normal(0, 0.1, (frames, 384)).astype(np.float32)
        np.save(root / "motions" / f"{i}.npy", motion)
        (root / "texts" / f"{i}.txt").write_text(f"{ACTIONS[i % 4]}.\n{ACTIONS[i % 4]} slowly.\n")
    return root


def test_split_is_deterministic_and_disjoint():
    ids = [str(i) for i in range(100)]
    a = split_ids(ids)
    assert a == split_ids(ids)
    train, hold = a
    assert not set(train) & set(hold) and len(hold) == 10 and len(train) + len(hold) == 100


def test_normalizer_uses_only_given_ids(tiny_data):
    ids = [str(i) for i in range(40)]
    train, _ = split_ids(ids)
    norm = Normalizer.fit(train, tiny_data)
    stacked = np.concatenate([np.load(tiny_data / "motions" / f"{m}.npy") for m in train])
    np.testing.assert_allclose(norm.mean, stacked.mean(0), atol=1e-4)


def test_crop_and_collate_mask():
    m = np.arange(300 * 2, dtype=np.float32).reshape(300, 2)
    assert crop(m, 196, None).shape == (196, 2) and crop(m[:50], 196, None).shape == (50, 2)
    batch = [(torch.ones(5, 3), "a", "1"), (torch.ones(2, 3), "b", "2")]
    padded, mask, texts, ids = collate(batch)
    assert padded.shape == (2, 5, 3) and mask.sum().item() == 7 and not mask[1, 2:].any()


def test_query_groups_hide_the_target_among_distractors():
    ids = [str(i) for i in range(50)]
    for target, candidates in query_groups(ids, n_candidates=32):
        assert target in candidates and len(set(candidates)) == 32


def test_metrics():
    ranks = np.array([1, 1, 2, 11])
    assert recall_at_k(ranks) == {"R@1": 0.5, "R@5": 0.75, "R@10": 0.75}
    assert challenge_score(np.ones(10)) == pytest.approx(1.0)


def test_clip_loss_prefers_aligned_pairs():
    torch.manual_seed(0)
    emb = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    scale = torch.tensor(np.log(20.0))
    assert clip_loss(emb, emb, scale) < clip_loss(emb, emb.roll(1, 0), scale)


def test_short_training_learns_and_search_api_works(tiny_data, tmp_path):
    from t2m import api
    from t2m.train import train

    ids = [str(i) for i in range(40)]
    result = train(epochs=5, batch_size=8, lr=1e-3, artifacts=tmp_path, data_dir=tiny_data, ids=ids)
    assert result["history"][-1]["loss"] < result["history"][0]["loss"]
    assert (tmp_path / "dual_encoder.pt").exists()

    from t2m.model import DualEncoder
    from t2m.train import embed

    model = DualEncoder()
    model.load_state_dict(torch.load(tmp_path / "dual_encoder.pt"))
    dataset = MotionTextDataset(ids, Normalizer.load(tmp_path / "normalizer.npz"), train=False, data_dir=tiny_data)
    emb, emb_ids = embed(model, dataset, "cpu")
    api.app.state.search = api.MotionSearch(model, emb, emb_ids, dataset.texts, set(ids[:30]), "cpu")
    with TestClient(api.app) as client:
        assert client.get("/health").json()["motions_indexed"] == 40
        body = client.post("/search", json={"text": "they dance together", "k": 3}).json()
        assert len(body["matches"]) == 3 and body["matches"][0]["score"] >= body["matches"][-1]["score"]
        assert client.post("/search", json={"text": "x"}).status_code == 422
