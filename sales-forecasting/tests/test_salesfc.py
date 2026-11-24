"""Fast tests: metric, loss, model shapes and API contract on synthetic data (no M5 download)."""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from salesfc.data import HORIZON
from salesfc.tide import LOOKBACK, TiDE, tweedie_loss


def test_tweedie_loss_is_minimised_at_the_true_mean():
    y = torch.full((1000, HORIZON), 3.0)
    losses = {mu: tweedie_loss(torch.full_like(y, np.log(mu)), y).item() for mu in (1.0, 3.0, 9.0)}
    assert losses[3.0] < losses[1.0] and losses[3.0] < losses[9.0]


def test_tide_output_shape_and_gradients():
    torch.manual_seed(0)
    model = TiDE(n_cov=20, cardinalities=[50, 7, 3, 10, 3])
    y = torch.rand(8, LOOKBACK)
    cov = torch.rand(8, LOOKBACK + HORIZON, 20)
    static = torch.tensor([[1, 2, 0, 3, 1]] * 8)
    out = model(y, cov, static)
    assert out.shape == (8, HORIZON)
    tweedie_loss(out, torch.rand(8, HORIZON) * 3).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_rolling_mean_uses_only_the_past():
    from salesfc.features import _rolling_mean

    m = np.arange(20, dtype=np.float32)[None]  # value = day index
    out = _rolling_mean(m, np.array([15]), offset=5, window=3)  # days 8, 9, 10: nothing after day 15 - 5
    assert out[0, 0] == pytest.approx(9.0)


def test_api_contract():
    from salesfc import api

    engine = api.Engine(["FOODS_1_001_CA_1", "FOODS_1_002_CA_1"], np.ones((2, HORIZON)) * [[1.5], [0.2]],
                        [f"2016-05-{d:02d}" for d in range(1, HORIZON + 1)], np.array([1.4, 0.3]), "test")
    api.app.state.engine = engine
    with TestClient(api.app) as client:
        assert client.get("/health").json()["series"] == 2
        body = client.get("/forecast/FOODS_1_001/CA_1").json()
        assert len(body["forecast"]) == HORIZON and body["forecast"][0]["units"] == 1.5
        assert client.get("/forecast/NOPE/CA_1").status_code == 404
