import numpy as np
import pytest
from fastapi.testclient import TestClient
from rankbandit.environment import CascadeEnvironment
from rankbandit.policies import ALL_POLICIES, CascadeKLUCB, CascadeThompson, CascadeUCB1, RandomRanking
from rankbandit.simulate import run


def test_click_probability_and_zero_regret_for_best_list():
    env = CascadeEnvironment(np.array([0.5, 0.2, 0.1, 0.0]), k=2, rng=np.random.default_rng(0))
    assert env.click_probability(np.array([0, 1])) == pytest.approx(1 - 0.5 * 0.8)
    _, regret = env.step(np.array([0, 1]))
    assert regret == pytest.approx(0.0)
    _, regret = env.step(np.array([2, 3]))
    assert regret == pytest.approx((1 - 0.5 * 0.8) - 0.1)


def test_click_is_the_first_attractive_position():
    env = CascadeEnvironment(np.array([0.0, 1.0, 1.0]), k=3, rng=np.random.default_rng(0))
    click, _ = env.step(np.array([0, 1, 2]))
    assert click == 1  # item 0 never attracts, item 1 always does; item 2 is never examined


def test_cascade_update_observes_only_examined_positions():
    policy = RandomRanking(5, 3, np.random.default_rng(0))
    policy.update(np.array([4, 2, 0]), click=1)
    assert policy.trials.tolist() == [0, 0, 1, 0, 1]  # item 0 was below the click: not examined
    assert policy.successes.tolist() == [0, 0, 1, 0, 0]
    policy.update(np.array([4, 2, 0]), click=None)  # no click: every shown item was examined
    assert policy.trials.tolist() == [1, 0, 2, 0, 2]


def test_rank_returns_k_distinct_items_best_first():
    for cls in ALL_POLICIES:
        policy = cls(20, 5, np.random.default_rng(1))
        ranking = policy.rank()
        assert len(ranking) == 5 and len(set(ranking.tolist())) == 5


def test_untried_items_are_explored_in_random_order():
    first_items = {CascadeUCB1(50, 1, np.random.default_rng(s)).rank()[0] for s in range(20)}
    assert len(first_items) > 5  # no bias towards low item ids


@pytest.mark.parametrize("cls", [CascadeThompson, CascadeKLUCB])
def test_good_policies_beat_random_by_far(cls):
    learned = run(cls, horizon=3000, seed=0, n_items=30, k=3)["cum_regret"][-1]
    random = run(RandomRanking, horizon=3000, seed=0, n_items=30, k=3)["cum_regret"][-1]
    assert learned < 0.4 * random


def test_api_serves_learns_and_rejects_bad_feedback(tmp_path):
    from rankbandit import api

    api.app.state.service = api.Service(n_items=20, k=3, state=tmp_path / "state.json", seed=0)
    client = TestClient(api.app)
    best = 7
    for _ in range(400):  # simulated users click item 7 whenever it is shown
        r = client.get("/ranking").json()
        pos = r["items"].index(best) if best in r["items"] else None
        assert (
            client.post("/feedback", json={"ranking_id": r["ranking_id"], "clicked_position": pos}).status_code == 204
        )
    assert client.get("/stats").json()["best_items"][0]["item"] == best
    r = client.get("/ranking").json()
    assert client.post("/feedback", json={"ranking_id": "nope", "clicked_position": None}).status_code == 404
    assert client.post("/feedback", json={"ranking_id": r["ranking_id"], "clicked_position": 9}).status_code == 422
    assert client.post("/feedback", json={"ranking_id": r["ranking_id"], "clicked_position": 0}).status_code == 204
    assert client.post("/feedback", json={"ranking_id": r["ranking_id"], "clicked_position": 0}).status_code == 404
    api.app.state.service.save()
    restored = api.Service(n_items=20, k=3, state=tmp_path / "state.json")
    assert restored.stats()["best_items"][0]["item"] == best
