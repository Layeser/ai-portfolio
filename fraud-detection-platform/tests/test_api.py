import fraudguard.api.main as api
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tiny_project):
    mp = pytest.MonkeyPatch()
    mp.setattr(api, "ARTIFACTS_DIR", tiny_project["artifacts"])
    mp.setattr(api, "DATA_DIR", tiny_project["data"])
    with TestClient(api.app) as c:
        yield c
    mp.undo()


def _customer_profile(tiny_data):
    tx, _ = tiny_data
    cid = tx["customer_id"].value_counts().index[0]
    usual = tx[(tx.customer_id == cid) & (tx.is_fraud == 0)].iloc[-1]
    return cid, usual


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"]


def test_score_contract(client, tiny_data):
    cid, usual = _customer_profile(tiny_data)
    payload = {"tx_id": "LIVE-1", "customer_id": cid, "timestamp": "2026-05-01T12:00:00", "amount": 20.0,
               "merchant_category": "grocery", "country": usual.country, "device_id": usual.device_id}
    body = client.post("/score", json=payload).json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in {"allow", "review", "block"}
    assert isinstance(body["rule_hits"], list) and isinstance(body["top_reasons"], list)


def test_account_takeover_pattern_scores_higher_than_usual_purchase(client, tiny_data):
    cid, usual = _customer_profile(tiny_data)
    base = {"customer_id": cid, "country": usual.country, "device_id": usual.device_id}
    normal = client.post("/score", json={**base, "tx_id": "N1", "timestamp": "2026-05-02T13:00:00",
                                         "amount": 25.0, "merchant_category": "grocery"}).json()
    attack = None
    for i in range(4):
        attack = client.post("/score", json={**base, "tx_id": f"A{i}", "timestamp": f"2026-05-02T03:0{i}:00",
                                             "amount": 900.0, "merchant_category": "electronics",
                                             "country": "NG", "device_id": "dev_unknown"}).json()
    assert attack["fraud_score"] > normal["fraud_score"]
    assert attack["decision"] in {"review", "block"}
    assert "NEW_DEVICE_FOREIGN" in attack["rule_hits"]


@pytest.mark.parametrize("patch", [{"amount": -5}, {"merchant_category": "casino"}, {"country": "FRA"}])
def test_invalid_payloads_are_rejected(client, patch):
    payload = {"tx_id": "X", "customer_id": "C00001", "timestamp": "2026-05-01T12:00:00", "amount": 10.0,
               "merchant_category": "grocery", "country": "FR", "device_id": "d"}
    assert client.post("/score", json={**payload, **patch}).status_code == 422


def test_prometheus_metrics_exposed(client):
    assert "fraudguard_scored_total" in client.get("/metrics").text
