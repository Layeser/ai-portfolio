"""Tests run on a small synthetic catalogue: no private data and no GPU needed."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prodcat.data import build_text, clean, leakage_report, validate
from prodcat.metrics import coverage_curve, hierarchical_prf, level_accuracy, threshold_for_accuracy
from prodcat.models import TfidfModel
from prodcat.taxonomy import Taxonomy

PATHS = ["root/elec/tv/oled", "root/elec/tv/led", "root/elec/audio/headphones", "root/home/garden/hose",
         "root/home/kitchen/wine_cellar"]
WORDS = {"oled": "téléviseur OLED 4K dalle organique", "led": "téléviseur LED rétroéclairage", "headphones":
         "casque audio bluetooth réduction de bruit", "hose": "tuyau d'arrosage jardin 20 m",
         "wine_cellar": "cave à vin bouteilles deux zones température"}


@pytest.fixture(scope="module")
def tax():
    return Taxonomy.from_paths(PATHS)


@pytest.fixture(scope="module")
def catalogue():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        path = PATHS[i % len(PATHS)]
        leaf = path.split("/")[-1]
        rows.append({"product_id": f"p{i}", "category_id": leaf, "category_path": path,
                     "title": f"{WORDS[leaf]} modèle {rng.integers(1000)}",
                     "description": "<b>Promo</b> &amp; livraison",
                     "brand": rng.choice(["ACME", "Nordia", None]), "color": rng.choice(["noir", None])})
    df = pd.DataFrame(rows)
    df["text"] = build_text(df)
    return df


def test_clean_strips_html_and_entities():
    assert clean("<p>Cave&nbsp;à <b>vin</b></p>") == "Cave à vin"
    assert clean(None) == ""


def test_validation_catches_inconsistent_paths(catalogue):
    assert validate(catalogue, "ok") == []
    broken = catalogue.copy()
    broken.loc[0, "category_path"] = "root/elec/tv/led"
    assert any("category_path" in issue for issue in validate(broken, "bad"))


def test_leakage_report_flags_shared_titles(catalogue):
    report, keep = leakage_report(catalogue.iloc[:200], catalogue.iloc[:50])
    assert report["test_rows_with_title_seen_in_train"] == 50 and not keep.any()


def test_taxonomy_paths_and_rollup(tax):
    assert tax.path("oled") == ["root", "elec", "tv", "oled"]
    assert set(tax.leaves) == {p.split("/")[-1] for p in PATHS}
    probs = np.array([0.45, 0.45, 0.04, 0.03, 0.03])
    labels = ["oled", "led", "headphones", "hose", "wine_cellar"]
    node, p = tax.deepest_confident_node(probs, labels, threshold=0.8)
    assert node == "tv" and p == pytest.approx(0.9)  # unsure between siblings, sure about the parent


def test_hierarchical_metrics_give_partial_credit_to_siblings(tax):
    sibling = hierarchical_prf(["oled"], ["led"], tax)["hF1"]
    far = hierarchical_prf(["oled"], ["hose"], tax)["hF1"]
    assert hierarchical_prf(["oled"], ["oled"], tax)["hF1"] == 1.0
    assert far == 0 < sibling < 1  # only the root is shared, and the root is excluded
    assert level_accuracy(["oled"], ["led"], tax, max_level=3) == {"level_1": 1.0, "level_2": 1.0, "level_3": 0.0}


def test_threshold_reaches_target_accuracy():
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.5, 0.4])
    ok = np.array([1, 1, 1, 0, 1, 0])
    t = threshold_for_accuracy(conf, ok, 0.95)
    assert t == 0.9
    _, coverage, acc = coverage_curve(conf, ok)
    assert coverage[-1] == 1.0 and acc[2] == 1.0


@pytest.fixture(scope="module")
def service(catalogue, tax):
    from prodcat.api import Service

    model = TfidfModel(C=5).fit(catalogue["text"], catalogue["category_id"])
    return Service(model, tax, threshold=0.5)


def test_model_learns_the_catalogue(service, catalogue):
    probs = service.model.predict_proba(catalogue["text"])
    pred = np.array(service.model.classes_)[probs.argmax(1)]
    assert (pred == catalogue["category_id"]).mean() > 0.95


def test_api_contract_and_routing(service):
    from prodcat.api import app

    app.state.service = service
    with TestClient(app) as client:
        assert client.get("/health").json()["model"] == "tfidf"
        body = client.post("/classify", json={"title": "Cave à vin 38 bouteilles deux zones"}).json()
        assert body["category_id"] == "wine_cellar"
        assert body["path"] == ["root", "home", "kitchen", "wine_cellar"]
        assert body["decision"] in {"auto", "review"} and len(body["top3"]) == 3
        batch = client.post("/classify/batch", json=[{"title": "casque bluetooth"}, {"title": "tuyau jardin"}])
        assert [b["category_id"] for b in batch.json()] == ["headphones", "hose"]
        assert client.post("/classify", json={"title": ""}).status_code == 422
        assert client.post("/classify/batch", json=[]).status_code == 422
