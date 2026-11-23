import json

import numpy as np
import pandas as pd
from fraudguard.model import FraudModel
from fraudguard.monitoring.drift import drift_report, psi, status


def test_training_outputs_and_model_beats_rules(tiny_project):
    m = tiny_project["metrics"]
    assert m["pr_auc"]["lightgbm"] > m["pr_auc"]["rules"]
    assert m["operating_points"]["block"]["threshold"] >= m["operating_points"]["review"]["threshold"]
    for f in ["model.txt", "model_meta.json", "metrics.json", "alerts.parquet", "reference_features.parquet"]:
        assert (tiny_project["artifacts"] / f).exists(), f
    assert (tiny_project["figures"] / "pr_curves.png").exists()


def test_model_explanations_are_positive_contributions(tiny_project):
    model = FraudModel.load(tiny_project["artifacts"])
    alerts = pd.read_parquet(tiny_project["artifacts"] / "alerts.parquet")
    for reasons in model.explain(alerts.head(5)):
        assert all(r["contribution"] > 0 for r in reasons)
    assert all(len(json.loads(r)) <= 3 for r in alerts["top_reasons"])


def test_psi_detects_shift_and_not_noise():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    assert psi(ref, rng.normal(0, 1, 20_000)) < 0.02
    assert psi(ref, rng.normal(1, 1, 20_000)) > 0.25
    assert status(0.05) == "stable" and status(0.2) == "warning" and status(0.4) == "drift"


def test_drift_report_flags_shifted_period(tiny_project):
    model = FraudModel.load(tiny_project["artifacts"])
    reference = pd.read_parquet(tiny_project["artifacts"] / "reference_features.parquet")
    shifted = reference.assign(is_foreign=1)
    report = drift_report(reference, shifted, model)
    assert "is_foreign" in report["drifted_features"]
    assert report["retrain_recommended"]
    assert not drift_report(reference, reference, model)["drifted_features"]
