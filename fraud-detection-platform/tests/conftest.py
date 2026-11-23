"""Shared fixtures: a tiny simulated dataset and a model trained on it (~10 s)."""

from dataclasses import replace

import pytest
from fraudguard.data.simulate import SimulationConfig, simulate
from fraudguard.train import train

TINY = SimulationConfig(n_customers=400, n_days=120, victim_rate=0.25, seed=7)


@pytest.fixture(scope="session")
def tiny_data():
    return simulate(TINY)


@pytest.fixture(scope="session")
def tiny_project(tmp_path_factory, tiny_data):
    """Directories with data + trained artifacts, like a real run of the pipeline."""
    root = tmp_path_factory.mktemp("fraudguard")
    data_dir, artifacts_dir, figures_dir = root / "data", root / "artifacts", root / "figures"
    data_dir.mkdir()
    tx, customers = tiny_data
    tx.to_parquet(data_dir / "transactions.parquet", index=False)
    customers.to_parquet(data_dir / "customers.parquet", index=False)
    drift_tx, _ = simulate(replace(TINY, n_days=60, online_share_boost=1.0, travel_rate=0.4, seed=8))
    drift_tx.to_parquet(data_dir / "transactions_drift.parquet", index=False)
    metrics = train(data_dir, artifacts_dir, figures_dir, num_boost_round=300, log_mlflow=False)
    return {"root": root, "data": data_dir, "artifacts": artifacts_dir, "figures": figures_dir,
            "metrics": metrics}
