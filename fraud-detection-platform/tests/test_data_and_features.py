import numpy as np
import pandas as pd
from fraudguard.data.simulate import SimulationConfig, simulate
from fraudguard.features import FEATURES, compute_features, latest_features
from fraudguard.rules import fired_rules, rule_hits


def test_simulation_is_deterministic_and_imbalanced():
    cfg = SimulationConfig(n_customers=150, n_days=40, seed=3)
    tx1, _ = simulate(cfg)
    tx2, _ = simulate(cfg)
    pd.testing.assert_frame_equal(tx1, tx2)
    assert 0 < tx1["is_fraud"].mean() < 0.05


def test_device_ids_do_not_leak_the_label(tiny_data):
    tx, _ = tiny_data
    assert tx["device_id"].str.startswith("dev_").all()
    assert not tx["device_id"].str.contains("FRAUD|TRIP", case=False).any()


def test_features_are_point_in_time(tiny_data):
    """Removing the future must not change any feature of the past (no leakage)."""
    tx, customers = tiny_data
    cutoff = tx["timestamp"].quantile(0.6)
    full = compute_features(tx, customers).set_index("tx_id")
    past_only = compute_features(tx[tx["timestamp"] <= cutoff], customers).set_index("tx_id")
    common = past_only.index
    np.testing.assert_allclose(full.loc[common, FEATURES].to_numpy(float),
                               past_only.loc[common, FEATURES].to_numpy(float))


def test_online_features_equal_offline_features(tiny_data):
    """The API path (history + new tx) reproduces the training features: no train/serve skew."""
    tx, customers = tiny_data
    offline = compute_features(tx, customers).set_index("tx_id")
    customer = tx["customer_id"].value_counts().index[0]
    rows = tx[tx["customer_id"] == customer].sort_values("timestamp")
    for i in [5, len(rows) // 2, len(rows) - 1]:
        online = latest_features(rows.iloc[:i], rows.iloc[[i]], customers)
        expected = offline.loc[rows.iloc[i]["tx_id"], FEATURES].to_numpy(float)
        np.testing.assert_allclose(online[FEATURES].to_numpy(float), expected)


def test_card_testing_rule_fires_on_micro_payment_burst():
    f = pd.DataFrame([{**dict.fromkeys(FEATURES, 0), "amount": 1.5, "tx_count_1h": 4,
                       "amount_to_mean_ratio": 0.1}])
    assert rule_hits(f).loc[0, "CARD_TESTING_BURST"]
    assert "CARD_TESTING_BURST" in fired_rules(f.iloc[0])
