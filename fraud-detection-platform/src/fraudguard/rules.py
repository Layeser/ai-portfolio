"""Expert rules: the baseline every fraud team starts from.

Rules are kept next to the ML score in production: they are auditable,
explain alerts in business language and catch known patterns from day one.
The ML model is benchmarked against them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    weight: float
    condition: Callable[[pd.DataFrame], pd.Series]


RULES: list[Rule] = [
    Rule(
        "CARD_TESTING_BURST",
        "3+ transactions in the previous hour and a micro-payment (< 5 EUR)",
        3.0,
        lambda f: (f["tx_count_1h"] >= 3) & (f["amount"] < 5),
    ),
    Rule(
        "NEW_DEVICE_FOREIGN",
        "First use of this device, from a country that is not the customer's home",
        3.0,
        lambda f: (f["new_device_tx_24h"] + f["new_device"] >= 1) & (f["is_foreign"] == 1),
    ),
    Rule(
        "HIGH_AMOUNT_VS_HISTORY",
        "Amount more than 5x the customer's average past amount",
        2.0,
        lambda f: f["amount_to_mean_ratio"] > 5,
    ),
    Rule(
        "RISKY_MERCHANT_NEW_DEVICE",
        "Gift cards, crypto or electronics bought from a recently new device",
        2.0,
        lambda f: f["merchant_code"].isin([8, 9, 10]) & (f["new_device_tx_24h"] + f["new_device"] >= 1),
    ),
    Rule(
        "NIGHT_HIGH_AMOUNT",
        "Night-time transaction (23h-6h) more than 3x the usual amount",
        1.0,
        lambda f: (f["is_night"] == 1) & (f["amount_to_mean_ratio"] > 3),
    ),
    Rule(
        "VELOCITY_24H",
        "10+ transactions in the previous 24 hours",
        1.0,
        lambda f: f["tx_count_24h"] >= 10,
    ),
]
RULES_BY_NAME = {r.name: r for r in RULES}


def rule_hits(features: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix: one column per rule."""
    return pd.DataFrame({r.name: r.condition(features).astype(bool) for r in RULES}, index=features.index)


def rule_score(features: pd.DataFrame) -> np.ndarray:
    """Weighted number of rules fired, scaled to [0, 1] (the rules-only baseline)."""
    hits = rule_hits(features)
    weights = np.array([r.weight for r in RULES])
    return hits.to_numpy() @ weights / weights.sum()


def fired_rules(features_row: pd.Series) -> list[str]:
    frame = features_row.to_frame().T.infer_objects()
    hits = rule_hits(frame).iloc[0]
    return [name for name, hit in hits.items() if hit]
