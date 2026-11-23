"""Point-in-time feature engineering.

Every behavioural feature of a transaction is computed **only from the
customer's earlier transactions**. This matters twice:

* offline, it prevents label leakage (a feature must never "see" the future);
* online, the scoring API calls the *same function* on the customer's history
  plus the incoming transaction, so there is no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudguard.config import MERCHANT_CATEGORIES

FEATURES = [
    "amount",
    "log_amount",
    "hour",
    "is_night",
    "is_online",
    "merchant_code",
    "is_foreign",
    "new_device",
    "new_country",
    "amount_to_mean_ratio",
    "tx_count_1h",
    "tx_count_24h",
    "amount_sum_24h",
    "new_device_tx_24h",
    "secs_since_last_tx",
]
CATEGORICAL_FEATURES = ["merchant_code"]

_MERCHANT_CODE = {m: i for i, m in enumerate(MERCHANT_CATEGORIES)}
_NO_PREVIOUS_TX = 30 * 86400.0


def _rolling(df: pd.DataFrame, column: str, window: str, how: str) -> np.ndarray:
    """Rolling aggregate per customer over the *previous* transactions only (closed='left')."""
    rolled = (
        df.set_index("timestamp")
        .groupby("customer_id", sort=False)[column]
        .rolling(window, closed="left")
        .agg(how)
    )
    return rolled.to_numpy()


def compute_features(tx: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Add the model features to a transaction table.

    `tx` needs: tx_id, timestamp, customer_id, amount, merchant_category,
    channel, country, device_id. `customers` maps customer_id -> home_country.
    The result is sorted by (customer_id, timestamp).
    """
    df = tx.merge(customers[["customer_id", "home_country"]], on="customer_id", how="left")
    df["home_country"] = df["home_country"].fillna(df["country"])  # unknown customer: no foreign flag
    df = df.sort_values(["customer_id", "timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    by_customer = df.groupby("customer_id", sort=False)

    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 23)).astype(int)
    df["is_online"] = (df["channel"] == "online").astype(int)
    df["merchant_code"] = df["merchant_category"].map(_MERCHANT_CODE).fillna(-1).astype(int)
    df["is_foreign"] = (df["country"] != df["home_country"]).astype(int)

    # First time this customer uses this device / pays from this country.
    df["new_device"] = (df.groupby(["customer_id", "device_id"], sort=False).cumcount() == 0).astype(int)
    df["new_country"] = (df.groupby(["customer_id", "country"], sort=False).cumcount() == 0).astype(int)

    # Ratio to the customer's average *past* amount (expanding mean, current row excluded).
    previous_count = by_customer.cumcount()
    previous_sum = by_customer["amount"].cumsum() - df["amount"]
    past_mean = previous_sum / previous_count.replace(0, np.nan)
    df["amount_to_mean_ratio"] = (df["amount"] / past_mean).fillna(1.0)

    # Velocity features over the previous 1 h / 24 h.
    df["tx_count_1h"] = np.nan_to_num(_rolling(df, "amount", "1h", "count"))
    df["tx_count_24h"] = np.nan_to_num(_rolling(df, "amount", "24h", "count"))
    df["amount_sum_24h"] = np.nan_to_num(_rolling(df, "amount", "24h", "sum"))
    df["new_device_tx_24h"] = np.nan_to_num(_rolling(df, "new_device", "24h", "sum"))

    elapsed = by_customer["timestamp"].diff().dt.total_seconds()
    df["secs_since_last_tx"] = elapsed.fillna(_NO_PREVIOUS_TX)
    return df


def latest_features(history: pd.DataFrame, new_tx: pd.DataFrame, customers: pd.DataFrame) -> pd.Series:
    """Online path: features of `new_tx` given the customer's `history` (same code as offline)."""
    frame = pd.concat([history, new_tx], ignore_index=True)
    feats = compute_features(frame, customers)
    return feats.loc[feats["tx_id"] == new_tx["tx_id"].iloc[0]].iloc[0]
