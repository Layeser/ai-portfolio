"""Read-only access to alerts and customer data for the investigation agent.

Ground-truth labels (`is_fraud`, `fraud_scenario`) are dropped on load: the
agent must reason like an analyst, from evidence only.
"""

from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

import pandas as pd

from fraudguard.config import ARTIFACTS_DIR, DATA_DIR
from fraudguard.rules import RULES_BY_NAME

LABEL_COLUMNS = ["is_fraud", "fraud_scenario"]
ALERT_FIELDS = ["tx_id", "customer_id", "timestamp", "amount", "merchant_category", "country", "device_id",
                "fraud_score", "decision"]
SIGNAL_FIELDS = ["is_foreign", "new_device", "new_country", "amount_to_mean_ratio", "tx_count_1h",
                 "tx_count_24h", "amount_sum_24h", "new_device_tx_24h", "is_night"]


def _jsonable(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        if isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        elif isinstance(v, float):
            out[k] = round(v, 4)
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


class CaseStore:
    def __init__(self, artifacts_dir: Path | None = None, data_dir: Path | None = None):
        # Resolved at call time so tests / containers can redirect them with env variables.
        self.artifacts_dir = Path(artifacts_dir or os.getenv("FRAUDGUARD_ARTIFACTS_DIR", ARTIFACTS_DIR))
        self.data_dir = Path(data_dir or os.getenv("FRAUDGUARD_DATA_DIR", DATA_DIR))

    @cached_property
    def alerts(self) -> pd.DataFrame:
        alerts = pd.read_parquet(self.artifacts_dir / "alerts.parquet")
        return alerts.drop(columns=LABEL_COLUMNS, errors="ignore").set_index("tx_id", drop=False)

    @cached_property
    def transactions(self) -> pd.DataFrame:
        tx = pd.read_parquet(self.data_dir / "transactions.parquet").drop(columns=LABEL_COLUMNS)
        return tx.sort_values("timestamp")

    @cached_property
    def customers(self) -> pd.DataFrame:
        return pd.read_parquet(self.data_dir / "customers.parquet").set_index("customer_id")

    # ---- tools -------------------------------------------------------------------------------

    def list_open_alerts(self, limit: int = 10) -> list[dict]:
        top = self.alerts.sort_values("fraud_score", ascending=False).head(limit)
        return [_jsonable(r) for r in top[ALERT_FIELDS].to_dict("records")]

    def get_alert(self, tx_id: str) -> dict:
        if tx_id not in self.alerts.index:
            return {"error": f"unknown alert {tx_id}"}
        row = self.alerts.loc[tx_id]
        alert = _jsonable(row[ALERT_FIELDS].to_dict())
        alert["signals"] = _jsonable(row[SIGNAL_FIELDS].to_dict())
        alert["rule_hits"] = [
            {"rule": name, "description": RULES_BY_NAME[name].description}
            for name in json.loads(row["rule_hits"])
        ]
        alert["model_top_reasons"] = json.loads(row["top_reasons"])
        return alert

    def get_customer_profile(self, customer_id: str, as_of: str) -> dict:
        """Behaviour over the 90 days *before* `as_of`."""
        if customer_id not in self.customers.index:
            return {"error": f"unknown customer {customer_id}"}
        end = pd.Timestamp(as_of)
        tx = self.transactions
        past = tx[(tx.customer_id == customer_id) & (tx.timestamp < end)
                  & (tx.timestamp >= end - pd.Timedelta(days=90))]
        hours = past.timestamp.dt.hour
        return _jsonable({
            "customer_id": customer_id,
            "home_country": self.customers.loc[customer_id, "home_country"],
            "tx_last_90d": int(len(past)),
            "median_amount": float(past.amount.median()) if len(past) else None,
            "p95_amount": float(past.amount.quantile(0.95)) if len(past) else None,
            "max_amount": float(past.amount.max()) if len(past) else None,
            "known_devices": past.device_id.value_counts().head(5).to_dict(),
            "countries": past.country.value_counts().to_dict(),
            "top_merchant_categories": past.merchant_category.value_counts().head(5).to_dict(),
            "usual_hours": f"{int(hours.quantile(0.1))}h-{int(hours.quantile(0.9))}h" if len(past) else None,
            "online_share": round(float((past.channel == "online").mean()), 3) if len(past) else None,
        })

    def get_recent_activity(self, customer_id: str, around: str, hours_before: int = 24,
                            hours_after: int = 6) -> list[dict]:
        """Transactions of the customer around a timestamp (analysts look after the alert too)."""
        t = pd.Timestamp(around)
        tx = self.transactions
        window = tx[(tx.customer_id == customer_id) & (tx.timestamp >= t - pd.Timedelta(hours=hours_before))
                    & (tx.timestamp <= t + pd.Timedelta(hours=hours_after))]
        window = window.assign(is_alert=window.tx_id.isin(self.alerts.index))
        cols = ["tx_id", "timestamp", "amount", "merchant_category", "country", "device_id", "is_alert"]
        return [_jsonable(r) for r in window[cols].head(40).to_dict("records")]
