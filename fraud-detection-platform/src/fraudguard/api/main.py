"""Real-time scoring API.

POST /score receives one card transaction and returns, in a few milliseconds:
the fraud probability, a decision (allow / review / block), the expert rules
that fired and the top SHAP reasons, so an analyst understands every alert.

The customer history lives in an in-memory store here; in production it would
be Redis (online feature store) fed by the CDC / Kafka stream.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from fraudguard import __version__
from fraudguard.config import ARTIFACTS_DIR, DATA_DIR, MERCHANT_CATEGORIES, MERCHANT_CHANNEL
from fraudguard.features import latest_features
from fraudguard.model import FraudModel
from fraudguard.rules import fired_rules

HISTORY_DAYS = 30
HISTORY_COLUMNS = ["tx_id", "timestamp", "customer_id", "amount", "merchant_category", "channel",
                   "country", "device_id"]

SCORED = Counter("fraudguard_scored_total", "Transactions scored", ["decision"])
LATENCY = Histogram("fraudguard_score_latency_seconds", "Scoring latency",
                    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0))

MerchantCategory = Literal[tuple(MERCHANT_CATEGORIES)]  # type: ignore[valid-type]


class Transaction(BaseModel):
    tx_id: str = Field(examples=["LIVE-0001"])
    customer_id: str = Field(examples=["C00042"])
    timestamp: datetime = Field(examples=["2026-04-30T03:12:00"])
    amount: float = Field(gt=0, examples=[899.0])
    merchant_category: MerchantCategory = Field(examples=["electronics"])
    country: str = Field(min_length=2, max_length=2, examples=["NG"])
    device_id: str = Field(examples=["unknown-device-77"])


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float


class ScoreResponse(BaseModel):
    tx_id: str
    fraud_score: float
    decision: Literal["allow", "review", "block"]
    rule_hits: list[str]
    top_reasons: list[Reason]
    model_version: str
    latency_ms: float


class FeatureStore:
    """Per-customer recent history (stand-in for a Redis online feature store)."""

    def __init__(self, customers: pd.DataFrame, history: pd.DataFrame | None = None):
        self.customers = customers
        self._history: dict[str, list[dict]] = defaultdict(list)
        if history is not None:
            for cid, group in history[HISTORY_COLUMNS].groupby("customer_id"):
                self._history[cid] = group.to_dict("records")

    def history(self, customer_id: str) -> pd.DataFrame:
        return pd.DataFrame(self._history.get(customer_id, []), columns=HISTORY_COLUMNS)

    def append(self, row: dict) -> None:
        rows = self._history[row["customer_id"]]
        rows.append(row)
        cutoff = row["timestamp"] - pd.Timedelta(days=HISTORY_DAYS)
        self._history[row["customer_id"]] = [r for r in rows if r["timestamp"] >= cutoff]


state: dict = {}


def load_state() -> None:
    customers = pd.read_parquet(DATA_DIR / "customers.parquet")
    tx = pd.read_parquet(DATA_DIR / "transactions.parquet", columns=HISTORY_COLUMNS)
    recent = tx[tx["timestamp"] >= tx["timestamp"].max() - pd.Timedelta(days=HISTORY_DAYS)]
    state["model"] = FraudModel.load(ARTIFACTS_DIR)
    state["store"] = FeatureStore(customers, recent)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_state()
    yield
    state.clear()


app = FastAPI(title="FraudGuard scoring API", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    model: FraudModel | None = state.get("model")
    return {"status": "ok" if model else "loading", "model_version": model.version if model else None}


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction) -> ScoreResponse:
    model: FraudModel | None = state.get("model")
    if model is None:
        raise HTTPException(503, "model not loaded")
    start = time.perf_counter()
    store: FeatureStore = state["store"]

    row = tx.model_dump()
    row["timestamp"] = pd.Timestamp(tx.timestamp).tz_localize(None)
    row["channel"] = MERCHANT_CHANNEL[tx.merchant_category]
    new = pd.DataFrame([row])[HISTORY_COLUMNS]

    feats = latest_features(store.history(tx.customer_id), new, store.customers).to_frame().T.infer_objects()
    fraud_score = float(model.predict(feats)[0])
    decision = model.decide(fraud_score)
    reasons = model.explain(feats)[0]
    rules = fired_rules(feats.iloc[0])
    store.append(row)

    elapsed = time.perf_counter() - start
    SCORED.labels(decision).inc()
    LATENCY.observe(elapsed)
    return ScoreResponse(
        tx_id=tx.tx_id,
        fraud_score=round(fraud_score, 5),
        decision=decision,
        rule_hits=rules,
        top_reasons=[Reason(**r) for r in reasons],
        model_version=model.version,
        latency_ms=round(elapsed * 1000, 2),
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
