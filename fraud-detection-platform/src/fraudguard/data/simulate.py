"""Synthetic card-transaction generator with labelled fraud scenarios.

Real fraud data is confidential, so the platform is built on a simulator that
reproduces the properties that make fraud detection hard:

* strong class imbalance (well under 1 % of transactions are fraudulent);
* per-customer behaviour: usual amounts, active hours, devices, home country;
* three fraud scenarios of increasing difficulty (account takeover, card
  testing, "stealth" fraud from a known device);
* legitimate behaviour that *looks* risky (travel abroad, new phone, large
  purchases), so the classes are not trivially separable.

Drift knobs (`online_share_boost`, `travel_rate`, `new_fraud_pattern`) let the
monitoring module simulate a production period that differs from training.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from fraudguard.config import DATA_DIR, MERCHANT_CATEGORIES, MERCHANT_CHANNEL, SEED

HOME_COUNTRIES = ["FR", "BE", "ES", "DE", "IT"]
HOME_WEIGHTS = [0.85, 0.05, 0.04, 0.03, 0.03]
TRAVEL_COUNTRIES = ["ES", "IT", "PT", "GB", "US", "MA", "DE"]
RISKY_COUNTRIES = ["NG", "RU", "CN", "BR", "US", "GB"]

LEGIT_MERCHANT_P = {
    "grocery": 0.30,
    "restaurant": 0.15,
    "transport": 0.10,
    "fuel": 0.07,
    "retail": 0.12,
    "online_retail": 0.12,
    "streaming": 0.04,
    "travel": 0.04,
    "electronics": 0.04,
    "gift_cards": 0.015,
    "crypto": 0.005,
}
# Typical basket size of each category, relative to the customer's usual amount.
AMOUNT_MULTIPLIER = {
    "grocery": 1.0,
    "restaurant": 0.9,
    "transport": 0.5,
    "fuel": 1.4,
    "retail": 1.5,
    "online_retail": 1.3,
    "streaming": 0.4,
    "travel": 6.0,
    "electronics": 5.0,
    "gift_cards": 1.5,
    "crypto": 4.0,
}

COLUMNS = [
    "timestamp",
    "customer_id",
    "amount",
    "merchant_category",
    "channel",
    "country",
    "device_id",
    "is_fraud",
    "fraud_scenario",
]


@dataclass(frozen=True)
class SimulationConfig:
    n_customers: int = 4000
    n_days: int = 120
    start: str = "2026-01-01"
    victim_rate: float = 0.08  # share of customers hit by one fraud incident
    travel_rate: float = 0.15  # share of customers who travel abroad once
    device_change_rate: float = 0.10  # share of customers who change phone
    online_share_boost: float = 0.0  # drift knob: more online shopping
    new_fraud_pattern: bool = False  # drift knob: gift-card mule fraud appears
    seed: int = SEED


def make_customers(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_customers
    ids = np.array([f"C{i:05d}" for i in range(n)])
    n_devices = rng.choice([1, 2], size=n, p=[0.7, 0.3])
    return pd.DataFrame(
        {
            "customer_id": ids,
            "home_country": rng.choice(HOME_COUNTRIES, size=n, p=HOME_WEIGHTS),
            "daily_rate": rng.gamma(shape=2.0, scale=0.6, size=n),
            "amount_mu": rng.normal(3.1, 0.45, size=n),
            "amount_sigma": rng.uniform(0.5, 0.9, size=n),
            "active_hour": np.clip(rng.normal(14, 3, size=n), 8, 21),
            "n_devices": n_devices,
        }
    )


def _merchant_probabilities(online_boost: float) -> np.ndarray:
    p = np.array([LEGIT_MERCHANT_P[m] for m in MERCHANT_CATEGORIES])
    online = np.array([MERCHANT_CHANNEL[m] == "online" for m in MERCHANT_CATEGORIES])
    p = np.where(online, p * (1 + online_boost), p)
    return p / p.sum()


def _legit_transactions(
    cfg: SimulationConfig, customers: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    start = pd.Timestamp(cfg.start)
    counts = rng.poisson(customers["daily_rate"].to_numpy() * cfg.n_days)
    idx = np.repeat(np.arange(len(customers)), counts)
    c = customers.iloc[idx].reset_index(drop=True)
    n = len(c)

    day = rng.integers(0, cfg.n_days, size=n)
    hour = rng.normal(c["active_hour"], 3.5) % 24
    timestamp = start + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour * 3600, unit="s")

    merchant = rng.choice(MERCHANT_CATEGORIES, size=n, p=_merchant_probabilities(cfg.online_share_boost))
    multiplier = pd.Series(merchant).map(AMOUNT_MULTIPLIER).to_numpy()
    amount = rng.lognormal(c["amount_mu"], c["amount_sigma"]) * multiplier

    # Customers with two devices use their primary one 80 % of the time.
    second = (c["n_devices"].to_numpy() == 2) & (rng.random(n) < 0.2)
    device = np.where(second, c["customer_id"] + "-D2", c["customer_id"] + "-D1")

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "customer_id": c["customer_id"],
            "amount": np.round(np.maximum(amount, 0.5), 2),
            "merchant_category": merchant,
            "channel": pd.Series(merchant).map(MERCHANT_CHANNEL).to_numpy(),
            "country": c["home_country"],
            "device_id": device,
            "is_fraud": 0,
            "fraud_scenario": "none",
        }
    )


def _apply_legit_risky_behaviour(
    cfg: SimulationConfig, tx: pd.DataFrame, customers: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Travel and phone changes: legitimate events that resemble fraud signals."""
    start = pd.Timestamp(cfg.start)
    tx = tx.copy()
    cust = tx["customer_id"].to_numpy()
    ts = tx["timestamp"].to_numpy()

    travellers = rng.choice(
        customers["customer_id"], size=int(cfg.travel_rate * len(customers)), replace=False
    )
    for cid in travellers:
        trip_start = start + pd.Timedelta(days=int(rng.integers(0, cfg.n_days - 8)))
        trip_end = trip_start + pd.Timedelta(days=int(rng.integers(3, 9)))
        mask = (cust == cid) & (ts >= trip_start.to_datetime64()) & (ts < trip_end.to_datetime64())
        tx.loc[mask, "country"] = rng.choice(TRAVEL_COUNTRIES)
        if rng.random() < 0.2:  # holiday phone / borrowed device
            tx.loc[mask, "device_id"] = f"{cid}-TRIP"

    changers = rng.choice(
        customers["customer_id"], size=int(cfg.device_change_rate * len(customers)), replace=False
    )
    for cid in changers:
        change_at = start + pd.Timedelta(days=int(rng.integers(10, cfg.n_days)))
        mask = (cust == cid) & (ts >= change_at.to_datetime64()) & (tx["device_id"].to_numpy() == f"{cid}-D1")
        tx.loc[mask, "device_id"] = f"{cid}-D1b"
    return tx


def _fraud_rows(
    scenario: str, cust: pd.Series, t0: pd.Timestamp, rng: np.random.Generator, incident: int
) -> list[dict]:
    mu = cust["amount_mu"]
    cid = cust["customer_id"]
    fraud_device = f"FRAUD-{incident:05d}"
    rows: list[dict] = []

    def add(ts, amount, merchant, country, device):
        rows.append(
            {
                "timestamp": ts,
                "customer_id": cid,
                "amount": round(float(max(amount, 0.5)), 2),
                "merchant_category": merchant,
                "channel": MERCHANT_CHANNEL[merchant],
                "country": country,
                "device_id": device,
                "is_fraud": 1,
                "fraud_scenario": scenario,
            }
        )

    if scenario == "account_takeover":
        country = rng.choice(RISKY_COUNTRIES) if rng.random() < 0.6 else cust["home_country"]
        for dt in np.sort(rng.uniform(0, 3 * 3600, size=rng.integers(2, 7))):
            merchant = rng.choice(
                ["electronics", "gift_cards", "crypto", "online_retail"], p=[0.35, 0.25, 0.15, 0.25]
            )
            add(t0 + pd.Timedelta(seconds=dt), rng.lognormal(mu + 1.6, 0.6), merchant, country, fraud_device)
    elif scenario == "card_testing":
        country = rng.choice(RISKY_COUNTRIES) if rng.random() < 0.5 else cust["home_country"]
        last = 0.0
        for dt in np.sort(rng.uniform(0, 20 * 60, size=rng.integers(3, 9))):
            add(
                t0 + pd.Timedelta(seconds=dt),
                rng.uniform(0.5, 5),
                rng.choice(["online_retail", "streaming"]),
                country,
                fraud_device,
            )
            last = dt
        for dt in np.sort(rng.uniform(last, last + 3600, size=rng.integers(1, 3))):
            add(
                t0 + pd.Timedelta(seconds=dt),
                rng.lognormal(mu + 1.8, 0.5),
                rng.choice(["electronics", "gift_cards"]),
                country,
                fraud_device,
            )
    elif scenario == "stealth":
        for dt in np.sort(rng.uniform(0, 48 * 3600, size=rng.integers(1, 4))):
            add(
                t0 + pd.Timedelta(seconds=dt),
                rng.lognormal(mu + 1.1, 0.5),
                rng.choice(["retail", "online_retail", "electronics"]),
                cust["home_country"],
                f"{cid}-D1",
            )
    elif scenario == "gift_card_mule":
        for dt in np.sort(rng.uniform(0, 6 * 3600, size=rng.integers(3, 7))):
            add(
                t0 + pd.Timedelta(seconds=dt),
                rng.uniform(50, 150),
                "gift_cards",
                cust["home_country"],
                f"{cid}-D1",
            )
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(scenario)
    return rows


def _fraud_transactions(
    cfg: SimulationConfig, customers: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    start = pd.Timestamp(cfg.start)
    if cfg.new_fraud_pattern:
        scenarios = ["account_takeover", "card_testing", "stealth", "gift_card_mule"]
        probs = [0.35, 0.25, 0.15, 0.25]
    else:
        scenarios = ["account_takeover", "card_testing", "stealth"]
        probs = [0.45, 0.30, 0.25]

    victims = rng.choice(len(customers), size=int(cfg.victim_rate * len(customers)), replace=False)
    rows: list[dict] = []
    for incident, v in enumerate(victims):
        t0 = start + pd.Timedelta(seconds=float(rng.uniform(0, (cfg.n_days - 2) * 86400)))
        rows += _fraud_rows(rng.choice(scenarios, p=probs), customers.iloc[v], t0, rng, incident)
    return pd.DataFrame(rows, columns=COLUMNS)


def _opaque_device_id(name: str) -> str:
    return "dev_" + hashlib.sha1(name.encode()).hexdigest()[:10]


def simulate(cfg: SimulationConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, customers). Deterministic for a given config."""
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    customers = make_customers(cfg, rng)
    legit = _apply_legit_risky_behaviour(cfg, _legit_transactions(cfg, customers, rng), customers, rng)
    fraud = _fraud_transactions(cfg, customers, rng)

    tx = pd.concat([legit, fraud], ignore_index=True)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"]).astype("datetime64[ns]").dt.floor("s")
    # Opaque device ids: the generator's internal names (e.g. FRAUD-xxx) would leak the label.
    tx["device_id"] = tx["device_id"].map(_opaque_device_id)
    tx = tx.sort_values(["timestamp", "customer_id"], kind="mergesort").reset_index(drop=True)
    tx.insert(0, "tx_id", [f"T{i:07d}" for i in range(len(tx))])
    return tx, customers[["customer_id", "home_country"]]


def drift_config(base: SimulationConfig | None = None) -> SimulationConfig:
    """A 'production month' that differs from training: more online and travel, new fraud pattern."""
    base = base or SimulationConfig()
    return replace(
        base,
        n_days=60,
        start="2026-05-01",
        online_share_boost=1.0,
        travel_rate=0.40,
        new_fraud_pattern=True,
        seed=base.seed + 1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-customers", type=int, default=SimulationConfig.n_customers)
    parser.add_argument("--n-days", type=int, default=SimulationConfig.n_days)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = SimulationConfig(n_customers=args.n_customers, n_days=args.n_days)
    tx, customers = simulate(cfg)
    tx.to_parquet(DATA_DIR / "transactions.parquet", index=False)
    customers.to_parquet(DATA_DIR / "customers.parquet", index=False)

    drift_tx, _ = simulate(replace(drift_config(cfg), n_customers=cfg.n_customers))
    drift_tx.to_parquet(DATA_DIR / "transactions_drift.parquet", index=False)

    rate = tx["is_fraud"].mean()
    print(f"{len(tx):,} transactions, {tx['is_fraud'].sum():,} fraudulent ({rate:.2%})")
    print(tx.loc[tx.is_fraud == 1, "fraud_scenario"].value_counts().to_string())
    print(f"drift period: {len(drift_tx):,} transactions ({drift_tx['is_fraud'].mean():.2%} fraud)")


if __name__ == "__main__":
    main()
