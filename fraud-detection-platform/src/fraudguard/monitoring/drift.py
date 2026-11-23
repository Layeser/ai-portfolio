"""Data-drift and performance monitoring.

Compares a production period with the training reference using the
Population Stability Index (PSI), the standard drift metric in credit and
fraud risk:

    PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)   over quantile bins of the reference

Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.

Because fraud labels arrive late (chargebacks take weeks), drift on inputs and
on the score distribution is the early warning; PR-AUC on the period is
computed once labels are available.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from fraudguard.config import ARTIFACTS_DIR, DATA_DIR, FIGURES_DIR, REPORTS_DIR  # noqa: E402
from fraudguard.features import CATEGORICAL_FEATURES, FEATURES, compute_features  # noqa: E402
from fraudguard.model import FraudModel  # noqa: E402

WARNING_PSI = 0.10
ALERT_PSI = 0.25
_EPS = 1e-4


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, categorical: bool = False) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if categorical or len(np.unique(reference)) <= bins:
        values = np.union1d(np.unique(reference), np.unique(current))
        ref_p = np.array([(reference == v).mean() for v in values])
        cur_p = np.array([(current == v).mean() for v in values])
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_p = np.histogram(reference, edges)[0] / len(reference)
        cur_p = np.histogram(current, edges)[0] / len(current)
    ref_p, cur_p = np.clip(ref_p, _EPS, None), np.clip(cur_p, _EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def status(value: float) -> str:
    return "drift" if value > ALERT_PSI else "warning" if value > WARNING_PSI else "stable"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, model: FraudModel) -> dict:
    features = {
        f: round(psi(reference[f], current[f], categorical=f in CATEGORICAL_FEATURES), 4) for f in FEATURES
    }
    ref_scores, cur_scores = model.predict(reference), model.predict(current)
    report = {
        "features": {f: {"psi": v, "status": status(v)} for f, v in features.items()},
        "score_psi": round(psi(ref_scores, cur_scores), 4),
        "alert_rate": {
            "reference": round(float((ref_scores >= model.review_threshold).mean()), 5),
            "current": round(float((cur_scores >= model.review_threshold).mean()), 5),
        },
    }
    report["drifted_features"] = [f for f, v in features.items() if v > ALERT_PSI]
    if "is_fraud" in current and current["is_fraud"].sum() > 0:  # delayed labels available
        report["pr_auc"] = {
            "reference_test": model.meta["metrics"]["pr_auc"]["lightgbm"],
            "current": round(average_precision_score(current["is_fraud"], cur_scores), 4),
        }
    report["retrain_recommended"] = bool(report["drifted_features"]) or report["score_psi"] > ALERT_PSI
    return report


def plot_psi(report: dict, path: Path) -> None:
    items = sorted(report["features"].items(), key=lambda kv: kv[1]["psi"])
    names = [k for k, _ in items]
    values = [v["psi"] for _, v in items]
    colors = ["#C0392B" if v > ALERT_PSI else "#E0A33B" if v > WARNING_PSI else "#167D9A" for v in values]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(names, values, color=colors)
    ax.axvline(WARNING_PSI, color="#E0A33B", ls="--", lw=1, label="warning (0.10)")
    ax.axvline(ALERT_PSI, color="#C0392B", ls="--", lw=1, label="drift (0.25)")
    ax.set(xlabel="PSI vs training reference", title="Feature drift — simulated production period")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    model = FraudModel.load(ARTIFACTS_DIR)
    reference = pd.read_parquet(ARTIFACTS_DIR / "reference_features.parquet")
    customers = pd.read_parquet(DATA_DIR / "customers.parquet")
    drift_tx = pd.read_parquet(DATA_DIR / "transactions_drift.parquet")
    feats = compute_features(drift_tx, customers)
    day = (feats["timestamp"] - feats["timestamp"].min().normalize()).dt.days
    current = feats[day >= 30]  # first 30 days = customer history warm-up

    report = drift_report(reference, current, model)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2))
    plot_psi(report, FIGURES_DIR / "drift_psi.png")
    print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
