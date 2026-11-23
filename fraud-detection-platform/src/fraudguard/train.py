"""Train and evaluate the fraud model: rules vs logistic regression vs LightGBM.

Design choices (each one is a classic interview question):

* **Time-based split** (train -> validation -> test in chronological order):
  a random split would leak future fraud patterns into training.
* **PR-AUC as the main metric**: with ~0.25 % positives, ROC-AUC looks
  excellent even for weak models; PR-AUC reflects the alert quality analysts see.
* **Thresholds chosen on validation, reported on test**, from business
  constraints: `block` needs precision >= 90 %, `review` needs precision >= 30 %
  (an analyst can afford ~2 false alerts per true fraud).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: E402

from fraudguard.config import DATA_DIR, FIGURES_DIR, ROOT, SEED  # noqa: E402
from fraudguard.features import CATEGORICAL_FEATURES, FEATURES, compute_features  # noqa: E402
from fraudguard.model import FraudModel  # noqa: E402
from fraudguard.rules import rule_hits, rule_score  # noqa: E402

WARMUP_DAYS = 30  # history only: lets "new device / new country" features mean something
VALID_START_DAY = 90
TEST_START_DAY = 105
BLOCK_MIN_PRECISION = 0.90
REVIEW_MIN_PRECISION = 0.30

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "average_precision",
    "verbose": -1,
    "seed": SEED,
}


def time_split(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = (feats["timestamp"] - feats["timestamp"].min().normalize()).dt.days
    train = feats[(day >= WARMUP_DAYS) & (day < VALID_START_DAY)]
    valid = feats[(day >= VALID_START_DAY) & (day < TEST_START_DAY)]
    test = feats[day >= TEST_START_DAY]
    return train, valid, test


def threshold_for_precision(y: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    """Lowest threshold (= highest recall) whose precision reaches `min_precision`."""
    precision, _, thresholds = precision_recall_curve(y, scores)
    ok = np.where(precision[:-1] >= min_precision)[0]
    return float(thresholds[ok[0]]) if len(ok) else float(thresholds[-1])


def operating_point(y: np.ndarray, scores: np.ndarray, amount: np.ndarray, threshold: float) -> dict:
    flagged = scores >= threshold
    tp = int((flagged & (y == 1)).sum())
    return {
        "threshold": round(threshold, 4),
        "precision": round(tp / max(flagged.sum(), 1), 4),
        "recall": round(tp / max(y.sum(), 1), 4),
        "fraud_amount_recall": round(float(amount[flagged & (y == 1)].sum() / amount[y == 1].sum()), 4),
        "alerts_per_10k_tx": round(float(flagged.mean() * 10_000), 1),
    }


def logistic_baseline() -> object:
    numeric = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return make_pipeline(pre, LogisticRegression(class_weight="balanced", max_iter=2000))


def plot_pr_curves(y: np.ndarray, curves: dict[str, np.ndarray], points: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = {"Rules": "#9AA5B1", "Logistic regression": "#E0A33B", "LightGBM": "#167D9A"}
    for name, scores in curves.items():
        p, r, _ = precision_recall_curve(y, scores)
        ax.plot(r, p, label=f"{name} (PR-AUC {average_precision_score(y, scores):.3f})",
                color=colors[name], lw=2.2 if name == "LightGBM" else 1.6)
    for label, pt, marker in [("block", points["block"], "s"), ("review", points["review"], "o")]:
        ax.scatter(pt["recall"], pt["precision"], color="#17324D", marker=marker, zorder=5, s=46)
        ax.annotate(f"{label} @ {pt['threshold']:.2f}", (pt["recall"], pt["precision"]),
                    textcoords="offset points", xytext=(-80, -14), fontsize=8.5, color="#17324D")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.02),
           title="Test set — precision/recall (chronological split)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_importance(model: FraudModel, sample: pd.DataFrame, path: Path) -> None:
    contrib = model.booster.predict(sample[FEATURES].astype(float), pred_contrib=True)[:, :-1]
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(importance.index, importance.values, color="#167D9A")
    ax.set(xlabel="mean |SHAP value| (log-odds)", title="What drives the fraud score")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(
    data_dir: Path = DATA_DIR,
    artifacts_dir: Path = ROOT / "artifacts",  # never the shipped models/pretrained
    figures_dir: Path = FIGURES_DIR,
    num_boost_round: int = 2000,
    log_mlflow: bool = True,
) -> dict:
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    customers = pd.read_parquet(data_dir / "customers.parquet")
    feats = compute_features(tx, customers)
    train_df, valid_df, test_df = time_split(feats)
    y_train, y_valid, y_test = (d["is_fraud"].to_numpy() for d in (train_df, valid_df, test_df))

    # 1) rules baseline
    rules_test = rule_score(test_df)

    # 2) logistic regression baseline
    logreg = logistic_baseline().fit(train_df[FEATURES], y_train)
    logreg_test = logreg.predict_proba(test_df[FEATURES])[:, 1]

    # 3) LightGBM with early stopping on the validation period
    dtrain = lgb.Dataset(train_df[FEATURES].astype(float), y_train, categorical_feature=CATEGORICAL_FEATURES)
    dvalid = lgb.Dataset(valid_df[FEATURES].astype(float), y_valid, reference=dtrain)
    booster = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=num_boost_round, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    valid_scores = booster.predict(valid_df[FEATURES].astype(float), num_iteration=booster.best_iteration)
    thresholds = {
        "block": threshold_for_precision(y_valid, valid_scores, BLOCK_MIN_PRECISION),
        "review": threshold_for_precision(y_valid, valid_scores, REVIEW_MIN_PRECISION),
    }
    model = FraudModel(booster, {"thresholds": thresholds})
    lgb_test = model.predict(test_df)

    amount = test_df["amount"].to_numpy()
    points = {name: operating_point(y_test, lgb_test, amount, t) for name, t in thresholds.items()}
    review_flag = lgb_test >= thresholds["review"]
    recall_by_scenario = (
        test_df.assign(flag=review_flag)[test_df["is_fraud"] == 1]
        .groupby("fraud_scenario")["flag"].mean().round(4).to_dict()
    )
    metrics = {
        "pr_auc": {
            "rules": round(average_precision_score(y_test, rules_test), 4),
            "logistic_regression": round(average_precision_score(y_test, logreg_test), 4),
            "lightgbm": round(average_precision_score(y_test, lgb_test), 4),
        },
        "roc_auc_lightgbm": round(roc_auc_score(y_test, lgb_test), 4),
        "operating_points": points,
        "recall_by_scenario_at_review": recall_by_scenario,
        "best_iteration": booster.best_iteration,
        "rows": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
        "fraud_rate_test": round(float(y_test.mean()), 5),
    }
    model.meta = {
        "version": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        "features": FEATURES,
        "thresholds": thresholds,
        "metrics": metrics,
        "train_period": [str(train_df.timestamp.min()), str(train_df.timestamp.max())],
    }
    model.save(artifacts_dir)

    # Reference sample for drift monitoring + scored alert queue for the agent.
    train_df.sample(min(50_000, len(train_df)), random_state=SEED)[FEATURES + ["is_fraud"]].to_parquet(
        artifacts_dir / "reference_features.parquet", index=False
    )
    alerts = test_df.assign(fraud_score=lgb_test).loc[review_flag].copy()
    alerts["decision"] = [model.decide(s) for s in alerts["fraud_score"]]
    alerts["top_reasons"] = [json.dumps(r) for r in model.explain(alerts)]
    hits = rule_hits(alerts)
    alerts["rule_hits"] = [json.dumps([c for c in hits.columns if row[c]]) for _, row in hits.iterrows()]
    alerts.to_parquet(artifacts_dir / "alerts.parquet", index=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    curves = {"Rules": rules_test, "Logistic regression": logreg_test, "LightGBM": lgb_test}
    plot_pr_curves(y_test, curves, points, figures_dir / "pr_curves.png")
    plot_importance(model, test_df.sample(min(20_000, len(test_df)), random_state=SEED),
                    figures_dir / "feature_importance.png")
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if log_mlflow:
        _log_to_mlflow(artifacts_dir, figures_dir, metrics)
    return metrics


def _log_to_mlflow(artifacts_dir: Path, figures_dir: Path, metrics: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{artifacts_dir / 'mlflow.db'}")
    mlflow.set_experiment("fraudguard")
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_params({k: v for k, v in LGB_PARAMS.items() if k != "verbose"})
        mlflow.log_params({"block_min_precision": BLOCK_MIN_PRECISION,
                           "review_min_precision": REVIEW_MIN_PRECISION})
        for name, value in metrics["pr_auc"].items():
            mlflow.log_metric(f"pr_auc_{name}", value)
        mlflow.log_metric("roc_auc_lightgbm", metrics["roc_auc_lightgbm"])
        for point, values in metrics["operating_points"].items():
            for k, v in values.items():
                mlflow.log_metric(f"{point}_{k}", v)
        for file in ("model.txt", "model_meta.json", "metrics.json"):
            mlflow.log_artifact(str(artifacts_dir / file))
        for fig in figures_dir.glob("*.png"):
            mlflow.log_artifact(str(fig), artifact_path="figures")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FraudGuard model")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    metrics = train(log_mlflow=not args.no_mlflow)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
