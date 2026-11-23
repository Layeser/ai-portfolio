"""Train, evaluate and package a product classifier.

    python -m prodcat.train --model tfidf      # ~2 min on CPU
    python -m prodcat.train --model xlmr       # ~10 min on a laptop GPU
    python -m prodcat.train --report           # comparison figures from saved metrics

Evaluation protocol: stratified 10 % of the train set for model selection and threshold
choice; the official test set is used once, reported in full and without the test rows whose
title already appears in train (exact-duplicate leakage).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from prodcat.data import ROOT, TAXONOMY_PATH, leakage_report, load, train_valid_split  # noqa: E402
from prodcat.metrics import classification_report, coverage_curve, threshold_for_accuracy  # noqa: E402
from prodcat.models import TfidfModel, TransformerModel  # noqa: E402
from prodcat.taxonomy import Taxonomy  # noqa: E402

ARTIFACTS = Path(os.getenv("PRODCAT_ARTIFACTS_DIR", ROOT / "artifacts"))
REPORTS = ROOT / "reports"
AUTO_ACCURACY_TARGET = 0.95  # accuracy required on auto-categorised products


def evaluate(model, frame, tax: Taxonomy, threshold: float | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    probs = model.predict_proba(frame["text"])
    classes = np.array(model.classes_)
    pred = classes[probs.argmax(1)]
    report = classification_report(list(frame["category_id"]), list(pred), tax)
    confidence, correct = probs.max(1), pred == frame["category_id"].to_numpy()
    if threshold is not None:
        auto = confidence >= threshold
        report["auto"] = {"threshold": round(threshold, 4), "coverage": round(float(auto.mean()), 4),
                          "accuracy_on_auto": round(float(correct[auto].mean()), 4) if auto.any() else None}
    return report, confidence, correct


def train(model_name: str, artifacts: Path = ARTIFACTS, data_dir: Path | None = None, log_mlflow: bool = True,
          **model_kwargs) -> dict:
    kwargs = {"data_dir": data_dir} if data_dir else {}
    tax = Taxonomy.load(TAXONOMY_PATH) if TAXONOMY_PATH.exists() else None
    train_df, test_df = load("train", **kwargs), load("test", **kwargs)
    if tax is None or not set(train_df["category_id"]) <= set(tax.parent):
        tax = Taxonomy.from_paths(list(train_df["category_path"]) + list(test_df["category_path"]))
    fit_df, valid_df = train_valid_split(train_df)
    leak, dedup_mask = leakage_report(train_df, test_df)

    if model_name == "tfidf":
        model = TfidfModel(**model_kwargs).fit(fit_df["text"], fit_df["category_id"])
    else:
        model = TransformerModel(**model_kwargs).fit(fit_df["text"], fit_df["category_id"],
                                                     valid_df["text"], valid_df["category_id"])

    valid_report, v_conf, v_ok = evaluate(model, valid_df, tax)
    threshold = threshold_for_accuracy(v_conf, v_ok, AUTO_ACCURACY_TARGET)
    test_report, t_conf, t_ok = evaluate(model, test_df, tax, threshold)
    dedup_report, _, _ = evaluate(model, test_df[dedup_mask.to_numpy()], tax, threshold)

    metrics = {"model": model_name, "valid": valid_report, "test": test_report, "test_dedup": dedup_report,
               "leakage": leak, "auto_accuracy_target": AUTO_ACCURACY_TARGET,
               "rows": {"fit": len(fit_df), "valid": len(valid_df), "test": len(test_df)}}
    if isinstance(model, TransformerModel):
        metrics["history"] = model.history
    out = artifacts / model_name
    model.save(out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "threshold.json").write_text(json.dumps({"auto_threshold": threshold}))
    np.savez(out / "test_confidence.npz", confidence=t_conf, correct=t_ok)
    if log_mlflow:
        _log_mlflow(artifacts, model_name, metrics)
    return metrics


def _log_mlflow(artifacts: Path, name: str, metrics: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{artifacts / 'mlflow.db'}")
    mlflow.set_experiment("prodcat")
    with mlflow.start_run(run_name=name):
        for split in ("valid", "test", "test_dedup"):
            for k in ("accuracy", "f1_macro", "hF1"):
                mlflow.log_metric(f"{split}_{k}", metrics[split][k])
        mlflow.log_metric("auto_coverage", metrics["test"]["auto"]["coverage"])
        mlflow.log_metric("auto_accuracy", metrics["test"]["auto"]["accuracy_on_auto"])
        mlflow.log_artifact(str(artifacts / name / "metrics.json"))


def report(artifacts: Path = ARTIFACTS) -> None:
    REPORTS.joinpath("figures").mkdir(parents=True, exist_ok=True)
    names = [n for n in ("tfidf", "xlmr") if (artifacts / n / "metrics.json").exists()]
    labels = {"tfidf": "TF-IDF + linear SVM (calibrated)", "xlmr": "XLM-RoBERTa fine-tuned"}
    colors = {"tfidf": "#E0A33B", "xlmr": "#167D9A"}

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for n in names:
        z = np.load(artifacts / n / "test_confidence.npz")
        _, coverage, acc = coverage_curve(z["confidence"], z["correct"])
        ax.plot(coverage, acc, label=labels[n], color=colors[n], lw=2)
        m = json.loads((artifacts / n / "metrics.json").read_text())["test"]["auto"]
        ax.scatter(m["coverage"], m["accuracy_on_auto"], color=colors[n], zorder=5)
        offset = {"tfidf": (8, 6), "xlmr": (-62, -18)}[n]  # keep the two labels apart
        ax.annotate(f"{m['coverage']:.0%} auto", (m["coverage"], m["accuracy_on_auto"]),
                    textcoords="offset points", xytext=offset, fontsize=8.5, color=colors[n])
    ax.axhline(AUTO_ACCURACY_TARGET, ls="--", lw=1, color="#52616B", label="95 % accuracy target")
    ax.set(xlabel="share of products categorised automatically", ylabel="accuracy on those products",
           title="Automation vs accuracy (test set)", ylim=(0.6, 1.005))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "coverage_accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.38
    for i, n in enumerate(names):
        levels = json.loads((artifacts / n / "metrics.json").read_text())["test"]["level_accuracy"]
        x = np.arange(len(levels)) + (i - (len(names) - 1) / 2) * width
        bars = ax.bar(x, list(levels.values()), width, label=labels[n], color=colors[n])
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5)
    ax.set_xticks(np.arange(4), ["level 1", "level 2", "level 3", "level 4"])
    ax.set(ylim=(0.7, 1.07), ylabel="accuracy of the path prefix", title="Errors stay close in the tree")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "level_accuracy.png", dpi=160)
    plt.close(fig)

    summary = {n: json.loads((artifacts / n / "metrics.json").read_text()) for n in names}
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tfidf", "xlmr"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.model:
        kwargs = {"epochs": args.epochs, "lr": args.lr} if args.model == "xlmr" else {}
        metrics = train(args.model, **kwargs)
        print(json.dumps({k: metrics[k] for k in ("test", "test_dedup", "leakage")}, indent=2))
    if args.report or not args.model:
        report()


if __name__ == "__main__":
    main()
