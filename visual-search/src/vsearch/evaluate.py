"""Train every encoder, index CIFAR-10 train (50k gallery) and query it with CIFAR-10 test (10k).

    python -m vsearch.evaluate                     # all encoders (GPU recommended: ~45 min)
    python -m vsearch.evaluate --encoders pixels_pca autoencoder --ae-epochs 2   # quick run

Metrics (a retrieved image is relevant when it has the query's class; labels are only used here):
* P@10       share of the 10 nearest gallery images with the query's class
* mAP@10     average precision over the top 10 (rewards relevant results ranked first)
* kNN acc    20-nearest-neighbour vote, i.e. "label the query by its search results"
* linear     logistic regression on the frozen embeddings (standard representation-learning probe)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vsearch.data import CLASSES, ROOT, load_cifar10  # noqa: E402
from vsearch.encoders import ENCODERS, ConvAutoencoder, SimCLR  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"


def normalise(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_index(gallery: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(gallery.shape[1])  # cosine similarity on L2-normalised vectors
    index.add(normalise(gallery))
    return index


def retrieval_metrics(
    neighbours: np.ndarray, gallery_labels: np.ndarray, query_labels: np.ndarray, k: int = 10
) -> dict:
    relevant = gallery_labels[neighbours[:, :k]] == query_labels[:, None]
    precision_at = np.cumsum(relevant, axis=1) / np.arange(1, k + 1)
    ap = (precision_at * relevant).sum(1) / np.maximum(relevant.sum(1), 1)
    votes = gallery_labels[neighbours[:, :20]]
    knn = np.array([np.bincount(v, minlength=10).argmax() for v in votes])
    return {
        f"P@{k}": round(float(relevant.mean()), 4),
        f"mAP@{k}": round(float(ap.mean()), 4),
        "kNN_accuracy": round(float((knn == query_labels).mean()), 4),
    }


def linear_probe(train_x, train_y, test_x, test_y) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(train_x), train_y)
    return round(float(clf.score(scaler.transform(test_x), test_y)), 4)


def fit_encoder(name: str, train_images: np.ndarray, args) -> object:
    cls = ENCODERS[name]
    if cls is ConvAutoencoder:
        return cls().fit(train_images, epochs=args.ae_epochs)
    if cls is SimCLR:
        return cls().fit(train_images, epochs=args.simclr_epochs)
    return cls().fit(train_images)


def plot_examples(name: str, test_images, train_images, neighbours, test_labels, train_labels, path: Path):
    rng = np.random.default_rng(1)
    queries = rng.choice(len(test_images), 6, replace=False)
    fig, axes = plt.subplots(6, 7, figsize=(7, 6.4))
    for row, q in enumerate(queries):
        axes[row, 0].imshow(test_images[q])
        axes[row, 0].set_title(f"query: {CLASSES[test_labels[q]]}", fontsize=6.5)
        for col, g in enumerate(neighbours[q, :6], start=1):
            ok = train_labels[g] == test_labels[q]
            axes[row, col].imshow(train_images[g])
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#167D9A" if ok else "#C0392B")
                spine.set_linewidth(2)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name}: query (left) and its 6 nearest gallery images (blue = same class)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(results: dict, path: Path) -> None:
    names = list(results)
    metrics = ["P@10", "kNN_accuracy", "linear_probe"]
    labels = {"P@10": "search P@10", "kNN_accuracy": "kNN accuracy", "linear_probe": "linear probe"}
    colors = ["#9AA5B1", "#E0A33B", "#167D9A"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (m, c) in enumerate(zip(metrics, colors, strict=True)):
        bars = ax.bar(x + (i - 1) * 0.27, [results[n][m] for n in names], 0.27, label=labels[m], color=c)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, [n.replace("_", " + ") if n == "pixels_pca" else n for n in names])
    ax.set(ylim=(0, 1), ylabel="CIFAR-10 test", title="Which representation finds similar images?")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--simclr-epochs", type=int, default=60)
    args = parser.parse_args()

    train_images, train_labels, test_images, test_labels = load_cifar10()
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results_path = ROOT / "reports" / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for name in args.encoders:
        start = time.time()
        encoder = fit_encoder(name, train_images, args)
        fit_minutes = (time.time() - start) / 60
        gallery, queries = encoder.embed(train_images), encoder.embed(test_images)
        index = build_index(gallery)
        t = time.time()
        _, neighbours = index.search(normalise(queries), 20)
        search_ms = (time.time() - t) / len(queries) * 1000
        results[name] = {
            **retrieval_metrics(neighbours, train_labels, test_labels),
            "linear_probe": linear_probe(gallery, train_labels, queries, test_labels),
            "dim": int(gallery.shape[1]),
            "fit_minutes": round(fit_minutes, 1),
            "search_ms_per_query": round(search_ms, 3),
        }
        print(name, results[name], flush=True)
        np.save(ARTIFACTS / f"gallery_{name}.npy", gallery.astype(np.float32))
        if isinstance(encoder, torch.nn.Module):
            torch.save(encoder.state_dict(), ARTIFACTS / f"{name}.pt")
        plot_examples(
            name, test_images, train_images, neighbours, test_labels, train_labels, FIGURES / f"examples_{name}.png"
        )
        results_path.write_text(json.dumps(results, indent=2))
    plot_summary({n: results[n] for n in ENCODERS if n in results}, FIGURES / "comparison.png")


if __name__ == "__main__":
    main()
