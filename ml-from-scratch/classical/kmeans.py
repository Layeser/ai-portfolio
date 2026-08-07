"""
K-Means clustering from scratch (NumPy).

Lloyd's algorithm: assign → update centroids → repeat.
Dataset: 3 isotropic Gaussian blobs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import ensure_outputs, set_seed


class KMeans:
    def __init__(self, k: int = 3, n_iters: int = 100, seed: int = 0):
        self.k = k
        self.n_iters = n_iters
        self.seed = seed
        self.centroids: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.seed)
        # init: random points from data
        idx = rng.choice(len(X), size=self.k, replace=False)
        self.centroids = X[idx].copy()
        for _ in range(self.n_iters):
            labels = self.predict(X)
            new_c = np.array(
                [
                    X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                    for j in range(self.k)
                ]
            )
            if np.allclose(new_c, self.centroids):
                break
            self.centroids = new_c
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.centroids is not None
        d2 = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1)


def main() -> None:
    set_seed(0)
    X, y_true = make_blobs(n_samples=450, centers=3, cluster_std=0.8, random_state=0)
    km = KMeans(k=3).fit(X)
    labels = km.predict(X)

    out = ensure_outputs("kmeans")
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=15, cmap="viridis")
    plt.scatter(km.centroids[:, 0], km.centroids[:, 1], c="red", s=120, marker="x")
    plt.title("K-Means (from scratch)")
    plt.tight_layout()
    plt.savefig(out / "clusters.png", dpi=140)
    print(f"saved {out / 'clusters.png'}")
    # crude purity vs true labels (best permutation not needed for demo)
    print("done — centroids:\n", km.centroids)


if __name__ == "__main__":
    main()
