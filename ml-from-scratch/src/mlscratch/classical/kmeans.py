"""
K-Means clustering from scratch (NumPy).

Lloyd's algorithm: assign → update centroids → repeat, with k-means++ initialisation and restarts.
Dataset: 3 isotropic Gaussian blobs.
Explainer: docs/kmeans.md
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from mlscratch.utils.common import ensure_outputs, set_seed


def squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(n, k) squared Euclidean distances via ‖x‖² − 2 x·c + ‖c‖²: O(n·k) memory, no (n, k, d) array."""
    d2 = (X**2).sum(axis=1)[:, None] - 2 * X @ C.T + (C**2).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0)  # round-off can make a distance slightly negative


class KMeans:
    """Lloyd's k-means.

    init="k-means++" spreads the first centroids out (each new one drawn with probability proportional to its squared
    distance to the closest centroid already chosen); init="random" picks k data points. The best of `n_init` runs,
    by inertia, is kept. Stops when the centroids move less than `tol` × (mean feature variance).
    """

    def __init__(self, k: int = 3, n_iters: int = 100, seed: int = 0, init: str = "k-means++",
                 n_init: int = 10, tol: float = 1e-4):
        if init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'")
        self.k, self.n_iters, self.seed, self.init, self.n_init, self.tol = k, n_iters, seed, init, n_init, tol
        self.centroids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.init == "random":
            return X[rng.choice(len(X), size=self.k, replace=False)].copy()
        centroids = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = squared_distances(X, np.array(centroids)).min(axis=1)
            centroids.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(centroids)

    def _lloyd(self, X: np.ndarray, centroids: np.ndarray, tol: float):
        for it in range(1, self.n_iters + 1):
            d2 = squared_distances(X, centroids)
            labels = d2.argmin(axis=1)
            new_c = centroids.copy()
            for j in range(self.k):
                members = labels == j
                if members.any():
                    new_c[j] = X[members].mean(axis=0)
                else:  # empty cluster: move it to the point worst served by its current centroid
                    far = d2[np.arange(len(X)), labels].argmax()
                    new_c[j], labels[far] = X[far], j
            shift = ((new_c - centroids) ** 2).sum()
            centroids = new_c
            if shift <= tol:
                return centroids, it, True
        return centroids, self.n_iters, False

    def fit(self, X: np.ndarray) -> KMeans:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2-D array")
        if not 1 <= self.k <= len(X):
            raise ValueError(f"k must be between 1 and n = {len(X)}")
        rng = np.random.default_rng(self.seed)
        tol = self.tol * X.var(axis=0).mean()  # scale-free stopping rule
        best = None
        for _ in range(self.n_init):
            centroids, n_iter, converged = self._lloyd(X, self._init_centroids(X, rng), tol)
            labels = squared_distances(X, centroids).argmin(axis=1)
            inertia = float(((X - centroids[labels]) ** 2).sum())
            if best is None or inertia < best[0]:
                best = (inertia, centroids, labels, n_iter, converged)
        self.inertia_, self.centroids, self.labels_, self.n_iter_, converged = best
        if not converged:
            warnings.warn(f"k-means did not converge in {self.n_iters} iterations", stacklevel=2)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("call fit() first")
        return squared_distances(np.asarray(X, dtype=float), self.centroids).argmin(axis=1)


def main() -> None:
    set_seed(0)
    X, _ = make_blobs(n_samples=450, centers=3, cluster_std=0.8, random_state=0)
    km = KMeans(k=3).fit(X)

    out = ensure_outputs("kmeans")
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, s=15, cmap="viridis")
    plt.scatter(km.centroids[:, 0], km.centroids[:, 1], c="red", s=120, marker="x")
    plt.title("K-Means (from scratch)")
    plt.tight_layout()
    plt.savefig(out / "clusters.png", dpi=140)
    print(f"saved {out / 'clusters.png'}")
    print(f"inertia {km.inertia_:.1f} after {km.n_iter_} iterations — centroids:\n", km.centroids)


if __name__ == "__main__":
    main()
