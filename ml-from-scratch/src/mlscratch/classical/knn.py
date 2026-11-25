"""
k-Nearest Neighbors from scratch (NumPy).

No training — store the dataset; predict by majority vote among k closest points.
Dataset: Iris (classic).
Explainer: docs/knn.md
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from mlscratch.utils.common import accuracy, set_seed, train_test_split


def squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(len(A), len(B)) squared Euclidean distances via ‖a‖² − 2 a·b + ‖b‖²: no (n_query, n_train, d) array."""
    d2 = (A**2).sum(axis=1)[:, None] - 2 * A @ B.T + (B**2).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0)


class KNearestNeighbors:
    """k-NN classifier. weights="uniform": majority vote; weights="distance": votes weighted by 1 / distance
    (an exact match wins outright). Ties between classes go to the smallest label, as in scikit-learn.
    Queries are processed in chunks of `batch_size` rows to bound memory."""

    def __init__(self, k: int = 5, weights: str = "uniform", batch_size: int = 1024):
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k, self.weights, self.batch_size = k, weights, batch_size
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNearestNeighbors:
        self.X = np.asarray(X, dtype=float)
        self.classes_, self.y = np.unique(np.asarray(y), return_inverse=True)  # labels encoded 0..C-1
        if not 1 <= self.k <= len(self.X):
            raise ValueError(f"k must be between 1 and n = {len(self.X)}")
        return self

    def _predict_batch(self, Q: np.ndarray) -> np.ndarray:
        d2 = squared_distances(Q, self.X)
        nn = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]  # k smallest, unordered: O(n) per row
        labels = self.y[nn]
        if self.weights == "uniform":
            w = np.ones_like(labels, dtype=float)
        else:
            dist = np.sqrt(np.take_along_axis(d2, nn, axis=1))
            exact = dist == 0
            w = np.where(exact.any(axis=1, keepdims=True), exact.astype(float), 1.0 / np.where(exact, 1.0, dist))
        votes = np.zeros((len(Q), len(self.classes_)))
        np.add.at(votes, (np.arange(len(Q))[:, None], labels), w)
        return votes.argmax(axis=1)  # argmax returns the first maximum: smallest label wins ties

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X is None:
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=float)
        out = [self._predict_batch(X[i : i + self.batch_size]) for i in range(0, len(X), self.batch_size)]
        return self.classes_[np.concatenate(out)]


def main() -> None:
    set_seed(0)
    data = load_iris()
    X, y = data.data, data.target
    # standardize
    mu, sigma = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sigma
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)
    for k in [1, 3, 5, 11]:
        pred = KNearestNeighbors(k=k).fit(X_tr, y_tr).predict(X_te)
        print(f"k={k:2d}  acc={accuracy(y_te, pred):.3f}")


if __name__ == "__main__":
    main()
