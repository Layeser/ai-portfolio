"""
k-Nearest Neighbors from scratch (NumPy).

No training — store the dataset; predict by majority vote among k closest points.
Dataset: Iris (classic).
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from mlscratch.utils.common import accuracy, set_seed, train_test_split


class KNearestNeighbors:
    def __init__(self, k: int = 5):
        self.k = k
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNearestNeighbors:
        self.X = X.astype(float)
        self.y = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.X is not None and self.y is not None
        # (n_query, n_train) pairwise L2 distances
        d2 = ((X[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
        nn = np.argsort(d2, axis=1)[:, : self.k]
        votes = self.y[nn]
        # majority vote per row
        out = []
        for row in votes:
            vals, counts = np.unique(row, return_counts=True)
            out.append(vals[np.argmax(counts)])
        return np.array(out)


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
