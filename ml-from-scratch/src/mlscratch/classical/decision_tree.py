"""
Decision Tree (CART-style, Gini) from scratch (NumPy).

Binary splits on a single feature threshold; recursion until max_depth / pure leaf.
Dataset: Iris (multi-class).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris

from mlscratch.utils.common import accuracy, set_seed, train_test_split


def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p**2))


@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None
    value: int | None = None  # leaf class


class DecisionTree:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Node | None = None

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain, best_feat, best_thr = 0.0, None, None
        parent = gini(y)
        n, d = X.shape
        for j in range(d):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                left = y[X[:, j] <= t]
                right = y[X[:, j] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent - (
                    len(left) / n * gini(left) + len(right) / n * gini(right)
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, j, float(t)
        return best_feat, best_thr, best_gain

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            vals, counts = np.unique(y, return_counts=True)
            return Node(value=int(vals[np.argmax(counts)]))

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            vals, counts = np.unique(y, return_counts=True)
            return Node(value=int(vals[np.argmax(counts)]))

        left_mask = X[:, feat] <= thr
        return Node(
            feature=feat,
            threshold=thr,
            left=self._grow(X[left_mask], y[left_mask], depth + 1),
            right=self._grow(X[~left_mask], y[~left_mask], depth + 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTree:
        self.root = self._grow(X, y, 0)
        return self

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        if node.value is not None:
            return node.value
        assert node.feature is not None and node.threshold is not None
        branch = node.left if x[node.feature] <= node.threshold else node.right
        assert branch is not None
        return self._predict_one(x, branch)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        return np.array([self._predict_one(x, self.root) for x in X])


def main() -> None:
    set_seed(0)
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3)
    model = DecisionTree(max_depth=4).fit(X_tr, y_tr)
    print(f"test accuracy={accuracy(y_te, model.predict(X_te)):.3f}")


if __name__ == "__main__":
    main()
