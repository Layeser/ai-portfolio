"""
Decision Tree (CART-style, Gini or entropy) from scratch (NumPy).

Binary splits on a single feature threshold; recursion until max_depth / pure leaf.
Dataset: Iris (multi-class).
Explainer: docs/decision_tree.md
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


def _impurity(counts: np.ndarray, criterion: str) -> np.ndarray:
    """Impurity of many candidate nodes at once; counts has shape (..., n_classes)."""
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(total, 1)
    if criterion == "gini":
        return 1.0 - (p**2).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=-1)


@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None
    value: int | None = None  # leaf class (index into classes_)
    proba: np.ndarray | None = None  # class frequencies at the node


class DecisionTree:
    """CART classifier. For each feature, the samples are sorted once and every threshold is scored from cumulative
    class counts, so a node costs O(d · n log n) instead of O(d · n²). Thresholds are midpoints between consecutive
    distinct values, as in scikit-learn."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 criterion: str = "gini"):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root: Node | None = None

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        onehot = np.eye(self.n_classes_)[y]  # (n, C)
        parent = float(_impurity(onehot.sum(axis=0), self.criterion))
        best_gain, best_feat, best_thr = 0.0, None, None
        m = self.min_samples_leaf
        for j in range(d):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            left = np.cumsum(onehot[order], axis=0)[:-1]  # class counts left of each cut, (n-1, C)
            right = left[-1] + onehot[order][-1] - left
            valid = xs[1:] > xs[:-1]  # cut only between distinct values
            n_left = np.arange(1, n)
            valid &= (n_left >= m) & (n - n_left >= m)
            if not valid.any():
                continue
            child = (n_left * _impurity(left, self.criterion) + (n - n_left) * _impurity(right, self.criterion)) / n
            gain = np.where(valid, parent - child, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_gain, best_feat, best_thr = float(gain[i]), j, float((xs[i] + xs[i + 1]) / 2)
        return best_feat, best_thr, best_gain

    def _leaf(self, y: np.ndarray) -> Node:
        counts = np.bincount(y, minlength=self.n_classes_)
        return Node(value=int(counts.argmax()), proba=counts / counts.sum())

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return self._leaf(y)
        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            return self._leaf(y)
        self.importances_[feat] += gain * len(y)  # impurity decrease, weighted by the node size
        left_mask = X[:, feat] <= thr
        node = self._leaf(y)
        node.feature, node.threshold, node.value = feat, thr, None
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTree:
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(np.asarray(y), return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.importances_ = np.zeros(X.shape[1])
        self.root = self._grow(X, y_enc, 0)
        total = self.importances_.sum()
        self.feature_importances_ = self.importances_ / total if total > 0 else self.importances_
        return self

    def _leaf_for(self, x: np.ndarray) -> Node:
        node = self.root
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("call fit() first")
        return np.array([self._leaf_for(x).proba for x in np.asarray(X, dtype=float)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("call fit() first")
        return self.classes_[[self._leaf_for(x).value for x in np.asarray(X, dtype=float)]]

    def depth(self) -> int:
        def walk(node: Node) -> int:
            return 0 if node.value is not None else 1 + max(walk(node.left), walk(node.right))

        return walk(self.root)


def main() -> None:
    set_seed(0)
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3)
    model = DecisionTree(max_depth=4).fit(X_tr, y_tr)
    print(f"test accuracy={accuracy(y_te, model.predict(X_te)):.3f}")
    print("feature importances:", np.round(model.feature_importances_, 3))


if __name__ == "__main__":
    main()
