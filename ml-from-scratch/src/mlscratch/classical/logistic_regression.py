"""
Binary Logistic Regression from scratch (NumPy).

Model:  p = sigmoid(X @ w + b)
Loss:   binary cross-entropy
Dataset: sklearn make_classification (labels only; model is ours)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

from mlscratch.utils.common import accuracy, ensure_outputs, set_seed, train_test_split


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    def __init__(self, lr: float = 0.1, n_iters: int = 2000):
        self.lr = lr
        self.n_iters = n_iters
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iters):
            p = sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (X.T @ err) / n
            self.b -= self.lr * np.mean(err)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


def main() -> None:
    set_seed(0)
    X, y = make_classification(
        n_samples=600, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=1.2, random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    model = LogisticRegression().fit(X_tr, y_tr)
    acc = accuracy(y_te, model.predict(X_te))
    print(f"test accuracy={acc:.3f}")

    out = ensure_outputs("logistic_regression")
    # decision boundary
    xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    ys = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
    xx, yy = np.meshgrid(xs, ys)
    zz = model.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.7)
    plt.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="RdBu", edgecolors="k", s=20)
    plt.title(f"Logistic Regression  acc={acc:.2f}")
    plt.tight_layout()
    plt.savefig(out / "boundary.png", dpi=140)
    print(f"saved {out / 'boundary.png'}")


if __name__ == "__main__":
    main()
