"""
Binary Logistic Regression from scratch (NumPy).

Model:  p = sigmoid(X @ w + b)
Loss:   binary cross-entropy (+ optional L2 penalty, optional class weights)
Dataset: sklearn make_classification (labels only; model is ours)
Explainer: docs/logistic_regression.md
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
    """Binary logistic regression by batch gradient descent.

    alpha: L2 penalty on w (not on b). class_weight="balanced" weighs each sample by n / (2·n_class), so both classes
    contribute equally to the loss. Stops early when the loss improves by less than `tol`.
    """

    def __init__(self, lr: float = 0.1, n_iters: int = 2000, alpha: float = 0.0, class_weight: str | None = None,
                 tol: float = 0.0):
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")
        self.lr, self.n_iters, self.alpha, self.class_weight, self.tol = lr, n_iters, alpha, class_weight, tol
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.losses: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("binary labels 0/1 expected")
        n, d = X.shape
        if self.class_weight == "balanced":
            counts = np.bincount(y.astype(int), minlength=2)
            sw = (n / (2 * counts))[y.astype(int)]
        else:
            sw = np.ones(n)
        self.w = np.zeros(d)
        self.b = 0.0
        self.losses = []
        for _ in range(self.n_iters):
            p = sigmoid(X @ self.w + self.b)
            err = sw * (p - y)  # gradient of the weighted cross-entropy with respect to the logit
            self.w -= self.lr * ((X.T @ err) / n + self.alpha * self.w)
            self.b -= self.lr * np.mean(err)
            eps = 1e-12
            ce = -np.mean(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            self.losses.append(float(ce + 0.5 * self.alpha * self.w @ self.w))
            if self.tol and len(self.losses) > 1 and self.losses[-2] - self.losses[-1] < self.tol:
                break
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("call fit() first")
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


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
    plt.contourf(xx, yy, zz, levels=5, cmap="RdBu", alpha=0.7)
    plt.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="RdBu", edgecolors="k", s=20)
    plt.title(f"Logistic Regression  acc={acc:.2f}")
    plt.tight_layout()
    plt.savefig(out / "boundary.png", dpi=140)
    print(f"saved {out / 'boundary.png'}")


if __name__ == "__main__":
    main()
