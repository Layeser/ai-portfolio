"""
Gaussian Naive Bayes from scratch (NumPy).

Assumes features independent given class; each feature ~ N(mu_{c,j}, sigma_{c,j}^2), here: Gaussian distribution.
Dataset: Iris.
Explainer: docs/naive_bayes.md
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from mlscratch.utils.common import accuracy, set_seed, train_test_split


class GaussianNaiveBayes:
    """Per class c and feature j, a Gaussian N(mean_[c, j], var_[c, j]); class log-priors from frequencies.

    var_smoothing adds var_smoothing × (largest feature variance) to every variance, as scikit-learn does, so a
    feature that is constant within a class does not produce a zero variance.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianNaiveBayes:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        eps = self.var_smoothing * X.var(axis=0).max()
        self.mean_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])  # (C, d)
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_]) + eps  # (C, d)
        self.log_prior_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))  # (C,)
        return self

    def joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """(n, C) log p(c) + Σ_j log N(x_j | mean_[c, j], var_[c, j]), all classes at once."""
        X = np.asarray(X, dtype=float)
        ll = -0.5 * (np.log(2 * np.pi * self.var_)[None] + (X[:, None, :] - self.mean_[None]) ** 2 / self.var_[None])
        return self.log_prior_[None] + ll.sum(axis=2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)  # log-sum-exp trick: exp never overflows
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]


def main() -> None:
    set_seed(0)
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3)
    model = GaussianNaiveBayes().fit(X_tr, y_tr)
    print(f"test accuracy={accuracy(y_te, model.predict(X_te)):.3f}")


if __name__ == "__main__":
    main()
