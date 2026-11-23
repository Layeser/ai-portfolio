"""
Gaussian Naive Bayes from scratch (NumPy).

Assumes features independent given class; each feature ~ N(mu_{c,j}, sigma_{c,j}^2).
Dataset: Iris.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from mlscratch.utils.common import accuracy, set_seed, train_test_split


class GaussianNaiveBayes:
    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianNaiveBayes:
        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.mean_ = {}
        self.var_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / len(X)
            self.mean_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0) + 1e-9
        return self

    def _log_gaussian(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            best_c, best_score = None, -np.inf
            for c in self.classes_:
                score = np.log(self.priors_[c]) + self._log_gaussian(
                    x, self.mean_[c], self.var_[c]
                ).sum()
                if score > best_score:
                    best_score, best_c = score, c
            preds.append(best_c)
        return np.array(preds)


def main() -> None:
    set_seed(0)
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3)
    model = GaussianNaiveBayes().fit(X_tr, y_tr)
    print(f"test accuracy={accuracy(y_te, model.predict(X_te)):.3f}")


if __name__ == "__main__":
    main()
