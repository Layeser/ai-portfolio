"""
Principal Component Analysis from scratch (NumPy).

1) Center X
2) Covariance C = X^T X / (n-1)
3) Eigen-decomposition; take top-k eigenvectors
Dataset: Iris → 2D projection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import ensure_outputs, set_seed


class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / (len(X) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending order
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        self.components_ = eigvecs[:, : self.n_components]
        total = eigvals.sum()
        self.explained_variance_ratio_ = eigvals[: self.n_components] / total
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def main() -> None:
    set_seed(0)
    data = load_iris()
    Z = PCA(2).fit_transform(data.data)
    pca = PCA(2).fit(data.data)
    print("explained variance ratio:", pca.explained_variance_ratio_)

    out = ensure_outputs("pca")
    plt.figure(figsize=(5, 4))
    plt.scatter(Z[:, 0], Z[:, 1], c=data.target, cmap="viridis", s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA on Iris (from scratch)")
    plt.tight_layout()
    plt.savefig(out / "iris_pca.png", dpi=140)
    print(f"saved {out / 'iris_pca.png'}")


if __name__ == "__main__":
    main()
