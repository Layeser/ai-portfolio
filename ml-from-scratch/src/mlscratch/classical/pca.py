"""
Principal Component Analysis from scratch (NumPy).

1) Center X
2) Covariance C = X^T X / (n-1)  (or, equivalently, an SVD of the centred data)
3) Eigen-decomposition; take top-k eigenvectors
Dataset: Iris → 2D projection.
Explainer: docs/pca.md
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

from mlscratch.utils.common import ensure_outputs, set_seed


def _check_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError(f"expected a 2-D array with at least 2 rows, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    return X


class PCA:
    """Principal components by eigendecomposition of the covariance ("eigh") or SVD of the centred data ("svd").

    `components_` has shape (d, k): one component per column. Signs are made deterministic: in every component,
    the loading with the largest absolute value is positive.
    """

    def __init__(self, n_components: int = 2, solver: str = "eigh", whiten: bool = False):
        if solver not in ("eigh", "svd"):
            raise ValueError("solver must be 'eigh' or 'svd'")
        self.n_components = n_components
        self.solver = solver
        self.whiten = whiten
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> PCA:
        X = _check_array(X)
        n, d = X.shape
        if not 1 <= self.n_components <= min(n, d):
            raise ValueError(f"n_components must be between 1 and min(n, d) = {min(n, d)}")
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        if self.solver == "eigh":
            cov = (Xc.T @ Xc) / (n - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)  # symmetric matrix: real eigenvalues, orthonormal eigenvectors
            order = np.argsort(eigvals)[::-1]  # eigh returns ascending order
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        else:
            _, s, vt = np.linalg.svd(Xc, full_matrices=False)  # never forms X^T X
            eigvals, eigvecs = s**2 / (n - 1), vt.T
        eigvals = np.clip(eigvals, 0.0, None)  # round-off can leave tiny negative eigenvalues
        # deterministic sign: v and -v are both valid; make the largest loading of each component positive
        signs = np.sign(eigvecs[np.argmax(np.abs(eigvecs), axis=0), np.arange(eigvecs.shape[1])])
        eigvecs = eigvecs * np.where(signs == 0, 1.0, signs)
        k = self.n_components
        self.components_ = eigvecs[:, :k]
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = eigvals[:k] / eigvals.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("call fit() first")
        Z = (np.asarray(X, dtype=float) - self.mean_) @ self.components_
        if self.whiten:
            Z = Z / np.sqrt(self.explained_variance_)  # unit variance along every component
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map projections back to the original space (exact only if all components are kept)."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("call fit() first")
        Z = np.asarray(Z, dtype=float)
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_.T + self.mean_


def main() -> None:
    set_seed(0)
    data = load_iris()
    print("data shape:", data.data.shape)
    pca = PCA(2).fit(data.data)
    Z = pca.transform(data.data)
    print("explained variance ratio:", pca.explained_variance_ratio_)
    print("reconstruction MSE with 2 components:", float(np.mean((pca.inverse_transform(Z) - data.data) ** 2)))

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
