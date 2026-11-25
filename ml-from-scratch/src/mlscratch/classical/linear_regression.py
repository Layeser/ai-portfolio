"""
Linear Regression from scratch (NumPy).

Model:  y_hat = X @ w + b
Loss:   MSE (+ optional L2 penalty alpha·‖w‖², i.e. ridge)
Update: gradient descent, or the exact least-squares solution

Dataset: synthetic y = 3x + 2 + noise  (classic teaching toy)
Explainer: docs/linear_regression.md
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mlscratch.utils.common import ensure_outputs, mse, set_seed, train_test_split


def _check_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"expected X (n, d) and y (n,), got {X.shape} and {y.shape}")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X or y contains NaN or infinite values")
    return X, y


class LinearRegression:
    """Least squares, optionally ridge-penalised (`alpha` > 0). The intercept `b` is never penalised."""

    def __init__(self, lr: float = 0.05, n_iters: int = 1000, alpha: float = 0.0, tol: float = 0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha
        self.tol = tol  # stop gradient descent when the loss improves by less than tol
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.losses: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """Batch gradient descent on MSE + alpha·‖w‖²."""
        X, y = _check_xy(X, y)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.losses = []
        for _ in range(self.n_iters):
            y_hat = X @ self.w + self.b
            err = y_hat - y
            self.w -= self.lr * ((X.T @ err) / n + self.alpha * self.w)
            self.b -= self.lr * np.mean(err)
            loss = float(np.mean(err**2) + self.alpha * self.w @ self.w)
            if not np.isfinite(loss):
                raise FloatingPointError("gradient descent diverged: lower lr or standardise the features")
            self.losses.append(loss)
            if self.tol and len(self.losses) > 1 and self.losses[-2] - loss < self.tol:
                break
        return self

    def fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """Exact solution. Ridge solves (Xcᵀ Xc + n·alpha·I) w = Xcᵀ yc on centred data, so b is not penalised;
        plain least squares uses lstsq, which never forms Xᵀ X."""
        X, y = _check_xy(X, y)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        if self.alpha > 0:
            A = Xc.T @ Xc + len(X) * self.alpha * np.eye(X.shape[1])
            self.w = np.linalg.solve(A, Xc.T @ yc)
        else:
            self.w = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        self.b = float(y_mean - x_mean @ self.w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("call fit() or fit_closed_form() first")
        return np.asarray(X, dtype=float) @ self.w + self.b


def main() -> None:
    set_seed(0)
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    y = 3.0 * X[:, 0] + 2.0 + 0.2 * np.random.randn(len(X))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    model = LinearRegression(lr=0.1, n_iters=500).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"GD  MSE={mse(y_te, pred):.4f}  w≈{model.w[0]:.3f}  b≈{model.b:.3f}")

    closed = LinearRegression().fit_closed_form(X_tr, y_tr)
    print(f"LS  MSE={mse(y_te, closed.predict(X_te)):.4f}  w≈{closed.w[0]:.3f}  b≈{closed.b:.3f}")

    out = ensure_outputs("linear_regression")
    plt.figure(figsize=(5, 3.5))
    plt.scatter(X_te[:, 0], y_te, s=12, alpha=0.6, label="test")
    xs = np.linspace(-1, 1, 100).reshape(-1, 1)
    plt.plot(xs[:, 0], model.predict(xs), "r-", label="fit")
    plt.legend()
    plt.title("Linear Regression (from scratch)")
    plt.tight_layout()
    plt.savefig(out / "fit.png", dpi=140)
    print(f"saved {out / 'fit.png'}")


if __name__ == "__main__":
    main()
