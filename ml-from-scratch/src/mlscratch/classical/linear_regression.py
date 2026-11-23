"""
Linear Regression from scratch (NumPy).

Model:  y_hat = X @ w + b
Loss:   MSE
Update: gradient descent (also closed-form via normal equations)

Dataset: synthetic y = 3x + 2 + noise  (classic teaching toy)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mlscratch.utils.common import ensure_outputs, mse, set_seed, train_test_split


class LinearRegression:
    def __init__(self, lr: float = 0.05, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.losses: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iters):
            y_hat = X @ self.w + self.b
            err = y_hat - y
            self.w -= self.lr * (X.T @ err) / n
            self.b -= self.lr * np.mean(err)
            self.losses.append(float(np.mean(err**2)))
        return self

    def fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """Normal equations with bias column: w = (X^T X)^-1 X^T y."""
        Xb = np.c_[np.ones(len(X)), X]
        theta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
        self.b = float(theta[0])
        self.w = theta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return X @ self.w + self.b


def main() -> None:
    set_seed(0)
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    y = 3.0 * X[:, 0] + 2.0 + 0.2 * np.random.randn(len(X))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    model = LinearRegression(lr=0.1, n_iters=500).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"GD  MSE={mse(y_te, pred):.4f}  w≈{model.w[0]:.3f}  b≈{model.b:.3f}")

    closed = LinearRegression().fit_closed_form(X_tr, y_tr)
    print(f"NE  MSE={mse(y_te, closed.predict(X_te)):.4f}  w≈{closed.w[0]:.3f}  b≈{closed.b:.3f}")

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
