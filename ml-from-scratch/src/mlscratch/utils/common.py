"""Tiny shared utilities — keep dependencies light."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # project root (src/mlscratch/utils/common.py)
OUTPUTS = ROOT / "outputs"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_test_split(
    *arrays: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> list[np.ndarray]:
    """Simple index-based split (no sklearn required for core logic)."""
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    out: list[np.ndarray] = []
    for a in arrays:
        out.append(a[train_idx])
        out.append(a[test_idx])
    return out


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def ensure_outputs(name: str) -> Path:
    path = OUTPUTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path
