"""CIFAR-10 as uint8 arrays (downloaded by torchvision on first use)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("VSEARCH_DATA_DIR", ROOT / "data"))
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(train_images, train_labels, test_images, test_labels); images are (N, 32, 32, 3) uint8."""
    import torchvision

    train = torchvision.datasets.CIFAR10(str(data_dir), train=True, download=True)
    test = torchvision.datasets.CIFAR10(str(data_dir), train=False, download=True)
    return train.data, np.array(train.targets), test.data, np.array(test.targets)
