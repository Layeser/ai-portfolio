"""Project paths and shared constants.

Every path can be overridden with an environment variable, so tests and the
Docker image can point the code at another artifacts directory.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("FRAUDGUARD_DATA_DIR", ROOT / "data"))
PRETRAINED_DIR = ROOT / "models" / "pretrained"  # shipped with the repository: no training needed


def _default_artifacts() -> Path:
    """Your own training run (artifacts/) if it exists, otherwise the pretrained model in the repo."""
    local = ROOT / "artifacts"
    return local if (local / "model.txt").exists() else PRETRAINED_DIR


ARTIFACTS_DIR = Path(os.getenv("FRAUDGUARD_ARTIFACTS_DIR", _default_artifacts()))
REPORTS_DIR = Path(os.getenv("FRAUDGUARD_REPORTS_DIR", ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42

# Merchant categories and their channel. The order defines the integer code fed
# to LightGBM, so it must never be reordered once a model is trained.
MERCHANT_CHANNEL = {
    "grocery": "pos",
    "restaurant": "pos",
    "transport": "pos",
    "fuel": "pos",
    "retail": "pos",
    "online_retail": "online",
    "streaming": "online",
    "travel": "online",
    "electronics": "online",
    "gift_cards": "online",
    "crypto": "online",
}
MERCHANT_CATEGORIES = list(MERCHANT_CHANNEL)
