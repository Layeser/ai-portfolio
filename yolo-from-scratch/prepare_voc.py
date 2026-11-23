#!/usr/bin/env python3
"""Download PASCAL VOC and convert to YOLO format.

Usage:
    # Quick start — VOC2007 only, limited samples (~900 MB download)
    python prepare_voc.py

    # Full VOC2007 + VOC2012 (larger download ~3 GB)
    python prepare_voc.py --years 2007 2012 --max-train 0 --max-val 0

    # Small subset for debugging (500 train / 100 val)
    python prepare_voc.py --max-train 500 --max-val 100

    # Train on 3 classes only
    python prepare_voc.py --classes person car dog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from utils.voc import VOC_CLASSES, prepare_voc_dataset


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare PASCAL VOC dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="YAML config (reads data.voc section if present)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/voc from config or ./data/voc)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=None,
        help="VOC years to download, e.g. 2007 or 2007 2012",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Max training images (0 = no limit, default from config or 2000)",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Max validation images (0 = no limit, default from config or 500)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=f"Subset of classes. All: {', '.join(VOC_CLASSES[:5])}...",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download VOC archives even if already present",
    )
    return parser.parse_args()


def _resolve_limit(value: int | None, config_value: int | None, default: int) -> int | None:
    """Convert CLI/config limit to None (= no cap) or positive int."""
    if value is not None:
        return None if value == 0 else value
    if config_value is not None:
        return None if config_value == 0 else config_value
    return default


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    voc_cfg = {}
    if cfg_path.exists():
        cfg = load_config(cfg_path)
        voc_cfg = cfg.get("data", {}).get("voc", {})

    output_dir = args.output or voc_cfg.get("output_dir", "data/voc")
    years = args.years or voc_cfg.get("years", ["2007"])
    years = [str(y) for y in years]

    max_train = _resolve_limit(
        args.max_train,
        voc_cfg.get("max_train_samples"),
        default=2000,
    )
    max_val = _resolve_limit(
        args.max_val,
        voc_cfg.get("max_val_samples"),
        default=500,
    )
    selected_classes = args.classes or voc_cfg.get("classes")

    prepare_voc_dataset(
        output_dir=PROJECT_ROOT / output_dir,
        years=years,
        max_train_samples=max_train,
        max_val_samples=max_val,
        selected_classes=selected_classes,
        force_download=args.force_download or voc_cfg.get("force_download", False),
    )


if __name__ == "__main__":
    main()
