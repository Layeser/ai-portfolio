"""Carve a validation split out of the training images (seeded), so checkpoint selection never sees
the VOC 2007 test set, which is reserved for the final mAP.

    python scripts/make_holdout.py data/voc/train data/voc/holdout 1000
"""

import random
import shutil
import sys
from pathlib import Path


def main(train_dir: Path, holdout_dir: Path, n: int, seed: int = 42) -> None:
    images = sorted((train_dir / "images").glob("*.jpg"))
    chosen = random.Random(seed).sample(images, n)
    for sub in ("images", "labels"):
        (holdout_dir / sub).mkdir(parents=True, exist_ok=True)
    for img in chosen:
        shutil.move(str(img), holdout_dir / "images" / img.name)
        label = train_dir / "labels" / f"{img.stem}.txt"
        if label.exists():
            shutil.move(str(label), holdout_dir / "labels" / label.name)
    print(f"moved {n} images from {train_dir} to {holdout_dir}; {len(images) - n} left for training")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
