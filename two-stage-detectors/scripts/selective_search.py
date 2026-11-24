"""Precompute Selective Search proposals ("fast" mode, as in R-CNN) for VOC 2007 trainval + test.

    python scripts/selective_search.py --workers 8

One .npy per image in data/proposals/ (int16 x1, y1, x2, y2, original pixels, at most 2000 boxes,
duplicates and boxes thinner than 10 px removed). Resumable: images already done are skipped.
Used by R-CNN and Fast R-CNN; Faster R-CNN learns its own proposals instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from twostage.voc import PROPOSALS_DIR, VOC_ROOT, split_ids  # noqa: E402


def propose(image_id: str) -> int:
    import cv2

    cv2.setNumThreads(1)
    out = PROPOSALS_DIR / f"{image_id}.npy"
    if out.exists():
        return 0
    image = cv2.imread(str(VOC_ROOT / "VOC2007" / "JPEGImages" / f"{image_id}.jpg"))
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()  # (x, y, w, h)
    boxes = np.stack([rects[:, 0], rects[:, 1], rects[:, 0] + rects[:, 2] - 1, rects[:, 1] + rects[:, 3] - 1], 1)
    boxes = boxes[(rects[:, 2] >= 10) & (rects[:, 3] >= 10)]
    _, first = np.unique(boxes, axis=0, return_index=True)
    boxes = boxes[np.sort(first)][:2000]  # keep Selective Search's own ranking
    np.save(out, boxes.astype(np.int16))
    return len(boxes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    ids = split_ids("trainval") + split_ids("test")
    start, counts = time.time(), []
    with Pool(args.workers, maxtasksperchild=200) as pool:
        for k, n in enumerate(pool.imap_unordered(propose, ids, chunksize=8), 1):
            if n:
                counts.append(n)
            if k % 500 == 0:
                print(f"{k}/{len(ids)} images, {time.time() - start:.0f}s, mean boxes {np.mean(counts or [0]):.0f}",
                      flush=True)
    print(f"done: {len(ids)} images in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
