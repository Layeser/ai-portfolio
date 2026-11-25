"""Figure and numbers for docs/yolov1.md: what the 7x7 grid can and cannot represent on VOC 2007 test.

    python scripts/explainer_figures.py      # CPU, about 20 s, reads data/voc/val (no weights needed)

YOLOv1's target holds one object per cell. Every object whose centre falls in an already occupied cell is absent
from the training target: this script counts them, per class, on the 4,952 test images, and compares two candidate
explanations of the per-class AP in reports/eval_voc2007_test.json: these collisions, and object size.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.dataset import YOLODataset  # noqa: E402
from utils.voc import VOC_CLASSES  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
BLUE, GREY, ORANGE = "#167D9A", "#9AA5B1", "#E0A33B"
S, SIZE = 7, 448


def cells(boxes) -> list[tuple[int, int]]:
    """Grid cell of each box centre, with the same rule as utils.boxes.encode_boxes."""
    c = SIZE / S
    return [(min(int(y / c), S - 1), min(int(x / c), S - 1)) for x, y, _, _ in boxes.tolist()]


def main() -> None:
    ds = YOLODataset(ROOT / "data" / "voc" / "val", image_size=SIZE, grid_size=S)
    total, lost = Counter(), Counter()
    sizes: dict[int, list[float]] = {}
    per_image = []
    for i in range(len(ds)):
        boxes, labels = ds.annotations(i)
        cell_of = cells(boxes)
        counts = Counter(cell_of)
        # encode_boxes writes the objects in order, so in a shared cell only the last one survives
        survivor = {cell: k for k, cell in enumerate(cell_of)}
        n_lost = 0
        for k, (cell, label) in enumerate(zip(cell_of, labels.tolist(), strict=True)):
            total[label] += 1
            sizes.setdefault(label, []).append(float((boxes[k, 2] * boxes[k, 3]).sqrt()) / SIZE)
            if survivor[cell] != k:
                lost[label] += 1
                n_lost += 1
        per_image.append((n_lost, max(counts.values(), default=0), i))

    n_obj, n_lost = sum(total.values()), sum(lost.values())
    rate = {VOC_CLASSES[c]: lost[c] / total[c] for c in total}
    ap = dict(json.loads((ROOT / "reports" / "eval_voc2007_test.json").read_text())["per_class_AP"])
    names = list(VOC_CLASSES)
    x = np.array([100 * rate[n] for n in names])
    size = np.array([np.median(sizes[c]) for c in range(len(names))])
    y = np.array([ap[n] for n in names])

    def spearman(a, b) -> float:
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return round(float(np.corrcoef(ra, rb)[0, 1]), 2)

    m = {
        "objects": n_obj,
        "objects_absent_from_target": n_lost,
        "share_absent_pct": round(100 * n_lost / n_obj, 1),
        "images_with_a_collision_pct": round(100 * sum(p[0] > 0 for p in per_image) / len(per_image), 1),
        "share_absent_per_class_pct": {n: round(100 * rate[n], 1) for n in names},
        "median_object_size_per_class": {n: round(float(v), 3) for n, v in zip(names, size, strict=True)},
        "spearman_absent_share_vs_ap": spearman(x, y),
        "spearman_median_size_vs_ap": spearman(size, y),
    }

    # Left: the image with the most objects in one cell, among images with at most 12 objects (readable)
    _, _, idx = max(p for p in per_image if len(ds.annotations(p[2])[0]) <= 12)
    boxes, labels = ds.annotations(idx)
    cell_of = cells(boxes)
    survivor = {cell: k for k, cell in enumerate(cell_of)}

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    # Boxes only, no photograph: the geometry is the point
    c = SIZE / S  # zoom on the occupied cells, plus a margin, aligned on the grid
    x1, y1 = (boxes[:, :2] - boxes[:, 2:] / 2).min(0).values.tolist()
    x2, y2 = (boxes[:, :2] + boxes[:, 2:] / 2).max(0).values.tolist()
    a0.set_xlim(max(0, (x1 // c) * c), min(SIZE, (x2 // c + 1) * c))
    a0.set_ylim(min(SIZE, (y2 // c + 1) * c), max(0, (y1 // c) * c))
    a0.set_aspect("equal")
    a0.set_facecolor("#F4F4F4")
    for g in range(1, S):
        a0.axhline(g * SIZE / S, color=GREY, linewidth=0.6)
        a0.axvline(g * SIZE / S, color=GREY, linewidth=0.6)
    for k, ((cx, cy, w, h), cell) in enumerate(zip(boxes.tolist(), cell_of, strict=True)):
        kept = survivor[cell] == k
        color = BLUE if kept else ORANGE
        a0.add_patch(patches.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor=color,
                                       linewidth=1.6, linestyle="-" if kept else "--"))
        a0.plot(cx, cy, "o" if kept else "x", color=color, markersize=6, markeredgewidth=2)
    for k, ((cx, cy, w, h), label) in enumerate(zip(boxes.tolist(), labels.tolist(), strict=True)):
        a0.annotate(VOC_CLASSES[label], (cx - w / 2, cy - h / 2), xytext=(1, 2), textcoords="offset points",
                    fontsize=7, color=BLUE if survivor[cell_of[k]] == k else ORANGE)
    a0.set_xticks([])
    a0.set_yticks([])
    a0.set_title(f"Test image {ds.image_paths[idx].stem}, zoom on the grid: {sum(survivor[c] != k for k, c in enumerate(cell_of))} of {len(cell_of)} objects "
                 "share a cell\n(dashed: absent from the training target)", fontsize=9)

    a1.scatter(size, y, s=12 + 4 * x, color=BLUE, alpha=0.85, zorder=3)
    below = {"bird", "cow", "pottedplant"}  # their neighbours' labels would overlap
    for n, xi, yi in zip(names, size, y, strict=True):
        a1.annotate(n, (xi, yi), textcoords="offset points", xytext=(4, -9 if n in below else 3), fontsize=7,
                    color="#333333")
    a1.set_xlabel("median object size in the class: √(w·h) / image side")
    a1.set_ylabel("test AP@0.5 of the trained model")
    a1.grid(alpha=0.25)
    a1.set_title(f"Per-class AP follows object size (Spearman ρ = {m['spearman_median_size_vs_ap']}), "
                 f"much more than\nthe share of objects lost to shared cells (ρ = {m['spearman_absent_share_vs_ap']}; "
                 "marker area)", fontsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "grid_collisions.png", dpi=150, facecolor="white")
    (ROOT / "reports" / "grid_collisions.json").write_text(json.dumps(m, indent=2) + "\n")
    for k, v in m.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
