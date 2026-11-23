"""README figures: training curves, per-class AP, detections on VOC 2007 test images.

    python scripts/make_figures.py --weights checkpoints/yolov1/yolov1_voc2007_fp16.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIGURES = ROOT / "reports" / "figures"
BLUE, GREY, ORANGE = "#167D9A", "#9AA5B1", "#E0A33B"


def training_curves() -> None:
    h = json.loads((ROOT / "reports" / "training_history.json").read_text())
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(epochs, h["train_loss"], color=GREY, label="train")
    ax1.plot(epochs, h["val_loss"], color=BLUE, label="holdout (checkpoint selection)")
    ax1.set(xlabel="epoch", ylabel="YOLO loss", title="Final run: holdout tracks train (no overfitting)")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)
    names, values = list(h["history_test_mAP50"]), list(h["history_test_mAP50"].values())
    bars = ax2.barh(range(len(names)), values, color=[GREY] * (len(names) - 1) + [BLUE])
    ax2.bar_label(bars, fmt="%.4g", fontsize=8, padding=3)
    ax2.set_yticks(range(len(names)), names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set(xlabel="VOC 2007 test mAP@0.5", xlim=(0, 0.3), title="Debugging history")
    fig.tight_layout()
    fig.savefig(FIGURES / "training.png", dpi=160)
    plt.close(fig)


def per_class_ap() -> None:
    ap = json.loads((ROOT / "reports" / "eval_voc2007_test.json").read_text())["per_class_AP"]
    items = sorted(ap.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([k for k, _ in items], [v for _, v in items], color=[ORANGE if v < 0.1 else BLUE for _, v in items])
    ax.set(xlabel="AP@0.5 (VOC 2007 test, 4,952 images)", title="Small objects are the weak spot (orange: AP < 0.1)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "per_class_ap.png", dpi=160)
    plt.close(fig)


def detections(weights: Path, ids: list[str], conf: float) -> None:
    from api import Detector

    detector = Detector.load(weights)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, image_id in zip(axes.flat, ids, strict=True):
        image = Image.open(ROOT / "data" / "voc" / "val" / "images" / f"{image_id}.jpg").convert("RGB")
        found, *_ = detector(image, conf)
        ax.imshow(image)
        for d in found:
            x1, y1, x2, y2 = d.box_xyxy
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=ORANGE, lw=2))
            ax.text(x1, y1, f"{d.label} {d.score:.2f}", color="white", fontsize=7, va="bottom",
                    bbox={"facecolor": ORANGE, "pad": 1, "lw": 0})
        ax.axis("off")
    fig.suptitle(f"Detections on VOC 2007 test images (confidence ≥ {conf})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "detections.jpg", dpi=120, pil_kwargs={"quality": 85})  # photos: JPEG is 5x smaller
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/yolov1/yolov1_voc2007_fp16.pt")
    parser.add_argument("--ids", nargs=6,  # first six test images: not cherry-picked
                        default=["000001", "000002", "000003", "000004", "000006", "000008"])
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    training_curves()
    per_class_ap()
    detections(args.weights, args.ids, args.conf)
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
