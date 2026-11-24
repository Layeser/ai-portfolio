"""README figures: holdout mAP per epoch, test mAP vs speed, detections on VOC 2007 test images.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from twostage.evaluate import _to  # noqa: E402
from twostage.models import build  # noqa: E402
from twostage.voc import CLASSES, VOCDetection, split_ids  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
NAMES = {"rcnn": "R-CNN", "fast_rcnn": "Fast R-CNN", "faster_rcnn": "Faster R-CNN"}
COLORS = {"rcnn": "#9AA5B1", "fast_rcnn": "#E0A33B", "faster_rcnn": "#167D9A"}


def available(prefix: str) -> list[str]:
    return [m for m in NAMES if (ROOT / "reports" / f"{prefix}_{m}{'_test' if prefix == 'eval' else ''}.json").exists()]


def training_curves() -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for m in available("history"):
        h = json.loads((ROOT / "reports" / f"history_{m}.json").read_text())
        ax.plot([r["epoch"] for r in h], [r["holdout_mAP07"] for r in h], "o-", color=COLORS[m], label=NAMES[m])
    ax.set(xlabel="epoch", ylabel="holdout mAP@0.5 (VOC07 metric)", title="Learning curves (300 held-out images)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "training.png", dpi=160)
    plt.close(fig)


def speed_accuracy() -> None:
    models = available("eval")
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for m in models:
        r = json.loads((ROOT / "reports" / f"eval_{m}_test.json").read_text())
        ax.scatter(r["seconds_per_image"], r["mAP07"], s=90, color=COLORS[m], zorder=3)
        ax.annotate(f"{NAMES[m]}\n{r['mAP07']:.3f}", (r["seconds_per_image"], r["mAP07"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xscale("log")
    ax.set(xlabel="seconds per image (GPU, log scale; Selective Search not included)",
           ylabel="VOC 2007 test mAP@0.5", title="Each generation is faster and more accurate")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "speed_accuracy.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def detections(model_name: str = "faster_rcnn", n: int = 6, conf: float = 0.5) -> None:
    weights = ROOT / "checkpoints" / model_name / "best.pt"
    if not weights.exists():
        return
    model = build(model_name, pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(weights, map_location="cuda")["model"])
    ids = split_ids("test")[:n]  # first test images: not cherry-picked
    data = VOCDetection(ids, train=False, proposals=model.needs_proposals)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (image, target) in zip(axes.flat, (data[i] for i in range(n)), strict=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            det = model([image.cuda()], [_to(target, "cuda")])[0]
        ax.imshow(image.permute(1, 2, 0).numpy())
        for box, score, label in zip(det["boxes"].float().cpu(), det["scores"].float().cpu(), det["labels"].cpu(),
                                     strict=True):
            if score < conf:
                continue
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#E0A33B", lw=2))
            ax.text(x1, y1, f"{CLASSES[label - 1]} {score:.2f}", color="white", fontsize=7, va="bottom",
                    bbox={"facecolor": "#E0A33B", "pad": 1, "lw": 0})
        ax.axis("off")
    fig.suptitle(f"{NAMES[model_name]} on the first VOC 2007 test images (score ≥ {conf})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "detections.jpg", dpi=110, pil_kwargs={"quality": 85})
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    training_curves()
    speed_accuracy()
    detections()
    print("figures written to", FIGURES)
