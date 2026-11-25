"""Figure and numbers for docs/two_stage.md: what the first stage hands to the second, measured on VOC 2007 test.

    python scripts/explainer_figures.py      # CPU, about a minute: cached proposals + annotations, no weights

1. Selective Search recall: share of ground-truth objects covered by at least one of the top-N cached proposals
   (IoU >= 0.5 and >= 0.7), for N from 10 to all.
2. RPN anchors: for each object, the best IoU among the 9 anchors per position of `models.RPN`, on the 600-pixel
   resized image the detectors actually see, ignoring cross-boundary anchors as `RPN.loss` does. This is before any
   learning: what the box regression has to correct.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from twostage.boxes import box_iou  # noqa: E402
from twostage.models import RPN  # noqa: E402
from twostage.voc import VOC_ROOT, load_annotation, load_proposals, resize_scale, split_ids  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
BLUE, GREY, ORANGE = "#167D9A", "#9AA5B1", "#E0A33B"
TOPS = [10, 30, 100, 300, 500, 1000, 2000]


def image_size(image_id: str) -> tuple[int, int]:
    import xml.etree.ElementTree as ET

    size = ET.parse(VOC_ROOT / "VOC2007" / "Annotations" / f"{image_id}.xml").find("size")
    return int(size.find("width").text), int(size.find("height").text)


def main() -> None:
    torch.set_grad_enabled(False)
    rpn = RPN()
    ids = split_ids("test")
    best_ss = {t: [] for t in TOPS}  # per object: best IoU among the top-t proposals
    best_anchor, sides, n_proposals = [], [], []
    for image_id in ids:
        ann = load_annotation(image_id)
        gt = torch.from_numpy(ann["boxes"][~ann["difficult"]])  # difficult objects are ignored by VOC evaluation
        if len(gt) == 0:
            continue
        props = torch.from_numpy(load_proposals(image_id))
        n_proposals.append(len(props))
        iou = box_iou(gt, props)  # (G, P), proposals in their stored (ranked) order
        for t in TOPS:
            best_ss[t].append(iou[:, :t].max(1).values if iou.shape[1] else torch.zeros(len(gt)))
        w, h = image_size(image_id)
        s = resize_scale(w, h)
        W, H = round(w * s), round(h * s)  # the image the detectors see
        fh, fw = -(-H // 32) * 2, -(-W // 32) * 2  # batch_images pads to a multiple of 32; features at stride 16
        anchors = rpn.anchors(fh, fw)
        inside = (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] < W) & (anchors[:, 3] < H)
        best_anchor.append(box_iou(gt * s, anchors[inside]).max(1).values)  # as in RPN.loss: cross-boundary ignored
        sides.append(((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).sqrt() * s)

    ss = {t: torch.cat(v).numpy() for t, v in best_ss.items()}
    anc = torch.cat(best_anchor).numpy()
    side = torch.cat(sides).numpy()
    m = {
        "images": len(n_proposals),
        "objects_non_difficult": int(len(anc)),
        "selective_search_proposals_per_image_median": int(np.median(n_proposals)),
        "selective_search_recall_iou05": {t: round(float((ss[t] >= 0.5).mean()), 3) for t in TOPS},
        "selective_search_recall_iou07": {t: round(float((ss[t] >= 0.7).mean()), 3) for t in TOPS},
        "anchors_best_iou_ge_07": round(float((anc >= 0.7).mean()), 3),
        "anchors_best_iou_ge_05": round(float((anc >= 0.5).mean()), 3),
        "anchors_best_iou_below_03": round(float((anc < 0.3).mean()), 3),
        "anchors_best_iou_median": round(float(np.median(anc)), 3),
        "median_side_px_objects_best_anchor_below_03": round(float(np.median(side[anc < 0.3])), 1),
        "median_side_px_all_objects": round(float(np.median(side)), 1),
    }

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    a0.plot(TOPS, [100 * m["selective_search_recall_iou05"][t] for t in TOPS], "o-", color=BLUE, label="IoU ≥ 0.5")
    a0.plot(TOPS, [100 * m["selective_search_recall_iou07"][t] for t in TOPS], "s--", color=ORANGE,
            label="IoU ≥ 0.7")
    a0.axvline(500, color=GREY, linewidth=1, linestyle=":")
    a0.annotate("R-CNN keeps 500\nat test time", (500, 40), xytext=(6, 0), textcoords="offset points", fontsize=8,
                color="#555555")
    a0.set_xscale("log")
    a0.set_xticks(TOPS, [str(t) for t in TOPS])
    a0.minorticks_off()
    a0.set_ylim(0, 100)
    a0.set_xlabel("top-N Selective Search proposals kept (log scale)")
    a0.set_ylabel("objects covered (%)")
    a0.set_title("Selective Search: high recall needs hundreds of boxes,\nand few of them are tight", fontsize=9)
    a0.legend(frameon=False, fontsize=8, loc="lower right")
    a0.grid(alpha=0.25, which="both")

    a1.hist(anc, bins=np.linspace(0, 1, 41), color=BLUE, alpha=0.85)
    for x, label in [(0.3, "negative below 0.3"), (0.7, "positive from 0.7")]:
        a1.axvline(x, color=ORANGE, linewidth=1.2, linestyle="--")
        a1.annotate(label, (x, a1.get_ylim()[1] * 0.92), xytext=(4, 0), textcoords="offset points", fontsize=8,
                    color="#555555", bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1})
    a1.set_xlabel("best IoU between an object and the 9 anchors at any position")
    a1.set_ylabel("objects")
    a1.set_title(f"RPN anchors before learning: only {100 * m['anchors_best_iou_ge_07']:.0f} % of objects reach the "
                 f"positive\nthreshold, hence the extra rule \"best anchor of each object is positive\"", fontsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "first_stage.png", dpi=150, facecolor="white")
    (ROOT / "reports" / "first_stage.json").write_text(json.dumps(m, indent=2) + "\n")
    for k, v in m.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
