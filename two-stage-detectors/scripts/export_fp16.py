"""Make light, inference-only checkpoints for release: model weights only, stored in float16.

`load_state_dict` copies them into the float32 model, so no change is needed to load them.

    python scripts/export_fp16.py      # checkpoints/<model>/best.pt -> checkpoints/<model>/<model>_voc2007_fp16.pt
"""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["rcnn", "fast_rcnn", "faster_rcnn"]


def export(src: Path, dst: Path) -> None:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    weights = {k: v.half() if v.is_floating_point() else v for k, v in checkpoint["model"].items()}
    torch.save({"model": weights, "epoch": checkpoint.get("epoch"),
                "holdout_mAP07": checkpoint.get("holdout_mAP07")}, dst)
    print(f"{src.relative_to(ROOT)} ({src.stat().st_size / 1e6:.0f} MB) -> {dst.relative_to(ROOT)} "
          f"({dst.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    for m in MODELS:
        export(ROOT / "checkpoints" / m / "best.pt", ROOT / "checkpoints" / m / f"{m}_voc2007_fp16.pt")
