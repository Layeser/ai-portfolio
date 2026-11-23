"""Make a light, inference-only checkpoint for release.

Training checkpoints also hold the optimizer state (~2x the weights). This keeps only the model
weights, stored in float16 (half the size again); `load_checkpoint` casts them back to float32.

    python scripts/export_fp16.py checkpoints/yolov1/last.pt checkpoints/yolov1/yolov1_voc2007_fp16.pt
"""

import sys
from pathlib import Path

import torch


def export(src: Path, dst: Path) -> None:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    weights = {k: v.half() if v.is_floating_point() else v for k, v in checkpoint["model_state_dict"].items()}
    torch.save({"model_state_dict": weights, "model_meta": checkpoint.get("model_meta", {}),
                "epoch": checkpoint.get("epoch")}, dst)
    print(f"{src} ({src.stat().st_size / 1e6:.0f} MB) -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
