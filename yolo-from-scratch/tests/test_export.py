import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_fp16 import export  # noqa: E402
from src.model import YOLOv1  # noqa: E402
from utils.helpers import load_checkpoint  # noqa: E402


def test_fp16_release_checkpoint_loads_and_predicts_the_same(tmp_path):
    torch.manual_seed(0)
    model = YOLOv1(backbone_type="custom").eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    full = tmp_path / "last.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": 3, "model_meta": {"backbone_type": "custom"}}, full)
    light = tmp_path / "release.pt"
    export(full, light)

    restored = YOLOv1(backbone_type="custom").eval()
    load_checkpoint(light, restored)
    x = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x), atol=2e-2)
    assert light.stat().st_size < 0.6 * full.stat().st_size
