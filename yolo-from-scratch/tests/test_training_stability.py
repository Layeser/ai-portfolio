import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loss import YOLOv1Loss
from src.model import YOLOv1


def test_loss_stays_finite_on_single_batch():
    torch.manual_seed(0)
    model = YOLOv1(num_classes=20, num_boxes=2, grid_size=7)
    criterion = YOLOv1Loss(grid_size=7, num_boxes=2, num_classes=20)

    images = torch.randn(2, 3, 448, 448)
    targets = torch.zeros(2, 7, 7, 25, dtype=torch.float32)
    targets[..., 0] = 1.0
    targets[..., 1] = 0.25
    targets[..., 2] = 0.25
    targets[..., 3] = 0.1
    targets[..., 4] = 0.1
    targets[..., 5 + 0] = 1.0

    predictions = model(images)
    loss, components = criterion(predictions, targets)

    assert torch.isfinite(loss).item(), components
    assert all(torch.isfinite(torch.tensor(value)).item() for value in components.values())
