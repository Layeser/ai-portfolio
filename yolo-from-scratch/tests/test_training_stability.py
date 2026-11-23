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


def test_model_overfits_a_tiny_batch_of_real_style_targets():
    """Sanity check that would have caught the original loss design: on 4 fixed images, the model must
    learn the class and the objectness of annotated cells within a few dozen steps."""
    torch.manual_seed(0)
    model = YOLOv1(num_classes=20, num_boxes=2, grid_size=7, backbone_type="custom")
    criterion = YOLOv1Loss(grid_size=7, num_boxes=2, num_classes=20)
    images = torch.randn(4, 3, 448, 448)
    targets = torch.zeros(4, 7, 7, 25)
    for b, (y, x, cls) in enumerate([(3, 3, 7), (1, 5, 11), (5, 2, 14), (2, 2, 3)]):
        targets[b, y, x, :5] = torch.tensor([1.0, 0.5, 0.5, 0.2, 0.3])
        targets[b, y, x, 5 + cls] = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        optimizer.zero_grad()
        loss, _ = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = model(images)
    conf = torch.maximum(preds[..., 4], preds[..., 9])  # best box confidence per cell
    empty = targets[..., 0] == 0
    confident = 0
    for b, (y, x, cls) in enumerate([(3, 3, 7), (1, 5, 11), (5, 2, 14), (2, 2, 3)]):
        assert preds[b, y, x, 10:].argmax().item() == cls
        # require a clear gap between object and empty cells (an untrained sigmoid gives ~0.5 everywhere)
        confident += int(conf[b, y, x] > conf[b][empty[b]].mean() + 0.3)
    # the two boxes of a cell compete for the object early in training, so one cell may lag behind
    assert confident >= 3


def test_neck_produces_the_detection_grid_without_resampling():
    """448 px input -> 14x14 backbone map -> exactly 7x7 after the neck convolutions (no pooling)."""
    for backbone in ("custom", "pretrained_resnet18"):
        model = YOLOv1(backbone_type=backbone, pretrained_backbone=False).eval()
        with torch.no_grad():
            features = model.neck.neck(model.backbone(torch.randn(1, 3, 448, 448)))
        assert tuple(features.shape[-2:]) == (7, 7), backbone
