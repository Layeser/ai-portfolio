import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect import scale_boxes_to_image
from src.model import YOLOv1


def test_scale_boxes_to_image_with_letterbox_padding() -> None:
    boxes = torch.tensor([[224.0, 224.0, 112.0, 112.0]], dtype=torch.float32)

    scaled = scale_boxes_to_image(
        boxes,
        original_size=(896, 448),
        model_size=448,
    )

    expected = torch.tensor([[448.0, 224.0, 224.0, 224.0]], dtype=torch.float32)
    assert torch.allclose(scaled, expected)


def test_resnet50_backbone_can_instantiate_and_forward() -> None:
    model = YOLOv1(num_classes=20, num_boxes=2, grid_size=7, backbone_type="pretrained_resnet50")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(x)

    assert outputs.shape == (1, 7, 7, 2 * 5 + 20)
