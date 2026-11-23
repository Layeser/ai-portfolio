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


def test_random_scale_translate_moves_boxes_with_the_image():
    """Paint a white square, augment, and check the transformed box still covers the white pixels."""
    import numpy as np
    from PIL import Image
    from utils.dataset import random_scale_translate

    torch.manual_seed(3)
    img = Image.new("RGB", (448, 448), (0, 0, 0))
    img.paste((255, 255, 255), (150, 180, 250, 260))  # box: cx=200, cy=220, w=100, h=80
    boxes = torch.tensor([[200.0, 220.0, 100.0, 80.0]])
    for _ in range(10):
        out, new_boxes, labels = random_scale_translate(img, boxes, torch.tensor([3]))
        white = np.argwhere(np.asarray(out)[..., 0] > 128)
        if len(new_boxes) == 0:
            continue
        cx, cy, w, h = new_boxes[0].tolist()
        ys, xs = white[:, 0], white[:, 1]
        assert abs(xs.mean() - cx) < 3 and abs(ys.mean() - cy) < 3
        assert abs((xs.max() - xs.min()) - w) < 4 and abs((ys.max() - ys.min()) - h) < 4
        assert labels.tolist() == [3]


def test_boxes_pushed_outside_are_dropped():
    from PIL import Image
    from utils.dataset import random_scale_translate

    img = Image.new("RGB", (448, 448))
    boxes = torch.tensor([[5.0, 5.0, 10.0, 10.0]])  # tiny box in the corner
    torch.manual_seed(0)
    results = [len(random_scale_translate(img, boxes, torch.tensor([1]))[1]) for _ in range(30)]
    assert 0 in results  # often translated out of the image, then dropped
