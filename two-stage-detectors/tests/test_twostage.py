"""Fast CPU tests on synthetic data (no VOC download, no pretrained weights)."""

import numpy as np
import pytest
import torch
from torchvision.ops import box_iou as tv_box_iou
from twostage.boxes import box_iou, decode, encode, match, sample
from twostage.evaluate import average_precision
from twostage.models import RCNN, RPN, FasterRCNN, FastRCNN


def random_boxes(n, size=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * size
    wh = torch.rand(n, 2, generator=g) * size / 2 + 4
    return torch.cat([xy, xy + wh], 1)


def test_iou_matches_torchvision():
    a, b = random_boxes(30), random_boxes(20, seed=1)
    assert torch.allclose(box_iou(a, b), tv_box_iou(a, b), atol=1e-5)


def test_encode_decode_roundtrip():
    ref, gt = random_boxes(50), random_boxes(50, seed=2)
    for weights in [(1, 1, 1, 1), (10, 10, 5, 5)]:
        assert torch.allclose(decode(encode(gt, ref, weights), ref, weights), gt, atol=1e-3)


def test_match_and_sample():
    gt = torch.tensor([[0.0, 0, 100, 100]])
    props = torch.tensor([[0.0, 0, 100, 100], [0, 0, 60, 100], [0, 0, 30, 100], [300, 300, 400, 400]])
    labels, _ = match(props, gt, torch.tensor([7]), fg_iou=0.5, bg_low=0.1)
    assert labels.tolist() == [7, 7, 0, -1]  # IoU 1, 0.6, 0.3 (bg), 0 (ignored below 0.1)
    pos, neg = sample(labels, num=4, positive_fraction=0.25)
    assert len(pos) == 1 and len(neg) == 1


def test_ap_perfect_and_07_metric():
    recall, precision = np.array([0.5, 1.0]), np.array([1.0, 1.0])
    assert average_precision(recall, precision, False) == pytest.approx(1.0)
    assert average_precision(recall, precision, True) == pytest.approx(1.0)


def test_rpn_anchors_cover_the_image():
    rpn = RPN()
    anchors = rpn.anchors(4, 5)
    assert anchors.shape == (4 * 5 * 9, 4)
    centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    assert centers[:, 0].min() == pytest.approx(8) and centers[:, 0].max() == pytest.approx(72)


def synthetic_batch(with_proposals: bool):
    images = [torch.rand(3, 160, 192), torch.rand(3, 128, 160)]
    targets = []
    for _ in images:
        gt = torch.tensor([[20.0, 30, 90, 110]])
        t = {"boxes": gt, "labels": torch.tensor([3]), "image_id": "x", "scale": 1.0}
        if with_proposals:
            t["proposals"] = torch.cat([gt + torch.randn(20, 4) * 5, random_boxes(30, size=120)])
        targets.append(t)
    return images, targets


@pytest.mark.parametrize("cls", [FastRCNN, FasterRCNN, RCNN])
def test_train_and_inference_shapes(cls):
    torch.manual_seed(0)
    model = cls(pretrained=False)
    if cls is RCNN:
        model.crop, model.test_proposals = 64, 20  # keep the CPU test fast
    images, targets = synthetic_batch(model.needs_proposals)
    model.train()
    losses = model(images, targets)
    assert all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
    model.eval()
    with torch.no_grad():
        out = model(images, targets)
    assert len(out) == 2 and set(out[0]) == {"boxes", "scores", "labels"}
    if len(out[0]["boxes"]):
        assert (out[0]["labels"] >= 1).all() and (out[0]["boxes"][:, 2] <= 191).all()
