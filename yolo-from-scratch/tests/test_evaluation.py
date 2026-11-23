import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import average_precision, batched_nms, voc_ap  # noqa: E402


def test_voc_ap_perfect_and_half():
    assert voc_ap(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    # one true positive then one false positive, 2 objects: recall reaches 0.5 with precision 1
    assert voc_ap(np.array([0.5, 0.5]), np.array([1.0, 0.5])) == pytest.approx(0.5)


def test_average_precision_counts_duplicates_as_false_positives():
    gt = {0: torch.tensor([[50.0, 50.0, 20.0, 20.0]])}
    dets = [(0, 0.9, torch.tensor([50.0, 50.0, 20.0, 20.0])), (0, 0.8, torch.tensor([51.0, 50.0, 20.0, 20.0]))]
    assert average_precision(dets, gt) == pytest.approx(1.0)  # the duplicate comes after recall = 1
    shifted = [(0, 0.9, torch.tensor([90.0, 90.0, 20.0, 20.0]))]
    assert average_precision(shifted, gt) == 0.0


def test_batched_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = torch.tensor([[50.0, 50.0, 20.0, 20.0], [51.0, 50.0, 20.0, 20.0], [50.0, 50.0, 20.0, 20.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([0, 0, 1])
    kept = sorted(batched_nms(boxes, scores, labels, 0.5).tolist())
    assert kept == [0, 2]  # the class-0 duplicate is removed, the class-1 box survives
