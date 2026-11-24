"""R-CNN (2014) -> Fast R-CNN (2015) -> Faster R-CNN (2015), written from the papers on a common base.

All three use an ImageNet-pretrained ResNet-50 (frozen batch norm, conv1 + layer1 frozen) so that the
comparison isolates the *detection design*:

| model        | proposals                    | features                                   |
|--------------|------------------------------|--------------------------------------------|
| R-CNN        | Selective Search (external)  | one CNN pass **per region** (warped crop)   |
| Fast R-CNN   | Selective Search (external)  | one CNN pass **per image** + RoI pooling    |
| Faster R-CNN | Region Proposal Network      | one CNN pass per image, proposals learned   |

Fast / Faster R-CNN use the "C4" layout of the Faster R-CNN paper's ResNet experiments: conv1..layer3
on the whole image (stride 16), RoI pooling to 14x14, layer4 as the per-region head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import batched_nms, nms, roi_align, roi_pool
from torchvision.ops.misc import FrozenBatchNorm2d

from twostage.boxes import box_iou, clip, decode, encode, match, sample

NUM_CLASSES = 21  # 20 VOC classes + background
HEAD_WEIGHTS = (10.0, 10.0, 5.0, 5.0)  # standard scaling of regression targets for the detection head
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def resnet50(pretrained: bool = True) -> nn.Module:
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)
    for module in (net.conv1, net.layer1):
        for p in module.parameters():
            p.requires_grad_(False)
    return net


def batch_images(images: list[torch.Tensor], divisor: int = 32) -> torch.Tensor:
    """Normalise and zero-pad a list of (3, H, W) images into one (B, 3, H', W') tensor."""
    h = math.ceil(max(im.shape[1] for im in images) / divisor) * divisor
    w = math.ceil(max(im.shape[2] for im in images) / divisor) * divisor
    out = images[0].new_zeros(len(images), 3, h, w)
    for i, im in enumerate(images):
        out[i, :, : im.shape[1], : im.shape[2]] = (im - MEAN.to(im.device)) / STD.to(im.device)
    return out


def smooth_l1(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="sum")


def postprocess(scores: torch.Tensor, deltas: torch.Tensor, rois: torch.Tensor, image_size,
                score_thresh: float = 0.05, nms_thresh: float = 0.3, max_dets: int = 100) -> dict:
    """Per-class box decoding, score threshold, per-class NMS (as in the Fast R-CNN paper)."""
    probs = F.softmax(scores.float(), -1)
    boxes = clip(decode(deltas.float(), rois, HEAD_WEIGHTS), *image_size).view(-1, NUM_CLASSES, 4)
    boxes, probs = boxes[:, 1:].reshape(-1, 4), probs[:, 1:].reshape(-1)
    labels = torch.arange(1, NUM_CLASSES, device=rois.device).repeat(len(rois))
    keep = probs > score_thresh
    boxes, probs, labels = boxes[keep], probs[keep], labels[keep]
    wh = boxes[:, 2:] - boxes[:, :2]
    keep = (wh > 1).all(1)
    boxes, probs, labels = boxes[keep], probs[keep], labels[keep]
    keep = batched_nms(boxes, probs, labels, nms_thresh)[:max_dets]
    return {"boxes": boxes[keep], "scores": probs[keep], "labels": labels[keep]}


def head_targets(proposals, target, fg_iou: float, bg_low: float, per_image: int, fg_fraction: float):
    """Sample RoIs for the detection head and compute their class / regression targets."""
    gt = target["boxes"]
    rois = torch.cat([proposals, gt]) if len(gt) else proposals  # ground truth as extra positives
    labels, matched = match(rois, gt, target["labels"], fg_iou, bg_low)
    pos, neg = sample(labels, per_image, fg_fraction)
    idx = torch.cat([pos, neg])
    labels = labels[idx]
    labels[len(pos):] = 0
    deltas = encode(matched[idx], rois[idx], HEAD_WEIGHTS) if len(gt) else torch.zeros(len(idx), 4,
                                                                                       device=rois.device)
    return rois[idx], labels, deltas


def head_loss(scores, deltas, labels, target_deltas) -> dict:
    loss_cls = F.cross_entropy(scores.float(), labels)
    fg = torch.nonzero(labels > 0).flatten()
    pred = deltas.float().view(len(deltas), NUM_CLASSES, 4)[fg, labels[fg]]
    loss_box = smooth_l1(pred, target_deltas[fg], beta=1.0) / max(len(labels), 1)
    return {"loss_cls": loss_cls, "loss_box": loss_box}


class C4Head(nn.Module):
    """RoI pooling (14x14) -> ResNet layer4 -> global average pool -> class scores + per-class boxes."""

    def __init__(self, layer4: nn.Module, pooler: str = "align"):
        super().__init__()
        self.layer4, self.pooler = layer4, pooler
        self.cls = nn.Linear(2048, NUM_CLASSES)
        self.box = nn.Linear(2048, NUM_CLASSES * 4)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.normal_(self.box.weight, std=0.001)
        nn.init.zeros_(self.cls.bias)
        nn.init.zeros_(self.box.bias)

    def forward(self, features: torch.Tensor, rois: list[torch.Tensor]):
        if self.pooler == "align":
            x = roi_align(features, rois, 14, spatial_scale=1 / 16, sampling_ratio=0, aligned=True)
        else:  # RoIPool, exactly as in the Fast R-CNN paper
            x = roi_pool(features, rois, 14, spatial_scale=1 / 16)
        if self.training:
            x = self.layer4(x).mean((2, 3))
        else:  # up to 2000 RoIs per image at test time: run layer4 in chunks to bound GPU memory
            x = torch.cat([self.layer4(c).mean((2, 3)) for c in x.split(512)])
        return self.cls(x), self.box(x)


class FastRCNN(nn.Module):
    """Girshick, Fast R-CNN (ICCV 2015): shared image features, RoI pooling, multi-task loss."""

    needs_proposals = True

    def __init__(self, pretrained: bool = True, pooler: str = "align", train_proposals: int = 2000,
                 test_proposals: int = 2000):
        super().__init__()
        net = resnet50(pretrained)
        self.body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
        self.head = C4Head(net.layer4, pooler)
        self.train_proposals, self.test_proposals = train_proposals, test_proposals

    def proposals(self, images, features, targets) -> list[torch.Tensor]:
        n = self.train_proposals if self.training else self.test_proposals
        return [t["proposals"][:n].float() for t in targets]

    def extra_losses(self) -> dict:
        return {}

    def forward(self, images: list[torch.Tensor], targets: list[dict]):
        features = self.body(batch_images(images))
        proposals = self.proposals(images, features, targets)
        if self.training:
            sampled = [head_targets(p.detach(), t, 0.5, self.bg_low, 128, 0.25)
                       for p, t in zip(proposals, targets, strict=True)]
            scores, deltas = self.head(features, [s[0] for s in sampled])
            losses = head_loss(scores, deltas, torch.cat([s[1] for s in sampled]), torch.cat([s[2] for s in sampled]))
            return {**losses, **self.extra_losses()}
        scores, deltas = self.head(features, proposals)
        out, start = [], 0
        for im, p in zip(images, proposals, strict=True):
            end = start + len(p)
            out.append(postprocess(scores[start:end], deltas[start:end], p, im.shape[1:]))
            start = end
        return out

    bg_low = 0.1  # Fast R-CNN: background RoIs have IoU in [0.1, 0.5)


class RPN(nn.Module):
    """Ren et al., Faster R-CNN (NeurIPS 2015): 9 anchors per position (3 scales x 3 aspect ratios)."""

    def __init__(self, channels: int = 1024, scales=(128, 256, 512), ratios=(0.5, 1.0, 2.0), stride: int = 16):
        super().__init__()
        self.stride = stride
        base = []
        for r in ratios:
            for s in scales:
                w, h = s / math.sqrt(r), s * math.sqrt(r)
                base.append([-w / 2, -h / 2, w / 2, h / 2])
        self.register_buffer("base_anchors", torch.tensor(base), persistent=False)
        self.conv = nn.Conv2d(channels, 512, 3, padding=1)
        self.cls = nn.Conv2d(512, len(base), 1)
        self.box = nn.Conv2d(512, len(base) * 4, 1)
        for layer in (self.conv, self.cls, self.box):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def anchors(self, h: int, w: int) -> torch.Tensor:
        ys = (torch.arange(h, device=self.base_anchors.device) + 0.5) * self.stride
        xs = (torch.arange(w, device=self.base_anchors.device) + 0.5) * self.stride
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([cx, cy, cx, cy], -1).reshape(-1, 1, 4)
        return (centers + self.base_anchors[None]).reshape(-1, 4)  # (H*W*A, 4), position-major

    def forward(self, features, images, targets, training: bool):
        x = F.relu(self.conv(features))
        logits = self.cls(x).permute(0, 2, 3, 1).reshape(len(images), -1).float()
        deltas = self.box(x).permute(0, 2, 3, 1).reshape(len(images), -1, 4).float()
        anchors = self.anchors(*features.shape[-2:])
        pre_nms, post_nms = (12000, 2000) if training else (6000, 300)
        proposals = []
        for i, im in enumerate(images):
            with torch.no_grad():
                boxes = clip(decode(deltas[i].detach(), anchors), *im.shape[1:])
                scores = logits[i].detach()
                top = scores.topk(min(pre_nms, len(scores))).indices
                boxes, scores = boxes[top], scores[top]
                wh = boxes[:, 2:] - boxes[:, :2]
                keep = (wh >= 1).all(1)
                boxes, scores = boxes[keep], scores[keep]
                keep = nms(boxes, scores, 0.7)[:post_nms]
                proposals.append(boxes[keep])
        losses = {}
        if training:
            losses = self.loss(logits, deltas, anchors, images, targets)
        return proposals, losses

    def loss(self, logits, deltas, anchors, images, targets) -> dict:
        labels_all, idx_all, box_targets, box_idx = [], [], [], []
        for i, t in enumerate(targets):
            labels = torch.full((len(anchors),), -1, dtype=torch.long, device=anchors.device)
            h, w = images[i].shape[1:]
            inside = (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] < w) & (anchors[:, 3] < h)
            gt = t["boxes"]
            if len(gt):
                iou = box_iou(anchors, gt)
                iou[~inside] = -1  # the paper ignores cross-boundary anchors during training
                best, which = iou.max(1)
                labels[(best >= 0) & (best < 0.3)] = 0
                labels[best >= 0.7] = 1
                gt_best = iou.max(0).values
                hits = (iou == gt_best[None]) & (gt_best[None] > 0)  # each GT keeps its best anchor(s)
                labels[hits.any(1)] = 1
            else:
                labels[inside] = 0
                which = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
            pos, neg = sample(labels, 256, 0.5)
            idx = torch.cat([pos, neg])
            labels_all.append(labels[idx].float().clamp(min=0))
            idx_all.append(i * len(anchors) + idx)
            if len(pos):
                box_targets.append(encode(gt[which[pos]], anchors[pos]))
                box_idx.append(i * len(anchors) + pos)
        idx_all = torch.cat(idx_all)
        loss_obj = F.binary_cross_entropy_with_logits(logits.reshape(-1)[idx_all], torch.cat(labels_all))
        if box_idx:
            box_idx = torch.cat(box_idx)
            loss_box = smooth_l1(deltas.reshape(-1, 4)[box_idx], torch.cat(box_targets), beta=1 / 9) / len(idx_all)
        else:
            loss_box = deltas.sum() * 0
        return {"loss_rpn_obj": loss_obj, "loss_rpn_box": loss_box}


class FasterRCNN(FastRCNN):
    """Fast R-CNN whose proposals come from an RPN on the same features (approximate joint training)."""

    needs_proposals = False
    bg_low = 0.0

    def __init__(self, pretrained: bool = True, pooler: str = "align"):
        super().__init__(pretrained, pooler)
        self.rpn = RPN()
        self._rpn_losses: dict = {}

    def proposals(self, images, features, targets):
        proposals, self._rpn_losses = self.rpn(features, images, targets, self.training)
        return proposals

    def extra_losses(self) -> dict:
        return self._rpn_losses


class RCNN(nn.Module):
    """Girshick et al., R-CNN (CVPR 2014): every proposal is warped to 224x224 (with 16 px of context)
    and classified by its own CNN pass. Warping is done on the GPU with RoIAlign on the image itself.

    Departure from the paper (documented in the README): the fine-tuned softmax classifier and a
    class-specific box regressor are trained jointly, instead of separate SVMs and ridge regressors.
    """

    needs_proposals = True

    def __init__(self, pretrained: bool = True, crop: int = 224, train_rois: int = 64, test_proposals: int = 500,
                 chunk: int = 256):
        super().__init__()
        net = resnet50(pretrained)
        net.fc = nn.Identity()
        self.net = net
        self.cls = nn.Linear(2048, NUM_CLASSES)
        self.box = nn.Linear(2048, NUM_CLASSES * 4)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.normal_(self.box.weight, std=0.001)
        nn.init.zeros_(self.cls.bias)
        nn.init.zeros_(self.box.bias)
        self.crop, self.train_rois, self.test_proposals, self.chunk = crop, train_rois, test_proposals, chunk

    def warp(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Anisotropic warp with context padding: the box is enlarged so that, once warped to `crop`
        pixels, 16 pixels of surrounding context remain on each side (R-CNN, Appendix A)."""
        pad = 16 * (boxes[:, 2:] - boxes[:, :2]) / (self.crop - 32)
        enlarged = torch.cat([boxes[:, :2] - pad, boxes[:, 2:] + pad], 1)
        rois = torch.cat([enlarged.new_zeros(len(enlarged), 1), enlarged], 1)
        return roi_align(image[None], rois, self.crop, spatial_scale=1.0, sampling_ratio=2, aligned=True)

    def classify(self, crops: torch.Tensor):
        x = self.net(crops)
        return self.cls(x), self.box(x)

    def forward(self, images: list[torch.Tensor], targets: list[dict]):
        normed = [(im - MEAN.to(im.device)) / STD.to(im.device) for im in images]
        if self.training:
            crops, labels, deltas_t, rois = [], [], [], []
            for im, t in zip(normed, targets, strict=True):
                r, lab, d = head_targets(t["proposals"].float(), t, 0.5, 0.0, self.train_rois, 0.25)
                crops.append(self.warp(im, r))
                labels.append(lab)
                deltas_t.append(d)
            scores, deltas = self.classify(torch.cat(crops))
            return head_loss(scores, deltas, torch.cat(labels), torch.cat(deltas_t))
        out = []
        for im, raw, t in zip(normed, images, targets, strict=True):
            rois = t["proposals"][: self.test_proposals].float()
            scores, deltas = [], []
            for k in range(0, len(rois), self.chunk):
                s, d = self.classify(self.warp(im, rois[k : k + self.chunk]))
                scores.append(s)
                deltas.append(d)
            out.append(postprocess(torch.cat(scores), torch.cat(deltas), rois, raw.shape[1:]))
        return out


def build(name: str, pretrained: bool = True) -> nn.Module:
    return {"rcnn": RCNN, "fast_rcnn": FastRCNN, "faster_rcnn": FasterRCNN}[name](pretrained=pretrained)
