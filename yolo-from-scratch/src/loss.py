"""YOLOv1 multi-part loss (coordinate, confidence, classification)."""

import torch
import torch.nn as nn


class YOLOv1Loss(nn.Module):
    """YOLOv1 loss from Redmon et al. (2016).

    Components:
      1. Coordinate loss  — MSE on (x, y) and sqrt(w, h) for responsible predictor
      2. Confidence loss  — MSE on object / no-object cells (weighted)
      3. Classification   — MSE on one-hot class vector (with sqrt trick optional)
    """

    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            predictions: (B, S, S, B*5 + C) raw network output
            targets:     (B, S, S, 5 + C)  [obj, x, y, w, h, class_one_hot...]

        Returns:
            total_loss, dict of loss components for logging
        """
        device = predictions.device
        batch_size = predictions.size(0)
        s = self.grid_size
        b = self.num_boxes
        c = self.num_classes

        # Split predictions into box params and class scores
        pred_boxes = predictions[..., : b * 5].view(batch_size, s, s, b, 5)
        pred_classes = predictions[..., b * 5 :]

        obj_mask = targets[..., 0:1]          # (B, S, S, 1)
        noobj_mask = 1.0 - obj_mask
        box_targets = targets[..., 1:5]       # (B, S, S, 4)
        class_targets = targets[..., 5:]      # (B, S, S, C)

        # --- Responsible predictor: box with highest IoU to ground truth ---
        # IoU is computed in a common coordinate frame: (x, y) are cell-relative
        # while (w, h) are image-relative, so we rescale (w, h) to cell units.
        ious = self._pairwise_iou(
            self._to_cell_units(pred_boxes[..., :4]),
            self._to_cell_units(box_targets.unsqueeze(3)),
        )  # (B, S, S, b)
        best_iou, best_box_idx = ious.max(dim=-1, keepdim=True)  # (B, S, S, 1)

        responsible_pred = torch.gather(
            pred_boxes,
            dim=3,
            index=best_box_idx.unsqueeze(-1).expand(-1, -1, -1, -1, 5),
        ).squeeze(3)

        # --- 1. Coordinate loss (only cells with objects) ---
        coord_loss = (
            self.mse(responsible_pred[..., 0:2] * obj_mask, box_targets[..., 0:2] * obj_mask)
            + self.mse(
                torch.sqrt(torch.clamp(responsible_pred[..., 2:4], min=1e-6)) * obj_mask,
                torch.sqrt(torch.clamp(box_targets[..., 2:4], min=1e-6)) * obj_mask,
            )
        )

        # --- 2. Confidence loss ---
        # Object cells: target confidence = IoU(pred, gt) as in the paper (detached).
        conf_target = (best_iou * obj_mask).detach()
        obj_conf_loss = self.mse(responsible_pred[..., 4:5] * obj_mask, conf_target)

        # No-object cells: target confidence = 0 for all boxes
        noobj_conf_loss = self.mse(pred_boxes[..., 4] * noobj_mask, torch.zeros_like(pred_boxes[..., 4]) * noobj_mask)

        # --- 3. Classification loss (only cells with objects) ---
        class_loss = self.mse(pred_classes * obj_mask, class_targets * obj_mask)

        total = (
            self.lambda_coord * coord_loss
            + obj_conf_loss
            + self.lambda_noobj * noobj_conf_loss
            + class_loss
        ) / batch_size

        components = {
            "loss": total.item(),
            "coord": (self.lambda_coord * coord_loss / batch_size).item(),
            "conf_obj": (obj_conf_loss / batch_size).item(),
            "conf_noobj": (self.lambda_noobj * noobj_conf_loss / batch_size).item(),
            "class": (class_loss / batch_size).item(),
        }
        return total, components

    def _to_cell_units(self, boxes: torch.Tensor) -> torch.Tensor:
        """Rescale (w, h) from image-relative to cell-relative units.

        (x, y) are already cell-relative offsets in [0, 1]; multiplying the
        image-relative (w, h) by the grid size puts them in the same frame so
        the IoU is geometrically meaningful within a cell.
        """
        xy = boxes[..., :2]
        wh = boxes[..., 2:4] * self.grid_size
        return torch.cat([xy, wh], dim=-1)

    @staticmethod
    def _pairwise_iou(
        pred_boxes: torch.Tensor,
        target_box: torch.Tensor,
    ) -> torch.Tensor:
        """IoU between each predicted box and the single GT box in a cell."""
        px1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        py1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        px2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        py2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        tx1 = target_box[..., 0] - target_box[..., 2] / 2
        ty1 = target_box[..., 1] - target_box[..., 3] / 2
        tx2 = target_box[..., 0] + target_box[..., 2] / 2
        ty2 = target_box[..., 1] + target_box[..., 3] / 2

        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = (px2 - px1) * (py2 - py1)
        target_area = (tx2 - tx1) * (ty2 - ty1)
        union = pred_area + target_area - inter_area + 1e-6
        return inter_area / union
