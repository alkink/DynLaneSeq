from __future__ import annotations

from dataclasses import dataclass

from .loss_s1 import S1Criterion, S1LossConfig


@dataclass
class S2LossConfig(S1LossConfig):
    lambda_coarse: float = 0.5


class S2Criterion(S1Criterion):
    cfg: S2LossConfig

    def __init__(self, cfg: S2LossConfig | None = None):
        super().__init__(cfg or S2LossConfig())

    def forward(self, outputs, targets, matches):
        final_losses = super().forward(outputs["final"], targets, matches)
        seg_loss = self.compute_seg_loss(outputs, targets)
        coarse_point = self.compute_point_loss(outputs["coarse"], targets, matches)
        coarse_range = self.compute_range_loss(outputs["coarse"], targets, matches)
        coarse_line_iou = self.compute_line_iou_loss(outputs["coarse"], targets, matches)
        coarse = coarse_point + 0.5 * coarse_range
        if self.cfg.w_line_iou > 0:
            coarse = coarse + self.cfg.w_line_iou * coarse_line_iou
        if self.cfg.w_seg > 0:
            final_losses["loss_total"] = final_losses["loss_total"] + self.cfg.w_seg * seg_loss
            final_losses["loss_seg"] = seg_loss
        final_losses["loss_point_coarse"] = coarse_point
        final_losses["loss_range_coarse"] = coarse_range
        final_losses["loss_line_iou_coarse"] = coarse_line_iou
        final_losses["loss_coarse"] = coarse
        final_losses["loss_total"] = final_losses["loss_total"] + self.cfg.lambda_coarse * coarse
        return final_losses
