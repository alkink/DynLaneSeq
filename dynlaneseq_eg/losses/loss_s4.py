from __future__ import annotations

from dataclasses import dataclass

from .loss_s1 import S1Criterion, S1LossConfig


@dataclass
class S4LossConfig(S1LossConfig):
    lambda_stage1: float = 0.5
    lambda_coarse: float = 0.25


class S4Criterion(S1Criterion):
    cfg: S4LossConfig

    def __init__(self, cfg: S4LossConfig | None = None):
        super().__init__(cfg or S4LossConfig())

    def forward(self, outputs, targets, matches):
        stage2 = {
            "exist_logits": outputs["stage2"]["exist_logits"],
            "row_x_logits": outputs["stage2"]["row_x_logits"],
            "pred_x_rows": outputs["stage2"]["pred_x_rows"],
            "range_norm": outputs["stage2"]["range_norm"],
        }
        losses = super().forward(stage2, targets, matches)
        seg_loss = self.compute_seg_loss(outputs, targets)
        stage1_point = self.compute_point_loss(outputs["stage1"], targets, matches)
        stage1_token = self.compute_token_loss(outputs["stage1"], targets, matches)
        stage1_line_iou = self.compute_line_iou_loss(outputs["stage1"], targets, matches)
        coarse_point = self.compute_point_loss(outputs["coarse"], targets, matches)
        losses["loss_point_stage1"] = stage1_point
        losses["loss_token_stage1"] = stage1_token
        losses["loss_line_iou_stage1"] = stage1_line_iou
        losses["loss_point_coarse"] = coarse_point
        if self.cfg.w_seg > 0:
            losses["loss_total"] = losses["loss_total"] + self.cfg.w_seg * seg_loss
            losses["loss_seg"] = seg_loss
        losses["loss_total"] = (
            losses["loss_total"]
            + self.cfg.lambda_stage1
            * (
                self.cfg.w_point * stage1_point
                + self.cfg.w_token * stage1_token
                + self.cfg.w_line_iou * stage1_line_iou
            )
            + self.cfg.lambda_coarse * coarse_point
        )
        return losses
