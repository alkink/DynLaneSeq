from __future__ import annotations

import torch

from .loss_s1 import S1Criterion
from .loss_s2 import S2Criterion, S2LossConfig


class S3Criterion(S2Criterion):
    def __init__(self, cfg: S2LossConfig | None = None, matcher=None):
        super().__init__(cfg)
        self.matcher = matcher

    def forward(self, outputs, targets, matches):
        if not bool(getattr(self.cfg, "cascade_matching", False)) or self.matcher is None:
            return super().forward(outputs, targets, matches)
        if "coarse" not in outputs or "final" not in outputs:
            return super().forward(outputs, targets, matches)

        matches_coarse = matches
        matches_final = self.matcher(outputs["final"], targets)
        final_losses = S1Criterion.forward(self, outputs["final"], targets, matches_final)
        seg_loss = self.compute_seg_loss(outputs, targets)
        centerline_loss = self.compute_centerline_loss(outputs, targets)
        coarse_exist = self.compute_exist_loss(outputs["coarse"], matches_coarse)
        coarse_point = self.compute_point_loss(outputs["coarse"], targets, matches_coarse)
        coarse_range = self.compute_range_loss(outputs["coarse"], targets, matches_coarse)
        coarse_line_iou = self.compute_line_iou_loss(outputs["coarse"], targets, matches_coarse)
        coarse = (
            self.cfg.w_exist * coarse_exist
            + self.cfg.w_point * coarse_point
            + self.cfg.w_range * coarse_range
        )
        if self.cfg.w_line_iou > 0:
            coarse = coarse + self.cfg.w_line_iou * coarse_line_iou
        if self.cfg.w_seg > 0:
            final_losses["loss_total"] = final_losses["loss_total"] + self.cfg.w_seg * seg_loss
            final_losses["loss_seg"] = seg_loss
        if self.cfg.w_centerline > 0:
            final_losses["loss_total"] = final_losses["loss_total"] + self.cfg.w_centerline * centerline_loss
            final_losses["loss_centerline"] = centerline_loss
        final_losses["loss_exist_coarse"] = coarse_exist
        final_losses["loss_point_coarse"] = coarse_point
        final_losses["loss_range_coarse"] = coarse_range
        final_losses["loss_line_iou_coarse"] = coarse_line_iou
        final_losses["loss_coarse"] = coarse
        final_losses["loss_total"] = final_losses["loss_total"] + self.cfg.lambda_coarse * coarse
        offset_losses = self.compute_active_offset_losses(outputs, targets, matches_coarse)
        final_losses.update(offset_losses)
        final_losses["loss_total"] = (
            final_losses["loss_total"]
            + self.cfg.w_active_offset_reg * offset_losses["loss_active_offset_reg"]
            + self.cfg.w_active_offset_ce * offset_losses["loss_active_offset_ce"]
        )
        final_losses = self.add_geometry_draft_loss(final_losses, outputs, targets, matches_coarse)
        final_losses["cascade_match_changed_ratio"] = self.compute_match_changed_ratio(
            matches_coarse,
            matches_final,
            outputs["final"]["pred_x_rows"],
        )
        return final_losses

    @staticmethod
    def compute_match_changed_ratio(matches_coarse, matches_final, anchor: torch.Tensor) -> torch.Tensor:
        changed = 0
        total = 0
        for coarse, final in zip(matches_coarse, matches_final):
            coarse_map = {
                int(gt): int(pred)
                for pred, gt in zip(coarse["pred_indices"].detach().cpu().tolist(), coarse["gt_indices"].detach().cpu().tolist())
            }
            final_map = {
                int(gt): int(pred)
                for pred, gt in zip(final["pred_indices"].detach().cpu().tolist(), final["gt_indices"].detach().cpu().tolist())
            }
            for gt_idx, coarse_pred in coarse_map.items():
                if gt_idx not in final_map:
                    continue
                total += 1
                changed += int(coarse_pred != final_map[gt_idx])
        return anchor.new_tensor(float(changed) / float(max(total, 1)))
