from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .loss_s1 import S1Criterion, S1LossConfig


@dataclass
class S2LossConfig(S1LossConfig):
    lambda_coarse: float = 0.5
    w_active_offset_reg: float = 0.0
    w_active_offset_ce: float = 0.0
    active_offset_max: float = 32.0
    active_offset_label_smoothing: float = 0.0


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
        offset_losses = self.compute_active_offset_losses(outputs, targets, matches)
        final_losses.update(offset_losses)
        final_losses["loss_total"] = (
            final_losses["loss_total"]
            + self.cfg.w_active_offset_reg * offset_losses["loss_active_offset_reg"]
            + self.cfg.w_active_offset_ce * offset_losses["loss_active_offset_ce"]
        )
        return final_losses

    def compute_active_offset_losses(self, outputs, targets, matches):
        evidence = outputs.get("evidence", {}) if isinstance(outputs, dict) else {}
        logits = evidence.get("active_offset_logits")
        pred_delta = evidence.get("active_pred_delta_x_rows")
        center_x = evidence.get("active_center_x_rows")
        offsets = evidence.get("active_offsets_px")
        anchor = outputs["final"]["pred_x_rows"] if "final" in outputs else next(
            value for value in outputs.values() if isinstance(value, torch.Tensor)
        )
        zero = anchor.sum() * 0.0
        if logits is None or pred_delta is None or center_x is None or offsets is None:
            return {
                "loss_active_offset_reg": zero,
                "loss_active_offset_ce": zero,
                "active_offset_target_clamp_ratio": zero.detach(),
            }

        logits = logits.float()
        pred_delta = pred_delta.float()
        center_x = center_x.float()
        offsets = offsets.to(device=logits.device, dtype=pred_delta.dtype).float()
        offset_max = float(self.cfg.active_offset_max)
        if offset_max <= 0:
            offset_max = float(offsets.abs().max().item())
        total_reg = logits.sum() * 0.0
        total_ce = logits.sum() * 0.0
        count = 0
        clamp_count = 0
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(logits.device)
            gt_idx = match["gt_indices"].to(logits.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(logits.device, dtype=pred_delta.dtype)[gt_idx]
            mask = targets[bi]["valid_mask"].to(logits.device)[gt_idx].bool()
            pred = pred_delta[bi, pred_idx]
            center = center_x[bi, pred_idx]
            lane_logits = logits[bi, pred_idx]
            raw_delta = gt_x - center
            target_delta = raw_delta.clamp(min=-offset_max, max=offset_max)
            valid = mask & torch.isfinite(target_delta) & torch.isfinite(pred)
            if not valid.any():
                continue
            target_idx = (target_delta.unsqueeze(-1) - offsets.view(1, 1, -1)).abs().argmin(dim=-1)
            total_reg = total_reg + F.smooth_l1_loss(
                pred[valid] / float(self.cfg.input_w),
                target_delta[valid] / float(self.cfg.input_w),
                beta=self.cfg.smooth_l1_beta,
                reduction="sum",
            )
            total_ce = total_ce + F.cross_entropy(
                lane_logits[valid],
                target_idx[valid],
                reduction="sum",
                label_smoothing=float(self.cfg.active_offset_label_smoothing),
            )
            valid_count = int(valid.sum().item())
            count += valid_count
            clamp_count += int((raw_delta[valid].abs() > offset_max).sum().item())
        denom = max(count, 1)
        clamp_ratio = torch.tensor(float(clamp_count) / float(denom), device=logits.device, dtype=logits.dtype)
        return {
            "loss_active_offset_reg": total_reg / denom,
            "loss_active_offset_ce": total_ce / denom,
            "active_offset_target_clamp_ratio": clamp_ratio.detach(),
        }
