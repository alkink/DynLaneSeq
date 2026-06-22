from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .loss_s1 import S1Criterion, S1LossConfig


@dataclass
class S2LossConfig(S1LossConfig):
    lambda_coarse: float = 0.5
    coarse_anchor_mode: str = "legacy"
    coarse_dense_anchor: bool = False
    w_active_offset_reg: float = 0.0
    w_active_offset_ce: float = 0.0
    active_offset_max: float = 32.0
    active_offset_label_smoothing: float = 0.0
    active_offset_reg_beta_px: float = 0.0
    active_offset_reg_normalizer_px: float = 0.0
    active_offset_non_center_weight: float = 1.0
    active_offset_soft_target_sigma_px: float = 0.0
    active_offset_fine_target_sigma_px: float = 0.0
    active_offset_fine_weight: float = 1.0
    active_offset_magnitude_weight: float = 0.0
    w_active_offset_tangent: float = 0.0
    w_active_gate: float = 0.0
    active_gate_error_threshold_px: float = 4.0
    active_gate_pos_weight: float = 1.0
    cascade_matching: bool = False


class S2Criterion(S1Criterion):
    cfg: S2LossConfig

    def __init__(self, cfg: S2LossConfig | None = None):
        super().__init__(cfg or S2LossConfig())
        mode = str(self.cfg.coarse_anchor_mode).lower()
        if mode not in {"legacy", "full"}:
            raise ValueError(f"Unsupported coarse_anchor_mode: {self.cfg.coarse_anchor_mode}")

    def forward(self, outputs, targets, matches):
        if str(self.cfg.coarse_anchor_mode).lower() == "full":
            return self._forward_with_full_coarse_anchor(outputs, targets, matches)

        final_losses = super().forward(outputs["final"], targets, matches)
        seg_loss = self.compute_seg_loss(outputs, targets)
        centerline_loss = self.compute_centerline_loss(outputs, targets)
        coarse_point = self.compute_point_loss(outputs["coarse"], targets, matches)
        coarse_range = self.compute_range_loss(outputs["coarse"], targets, matches)
        coarse_line_iou = self.compute_line_iou_loss(outputs["coarse"], targets, matches)
        coarse = coarse_point + 0.5 * coarse_range
        if self.cfg.w_line_iou > 0:
            coarse = coarse + self.cfg.w_line_iou * coarse_line_iou
        if self.cfg.w_seg > 0:
            final_losses["loss_total"] = final_losses["loss_total"] + self.cfg.w_seg * seg_loss
            final_losses["loss_seg"] = seg_loss
        if self.cfg.w_centerline > 0:
            final_losses["loss_total"] = final_losses["loss_total"] + self.cfg.w_centerline * centerline_loss
            final_losses["loss_centerline"] = centerline_loss
        final_losses["loss_point_coarse"] = coarse_point
        final_losses["loss_range_coarse"] = coarse_range
        final_losses["loss_line_iou_coarse"] = coarse_line_iou
        final_losses["loss_coarse"] = coarse
        final_losses["loss_total"] = final_losses["loss_total"] + self.cfg.lambda_coarse * coarse
        offset_losses = self.compute_active_offset_losses(outputs, targets, matches)
        final_losses.update(offset_losses)
        gate_loss = self.compute_active_gate_loss(outputs, targets, matches)
        final_losses["loss_active_gate"] = gate_loss
        final_losses["loss_total"] = (
            final_losses["loss_total"]
            + self.cfg.w_active_offset_reg * offset_losses["loss_active_offset_reg"]
            + self.cfg.w_active_offset_ce * offset_losses["loss_active_offset_ce"]
            + self.cfg.w_active_offset_tangent * offset_losses["loss_active_offset_tangent"]
            + self.cfg.w_active_gate * gate_loss
        )
        return self.add_geometry_draft_loss(final_losses, outputs, targets, matches)

    def _forward_with_full_coarse_anchor(self, outputs, targets, matches):
        # S1Criterion delegates the wrapped output to S0Criterion, which applies
        # the complete weighted coarse objective when lambda_coarse is enabled.
        losses = super().forward(outputs, targets, matches)

        if self.cfg.coarse_dense_anchor and self.cfg.lambda_coarse > 0:
            coarse_dense_outputs = {
                "seg_logits": outputs.get("coarse_seg_logits"),
                "centerline_logits": outputs.get("coarse_centerline_logits"),
                "exist_logits": outputs["coarse"]["exist_logits"],
            }
            for scale_name in self.cfg.seg_extra_weights:
                coarse_dense_outputs[f"seg_logits_{scale_name}"] = outputs.get(
                    f"coarse_seg_logits_{scale_name}"
                )
            coarse_seg = (
                self.compute_seg_loss(coarse_dense_outputs, targets)
                if self.cfg.w_seg != 0
                else outputs["coarse"]["pred_x_rows"].sum() * 0.0
            )
            coarse_centerline = (
                self.compute_centerline_loss(coarse_dense_outputs, targets)
                if self.cfg.w_centerline != 0
                else outputs["coarse"]["pred_x_rows"].sum() * 0.0
            )
            coarse_dense = self.cfg.w_seg * coarse_seg + self.cfg.w_centerline * coarse_centerline
            losses["loss_seg_coarse"] = coarse_seg
            losses["loss_centerline_coarse"] = coarse_centerline
            losses["loss_coarse_dense"] = coarse_dense
            losses["loss_coarse_total"] = losses["loss_coarse_total"] + coarse_dense
            losses["loss_total"] = losses["loss_total"] + self.cfg.lambda_coarse * coarse_dense

        offset_losses = self.compute_active_offset_losses(outputs, targets, matches)
        losses.update(offset_losses)
        gate_loss = self.compute_active_gate_loss(outputs, targets, matches)
        losses["loss_active_gate"] = gate_loss
        losses["loss_total"] = (
            losses["loss_total"]
            + self.cfg.w_active_offset_reg * offset_losses["loss_active_offset_reg"]
            + self.cfg.w_active_offset_ce * offset_losses["loss_active_offset_ce"]
            + self.cfg.w_active_offset_tangent * offset_losses["loss_active_offset_tangent"]
            + self.cfg.w_active_gate * gate_loss
        )
        return losses

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
                "loss_active_offset_tangent": zero,
                "active_offset_target_clamp_ratio": zero.detach(),
                "active_offset_non_center_ratio": zero.detach(),
            }

        def compute_stage(
            stage_logits: torch.Tensor,
            stage_pred: torch.Tensor,
            stage_center: torch.Tensor,
            stage_offsets: torch.Tensor,
            offset_max: float,
            sigma_px: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            stage_logits = stage_logits.float()
            stage_pred = stage_pred.float()
            stage_center = stage_center.float()
            stage_offsets = stage_offsets.to(device=stage_logits.device, dtype=stage_pred.dtype).float()
            if offset_max <= 0:
                offset_max = float(stage_offsets.abs().max().item())
            total_reg = stage_logits.sum() * 0.0
            total_ce = stage_logits.sum() * 0.0
            count = stage_logits.new_tensor(0.0)
            clamp_count = stage_logits.new_tensor(0.0)
            non_center_count = stage_logits.new_tensor(0.0)
            center_idx = int(stage_offsets.abs().argmin().item())
            for bi, match in enumerate(matches):
                pred_idx = match["pred_indices"].to(stage_logits.device)
                gt_idx = match["gt_indices"].to(stage_logits.device)
                if pred_idx.numel() == 0:
                    continue
                gt_x = targets[bi]["x_rows"].to(stage_logits.device, dtype=stage_pred.dtype)[gt_idx]
                mask = targets[bi]["valid_mask"].to(stage_logits.device)[gt_idx].bool()
                pred = stage_pred[bi, pred_idx]
                center = stage_center[bi, pred_idx]
                lane_logits = stage_logits[bi, pred_idx]
                raw_delta = gt_x - center
                target_delta = raw_delta.clamp(min=-offset_max, max=offset_max)
                valid = mask & torch.isfinite(target_delta) & torch.isfinite(pred)
                valid_f = valid.to(dtype=pred.dtype)
                safe_target_delta = torch.where(valid, target_delta, pred.detach())
                reg_normalizer = float(self.cfg.active_offset_reg_normalizer_px)
                if reg_normalizer <= 0:
                    reg_normalizer = max(offset_max, 1.0)
                reg = F.smooth_l1_loss(
                    pred,
                    safe_target_delta,
                    beta=max(float(self.cfg.active_offset_reg_beta_px), 1e-3),
                    reduction="none",
                ) / reg_normalizer
                target_idx = (target_delta.unsqueeze(-1) - stage_offsets.view(1, 1, -1)).abs().argmin(dim=-1)
                if sigma_px > 0:
                    distance = stage_offsets.view(1, 1, -1) - target_delta.unsqueeze(-1)
                    target_prob = torch.softmax(-0.5 * distance.square() / (sigma_px * sigma_px), dim=-1)
                    smoothing = float(self.cfg.active_offset_label_smoothing)
                    if smoothing > 0:
                        target_prob = (1.0 - smoothing) * target_prob + smoothing / float(stage_offsets.numel())
                    ce = -(target_prob * F.log_softmax(lane_logits, dim=-1)).sum(dim=-1)
                else:
                    ce = F.cross_entropy(
                        lane_logits.reshape(-1, lane_logits.shape[-1]),
                        target_idx.reshape(-1),
                        reduction="none",
                        label_smoothing=float(self.cfg.active_offset_label_smoothing),
                    ).view_as(target_idx)
                sample_weight = torch.ones_like(ce)
                if self.cfg.active_offset_non_center_weight != 1.0:
                    sample_weight = sample_weight * torch.where(
                        target_idx == center_idx,
                        torch.ones_like(ce),
                        torch.full_like(ce, float(self.cfg.active_offset_non_center_weight)),
                    )
                if self.cfg.active_offset_magnitude_weight > 0:
                    magnitude = (raw_delta.abs() / max(offset_max, 1.0)).clamp(0.0, 1.0)
                    sample_weight = sample_weight * (1.0 + float(self.cfg.active_offset_magnitude_weight) * magnitude)
                total_reg = total_reg + (reg * sample_weight * valid_f).sum()
                total_ce = total_ce + (ce * sample_weight * valid_f).sum()
                count = count + (sample_weight * valid_f).sum()
                clamp_count = clamp_count + ((raw_delta.abs() > offset_max) & valid).to(stage_logits.dtype).sum()
                non_center_count = non_center_count + ((target_idx != center_idx) & valid).to(stage_logits.dtype).sum()
            denom = count.clamp_min(1.0)
            return total_reg / denom, total_ce / denom, clamp_count, non_center_count, denom

        coarse_logits = evidence.get("active_coarse_offset_logits")
        fine_logits = evidence.get("active_fine_offset_logits")
        if coarse_logits is not None and fine_logits is not None:
            coarse_result = compute_stage(
                coarse_logits,
                evidence["active_coarse_pred_delta_x_rows"],
                center_x,
                evidence["active_coarse_offsets_px"],
                float(self.cfg.active_offset_max),
                float(self.cfg.active_offset_soft_target_sigma_px),
            )
            fine_offsets = evidence["active_fine_offsets_px"]
            fine_max = float(fine_offsets.detach().abs().max().item())
            fine_result = compute_stage(
                fine_logits,
                evidence["active_fine_pred_delta_x_rows"],
                evidence["active_fine_center_x_rows"],
                fine_offsets,
                fine_max,
                float(self.cfg.active_offset_fine_target_sigma_px),
            )
            fine_weight = float(self.cfg.active_offset_fine_weight)
            normalizer = max(1.0 + fine_weight, 1e-6)
            loss_reg = (coarse_result[0] + fine_weight * fine_result[0]) / normalizer
            loss_ce = (coarse_result[1] + fine_weight * fine_result[1]) / normalizer
            clamp_count, non_center_count, denom = coarse_result[2], coarse_result[3], coarse_result[4]
        else:
            result = compute_stage(
                logits,
                pred_delta,
                center_x,
                offsets,
                float(self.cfg.active_offset_max),
                float(self.cfg.active_offset_soft_target_sigma_px),
            )
            loss_reg, loss_ce, clamp_count, non_center_count, denom = result

        total_tangent = pred_delta.sum() * 0.0
        tangent_count = pred_delta.new_tensor(0.0)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_delta.device)
            gt_idx = match["gt_indices"].to(pred_delta.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(pred_delta.device, dtype=pred_delta.dtype)[gt_idx]
            valid = targets[bi]["valid_mask"].to(pred_delta.device)[gt_idx].bool()
            pred = pred_delta[bi, pred_idx]
            target = gt_x - center_x[bi, pred_idx]
            pair_valid = valid[:, 1:] & valid[:, :-1] & torch.isfinite(target[:, 1:]) & torch.isfinite(target[:, :-1])
            pred_tangent = pred[:, 1:] - pred[:, :-1]
            target_tangent = target[:, 1:] - target[:, :-1]
            tangent = F.smooth_l1_loss(pred_tangent, target_tangent, beta=2.0, reduction="none") / 32.0
            total_tangent = total_tangent + (tangent * pair_valid.to(tangent.dtype)).sum()
            tangent_count = tangent_count + pair_valid.to(tangent.dtype).sum()
        loss_tangent = total_tangent / tangent_count.clamp_min(1.0)
        return {
            "loss_active_offset_reg": loss_reg,
            "loss_active_offset_ce": loss_ce,
            "loss_active_offset_tangent": loss_tangent,
            "active_offset_target_clamp_ratio": (clamp_count / denom).detach(),
            "active_offset_non_center_ratio": (non_center_count / denom).detach(),
        }

    def compute_active_gate_loss(self, outputs, targets, matches):
        evidence = outputs.get("evidence", {}) if isinstance(outputs, dict) else {}
        gate_logits = evidence.get("active_gate_logits")
        center_x = evidence.get("active_center_x_rows")
        anchor = outputs["final"]["pred_x_rows"]
        zero = anchor.sum() * 0.0
        if gate_logits is None or center_x is None:
            return zero
        total = gate_logits.sum() * 0.0
        count = gate_logits.new_tensor(0.0)
        threshold = float(self.cfg.active_gate_error_threshold_px)
        pos_weight = gate_logits.new_tensor(float(self.cfg.active_gate_pos_weight))
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(gate_logits.device)
            gt_idx = match["gt_indices"].to(gate_logits.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(gate_logits.device, dtype=center_x.dtype)[gt_idx]
            valid = targets[bi]["valid_mask"].to(gate_logits.device)[gt_idx].bool()
            error = (gt_x - center_x[bi, pred_idx]).abs()
            target_open = (error > threshold).to(gate_logits.dtype)
            lane_logits = gate_logits[bi, pred_idx].float()
            loss = F.binary_cross_entropy_with_logits(
                lane_logits,
                target_open.float(),
                pos_weight=pos_weight.float(),
                reduction="none",
            )
            valid_f = valid.to(loss.dtype)
            total = total + (loss * valid_f).sum()
            count = count + valid_f.sum()
        return total / count.clamp_min(1.0)
