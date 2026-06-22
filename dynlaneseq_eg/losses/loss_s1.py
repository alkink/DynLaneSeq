from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .loss_s0 import LossConfig, S0Criterion


@dataclass
class S1LossConfig(LossConfig):
    w_token: float = 0.5
    token_ignore_index: int = -100
    token_label_smoothing: float = 0.0
    w_visibility: float = 0.0
    visibility_pos_weight: float = 1.0
    w_tangent: float = 0.0
    w_curvature: float = 0.0
    w_delta_l2: float = 0.0


class S1Criterion(S0Criterion):
    cfg: S1LossConfig

    def __init__(self, cfg: S1LossConfig | None = None):
        super().__init__(cfg or S1LossConfig())

    def forward(self, outputs, targets, matches):
        base = super().forward(outputs, targets, matches)
        flat_outputs = outputs["final"] if "final" in outputs else outputs
        loss_token = self.compute_token_loss(flat_outputs, targets, matches)
        loss_visibility = (
            self.compute_visibility_loss(flat_outputs, targets, matches)
            if self.cfg.w_visibility != 0
            else flat_outputs["row_x_logits"].sum() * 0.0
        )
        loss_tangent = (
            self.compute_tangent_loss(flat_outputs, targets, matches)
            if self.cfg.w_tangent != 0
            else flat_outputs["row_x_logits"].sum() * 0.0
        )
        loss_curvature = (
            self.compute_curvature_loss(flat_outputs, targets, matches)
            if self.cfg.w_curvature != 0
            else flat_outputs["row_x_logits"].sum() * 0.0
        )
        loss_delta_l2 = (
            self.compute_delta_l2_loss(flat_outputs, targets, matches)
            if self.cfg.w_delta_l2 != 0
            else flat_outputs["row_x_logits"].sum() * 0.0
        )
        base["loss_token"] = loss_token
        base["loss_visibility"] = loss_visibility
        base["loss_tangent"] = loss_tangent
        base["loss_curvature"] = loss_curvature
        base["loss_delta_l2"] = loss_delta_l2
        base["loss_total"] = (
            base["loss_total"]
            + self.cfg.w_token * loss_token
            + self.cfg.w_visibility * loss_visibility
            + self.cfg.w_tangent * loss_tangent
            + self.cfg.w_curvature * loss_curvature
            + self.cfg.w_delta_l2 * loss_delta_l2
        )
        return base

    def compute_token_loss(self, outputs, targets, matches) -> torch.Tensor:
        logits = outputs["row_x_logits"]
        total = logits.sum() * 0.0
        count = logits.new_tensor(0.0)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(logits.device)
            gt_idx = match["gt_indices"].to(logits.device)
            if pred_idx.numel() == 0:
                continue
            pred = logits[bi, pred_idx]
            tgt = targets[bi]["x_bins"].to(logits.device)[gt_idx]
            loss = F.cross_entropy(
                pred.reshape(-1, pred.shape[-1]),
                tgt.reshape(-1),
                ignore_index=self.cfg.token_ignore_index,
                reduction="sum",
                label_smoothing=float(self.cfg.token_label_smoothing),
            )
            valid = (tgt != self.cfg.token_ignore_index).sum()
            total = total + loss
            count = count + valid.to(device=logits.device, dtype=logits.dtype)
        return total / count.clamp_min(1.0)

    def compute_visibility_loss(self, outputs, targets, matches) -> torch.Tensor:
        logits = outputs.get("row_visibility_logits")
        if logits is None:
            anchor = outputs["row_x_logits"]
            return anchor.sum() * 0.0
        total = logits.sum() * 0.0
        count = logits.new_tensor(0.0)
        pos_weight = None
        if self.cfg.visibility_pos_weight != 1.0:
            pos_weight = torch.tensor(
                [float(self.cfg.visibility_pos_weight)],
                device=logits.device,
                dtype=logits.dtype,
            )
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(logits.device)
            gt_idx = match["gt_indices"].to(logits.device)
            if pred_idx.numel() == 0:
                continue
            pred = logits[bi, pred_idx]
            tgt = targets[bi]["valid_mask"].to(logits.device, dtype=logits.dtype)[gt_idx]
            total = total + F.binary_cross_entropy_with_logits(
                pred,
                tgt,
                pos_weight=pos_weight,
                reduction="sum",
            )
            count = count + float(tgt.numel())
        return total / count.clamp_min(1.0)

    def compute_tangent_loss(self, outputs, targets, matches) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"]
        total = pred_x.sum() * 0.0
        count = pred_x.new_tensor(0.0)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(pred_x.device, dtype=pred_x.dtype)[gt_idx]
            mask = targets[bi]["valid_mask"].to(pred_x.device)[gt_idx].bool()
            pair_mask = mask[:, 1:] & mask[:, :-1]
            if not pair_mask.any():
                continue
            pred = pred_x[bi, pred_idx]
            pred_tangent = (pred[:, 1:] - pred[:, :-1]) / float(self.cfg.input_w)
            gt_tangent = (gt_x[:, 1:] - gt_x[:, :-1]) / float(self.cfg.input_w)
            loss = F.smooth_l1_loss(
                pred_tangent,
                gt_tangent,
                beta=self.cfg.smooth_l1_beta,
                reduction="none",
            )
            valid = pair_mask.to(dtype=pred_x.dtype)
            total = total + (loss * valid).sum()
            count = count + valid.sum()
        return total / count.clamp_min(1.0)

    def compute_curvature_loss(self, outputs, targets, matches) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"]
        total = pred_x.sum() * 0.0
        count = pred_x.new_tensor(0.0)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(pred_x.device, dtype=pred_x.dtype)[gt_idx]
            mask = targets[bi]["valid_mask"].to(pred_x.device)[gt_idx].bool()
            triplet_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
            if not triplet_mask.any():
                continue
            pred = pred_x[bi, pred_idx]
            pred_curv = (pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]) / float(self.cfg.input_w)
            gt_curv = (gt_x[:, 2:] - 2.0 * gt_x[:, 1:-1] + gt_x[:, :-2]) / float(self.cfg.input_w)
            loss = F.smooth_l1_loss(
                pred_curv,
                gt_curv,
                beta=self.cfg.smooth_l1_beta,
                reduction="none",
            )
            valid = triplet_mask.to(dtype=pred_x.dtype)
            total = total + (loss * valid).sum()
            count = count + valid.sum()
        return total / count.clamp_min(1.0)

    def compute_delta_l2_loss(self, outputs, targets, matches) -> torch.Tensor:
        pred_x = outputs.get("pred_x_rows")
        coarse = outputs.get("coarse")
        if pred_x is not None and isinstance(coarse, dict) and coarse.get("pred_x_rows") is not None:
            delta_x = (pred_x - coarse["pred_x_rows"].detach()) / float(self.cfg.input_w)
            total = delta_x.sum() * 0.0
            count = delta_x.new_tensor(0.0)
            for bi, match in enumerate(matches):
                pred_idx = match["pred_indices"].to(delta_x.device)
                gt_idx = match["gt_indices"].to(delta_x.device)
                if pred_idx.numel() == 0:
                    continue
                mask = targets[bi]["valid_mask"].to(delta_x.device)[gt_idx].bool()
                pred_delta = delta_x[bi, pred_idx]
                valid = mask.to(dtype=delta_x.dtype)
                total = total + (pred_delta.pow(2) * valid).sum()
                count = count + valid.sum()
            return total / count.clamp_min(1.0)

        delta = outputs.get("row_delta_logits")
        if delta is None:
            return outputs["row_x_logits"].sum() * 0.0
        total = delta.sum() * 0.0
        count = delta.new_tensor(0.0)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(delta.device)
            gt_idx = match["gt_indices"].to(delta.device)
            if pred_idx.numel() == 0:
                continue
            mask = targets[bi]["valid_mask"].to(delta.device)[gt_idx].bool()
            pred_delta = delta[bi, pred_idx]
            valid = mask.unsqueeze(-1).to(dtype=delta.dtype)
            total = total + (pred_delta.pow(2) * valid).sum()
            count = count + valid.sum() * float(pred_delta.shape[-1])
        return total / count.clamp_min(1.0)
