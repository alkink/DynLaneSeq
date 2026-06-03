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


class S1Criterion(S0Criterion):
    cfg: S1LossConfig

    def __init__(self, cfg: S1LossConfig | None = None):
        super().__init__(cfg or S1LossConfig())

    def forward(self, outputs, targets, matches):
        base = super().forward(outputs, targets, matches)
        flat_outputs = outputs["final"] if "final" in outputs else outputs
        loss_token = self.compute_token_loss(flat_outputs, targets, matches)
        loss_visibility = self.compute_visibility_loss(flat_outputs, targets, matches)
        base["loss_token"] = loss_token
        base["loss_visibility"] = loss_visibility
        base["loss_total"] = base["loss_total"] + self.cfg.w_token * loss_token + self.cfg.w_visibility * loss_visibility
        return base

    def compute_token_loss(self, outputs, targets, matches) -> torch.Tensor:
        logits = outputs["row_x_logits"]
        total = logits.sum() * 0.0
        count = 0
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
            count += int(valid.item())
        return total / max(count, 1)

    def compute_visibility_loss(self, outputs, targets, matches) -> torch.Tensor:
        logits = outputs.get("row_visibility_logits")
        if logits is None:
            anchor = outputs["row_x_logits"]
            return anchor.sum() * 0.0
        total = logits.sum() * 0.0
        count = 0
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
            count += int(tgt.numel())
        return total / max(count, 1)
