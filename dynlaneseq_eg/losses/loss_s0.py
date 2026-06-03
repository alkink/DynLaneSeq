from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from dynlaneseq_eg.modeling.common import sort_range_norm


@dataclass
class LossConfig:
    w_exist: float = 2.0
    w_point: float = 5.0
    w_range: float = 1.0
    w_smooth: float = 0.0
    smooth_l1_beta: float = 0.01
    input_w: int = 800
    input_h: int = 288
    no_lane_weight: float = 1.0
    smoothness_contiguous: bool = True
    w_line_iou: float = 0.0
    line_iou_radius: float = 15.0
    w_seg: float = 0.0
    seg_pos_weight: float = 1.0
    w_quality: float = 0.0


class S0Criterion(nn.Module):
    def __init__(self, cfg: LossConfig | None = None):
        super().__init__()
        self.cfg = cfg or LossConfig()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        raw_outputs = outputs
        if "final" in outputs:
            outputs = outputs["final"]
        elif "stage2" in outputs:
            outputs = outputs["stage2"]
        loss_exist = self.compute_exist_loss(outputs, matches)
        loss_point = self.compute_point_loss(outputs, targets, matches)
        loss_range = self.compute_range_loss(outputs, targets, matches)
        loss_smooth = self.compute_smoothness_loss(outputs, targets, matches)
        loss_line_iou = self.compute_line_iou_loss(outputs, targets, matches)
        loss_seg = self.compute_seg_loss(raw_outputs, targets)
        loss_quality = self.compute_quality_loss(outputs, targets, matches)
        total = (
            self.cfg.w_exist * loss_exist
            + self.cfg.w_point * loss_point
            + self.cfg.w_range * loss_range
            + self.cfg.w_smooth * loss_smooth
            + self.cfg.w_line_iou * loss_line_iou
            + self.cfg.w_seg * loss_seg
            + self.cfg.w_quality * loss_quality
        )
        return {
            "loss_total": total,
            "loss_exist": loss_exist,
            "loss_point": loss_point,
            "loss_range": loss_range,
            "loss_smooth": loss_smooth,
            "loss_line_iou": loss_line_iou,
            "loss_seg": loss_seg,
            "loss_quality": loss_quality,
        }

    def compute_exist_loss(self, outputs: dict[str, torch.Tensor], matches: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        logits = outputs["exist_logits"]
        b, n, _ = logits.shape
        target = torch.ones((b, n), dtype=torch.long, device=logits.device)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(logits.device)
            if pred_idx.numel() > 0:
                target[bi, pred_idx] = 0
        weight = torch.tensor([1.0, self.cfg.no_lane_weight], device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits.view(b * n, 2), target.view(b * n), weight=weight)

    def compute_point_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"]
        total = pred_x.sum() * 0.0
        count = 0
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(pred_x.device)[gt_idx]
            mask = targets[bi]["valid_mask"].to(pred_x.device)[gt_idx].bool()
            pred = pred_x[bi, pred_idx] / float(self.cfg.input_w)
            gt = gt_x / float(self.cfg.input_w)
            if mask.any():
                total = total + F.smooth_l1_loss(
                    pred[mask],
                    gt[mask],
                    beta=self.cfg.smooth_l1_beta,
                    reduction="sum",
                )
                count += int(mask.sum().item())
        return total / max(count, 1)

    def compute_line_iou_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"]
        total = pred_x.sum() * 0.0
        count = 0
        radius = float(self.cfg.line_iou_radius)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(pred_x.device)[gt_idx]
            mask = targets[bi]["valid_mask"].to(pred_x.device)[gt_idx].bool()
            pred = pred_x[bi, pred_idx]
            for pred_lane, gt_lane, lane_mask in zip(pred, gt_x, mask):
                if lane_mask.any():
                    px1 = pred_lane[lane_mask] - radius
                    px2 = pred_lane[lane_mask] + radius
                    gx1 = gt_lane[lane_mask] - radius
                    gx2 = gt_lane[lane_mask] + radius
                    overlap = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0.0)
                    union = (4.0 * radius - overlap).clamp(min=1e-6)
                    iou = overlap / union
                    enclosing = (torch.maximum(px2, gx2) - torch.minimum(px1, gx1)).clamp(min=1e-6)
                    giou = iou - (enclosing - union) / enclosing
                    total = total + (1.0 - giou).mean()
                    count += 1
        return total / max(count, 1)

    def compute_quality_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        logits = outputs.get("quality_logits")
        if logits is None:
            return outputs["pred_x_rows"].sum() * 0.0
        quality_logits = logits.float()
        target_quality = torch.zeros_like(quality_logits)
        pred_x = outputs.get("quality_pred_x_rows", outputs["pred_x_rows"]).float()
        radius = float(self.cfg.line_iou_radius)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(pred_x.device, dtype=pred_x.dtype)[gt_idx]
            mask = targets[bi]["valid_mask"].to(pred_x.device)[gt_idx].bool()
            pred = pred_x[bi, pred_idx]
            qualities = []
            for pred_lane, gt_lane, lane_mask in zip(pred, gt_x, mask):
                if not lane_mask.any():
                    qualities.append(pred_lane.sum() * 0.0)
                    continue
                px1 = pred_lane[lane_mask] - radius
                px2 = pred_lane[lane_mask] + radius
                gx1 = gt_lane[lane_mask] - radius
                gx2 = gt_lane[lane_mask] + radius
                overlap = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0.0)
                union = (4.0 * radius - overlap).clamp(min=1e-6)
                qualities.append((overlap / union).mean())
            target_quality[bi, pred_idx] = torch.stack(qualities).detach().to(dtype=target_quality.dtype)
        return F.binary_cross_entropy_with_logits(quality_logits, target_quality)

    def compute_seg_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        seg_logits = outputs.get("seg_logits")
        if seg_logits is None:
            flat = outputs.get("final") or outputs.get("stage2") or {}
            seg_logits = flat.get("seg_logits") if isinstance(flat, dict) else None
        if seg_logits is None:
            anchor = outputs.get("exist_logits")
            if anchor is None:
                anchor = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
            return anchor.sum() * 0.0
        valid_indices = []
        seg_targets = []
        for idx, target in enumerate(targets):
            if "seg_mask" not in target:
                return seg_logits.sum() * 0.0
            seg_mask = target["seg_mask"]
            seg_valid = target.get("seg_valid", True)
            if isinstance(seg_valid, torch.Tensor):
                seg_valid = bool(seg_valid.detach().cpu().item())
            if not seg_valid:
                continue
            has_lane = int(target["x_rows"].shape[0]) > 0
            if has_lane and float(seg_mask.detach().max().cpu()) == 0.0:
                continue
            valid_indices.append(idx)
            seg_targets.append(seg_mask.to(seg_logits.device, dtype=seg_logits.dtype))
        if not valid_indices:
            return seg_logits.sum() * 0.0
        seg_logits = seg_logits[valid_indices]
        seg_target = torch.stack(seg_targets, dim=0)
        if seg_target.shape[-2:] != seg_logits.shape[-2:]:
            seg_target = F.interpolate(seg_target, size=seg_logits.shape[-2:], mode="nearest")
        pos_weight = None
        if self.cfg.seg_pos_weight != 1.0:
            pos_weight = torch.tensor([self.cfg.seg_pos_weight], device=seg_logits.device, dtype=seg_logits.dtype)
        return F.binary_cross_entropy_with_logits(seg_logits, seg_target, pos_weight=pos_weight)

    def compute_range_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_range = sort_range_norm(outputs["range_norm"])
        total = pred_range.sum() * 0.0
        count = 0
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_range.device)
            gt_idx = match["gt_indices"].to(pred_range.device)
            if pred_idx.numel() == 0:
                continue
            gt_range = targets[bi]["range_y"].to(pred_range.device)[gt_idx] / float(self.cfg.input_h)
            pred = pred_range[bi, pred_idx]
            total = total + F.smooth_l1_loss(pred, gt_range, beta=self.cfg.smooth_l1_beta, reduction="sum")
            count += int(pred.numel())
        return total / max(count, 1)

    def compute_smoothness_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"]
        total = pred_x.sum() * 0.0
        count = 0
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            for pidx, gidx in zip(pred_idx.tolist(), gt_idx.tolist()):
                mask = targets[bi]["valid_mask"].to(pred_x.device)[gidx].bool()
                if not self.cfg.smoothness_contiguous:
                    if int(mask.sum().item()) >= 3:
                        lane_x = pred_x[bi, pidx][mask]
                        d2 = lane_x[2:] - 2.0 * lane_x[1:-1] + lane_x[:-2]
                        total = total + (d2 / float(self.cfg.input_w)).abs().mean()
                        count += 1
                    continue
                triplet_mask = mask[2:] & mask[1:-1] & mask[:-2]
                if triplet_mask.any():
                    lane_x = pred_x[bi, pidx]
                    d2 = lane_x[2:] - 2.0 * lane_x[1:-1] + lane_x[:-2]
                    total = total + (d2[triplet_mask] / float(self.cfg.input_w)).abs().sum()
                    count += int(triplet_mask.sum().item())
        return total / max(count, 1)
