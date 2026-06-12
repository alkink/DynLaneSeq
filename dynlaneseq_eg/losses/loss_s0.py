from __future__ import annotations

from dataclasses import dataclass, field

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
    exist_loss_type: str = "ce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    smoothness_contiguous: bool = True
    w_line_iou: float = 0.0
    line_iou_radius: float = 15.0
    w_seg: float = 0.0
    seg_pos_weight: float = 1.0
    seg_extra_weights: dict[str, float] = field(default_factory=dict)
    w_quality: float = 0.0
    w_centerline: float = 0.0
    centerline_sigma_bins: float = 1.5
    centerline_pos_weight: float = 1.0
    w_dynamic_proposal_heatmap: float = 0.0
    w_dynamic_proposal_x: float = 0.0
    w_dynamic_proposal_range: float = 0.0
    dynamic_proposal_sigma_bins: float = 1.5
    dynamic_proposal_seed_radius_bins: int = 2
    dynamic_proposal_heatmap_pos_weight: float = 1.0
    lambda_coarse: float = 0.0
    lambda_geometry_draft: float = 0.0


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
        loss_centerline = self.compute_centerline_loss(raw_outputs, targets)
        dynamic_proposal_losses = self.compute_dynamic_proposal_losses(raw_outputs, targets)
        total = (
            self.cfg.w_exist * loss_exist
            + self.cfg.w_point * loss_point
            + self.cfg.w_range * loss_range
            + self.cfg.w_smooth * loss_smooth
            + self.cfg.w_line_iou * loss_line_iou
            + self.cfg.w_seg * loss_seg
            + self.cfg.w_quality * loss_quality
            + self.cfg.w_centerline * loss_centerline
            + self.cfg.w_dynamic_proposal_heatmap * dynamic_proposal_losses["heatmap"]
            + self.cfg.w_dynamic_proposal_x * dynamic_proposal_losses["x"]
            + self.cfg.w_dynamic_proposal_range * dynamic_proposal_losses["range"]
        )
        out = {
            "loss_total": total,
            "loss_exist": loss_exist,
            "loss_point": loss_point,
            "loss_range": loss_range,
            "loss_smooth": loss_smooth,
            "loss_line_iou": loss_line_iou,
            "loss_seg": loss_seg,
            "loss_quality": loss_quality,
            "loss_centerline": loss_centerline,
            "loss_dynamic_proposal_heatmap": dynamic_proposal_losses["heatmap"],
            "loss_dynamic_proposal_x": dynamic_proposal_losses["x"],
            "loss_dynamic_proposal_range": dynamic_proposal_losses["range"],
        }
        if self.cfg.lambda_coarse > 0 and isinstance(raw_outputs.get("coarse"), dict):
            coarse = raw_outputs["coarse"]
            coarse_exist = self.compute_exist_loss(coarse, matches)
            coarse_point = self.compute_point_loss(coarse, targets, matches)
            coarse_range = self.compute_range_loss(coarse, targets, matches)
            coarse_smooth = self.compute_smoothness_loss(coarse, targets, matches)
            coarse_line_iou = self.compute_line_iou_loss(coarse, targets, matches)
            coarse_quality = self.compute_quality_loss(coarse, targets, matches)
            coarse_total = (
                self.cfg.w_exist * coarse_exist
                + self.cfg.w_point * coarse_point
                + self.cfg.w_range * coarse_range
                + self.cfg.w_smooth * coarse_smooth
                + self.cfg.w_line_iou * coarse_line_iou
                + self.cfg.w_quality * coarse_quality
            )
            total = total + self.cfg.lambda_coarse * coarse_total
            out.update(
                {
                    "loss_total": total,
                    "loss_coarse_total": coarse_total,
                    "loss_exist_coarse": coarse_exist,
                    "loss_point_coarse": coarse_point,
                    "loss_range_coarse": coarse_range,
                    "loss_smooth_coarse": coarse_smooth,
                    "loss_line_iou_coarse": coarse_line_iou,
                    "loss_quality_coarse": coarse_quality,
                }
            )
        out = self.add_geometry_draft_loss(out, raw_outputs, targets, matches)
        return out

    def add_geometry_draft_loss(
        self,
        losses: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if self.cfg.lambda_geometry_draft <= 0 or not isinstance(outputs.get("s0_geometry_draft"), dict):
            return losses
        draft = outputs["s0_geometry_draft"]
        draft_exist = self.compute_exist_loss(draft, matches)
        draft_point = self.compute_point_loss(draft, targets, matches)
        draft_range = self.compute_range_loss(draft, targets, matches)
        draft_smooth = self.compute_smoothness_loss(draft, targets, matches)
        draft_line_iou = self.compute_line_iou_loss(draft, targets, matches)
        draft_quality = self.compute_quality_loss(draft, targets, matches)
        draft_total = (
            self.cfg.w_exist * draft_exist
            + self.cfg.w_point * draft_point
            + self.cfg.w_range * draft_range
            + self.cfg.w_smooth * draft_smooth
            + self.cfg.w_line_iou * draft_line_iou
            + self.cfg.w_quality * draft_quality
        )
        losses = dict(losses)
        losses["loss_total"] = losses["loss_total"] + self.cfg.lambda_geometry_draft * draft_total
        losses.update(
            {
                "loss_geometry_draft_total": draft_total,
                "loss_exist_geometry_draft": draft_exist,
                "loss_point_geometry_draft": draft_point,
                "loss_range_geometry_draft": draft_range,
                "loss_smooth_geometry_draft": draft_smooth,
                "loss_line_iou_geometry_draft": draft_line_iou,
                "loss_quality_geometry_draft": draft_quality,
            }
        )
        return losses

    def compute_exist_loss(self, outputs: dict[str, torch.Tensor], matches: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        logits = outputs["exist_logits"]
        b, n, _ = logits.shape
        target = torch.ones((b, n), dtype=torch.long, device=logits.device)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(logits.device)
            if pred_idx.numel() > 0:
                target[bi, pred_idx] = 0
        if str(self.cfg.exist_loss_type).lower() == "focal":
            lane_target = (target == 0).to(dtype=logits.dtype)
            lane_logit = logits[..., 0] - logits[..., 1]
            ce = F.binary_cross_entropy_with_logits(lane_logit, lane_target, reduction="none")
            prob = torch.sigmoid(lane_logit)
            p_t = prob * lane_target + (1.0 - prob) * (1.0 - lane_target)
            alpha = float(self.cfg.focal_alpha)
            alpha_t = alpha * lane_target + (1.0 - alpha) * (1.0 - lane_target)
            loss = alpha_t * (1.0 - p_t).pow(float(self.cfg.focal_gamma)) * ce
            return loss.mean()
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
        seg_items: list[tuple[torch.Tensor, float]] = []
        if seg_logits is not None:
            seg_items.append((seg_logits, 1.0))
        for scale_name, weight in self.cfg.seg_extra_weights.items():
            extra_logits = outputs.get(f"seg_logits_{scale_name}")
            if extra_logits is not None and float(weight) != 0.0:
                seg_items.append((extra_logits, float(weight)))
        if not seg_items:
            flat = outputs.get("final") or outputs.get("stage2") or {}
            seg_logits = flat.get("seg_logits") if isinstance(flat, dict) else None
            if seg_logits is not None:
                seg_items.append((seg_logits, 1.0))
        if not seg_items:
            anchor = outputs.get("exist_logits")
            if anchor is None:
                flat = outputs.get("final") or outputs.get("stage2") or outputs.get("coarse") or {}
                anchor = flat.get("exist_logits") if isinstance(flat, dict) else None
            if anchor is None:
                anchor = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
            return anchor.sum() * 0.0
        seg_logits = seg_items[0][0]
        seg_targets = []
        valid_weights = []
        for target in targets:
            if "seg_mask" not in target:
                return seg_logits.sum() * 0.0
            seg_mask = target["seg_mask"].to(seg_logits.device, dtype=seg_logits.dtype)
            seg_valid = target.get("seg_valid", True)
            if isinstance(seg_valid, torch.Tensor):
                valid = seg_valid.to(seg_logits.device, dtype=seg_logits.dtype).reshape(-1)[0]
            else:
                valid = torch.tensor(float(bool(seg_valid)), device=seg_logits.device, dtype=seg_logits.dtype)
            has_lane = int(target["x_rows"].shape[0]) > 0
            if has_lane:
                valid = valid * (seg_mask.detach().amax() > 0).to(dtype=seg_logits.dtype)
            seg_targets.append(seg_mask)
            valid_weights.append(valid)
        seg_target = torch.stack(seg_targets, dim=0)
        sample_weights = torch.stack(valid_weights, dim=0).to(device=seg_logits.device, dtype=seg_logits.dtype)
        sample_denom = sample_weights.sum().clamp_min(1.0)
        pos_weight = None
        if self.cfg.seg_pos_weight != 1.0:
            pos_weight = torch.tensor([self.cfg.seg_pos_weight], device=seg_logits.device, dtype=seg_logits.dtype)
        total = seg_logits.sum() * 0.0
        for logits, weight in seg_items:
            target = seg_target
            if target.shape[-2:] != logits.shape[-2:]:
                target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
            pw = pos_weight
            if pw is not None and pw.dtype != logits.dtype:
                pw = pw.to(dtype=logits.dtype)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
            loss = loss.flatten(1).mean(dim=1)
            total = total + float(weight) * (loss * sample_weights).sum() / sample_denom
        return total

    def compute_centerline_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        logits = outputs.get("centerline_logits")
        if logits is None:
            anchor = outputs.get("exist_logits")
            if anchor is None:
                flat = outputs.get("final") or outputs.get("stage2") or outputs.get("coarse") or {}
                anchor = flat.get("exist_logits") if isinstance(flat, dict) else None
            if anchor is None:
                anchor = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
            return anchor.sum() * 0.0

        b, _, num_rows, x_bins = logits.shape
        device = logits.device
        dtype = logits.dtype
        target_map = torch.zeros((b, 1, num_rows, x_bins), device=device, dtype=dtype)
        grid = torch.arange(x_bins, device=device, dtype=dtype).view(1, 1, x_bins)
        sigma = max(float(self.cfg.centerline_sigma_bins), 1e-3)
        bin_width = float(self.cfg.input_w) / float(x_bins)
        for bi, target in enumerate(targets):
            x_rows = target["x_rows"].to(device=device, dtype=dtype)
            valid_mask = target["valid_mask"].to(device=device).bool()
            if x_rows.numel() == 0:
                continue
            row_count = min(int(x_rows.shape[1]), int(num_rows))
            x_rows = x_rows[:, :row_count]
            valid_mask = valid_mask[:, :row_count]
            centers = (x_rows / bin_width).clamp(min=0.0, max=float(x_bins - 1))
            valid = valid_mask & torch.isfinite(centers) & (x_rows >= 0.0) & (x_rows <= float(self.cfg.input_w))
            if not valid.any():
                continue
            diff = grid - centers.unsqueeze(-1)
            gauss = torch.exp(-0.5 * (diff / sigma).pow(2))
            gauss = gauss * valid.unsqueeze(-1).to(dtype=dtype)
            target_map[bi, 0, :row_count] = gauss.amax(dim=0)
        pos_weight = None
        if self.cfg.centerline_pos_weight != 1.0:
            pos_weight = torch.tensor([self.cfg.centerline_pos_weight], device=device, dtype=dtype)
        return F.binary_cross_entropy_with_logits(logits, target_map, pos_weight=pos_weight)

    def compute_dynamic_proposal_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        dynamic = outputs.get("dynamic_proposals")
        if not isinstance(dynamic, dict) or not isinstance(dynamic.get("dense"), dict):
            anchor = self._zero_anchor(outputs)
            zero = anchor.sum() * 0.0
            return {"heatmap": zero, "x": zero, "range": zero}

        dense = dynamic["dense"]
        heatmap_logits = dense["heatmap_logits"]
        dense_x = dense["x_rows"]
        dense_range = sort_range_norm(dense["range_norm"].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        b, _, feat_h, feat_w = heatmap_logits.shape
        _, num_rows_pred, _, _ = dense_x.shape
        device = heatmap_logits.device
        dtype = heatmap_logits.dtype
        heatmap_target = torch.zeros((b, 1, feat_h, feat_w), device=device, dtype=dtype)
        grid_x = torch.arange(feat_w, device=device, dtype=dtype)
        sigma = max(float(self.cfg.dynamic_proposal_sigma_bins), 1e-3)
        radius = max(int(self.cfg.dynamic_proposal_seed_radius_bins), 0)
        x_loss = dense_x.sum() * 0.0
        range_loss = dense_range.sum() * 0.0
        x_denom = dense_x.new_tensor(0.0)
        range_denom = dense_range.new_tensor(0.0)

        for bi, target in enumerate(targets):
            x_rows = target["x_rows"].to(device=device, dtype=dense_x.dtype)
            valid_mask = target["valid_mask"].to(device=device).bool()
            if x_rows.numel() == 0:
                continue
            lane_count = int(x_rows.shape[0])
            row_count = min(int(x_rows.shape[1]), int(num_rows_pred))
            if row_count <= 0:
                continue
            for lane_idx in range(lane_count):
                lane_x = x_rows[lane_idx, :row_count]
                lane_valid = valid_mask[lane_idx, :row_count]
                finite_valid = lane_valid & torch.isfinite(lane_x) & (lane_x >= 0.0) & (lane_x <= float(self.cfg.input_w))
                valid_rows = finite_valid.nonzero(as_tuple=False).flatten()
                if valid_rows.numel() == 0:
                    continue

                seed_row = valid_rows[-1]
                if row_count == 1:
                    feat_y = torch.zeros((), device=device, dtype=torch.long)
                else:
                    feat_y = torch.round(seed_row.to(dtype=dense_x.dtype) * float(feat_h - 1) / float(row_count - 1)).long()
                seed_x = lane_x[seed_row]
                seed_x_bin = (seed_x / float(self.cfg.input_w) * float(feat_w)).clamp(0.0, float(feat_w - 1))
                heat = torch.exp(-0.5 * ((grid_x - seed_x_bin.to(dtype=dtype)) / sigma).pow(2))
                heatmap_target[bi, 0, feat_y] = torch.maximum(heatmap_target[bi, 0, feat_y], heat)

                center_bin = int(torch.round(seed_x_bin).clamp(0, feat_w - 1).item())
                for offset in range(-radius, radius + 1):
                    feat_x = center_bin + offset
                    if feat_x < 0 or feat_x >= feat_w:
                        continue
                    weight = dense_x.new_tensor(float(torch.exp(torch.tensor(-0.5 * (float(offset) / sigma) ** 2))))
                    pred_lane = dense_x[bi, :row_count, feat_y, feat_x]
                    gt_lane = lane_x[:row_count]
                    mask = finite_valid[:row_count]
                    if mask.any():
                        x_loss = x_loss + weight * F.smooth_l1_loss(
                            pred_lane[mask] / float(self.cfg.input_w),
                            gt_lane[mask] / float(self.cfg.input_w),
                            beta=self.cfg.smooth_l1_beta,
                            reduction="sum",
                        )
                        x_denom = x_denom + weight * mask.to(dtype=dense_x.dtype).sum()
                    if "range_y" in target:
                        gt_range = target["range_y"].to(device=device, dtype=dense_range.dtype)[lane_idx] / float(self.cfg.input_h)
                        pred_range = dense_range[bi, :, feat_y, feat_x]
                        range_loss = range_loss + weight * F.smooth_l1_loss(
                            pred_range,
                            sort_range_norm(gt_range.view(1, 1, 2)).view(2),
                            beta=self.cfg.smooth_l1_beta,
                            reduction="sum",
                        )
                        range_denom = range_denom + weight * 2.0

        pos_weight = None
        if self.cfg.dynamic_proposal_heatmap_pos_weight != 1.0:
            pos_weight = torch.tensor([self.cfg.dynamic_proposal_heatmap_pos_weight], device=device, dtype=dtype)
        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_logits, heatmap_target, pos_weight=pos_weight)
        x_loss = x_loss / x_denom.clamp_min(1.0)
        range_loss = range_loss / range_denom.clamp_min(1.0)
        return {"heatmap": heatmap_loss, "x": x_loss, "range": range_loss}

    def _zero_anchor(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, dict):
                try:
                    return self._zero_anchor(value)
                except StopIteration:
                    pass
        raise StopIteration("No tensor found in outputs")

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
