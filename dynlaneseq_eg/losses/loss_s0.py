from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from dynlaneseq_eg.modeling.common import sort_range_norm
from .matcher_s0 import HungarianMatcherS0


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
    w_row_dfl: float = 0.0
    row_dfl_warmup_iters: int = 0
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
    lambda_intermediate: float = 0.0
    intermediate_layer_weights: tuple[float, ...] = ()
    geometry_reduction: str = "global_rows"


class S0Criterion(nn.Module):
    def __init__(self, cfg: LossConfig | None = None, matcher: HungarianMatcherS0 | None = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.matcher = matcher
        self._iteration = 0
        reduction = str(self.cfg.geometry_reduction).strip().lower()
        if reduction not in {
            "global",
            "global_rows",
            "row",
            "rows",
            "lane",
            "lane_mean",
            "per_lane",
        }:
            raise ValueError(
                f"Unsupported loss.geometry_reduction: {self.cfg.geometry_reduction!r}"
            )

    def lane_balanced_geometry(self) -> bool:
        return str(self.cfg.geometry_reduction).strip().lower() in {
            "lane",
            "lane_mean",
            "per_lane",
        }

    def set_iteration(self, iteration: int) -> None:
        self._iteration = int(iteration)

    def row_dfl_weight(self) -> float:
        weight = float(self.cfg.w_row_dfl)
        warmup = int(self.cfg.row_dfl_warmup_iters)
        if weight == 0.0 or warmup <= 0:
            return weight
        return weight * min(1.0, float(self._iteration + 1) / float(warmup))

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
        zero = self._zero_anchor(raw_outputs).sum() * 0.0
        loss_exist = self.compute_exist_loss(outputs, matches) if self.cfg.w_exist != 0 else zero
        loss_point = self.compute_point_loss(outputs, targets, matches) if self.cfg.w_point != 0 else zero
        loss_range = self.compute_range_loss(outputs, targets, matches) if self.cfg.w_range != 0 else zero
        loss_smooth = self.compute_smoothness_loss(outputs, targets, matches) if self.cfg.w_smooth != 0 else zero
        loss_line_iou = self.compute_line_iou_loss(outputs, targets, matches) if self.cfg.w_line_iou != 0 else zero
        loss_seg = self.compute_seg_loss(raw_outputs, targets) if self.cfg.w_seg != 0 else zero
        loss_quality = self.compute_quality_loss(outputs, targets, matches) if self.cfg.w_quality != 0 else zero
        loss_centerline = self.compute_centerline_loss(raw_outputs, targets) if self.cfg.w_centerline != 0 else zero
        row_dfl_weight = self.row_dfl_weight()
        loss_row_dfl = self.compute_row_dfl_loss(outputs, targets, matches) if row_dfl_weight != 0 else zero
        if (
            self.cfg.w_dynamic_proposal_heatmap != 0
            or self.cfg.w_dynamic_proposal_x != 0
            or self.cfg.w_dynamic_proposal_range != 0
        ):
            dynamic_proposal_losses = self.compute_dynamic_proposal_losses(raw_outputs, targets)
        else:
            dynamic_proposal_losses = {"heatmap": zero, "x": zero, "range": zero}
        total = (
            self.cfg.w_exist * loss_exist
            + self.cfg.w_point * loss_point
            + self.cfg.w_range * loss_range
            + self.cfg.w_smooth * loss_smooth
            + self.cfg.w_line_iou * loss_line_iou
            + self.cfg.w_seg * loss_seg
            + self.cfg.w_quality * loss_quality
            + self.cfg.w_centerline * loss_centerline
            + row_dfl_weight * loss_row_dfl
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
            "loss_row_dfl": loss_row_dfl,
            "weight_row_dfl": zero.new_tensor(row_dfl_weight),
            "loss_dynamic_proposal_heatmap": dynamic_proposal_losses["heatmap"],
            "loss_dynamic_proposal_x": dynamic_proposal_losses["x"],
            "loss_dynamic_proposal_range": dynamic_proposal_losses["range"],
        }
        if self.cfg.lambda_coarse > 0 and isinstance(raw_outputs.get("coarse"), dict):
            coarse = raw_outputs["coarse"]
            coarse_exist = self.compute_exist_loss(coarse, matches) if self.cfg.w_exist != 0 else zero
            coarse_point = self.compute_point_loss(coarse, targets, matches) if self.cfg.w_point != 0 else zero
            coarse_range = self.compute_range_loss(coarse, targets, matches) if self.cfg.w_range != 0 else zero
            coarse_smooth = self.compute_smoothness_loss(coarse, targets, matches) if self.cfg.w_smooth != 0 else zero
            coarse_line_iou = self.compute_line_iou_loss(coarse, targets, matches) if self.cfg.w_line_iou != 0 else zero
            coarse_quality = self.compute_quality_loss(coarse, targets, matches) if self.cfg.w_quality != 0 else zero
            coarse_row_dfl = self.compute_row_dfl_loss(coarse, targets, matches) if row_dfl_weight != 0 else zero
            coarse_total = (
                self.cfg.w_exist * coarse_exist
                + self.cfg.w_point * coarse_point
                + self.cfg.w_range * coarse_range
                + self.cfg.w_smooth * coarse_smooth
                + self.cfg.w_line_iou * coarse_line_iou
                + self.cfg.w_quality * coarse_quality
                + row_dfl_weight * coarse_row_dfl
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
                    "loss_row_dfl_coarse": coarse_row_dfl,
                }
            )
        out = self.add_geometry_draft_loss(out, raw_outputs, targets, matches)
        out = self.add_intermediate_losses(out, raw_outputs, targets)
        return out

    def add_intermediate_losses(
        self,
        losses: dict[str, torch.Tensor],
        outputs: dict[str, object],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Deeply supervise intermediate structured decoder states.

        Each layer receives its own assignment, matching CondLSTR's training
        principle while retaining the final layer as the only quality-calibrated
        output.  Encoder auxiliary objectives, smoothness, and quality are not
        duplicated here.
        """
        strength = float(self.cfg.lambda_intermediate)
        if strength <= 0.0:
            return losses
        aux_outputs = outputs.get("aux_outputs")
        if not isinstance(aux_outputs, (list, tuple)) or not aux_outputs:
            raise ValueError("lambda_intermediate > 0 requires non-empty model aux_outputs")
        if self.matcher is None:
            raise ValueError("lambda_intermediate > 0 requires an auxiliary matcher")

        configured_weights = tuple(float(v) for v in self.cfg.intermediate_layer_weights)
        if configured_weights:
            if len(configured_weights) != len(aux_outputs):
                raise ValueError(
                    "intermediate_layer_weights must match the number of auxiliary decoder layers: "
                    f"got {len(configured_weights)} weights for {len(aux_outputs)} outputs"
                )
            layer_weights = configured_weights
        else:
            layer_weights = tuple(1.0 for _ in aux_outputs)
        if any(weight < 0.0 for weight in layer_weights) or sum(layer_weights) <= 0.0:
            raise ValueError("intermediate_layer_weights must be non-negative with a positive sum")

        normalizer = float(sum(layer_weights))
        row_dfl_weight = self.row_dfl_weight()
        aggregate = losses["loss_total"].new_zeros(())
        component_sums = {
            "exist": aggregate.clone(),
            "point": aggregate.clone(),
            "range": aggregate.clone(),
            "line_iou": aggregate.clone(),
            "row_dfl": aggregate.clone(),
        }
        out = dict(losses)
        for layer_index, (aux, layer_weight) in enumerate(zip(aux_outputs, layer_weights), start=1):
            if not isinstance(aux, dict):
                raise TypeError("every auxiliary decoder output must be a dictionary")
            aux_matches = self.matcher(aux, targets)
            zero = self._zero_anchor(aux).sum() * 0.0
            aux_exist = self.compute_exist_loss(aux, aux_matches) if self.cfg.w_exist != 0 else zero
            aux_point = self.compute_point_loss(aux, targets, aux_matches) if self.cfg.w_point != 0 else zero
            aux_range = self.compute_range_loss(aux, targets, aux_matches) if self.cfg.w_range != 0 else zero
            aux_line_iou = (
                self.compute_line_iou_loss(aux, targets, aux_matches)
                if self.cfg.w_line_iou != 0
                else zero
            )
            aux_row_dfl = (
                self.compute_row_dfl_loss(aux, targets, aux_matches)
                if row_dfl_weight != 0
                else zero
            )
            aux_total = (
                self.cfg.w_exist * aux_exist
                + self.cfg.w_point * aux_point
                + self.cfg.w_range * aux_range
                + self.cfg.w_line_iou * aux_line_iou
                + row_dfl_weight * aux_row_dfl
            )
            normalized_weight = float(layer_weight) / normalizer
            aggregate = aggregate + normalized_weight * aux_total
            component_sums["exist"] = component_sums["exist"] + normalized_weight * aux_exist
            component_sums["point"] = component_sums["point"] + normalized_weight * aux_point
            component_sums["range"] = component_sums["range"] + normalized_weight * aux_range
            component_sums["line_iou"] = component_sums["line_iou"] + normalized_weight * aux_line_iou
            component_sums["row_dfl"] = component_sums["row_dfl"] + normalized_weight * aux_row_dfl
            out[f"loss_intermediate_l{layer_index}_total"] = aux_total

        out["loss_total"] = out["loss_total"] + strength * aggregate
        out["loss_intermediate_total"] = aggregate
        out["loss_intermediate_exist"] = component_sums["exist"]
        out["loss_intermediate_point"] = component_sums["point"]
        out["loss_intermediate_range"] = component_sums["range"]
        out["loss_intermediate_line_iou"] = component_sums["line_iou"]
        out["loss_intermediate_row_dfl"] = component_sums["row_dfl"]
        out["weight_intermediate"] = aggregate.new_tensor(strength)
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
        zero = self._zero_anchor(draft).sum() * 0.0
        draft_exist = self.compute_exist_loss(draft, matches) if self.cfg.w_exist != 0 else zero
        draft_point = self.compute_point_loss(draft, targets, matches) if self.cfg.w_point != 0 else zero
        draft_range = self.compute_range_loss(draft, targets, matches) if self.cfg.w_range != 0 else zero
        draft_smooth = self.compute_smoothness_loss(draft, targets, matches) if self.cfg.w_smooth != 0 else zero
        draft_line_iou = self.compute_line_iou_loss(draft, targets, matches) if self.cfg.w_line_iou != 0 else zero
        draft_quality = self.compute_quality_loss(draft, targets, matches) if self.cfg.w_quality != 0 else zero
        row_dfl_weight = self.row_dfl_weight()
        draft_row_dfl = self.compute_row_dfl_loss(draft, targets, matches) if row_dfl_weight != 0 else zero
        draft_total = (
            self.cfg.w_exist * draft_exist
            + self.cfg.w_point * draft_point
            + self.cfg.w_range * draft_range
            + self.cfg.w_smooth * draft_smooth
            + self.cfg.w_line_iou * draft_line_iou
            + self.cfg.w_quality * draft_quality
            + row_dfl_weight * draft_row_dfl
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
                "loss_row_dfl_geometry_draft": draft_row_dfl,
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

    @staticmethod
    def _zero_anchor(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, dict):
                try:
                    return S0Criterion._zero_anchor(value)
                except StopIteration:
                    continue
        raise StopIteration

    def compute_point_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"]
        total = pred_x.sum() * 0.0
        count = pred_x.new_tensor(0.0)
        lane_balanced = self.lane_balanced_geometry()
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(pred_x.device)[gt_idx]
            mask = targets[bi]["valid_mask"].to(pred_x.device)[gt_idx].bool()
            pred = pred_x[bi, pred_idx] / float(self.cfg.input_w)
            gt = gt_x / float(self.cfg.input_w)
            valid = mask.to(dtype=pred.dtype)
            loss = F.smooth_l1_loss(pred, gt, beta=self.cfg.smooth_l1_beta, reduction="none")
            if lane_balanced:
                valid_count = valid.sum(dim=-1)
                lane_loss = (loss * valid).sum(dim=-1) / valid_count.clamp_min(1.0)
                valid_lane = valid_count > 0
                total = total + (
                    lane_loss * valid_lane.to(dtype=lane_loss.dtype)
                ).sum()
                count = count + valid_lane.to(dtype=count.dtype).sum()
            else:
                total = total + (loss * valid).sum()
                count = count + valid.sum()
        return total / count.clamp_min(1.0)

    def compute_row_dfl_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        logits = outputs.get("row_x_logits")
        if logits is None:
            return outputs["pred_x_rows"].sum() * 0.0
        b, n, num_rows, x_bins = logits.shape
        del b, n
        # Preserve a differentiable zero without materializing an FP32 copy of
        # every slot/row/bin.  Only matched lane logits contribute to DFL and
        # are converted below after indexing.
        total = logits.sum(dtype=torch.float32) * 0.0
        count = total.new_tensor(0.0)
        lane_balanced = self.lane_balanced_geometry()
        bin_width = float(self.cfg.input_w) / float(x_bins)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(logits.device)
            gt_idx = match["gt_indices"].to(logits.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(logits.device, dtype=torch.float32)[gt_idx]
            mask = targets[bi]["valid_mask"].to(logits.device)[gt_idx].bool()
            row_count = min(int(gt_x.shape[-1]), int(num_rows))
            if row_count <= 0:
                continue
            pred_logits = logits[bi, pred_idx, :row_count].float()
            gt_x = gt_x[:, :row_count]
            mask = mask[:, :row_count]
            valid = mask & torch.isfinite(gt_x) & (gt_x >= 0.0) & (gt_x <= float(self.cfg.input_w))

            # Avoid a Python boolean conversion of a CUDA tensor here.  Deep
            # supervision reaches this path once per image and decoder output;
            # the old ``if not valid.any()`` therefore serialized the stream
            # many times per optimizer step.  Invalid values are made safe
            # before indexing and remain exactly zero-weighted below.
            safe_gt_x = torch.where(valid, gt_x, torch.zeros_like(gt_x))
            target_bin = (safe_gt_x / bin_width).clamp(0.0, float(x_bins - 1))
            left = target_bin.floor().long()
            right = (left + 1).clamp(max=x_bins - 1)
            right_w = target_bin - left.to(dtype=target_bin.dtype)
            left_w = 1.0 - right_w
            same = right == left
            left_w = torch.where(same, torch.ones_like(left_w), left_w)
            right_w = torch.where(same, torch.zeros_like(right_w), right_w)

            log_probs = F.log_softmax(pred_logits, dim=-1)
            left_lp = log_probs.gather(-1, left.unsqueeze(-1)).squeeze(-1)
            right_lp = log_probs.gather(-1, right.unsqueeze(-1)).squeeze(-1)
            loss = -(left_w * left_lp + right_w * right_lp)
            valid_f = valid.to(dtype=loss.dtype)
            if lane_balanced:
                valid_count = valid_f.sum(dim=-1)
                lane_loss = (loss * valid_f).sum(dim=-1) / valid_count.clamp_min(1.0)
                valid_lane = valid_count > 0
                total = total + (
                    lane_loss * valid_lane.to(dtype=lane_loss.dtype)
                ).sum()
                count = count + valid_lane.to(dtype=count.dtype).sum()
            else:
                total = total + (loss * valid_f).sum()
                count = count + valid_f.sum()
        return total / count.clamp_min(1.0)

    def compute_line_iou_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"]
        total = pred_x.sum() * 0.0
        count = pred_x.new_tensor(0.0)
        radius = float(self.cfg.line_iou_radius)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(pred_x.device)[gt_idx]
            mask = targets[bi]["valid_mask"].to(pred_x.device)[gt_idx].bool()
            pred = pred_x[bi, pred_idx]
            px1 = pred - radius
            px2 = pred + radius
            gx1 = gt_x - radius
            gx2 = gt_x + radius
            overlap = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0.0)
            union = (4.0 * radius - overlap).clamp(min=1e-6)
            iou = overlap / union
            enclosing = (torch.maximum(px2, gx2) - torch.minimum(px1, gx1)).clamp(min=1e-6)
            giou = iou - (enclosing - union) / enclosing
            valid = mask.to(dtype=pred_x.dtype)
            valid_count = valid.sum(dim=-1)
            lane_loss = ((1.0 - giou) * valid).sum(dim=-1) / valid_count.clamp_min(1.0)
            valid_lane = valid_count > 0
            total = total + (lane_loss * valid_lane.to(dtype=lane_loss.dtype)).sum()
            count = count + valid_lane.to(dtype=count.dtype).sum()
        return total / count.clamp_min(1.0)

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
            px1 = pred - radius
            px2 = pred + radius
            gx1 = gt_x - radius
            gx2 = gt_x + radius
            overlap = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0.0)
            union = (4.0 * radius - overlap).clamp(min=1e-6)
            valid = mask.to(dtype=pred_x.dtype)
            valid_count = valid.sum(dim=-1)
            qualities = ((overlap / union) * valid).sum(dim=-1) / valid_count.clamp_min(1.0)
            qualities = qualities * (valid_count > 0).to(dtype=qualities.dtype)
            target_quality[bi, pred_idx] = qualities.detach().to(dtype=target_quality.dtype)
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
            # Keep empty/invalid images on the tensor path instead of forcing
            # a device synchronization through ``Tensor.__bool__``.
            safe_centers = torch.where(valid, centers, torch.zeros_like(centers))
            diff = grid - safe_centers.unsqueeze(-1)
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
