from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import line_iou_against_gt
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, input_to_grid, nested_to_device


SOURCE_NAMES = ("c2", "p2", "pyramid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a trained lane detector and train equal-capacity sequence probes "
            "on GT-curve-aligned C2, P2, and top-down pyramid profiles. The held-out "
            "metrics determine whether lanes missed by the decoder are represented "
            "in its visual features. This is a diagnostic, not a benchmark result."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--probe-rows", type=int, default=80)
    parser.add_argument("--probe-layers", type=int, default=2)
    parser.add_argument(
        "--offsets-px",
        type=float,
        nargs="+",
        default=[-48, -40, -32, -24, -16, -8, 0, 8, 16, 24, 32, 40, 48],
    )
    parser.add_argument("--max-train-shift-px", type=float, default=36.0)
    parser.add_argument("--point-loss-weight", type=float, default=2.0)
    parser.add_argument("--eval-max-batches", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--base-hit-threshold", type=float, default=0.5)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--save-probes", default="")
    parser.add_argument("--load-probes", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class LaneExamples:
    image_indices: torch.Tensor
    gt_x: torch.Tensor
    anchor_x: torch.Tensor
    target_residual: torch.Tensor
    valid: torch.Tensor
    lane_indices: torch.Tensor
    pattern_indices: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.gt_x.shape[0])


class CurveAlignedSequenceProbe(nn.Module):
    """Equal-capacity curve evidence readout for one or more frozen scales."""

    def __init__(
        self,
        *,
        common_channels: int,
        hidden_dim: int,
        num_rows: int,
        offsets_px: list[float],
        max_scales: int = 3,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four")
        self.common_channels = int(common_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_rows = int(num_rows)
        self.max_scales = int(max_scales)
        self.register_buffer(
            "offsets_px",
            torch.tensor(offsets_px, dtype=torch.float32),
            persistent=True,
        )
        self.input_norm = nn.LayerNorm(self.common_channels)
        self.input_proj = nn.Sequential(
            nn.Linear(self.common_channels, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.scale_embedding = nn.Parameter(torch.zeros(self.max_scales, self.hidden_dim))
        self.scale_score = nn.Linear(self.hidden_dim, 1)
        self.offset_embedding = nn.Embedding(len(offsets_px), self.hidden_dim)
        self.row_embedding = nn.Embedding(self.num_rows, self.hidden_dim)
        self.local_mixer = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=4 * self.hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.row_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.local_score = nn.Linear(self.hidden_dim, 1)
        self.context_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.normal_(self.scale_embedding, std=0.02)
        nn.init.normal_(self.offset_embedding.weight, std=0.02)
        nn.init.normal_(self.row_embedding.weight, std=0.02)

    def forward(
        self,
        profiles: torch.Tensor,
        scale_mask: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """profiles=[lanes,rows,offsets,scales,channels]."""
        if profiles.ndim != 5:
            raise ValueError(f"profiles must be rank 5, got {tuple(profiles.shape)}")
        lanes, rows, offsets, scales, channels = profiles.shape
        if rows != self.num_rows:
            raise ValueError(f"Expected {self.num_rows} rows, got {rows}")
        if scales != self.max_scales:
            raise ValueError(f"Expected {self.max_scales} scales, got {scales}")
        if channels != self.common_channels:
            raise ValueError(f"Expected {self.common_channels} channels, got {channels}")
        if scale_mask.shape != (lanes, self.max_scales):
            raise ValueError(
                f"scale_mask must be {(lanes, self.max_scales)}, got {tuple(scale_mask.shape)}"
            )

        normalized = self.input_norm(profiles.float())
        hidden = self.input_proj(normalized)
        hidden = hidden + self.scale_embedding.view(
            1, 1, 1, self.max_scales, self.hidden_dim
        )
        scale_logits = self.scale_score(hidden).squeeze(-1)
        active = scale_mask[:, None, None, :].bool()
        scale_logits = scale_logits.masked_fill(~active, torch.finfo(scale_logits.dtype).min)
        scale_weights = torch.softmax(scale_logits, dim=-1)
        hidden = (scale_weights.unsqueeze(-1) * hidden).sum(dim=3)

        hidden = (
            hidden
            + self.row_embedding.weight.view(1, rows, 1, self.hidden_dim)
            + self.offset_embedding.weight.view(1, 1, offsets, self.hidden_dim)
        )
        mixed = self.local_mixer(
            hidden.reshape(lanes * rows, offsets, self.hidden_dim).transpose(1, 2)
        ).transpose(1, 2)
        hidden = hidden + mixed.reshape(lanes, rows, offsets, self.hidden_dim)
        local_logits = self.local_score(hidden).squeeze(-1)
        local_weights = torch.softmax(local_logits, dim=-1)
        row_summary = (local_weights.unsqueeze(-1) * hidden).sum(dim=2)
        row_context = self.row_encoder(
            row_summary,
            src_key_padding_mask=~valid_rows.bool(),
        )
        context = self.context_proj(row_context)
        logits = local_logits + (hidden * context.unsqueeze(2)).sum(dim=-1) / math.sqrt(
            float(self.hidden_dim)
        )
        probabilities = torch.softmax(logits, dim=-1)
        residual = (
            probabilities * self.offsets_px.to(device=logits.device, dtype=logits.dtype)
        ).sum(dim=-1)
        return {
            "logits": logits,
            "residual": residual,
            "scale_weights": scale_weights,
        }


def _pad_channels(feature: torch.Tensor, common_channels: int) -> torch.Tensor:
    channels = int(feature.shape[-1])
    if channels > int(common_channels):
        raise ValueError(
            f"Feature has {channels} channels but common width is {common_channels}"
        )
    if channels == int(common_channels):
        return feature
    return F.pad(feature, (0, int(common_channels) - channels))


def _sample_feature_profiles(
    feature: torch.Tensor,
    image_index: int,
    sample_x: torch.Tensor,
    *,
    input_w: int,
    input_h: int,
) -> torch.Tensor:
    """Sample one feature map and return [lanes,rows,offsets,channels]."""
    if sample_x.ndim != 3:
        raise ValueError("sample_x must have shape [lanes,rows,offsets]")
    lanes, rows, offsets = sample_x.shape
    y = fixed_y_rows(rows, input_h, device=sample_x.device, dtype=sample_x.dtype)
    y = y.view(1, rows, 1).expand(lanes, rows, offsets)
    grid = input_to_grid(
        sample_x.clamp(0.0, float(input_w - 1)),
        y,
        input_w=input_w,
        input_h=input_h,
    )
    sampled = F.grid_sample(
        feature[image_index : image_index + 1].float(),
        grid.reshape(1, lanes * rows * offsets, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).reshape(
        lanes, rows, offsets, int(feature.shape[1])
    )


def _extract_frozen_sources(
    model: nn.Module,
    images: torch.Tensor,
) -> tuple[dict[str, list[torch.Tensor]], torch.Tensor]:
    """Reproduce SimpleFPN and expose trained top-down states without extra heads."""
    encoder = model.encoder
    feats = encoder.backbone(images)
    lateral = encoder.fpn.lateral
    p5 = lateral["c5"](feats["c5"])
    p4 = lateral["c4"](feats["c4"]) + F.interpolate(
        p5, size=feats["c4"].shape[-2:], mode="nearest"
    )
    p3 = lateral["c3"](feats["c3"]) + F.interpolate(
        p4, size=feats["c3"].shape[-2:], mode="nearest"
    )
    p2_pre = lateral["c2"](feats["c2"]) + F.interpolate(
        p3, size=feats["c2"].shape[-2:], mode="nearest"
    )
    p2 = encoder.proj(encoder.fpn.output(p2_pre))
    return {
        "c2": [feats["c2"]],
        "p2": [p2],
        "pyramid": [p2, p3, p4],
    }, p2


def _selected_row_indices(num_rows: int, probe_rows: int, device: torch.device) -> torch.Tensor:
    if probe_rows > num_rows:
        raise ValueError("probe_rows cannot exceed model num_rows")
    return torch.round(
        torch.linspace(0, num_rows - 1, probe_rows, device=device)
    ).long()


def _training_residuals(
    lane_count: int,
    num_rows: int,
    max_shift: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, num_rows, device=device).view(1, num_rows)
    a0 = torch.empty(lane_count, 1, device=device).uniform_(-0.75, 0.75)
    a1 = torch.empty(lane_count, 1, device=device).uniform_(-0.65, 0.65)
    a2 = torch.empty(lane_count, 1, device=device).uniform_(-0.35, 0.35)
    residual = float(max_shift) * (
        a0 + a1 * t + a2 * torch.sin(math.pi * t)
    )
    return residual.clamp(-float(max_shift), float(max_shift))


def _evaluation_residuals(num_rows: int, *, device: torch.device) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, num_rows, device=device)
    return torch.stack(
        (
            torch.full_like(t, -32.0),
            torch.full_like(t, 32.0),
            32.0 * t,
            -32.0 * t,
        ),
        dim=0,
    )


def _build_lane_examples(
    targets: list[dict[str, torch.Tensor]],
    *,
    row_indices: torch.Tensor,
    input_w: int,
    max_offset: float,
    training: bool,
    max_train_shift: float,
) -> LaneExamples | None:
    image_parts: list[torch.Tensor] = []
    gt_parts: list[torch.Tensor] = []
    anchor_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    lane_parts: list[torch.Tensor] = []
    pattern_parts: list[torch.Tensor] = []
    probe_rows = int(row_indices.numel())
    for image_index, target in enumerate(targets):
        gt_full = target["x_rows"].float()
        valid_full = target["valid_mask"].bool()
        if gt_full.numel() == 0:
            continue
        gt = gt_full[:, row_indices]
        valid = valid_full[:, row_indices]
        lane_ids = torch.arange(gt.shape[0], device=gt.device)
        keep = valid.sum(dim=-1) >= 5
        gt = gt[keep]
        valid = valid[keep]
        lane_ids = lane_ids[keep]
        if gt.numel() == 0:
            continue

        if training:
            residual = _training_residuals(
                int(gt.shape[0]),
                probe_rows,
                max_train_shift,
                device=gt.device,
            )
            patterns = torch.zeros(gt.shape[0], device=gt.device, dtype=torch.long)
        else:
            templates = _evaluation_residuals(probe_rows, device=gt.device)
            pattern_count = int(templates.shape[0])
            gt = gt.repeat_interleave(pattern_count, dim=0)
            valid = valid.repeat_interleave(pattern_count, dim=0)
            lane_ids = lane_ids.repeat_interleave(pattern_count)
            residual = templates.repeat(int(gt.shape[0] // pattern_count), 1)
            patterns = torch.arange(
                pattern_count, device=gt.device, dtype=torch.long
            ).repeat(int(gt.shape[0] // pattern_count))

        anchor = gt - residual
        safe = (
            valid
            & torch.isfinite(gt)
            & (anchor >= float(max_offset))
            & (anchor <= float(input_w - 1) - float(max_offset))
        )
        keep = safe.sum(dim=-1) >= 5
        if not bool(keep.any()):
            continue
        gt = gt[keep]
        anchor = anchor[keep]
        residual = residual[keep]
        safe = safe[keep]
        lane_ids = lane_ids[keep]
        patterns = patterns[keep]
        anchor = torch.where(safe, anchor, torch.full_like(anchor, float(input_w) * 0.5))

        count = int(gt.shape[0])
        image_parts.append(
            torch.full((count,), image_index, device=gt.device, dtype=torch.long)
        )
        gt_parts.append(gt)
        anchor_parts.append(anchor)
        residual_parts.append(residual)
        valid_parts.append(safe)
        lane_parts.append(lane_ids)
        pattern_parts.append(patterns)
    if not gt_parts:
        return None
    return LaneExamples(
        image_indices=torch.cat(image_parts),
        gt_x=torch.cat(gt_parts),
        anchor_x=torch.cat(anchor_parts),
        target_residual=torch.cat(residual_parts),
        valid=torch.cat(valid_parts),
        lane_indices=torch.cat(lane_parts),
        pattern_indices=torch.cat(pattern_parts),
    )


def _profiles_for_source(
    source_features: list[torch.Tensor],
    examples: LaneExamples,
    offsets: torch.Tensor,
    *,
    common_channels: int,
    max_scales: int,
    input_w: int,
    input_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_profiles: list[torch.Tensor] = []
    for image_index in examples.image_indices.unique(sorted=True).tolist():
        selected = examples.image_indices == int(image_index)
        sample_x = examples.anchor_x[selected].unsqueeze(-1) + offsets.view(1, 1, -1)
        scale_profiles = []
        for feature in source_features:
            sampled = _sample_feature_profiles(
                feature,
                int(image_index),
                sample_x,
                input_w=input_w,
                input_h=input_h,
            )
            sampled = F.normalize(sampled, p=2.0, dim=-1, eps=1e-6)
            scale_profiles.append(_pad_channels(sampled, common_channels))
        while len(scale_profiles) < int(max_scales):
            scale_profiles.append(torch.zeros_like(scale_profiles[0]))
        all_profiles.append(torch.stack(scale_profiles[:max_scales], dim=3))
    profiles = torch.cat(all_profiles, dim=0)
    active_scales = min(len(source_features), int(max_scales))
    scale_mask = torch.zeros(
        examples.count,
        max_scales,
        device=profiles.device,
        dtype=torch.bool,
    )
    scale_mask[:, :active_scales] = True
    return profiles, scale_mask


def _probe_loss(
    outputs: dict[str, torch.Tensor],
    target_residual: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    *,
    point_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_index = (
        target_residual.unsqueeze(-1) - offsets.view(1, 1, -1)
    ).abs().argmin(dim=-1)
    logits = outputs["logits"][valid]
    target_index = target_index[valid]
    ce = F.cross_entropy(logits.float(), target_index, reduction="mean")
    scale = float(offsets.abs().max().clamp_min(1.0))
    point = F.smooth_l1_loss(
        outputs["residual"][valid].float() / scale,
        target_residual[valid].float() / scale,
        beta=0.1,
        reduction="mean",
    )
    loss = ce + float(point_weight) * point
    return loss, {
        "ce": float(ce.detach()),
        "point": float(point.detach()),
        "total": float(loss.detach()),
    }


class BucketStats:
    def __init__(self) -> None:
        self.examples = 0
        self.rows = 0
        self.anchor_abs_error = 0.0
        self.corrected_abs_error = 0.0
        self.anchor_iou = 0.0
        self.corrected_iou = 0.0
        self.anchor_hit_050 = 0
        self.corrected_hit_050 = 0
        self.anchor_hit_070 = 0
        self.corrected_hit_070 = 0
        self.recovered_050 = 0
        self.lost_050 = 0
        self.direction_rows = 0
        self.direction_correct = 0
        self.scale_weight_sum: torch.Tensor | None = None

    def update_batch(
        self,
        *,
        anchor: torch.Tensor,
        corrected: torch.Tensor,
        gt: torch.Tensor,
        valid: torch.Tensor,
        target_residual: torch.Tensor,
        predicted_residual: torch.Tensor,
        scale_weights: torch.Tensor,
        anchor_iou: torch.Tensor,
        corrected_iou: torch.Tensor,
        selected: torch.Tensor,
    ) -> None:
        selected = selected.bool()
        if not bool(selected.any()):
            return
        anchor = anchor[selected]
        corrected = corrected[selected]
        gt = gt[selected]
        valid = valid[selected]
        target_residual = target_residual[selected]
        predicted_residual = predicted_residual[selected]
        anchor_iou = anchor_iou[selected]
        corrected_iou = corrected_iou[selected]
        scale_weights = scale_weights[selected]
        example_count = int(selected.sum())
        row_count = int(valid.sum())
        self.examples += example_count
        self.rows += row_count
        self.anchor_abs_error += float((anchor[valid] - gt[valid]).abs().sum())
        self.corrected_abs_error += float((corrected[valid] - gt[valid]).abs().sum())
        self.anchor_iou += float(anchor_iou.sum())
        self.corrected_iou += float(corrected_iou.sum())
        anchor_050 = anchor_iou >= 0.5
        corrected_050 = corrected_iou >= 0.5
        self.anchor_hit_050 += int(anchor_050.sum())
        self.corrected_hit_050 += int(corrected_050.sum())
        self.anchor_hit_070 += int((anchor_iou >= 0.7).sum())
        self.corrected_hit_070 += int((corrected_iou >= 0.7).sum())
        self.recovered_050 += int((~anchor_050 & corrected_050).sum())
        self.lost_050 += int((anchor_050 & ~corrected_050).sum())
        direction_mask = valid & (target_residual.abs() >= 4.0)
        self.direction_rows += int(direction_mask.sum())
        self.direction_correct += int(
            (
                torch.sign(predicted_residual[direction_mask])
                == torch.sign(target_residual[direction_mask])
            ).sum()
        )
        valid_weight = valid[:, :, None, None].to(dtype=scale_weights.dtype)
        weights = (
            (scale_weights * valid_weight).sum(dim=(1, 2))
            / (
                valid.sum(dim=1, keepdim=True).to(dtype=scale_weights.dtype)
                * float(scale_weights.shape[2])
            ).clamp_min(1.0)
        ).sum(dim=0).detach().cpu()
        self.scale_weight_sum = (
            weights if self.scale_weight_sum is None else self.scale_weight_sum + weights
        )

    def summary(self) -> dict[str, Any]:
        count = max(self.examples, 1)
        rows = max(self.rows, 1)
        scale_weights = (
            (self.scale_weight_sum / count).tolist()
            if self.scale_weight_sum is not None
            else []
        )
        return {
            "lane_patterns": self.examples,
            "rows": self.rows,
            "anchor_row_mae_px": self.anchor_abs_error / rows,
            "corrected_row_mae_px": self.corrected_abs_error / rows,
            "row_mae_gain_px": (self.anchor_abs_error - self.corrected_abs_error) / rows,
            "anchor_mean_iou": self.anchor_iou / count,
            "corrected_mean_iou": self.corrected_iou / count,
            "mean_iou_gain": (self.corrected_iou - self.anchor_iou) / count,
            "anchor_recall_050": self.anchor_hit_050 / count,
            "corrected_recall_050": self.corrected_hit_050 / count,
            "anchor_recall_070": self.anchor_hit_070 / count,
            "corrected_recall_070": self.corrected_hit_070 / count,
            "recovered_anchor_misses_050": self.recovered_050,
            "lost_anchor_hits_050": self.lost_050,
            "direction_accuracy": self.direction_correct / max(self.direction_rows, 1),
            "mean_scale_weights": scale_weights,
        }


def _paired_line_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    line_width: float,
) -> torch.Tensor:
    """Vectorized line IoU for paired [examples,rows] curves."""
    radius = float(line_width) * 0.5
    predictions = predictions.float()
    targets = targets.float()
    valid = valid.bool() & torch.isfinite(predictions) & torch.isfinite(targets)
    overlap = (
        torch.minimum(predictions + radius, targets + radius)
        - torch.maximum(predictions - radius, targets - radius)
    ).clamp_min(0.0)
    overlap = overlap.masked_fill(~valid, 0.0)
    union = (2.0 * float(line_width) - overlap).masked_fill(~valid, 0.0)
    return overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1e-6)


def _base_iou_buckets(
    predictions: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    *,
    group_size: int,
    line_width: float,
) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for image_index, target in enumerate(targets):
        candidates = predictions[image_index, :group_size].float()
        gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
        valid = target["valid_mask"].to(device=candidates.device).bool()
        for lane_index in range(int(gt_x.shape[0])):
            if int(valid[lane_index].sum()) < 5:
                continue
            ious = line_iou_against_gt(
                candidates,
                gt_x[lane_index],
                valid[lane_index],
                line_width=float(line_width),
            )
            result[(image_index, lane_index)] = (
                float(ious.max()) if ious.numel() else 0.0
            )
    return result


@torch.inference_mode()
def _evaluate(
    *,
    model: nn.Module,
    probes: dict[str, CurveAlignedSequenceProbe],
    loader,
    device: torch.device,
    channels_last: bool,
    amp_dtype: torch.dtype | None,
    row_indices: torch.Tensor,
    offsets: torch.Tensor,
    common_channels: int,
    input_w: int,
    input_h: int,
    group_size: int,
    max_batches: int,
    line_width: float,
    base_hit_threshold: float,
) -> dict[str, Any]:
    for probe in probes.values():
        probe.eval()
    stats = {
        name: {
            "all": BucketStats(),
            "base_hit": BucketStats(),
            "base_miss": BucketStats(),
        }
        for name in SOURCE_NAMES
    }
    lane_best: dict[str, dict[str, dict[tuple[int, int, int], list[float]]]] = {
        name: {
            "all": {},
            "base_hit": {},
            "base_miss": {},
        }
        for name in SOURCE_NAMES
    }
    images_seen = 0
    gt_lanes = 0
    base_hits = 0
    displayed_batches = len(loader)
    if max_batches > 0:
        displayed_batches = min(displayed_batches, int(max_batches))
    for batch_index, (images, targets, _metas) in enumerate(
        tqdm(
            loader,
            total=displayed_batches,
            desc="GT-curve feature probe eval",
            ncols=96,
        )
    ):
        if max_batches > 0 and batch_index >= max_batches:
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format,
        )
        targets = nested_to_device(targets, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None and device.type == "cuda",
        ):
            sources, p2 = _extract_frozen_sources(model, images)
            predictions = model.structured_query_head(p2, inference_only=True)[
                "pred_x_rows"
            ]
        base_iou = _base_iou_buckets(
            predictions,
            targets,
            group_size=group_size,
            line_width=line_width,
        )
        gt_lanes += len(base_iou)
        base_hits += sum(value >= base_hit_threshold for value in base_iou.values())
        examples = _build_lane_examples(
            targets,
            row_indices=row_indices,
            input_w=input_w,
            max_offset=float(offsets.abs().max()),
            training=False,
            max_train_shift=0.0,
        )
        if examples is None:
            images_seen += int(images.shape[0])
            continue
        base_values = torch.tensor(
            [
                base_iou.get((int(image_index), int(lane_index)), 0.0)
                for image_index, lane_index in zip(
                    examples.image_indices.tolist(),
                    examples.lane_indices.tolist(),
                )
            ],
            device=device,
            dtype=torch.float32,
        )
        bucket_masks = {
            "all": torch.ones(examples.count, device=device, dtype=torch.bool),
            "base_hit": base_values >= float(base_hit_threshold),
            "base_miss": base_values < float(base_hit_threshold),
        }
        anchor_iou = _paired_line_iou(
            examples.anchor_x,
            examples.gt_x,
            examples.valid,
            line_width=line_width,
        )
        for source_name in SOURCE_NAMES:
            profiles, scale_mask = _profiles_for_source(
                sources[source_name],
                examples,
                offsets,
                common_channels=common_channels,
                max_scales=3,
                input_w=input_w,
                input_h=input_h,
            )
            outputs = probes[source_name](profiles, scale_mask, examples.valid)
            corrected = examples.anchor_x + outputs["residual"]
            corrected_iou = _paired_line_iou(
                corrected,
                examples.gt_x,
                examples.valid,
                line_width=line_width,
            )
            values = dict(
                anchor=examples.anchor_x,
                corrected=corrected,
                gt=examples.gt_x,
                valid=examples.valid,
                target_residual=examples.target_residual,
                predicted_residual=outputs["residual"],
                scale_weights=outputs["scale_weights"],
                anchor_iou=anchor_iou,
                corrected_iou=corrected_iou,
            )
            for bucket_name, selected in bucket_masks.items():
                stats[source_name][bucket_name].update_batch(
                    **values,
                    selected=selected,
                )

            anchor_values = anchor_iou.detach().cpu().tolist()
            corrected_values = corrected_iou.detach().cpu().tolist()
            image_values = examples.image_indices.detach().cpu().tolist()
            lane_values = examples.lane_indices.detach().cpu().tolist()
            hit_values = bucket_masks["base_hit"].detach().cpu().tolist()
            for example_index, (
                local_image,
                lane_index,
                is_base_hit,
                anchor_value,
                corrected_value,
            ) in enumerate(
                zip(
                    image_values,
                    lane_values,
                    hit_values,
                    anchor_values,
                    corrected_values,
                )
            ):
                bucket = "base_hit" if is_base_hit else "base_miss"
                global_key = (batch_index, int(local_image), int(lane_index))
                for bucket_name in ("all", bucket):
                    entry = lane_best[source_name][bucket_name].setdefault(
                        global_key, [0.0, 0.0]
                    )
                    entry[0] = max(entry[0], float(anchor_value))
                    entry[1] = max(entry[1], float(corrected_value))
                _ = example_index
        images_seen += int(images.shape[0])

    result: dict[str, Any] = {
        "images": images_seen,
        "gt_lanes": gt_lanes,
        "base_group0_recall_050": base_hits / max(gt_lanes, 1),
        "sources": {},
    }
    for source_name in SOURCE_NAMES:
        source_result: dict[str, Any] = {}
        for bucket_name, bucket_stats in stats[source_name].items():
            summary = bucket_stats.summary()
            lane_values = lane_best[source_name][bucket_name].values()
            lane_count = len(lane_best[source_name][bucket_name])
            best_anchor_hits = sum(values[0] >= 0.5 for values in lane_values)
            best_corrected_hits = sum(values[1] >= 0.5 for values in lane_values)
            summary["gt_lanes"] = lane_count
            summary["best_of_four_anchor_recall_050"] = best_anchor_hits / max(
                lane_count, 1
            )
            summary["best_of_four_corrected_recall_050"] = (
                best_corrected_hits / max(lane_count, 1)
            )
            source_result[bucket_name] = summary
        result["sources"][source_name] = source_result
    return result


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    channels_last = (
        bool(cfg.get("training", {}).get("channels_last", False))
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    input_w = int(model_cfg.get("input_w", 800))
    input_h = int(model_cfg.get("input_h", 288))
    num_rows = int(model_cfg.get("num_rows", 72))
    fpn_channels = int(model_cfg.get("fpn_channels", 128))
    dim = int(model_cfg.get("dim", fpn_channels))
    common_channels = max(
        int(model.encoder.backbone.out_channels["c2"]),
        fpn_channels,
        dim,
    )
    structured_cfg = model_cfg["structured_query"]
    num_instances = int(structured_cfg["num_instances"])
    num_groups = int(structured_cfg.get("num_groups", 1))
    group_size = num_instances // max(num_groups, 1)
    offsets = torch.tensor(args.offsets_px, device=device, dtype=torch.float32)
    if not bool((offsets[1:] > offsets[:-1]).all()):
        raise ValueError("--offsets-px must be strictly increasing")
    row_indices = _selected_row_indices(num_rows, int(args.probe_rows), device)

    probes = {
        name: CurveAlignedSequenceProbe(
            common_channels=common_channels,
            hidden_dim=int(args.hidden_dim),
            num_rows=int(args.probe_rows),
            offsets_px=[float(value) for value in args.offsets_px],
            max_scales=3,
            num_layers=int(args.probe_layers),
        ).to(device)
        for name in SOURCE_NAMES
    }
    # Source comparisons must not inherit a lucky random initialization.
    # Each probe is an independent module, but all start from identical weights.
    if not args.load_probes:
        reference_state = probes[SOURCE_NAMES[0]].state_dict()
        for source_name in SOURCE_NAMES[1:]:
            probes[source_name].load_state_dict(reference_state, strict=True)
    parameter_counts = {
        name: sum(parameter.numel() for parameter in probe.parameters())
        for name, probe in probes.items()
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Probe parameter counts differ: {parameter_counts}")
    probe_training_steps = int(args.train_steps)
    if args.load_probes:
        saved_payload = torch.load(args.load_probes, map_location="cpu")
        for name, probe in probes.items():
            probe.load_state_dict(saved_payload["probes"][name], strict=True)
        probe_training_steps = int(
            saved_payload.get("args", {}).get("train_steps", probe_training_steps)
        )

    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    history: list[dict[str, Any]] = []
    if int(args.train_steps) > 0 and not args.load_probes:
        train_loader = build_dataloader(cfg, split="train", training=True)
        optimizer = torch.optim.AdamW(
            [parameter for probe in probes.values() for parameter in probe.parameters()],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        train_iterator = iter(train_loader)
        for probe in probes.values():
            probe.train()
        progress = tqdm(range(1, int(args.train_steps) + 1), desc="GT-curve probe train", ncols=96)
        for step in progress:
            try:
                images, targets, _metas = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                images, targets, _metas = next(train_iterator)
            images = images.to(
                device,
                non_blocking=True,
                memory_format=(
                    torch.channels_last if channels_last else torch.contiguous_format
                ),
            )
            targets = nested_to_device(targets, device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                sources, _p2 = _extract_frozen_sources(model, images)
            examples = _build_lane_examples(
                targets,
                row_indices=row_indices,
                input_w=input_w,
                max_offset=float(offsets.abs().max()),
                training=True,
                max_train_shift=float(args.max_train_shift_px),
            )
            if examples is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            losses: dict[str, torch.Tensor] = {}
            metrics: dict[str, dict[str, float]] = {}
            for source_name in SOURCE_NAMES:
                profiles, scale_mask = _profiles_for_source(
                    sources[source_name],
                    examples,
                    offsets,
                    common_channels=common_channels,
                    max_scales=3,
                    input_w=input_w,
                    input_h=input_h,
                )
                outputs = probes[source_name](profiles, scale_mask, examples.valid)
                loss, source_metrics = _probe_loss(
                    outputs,
                    examples.target_residual,
                    examples.valid,
                    offsets,
                    point_weight=float(args.point_loss_weight),
                )
                losses[source_name] = loss
                metrics[source_name] = source_metrics
            total_loss = torch.stack(list(losses.values())).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for probe in probes.values() for parameter in probe.parameters()],
                max_norm=5.0,
            )
            optimizer.step()
            record = {
                "step": step,
                "lane_examples": examples.count,
                "total": float(total_loss.detach()),
                **{
                    f"{source}/{key}": value
                    for source, values in metrics.items()
                    for key, value in values.items()
                },
            }
            if step == 1 or step % int(args.log_interval) == 0 or step == int(args.train_steps):
                history.append(record)
                progress.set_postfix(
                    {
                        name: f"{metrics[name]['total']:.3f}"
                        for name in SOURCE_NAMES
                    }
                )

    if args.save_probes:
        save_path = Path(args.save_probes)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "probes": {name: probe.state_dict() for name, probe in probes.items()},
                "args": vars(args),
                "parameter_counts": parameter_counts,
            },
            save_path,
        )

    eval_loader = build_dataloader(cfg, split="val", training=False)
    evaluation = _evaluate(
        model=model,
        probes=probes,
        loader=eval_loader,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        row_indices=row_indices,
        offsets=offsets,
        common_channels=common_channels,
        input_w=input_w,
        input_h=input_h,
        group_size=group_size,
        max_batches=int(args.eval_max_batches),
        line_width=float(args.line_width),
        base_hit_threshold=float(args.base_hit_threshold),
    )
    payload = {
        "diagnostic_only": True,
        "interpretation": (
            "GT geometry defines controlled offset corridors. High held-out correction "
            "on base-missed lanes means the frozen source contains curve-level evidence; "
            "poor correction means that source does not expose a readily decodable signal."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "train_split": "train",
        "eval_split": "val",
        "probe_training_steps": probe_training_steps,
        "evaluation_invocation_train_steps": int(args.train_steps),
        "probe_rows": int(args.probe_rows),
        "offsets_px": [float(value) for value in args.offsets_px],
        "evaluation_patterns_px": [
            "global -32",
            "global +32",
            "linear -32 to +32",
            "linear +32 to -32",
        ],
        "feature_sources": {
            "c2": "raw stride-4 backbone C2",
            "p2": "projected fused P2 consumed by LaneRowNet",
            "pyramid": "projected P2 plus trained top-down P3 and P4 states",
        },
        "fairness": (
            "All sources use identical augmented train samples, held-out validation "
            "lanes, offsets, sequence architecture, optimizer steps, and trainable "
            "parameter count. Lower-channel sources are zero-padded, not compressed."
        ),
        "probe_parameter_counts": parameter_counts,
        "training_history": history,
        "evaluation": evaluation,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
