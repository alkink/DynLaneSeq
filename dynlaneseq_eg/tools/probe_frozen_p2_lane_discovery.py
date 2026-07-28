from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import line_iou_against_gt
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import soft_expected_x
from dynlaneseq_eg.modeling.structured_queries import RowAwareCrossAttentionLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small lane-discovery head on frozen checkpoint P2 features and "
            "measure whether it recovers GT lanes missed by the structured decoder. "
            "The probe is diagnostic only and never updates the base model."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--probe-height", type=int, default=80)
    parser.add_argument("--probe-width", type=int, default=200)
    parser.add_argument("--decoder-rows", type=int, default=80)
    parser.add_argument("--heatmap-sigma", type=float, default=1.5)
    parser.add_argument("--curve-loss-weight", type=float, default=1.0)
    parser.add_argument("--curve-beta", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--peak-nms-radius", type=int, default=2)
    parser.add_argument("--eval-max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--load-probe", default="")
    parser.add_argument("--save-probe", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FrozenP2LaneDiscoveryProbe(nn.Module):
    """Endpoint heatmap plus a seed-conditioned row decoder on frozen P2."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_rows: int,
        decoder_rows: int,
        probe_height: int,
        probe_width: int,
        input_w: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_rows = int(num_rows)
        self.decoder_rows = int(decoder_rows)
        self.probe_height = int(probe_height)
        self.probe_width = int(probe_width)
        self.input_w = int(input_w)
        self.curve_bins = int(probe_width)
        self.tower = nn.Sequential(
            nn.Conv2d(int(in_dim) + 2, int(hidden_dim), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, int(hidden_dim)),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, int(hidden_dim)),
            nn.GELU(),
        )
        self.heatmap = nn.Conv2d(int(hidden_dim), 1, kernel_size=1)
        self.seed_coord = nn.Sequential(
            nn.Linear(2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.row_embedding = nn.Embedding(self.decoder_rows, int(hidden_dim))
        self.x_embedding = nn.Embedding(self.probe_width, int(hidden_dim))
        self.row_decoder = RowAwareCrossAttentionLayer(
            dim=int(hidden_dim),
            num_heads=4,
            ff_dim=4 * int(hidden_dim),
            dropout=0.0,
            num_groups=1,
        )
        self.row_x = nn.Linear(int(hidden_dim), self.curve_bins)
        nn.init.constant_(self.heatmap.bias, -4.6)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.x_embedding.weight, std=0.02)

    def forward(self, p2: torch.Tensor) -> dict[str, torch.Tensor]:
        resized = F.interpolate(
            p2,
            size=(self.probe_height, self.probe_width),
            mode="bilinear",
            align_corners=False,
        )
        batch = int(resized.shape[0])
        yy = torch.linspace(-1.0, 1.0, self.probe_height, device=resized.device, dtype=resized.dtype)
        xx = torch.linspace(-1.0, 1.0, self.probe_width, device=resized.device, dtype=resized.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        features = self.tower(torch.cat((resized, coords), dim=1))
        return {
            "heatmap_logits": self.heatmap(features).squeeze(1),
            "hidden": features,
        }

    def decode_curves(
        self,
        hidden: torch.Tensor,
        seed_yx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode full row curves from spatially distinct endpoint seeds."""
        if hidden.shape[0] != 1:
            raise ValueError("decode_curves currently expects one image at a time")
        if seed_yx.ndim != 2 or seed_yx.shape[-1] != 2:
            raise ValueError(f"seed_yx must have shape [N,2], got {tuple(seed_yx.shape)}")
        num_seeds = int(seed_yx.shape[0])
        if num_seeds == 0:
            return {
                "row_x_logits": hidden.new_zeros((1, 0, self.decoder_rows, self.curve_bins)),
                "pred_x_rows": hidden.new_zeros((1, 0, self.num_rows)),
            }
        seed_y = seed_yx[:, 0].long().clamp(0, self.probe_height - 1)
        seed_x = seed_yx[:, 1].long().clamp(0, self.probe_width - 1)
        seed_features = hidden[0, :, seed_y, seed_x].transpose(0, 1).contiguous()
        normalized_x = seed_x.float() / float(max(self.probe_width - 1, 1)) * 2.0 - 1.0
        normalized_y = seed_y.float() / float(max(self.probe_height - 1, 1)) * 2.0 - 1.0
        seed_features = seed_features + self.seed_coord(
            torch.stack((normalized_x, normalized_y), dim=-1).to(dtype=hidden.dtype)
        )
        row = self.row_embedding.weight.to(device=hidden.device, dtype=hidden.dtype)
        row_tokens = seed_features[:, None, :] + row[None, :, :]
        row_tokens = row_tokens.unsqueeze(0)

        evidence = F.interpolate(
            hidden,
            size=(self.decoder_rows, self.probe_width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()
        x_pos = self.x_embedding.weight.to(device=hidden.device, dtype=hidden.dtype)
        keys = evidence + x_pos.view(1, 1, self.probe_width, self.hidden_dim)
        row_tokens = self.row_decoder(row_tokens, evidence, keys, num_groups=1)
        logits = self.row_x(row_tokens)
        decoded_x = soft_expected_x(
            logits,
            input_w=self.input_w,
            x_bins=self.curve_bins,
        )
        if self.decoder_rows == self.num_rows:
            pred_x_rows = decoded_x
        else:
            batch, lanes, _rows = decoded_x.shape
            pred_x_rows = F.interpolate(
                decoded_x.reshape(batch * lanes, 1, self.decoder_rows),
                size=self.num_rows,
                mode="linear",
                align_corners=False,
            ).reshape(batch, lanes, self.num_rows)
        return {
            "row_x_logits": logits,
            "pred_x_rows": pred_x_rows,
        }


def _gaussian_heatmap_targets(
    targets: list[dict[str, torch.Tensor]],
    *,
    height: int,
    width: int,
    num_rows: int,
    input_w: float,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[list[tuple[int, int, int]]]]:
    heatmap = torch.zeros((len(targets), height, width), device=device, dtype=dtype)
    seeds: list[list[tuple[int, int, int]]] = [[] for _ in targets]
    radius = max(int(math.ceil(3.0 * float(sigma))), 1)
    for batch_index, target in enumerate(targets):
        x_rows = target["x_rows"].to(device=device, dtype=dtype)
        valid_mask = target["valid_mask"].to(device=device).bool()
        for lane_index in range(int(x_rows.shape[0])):
            valid_rows = valid_mask[lane_index].nonzero(as_tuple=False).flatten()
            if valid_rows.numel() == 0:
                continue
            seed_row = int(valid_rows[-1])
            seed_x = float(x_rows[lane_index, seed_row])
            y = int(round(seed_row * float(height - 1) / float(max(num_rows - 1, 1))))
            x = int(round(seed_x * float(width - 1) / float(input_w)))
            y = max(0, min(height - 1, y))
            x = max(0, min(width - 1, x))
            seeds[batch_index].append((lane_index, y, x))
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            grid_y = torch.arange(y0, y1, device=device, dtype=dtype)
            grid_x = torch.arange(x0, x1, device=device, dtype=dtype)
            distance = (grid_y[:, None] - float(y)).pow(2) + (grid_x[None, :] - float(x)).pow(2)
            gaussian = torch.exp(-0.5 * distance / float(sigma * sigma))
            heatmap[batch_index, y0:y1, x0:x1] = torch.maximum(
                heatmap[batch_index, y0:y1, x0:x1],
                gaussian,
            )
    return heatmap, seeds


def _centernet_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits.float()).clamp(1e-5, 1.0 - 1e-5)
    target = target.float()
    positive = target >= 1.0 - 1e-6
    negative = ~positive
    negative_weight = (1.0 - target).pow(4.0)
    positive_loss = -(probabilities.log() * (1.0 - probabilities).pow(2.0) * positive).sum()
    negative_loss = -(
        (1.0 - probabilities).log()
        * probabilities.pow(2.0)
        * negative_weight
        * negative
    ).sum()
    num_positive = positive.sum().clamp_min(1)
    return (positive_loss + negative_loss) / num_positive


def _curve_loss(
    probe: FrozenP2LaneDiscoveryProbe,
    hidden: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    seeds: list[list[tuple[int, int, int]]],
    *,
    input_w: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_ce = hidden.sum() * 0.0
    total_point = hidden.sum() * 0.0
    ce_count = 0
    point_count = 0
    decoder_row_indices = torch.round(
        torch.linspace(
            0,
            probe.num_rows - 1,
            probe.decoder_rows,
            device=hidden.device,
        )
    ).long()
    for batch_index, sample_seeds in enumerate(seeds):
        if not sample_seeds:
            continue
        target = targets[batch_index]
        x_rows = target["x_rows"].to(device=hidden.device, dtype=hidden.dtype)
        valid_mask = target["valid_mask"].to(device=hidden.device).bool()
        seed_yx = torch.tensor(
            [[y, x] for _lane_index, y, x in sample_seeds],
            device=hidden.device,
            dtype=torch.long,
        )
        decoded = probe.decode_curves(hidden[batch_index : batch_index + 1], seed_yx)
        logits = decoded["row_x_logits"][0]
        predictions = decoded["pred_x_rows"][0]
        for proposal_index, (lane_index, _y, _x) in enumerate(sample_seeds):
            valid = valid_mask[lane_index]
            if not bool(valid.any()):
                continue
            gt_x = x_rows[lane_index]
            gt_bins = torch.round(
                gt_x / float(input_w) * float(probe.curve_bins - 1)
            ).long().clamp(0, probe.curve_bins - 1)
            decoder_valid = valid[decoder_row_indices]
            if bool(decoder_valid.any()):
                total_ce = total_ce + F.cross_entropy(
                    logits[proposal_index, decoder_valid],
                    gt_bins[decoder_row_indices][decoder_valid],
                    reduction="sum",
                )
                ce_count += int(decoder_valid.sum())
            total_point = total_point + F.smooth_l1_loss(
                predictions[proposal_index, valid] / float(input_w),
                gt_x[valid] / float(input_w),
                beta=float(beta),
                reduction="sum",
            )
            point_count += int(valid.sum())
    ce = total_ce / max(ce_count, 1)
    point = total_point / max(point_count, 1)
    return ce + 5.0 * point, ce, point


def _topk_seeds(
    outputs: dict[str, torch.Tensor],
    *,
    top_k: int,
    nms_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    heatmap = torch.sigmoid(outputs["heatmap_logits"].float())
    kernel = 2 * int(nms_radius) + 1
    pooled = F.max_pool2d(heatmap.unsqueeze(1), kernel_size=kernel, stride=1, padding=int(nms_radius)).squeeze(1)
    local = heatmap.masked_fill(heatmap < pooled, 0.0)
    flat = local.flatten(1)
    k = min(int(top_k), int(flat.shape[1]))
    scores, indices = torch.topk(flat, k=k, dim=1)
    width = int(heatmap.shape[-1])
    peak_y = torch.div(indices, width, rounding_mode="floor")
    peak_x = indices % width
    peaks = torch.stack((peak_y, peak_x), dim=-1)
    return scores, peaks


def _best_iou_per_gt(
    candidates: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    line_width: float,
) -> torch.Tensor:
    gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
    valid = target["valid_mask"].to(device=candidates.device).bool()
    values: list[torch.Tensor] = []
    for lane_index in range(int(gt_x.shape[0])):
        if int(valid[lane_index].sum()) < 5:
            continue
        ious = line_iou_against_gt(
            candidates,
            gt_x[lane_index],
            valid[lane_index],
            line_width=float(line_width),
        )
        values.append(ious.max() if ious.numel() else candidates.new_zeros(()))
    return torch.stack(values).float() if values else candidates.new_zeros((0,), dtype=torch.float32)


@dataclass
class ProbeMetrics:
    gt: int = 0
    base_hits_050: int = 0
    base_hits_070: int = 0
    probe_hits_050: int = 0
    probe_hits_070: int = 0
    union_hits_050: int = 0
    union_hits_070: int = 0
    base_misses_050: int = 0
    recovered_base_misses_050: int = 0
    base_misses_070: int = 0
    recovered_base_misses_070: int = 0
    seed_total: int = 0
    seed_hits_16px: int = 0
    seed_hits_32px: int = 0
    seed_hits_64px: int = 0
    missed_seed_total_050: int = 0
    missed_seed_hits_32px_050: int = 0
    proposal_gt_hits_050: int = 0
    proposal_unique_gt_hits_050: int = 0
    teacher_seed_paired_hits_050: int = 0
    teacher_seed_paired_hits_070: int = 0
    teacher_seed_best_hits_050: int = 0
    teacher_seed_best_hits_070: int = 0

    def summary(self) -> dict[str, float | int]:
        return {
            "gt_lanes": self.gt,
            "base_group_recall_050": self.base_hits_050 / max(self.gt, 1),
            "base_group_recall_070": self.base_hits_070 / max(self.gt, 1),
            "probe_recall_050": self.probe_hits_050 / max(self.gt, 1),
            "probe_recall_070": self.probe_hits_070 / max(self.gt, 1),
            "union_recall_050": self.union_hits_050 / max(self.gt, 1),
            "union_recall_070": self.union_hits_070 / max(self.gt, 1),
            "union_gain_050_points": 100.0
            * (self.union_hits_050 - self.base_hits_050)
            / max(self.gt, 1),
            "union_gain_070_points": 100.0
            * (self.union_hits_070 - self.base_hits_070)
            / max(self.gt, 1),
            "base_misses_050": self.base_misses_050,
            "recovered_base_misses_050": self.recovered_base_misses_050,
            "miss_recovery_fraction_050": self.recovered_base_misses_050
            / max(self.base_misses_050, 1),
            "base_misses_070": self.base_misses_070,
            "recovered_base_misses_070": self.recovered_base_misses_070,
            "miss_recovery_fraction_070": self.recovered_base_misses_070
            / max(self.base_misses_070, 1),
            "seed_recall_16px": self.seed_hits_16px / max(self.seed_total, 1),
            "seed_recall_32px": self.seed_hits_32px / max(self.seed_total, 1),
            "seed_recall_64px": self.seed_hits_64px / max(self.seed_total, 1),
            "missed_lane_seed_recall_32px_050": self.missed_seed_hits_32px_050
            / max(self.missed_seed_total_050, 1),
            "probe_duplicate_fraction_050": (
                self.proposal_gt_hits_050 - self.proposal_unique_gt_hits_050
            )
            / max(self.proposal_gt_hits_050, 1),
            "teacher_seed_paired_recall_050": self.teacher_seed_paired_hits_050
            / max(self.gt, 1),
            "teacher_seed_paired_recall_070": self.teacher_seed_paired_hits_070
            / max(self.gt, 1),
            "teacher_seed_best_recall_050": self.teacher_seed_best_hits_050
            / max(self.gt, 1),
            "teacher_seed_best_recall_070": self.teacher_seed_best_hits_070
            / max(self.gt, 1),
        }


def _seed_distances_px(
    peaks: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    probe_height: int,
    probe_width: int,
    num_rows: int,
    input_w: float,
    input_h: float,
) -> torch.Tensor:
    x_rows = target["x_rows"].to(device=peaks.device, dtype=torch.float32)
    valid_mask = target["valid_mask"].to(device=peaks.device).bool()
    distances: list[torch.Tensor] = []
    for lane_index in range(int(x_rows.shape[0])):
        valid_rows = valid_mask[lane_index].nonzero(as_tuple=False).flatten()
        if valid_rows.numel() == 0:
            continue
        seed_row = valid_rows[-1].float()
        seed_x = x_rows[lane_index, valid_rows[-1]]
        gt_y = seed_row * float(probe_height - 1) / float(max(num_rows - 1, 1))
        gt_x = seed_x * float(probe_width - 1) / float(input_w)
        dy = (peaks[:, 0].float() - gt_y) * float(input_h) / float(probe_height)
        dx = (peaks[:, 1].float() - gt_x) * float(input_w) / float(probe_width)
        distances.append(torch.sqrt(dx.pow(2) + dy.pow(2)).min())
    return torch.stack(distances) if distances else peaks.new_zeros((0,), dtype=torch.float32)


def _update_duplicate_stats(
    metrics: ProbeMetrics,
    candidates: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    line_width: float,
) -> None:
    gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
    valid = target["valid_mask"].to(device=candidates.device).bool()
    assigned: list[int] = []
    for candidate_index in range(int(candidates.shape[0])):
        best_iou = -1.0
        best_gt = -1
        for gt_index in range(int(gt_x.shape[0])):
            if int(valid[gt_index].sum()) < 5:
                continue
            iou = line_iou_against_gt(
                candidates[candidate_index : candidate_index + 1],
                gt_x[gt_index],
                valid[gt_index],
                line_width=float(line_width),
            )
            value = float(iou[0]) if iou.numel() else 0.0
            if value > best_iou:
                best_iou = value
                best_gt = gt_index
        if best_iou >= 0.5 and best_gt >= 0:
            assigned.append(best_gt)
    metrics.proposal_gt_hits_050 += len(assigned)
    metrics.proposal_unique_gt_hits_050 += len(set(assigned))


def _teacher_seed_coordinates(
    target: dict[str, torch.Tensor],
    *,
    probe_height: int,
    probe_width: int,
    num_rows: int,
    input_w: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    x_rows = target["x_rows"].to(device=device, dtype=torch.float32)
    valid_mask = target["valid_mask"].to(device=device).bool()
    coordinates: list[list[int]] = []
    lane_indices: list[int] = []
    for lane_index in range(int(x_rows.shape[0])):
        valid_rows = valid_mask[lane_index].nonzero(as_tuple=False).flatten()
        if valid_rows.numel() == 0:
            continue
        seed_row = int(valid_rows[-1])
        seed_x = float(x_rows[lane_index, seed_row])
        y = int(round(seed_row * float(probe_height - 1) / float(max(num_rows - 1, 1))))
        x = int(round(seed_x * float(probe_width - 1) / float(input_w)))
        coordinates.append(
            [
                max(0, min(probe_height - 1, y)),
                max(0, min(probe_width - 1, x)),
            ]
        )
        lane_indices.append(lane_index)
    if not coordinates:
        return torch.zeros((0, 2), device=device, dtype=torch.long), lane_indices
    return torch.tensor(coordinates, device=device, dtype=torch.long), lane_indices


def _update_teacher_seed_stats(
    metrics: ProbeMetrics,
    teacher_candidates: torch.Tensor,
    lane_indices: list[int],
    target: dict[str, torch.Tensor],
    *,
    line_width: float,
) -> None:
    gt_x = target["x_rows"].to(device=teacher_candidates.device, dtype=teacher_candidates.dtype)
    valid = target["valid_mask"].to(device=teacher_candidates.device).bool()
    paired_values: list[torch.Tensor] = []
    for proposal_index, lane_index in enumerate(lane_indices):
        iou = line_iou_against_gt(
            teacher_candidates[proposal_index : proposal_index + 1],
            gt_x[lane_index],
            valid[lane_index],
            line_width=float(line_width),
        )
        paired_values.append(iou[0] if iou.numel() else teacher_candidates.new_zeros(()))
    if paired_values:
        paired = torch.stack(paired_values)
        metrics.teacher_seed_paired_hits_050 += int((paired >= 0.5).sum())
        metrics.teacher_seed_paired_hits_070 += int((paired >= 0.7).sum())
    best = _best_iou_per_gt(teacher_candidates, target, line_width=line_width)
    metrics.teacher_seed_best_hits_050 += int((best >= 0.5).sum())
    metrics.teacher_seed_best_hits_070 += int((best >= 0.7).sum())


@torch.no_grad()
def evaluate_probe(
    base_model: nn.Module,
    probe: FrozenP2LaneDiscoveryProbe,
    loader: Iterable,
    *,
    device: torch.device,
    channels_last: bool,
    amp_dtype: torch.dtype | None,
    max_batches: int,
    top_k: int,
    nms_radius: int,
    input_w: float,
    input_h: float,
    num_rows: int,
    num_instances: int,
    num_groups: int,
    line_width: float,
) -> ProbeMetrics:
    metrics = ProbeMetrics()
    group_size = num_instances // max(num_groups, 1)
    base_model.eval()
    probe.eval()
    autocast_enabled = amp_dtype is not None and device.type == "cuda"
    for batch_index, (images, targets, _metas) in enumerate(
        tqdm(loader, ncols=88, desc="frozen-P2 probe eval")
    ):
        if max_batches > 0 and batch_index >= max_batches:
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format,
        )
        amp_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if autocast_enabled
            else nullcontext()
        )
        with amp_context:
            features = base_model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )["features"]
            base_outputs = base_model.structured_query_head(features)
        probe_outputs = probe(features.float())
        _scores, peaks = _topk_seeds(
            probe_outputs,
            top_k=top_k,
            nms_radius=nms_radius,
        )

        for sample_index, target in enumerate(targets):
            base_candidates = base_outputs["pred_x_rows"][sample_index, :group_size].float()
            decoded = probe.decode_curves(
                probe_outputs["hidden"][sample_index : sample_index + 1],
                peaks[sample_index],
            )
            probe_candidates = decoded["pred_x_rows"][0].float()
            teacher_yx, teacher_lane_indices = _teacher_seed_coordinates(
                target,
                probe_height=probe.probe_height,
                probe_width=probe.probe_width,
                num_rows=num_rows,
                input_w=input_w,
                device=device,
            )
            teacher_decoded = probe.decode_curves(
                probe_outputs["hidden"][sample_index : sample_index + 1],
                teacher_yx,
            )
            teacher_candidates = teacher_decoded["pred_x_rows"][0].float()
            base_iou = _best_iou_per_gt(base_candidates, target, line_width=line_width)
            probe_iou = _best_iou_per_gt(probe_candidates, target, line_width=line_width)
            if base_iou.shape != probe_iou.shape:
                raise RuntimeError("Base/probe GT count mismatch")
            union_iou = torch.maximum(base_iou, probe_iou)
            metrics.gt += int(base_iou.numel())
            metrics.base_hits_050 += int((base_iou >= 0.5).sum())
            metrics.base_hits_070 += int((base_iou >= 0.7).sum())
            metrics.probe_hits_050 += int((probe_iou >= 0.5).sum())
            metrics.probe_hits_070 += int((probe_iou >= 0.7).sum())
            metrics.union_hits_050 += int((union_iou >= 0.5).sum())
            metrics.union_hits_070 += int((union_iou >= 0.7).sum())
            miss_050 = base_iou < 0.5
            miss_070 = base_iou < 0.7
            metrics.base_misses_050 += int(miss_050.sum())
            metrics.base_misses_070 += int(miss_070.sum())
            metrics.recovered_base_misses_050 += int((miss_050 & (probe_iou >= 0.5)).sum())
            metrics.recovered_base_misses_070 += int((miss_070 & (probe_iou >= 0.7)).sum())

            distances = _seed_distances_px(
                peaks[sample_index],
                target,
                probe_height=probe.probe_height,
                probe_width=probe.probe_width,
                num_rows=num_rows,
                input_w=input_w,
                input_h=input_h,
            )
            metrics.seed_total += int(distances.numel())
            metrics.seed_hits_16px += int((distances <= 16.0).sum())
            metrics.seed_hits_32px += int((distances <= 32.0).sum())
            metrics.seed_hits_64px += int((distances <= 64.0).sum())
            if distances.shape == miss_050.shape:
                metrics.missed_seed_total_050 += int(miss_050.sum())
                metrics.missed_seed_hits_32px_050 += int(((distances <= 32.0) & miss_050).sum())
            _update_duplicate_stats(
                metrics,
                probe_candidates,
                target,
                line_width=line_width,
            )
            _update_teacher_seed_stats(
                metrics,
                teacher_candidates,
                teacher_lane_indices,
                target,
                line_width=line_width,
            )
    return metrics


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    dataloader_cfg = cfg.setdefault("dataloader", {})
    dataloader_cfg["num_workers"] = int(args.num_workers)
    dataloader_cfg["eval_batch_size"] = int(args.eval_batch_size)
    dataloader_cfg["persistent_workers"] = bool(int(args.num_workers) > 0)
    training_cfg = cfg.setdefault("training", {})
    training_cfg["batch_size"] = int(args.batch_size)
    training_cfg["seed"] = int(args.seed)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    channels_last = bool(training_cfg.get("channels_last", False)) and device.type == "cuda"
    base_model = build_model(cfg)
    load_checkpoint(args.checkpoint, base_model, strict=False)
    base_model.requires_grad_(False)
    base_model = base_model.to(device).eval()
    if channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)

    num_rows = int(model_cfg.get("num_rows", 72))
    input_w = float(model_cfg.get("input_w", 800))
    input_h = float(model_cfg.get("input_h", 288))
    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(structured_cfg.get("num_instances", model_cfg.get("num_slots", 0)))
    num_groups = int(structured_cfg.get("num_groups", 1))
    probe = FrozenP2LaneDiscoveryProbe(
        in_dim=int(model_cfg.get("dim", 256)),
        hidden_dim=int(args.hidden_dim),
        num_rows=num_rows,
        decoder_rows=int(args.decoder_rows),
        probe_height=int(args.probe_height),
        probe_width=int(args.probe_width),
        input_w=int(input_w),
    ).to(device)
    if args.load_probe:
        try:
            probe_payload = torch.load(args.load_probe, map_location="cpu", weights_only=False)
        except TypeError:
            probe_payload = torch.load(args.load_probe, map_location="cpu")
        state_dict = probe_payload.get("probe", probe_payload)
        probe.load_state_dict(state_dict, strict=True)
        print(f"loaded_probe: {args.load_probe}")
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_loader = build_dataloader(cfg, split="train", training=True)
    eval_loader = build_dataloader(cfg, split="val", training=False)

    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    if amp_dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    autocast_enabled = amp_dtype is not None and device.type == "cuda"

    probe.train()
    train_iterator = iter(train_loader)
    running = {"total": 0.0, "heatmap": 0.0, "curve": 0.0}
    for step in range(1, int(args.train_steps) + 1):
        try:
            images, targets, _metas = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, targets, _metas = next(train_iterator)
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format,
        )
        with torch.no_grad():
            amp_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if autocast_enabled
                else nullcontext()
            )
            with amp_context:
                p2 = base_model.encoder.forward_features(
                    images,
                    inference_only=True,
                    structured_only=True,
                )["features"]
        outputs = probe(p2.float())
        heatmap_target, seeds = _gaussian_heatmap_targets(
            targets,
            height=probe.probe_height,
            width=probe.probe_width,
            num_rows=num_rows,
            input_w=input_w,
            sigma=float(args.heatmap_sigma),
            device=device,
            dtype=outputs["heatmap_logits"].dtype,
        )
        loss_heatmap = _centernet_focal_loss(outputs["heatmap_logits"], heatmap_target)
        loss_curve, loss_curve_ce, loss_curve_point = _curve_loss(
            probe,
            outputs["hidden"],
            targets,
            seeds,
            input_w=input_w,
            beta=float(args.curve_beta),
        )
        loss_total = loss_heatmap + float(args.curve_loss_weight) * loss_curve
        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        nn.utils.clip_grad_norm_(probe.parameters(), max_norm=5.0)
        optimizer.step()

        running["total"] += float(loss_total.detach())
        running["heatmap"] += float(loss_heatmap.detach())
        running["curve"] += float(loss_curve.detach())
        if step % int(args.log_interval) == 0 or step == int(args.train_steps):
            denom = int(args.log_interval) if step % int(args.log_interval) == 0 else step % int(args.log_interval)
            print(
                f"probe step {step:05d}/{int(args.train_steps):05d} | "
                f"loss {running['total'] / max(denom, 1):.4f} | "
                f"heatmap {running['heatmap'] / max(denom, 1):.4f} | "
                f"curve {running['curve'] / max(denom, 1):.4f} | "
                f"curve_ce {float(loss_curve_ce.detach()):.4f} | "
                f"curve_point {float(loss_curve_point.detach()):.4f}",
                flush=True,
            )
            running = {"total": 0.0, "heatmap": 0.0, "curve": 0.0}

    metrics = evaluate_probe(
        base_model,
        probe,
        eval_loader,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        max_batches=int(args.eval_max_batches),
        top_k=int(args.top_k),
        nms_radius=int(args.peak_nms_radius),
        input_w=input_w,
        input_h=input_h,
        num_rows=num_rows,
        num_instances=num_instances,
        num_groups=num_groups,
        line_width=float(args.line_width),
    )
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    payload = {
        "diagnostic_only": True,
        "base_model_frozen": True,
        "train_split": "train",
        "eval_split": "val",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed": int(args.seed),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "probe_parameters": int(parameter_count),
        "probe_shape": {
            "height": probe.probe_height,
            "width": probe.probe_width,
            "decoder_rows": probe.decoder_rows,
            "top_k": int(args.top_k),
        },
        "metrics": metrics.summary(),
    }
    print(json.dumps(payload, indent=2))

    if args.save_probe:
        probe_path = Path(args.save_probe)
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"probe": probe.state_dict(), "metadata": payload}, probe_path)
        print(f"probe_checkpoint: {probe_path}")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
