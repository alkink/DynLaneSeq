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
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a mature LaneRowNet checkpoint and train only a small "
            "query-conditioned dense curve head. The probe tests whether P2 and "
            "the frozen lane-row states contain lane-specific spatial evidence "
            "that the current row output head fails to acquire. It is diagnostic "
            "only and never updates the checkpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--evidence-width", type=int, default=400)
    parser.add_argument(
        "--conditioning-mode",
        choices=("per_row", "lane_shared"),
        default="per_row",
        help=(
            "per_row reproduces the original flexible probe. lane_shared "
            "generates one dynamic visual filter per lane and reuses it over "
            "all rows, matching the whole-lane coherence bias of CondLSTR."
        ),
    )
    parser.add_argument(
        "--explicit-coordinates",
        action="store_true",
        help=(
            "Append normalized image-row and horizontal coordinates to the "
            "visual evidence before the probe tower, as in dynamic mask heads."
        ),
    )
    parser.add_argument(
        "--state-source",
        choices=("final", "initial"),
        default="final",
        help=(
            "Use the frozen final decoder row states, or the image-blind "
            "instance+row states before the first decoder block."
        ),
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--eval-max-batches", type=int, default=16)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--load-probe", default="")
    parser.add_argument("--save-probe", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


class QueryConditionedDenseCurveProbe(nn.Module):
    """Dense lane-row association from frozen row states and frozen P2.

    For every lane-row state, the probe produces a dynamic vector and scores it
    against every horizontal P2 location on that row. This is deliberately more
    direct than the model's current ``Linear(D, K)`` output head: the final
    coordinate is still conditioned on image evidence at prediction time.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        state_dim: int,
        hidden_dim: int,
        num_rows: int,
        evidence_width: int,
        input_w: int,
        conditioning_mode: str = "per_row",
        explicit_coordinates: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % 8 != 0:
            raise ValueError("hidden_dim must be divisible by 8 for GroupNorm")
        conditioning_mode = str(conditioning_mode).strip().lower()
        if conditioning_mode not in {"per_row", "lane_shared"}:
            raise ValueError(
                "conditioning_mode must be 'per_row' or 'lane_shared'"
            )
        self.num_rows = int(num_rows)
        self.evidence_width = int(evidence_width)
        self.input_w = int(input_w)
        self.hidden_dim = int(hidden_dim)
        self.conditioning_mode = conditioning_mode
        self.explicit_coordinates = bool(explicit_coordinates)
        feature_in_dim = int(in_dim) + (2 if self.explicit_coordinates else 0)
        self.feature_tower = nn.Sequential(
            nn.Conv2d(feature_in_dim, int(hidden_dim), kernel_size=1, bias=False),
            nn.GroupNorm(8, int(hidden_dim)),
            nn.GELU(),
            nn.Conv2d(
                int(hidden_dim),
                int(hidden_dim),
                kernel_size=3,
                padding=1,
                groups=int(hidden_dim),
                bias=False,
            ),
            nn.GroupNorm(8, int(hidden_dim)),
            nn.GELU(),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(int(state_dim)),
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.row_bias = nn.Sequential(
            nn.LayerNorm(int(state_dim)),
            nn.Linear(int(state_dim), 1),
        )

    def encode_features(self, p2: torch.Tensor) -> torch.Tensor:
        if p2.ndim != 4:
            raise ValueError(f"p2 must be [B,C,H,W], got {tuple(p2.shape)}")
        if p2.shape[-2:] != (self.num_rows, self.evidence_width):
            p2 = F.interpolate(
                p2,
                size=(self.num_rows, self.evidence_width),
                mode="bilinear",
                align_corners=False,
            )
        if self.explicit_coordinates:
            batch, _channels, height, width = p2.shape
            row_coord = torch.linspace(
                -1.0,
                1.0,
                height,
                device=p2.device,
                dtype=p2.dtype,
            ).view(1, 1, height, 1)
            x_coord = torch.linspace(
                -1.0,
                1.0,
                width,
                device=p2.device,
                dtype=p2.dtype,
            ).view(1, 1, 1, width)
            coordinates = torch.cat(
                [
                    row_coord.expand(batch, -1, -1, width),
                    x_coord.expand(batch, -1, height, -1),
                ],
                dim=1,
            )
            p2 = torch.cat([p2, coordinates], dim=1)
        return self.feature_tower(p2)

    def score_encoded(
        self,
        encoded_features: torch.Tensor,
        row_states: torch.Tensor,
    ) -> torch.Tensor:
        if row_states.ndim != 4:
            raise ValueError(
                f"row_states must be [B,N,R,D], got {tuple(row_states.shape)}"
            )
        if int(row_states.shape[2]) != self.num_rows:
            raise ValueError(
                f"row_states has {row_states.shape[2]} rows, expected {self.num_rows}"
            )
        if int(encoded_features.shape[0]) != int(row_states.shape[0]):
            raise ValueError("feature/state batch sizes differ")
        visual = encoded_features.permute(0, 2, 3, 1).contiguous()
        if self.conditioning_mode == "per_row":
            dynamic = self.state_projection(row_states)
            logits = torch.einsum("bnrd,brxd->bnrx", dynamic, visual)
            bias = self.row_bias(row_states)
        else:
            # A single dynamic filter must explain the complete lane.  Unlike
            # the per-row probe, it cannot select a different visual template
            # independently at every height.
            lane_state = row_states.mean(dim=2)
            dynamic = self.state_projection(lane_state)
            logits = torch.einsum("bnd,brxd->bnrx", dynamic, visual)
            bias = self.row_bias(lane_state).unsqueeze(2)
        logits = logits / math.sqrt(float(self.hidden_dim))
        return logits + bias

    def forward(self, p2: torch.Tensor, row_states: torch.Tensor) -> torch.Tensor:
        return self.score_encoded(self.encode_features(p2), row_states)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits.float(), dim=-1)
        centers = (
            torch.arange(
                self.evidence_width,
                device=logits.device,
                dtype=probabilities.dtype,
            )
            + 0.5
        ) * (float(self.input_w) / float(self.evidence_width))
        return (probabilities * centers).sum(dim=-1)


def _group_zero_matches(
    match: dict[str, torch.Tensor],
    *,
    group_size: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for pred_index, gt_index in zip(
        match["pred_indices"].tolist(),
        match["gt_indices"].tolist(),
    ):
        if 0 <= int(pred_index) < int(group_size):
            pairs.append((int(pred_index), int(gt_index)))
    return pairs


def matched_dense_curve_loss(
    logits: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    *,
    group_size: int,
    input_w: int,
) -> tuple[torch.Tensor, int, int]:
    """Cross-entropy over valid rows of group-zero matched lanes.

    The reduction first averages rows within each lane and then averages lanes.
    Short lanes therefore receive the same supervision weight as long lanes.
    """

    if logits.ndim != 4:
        raise ValueError("logits must have shape [B,N,R,X]")
    width = int(logits.shape[-1])
    lane_losses: list[torch.Tensor] = []
    valid_rows = 0
    for batch_index, (target, match) in enumerate(zip(targets, matches)):
        gt_x = target["x_rows"].to(device=logits.device, dtype=torch.float32)
        gt_valid = target["valid_mask"].to(device=logits.device).bool()
        for pred_index, gt_index in _group_zero_matches(
            match,
            group_size=int(group_size),
        ):
            valid = gt_valid[gt_index]
            if not bool(valid.any()):
                continue
            bins = torch.floor(
                gt_x[gt_index] / float(input_w) * float(width)
            ).long().clamp(0, width - 1)
            lane_losses.append(
                F.cross_entropy(
                    logits[batch_index, pred_index, valid].float(),
                    bins[valid],
                    reduction="mean",
                )
            )
            valid_rows += int(valid.sum())
    if not lane_losses:
        return logits.sum() * 0.0, 0, 0
    return torch.stack(lane_losses).mean(), len(lane_losses), valid_rows


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
    return (
        torch.stack(values).float()
        if values
        else candidates.new_zeros((0,), dtype=torch.float32)
    )


def _paired_iou(
    candidates: torch.Tensor,
    target: dict[str, torch.Tensor],
    pairs: list[tuple[int, int]],
    *,
    line_width: float,
) -> torch.Tensor:
    gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
    valid = target["valid_mask"].to(device=candidates.device).bool()
    values: list[torch.Tensor] = []
    for pred_index, gt_index in pairs:
        iou = line_iou_against_gt(
            candidates[pred_index : pred_index + 1],
            gt_x[gt_index],
            valid[gt_index],
            line_width=float(line_width),
        )
        values.append(iou[0] if iou.numel() else candidates.new_zeros(()))
    return (
        torch.stack(values).float()
        if values
        else candidates.new_zeros((0,), dtype=torch.float32)
    )


@dataclass
class CurveModeMetrics:
    gt_lanes: int = 0
    hits_050: int = 0
    hits_070: int = 0
    paired_lanes: int = 0
    paired_iou_sum: float = 0.0

    def update(self, best: torch.Tensor, paired: torch.Tensor) -> None:
        self.gt_lanes += int(best.numel())
        self.hits_050 += int((best >= 0.5).sum())
        self.hits_070 += int((best >= 0.7).sum())
        self.paired_lanes += int(paired.numel())
        self.paired_iou_sum += float(paired.sum())

    def summary(self) -> dict[str, float | int]:
        return {
            "gt_lanes": self.gt_lanes,
            "recall_050": self.hits_050 / max(self.gt_lanes, 1),
            "recall_070": self.hits_070 / max(self.gt_lanes, 1),
            "paired_mean_iou": self.paired_iou_sum / max(self.paired_lanes, 1),
        }


@dataclass
class ProbeComparison:
    gt_lanes: int = 0
    base_hits_050: int = 0
    base_hits_070: int = 0
    union_hits_050: int = 0
    union_hits_070: int = 0
    recovered_misses_050: int = 0
    recovered_misses_070: int = 0
    base_misses_050: int = 0
    base_misses_070: int = 0
    paired_lanes: int = 0
    paired_base_iou_sum: float = 0.0
    paired_probe_iou_sum: float = 0.0
    paired_improved: int = 0
    paired_worsened: int = 0

    def update(
        self,
        base_best: torch.Tensor,
        probe_best: torch.Tensor,
        base_paired: torch.Tensor,
        probe_paired: torch.Tensor,
    ) -> None:
        union = torch.maximum(base_best, probe_best)
        miss_050 = base_best < 0.5
        miss_070 = base_best < 0.7
        self.gt_lanes += int(base_best.numel())
        self.base_hits_050 += int((base_best >= 0.5).sum())
        self.base_hits_070 += int((base_best >= 0.7).sum())
        self.union_hits_050 += int((union >= 0.5).sum())
        self.union_hits_070 += int((union >= 0.7).sum())
        self.base_misses_050 += int(miss_050.sum())
        self.base_misses_070 += int(miss_070.sum())
        self.recovered_misses_050 += int((miss_050 & (probe_best >= 0.5)).sum())
        self.recovered_misses_070 += int((miss_070 & (probe_best >= 0.7)).sum())
        if base_paired.numel() != probe_paired.numel():
            raise ValueError("base/probe paired counts differ")
        self.paired_lanes += int(base_paired.numel())
        self.paired_base_iou_sum += float(base_paired.sum())
        self.paired_probe_iou_sum += float(probe_paired.sum())
        self.paired_improved += int((probe_paired > base_paired + 0.02).sum())
        self.paired_worsened += int((probe_paired < base_paired - 0.02).sum())

    def summary(self) -> dict[str, float | int]:
        base_recall_050 = self.base_hits_050 / max(self.gt_lanes, 1)
        base_recall_070 = self.base_hits_070 / max(self.gt_lanes, 1)
        union_recall_050 = self.union_hits_050 / max(self.gt_lanes, 1)
        union_recall_070 = self.union_hits_070 / max(self.gt_lanes, 1)
        return {
            "gt_lanes": self.gt_lanes,
            "base_recall_050": base_recall_050,
            "base_recall_070": base_recall_070,
            "union_recall_050": union_recall_050,
            "union_recall_070": union_recall_070,
            "union_gain_050_points": 100.0 * (union_recall_050 - base_recall_050),
            "union_gain_070_points": 100.0 * (union_recall_070 - base_recall_070),
            "base_misses_050": self.base_misses_050,
            "recovered_base_misses_050": self.recovered_misses_050,
            "miss_recovery_fraction_050": self.recovered_misses_050
            / max(self.base_misses_050, 1),
            "base_misses_070": self.base_misses_070,
            "recovered_base_misses_070": self.recovered_misses_070,
            "miss_recovery_fraction_070": self.recovered_misses_070
            / max(self.base_misses_070, 1),
            "paired_base_mean_iou": self.paired_base_iou_sum
            / max(self.paired_lanes, 1),
            "paired_probe_mean_iou": self.paired_probe_iou_sum
            / max(self.paired_lanes, 1),
            "paired_improved_gt_2pts": self.paired_improved,
            "paired_worsened_gt_2pts": self.paired_worsened,
        }


def _roll_batch(tensor: torch.Tensor) -> torch.Tensor:
    if int(tensor.shape[0]) <= 1:
        return tensor.flip(-1)
    return tensor.roll(shifts=1, dims=0)


def probe_row_states(
    structured_head: nn.Module,
    outputs: dict[str, Any],
    *,
    source: str,
    group_size: int,
    batch_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    source = str(source).strip().lower()
    if source == "final":
        return outputs["structured_row_tokens"][:, :group_size].to(dtype=dtype)
    if source != "initial":
        raise ValueError(f"Unsupported state source: {source!r}")
    instance_tokens = getattr(structured_head, "instance_tokens").weight[:group_size]
    row_tokens = getattr(structured_head, "row_tokens").weight
    initial = instance_tokens[:, None, :] + row_tokens[None, :, :]
    return (
        initial.unsqueeze(0)
        .expand(int(batch_size), -1, -1, -1)
        .to(dtype=dtype)
    )


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    probe: QueryConditionedDenseCurveProbe,
    matcher: Any,
    loader: Iterable,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    group_size: int,
    line_width: float,
    state_source: str,
) -> tuple[dict[str, Any], int]:
    model.eval()
    probe.eval()
    modes = {
        "dense_correct": CurveModeMetrics(),
        "dense_wrong_image": CurveModeMetrics(),
        "dense_wrong_state": CurveModeMetrics(),
        "dense_zero_image": CurveModeMetrics(),
        "blend_050": CurveModeMetrics(),
    }
    comparison = ProbeComparison()
    images_seen = 0
    for images, targets, _metas in tqdm(
        loader,
        desc="query-conditioned dense probe eval",
        ncols=104,
    ):
        images = images.to(device)
        targets = nested_to_device(targets, device)
        with _amp_context(device, amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
            outputs = model.structured_query_head(
                encoded["features"],
                inference_only=False,
            )
        matches = matcher(outputs, targets)
        p2 = encoded["features"].float()
        row_states = probe_row_states(
            model.structured_query_head,
            outputs,
            source=state_source,
            group_size=group_size,
            batch_size=int(images.shape[0]),
        )
        encoded_correct = probe.encode_features(p2)
        correct_logits = probe.score_encoded(encoded_correct, row_states)
        wrong_image_logits = probe.score_encoded(
            probe.encode_features(_roll_batch(p2)),
            row_states,
        )
        wrong_state_logits = probe.score_encoded(
            encoded_correct,
            _roll_batch(row_states),
        )
        zero_image_logits = probe.score_encoded(
            probe.encode_features(torch.zeros_like(p2)),
            row_states,
        )
        curves = {
            "dense_correct": probe.decode(correct_logits),
            "dense_wrong_image": probe.decode(wrong_image_logits),
            "dense_wrong_state": probe.decode(wrong_state_logits),
            "dense_zero_image": probe.decode(zero_image_logits),
        }
        base = outputs["pred_x_rows"][:, :group_size].float()
        curves["blend_050"] = 0.5 * base + 0.5 * curves["dense_correct"]

        for batch_index, (target, match) in enumerate(zip(targets, matches)):
            pairs = _group_zero_matches(match, group_size=group_size)
            base_best = _best_iou_per_gt(
                base[batch_index],
                target,
                line_width=line_width,
            )
            base_paired = _paired_iou(
                base[batch_index],
                target,
                pairs,
                line_width=line_width,
            )
            normal_best: torch.Tensor | None = None
            normal_paired: torch.Tensor | None = None
            for name, candidates in curves.items():
                best = _best_iou_per_gt(
                    candidates[batch_index],
                    target,
                    line_width=line_width,
                )
                paired = _paired_iou(
                    candidates[batch_index],
                    target,
                    pairs,
                    line_width=line_width,
                )
                modes[name].update(best, paired)
                if name == "dense_correct":
                    normal_best = best
                    normal_paired = paired
            if normal_best is None or normal_paired is None:
                raise RuntimeError("dense_correct metrics were not computed")
            comparison.update(
                base_best,
                normal_best,
                base_paired,
                normal_paired,
            )
        images_seen += int(images.shape[0])
    return {
        "modes": {name: metric.summary() for name, metric in modes.items()},
        "base_vs_dense": comparison.summary(),
    }, images_seen


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["persistent_workers"] = bool(int(args.num_workers) > 0)
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg["training"]["seed"] = int(args.seed)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    if (
        amp_dtype == torch.bfloat16
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.float16

    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.requires_grad_(False)
    model = model.to(device).eval()
    matcher = build_matcher(cfg)

    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(
        structured_cfg.get("num_instances", model_cfg.get("num_slots", 0))
    )
    num_groups = int(structured_cfg.get("num_groups", 1))
    if num_instances % num_groups != 0:
        raise ValueError("num_instances must be divisible by num_groups")
    group_size = num_instances // num_groups
    probe = QueryConditionedDenseCurveProbe(
        in_dim=int(model_cfg.get("dim", 256)),
        state_dim=int(model_cfg.get("dim", 256)),
        hidden_dim=int(args.hidden_dim),
        num_rows=int(model_cfg.get("num_rows", 72)),
        evidence_width=int(args.evidence_width),
        input_w=int(model_cfg.get("input_w", 800)),
        conditioning_mode=args.conditioning_mode,
        explicit_coordinates=bool(args.explicit_coordinates),
    ).to(device)
    if args.load_probe:
        payload = torch.load(args.load_probe, map_location="cpu")
        probe.load_state_dict(payload.get("probe", payload), strict=True)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_loader = build_dataloader(cfg, split="train", training=True)
    train_iterator = iter(train_loader)
    if not args.load_probe and int(args.train_steps) > 0:
        probe.train()
        running_loss = 0.0
        running_lanes = 0
        running_rows = 0
        progress = tqdm(
            range(1, int(args.train_steps) + 1),
            desc="query-conditioned dense probe train",
            ncols=104,
        )
        for step in progress:
            try:
                images, targets, _metas = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                images, targets, _metas = next(train_iterator)
            images = images.to(device)
            targets = nested_to_device(targets, device)
            with torch.no_grad(), _amp_context(device, amp_dtype):
                encoded = model.encoder.forward_features(
                    images,
                    inference_only=True,
                    structured_only=True,
                )
                outputs = model.structured_query_head(
                    encoded["features"],
                    inference_only=False,
                )
                matches = matcher(outputs, targets)
                p2 = encoded["features"].float()
                row_states = probe_row_states(
                    model.structured_query_head,
                    outputs,
                    source=args.state_source,
                    group_size=group_size,
                    batch_size=int(images.shape[0]),
                )
            logits = probe(p2, row_states)
            loss, lane_count, row_count = matched_dense_curve_loss(
                logits,
                targets,
                matches,
                group_size=group_size,
                input_w=int(model_cfg.get("input_w", 800)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach())
            running_lanes += int(lane_count)
            running_rows += int(row_count)
            if step % int(args.log_interval) == 0 or step == int(args.train_steps):
                interval = (
                    int(args.log_interval)
                    if step % int(args.log_interval) == 0
                    else step % int(args.log_interval)
                )
                progress.set_postfix(
                    loss=f"{running_loss / max(interval, 1):.3f}",
                    lanes=running_lanes,
                    rows=running_rows,
                )
                running_loss = 0.0
                running_lanes = 0
                running_rows = 0

    eval_loader = build_dataloader(cfg, split=args.split, training=False)
    eval_loader, sampled_indices = select_diagnostic_loader(
        eval_loader,
        strategy=args.sample_strategy,
        max_batches=int(args.eval_max_batches),
        num_workers=int(args.num_workers),
    )
    metrics, images_seen = evaluate_probe(
        model,
        probe,
        matcher,
        eval_loader,
        device=device,
        amp_dtype=amp_dtype,
        group_size=group_size,
        line_width=float(args.line_width),
        state_source=args.state_source,
    )
    dense = metrics["modes"]["dense_correct"]
    wrong_image = metrics["modes"]["dense_wrong_image"]
    comparison = metrics["base_vs_dense"]
    positive_gate = bool(
        float(comparison["union_gain_050_points"]) >= 5.0
        and float(dense["recall_050"]) >= float(wrong_image["recall_050"]) + 0.03
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "The base model is frozen. A positive result proves recoverable "
            "query-conditioned P2 evidence, not the performance of a jointly "
            "trained final architecture."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "split": args.split,
        "seed": int(args.seed),
        "train_steps": int(args.train_steps),
        "state_source": args.state_source,
        "conditioning_mode": args.conditioning_mode,
        "explicit_coordinates": bool(args.explicit_coordinates),
        "images": int(images_seen),
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "group_size": int(group_size),
        "probe_parameters": sum(
            parameter.numel() for parameter in probe.parameters()
        ),
        "positive_gate_definition": (
            "union gain at IoU 0.50 >= 5.0 points AND correct-image dense "
            "recall >= wrong-image control by 3.0 points"
        ),
        "positive_gate": positive_gate,
        "metrics": metrics,
    }
    compact = dict(payload)
    compact.pop("sampled_dataset_indices")
    print(json.dumps(compact, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output_path}")
    if args.save_probe:
        save_path = Path(args.save_probe)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"probe": probe.state_dict(), "metadata": compact}, save_path)
        print(f"probe_checkpoint: {save_path}")


if __name__ == "__main__":
    main()
