from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
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
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an independent ordered-curve proposal head on one frozen visual "
            "source. The probe never consumes the model's lane queries, row states, "
            "decoder outputs, or assignments. It tests whether that feature source "
            "alone can acquire coarse references for GT lanes missed by the "
            "structured decoder."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-proposals", type=int, default=4)
    parser.add_argument(
        "--feature-source",
        choices=("p2", "c2", "c3"),
        default="p2",
        help=(
            "Frozen visual source consumed by the independent head. Raw C2/C3 "
            "are zero-padded to the model dimension so every source uses the "
            "same trainable probe architecture and parameter count."
        ),
    )
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--probe-width", type=int, default=100)
    parser.add_argument("--x-bins", type=int, default=200)
    parser.add_argument("--eval-max-batches", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--sample-strategy", choices=("sequential", "uniform"), default="uniform")
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


class IndependentP2CurveProposalProbe(nn.Module):
    """Small fixed-order curve head that consumes only P2 image features."""

    def __init__(
        self,
        *,
        in_dim: int,
        feature_dim: int,
        hidden_dim: int,
        num_rows: int,
        probe_width: int,
        num_proposals: int,
        x_bins: int,
        input_w: int,
    ) -> None:
        super().__init__()
        if feature_dim % 8 != 0:
            raise ValueError("feature_dim must be divisible by 8 for GroupNorm")
        self.num_rows = int(num_rows)
        self.probe_width = int(probe_width)
        self.num_proposals = int(num_proposals)
        self.x_bins = int(x_bins)
        self.input_w = int(input_w)
        self.visual_tower = nn.Sequential(
            nn.Conv2d(int(in_dim) + 2, int(feature_dim), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, int(feature_dim)),
            nn.GELU(),
            nn.Conv2d(int(feature_dim), int(feature_dim), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, int(feature_dim)),
            nn.GELU(),
        )
        self.row_encoder = nn.Sequential(
            nn.Linear(int(feature_dim) * self.probe_width, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.row_embedding = nn.Embedding(self.num_rows, int(hidden_dim))
        self.vertical_mixer = nn.Sequential(
            nn.Conv1d(int(hidden_dim), int(hidden_dim), kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, int(hidden_dim)),
            nn.GELU(),
            nn.Conv1d(int(hidden_dim), int(hidden_dim), kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, int(hidden_dim)),
            nn.GELU(),
        )
        self.row_norm = nn.LayerNorm(int(hidden_dim))
        self.row_x = nn.Linear(
            int(hidden_dim),
            self.num_proposals * self.x_bins,
        )
        self.row_visibility = nn.Linear(int(hidden_dim), self.num_proposals)
        self.existence = nn.Linear(int(hidden_dim), self.num_proposals)
        nn.init.normal_(self.row_embedding.weight, std=0.02)

    def forward(self, p2: torch.Tensor) -> dict[str, torch.Tensor]:
        resized = F.interpolate(
            p2,
            size=(self.num_rows, self.probe_width),
            mode="bilinear",
            align_corners=False,
        )
        batch = int(resized.shape[0])
        yy = torch.linspace(-1.0, 1.0, self.num_rows, device=resized.device, dtype=resized.dtype)
        xx = torch.linspace(-1.0, 1.0, self.probe_width, device=resized.device, dtype=resized.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        visual = self.visual_tower(torch.cat((resized, coordinates), dim=1))

        row_input = visual.permute(0, 2, 1, 3).reshape(batch, self.num_rows, -1)
        row_state = self.row_encoder(row_input)
        global_state = self.global_encoder(visual.mean(dim=(2, 3))).unsqueeze(1)
        row_position = self.row_embedding.weight.to(device=visual.device, dtype=visual.dtype).unsqueeze(0)
        row_state = row_state + global_state + row_position
        mixed = self.vertical_mixer(row_state.transpose(1, 2)).transpose(1, 2)
        row_state = self.row_norm(row_state + mixed)

        logits = self.row_x(row_state).reshape(
            batch,
            self.num_rows,
            self.num_proposals,
            self.x_bins,
        ).permute(0, 2, 1, 3).contiguous()
        visibility = self.row_visibility(row_state).permute(0, 2, 1).contiguous()
        existence = self.existence(row_state.mean(dim=1))
        return {
            "row_x_logits": logits,
            "row_visibility_logits": visibility,
            "existence_logits": existence,
        }

    def decode(self, outputs: dict[str, torch.Tensor], *, method: str = "expected") -> torch.Tensor:
        logits = outputs["row_x_logits"]
        if method == "expected":
            return soft_expected_x(
                logits,
                input_w=self.input_w,
                x_bins=self.x_bins,
            )
        if method == "argmax":
            indices = logits.argmax(dim=-1).to(dtype=logits.dtype)
            return (indices + 0.5) * (float(self.input_w) / float(self.x_bins))
        raise ValueError(f"Unsupported decode method: {method!r}")


def canonicalize_feature_channels(
    feature: torch.Tensor,
    *,
    output_channels: int,
) -> torch.Tensor:
    """Losslessly pad a frozen feature map to a common channel count."""
    input_channels = int(feature.shape[1])
    output_channels = int(output_channels)
    if input_channels > output_channels:
        raise ValueError(
            f"Cannot losslessly canonicalize {input_channels} channels to "
            f"{output_channels}; increase the common channel count."
        )
    if input_channels == output_channels:
        return feature
    return F.pad(feature, (0, 0, 0, 0, 0, output_channels - input_channels))


def extract_frozen_feature_source(
    base_model: nn.Module,
    images: torch.Tensor,
    *,
    feature_source: str,
    canonical_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected frozen source and the projected P2 used by the base."""
    if feature_source not in {"p2", "c2", "c3"}:
        raise ValueError(f"Unsupported feature source: {feature_source!r}")
    encoder = base_model.encoder
    backbone_features = encoder.backbone(images)
    p2 = encoder.proj(encoder.fpn(backbone_features))
    selected = p2 if feature_source == "p2" else backbone_features[feature_source]
    selected = canonicalize_feature_channels(
        selected,
        output_channels=int(canonical_channels),
    )
    return selected, p2


def _lane_reference_x(x_rows: torch.Tensor, valid: torch.Tensor) -> float:
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return float("inf")
    bottom_count = min(8, int(valid_indices.numel()))
    bottom_indices = valid_indices[-bottom_count:]
    return float(x_rows[bottom_indices].mean())


def ordered_target_tensors(
    targets: list[dict[str, torch.Tensor]],
    *,
    num_proposals: int,
    num_rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    batch = len(targets)
    x_target = torch.zeros((batch, num_proposals, num_rows), device=device, dtype=torch.float32)
    valid_target = torch.zeros((batch, num_proposals, num_rows), device=device, dtype=torch.bool)
    exist_target = torch.zeros((batch, num_proposals), device=device, dtype=torch.float32)
    dropped_lanes = 0
    for batch_index, target in enumerate(targets):
        x_rows = target["x_rows"].to(device=device, dtype=torch.float32)
        valid = target["valid_mask"].to(device=device).bool()
        lane_indices = [
            lane_index
            for lane_index in range(int(x_rows.shape[0]))
            if int(valid[lane_index].sum()) >= 5
        ]
        lane_indices.sort(
            key=lambda lane_index: _lane_reference_x(
                x_rows[lane_index],
                valid[lane_index],
            )
        )
        dropped_lanes += max(0, len(lane_indices) - int(num_proposals))
        for slot_index, lane_index in enumerate(lane_indices[:num_proposals]):
            x_target[batch_index, slot_index] = x_rows[lane_index]
            valid_target[batch_index, slot_index] = valid[lane_index]
            exist_target[batch_index, slot_index] = 1.0
    return x_target, valid_target, exist_target, dropped_lanes


def compute_probe_loss(
    probe: IndependentP2CurveProposalProbe,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    x_target, valid_target, exist_target, dropped_lanes = ordered_target_tensors(
        targets,
        num_proposals=probe.num_proposals,
        num_rows=probe.num_rows,
        device=outputs["row_x_logits"].device,
    )
    logits = outputs["row_x_logits"].float()
    target_bins = torch.floor(
        x_target / float(probe.input_w) * float(probe.x_bins)
    ).long().clamp(0, probe.x_bins - 1)
    if bool(valid_target.any()):
        loss_distribution = F.cross_entropy(
            logits[valid_target],
            target_bins[valid_target],
            reduction="mean",
        )
        pred_x = probe.decode({"row_x_logits": logits}, method="expected")
        loss_point = F.smooth_l1_loss(
            pred_x[valid_target] / float(probe.input_w),
            x_target[valid_target] / float(probe.input_w),
            beta=0.01,
            reduction="mean",
        )
    else:
        loss_distribution = logits.sum() * 0.0
        loss_point = logits.sum() * 0.0
    loss_visibility = F.binary_cross_entropy_with_logits(
        outputs["row_visibility_logits"].float(),
        valid_target.float(),
    )
    loss_existence = F.binary_cross_entropy_with_logits(
        outputs["existence_logits"].float(),
        exist_target,
    )
    loss_total = (
        loss_distribution
        + 5.0 * loss_point
        + 0.25 * loss_visibility
        + 0.25 * loss_existence
    )
    components = {
        "total": loss_total,
        "distribution": loss_distribution,
        "point": loss_point,
        "visibility": loss_visibility,
        "existence": loss_existence,
    }
    return loss_total, components, dropped_lanes


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


def _paired_ordered_iou(
    candidates: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    line_width: float,
) -> torch.Tensor:
    x_target, valid_target, exist_target, _ = ordered_target_tensors(
        [target],
        num_proposals=int(candidates.shape[0]),
        num_rows=int(candidates.shape[-1]),
        device=candidates.device,
    )
    values: list[torch.Tensor] = []
    for slot_index in range(int(candidates.shape[0])):
        if not bool(exist_target[0, slot_index]):
            continue
        iou = line_iou_against_gt(
            candidates[slot_index : slot_index + 1],
            x_target[0, slot_index].to(dtype=candidates.dtype),
            valid_target[0, slot_index],
            line_width=float(line_width),
        )
        values.append(iou[0] if iou.numel() else candidates.new_zeros(()))
    return torch.stack(values).float() if values else candidates.new_zeros((0,), dtype=torch.float32)


@dataclass
class ModeMetrics:
    gt_lanes: int = 0
    hits_050: int = 0
    hits_070: int = 0
    paired_iou_sum: float = 0.0
    paired_iou_count: int = 0

    def update(self, best_iou: torch.Tensor, paired_iou: torch.Tensor) -> None:
        self.gt_lanes += int(best_iou.numel())
        self.hits_050 += int((best_iou >= 0.5).sum())
        self.hits_070 += int((best_iou >= 0.7).sum())
        self.paired_iou_sum += float(paired_iou.sum())
        self.paired_iou_count += int(paired_iou.numel())

    def summary(self) -> dict[str, float | int]:
        return {
            "gt_lanes": self.gt_lanes,
            "recall_050": self.hits_050 / max(self.gt_lanes, 1),
            "recall_070": self.hits_070 / max(self.gt_lanes, 1),
            "paired_mean_iou": self.paired_iou_sum / max(self.paired_iou_count, 1),
        }


@dataclass
class BaseUnionMetrics:
    gt_lanes: int = 0
    base_hits_050: int = 0
    base_hits_070: int = 0
    union_hits_050: int = 0
    union_hits_070: int = 0
    base_misses_050: int = 0
    recovered_misses_050: int = 0
    base_misses_070: int = 0
    recovered_misses_070: int = 0

    def update(self, base_iou: torch.Tensor, candidate_iou: torch.Tensor) -> None:
        if base_iou.shape != candidate_iou.shape:
            raise ValueError("Base/probe GT count mismatch")
        union = torch.maximum(base_iou, candidate_iou)
        miss_050 = base_iou < 0.5
        miss_070 = base_iou < 0.7
        self.gt_lanes += int(base_iou.numel())
        self.base_hits_050 += int((base_iou >= 0.5).sum())
        self.base_hits_070 += int((base_iou >= 0.7).sum())
        self.union_hits_050 += int((union >= 0.5).sum())
        self.union_hits_070 += int((union >= 0.7).sum())
        self.base_misses_050 += int(miss_050.sum())
        self.recovered_misses_050 += int((miss_050 & (candidate_iou >= 0.5)).sum())
        self.base_misses_070 += int(miss_070.sum())
        self.recovered_misses_070 += int((miss_070 & (candidate_iou >= 0.7)).sum())

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
            "miss_recovery_fraction_050": self.recovered_misses_050 / max(self.base_misses_050, 1),
            "base_misses_070": self.base_misses_070,
            "recovered_base_misses_070": self.recovered_misses_070,
            "miss_recovery_fraction_070": self.recovered_misses_070 / max(self.base_misses_070, 1),
        }


@torch.no_grad()
def evaluate_probe(
    base_model: nn.Module,
    probe: IndependentP2CurveProposalProbe,
    loader: Iterable,
    *,
    device: torch.device,
    channels_last: bool,
    amp_dtype: torch.dtype | None,
    group_size: int,
    line_width: float,
    feature_source: str = "p2",
    canonical_channels: int | None = None,
) -> dict[str, Any]:
    mode_names = (
        "correct_expected",
        "correct_argmax",
        "wrong_image",
        "zero_image",
        "horizontal_mean",
    )
    mode_metrics = {name: ModeMetrics() for name in mode_names}
    union_metrics = {
        name: BaseUnionMetrics()
        for name in ("correct_expected", "correct_argmax", "wrong_image", "zero_image", "horizontal_mean")
    }
    base_model.eval()
    probe.eval()
    autocast_enabled = amp_dtype is not None and device.type == "cuda"
    for images, targets, _metas in tqdm(
        loader,
        ncols=96,
        desc=f"independent {str(feature_source).upper()} proposal eval",
    ):
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
            selected_feature, p2 = extract_frozen_feature_source(
                base_model,
                images,
                feature_source=str(feature_source),
                canonical_channels=int(
                    canonical_channels
                    if canonical_channels is not None
                    else base_model.encoder.proj.out_channels
                ),
            )
            # V5+ protected ownership consumes detached P4/P5 semantic
            # features in addition to projected P2.  Calling the structured
            # head directly therefore no longer reproduces the production
            # forward contract.  Use the complete frozen model for the base
            # comparison while retaining ``selected_feature`` for the
            # independent probe arm.
            base_outputs = base_model(images)
        selected_float = selected_feature.float()
        wrong_indices = torch.roll(
            torch.arange(int(selected_feature.shape[0]), device=device),
            shifts=1,
        )
        conditions = {
            "correct_expected": selected_float,
            "correct_argmax": selected_float,
            "wrong_image": selected_float[wrong_indices],
            "zero_image": torch.zeros_like(selected_float),
            "horizontal_mean": selected_float.mean(dim=-1, keepdim=True).expand_as(
                selected_float
            ),
        }
        condition_outputs: dict[str, torch.Tensor] = {}
        for mode_name, condition_p2 in conditions.items():
            outputs = probe(condition_p2)
            decode_method = "argmax" if mode_name == "correct_argmax" else "expected"
            condition_outputs[mode_name] = probe.decode(outputs, method=decode_method).float()

        for sample_index, target in enumerate(targets):
            base_candidates = base_outputs["pred_x_rows"][sample_index, :group_size].float()
            base_iou = _best_iou_per_gt(base_candidates, target, line_width=line_width)
            for mode_name, candidates_batch in condition_outputs.items():
                candidates = candidates_batch[sample_index]
                candidate_iou = _best_iou_per_gt(candidates, target, line_width=line_width)
                paired_iou = _paired_ordered_iou(candidates, target, line_width=line_width)
                mode_metrics[mode_name].update(candidate_iou, paired_iou)
                union_metrics[mode_name].update(base_iou, candidate_iou)

    mode_summary = {name: metrics.summary() for name, metrics in mode_metrics.items()}
    union_summary = {name: metrics.summary() for name, metrics in union_metrics.items()}
    primary = union_summary["correct_expected"]
    wrong = union_summary["wrong_image"]
    correct_mode = mode_summary["correct_expected"]
    wrong_mode = mode_summary["wrong_image"]
    gain_over_wrong = (
        float(primary["union_gain_050_points"])
        - float(wrong["union_gain_050_points"])
    )
    paired_iou_gain = (
        float(correct_mode["paired_mean_iou"])
        - float(wrong_mode["paired_mean_iou"])
    )
    gate = (
        float(primary["union_gain_050_points"]) >= 5.0
        and gain_over_wrong >= 3.0
        and paired_iou_gain >= 0.05
    )
    return {
        "positive_gate_definition": (
            "correct-image expected-X union gain at IoU 0.50 >= 5.0 points; "
            "gain over wrong-image control >= 3.0 points; and paired mean-IoU "
            "gain over wrong-image control >= 0.05"
        ),
        "positive_gate": bool(gate),
        "image_specificity": {
            "correct_minus_wrong_union_gain_050_points": gain_over_wrong,
            "correct_minus_wrong_paired_mean_iou": paired_iou_gain,
        },
        "modes": mode_summary,
        "base_unions": union_summary,
    }


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
    input_w = int(model_cfg.get("input_w", 800))
    canonical_channels = int(model_cfg.get("dim", 256))
    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(structured_cfg.get("num_instances", model_cfg.get("num_slots", 0)))
    num_groups = int(structured_cfg.get("num_groups", 1))
    if num_groups < 1 or num_instances % num_groups != 0:
        raise ValueError("Invalid structured query grouping")
    group_size = num_instances // num_groups

    probe = IndependentP2CurveProposalProbe(
        in_dim=canonical_channels,
        feature_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim),
        num_rows=num_rows,
        probe_width=int(args.probe_width),
        num_proposals=int(args.num_proposals),
        x_bins=int(args.x_bins),
        input_w=input_w,
    ).to(device)
    loaded_probe_metadata: dict[str, Any] = {}
    if args.load_probe:
        try:
            probe_payload = torch.load(args.load_probe, map_location="cpu", weights_only=False)
        except TypeError:
            probe_payload = torch.load(args.load_probe, map_location="cpu")
        probe.load_state_dict(probe_payload.get("probe", probe_payload), strict=True)
        if isinstance(probe_payload, dict) and isinstance(probe_payload.get("metadata"), dict):
            loaded_probe_metadata = dict(probe_payload["metadata"])
        print(f"loaded_probe: {args.load_probe}")

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_loader = build_dataloader(cfg, split="train", training=True)
    raw_eval_loader = build_dataloader(cfg, split="val", training=False)
    eval_loader, sampled_indices = select_diagnostic_loader(
        raw_eval_loader,
        strategy=str(args.sample_strategy),
        max_batches=int(args.eval_max_batches),
        num_workers=int(args.num_workers),
    )

    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    if amp_dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    autocast_enabled = amp_dtype is not None and device.type == "cuda"

    probe.train()
    train_iterator = iter(train_loader)
    running = {
        "total": 0.0,
        "distribution": 0.0,
        "point": 0.0,
        "visibility": 0.0,
        "existence": 0.0,
    }
    dropped_lanes_total = 0
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
                selected_feature, _p2 = extract_frozen_feature_source(
                    base_model,
                    images,
                    feature_source=str(args.feature_source),
                    canonical_channels=canonical_channels,
                )
        outputs = probe(selected_feature.float())
        loss, components, dropped_lanes = compute_probe_loss(probe, outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(probe.parameters(), max_norm=5.0)
        optimizer.step()
        dropped_lanes_total += int(dropped_lanes)
        for name in running:
            running[name] += float(components[name].detach())
        if step % int(args.log_interval) == 0 or step == int(args.train_steps):
            window = int(args.log_interval)
            if step == int(args.train_steps) and step % int(args.log_interval) != 0:
                window = step % int(args.log_interval)
            print(
                f"independent {str(args.feature_source).upper()} proposal step "
                f"{step:05d}/{int(args.train_steps):05d} | "
                + " | ".join(
                    f"{name} {running[name] / max(window, 1):.4f}"
                    for name in running
                ),
                flush=True,
            )
            running = {name: 0.0 for name in running}

    evaluation = evaluate_probe(
        base_model,
        probe,
        eval_loader,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        group_size=group_size,
        line_width=float(args.line_width),
        feature_source=str(args.feature_source),
        canonical_channels=canonical_channels,
    )
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    payload = {
        "diagnostic_only": True,
        "warning": (
            f"The base model and {str(args.feature_source).upper()} source are frozen. "
            "A positive result proves that an independent supervised head can acquire "
            "missed lane references from that source; it is not a benchmark result "
            "or a jointly trained final architecture."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed": int(args.seed),
        "train_steps": int(args.train_steps),
        "loaded_probe": str(args.load_probe),
        "loaded_probe_prior_train_steps": int(
            loaded_probe_metadata.get(
                "total_probe_train_steps",
                loaded_probe_metadata.get("train_steps", 0),
            )
        ),
        "total_probe_train_steps": (
            int(args.train_steps)
            + int(
                loaded_probe_metadata.get(
                    "total_probe_train_steps",
                    loaded_probe_metadata.get("train_steps", 0),
                )
            )
        ),
        "train_batch_size": int(args.batch_size),
        "eval_split": "val",
        "sample_strategy": str(args.sample_strategy),
        "sampled_dataset_indices": sampled_indices,
        "images": len(sampled_indices),
        "probe_parameters": int(parameter_count),
        "probe_design": {
            "inputs": (
                f"frozen {str(args.feature_source).upper()} plus absolute x/y "
                "coordinate channels"
            ),
            "feature_source": str(args.feature_source),
            "canonical_channels": canonical_channels,
            "channel_equalization": (
                "raw C2/C3 channels are losslessly zero-padded to the model "
                "dimension; all sources therefore use an identical trainable head"
            ),
            "excluded_inputs": (
                "lane queries, row tokens/states, decoder outputs, matcher assignments, "
                "base predicted curves"
            ),
            "assignment": "GT lanes sorted left-to-right by their bottom visible rows",
            "num_proposals": int(args.num_proposals),
            "num_rows": num_rows,
            "probe_width": int(args.probe_width),
            "x_bins": int(args.x_bins),
            "primary_decode": "expected-X",
        },
        "dropped_training_lanes_beyond_proposal_count": int(dropped_lanes_total),
        "evaluation": evaluation,
    }
    print(json.dumps(payload, indent=2))

    if args.save_probe:
        path = Path(args.save_probe)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"probe": probe.state_dict(), "metadata": payload}, path)
        print(f"probe_checkpoint: {path}")
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"output_json: {path}")


if __name__ == "__main__":
    main()
