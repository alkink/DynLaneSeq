from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
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
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    official_proposal_gt_iou_matrix,
    sha256_file,
    trace_postprocess,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_official_set_selection import (
    SetAwareQualityProbe,
    _finish_counts,
    _new_counts,
    _stage_for_image,
    _topk_ids,
    _update_counts,
    official_unique_quality_targets,
    selection_features,
)
from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    pairwise_quality_ranking_loss,
    quality_focal_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a LaneRowNet checkpoint and test whether raw P2 evidence "
            "sampled along each final predicted curve can verify the correct "
            "lane set. A geometry/state-only residual set scorer is trained as "
            "an equal-protocol control. This is a diagnostic, not a benchmark."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-images", type=int, default=4096)
    parser.add_argument("--val-images", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--curve-samples", type=int, default=20)
    parser.add_argument(
        "--offsets-px",
        type=float,
        nargs="+",
        default=[-48.0, -24.0, -12.0, 0.0, 12.0, 24.0, 48.0],
    )
    parser.add_argument("--visual-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--row-layers", type=int, default=1)
    parser.add_argument("--row-heads", type=int, default=4)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--set-heads", type=int, default=8)
    parser.add_argument("--set-ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quality-focal-beta", type=float, default=2.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-target-margin", type=float, default=0.10)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--range-temperature", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quality-power", type=float, default=0.25)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--min-gain-050-points", type=float, default=1.0)
    parser.add_argument("--min-gain-070-points", type=float, default=0.5)
    parser.add_argument("--min-visual-over-control-points", type=float, default=0.25)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-probe", required=True)
    return parser.parse_args()


class CurveAlignedVisualVerifier(nn.Module):
    """Residual set scorer with query-conditioned evidence along each curve."""

    def __init__(
        self,
        *,
        base_dim: int,
        feature_channels: int,
        row_state_dim: int,
        curve_samples: int,
        offsets: int,
        visual_dim: int = 128,
        hidden_dim: int = 256,
        row_layers: int = 1,
        row_heads: int = 4,
        set_layers: int = 2,
        set_heads: int = 8,
        set_ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(visual_dim) % int(row_heads):
            raise ValueError("visual_dim must be divisible by row_heads")
        if int(hidden_dim) % int(set_heads):
            raise ValueError("hidden_dim must be divisible by set_heads")
        self.curve_samples = int(curve_samples)
        self.offsets = int(offsets)
        self.profile_norm = nn.LayerNorm(int(feature_channels))
        self.profile_projection = nn.Linear(int(feature_channels), int(visual_dim))
        self.state_norm = nn.LayerNorm(int(row_state_dim))
        self.state_projection = nn.Linear(int(row_state_dim), int(visual_dim))
        self.offset_embedding = nn.Parameter(torch.zeros(self.offsets, int(visual_dim)))
        self.row_embedding = nn.Parameter(torch.zeros(self.curve_samples, int(visual_dim)))
        self.local_mixer = nn.Sequential(
            nn.LayerNorm(int(visual_dim)),
            nn.Linear(int(visual_dim), int(visual_dim)),
            nn.GELU(),
        )
        self.offset_score = nn.Linear(int(visual_dim), 1)
        row_layer = nn.TransformerEncoderLayer(
            d_model=int(visual_dim),
            nhead=int(row_heads),
            dim_feedforward=4 * int(visual_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.row_encoder = nn.TransformerEncoder(
            row_layer,
            num_layers=int(row_layers),
            enable_nested_tensor=False,
        )
        self.row_score = nn.Linear(int(visual_dim), 1)
        self.base_projection = nn.Sequential(
            nn.LayerNorm(int(base_dim)),
            nn.Linear(int(base_dim), int(hidden_dim)),
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(2 * int(visual_dim)),
            nn.Linear(2 * int(visual_dim), int(hidden_dim)),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * int(hidden_dim)),
            nn.Linear(2 * int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(set_heads),
            dim_feedforward=int(set_ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=int(set_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)
        nn.init.normal_(self.offset_embedding, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)
        # The residual scorer exactly preserves the deployed score at step 0.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        base_features: torch.Tensor,
        profiles: torch.Tensor,
        row_states: torch.Tensor,
        row_weights: torch.Tensor,
    ) -> torch.Tensor:
        if profiles.ndim != 5:
            raise ValueError("profiles must have shape [B,N,R,K,C]")
        batch, candidates, rows, offsets, _channels = profiles.shape
        if rows != self.curve_samples or offsets != self.offsets:
            raise ValueError(
                f"expected {self.curve_samples}x{self.offsets} samples, "
                f"got {rows}x{offsets}"
            )
        if row_states.shape[:3] != (batch, candidates, rows):
            raise ValueError("row_states must align with profiles")
        if row_weights.shape != (batch, candidates, rows):
            raise ValueError("row_weights must have shape [B,N,R]")

        visual = self.profile_projection(self.profile_norm(profiles.float()))
        state = self.state_projection(self.state_norm(row_states.float()))
        visual = visual + state.unsqueeze(3)
        visual = visual + self.offset_embedding.view(1, 1, 1, offsets, -1)
        visual = visual + self.row_embedding.view(1, 1, rows, 1, -1)
        visual = visual + self.local_mixer(visual)
        offset_probability = torch.softmax(
            self.offset_score(visual).squeeze(-1),
            dim=-1,
        )
        row_visual = (offset_probability.unsqueeze(-1) * visual).sum(dim=3)
        encoded_rows = self.row_encoder(
            row_visual.reshape(batch * candidates, rows, -1)
        ).reshape(batch, candidates, rows, -1)

        weights = row_weights.float().clamp_min(1e-6)
        normalized_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weighted_mean = (encoded_rows * normalized_weights.unsqueeze(-1)).sum(dim=2)
        attention_logits = self.row_score(encoded_rows).squeeze(-1) + weights.log()
        attention = torch.softmax(attention_logits, dim=-1)
        attended = (encoded_rows * attention.unsqueeze(-1)).sum(dim=2)
        visual_summary = self.visual_projection(torch.cat((weighted_mean, attended), dim=-1))
        base_summary = self.base_projection(base_features.float())
        hidden = self.fusion(torch.cat((base_summary, visual_summary), dim=-1))
        hidden = self.set_encoder(hidden)
        return self.output(self.output_norm(hidden)).squeeze(-1)


def sample_curve_aligned_profiles(
    feature_map: torch.Tensor,
    pred_x_rows: torch.Tensor,
    *,
    row_indices: torch.Tensor,
    offsets_px: torch.Tensor,
    input_h: int,
    input_w: int,
) -> torch.Tensor:
    """Sample P2 around final curves and return [B,N,R,K,C]."""

    if feature_map.ndim != 4 or pred_x_rows.ndim != 3:
        raise ValueError("feature_map and pred_x_rows must be rank 4/3")
    batch, channels, _height, _width = feature_map.shape
    pred_batch, candidates, total_rows = pred_x_rows.shape
    if pred_batch != batch:
        raise ValueError("feature and prediction batch sizes differ")
    row_indices = row_indices.to(device=pred_x_rows.device, dtype=torch.long)
    offsets_px = offsets_px.to(device=pred_x_rows.device, dtype=torch.float32)
    selected_x = pred_x_rows.detach().float().index_select(-1, row_indices)
    rows = int(selected_x.shape[-1])
    offsets = int(offsets_px.numel())
    sample_x = selected_x.unsqueeze(-1) + offsets_px.view(1, 1, 1, offsets)
    sample_x = sample_x.clamp(0.0, float(max(int(input_w) - 1, 0)))
    all_y = fixed_y_rows(
        int(total_rows),
        int(input_h),
        device=pred_x_rows.device,
        dtype=torch.float32,
    )
    sample_y = all_y.index_select(0, row_indices).view(1, 1, rows, 1)
    sample_y = sample_y.expand(batch, candidates, rows, offsets)
    grid_x = 2.0 * sample_x / float(max(int(input_w) - 1, 1)) - 1.0
    grid_y = 2.0 * sample_y / float(max(int(input_h) - 1, 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
        batch,
        candidates * rows,
        offsets,
        2,
    )
    # Keep the frozen visual readout in FP32. CUDA grid_sample BF16 support is
    # inconsistent across the server PyTorch versions used by this project.
    with torch.autocast(device_type=feature_map.device.type, enabled=False):
        sampled = F.grid_sample(
            feature_map.detach().float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return sampled.permute(0, 2, 3, 1).reshape(
        batch,
        candidates,
        rows,
        offsets,
        channels,
    )


def sampled_range_weights(
    range_norm: torch.Tensor,
    *,
    row_indices: torch.Tensor,
    total_rows: int,
    temperature: float,
) -> torch.Tensor:
    ranges = sort_range_norm(range_norm.detach().float())
    y = row_indices.to(device=ranges.device, dtype=torch.float32)
    y = y / float(max(int(total_rows) - 1, 1))
    y = y.view(1, 1, -1)
    tau = max(float(temperature), 1e-4)
    return torch.sigmoid((y - ranges[..., :1]) / tau) * torch.sigmoid(
        (ranges[..., 1:] - y) / tau
    )


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def frozen_visual_batch(
    model: nn.Module,
    images: torch.Tensor,
    *,
    row_indices: torch.Tensor,
    offsets_px: torch.Tensor,
    input_h: int,
    input_w: int,
    range_temperature: float,
    amp_dtype: torch.dtype | None,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    with _amp_context(images.device, amp_dtype):
        encoder_outputs = model.encoder.forward_features(
            images,
            inference_only=True,
            structured_only=True,
        )
        outputs = model.structured_query_head(
            encoder_outputs["features"],
            inference_only=False,
        )
        base_features = selection_features(
            outputs,
            input_w=input_w,
            curve_samples=int(row_indices.numel()),
        )
    profiles = sample_curve_aligned_profiles(
        encoder_outputs["features"],
        outputs["pred_x_rows"],
        row_indices=row_indices,
        offsets_px=offsets_px,
        input_h=input_h,
        input_w=input_w,
    )
    row_states = outputs["structured_row_tokens"].detach().index_select(
        2,
        row_indices.to(outputs["structured_row_tokens"].device),
    )
    row_weights = sampled_range_weights(
        outputs["range_norm"],
        row_indices=row_indices,
        total_rows=int(outputs["pred_x_rows"].shape[-1]),
        temperature=range_temperature,
    )
    return outputs, base_features.detach(), profiles.detach(), row_states.detach(), row_weights.detach()


def current_score_probability(
    outputs: dict[str, torch.Tensor],
    *,
    quality_power: float,
) -> torch.Tensor:
    exist = torch.softmax(outputs["exist_logits"].detach().float(), dim=-1)[..., 0]
    quality = torch.sigmoid(outputs["quality_logits"].detach().float()).clamp_min(1e-6)
    return (exist * quality.pow(float(quality_power))).clamp(1e-6, 1.0 - 1e-6)


def official_batch(
    outputs: dict[str, torch.Tensor],
    metas: list[dict[str, Any]],
    *,
    line_width: float,
    min_valid_rows: int,
    row_visibility_thresh: float,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    targets: list[torch.Tensor] = []
    official_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    stages: list[dict[str, torch.Tensor]] = []
    for batch_index, meta in enumerate(metas):
        stage = _stage_for_image(outputs, batch_index)
        official_iou, candidate_valid = official_proposal_gt_iou_matrix(
            {"stages": {"main": stage}, "meta": meta},
            "main",
            line_width=line_width,
            min_valid_rows=min_valid_rows,
            row_visibility_thresh=row_visibility_thresh,
        )
        targets.append(official_unique_quality_targets(official_iou, candidate_valid))
        official_rows.append(official_iou)
        valid_rows.append(candidate_valid)
        stages.append(stage)
    return torch.stack(targets, dim=0), official_rows, valid_rows, stages


def build_subset_loader(
    cfg: dict[str, Any],
    *,
    split: str,
    images: int,
    batch_size: int,
    num_workers: int,
):
    local_cfg = deepcopy(cfg)
    local_cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    local_cfg["dataloader"]["num_workers"] = int(num_workers)
    if int(num_workers) == 0:
        local_cfg["dataloader"]["persistent_workers"] = False
    base_loader = build_dataloader(local_cfg, split=split, training=False)
    sample_count = min(int(images), len(base_loader.dataset))
    max_batches = math.ceil(sample_count / int(batch_size))
    loader, indices = select_diagnostic_loader(
        base_loader,
        strategy="uniform",
        max_batches=max_batches,
        num_workers=num_workers,
    )
    if len(indices) > sample_count:
        raise ValueError("diagnostic loader rounded beyond requested sample count")
    return loader, indices


def _move_images(
    images: torch.Tensor,
    *,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    if channels_last:
        return images.to(device, non_blocking=True, memory_format=torch.channels_last)
    return images.to(device, non_blocking=True)


def _residual_logits(base_probability: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return torch.logit(base_probability.clamp(1e-6, 1.0 - 1e-6)) + delta.float()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    control_probe: nn.Module,
    visual_probe: nn.Module,
    loader,
    *,
    device: torch.device,
    channels_last: bool,
    amp_dtype: torch.dtype | None,
    row_indices: torch.Tensor,
    offsets_px: torch.Tensor,
    input_h: int,
    input_w: int,
    range_temperature: float,
    quality_power: float,
    line_width: float,
    min_valid_rows: int,
    row_visibility_thresh: float,
    top_k: int,
    nms_distance: float,
    nms_min_overlap_points: int,
) -> tuple[dict[str, Any], list[str]]:
    model.eval()
    control_probe.eval()
    visual_probe.eval()
    names = ("current_exist_quality", "base_residual_set", "curve_aligned_visual")
    modes = {
        "raw_top4": {name: _new_counts() for name in names},
        "nms_top4": {name: _new_counts() for name in names},
    }
    oracle_hits = {0.5: 0, 0.7: 0}
    oracle_gt = 0
    image_paths: list[str] = []
    for images, _targets, metas in tqdm(loader, desc="curve verifier eval", ncols=80):
        images = _move_images(images, device=device, channels_last=channels_last)
        outputs, base_features, profiles, row_states, row_weights = frozen_visual_batch(
            model,
            images,
            row_indices=row_indices,
            offsets_px=offsets_px,
            input_h=input_h,
            input_w=input_w,
            range_temperature=range_temperature,
            amp_dtype=amp_dtype,
        )
        base_probability = current_score_probability(outputs, quality_power=quality_power)
        control_logits = _residual_logits(base_probability, control_probe(base_features.float()))
        visual_logits = _residual_logits(
            base_probability,
            visual_probe(base_features, profiles, row_states, row_weights),
        )
        scores_by_strategy = {
            "current_exist_quality": base_probability.cpu(),
            "base_residual_set": torch.sigmoid(control_logits).cpu(),
            "curve_aligned_visual": torch.sigmoid(visual_logits).cpu(),
        }
        _quality_targets, official_rows, valid_rows, stages = official_batch(
            outputs,
            metas,
            line_width=line_width,
            min_valid_rows=min_valid_rows,
            row_visibility_thresh=row_visibility_thresh,
        )
        for batch_index, official_iou in enumerate(official_rows):
            candidate_valid = valid_rows[batch_index].bool()
            oracle_gt += int(official_iou.shape[0])
            for threshold in (0.5, 0.7):
                oracle = cardinality_oracle_assignment(
                    official_iou,
                    threshold=threshold,
                    top_k=top_k,
                    candidate_valid=candidate_valid,
                )
                oracle_hits[threshold] += int(oracle.hit_count)
            for name, score_batch in scores_by_strategy.items():
                scores = score_batch[batch_index]
                raw_ids = _topk_ids(scores, candidate_valid, top_k)
                trace = trace_postprocess(
                    stages[batch_index],
                    input_h=input_h,
                    input_w=input_w,
                    score_thresh=-1.0,
                    quality_power=0.0,
                    min_valid_rows=min_valid_rows,
                    nms_distance_thresh_px=nms_distance,
                    nms_min_overlap_points=nms_min_overlap_points,
                    top_k=top_k,
                    row_visibility_thresh=row_visibility_thresh,
                    score_override={index: float(scores[index]) for index in range(len(scores))},
                )
                _update_counts(
                    modes["raw_top4"][name],
                    official_iou,
                    raw_ids,
                    scores=scores,
                    candidate_valid=candidate_valid,
                )
                _update_counts(
                    modes["nms_top4"][name],
                    official_iou,
                    trace["selected_ids"],
                    scores=scores,
                    candidate_valid=candidate_valid,
                )
        image_paths.extend(str(meta.get("image_path", "")) for meta in metas)
    return {
        "modes": {
            mode: {name: _finish_counts(counts) for name, counts in rows.items()}
            for mode, rows in modes.items()
        },
        "oracle_top4": {
            f"{threshold:.2f}": {
                "gt_lanes": int(oracle_gt),
                "tp": int(oracle_hits[threshold]),
                "recall": float(oracle_hits[threshold]) / float(max(oracle_gt, 1)),
            }
            for threshold in (0.5, 0.7)
        },
    }, image_paths


def verification_verdict(
    evaluation: dict[str, Any],
    *,
    min_gain_050_points: float,
    min_gain_070_points: float,
    min_visual_over_control_points: float,
) -> dict[str, Any]:
    rows = evaluation["modes"]["nms_top4"]
    baseline = rows["current_exist_quality"]
    control = rows["base_residual_set"]
    visual = rows["curve_aligned_visual"]

    def gains(row: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
        return {
            "f1_050_points": 100.0 * (float(row["f1_050"]) - float(reference["f1_050"])),
            "f1_070_points": 100.0 * (float(row["f1_070"]) - float(reference["f1_070"])),
        }

    control_over_current = gains(control, baseline)
    visual_over_current = gains(visual, baseline)
    visual_over_control = gains(visual, control)
    control_positive = bool(
        control_over_current["f1_050_points"] >= float(min_gain_050_points)
        and control_over_current["f1_070_points"] >= float(min_gain_070_points)
    )
    visual_positive = bool(
        visual_over_current["f1_050_points"] >= float(min_gain_050_points)
        and visual_over_current["f1_070_points"] >= float(min_gain_070_points)
    )
    evidence_specific = bool(
        visual_positive
        and visual_over_control["f1_050_points"] >= float(min_visual_over_control_points)
        and visual_over_control["f1_070_points"] >= float(min_visual_over_control_points)
    )
    if evidence_specific:
        recommendation = "curve_aligned_visual_evidence_positive_integrate_jointly"
    elif control_positive and not evidence_specific:
        recommendation = "residual_set_optimization_positive_visual_evidence_not_isolated"
    elif visual_positive:
        recommendation = "visual_arm_positive_but_not_separated_from_control"
    else:
        recommendation = "frozen_curve_verifier_negative_rework_joint_proposal_decoder"
    return {
        "gate": {
            "min_gain_f1_050_points": float(min_gain_050_points),
            "min_gain_f1_070_points": float(min_gain_070_points),
            "min_visual_over_control_each_threshold_points": float(
                min_visual_over_control_points
            ),
        },
        "control_over_current": control_over_current,
        "visual_over_current": visual_over_current,
        "visual_over_control": visual_over_control,
        "control_positive": control_positive,
        "visual_positive": visual_positive,
        "evidence_specific_positive": evidence_specific,
        "recommendation": recommendation,
    }


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "train_images",
        "val_images",
        "train_steps",
        "curve_samples",
        "visual_dim",
        "hidden_dim",
        "row_layers",
        "row_heads",
        "set_layers",
        "set_heads",
        "set_ff_dim",
        "min_valid_rows",
        "top_k",
        "nms_min_overlap_points",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    if len(args.offsets_px) < 3:
        raise ValueError("offsets_px must contain at least three samples")
    offsets = sorted(float(value) for value in args.offsets_px)
    if offsets != [float(value) for value in args.offsets_px] or len(set(offsets)) != len(offsets):
        raise ValueError("offsets_px must be unique and strictly increasing")
    if not (any(value < 0 for value in offsets) and 0.0 in offsets and any(value > 0 for value in offsets)):
        raise ValueError("offsets_px must span negative, zero, and positive values")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _set_seed(args.seed)
    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    model_cfg = cfg["model"]
    input_h = int(model_cfg["input_h"])
    input_w = int(model_cfg["input_w"])
    total_rows = int(model_cfg["num_rows"])
    curve_samples = min(int(args.curve_samples), total_rows)
    row_indices = torch.linspace(0, total_rows - 1, curve_samples).round().long().to(device)
    offsets_px = torch.tensor(args.offsets_px, dtype=torch.float32, device=device)

    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if model.structured_query_head is None:
        raise ValueError("curve-aligned verification requires a structured query head")
    model.structured_query_head.intermediate_supervision = False
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False) and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    train_loader, train_indices = build_subset_loader(
        cfg,
        split="train",
        images=args.train_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader, val_indices = build_subset_loader(
        cfg,
        split="val",
        images=args.val_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Resolve dimensions from one frozen batch without consuming the training
    # loop's deterministic order or optimizer state.
    preview_images, _preview_targets, _preview_metas = next(iter(train_loader))
    preview_images = _move_images(preview_images, device=device, channels_last=channels_last)
    preview = frozen_visual_batch(
        model,
        preview_images,
        row_indices=row_indices,
        offsets_px=offsets_px,
        input_h=input_h,
        input_w=input_w,
        range_temperature=args.range_temperature,
        amp_dtype=amp_dtype,
    )
    _preview_outputs, preview_base, preview_profiles, preview_states, _preview_weights = preview
    base_dim = int(preview_base.shape[-1])
    feature_channels = int(preview_profiles.shape[-1])
    row_state_dim = int(preview_states.shape[-1])
    del preview, preview_images, preview_base, preview_profiles, preview_states
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Reset after the preview so probe initialization is independent of
    # DataLoader worker creation and cache/prefetch behavior.
    _set_seed(args.seed)
    control_probe = SetAwareQualityProbe(
        base_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.set_layers,
        num_heads=args.set_heads,
        ff_dim=args.set_ff_dim,
        dropout=args.dropout,
    ).to(device)
    nn.init.zeros_(control_probe.output.weight)
    nn.init.zeros_(control_probe.output.bias)
    visual_probe = CurveAlignedVisualVerifier(
        base_dim=base_dim,
        feature_channels=feature_channels,
        row_state_dim=row_state_dim,
        curve_samples=curve_samples,
        offsets=len(args.offsets_px),
        visual_dim=args.visual_dim,
        hidden_dim=args.hidden_dim,
        row_layers=args.row_layers,
        row_heads=args.row_heads,
        set_layers=args.set_layers,
        set_heads=args.set_heads,
        set_ff_dim=args.set_ff_dim,
        dropout=args.dropout,
    ).to(device)
    parameters = list(control_probe.parameters()) + list(visual_probe.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    totals = {
        "control_quality": 0.0,
        "control_rank": 0.0,
        "visual_quality": 0.0,
        "visual_rank": 0.0,
        "target_mean": 0.0,
        "target_ge_050": 0.0,
    }
    running = {name: 0.0 for name in totals}
    train_paths: list[str] = []
    control_probe.train()
    visual_probe.train()
    progress = tqdm(total=int(args.train_steps), desc="curve verifier train", ncols=80)
    step = 0
    while step < int(args.train_steps):
        for images, _targets, metas in train_loader:
            if step >= int(args.train_steps):
                break
            images = _move_images(images, device=device, channels_last=channels_last)
            outputs, base_features, profiles, row_states, row_weights = frozen_visual_batch(
                model,
                images,
                row_indices=row_indices,
                offsets_px=offsets_px,
                input_h=input_h,
                input_w=input_w,
                range_temperature=args.range_temperature,
                amp_dtype=amp_dtype,
            )
            quality_targets, _official, _valid, _stages = official_batch(
                outputs,
                metas,
                line_width=args.line_width,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            quality_targets = quality_targets.to(device=device, dtype=torch.float32)
            base_probability = current_score_probability(
                outputs,
                quality_power=args.quality_power,
            )
            control_logits = _residual_logits(
                base_probability,
                control_probe(base_features.float()),
            )
            visual_logits = _residual_logits(
                base_probability,
                visual_probe(base_features, profiles, row_states, row_weights),
            )
            control_quality = quality_focal_loss(
                control_logits,
                quality_targets,
                beta=args.quality_focal_beta,
            )
            visual_quality = quality_focal_loss(
                visual_logits,
                quality_targets,
                beta=args.quality_focal_beta,
            )
            control_rank = pairwise_quality_ranking_loss(
                control_logits,
                quality_targets,
                target_margin=args.rank_target_margin,
            )
            visual_rank = pairwise_quality_ranking_loss(
                visual_logits,
                quality_targets,
                target_margin=args.rank_target_margin,
            )
            loss = control_quality + visual_quality + float(args.rank_loss_weight) * (
                control_rank + visual_rank
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()

            values = {
                "control_quality": float(control_quality.detach()),
                "control_rank": float(control_rank.detach()),
                "visual_quality": float(visual_quality.detach()),
                "visual_rank": float(visual_rank.detach()),
                "target_mean": float(quality_targets.mean()),
                "target_ge_050": float((quality_targets > 0.5).float().mean()),
            }
            for name, value in values.items():
                totals[name] += value
                running[name] += value
            train_paths.extend(str(meta.get("image_path", "")) for meta in metas)
            step += 1
            progress.update(1)
            if int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
                denominator = float(args.log_interval)
                progress.write(
                    f"step {step:05d}/{int(args.train_steps):05d} "
                    f"control={running['control_quality'] / denominator:.4f} "
                    f"visual={running['visual_quality'] / denominator:.4f} "
                    f"rank_c={running['control_rank'] / denominator:.4f} "
                    f"rank_v={running['visual_rank'] / denominator:.4f}"
                )
                for name in running:
                    running[name] = 0.0
    progress.close()

    evaluation, val_paths = evaluate(
        model,
        control_probe,
        visual_probe,
        val_loader,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        row_indices=row_indices,
        offsets_px=offsets_px,
        input_h=input_h,
        input_w=input_w,
        range_temperature=args.range_temperature,
        quality_power=args.quality_power,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        top_k=args.top_k,
        nms_distance=args.nms_distance,
        nms_min_overlap_points=args.nms_min_overlap_points,
    )
    overlap = set(train_paths) & set(val_paths)
    if overlap:
        raise ValueError(f"train/validation image overlap: {sorted(overlap)[0]}")
    verdict = verification_verdict(
        evaluation,
        min_gain_050_points=args.min_gain_050_points,
        min_gain_070_points=args.min_gain_070_points,
        min_visual_over_control_points=args.min_visual_over_control_points,
    )

    checkpoint_sha256 = sha256_file(args.checkpoint)
    payload = {
        "diagnostic_only": True,
        "warning": (
            "The detector is frozen. Official train-split CULane raster IoU "
            "supervises only diagnostic residual scorers; this is not a "
            "benchmark result or a deployable model."
        ),
        "question": (
            "Does raw P2 evidence sampled along final predicted curves add "
            "set-selection information absent from decoded geometry/state features?"
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "checkpoint_sha256": checkpoint_sha256,
        "feature_contract": {
            "base_feature_dim": base_dim,
            "p2_channels": feature_channels,
            "row_state_dim": row_state_dim,
            "curve_samples": curve_samples,
            "offsets_px": [float(value) for value in args.offsets_px],
            "p2_sampling": "bilinear_grid_sample_along_final_pred_x_rows",
            "visual_conditioning": "final_structured_row_states",
            "control": "same_residual_set_protocol_without_raw_p2_profiles",
        },
        "target": {
            "name": "official_raster_iou_unique_hungarian",
            "line_width": float(args.line_width),
        },
        "training": {
            "steps": int(args.train_steps),
            "batch_size": int(args.batch_size),
            "train_images_requested": int(args.train_images),
            "train_subset_size": len(train_indices),
            "unique_train_images_seen": len(set(train_paths)),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "quality_focal_beta": float(args.quality_focal_beta),
            "rank_loss_weight": float(args.rank_loss_weight),
            "rank_target_margin": float(args.rank_target_margin),
            "seed": int(args.seed),
            "mean_losses": {
                name: value / float(max(int(args.train_steps), 1))
                for name, value in totals.items()
            },
        },
        "probes": {
            "control_parameters": sum(p.numel() for p in control_probe.parameters()),
            "visual_parameters": sum(p.numel() for p in visual_probe.parameters()),
            "zero_initialized_residual_outputs": True,
            "permutation_equivariant_candidate_processing": True,
        },
        "evaluation_settings": {
            "split": "val",
            "val_images_requested": int(args.val_images),
            "val_subset_size": len(val_indices),
            "quality_power": float(args.quality_power),
            "top_k": int(args.top_k),
            "nms_distance": float(args.nms_distance),
            "score_threshold": None,
            "train_val_paths_disjoint": True,
        },
        "evaluation": evaluation,
        "verdict": verdict,
    }
    save_path = Path(args.save_probe)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "control_probe": control_probe.state_dict(),
            "visual_probe": visual_probe.state_dict(),
            "source_checkpoint": args.checkpoint,
            "source_iteration": int(checkpoint_iteration),
            "source_checkpoint_sha256": checkpoint_sha256,
            "base_dim": base_dim,
            "feature_channels": feature_channels,
            "row_state_dim": row_state_dim,
            "curve_samples": curve_samples,
            "offsets_px": [float(value) for value in args.offsets_px],
            "training_seed": int(args.seed),
            "training_steps": int(args.train_steps),
        },
        save_path,
    )
    payload["probe_checkpoint"] = str(save_path)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")
    print(f"probe_checkpoint: {save_path}")


if __name__ == "__main__":
    main()
