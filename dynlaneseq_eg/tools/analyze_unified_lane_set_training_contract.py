from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    official_proposal_gt_iou_matrix,
    recall_from_ids,
    unique_candidate_labels,
)
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the unified lane-set training contract at a frozen "
            "checkpoint: score/geometry gradient alignment, official-IoU "
            "score alignment, lane-count calibration, and direct-vs-oracle "
            "Top-K capacity."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument(
        "--gradient-images",
        type=int,
        default=2,
        help=(
            "Number of individual images on which full parameter gradients "
            "are measured. The remaining images use inference-only forward."
        ),
    )
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
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--score-thresholds",
        type=float,
        nargs="+",
        default=(0.20, 0.30),
        help=(
            "Frozen deployment points evaluated from one inference pass. "
            "They are reported separately and are not swept to choose a winner."
        ),
    )
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=(0.50, 0.75),
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


@dataclass
class ScalarStats:
    values: list[float] = field(default_factory=list)

    def update(self, value: float | None) -> None:
        if value is not None and math.isfinite(float(value)):
            self.values.append(float(value))

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.update(float(value))

    def summary(self) -> dict[str, float | int | None]:
        if not self.values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }
        values = np.asarray(self.values, dtype=np.float64)
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }


@dataclass
class GradientAlignmentStats:
    geometry_norm: ScalarStats = field(default_factory=ScalarStats)
    score_norm: ScalarStats = field(default_factory=ScalarStats)
    cosine: ScalarStats = field(default_factory=ScalarStats)
    score_to_geometry_norm_ratio: ScalarStats = field(
        default_factory=ScalarStats
    )
    score_projection_on_geometry: ScalarStats = field(
        default_factory=ScalarStats
    )
    valid_pairs: int = 0
    negative_pairs: int = 0

    def update(
        self,
        geometry_gradients: tuple[torch.Tensor | None, ...],
        score_gradients: tuple[torch.Tensor | None, ...],
    ) -> None:
        geometry_sq = 0.0
        score_sq = 0.0
        dot = 0.0
        for geometry, score in zip(geometry_gradients, score_gradients):
            if geometry is not None:
                geometry_f = geometry.detach().float()
                geometry_sq += float(torch.sum(geometry_f * geometry_f))
            else:
                geometry_f = None
            if score is not None:
                score_f = score.detach().float()
                score_sq += float(torch.sum(score_f * score_f))
            else:
                score_f = None
            if geometry_f is not None and score_f is not None:
                dot += float(torch.sum(geometry_f * score_f))

        geometry_norm = math.sqrt(max(geometry_sq, 0.0))
        score_norm = math.sqrt(max(score_sq, 0.0))
        self.geometry_norm.update(geometry_norm)
        self.score_norm.update(score_norm)
        if geometry_norm <= 0.0 or score_norm <= 0.0:
            return
        cosine = dot / (geometry_norm * score_norm)
        self.cosine.update(cosine)
        self.score_to_geometry_norm_ratio.update(score_norm / geometry_norm)
        # In a gradient-descent update, a negative value means that the score
        # objective locally moves against the geometry descent direction.
        self.score_projection_on_geometry.update(dot / geometry_sq)
        self.valid_pairs += 1
        self.negative_pairs += int(cosine < 0.0)

    def summary(self) -> dict[str, Any]:
        return {
            "geometry_gradient_norm": self.geometry_norm.summary(),
            "score_gradient_norm": self.score_norm.summary(),
            "cosine_geometry_vs_score": self.cosine.summary(),
            "score_to_geometry_norm_ratio": (
                self.score_to_geometry_norm_ratio.summary()
            ),
            "score_projection_on_geometry": (
                self.score_projection_on_geometry.summary()
            ),
            "negative_cosine_fraction": (
                float(self.negative_pairs) / float(self.valid_pairs)
                if self.valid_pairs
                else None
            ),
        }


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _prepare_config(
    path: str,
    *,
    dataset_root: str,
    eval_batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg = load_config(path)
    if dataset_root:
        cfg.setdefault("dataset", {})["root"] = dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        eval_batch_size
    )
    cfg.setdefault("dataloader", {})["num_workers"] = int(num_workers)
    if int(num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _stage_for_image(
    outputs: dict[str, torch.Tensor],
    image_index: int,
) -> dict[str, torch.Tensor]:
    fields = (
        "pred_x_rows",
        "range_norm",
        "exist_logits",
        "quality_logits",
        "bounded_delta_max_abs_by_layer",
        "bounded_delta_mean_abs_by_layer",
    )
    return {
        name: outputs[name][image_index].detach().float().cpu()
        for name in fields
        if isinstance(outputs.get(name), torch.Tensor)
    }


def _parameter_groups(
    model: torch.nn.Module,
) -> tuple[
    dict[str, tuple[torch.nn.Parameter, ...]],
    tuple[torch.nn.Parameter, ...],
    dict[torch.nn.Parameter, int],
]:
    grouped: dict[str, list[torch.nn.Parameter]] = defaultdict(list)
    parameter_index: dict[torch.nn.Parameter, int] = {}
    selected: list[torch.nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group: str | None = None
        if name.startswith("encoder.backbone."):
            group = "shared_backbone"
        elif name.startswith("encoder.fpn.lateral.c4.") or name.startswith(
            "encoder.fpn.lateral.c5."
        ):
            # C4/C5 feed both the top-down P2 geometry path and the P4/P5
            # semantic score view.
            group = "shared_fpn_lateral_c4_c5"
        elif name.startswith("encoder.fpn.lateral.c2.") or name.startswith(
            "encoder.fpn.lateral.c3."
        ):
            group = "p2_geometry_pyramid"
        elif name.startswith("encoder.fpn.output.") or name.startswith(
            "encoder.proj."
        ):
            group = "p2_geometry_pyramid"
        elif (
            name.startswith("encoder.fpn.pyramid_outputs.p4.")
            or name.startswith("encoder.fpn.pyramid_outputs.p5.")
            or name.startswith("encoder.ms_proj.p4.")
            or name.startswith("encoder.ms_proj.p5.")
        ):
            group = "p4_p5_semantic_pyramid"
        elif name.startswith("structured_query_head.lane_state_layers."):
            semantic_markers = (
                ".semantic_",
                ".semantic_attention.",
                ".norm_semantic_",
            )
            group = (
                "semantic_score_view"
                if any(marker in name for marker in semantic_markers)
                else "unified_lane_state_core"
            )
        elif name.startswith("structured_query_head.exist.") or name.startswith(
            "structured_query_head.decision_norm."
        ):
            group = "foreground_score_head"
        elif (
            name.startswith("structured_query_head.row_x.")
            or name.startswith("structured_query_head.row_delta_heads.")
            or name.startswith("structured_query_head.range.")
        ):
            group = "geometry_prediction_heads"
        elif (
            name.startswith("structured_query_head.layers.")
            or name.startswith("structured_query_head.feature_proj.")
            or name.startswith("structured_query_head.reference_")
            or name.startswith("structured_query_head.instance_tokens.")
            or name.startswith("structured_query_head.row_tokens.")
            or name.startswith("structured_query_head.x_tokens.")
        ):
            group = "row_reference_geometry_path"
        elif name.startswith("structured_query_head.row_norm.") or name.startswith(
            "structured_query_head.lane_norm."
        ):
            group = "shared_lane_readout_norms"
        if group is None:
            continue
        parameter_index[parameter] = len(selected)
        selected.append(parameter)
        grouped[group].append(parameter)

    return (
        {name: tuple(parameters) for name, parameters in grouped.items()},
        tuple(selected),
        parameter_index,
    )


def _slice_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: tuple[torch.nn.Parameter, ...],
    parameter_index: dict[torch.nn.Parameter, int],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(gradients[parameter_index[parameter]] for parameter in parameters)


def _loss_groups(
    criterion: torch.nn.Module,
    losses: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cfg = criterion.cfg
    row_dfl_weight = float(criterion.row_dfl_weight())
    intermediate_weight = float(cfg.lambda_intermediate)

    geometry = (
        float(cfg.w_point) * losses["loss_point"]
        + float(cfg.w_range) * losses["loss_range"]
        + float(cfg.w_smooth) * losses["loss_smooth"]
        + float(cfg.w_line_iou) * losses["loss_line_iou"]
        + row_dfl_weight * losses["loss_row_dfl"]
    )
    score = (
        float(cfg.w_exist) * losses["loss_exist"]
        + float(cfg.w_quality) * losses["loss_quality"]
        + float(cfg.w_cardinality) * losses["loss_cardinality"]
        + float(cfg.w_score_margin) * losses["loss_score_margin"]
        + float(cfg.w_set_selection) * losses["loss_set_selection"]
    )
    if intermediate_weight > 0.0:
        geometry = geometry + intermediate_weight * (
            float(cfg.w_point) * losses["loss_intermediate_point"]
            + float(cfg.w_range) * losses["loss_intermediate_range"]
            + float(cfg.w_line_iou)
            * losses["loss_intermediate_line_iou"]
            + row_dfl_weight * losses["loss_intermediate_row_dfl"]
        )
        score = score + (
            intermediate_weight
            * float(cfg.w_exist)
            * losses["loss_intermediate_exist"]
        )
    dense = (
        float(cfg.w_seg) * losses["loss_seg"]
        + float(cfg.w_centerline) * losses["loss_centerline"]
    )
    return {"geometry": geometry, "score": score, "dense": dense}


def _safe_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if float(left_array.std()) <= 1e-12 or float(right_array.std()) <= 1e-12:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _average_precision(scores: list[float], labels: list[int]) -> float | None:
    if not scores or len(scores) != len(labels) or sum(labels) == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if int(labels[index]) == 0:
            continue
        positive += 1
        precision_sum += float(positive) / float(rank)
    return precision_sum / float(sum(labels))


def _metric_summary(counter: dict[str, int]) -> dict[str, float | int]:
    tp = int(counter["tp"])
    fp = int(counter["fp"])
    fn = int(counter["fn"])
    precision = float(tp) / float(max(tp + fp, 1))
    recall = float(tp) / float(max(tp + fn, 1))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _build_verdict(
    gradient: dict[str, Any],
    alignment: dict[str, Any],
    capacity: dict[str, Any],
    count: dict[str, Any],
) -> dict[str, Any]:
    evidence: list[str] = []
    shared_groups = (
        "shared_backbone",
        "shared_fpn_lateral_c4_c5",
        "unified_lane_state_core",
        "row_reference_geometry_path",
    )
    conflicting = []
    for name in shared_groups:
        row = gradient.get(name, {})
        cosine = row.get("cosine_geometry_vs_score", {}).get("mean")
        ratio = row.get("score_to_geometry_norm_ratio", {}).get("mean")
        if cosine is not None and ratio is not None and cosine < -0.05 and ratio > 0.10:
            conflicting.append(name)
    if conflicting:
        evidence.append("material_score_geometry_gradient_conflict")

    pearson = alignment.get("pearson_score_vs_best_official_iou")
    unique_ap = alignment.get("unique_candidate_ap_050")
    if (pearson is not None and pearson < 0.25) or (
        unique_ap is not None and unique_ap < 0.50
    ):
        evidence.append("weak_score_localization_alignment")

    direct = capacity.get("0.50", {}).get("direct_topk_recall")
    oracle = capacity.get("0.50", {}).get("oracle_topk_recall")
    all_raw = capacity.get("0.50", {}).get("all_candidates_recall")
    if direct is not None and oracle is not None and oracle - direct >= 0.05:
        evidence.append("ranking_leaves_recoverable_geometry_unselected")
    if all_raw is not None and all_raw < 0.75:
        evidence.append("candidate_geometry_capacity_is_still_limited")

    count_mae = count.get("probability_mass_vs_training_count_mae")
    if count_mae is not None and count_mae > 0.75:
        evidence.append("foreground_probability_mass_is_miscalibrated")

    if not evidence:
        primary = "no_single_failure_confirmed_on_this_sample"
    elif "candidate_geometry_capacity_is_still_limited" in evidence and (
        "weak_score_localization_alignment" in evidence
        or "ranking_leaves_recoverable_geometry_unselected" in evidence
    ):
        primary = "mixed_geometry_and_scoring_bottleneck"
    elif "candidate_geometry_capacity_is_still_limited" in evidence:
        primary = "geometry_capacity_bottleneck"
    elif "material_score_geometry_gradient_conflict" in evidence:
        primary = "shared_path_gradient_conflict"
    else:
        primary = "score_assignment_calibration_bottleneck"
    return {
        "primary_signal": primary,
        "evidence": evidence,
        "conflicting_parameter_groups": conflicting,
        "warning": (
            "This is a frozen-checkpoint diagnostic. It identifies local "
            "training-contract evidence; it does not predict the exact F1 of "
            "a retrained intervention."
        ),
    }


def main() -> None:
    from tqdm import tqdm

    args = parse_args()
    if int(args.gradient_images) < 0:
        raise ValueError("gradient-images must be non-negative")
    if int(args.top_k) < 1:
        raise ValueError("top-k must be positive")
    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    cfg = _prepare_config(
        args.config,
        dataset_root=args.dataset_root,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    row_reference_cfg = (
        cfg.get("model", {})
        .get("structured_query", {})
        .get("row_reference", {})
    )
    bounded_delta_enabled = (
        str(row_reference_cfg.get("prediction_mode", "absolute")).strip().lower()
        == "bounded_delta"
    )
    configured_delta_offsets = tuple(
        float(value)
        for value in row_reference_cfg.get("delta_offsets_px", ())
    )
    bounded_delta_radius_px = (
        max(abs(value) for value in configured_delta_offsets)
        if configured_delta_offsets
        else None
    )
    model = build_model(cfg)
    checkpoint_iteration = int(
        load_checkpoint(args.checkpoint, model, strict=False)
    )
    model = model.to(device).eval()
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(checkpoint_iteration)
    parameter_groups, selected_parameters, parameter_index = _parameter_groups(
        model
    )
    if not selected_parameters:
        raise ValueError("no unified lane-set parameters were selected")

    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
    total_batches = (
        min(len(loader), int(args.max_batches))
        if int(args.max_batches) > 0
        else len(loader)
    )

    gradient_stats = {
        name: GradientAlignmentStats() for name in parameter_groups
    }
    weighted_loss_values = defaultdict(ScalarStats)
    reconstruction_error = ScalarStats()
    all_scores: list[float] = []
    all_best_iou: list[float] = []
    matched_scores: list[float] = []
    unmatched_scores: list[float] = []
    matched_best_iou: list[float] = []
    unmatched_best_iou: list[float] = []
    candidate_labels: dict[float, list[int]] = {
        float(threshold): [] for threshold in args.iou_thresholds
    }
    unique_labels: dict[float, list[int]] = {
        float(threshold): [] for threshold in args.iou_thresholds
    }
    capacity_counts = {
        float(threshold): {
            "gt": 0,
            "all_hits": 0,
            "direct_hits": 0,
            "oracle_hits": 0,
        }
        for threshold in args.iou_thresholds
    }
    deployed_counts = {
        float(score_threshold): {
            float(iou_threshold): {"tp": 0, "fp": 0, "fn": 0}
            for iou_threshold in args.iou_thresholds
        }
        for score_threshold in args.score_thresholds
    }
    probability_count_errors: list[float] = []
    threshold_count_errors: dict[float, list[float]] = {
        float(threshold): [] for threshold in args.score_thresholds
    }
    predicted_probability_counts: list[float] = []
    training_target_counts: list[float] = []
    official_target_counts: list[float] = []
    bounded_delta_maxima: list[list[float]] = []
    bounded_delta_means: list[list[float]] = []
    bounded_delta_violations = 0
    bounded_delta_observations = 0
    images_seen = 0
    gradient_images_seen = 0

    iterator = tqdm(
        loader,
        total=total_batches,
        desc="unified lane-set contract audit",
        ncols=100,
    )
    for batch_index, (images_cpu, targets_cpu, metas) in enumerate(iterator):
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        if device.type == "cuda":
            images = images_cpu.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images_cpu.to(device)
        targets = nested_to_device(targets_cpu, device)

        with torch.no_grad(), _amp_context(device, amp_dtype):
            metric_outputs = model(images, inference_only=True)
            metric_matches = matcher(metric_outputs, targets)

        for image_index, (target_cpu, meta, match) in enumerate(
            zip(targets_cpu, metas, metric_matches)
        ):
            stage = _stage_for_image(metric_outputs, image_index)
            layer_maxima = stage.get("bounded_delta_max_abs_by_layer")
            layer_means = stage.get("bounded_delta_mean_abs_by_layer")
            if isinstance(layer_maxima, torch.Tensor):
                maxima = layer_maxima.reshape(-1).tolist()
                means = (
                    layer_means.reshape(-1).tolist()
                    if isinstance(layer_means, torch.Tensor)
                    else [float("nan")] * len(maxima)
                )
                while len(bounded_delta_maxima) < len(maxima):
                    bounded_delta_maxima.append([])
                    bounded_delta_means.append([])
                for layer_index, (maximum, mean) in enumerate(zip(maxima, means)):
                    maximum = float(maximum)
                    bounded_delta_maxima[layer_index].append(maximum)
                    bounded_delta_means[layer_index].append(float(mean))
                    bounded_delta_observations += 1
                    if (
                        bounded_delta_radius_px is not None
                        and maximum > bounded_delta_radius_px + 1e-4
                    ):
                        bounded_delta_violations += 1
            record = {"stages": {"main": stage}, "meta": meta}
            official_iou, candidate_valid = official_proposal_gt_iou_matrix(
                record,
                "main",
                line_width=float(args.line_width),
                min_valid_rows=int(args.min_valid_rows),
                row_visibility_thresh=float(args.row_visibility_thresh),
            )
            scores = torch.softmax(stage["exist_logits"].float(), dim=-1)[..., 0]
            if int(official_iou.shape[0]) > 0:
                best_iou = official_iou.max(dim=0).values
            else:
                best_iou = torch.zeros_like(scores)
            valid_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten().tolist()
            matched_ids = set(
                int(value)
                for value in match["pred_indices"].detach().cpu().tolist()
            )
            for candidate_index in valid_ids:
                score = float(scores[candidate_index])
                iou = float(best_iou[candidate_index])
                all_scores.append(score)
                all_best_iou.append(iou)
                if candidate_index in matched_ids:
                    matched_scores.append(score)
                    matched_best_iou.append(iou)
                else:
                    unmatched_scores.append(score)
                    unmatched_best_iou.append(iou)
            official_target_count = float(int(official_iou.shape[0]))
            training_target_count = float(int(target_cpu["x_rows"].shape[0]))
            predicted_probability_count = float(scores[candidate_valid].sum())
            selected_by_threshold: dict[float, list[int]] = {}
            for score_threshold in args.score_thresholds:
                score_threshold = float(score_threshold)
                threshold_ids = [
                    index
                    for index in valid_ids
                    if float(scores[index]) >= score_threshold
                ]
                threshold_ids.sort(
                    key=lambda index: float(scores[index]),
                    reverse=True,
                )
                selected_by_threshold[score_threshold] = threshold_ids[
                    : int(args.top_k)
                ]
            training_target_counts.append(training_target_count)
            official_target_counts.append(official_target_count)
            predicted_probability_counts.append(predicted_probability_count)
            probability_count_errors.append(
                abs(predicted_probability_count - training_target_count)
            )
            for score_threshold, selected_ids in selected_by_threshold.items():
                threshold_count_errors[score_threshold].append(
                    abs(float(len(selected_ids)) - official_target_count)
                )

            direct_ids = sorted(
                valid_ids,
                key=lambda index: float(scores[index]),
                reverse=True,
            )[: int(args.top_k)]
            for threshold in args.iou_thresholds:
                threshold = float(threshold)
                labels, _unique_assignment = unique_candidate_labels(
                    official_iou,
                    threshold,
                    candidate_valid,
                )
                candidate_labels[threshold].extend(
                    int(float(best_iou[index]) > threshold)
                    for index in valid_ids
                )
                unique_labels[threshold].extend(
                    int(labels[index] == "unique_tp") for index in valid_ids
                )
                all_hits, gt_count, _ = recall_from_ids(
                    official_iou,
                    valid_ids,
                    threshold,
                )
                direct_hits, _gt_count, _ = recall_from_ids(
                    official_iou,
                    direct_ids,
                    threshold,
                )
                oracle = cardinality_oracle_assignment(
                    official_iou,
                    threshold,
                    int(args.top_k),
                    candidate_valid,
                )
                capacity_counts[threshold]["gt"] += int(gt_count)
                capacity_counts[threshold]["all_hits"] += int(all_hits)
                capacity_counts[threshold]["direct_hits"] += int(direct_hits)
                capacity_counts[threshold]["oracle_hits"] += int(
                    oracle.hit_count
                )

                for score_threshold, selected_ids in selected_by_threshold.items():
                    deployed_hits, deployed_gt, _ = recall_from_ids(
                        official_iou,
                        selected_ids,
                        threshold,
                    )
                    counter = deployed_counts[score_threshold][threshold]
                    counter["tp"] += int(deployed_hits)
                    counter["fp"] += int(len(selected_ids) - deployed_hits)
                    counter["fn"] += int(deployed_gt - deployed_hits)

        # Parameter-gradient measurement is intentionally performed one image
        # at a time. It is independent of the larger inference batch above and
        # remains safe on the 16 GiB training target.
        remaining = int(args.gradient_images) - gradient_images_seen
        for image_index in range(min(max(remaining, 0), int(images.shape[0]))):
            image = images[image_index : image_index + 1]
            gradient_targets = [targets[image_index]]
            with _amp_context(device, amp_dtype):
                outputs, matches = forward_with_matches(
                    model,
                    image,
                    gradient_targets,
                    matcher,
                    cfg,
                    checkpoint_iteration,
                )
                criterion.set_iteration(checkpoint_iteration)
                losses = criterion(outputs, gradient_targets, matches)
                grouped_losses = _loss_groups(criterion, losses)
            for name, loss in grouped_losses.items():
                weighted_loss_values[name].update(
                    float(loss.detach().float())
                )
            reconstructed = sum(grouped_losses.values())
            reconstruction_error.update(
                abs(
                    float(
                        (losses["loss_total"] - reconstructed)
                        .detach()
                        .float()
                    )
                )
            )
            geometry_gradients = torch.autograd.grad(
                grouped_losses["geometry"],
                selected_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            score_gradients = torch.autograd.grad(
                grouped_losses["score"],
                selected_parameters,
                retain_graph=False,
                allow_unused=True,
            )
            for name, parameters in parameter_groups.items():
                gradient_stats[name].update(
                    _slice_gradients(
                        geometry_gradients,
                        parameters,
                        parameter_index,
                    ),
                    _slice_gradients(
                        score_gradients,
                        parameters,
                        parameter_index,
                    ),
                )
            gradient_images_seen += 1
            del outputs, matches, losses, grouped_losses
            del geometry_gradients, score_gradients

        images_seen += int(images.shape[0])
        del metric_outputs, metric_matches, images, targets

    score_alignment: dict[str, Any] = {
        "candidates": len(all_scores),
        "pearson_score_vs_best_official_iou": _safe_correlation(
            all_scores,
            all_best_iou,
        ),
        "matched_mean_score": _mean(matched_scores),
        "unmatched_mean_score": _mean(unmatched_scores),
        "matched_mean_best_official_iou": _mean(matched_best_iou),
        "unmatched_mean_best_official_iou": _mean(unmatched_best_iou),
        "matched_fraction_iou_ge_050": (
            _mean([float(value > 0.50) for value in matched_best_iou])
        ),
        "matched_fraction_iou_ge_075": (
            _mean([float(value > 0.75) for value in matched_best_iou])
        ),
    }
    for threshold in args.iou_thresholds:
        threshold = float(threshold)
        suffix = f"{int(round(100.0 * threshold)):03d}"
        score_alignment[f"candidate_best_iou_ap_{suffix}"] = _average_precision(
            all_scores,
            candidate_labels[threshold],
        )
        score_alignment[f"unique_candidate_ap_{suffix}"] = _average_precision(
            all_scores,
            unique_labels[threshold],
        )

    capacity: dict[str, Any] = {}
    for threshold, counts in capacity_counts.items():
        gt = max(int(counts["gt"]), 1)
        capacity[f"{threshold:.2f}"] = {
            "gt_lanes": int(counts["gt"]),
            "all_candidates_recall": float(counts["all_hits"]) / gt,
            "direct_topk_recall": float(counts["direct_hits"]) / gt,
            "oracle_topk_recall": float(counts["oracle_hits"]) / gt,
            "oracle_minus_direct_recall_points": 100.0
            * float(counts["oracle_hits"] - counts["direct_hits"])
            / gt,
        }
    deployed = {
        f"score_{score_threshold:.2f}": {
            f"{iou_threshold:.2f}": _metric_summary(counts)
            for iou_threshold, counts in by_iou.items()
        }
        for score_threshold, by_iou in deployed_counts.items()
    }
    count_calibration = {
        "images": images_seen,
        "mean_training_target_lane_count": _mean(training_target_counts),
        "mean_official_target_lane_count": _mean(official_target_counts),
        "mean_foreground_probability_mass": _mean(
            predicted_probability_counts
        ),
        "probability_mass_vs_training_count_mae": _mean(
            probability_count_errors
        ),
        "thresholded_topk_vs_official_count_mae": {
            f"score_{threshold:.2f}": _mean(values)
            for threshold, values in threshold_count_errors.items()
        },
        "score_thresholds": [
            float(threshold) for threshold in args.score_thresholds
        ],
        "top_k": int(args.top_k),
    }
    gradient_summary = {
        name: stats.summary() for name, stats in gradient_stats.items()
    }
    bounded_delta_contract = {
        "enabled": bool(bounded_delta_enabled),
        "configured_offsets_px": list(configured_delta_offsets),
        "configured_radius_px": bounded_delta_radius_px,
        "observed_max_abs_px_by_layer": [
            max(values) if values else None for values in bounded_delta_maxima
        ],
        "observed_mean_abs_px_by_layer": [
            _mean(values) for values in bounded_delta_means
        ],
        "observed_global_max_abs_px": (
            max(max(values) for values in bounded_delta_maxima if values)
            if any(bounded_delta_maxima)
            else None
        ),
        "observations": int(bounded_delta_observations),
        "trust_region_violations": int(bounded_delta_violations),
        "trust_region_satisfied": bool(
            bounded_delta_enabled
            and bounded_delta_observations > 0
            and bounded_delta_violations == 0
        ),
    }
    payload = {
        "diagnostic_only": True,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "gradient_images": gradient_images_seen,
        "settings": {
            "amp_dtype": args.amp_dtype,
            "line_width": float(args.line_width),
            "min_valid_rows": int(args.min_valid_rows),
            "row_visibility_thresh": float(args.row_visibility_thresh),
            "top_k": int(args.top_k),
            "score_thresholds": [
                float(threshold) for threshold in args.score_thresholds
            ],
            "iou_thresholds": [float(value) for value in args.iou_thresholds],
            "intermediate_supervision_included_in_gradients": bool(
                float(criterion.cfg.lambda_intermediate) > 0.0
            ),
        },
        "selected_gradient_parameters": {
            name: sum(parameter.numel() for parameter in parameters)
            for name, parameters in parameter_groups.items()
        },
        "weighted_loss_values": {
            name: stats.summary()
            for name, stats in weighted_loss_values.items()
        },
        "loss_reconstruction_abs_error": reconstruction_error.summary(),
        "gradient_alignment": gradient_summary,
        "score_official_iou_alignment": score_alignment,
        "count_calibration": count_calibration,
        "capacity": capacity,
        "deployed_operating_points": deployed,
        "bounded_delta_contract": bounded_delta_contract,
    }
    payload["verdict"] = _build_verdict(
        gradient_summary,
        score_alignment,
        capacity,
        count_calibration,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = dict(payload)
    compact.pop("sampled_dataset_indices")
    print(json.dumps(compact, indent=2))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
