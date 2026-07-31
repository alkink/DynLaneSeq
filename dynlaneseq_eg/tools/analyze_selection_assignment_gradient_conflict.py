from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.data.lane_target_builder import decode_targets_to_points
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    _raster_lane_mask,
    lanes_to_original,
    official_proposal_gt_iou_matrix,
    proposal_gt_iou_matrix,
)
from dynlaneseq_eg.evaluation.culane_metric import load_culane_img_data
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
            "Audit whether the main Hungarian matcher and the unique "
            "set-selection target supervise different lane slots, and whether "
            "their gradients conflict at the final structured decoder state."
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
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--gradient-batches", type=int, default=4)
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
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--official-win-margin", type=float, default=0.02)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


@dataclass
class ScalarStats:
    values: list[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        if math.isfinite(float(value)):
            self.values.append(float(value))

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.update(float(value))

    def summary(self) -> dict[str, float | int]:
        if not self.values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        ordered = sorted(self.values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            median = ordered[middle]
        else:
            median = 0.5 * (ordered[middle - 1] + ordered[middle])
        return {
            "count": len(ordered),
            "mean": sum(ordered) / len(ordered),
            "median": median,
            "min": ordered[0],
            "max": ordered[-1],
        }


@dataclass
class GradientPairStats:
    cosines: ScalarStats = field(default_factory=ScalarStats)
    right_to_left_norm_ratios: ScalarStats = field(default_factory=ScalarStats)
    negative: int = 0
    valid: int = 0

    def update(self, left: torch.Tensor, right: torch.Tensor) -> None:
        left_flat = left.detach().double().reshape(-1)
        right_flat = right.detach().double().reshape(-1)
        left_norm = float(left_flat.norm())
        right_norm = float(right_flat.norm())
        if left_norm <= 0.0 or right_norm <= 0.0:
            return
        cosine = float(
            torch.dot(left_flat, right_flat) / (left_norm * right_norm)
        )
        self.cosines.update(cosine)
        self.right_to_left_norm_ratios.update(right_norm / left_norm)
        self.valid += 1
        self.negative += int(cosine < 0.0)

    def summary(self) -> dict[str, Any]:
        return {
            "cosine": self.cosines.summary(),
            "right_to_left_norm_ratio": (
                self.right_to_left_norm_ratios.summary()
            ),
            "negative_fraction": (
                float(self.negative) / float(self.valid)
                if self.valid
                else 0.0
            ),
        }


def _amp_context(
    device: torch.device,
    amp_dtype: torch.dtype | None,
):
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
    # Intermediate outputs do not contribute to the gradient with respect to
    # the final decoder state. Avoid materializing them in this diagnostic.
    cfg.setdefault("loss", {})["lambda_intermediate"] = 0.0
    return cfg


def _stage_for_image(
    outputs: dict[str, torch.Tensor],
    batch_index: int,
) -> dict[str, torch.Tensor]:
    fields = (
        "pred_x_rows",
        "range_norm",
        "exist_logits",
        "quality_logits",
        "selection_logits",
    )
    return {
        name: outputs[name][batch_index].detach().float().cpu()
        for name in fields
        if isinstance(outputs.get(name), torch.Tensor)
    }


def _target_to_official_mapping(
    target: dict[str, torch.Tensor],
    meta: dict[str, Any],
    *,
    line_width: float,
) -> tuple[dict[int, int], list[float]]:
    """Map filtered fixed-row targets back to original annotation lanes."""

    target_lanes = decode_targets_to_points(
        target["x_rows"].detach().float().cpu().numpy(),
        target["valid_mask"].detach().bool().cpu().numpy(),
        input_h=int(meta.get("input_h", 288)),
    )
    target_lanes = lanes_to_original(target_lanes, meta)
    annotation_path = str(meta.get("anno_path", ""))
    official_lanes = (
        load_culane_img_data(annotation_path) if annotation_path else []
    )
    if not target_lanes or not official_lanes:
        return {}, []

    image_h = int(meta.get("orig_h", 590))
    image_w = int(meta.get("orig_w", 1640))
    width = int(round(float(line_width)))
    target_masks = [
        _raster_lane_mask(lane, image_h, image_w, width)
        for lane in target_lanes
    ]
    official_masks = [
        _raster_lane_mask(lane, image_h, image_w, width)
        for lane in official_lanes
    ]
    matrix = np.zeros(
        (len(target_masks), len(official_masks)),
        dtype=np.float32,
    )
    target_counts = [int(mask.sum()) for mask in target_masks]
    official_counts = [int(mask.sum()) for mask in official_masks]
    for target_index, target_mask in enumerate(target_masks):
        for official_index, official_mask in enumerate(official_masks):
            intersection = int(
                cv2.countNonZero(
                    cv2.bitwise_and(target_mask, official_mask)
                )
            )
            union = (
                target_counts[target_index]
                + official_counts[official_index]
                - intersection
            )
            matrix[target_index, official_index] = (
                0.0 if union <= 0 else float(intersection) / float(union)
            )
    target_indices, official_indices = linear_sum_assignment(1.0 - matrix)
    mapping = {
        int(target_index): int(official_index)
        for target_index, official_index in zip(
            target_indices,
            official_indices,
        )
    }
    qualities = [
        float(matrix[target_index, official_index])
        for target_index, official_index in zip(
            target_indices,
            official_indices,
        )
    ]
    return mapping, qualities


def _main_assignment_map(
    match: dict[str, torch.Tensor],
) -> dict[int, int]:
    return {
        int(gt_index): int(pred_index)
        for pred_index, gt_index in zip(
            match["pred_indices"].detach().cpu().tolist(),
            match["gt_indices"].detach().cpu().tolist(),
        )
    }


def _selection_assignment_map(
    quality: torch.Tensor,
    valid_target_mask: torch.Tensor,
) -> dict[int, int]:
    """Reproduce the positive pure-quality targets used by set selection."""

    if int(quality.shape[0]) == 0 or int(quality.shape[1]) == 0:
        return {}
    gt_rows, candidate_columns = linear_sum_assignment(
        1.0 - quality.detach().float().cpu().numpy()
    )
    target_ids = torch.where(
        valid_target_mask.detach().bool().cpu()
    )[0].tolist()
    assignment: dict[int, int] = {}
    for gt_row, candidate_index in zip(gt_rows, candidate_columns):
        # The loss writes an assigned quality of exactly zero back into an
        # all-zero target tensor. Such a pair is not a positive teacher and
        # must not be counted as an assignment conflict.
        if float(quality[int(gt_row), int(candidate_index)]) <= 0.0:
            continue
        assignment[int(target_ids[int(gt_row)])] = int(candidate_index)
    return assignment


def compare_assignment_maps(
    main_by_gt: dict[int, int],
    selection_by_gt: dict[int, int],
    official_iou: torch.Tensor,
    target_to_official: dict[int, int],
    *,
    win_margin: float,
) -> dict[str, Any]:
    """Compare two assignments in official raster-IoU space."""

    common_gt = sorted(set(main_by_gt) & set(selection_by_gt))
    result: dict[str, Any] = {
        "gt": len(common_gt),
        "agreement": 0,
        "disagreement": 0,
        "main_iou": [],
        "selection_iou": [],
        "official_delta_selection_minus_main": [],
        "disagreement_main_iou": [],
        "disagreement_selection_iou": [],
        "disagreement_delta_selection_minus_main": [],
        "selection_wins": 0,
        "main_wins": 0,
        "ties": 0,
        "transitions": {
            "0.50": {
                "main_miss_selection_hit": 0,
                "main_hit_selection_miss": 0,
            },
            "0.70": {
                "main_miss_selection_hit": 0,
                "main_hit_selection_miss": 0,
            },
        },
    }
    for target_gt in common_gt:
        official_gt = target_to_official.get(int(target_gt))
        if official_gt is None or official_gt >= int(official_iou.shape[0]):
            continue
        main_candidate = int(main_by_gt[target_gt])
        selection_candidate = int(selection_by_gt[target_gt])
        main_iou = float(official_iou[official_gt, main_candidate])
        selection_iou = float(
            official_iou[official_gt, selection_candidate]
        )
        delta = selection_iou - main_iou
        result["main_iou"].append(main_iou)
        result["selection_iou"].append(selection_iou)
        result["official_delta_selection_minus_main"].append(delta)
        if main_candidate == selection_candidate:
            result["agreement"] += 1
        else:
            result["disagreement"] += 1
            result["disagreement_main_iou"].append(main_iou)
            result["disagreement_selection_iou"].append(selection_iou)
            result["disagreement_delta_selection_minus_main"].append(delta)
            if delta > float(win_margin):
                result["selection_wins"] += 1
            elif delta < -float(win_margin):
                result["main_wins"] += 1
            else:
                result["ties"] += 1
        for threshold in (0.5, 0.7):
            key = f"{threshold:.2f}"
            main_hit = main_iou >= threshold
            selection_hit = selection_iou >= threshold
            if not main_hit and selection_hit:
                result["transitions"][key][
                    "main_miss_selection_hit"
                ] += 1
            elif main_hit and not selection_hit:
                result["transitions"][key][
                    "main_hit_selection_miss"
                ] += 1
    return result


def _loss_groups(
    criterion: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    cfg = criterion.cfg
    zero = outputs["pred_x_rows"].sum() * 0.0
    classification = zero
    if float(cfg.w_exist) != 0.0:
        classification = classification + float(
            cfg.w_exist
        ) * criterion.compute_exist_loss(outputs, matches)
    if float(cfg.w_quality) != 0.0:
        classification = classification + float(
            cfg.w_quality
        ) * criterion.compute_quality_loss(outputs, targets, matches)

    geometry = zero
    if float(cfg.w_point) != 0.0:
        geometry = geometry + float(
            cfg.w_point
        ) * criterion.compute_point_loss(outputs, targets, matches)
    if float(cfg.w_range) != 0.0:
        geometry = geometry + float(
            cfg.w_range
        ) * criterion.compute_range_loss(outputs, targets, matches)
    if float(cfg.w_smooth) != 0.0:
        geometry = geometry + float(
            cfg.w_smooth
        ) * criterion.compute_smoothness_loss(outputs, targets, matches)
    if float(cfg.w_line_iou) != 0.0:
        geometry = geometry + float(
            cfg.w_line_iou
        ) * criterion.compute_line_iou_loss(outputs, targets, matches)
    row_dfl_weight = float(criterion.row_dfl_weight())
    if row_dfl_weight != 0.0:
        geometry = geometry + row_dfl_weight * (
            criterion.compute_row_dfl_loss(outputs, targets, matches)
        )

    if float(cfg.w_set_selection) == 0.0:
        raise ValueError(
            "The audited config must enable w_set_selection"
        )
    selection = float(cfg.w_set_selection) * (
        criterion.compute_set_selection_loss(outputs, targets)["total"]
    )
    return {
        "classification": classification,
        "geometry": geometry,
        "main": classification + geometry,
        "selection": selection,
    }


def _autograd_bundle(
    loss: torch.Tensor,
    row_state: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
    gradients = torch.autograd.grad(
        loss,
        (row_state, *parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    state_gradient = gradients[0]
    if state_gradient is None:
        state_gradient = torch.zeros_like(row_state)
    return state_gradient.detach(), tuple(
        gradient.detach() if gradient is not None else None
        for gradient in gradients[1:]
    )


def _add_parameter_gradients(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    output: list[torch.Tensor | None] = []
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is None:
            output.append(right_gradient)
        elif right_gradient is None:
            output.append(left_gradient)
        else:
            output.append(left_gradient + right_gradient)
    return tuple(output)


def _parameter_pair_tensors(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    left_parts: list[torch.Tensor] = []
    right_parts: list[torch.Tensor] = []
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is None and right_gradient is None:
            continue
        template = (
            left_gradient if left_gradient is not None else right_gradient
        )
        assert template is not None
        left_parts.append(
            (
                left_gradient
                if left_gradient is not None
                else torch.zeros_like(template)
            ).reshape(-1)
        )
        right_parts.append(
            (
                right_gradient
                if right_gradient is not None
                else torch.zeros_like(template)
            ).reshape(-1)
        )
    if not left_parts:
        return None
    return torch.cat(left_parts), torch.cat(right_parts)


def _summarize_assignment(
    counts: dict[str, int],
    scalars: dict[str, ScalarStats],
    transitions: dict[str, dict[str, int]],
) -> dict[str, Any]:
    gt = int(counts["gt"])
    disagreement = int(counts["disagreement"])
    comparable = int(counts["agreement"] + counts["disagreement"])
    return {
        **counts,
        "comparable_gt": comparable,
        "agreement_rate": (
            float(counts["agreement"]) / float(comparable)
            if comparable
            else 0.0
        ),
        "disagreement_rate": (
            float(disagreement) / float(comparable)
            if comparable
            else 0.0
        ),
        "selection_win_fraction_on_disagreement": (
            float(counts["selection_wins"]) / float(disagreement)
            if disagreement
            else 0.0
        ),
        "main_win_fraction_on_disagreement": (
            float(counts["main_wins"]) / float(disagreement)
            if disagreement
            else 0.0
        ),
        "official_iou": {
            name: stats.summary() for name, stats in scalars.items()
        },
        "threshold_transitions": transitions,
    }


def _verdict(
    assignment: dict[str, Any],
    gradient: dict[str, Any],
) -> dict[str, Any]:
    disagreement = float(assignment["disagreement_rate"])
    delta = float(
        assignment["official_iou"][
            "disagreement_delta_selection_minus_main"
        ]["mean"]
    )
    row_cosine = float(
        gradient["row_state"]["selection_vs_main"]["cosine"]["mean"]
    )
    parameter_cosine = float(
        gradient["final_decoder_parameters"][
            "selection_vs_main"
        ]["cosine"]["mean"]
    )
    substantial_disagreement = disagreement >= 0.20
    opposing_gradient = (
        row_cosine < -0.05 or parameter_cosine < -0.05
    )
    surrogate_misaligned = (
        int(assignment["disagreement"]) > 0 and delta < -0.005
    )
    if surrogate_misaligned:
        diagnosis = (
            "selection_target_surrogate_misaligned_with_official_metric"
        )
        next_action = (
            "replace the independent surrogate assignment with the main "
            "matcher assignment or an official-aligned shared target before "
            "another joint-training gate"
        )
    elif substantial_disagreement and opposing_gradient:
        diagnosis = "assignment_gradient_conflict_confirmed"
        next_action = (
            "share one assignment across existence, geometry, quality, and "
            "selection; then run one matched 5k causal gate"
        )
    elif substantial_disagreement:
        diagnosis = (
            "assignment_mismatch_present_without_opposing_gradient_proof"
        )
        next_action = (
            "run a shared-assignment 5k gate only if the selection-assigned "
            "slots have better official IoU"
        )
    elif opposing_gradient:
        diagnosis = (
            "objective_gradient_conflict_without_assignment_identity_shift"
        )
        next_action = (
            "rebalance or decouple selection gradients; changing matcher "
            "identity alone is unlikely to solve the bottleneck"
        )
    else:
        diagnosis = "assignment_conflict_not_supported"
        next_action = (
            "stop modifying assignment and return to proposal-state "
            "separability or decoder evidence acquisition"
        )
    return {
        "substantial_assignment_disagreement": substantial_disagreement,
        "opposing_shared_gradient": opposing_gradient,
        "selection_surrogate_officially_misaligned": surrogate_misaligned,
        "diagnosis": diagnosis,
        "next_action": next_action,
    }


def main() -> None:
    args = parse_args()
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
    model = build_model(cfg)
    checkpoint_iteration = int(
        load_checkpoint(args.checkpoint, model, strict=False)
    )
    model = model.to(device).eval()
    head = model.structured_query_head
    if head is None:
        raise ValueError("structured query head is required")
    if head.set_selection_head is None:
        raise ValueError("checkpoint/config must enable set_selection")
    head.intermediate_supervision = False
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(checkpoint_iteration)
    last_layer_parameters = tuple(head.layers[-1].parameters())

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

    assignment_counts = defaultdict(int)
    assignment_scalars = defaultdict(ScalarStats)
    transitions = {
        "0.50": {
            "main_miss_selection_hit": 0,
            "main_hit_selection_miss": 0,
        },
        "0.70": {
            "main_miss_selection_hit": 0,
            "main_hit_selection_miss": 0,
        },
    }
    gradient_stats = {
        "row_state": {
            "selection_vs_main": GradientPairStats(),
            "selection_vs_classification": GradientPairStats(),
            "selection_vs_geometry": GradientPairStats(),
            "classification_vs_geometry": GradientPairStats(),
        },
        "final_decoder_parameters": {
            "selection_vs_main": GradientPairStats(),
            "selection_vs_classification": GradientPairStats(),
            "selection_vs_geometry": GradientPairStats(),
            "classification_vs_geometry": GradientPairStats(),
        },
    }
    loss_values = defaultdict(ScalarStats)
    images_seen = 0
    gradient_images = 0

    for batch_index, (images, targets_cpu, metas) in enumerate(
        tqdm(
            loader,
            total=total_batches,
            desc="assignment/gradient conflict audit",
            ncols=100,
        )
    ):
        if int(args.max_batches) > 0 and batch_index >= int(
            args.max_batches
        ):
            break
        gradient_batch = batch_index < int(args.gradient_batches)
        if device.type == "cuda":
            images = images.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images.to(device)
        targets = nested_to_device(targets_cpu, device)

        with torch.no_grad(), _amp_context(device, amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
        if gradient_batch:
            with _amp_context(device, amp_dtype):
                outputs = head(
                    encoded["features"],
                    inference_only=False,
                )
        else:
            with torch.no_grad(), _amp_context(device, amp_dtype):
                outputs = head(
                    encoded["features"],
                    inference_only=False,
                )
        matches = matcher(outputs, targets)

        for image_index, (target, meta, match) in enumerate(
            zip(targets_cpu, metas, matches)
        ):
            stage = _stage_for_image(outputs, image_index)
            surrogate_iou, valid_gt, candidate_valid = (
                proposal_gt_iou_matrix(
                    stage,
                    target,
                    input_h=int(meta.get("input_h", 288)),
                    input_w=int(meta.get("input_w", 800)),
                    line_width=float(args.line_width),
                    min_valid_rows=int(args.min_valid_rows),
                )
            )
            main_by_gt = _main_assignment_map(match)
            selection_by_gt = _selection_assignment_map(
                surrogate_iou,
                valid_gt,
            )
            assignment_counts["main_assigned_gt"] += len(main_by_gt)
            assignment_counts["selection_positive_gt"] += len(
                selection_by_gt
            )
            assignment_counts["selection_missing_main_gt"] += len(
                set(main_by_gt) - set(selection_by_gt)
            )
            target_to_official, mapping_quality = (
                _target_to_official_mapping(
                    target,
                    meta,
                    line_width=float(args.line_width),
                )
            )
            official_iou, _official_candidate_valid = (
                official_proposal_gt_iou_matrix(
                    {"stages": {"main": stage}, "meta": meta},
                    "main",
                    line_width=float(args.line_width),
                    min_valid_rows=int(args.min_valid_rows),
                )
            )
            comparison = compare_assignment_maps(
                main_by_gt,
                selection_by_gt,
                official_iou,
                target_to_official,
                win_margin=float(args.official_win_margin),
            )
            assignment_counts["gt"] += int(comparison["gt"])
            for name in (
                "agreement",
                "disagreement",
                "selection_wins",
                "main_wins",
                "ties",
            ):
                assignment_counts[name] += int(comparison[name])
            main_positive = set(main_by_gt.values())
            selection_positive = set(selection_by_gt.values())
            union = main_positive | selection_positive
            assignment_scalars["positive_set_jaccard"].update(
                (
                    float(len(main_positive & selection_positive))
                    / float(len(union))
                    if union
                    else 1.0
                )
            )
            assignment_counts["selection_invalid_assignments"] += sum(
                not bool(candidate_valid[candidate])
                for candidate in selection_positive
                if candidate < int(candidate_valid.shape[0])
            )
            assignment_scalars["target_to_official_mapping_iou"].extend(
                mapping_quality
            )
            for name in (
                "main_iou",
                "selection_iou",
                "official_delta_selection_minus_main",
                "disagreement_main_iou",
                "disagreement_selection_iou",
                "disagreement_delta_selection_minus_main",
            ):
                assignment_scalars[name].extend(comparison[name])
            for threshold in ("0.50", "0.70"):
                for transition_name, count in comparison[
                    "transitions"
                ][threshold].items():
                    transitions[threshold][transition_name] += int(count)

        if gradient_batch:
            loss_groups = _loss_groups(
                criterion,
                outputs,
                targets,
                matches,
            )
            for name, loss in loss_groups.items():
                loss_values[name].update(float(loss.detach().float()))
            row_state = outputs["structured_row_tokens"]
            classification_state, classification_parameters = (
                _autograd_bundle(
                    loss_groups["classification"],
                    row_state,
                    last_layer_parameters,
                    retain_graph=True,
                )
            )
            geometry_state, geometry_parameters = _autograd_bundle(
                loss_groups["geometry"],
                row_state,
                last_layer_parameters,
                retain_graph=True,
            )
            selection_state, selection_parameters = _autograd_bundle(
                loss_groups["selection"],
                row_state,
                last_layer_parameters,
                retain_graph=False,
            )
            main_state = classification_state + geometry_state
            main_parameters = _add_parameter_gradients(
                classification_parameters,
                geometry_parameters,
            )
            state_pairs = {
                "selection_vs_main": (main_state, selection_state),
                "selection_vs_classification": (
                    classification_state,
                    selection_state,
                ),
                "selection_vs_geometry": (
                    geometry_state,
                    selection_state,
                ),
                "classification_vs_geometry": (
                    classification_state,
                    geometry_state,
                ),
            }
            for name, (left, right) in state_pairs.items():
                for image_index in range(int(left.shape[0])):
                    gradient_stats["row_state"][name].update(
                        left[image_index],
                        right[image_index],
                    )
            parameter_pairs = {
                "selection_vs_main": (
                    main_parameters,
                    selection_parameters,
                ),
                "selection_vs_classification": (
                    classification_parameters,
                    selection_parameters,
                ),
                "selection_vs_geometry": (
                    geometry_parameters,
                    selection_parameters,
                ),
                "classification_vs_geometry": (
                    classification_parameters,
                    geometry_parameters,
                ),
            }
            for name, (left, right) in parameter_pairs.items():
                tensors = _parameter_pair_tensors(left, right)
                if tensors is not None:
                    gradient_stats["final_decoder_parameters"][
                        name
                    ].update(*tensors)
            gradient_images += int(images.shape[0])

        images_seen += int(images.shape[0])
        del outputs, encoded, images, targets

    assignment_summary = _summarize_assignment(
        dict(assignment_counts),
        dict(assignment_scalars),
        transitions,
    )
    gradient_summary = {
        location: {
            name: stats.summary()
            for name, stats in comparisons.items()
        }
        for location, comparisons in gradient_stats.items()
    }
    payload = {
        "diagnostic_only": True,
        "warning": (
            "This audit measures target identity and instantaneous gradients "
            "at a frozen checkpoint. It does not report a deployable model or "
            "predict the exact F1 of a retrained shared-assignment variant."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "gradient_images": gradient_images,
        "settings": {
            "line_width": float(args.line_width),
            "min_valid_rows": int(args.min_valid_rows),
            "official_win_margin": float(args.official_win_margin),
            "amp_dtype": args.amp_dtype,
            "intermediate_supervision_materialized": False,
        },
        "assignment": assignment_summary,
        "weighted_loss_values": {
            name: stats.summary()
            for name, stats in loss_values.items()
        },
        "gradient_alignment": gradient_summary,
    }
    payload["verdict"] = _verdict(
        assignment_summary,
        gradient_summary,
    )
    compact = dict(payload)
    compact.pop("sampled_dataset_indices")
    print(json.dumps(compact, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
