from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import (
    ProposalRecallStats,
    line_iou_against_gt,
)
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causally audit whether the structured decoder follows horizontal "
            "P2 evidence or mainly preserves learned query priors. The encoder "
            "is run once; zero, x-mean, horizontal-shift, and batch-swap "
            "interventions are then applied directly to its P2 tensor."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
        help=(
            "Uniform spans the complete ordered split; sequential is retained "
            "only to reproduce older same-video diagnostics."
        ),
    )
    parser.add_argument("--shift-cols", type=int, default=16)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def shift_feature_x(features: torch.Tensor, columns: int) -> torch.Tensor:
    """Move feature content horizontally with zero fill and no wraparound."""

    if features.ndim != 4:
        raise ValueError("features must be [B,C,H,W]")
    columns = int(columns)
    width = int(features.shape[-1])
    if abs(columns) >= width:
        return torch.zeros_like(features)
    if columns == 0:
        return features.clone()
    shifted = torch.zeros_like(features)
    if columns > 0:
        shifted[..., columns:] = features[..., : width - columns]
    else:
        amount = -columns
        shifted[..., : width - amount] = features[..., amount:]
    return shifted


def shift_targets_x(
    targets: list[dict[str, torch.Tensor]],
    shift_px: float,
    *,
    input_w: int,
) -> list[dict[str, torch.Tensor]]:
    """Return lightweight targets translated in x for equivariance recall."""

    result: list[dict[str, torch.Tensor]] = []
    for target in targets:
        shifted = dict(target)
        x_rows = target["x_rows"].float() + float(shift_px)
        valid = target["valid_mask"].bool()
        valid = valid & torch.isfinite(x_rows)
        valid = valid & (x_rows >= 0.0) & (x_rows <= float(input_w - 1))
        shifted["x_rows"] = x_rows
        shifted["valid_mask"] = valid
        result.append(shifted)
    return result


@dataclass
class MeanStats:
    count: int = 0
    total: float = 0.0
    absolute_total: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float()
        finite = torch.isfinite(values)
        if not bool(finite.any()):
            return
        values = values[finite]
        self.count += int(values.numel())
        self.total += float(values.sum())
        self.absolute_total += float(values.abs().sum())

    def summary(self) -> dict[str, float | int]:
        count = max(self.count, 1)
        return {
            "count": self.count,
            "mean": self.total / count,
            "mean_abs": self.absolute_total / count,
        }


@dataclass
class ShiftResponseStats:
    expected_shift_px: float
    count: int = 0
    signed_total: float = 0.0
    absolute_total: float = 0.0
    expected_error_total: float = 0.0
    direction_correct: int = 0
    follows_half_shift: int = 0

    def update(self, delta: torch.Tensor) -> None:
        delta = delta.detach().float()
        finite = torch.isfinite(delta)
        if not bool(finite.any()):
            return
        delta = delta[finite]
        expected = float(self.expected_shift_px)
        self.count += int(delta.numel())
        self.signed_total += float(delta.sum())
        self.absolute_total += float(delta.abs().sum())
        self.expected_error_total += float((delta - expected).abs().sum())
        self.direction_correct += int((delta * expected > 0.0).sum())
        tolerance = 0.5 * abs(expected)
        self.follows_half_shift += int(((delta - expected).abs() <= tolerance).sum())

    def summary(self) -> dict[str, float | int]:
        count = max(self.count, 1)
        expected = float(self.expected_shift_px)
        mean_signed = self.signed_total / count
        return {
            "count": self.count,
            "expected_shift_px": expected,
            "mean_signed_output_shift_px": mean_signed,
            "mean_abs_output_shift_px": self.absolute_total / count,
            "response_ratio": mean_signed / expected if expected != 0.0 else 0.0,
            "mean_abs_error_to_expected_shift_px": self.expected_error_total / count,
            "correct_direction_fraction": self.direction_correct / count,
            "within_half_expected_shift_fraction": self.follows_half_shift / count,
        }


@dataclass
class SwapResponseStats:
    count: int = 0
    source_distance_total: float = 0.0
    donor_distance_total: float = 0.0
    donor_closer: int = 0

    def update(
        self,
        swapped: torch.Tensor,
        source: torch.Tensor,
        donor: torch.Tensor,
    ) -> None:
        source_distance = (swapped.detach().float() - source.detach().float()).abs()
        donor_distance = (swapped.detach().float() - donor.detach().float()).abs()
        finite = torch.isfinite(source_distance) & torch.isfinite(donor_distance)
        if not bool(finite.any()):
            return
        source_distance = source_distance[finite]
        donor_distance = donor_distance[finite]
        self.count += int(source_distance.numel())
        self.source_distance_total += float(source_distance.sum())
        self.donor_distance_total += float(donor_distance.sum())
        self.donor_closer += int((donor_distance < source_distance).sum())

    def summary(self) -> dict[str, float | int]:
        count = max(self.count, 1)
        source = self.source_distance_total / count
        donor = self.donor_distance_total / count
        return {
            "count": self.count,
            "mean_distance_to_source_px": source,
            "mean_distance_to_donor_px": donor,
            "donor_closer_fraction": self.donor_closer / count,
            "normalized_donor_preference": (
                (source - donor) / max(source + donor, 1e-6)
            ),
        }


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def _head_stages(
    head: torch.nn.Module,
    features: torch.Tensor,
    amp_dtype: torch.dtype | None,
) -> dict[str, dict[str, torch.Tensor]]:
    previous = bool(head.intermediate_supervision)
    head.intermediate_supervision = True
    try:
        with _amp_context(features.device, amp_dtype):
            outputs = head(features, inference_only=False)
    finally:
        head.intermediate_supervision = previous
    auxiliary = outputs.get("aux_outputs")
    if not isinstance(auxiliary, (list, tuple)):
        raise RuntimeError("Intermediate decoder outputs were not produced")
    stages = [*auxiliary, outputs]
    return {
        f"L{index}": {
            "pred_x_rows": stage["pred_x_rows"].detach().float(),
            "exist_logits": stage["exist_logits"].detach().float(),
            "exist_probability": torch.softmax(
                stage["exist_logits"].detach().float(),
                dim=-1,
            )[..., 0],
            "range_norm": stage["range_norm"].detach().float(),
        }
        for index, stage in enumerate(stages, start=1)
    }


def _best_lane_ious(
    candidates: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    line_width: float,
) -> list[float]:
    gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
    valid = target["valid_mask"].to(device=candidates.device).bool()
    values: list[float] = []
    for lane_index in range(int(gt_x.shape[0])):
        if int(valid[lane_index].sum()) < 5:
            continue
        ious = line_iou_against_gt(
            candidates,
            gt_x[lane_index],
            valid[lane_index],
            line_width=float(line_width),
        )
        values.append(float(ious.max()) if ious.numel() else 0.0)
    return values


def _update_recall(
    stats: ProposalRecallStats,
    stage: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    group_size: int,
    line_width: float,
) -> None:
    pred_x = stage["pred_x_rows"]
    for batch_index, target in enumerate(targets):
        candidates = pred_x[batch_index, :group_size]
        for best_iou in _best_lane_ious(
            candidates,
            target,
            line_width=float(line_width),
        ):
            stats.update(best_iou)


def _group_zero_assignments(
    match: dict[str, torch.Tensor],
    *,
    group_size: int,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for pred_index, gt_index in zip(
        match["pred_indices"].tolist(),
        match["gt_indices"].tolist(),
    ):
        if int(pred_index) < int(group_size):
            result[int(gt_index)] = int(pred_index)
    return result


def _update_matched_shift_response(
    stats: ShiftResponseStats,
    baseline_x: torch.Tensor,
    changed_x: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    *,
    group_size: int,
) -> None:
    for batch_index, (target, match) in enumerate(zip(targets, matches)):
        assignments = _group_zero_assignments(match, group_size=group_size)
        valid_all = target["valid_mask"].to(device=baseline_x.device).bool()
        for gt_index, pred_index in assignments.items():
            valid = valid_all[int(gt_index)]
            if int(valid.sum()) < 5:
                continue
            stats.update(
                changed_x[batch_index, pred_index, valid]
                - baseline_x[batch_index, pred_index, valid]
            )


def _summary_decision(
    payload: dict[str, Any],
) -> dict[str, Any]:
    final_layer = sorted(
        payload["stage_metrics"],
        key=lambda name: int(name[1:]),
    )[-1]
    final = payload["stage_metrics"][final_layer]
    ratios = [
        float(final[name]["matched_shift_response"]["response_ratio"])
        for name in ("shift_left", "shift_right")
    ]
    mean_ratio = sum(ratios) / len(ratios)
    baseline_recall = float(
        final["baseline"]["recall_original_target"]["recall@0.5"]
    )
    zero_recall = float(final["zero"]["recall_original_target"]["recall@0.5"])
    row_mean_recall = float(
        final["row_mean"]["recall_original_target"]["recall@0.5"]
    )
    return {
        "final_layer": final_layer,
        "mean_bidirectional_shift_response_ratio": mean_ratio,
        "zero_feature_recall_retention_fraction": (
            zero_recall / max(baseline_recall, 1e-6)
        ),
        "row_mean_recall_retention_fraction": (
            row_mean_recall / max(baseline_recall, 1e-6)
        ),
        "interpretation_rule": (
            "A low shift-response ratio together with high zero/x-mean recall "
            "retention supports query-prior domination. A substantial shift "
            "response and low ablated-feature retention supports genuine "
            "image grounding. Mixed values indicate that visual evidence is "
            "used but does not fully control row coordinates."
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if int(args.eval_batch_size) < 2:
        raise ValueError("--eval-batch-size must be at least two for batch swap")
    if int(args.shift_cols) <= 0:
        raise ValueError("--shift-cols must be positive")

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False
    model_cfg.setdefault("structured_query", {})["intermediate_supervision"] = True

    device = torch.device(args.device)
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    channels_last = (
        bool(cfg.get("training", {}).get("channels_last", False))
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    head = model.structured_query_head
    if head is None:
        raise ValueError("Image-grounding audit requires structured_query")
    matcher = build_matcher(cfg)
    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )

    num_instances = int(head.num_instances)
    num_groups = int(head.num_groups)
    group_size = num_instances // max(num_groups, 1)
    input_w = int(model_cfg.get("input_w", head.input_w))
    thresholds = tuple(float(value) for value in args.iou_thresholds)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)

    condition_names = (
        "baseline",
        "zero",
        "row_mean",
        "shift_left",
        "shift_right",
        "batch_swap",
    )
    recalls_original: dict[str, dict[str, ProposalRecallStats]] = {}
    recalls_shifted: dict[str, dict[str, ProposalRecallStats]] = {}
    recalls_donor: dict[str, dict[str, ProposalRecallStats]] = {}
    query_changes: dict[str, dict[str, MeanStats]] = {}
    exist_changes: dict[str, dict[str, MeanStats]] = {}
    shift_responses: dict[str, dict[str, ShiftResponseStats]] = {}
    swap_responses: dict[str, SwapResponseStats] = {}
    images_seen = 0
    feature_shape: tuple[int, ...] | None = None
    shift_px = 0.0

    total = len(loader)
    if int(args.max_batches) > 0:
        total = min(total, int(args.max_batches))
    progress = tqdm(
        enumerate(loader),
        total=total,
        desc="decoder image-grounding audit",
        ncols=100,
    )
    for batch_index, (images, targets, _metas) in progress:
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last
                if channels_last
                else torch.contiguous_format
            ),
        )
        targets = nested_to_device(targets, device)
        with _amp_context(device, amp_dtype):
            p2 = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )["features"]
        feature_shape = tuple(int(value) for value in p2.shape)
        shift_px = float(args.shift_cols) * float(input_w) / float(p2.shape[-1])
        conditions = {
            "baseline": p2,
            "zero": torch.zeros_like(p2),
            "row_mean": p2.mean(dim=-1, keepdim=True).expand_as(p2),
            "shift_left": shift_feature_x(p2, -int(args.shift_cols)),
            "shift_right": shift_feature_x(p2, int(args.shift_cols)),
            "batch_swap": torch.roll(p2, shifts=1, dims=0),
        }
        stage_outputs = {
            name: _head_stages(head, value, amp_dtype)
            for name, value in conditions.items()
        }
        del conditions
        baseline_stages = stage_outputs["baseline"]
        final_name = sorted(
            baseline_stages,
            key=lambda name: int(name[1:]),
        )[-1]
        final_for_matcher = baseline_stages[final_name]
        # Matcher geometry and existence are needed only to choose stable
        # group-zero lane/query identities for the response measurement.
        matches = matcher(final_for_matcher, targets)
        shifted_targets = {
            "shift_left": shift_targets_x(
                targets,
                -shift_px,
                input_w=input_w,
            ),
            "shift_right": shift_targets_x(
                targets,
                shift_px,
                input_w=input_w,
            ),
        }
        donor_indices = torch.roll(
            torch.arange(images.shape[0], device=device),
            shifts=1,
            dims=0,
        )
        donor_targets = [targets[int(index)] for index in donor_indices.tolist()]

        for layer_name, baseline_stage in baseline_stages.items():
            recalls_original.setdefault(layer_name, {})
            recalls_shifted.setdefault(layer_name, {})
            recalls_donor.setdefault(layer_name, {})
            query_changes.setdefault(layer_name, {})
            exist_changes.setdefault(layer_name, {})
            shift_responses.setdefault(layer_name, {})
            swap_responses.setdefault(layer_name, SwapResponseStats())
            for condition_name in condition_names:
                condition_stage = stage_outputs[condition_name][layer_name]
                recall = recalls_original[layer_name].setdefault(
                    condition_name,
                    ProposalRecallStats(thresholds=thresholds),
                )
                _update_recall(
                    recall,
                    condition_stage,
                    targets,
                    group_size=group_size,
                    line_width=float(args.line_width),
                )
                query_delta = query_changes[layer_name].setdefault(
                    condition_name,
                    MeanStats(),
                )
                query_delta.update(
                    condition_stage["pred_x_rows"][:, :group_size]
                    - baseline_stage["pred_x_rows"][:, :group_size]
                )
                exist_delta = exist_changes[layer_name].setdefault(
                    condition_name,
                    MeanStats(),
                )
                exist_delta.update(
                    condition_stage["exist_probability"][:, :group_size]
                    - baseline_stage["exist_probability"][:, :group_size]
                )

            for condition_name, expected_shift in (
                ("shift_left", -shift_px),
                ("shift_right", shift_px),
            ):
                condition_stage = stage_outputs[condition_name][layer_name]
                shifted_recall = recalls_shifted[layer_name].setdefault(
                    condition_name,
                    ProposalRecallStats(thresholds=thresholds),
                )
                _update_recall(
                    shifted_recall,
                    condition_stage,
                    shifted_targets[condition_name],
                    group_size=group_size,
                    line_width=float(args.line_width),
                )
                response = shift_responses[layer_name].setdefault(
                    condition_name,
                    ShiftResponseStats(expected_shift_px=expected_shift),
                )
                _update_matched_shift_response(
                    response,
                    baseline_stage["pred_x_rows"],
                    condition_stage["pred_x_rows"],
                    targets,
                    matches,
                    group_size=group_size,
                )

            swapped_stage = stage_outputs["batch_swap"][layer_name]
            donor_recall = recalls_donor[layer_name].setdefault(
                "batch_swap",
                ProposalRecallStats(thresholds=thresholds),
            )
            _update_recall(
                donor_recall,
                swapped_stage,
                donor_targets,
                group_size=group_size,
                line_width=float(args.line_width),
            )
            swap_responses[layer_name].update(
                swapped_stage["pred_x_rows"][:, :group_size],
                baseline_stage["pred_x_rows"][:, :group_size],
                baseline_stage["pred_x_rows"][donor_indices, :group_size],
            )
        images_seen += int(images.shape[0])

    stage_metrics: dict[str, Any] = {}
    for layer_name in sorted(recalls_original, key=lambda name: int(name[1:])):
        stage_metrics[layer_name] = {}
        for condition_name in condition_names:
            condition_payload: dict[str, Any] = {
                "recall_original_target": recalls_original[layer_name][
                    condition_name
                ].summary(),
                "querywise_output_change_px": query_changes[layer_name][
                    condition_name
                ].summary(),
                "existence_probability_change": exist_changes[layer_name][
                    condition_name
                ].summary(),
            }
            if condition_name in shift_responses[layer_name]:
                condition_payload["matched_shift_response"] = shift_responses[
                    layer_name
                ][condition_name].summary()
                condition_payload["recall_shifted_target"] = recalls_shifted[
                    layer_name
                ][condition_name].summary()
            if condition_name == "batch_swap":
                condition_payload["recall_donor_target"] = recalls_donor[
                    layer_name
                ][condition_name].summary()
                condition_payload["swap_response"] = swap_responses[
                    layer_name
                ].summary()
            stage_metrics[layer_name][condition_name] = condition_payload

    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "warning": (
            "P2 interventions are causal decoder diagnostics, not natural "
            "images or benchmark results."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "feature_shape_last_batch": feature_shape,
        "num_instances": num_instances,
        "num_groups": num_groups,
        "group_size": group_size,
        "shift_columns": int(args.shift_cols),
        "shift_pixels": shift_px,
        "stage_metrics": stage_metrics,
    }
    payload["decision_summary"] = _summary_decision(payload)
    print(json.dumps(payload, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
