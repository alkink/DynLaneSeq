from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import (
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.losses.matcher_s0 import (
    HungarianMatcherS0,
    MatcherConfig,
)
from dynlaneseq_eg.modeling.common import nested_to_device, sort_range_norm
from dynlaneseq_eg.tools.analyze_selection_assignment_gradient_conflict import (
    _amp_context,
    _prepare_config,
    _selection_assignment_map,
    _stage_for_image,
    _target_to_official_mapping,
)
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    official_proposal_gt_iou_matrix,
    proposal_gt_iou_matrix,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the S0 Hungarian cost at a frozen checkpoint. Compare "
            "one-factor matcher counterfactuals in official CULane raster-IoU "
            "space without training or changing predictions."
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
        median = (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        return {
            "count": len(ordered),
            "mean": sum(ordered) / len(ordered),
            "median": median,
            "min": ordered[0],
            "max": ordered[-1],
        }


@dataclass
class VariantStats:
    assigned: int = 0
    official_iou: ScalarStats = field(default_factory=ScalarStats)
    exist_probability: ScalarStats = field(default_factory=ScalarStats)
    raw_costs: dict[str, ScalarStats] = field(
        default_factory=lambda: defaultdict(ScalarStats)
    )
    hits: dict[str, int] = field(
        default_factory=lambda: {"0.50": 0, "0.70": 0}
    )

    def summary(self) -> dict[str, Any]:
        return {
            "assigned_gt": self.assigned,
            "mean_official_iou": self.official_iou.summary(),
            "selected_exist_probability": (
                self.exist_probability.summary()
            ),
            "selected_raw_cost_components": {
                name: stats.summary()
                for name, stats in self.raw_costs.items()
            },
            "threshold_hits": dict(self.hits),
            "recall": {
                threshold: (
                    float(count) / float(self.assigned)
                    if self.assigned
                    else 0.0
                )
                for threshold, count in self.hits.items()
            },
        }


@dataclass
class PairStats:
    comparable: int = 0
    agreement: int = 0
    variant_wins: int = 0
    reference_wins: int = 0
    ties: int = 0
    iou_delta: ScalarStats = field(default_factory=ScalarStats)
    transitions: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "0.50": {
                "reference_miss_variant_hit": 0,
                "reference_hit_variant_miss": 0,
            },
            "0.70": {
                "reference_miss_variant_hit": 0,
                "reference_hit_variant_miss": 0,
            },
        }
    )

    def update(
        self,
        reference_candidate: int,
        variant_candidate: int,
        reference_iou: float,
        variant_iou: float,
        *,
        win_margin: float,
    ) -> None:
        self.comparable += 1
        self.agreement += int(reference_candidate == variant_candidate)
        delta = float(variant_iou) - float(reference_iou)
        self.iou_delta.update(delta)
        if delta > float(win_margin):
            self.variant_wins += 1
        elif delta < -float(win_margin):
            self.reference_wins += 1
        else:
            self.ties += 1
        for threshold in (0.5, 0.7):
            key = f"{threshold:.2f}"
            reference_hit = float(reference_iou) >= threshold
            variant_hit = float(variant_iou) >= threshold
            if not reference_hit and variant_hit:
                self.transitions[key][
                    "reference_miss_variant_hit"
                ] += 1
            elif reference_hit and not variant_hit:
                self.transitions[key][
                    "reference_hit_variant_miss"
                ] += 1

    def summary(self) -> dict[str, Any]:
        return {
            "comparable_gt": self.comparable,
            "assignment_agreement_rate": (
                float(self.agreement) / float(self.comparable)
                if self.comparable
                else 0.0
            ),
            "variant_wins": self.variant_wins,
            "reference_wins": self.reference_wins,
            "ties": self.ties,
            "official_iou_delta_variant_minus_reference": (
                self.iou_delta.summary()
            ),
            "threshold_transitions": self.transitions,
        }


def matcher_cost_components(
    matcher: HungarianMatcherS0,
    exist_logits: torch.Tensor,
    pred_x_rows: torch.Tensor,
    range_norm: torch.Tensor,
    target: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the exact unweighted matrices used by S0 matching."""

    cfg = matcher.cfg
    device = pred_x_rows.device
    gt_x = target["x_rows"].to(device)
    gt_mask = target["valid_mask"].to(device).bool()
    gt_range = target["range_y"].to(device)
    candidates = int(pred_x_rows.shape[0])
    gt_count = int(gt_x.shape[0])
    if gt_count == 0:
        empty = pred_x_rows.new_zeros((candidates, 0))
        return {
            "object": empty,
            "point": empty,
            "range": empty,
            "line_iou": empty,
        }

    p_lane = torch.softmax(exist_logits, dim=-1)[:, 0]
    object_cost_type = str(cfg.object_cost_type).strip().lower()
    if object_cost_type in {
        "neg_probability",
        "negative_probability",
        "minus_p",
    }:
        object_cost = -p_lane.view(candidates, 1).expand(
            candidates,
            gt_count,
        )
    elif object_cost_type in {
        "neg_log_probability",
        "negative_log_probability",
        "nll",
    }:
        object_cost = -torch.log(
            p_lane.clamp_min(cfg.eps)
        ).view(candidates, 1).expand(candidates, gt_count)
    else:
        raise ValueError(
            f"Unsupported matcher.object_cost_type: "
            f"{cfg.object_cost_type!r}"
        )

    difference = (
        pred_x_rows[:, None, :] - gt_x[None, :, :]
    ).abs() / float(cfg.input_w)
    mask = gt_mask[None, :, :].expand(candidates, gt_count, -1)
    valid_count = mask.sum(dim=-1).clamp_min(1)
    point_cost = (
        difference * mask.float()
    ).sum(dim=-1) / valid_count
    point_cost = torch.where(
        gt_mask.sum(dim=-1).view(1, gt_count) > 0,
        point_cost,
        torch.full_like(point_cost, 1e6),
    )

    predicted_range = sort_range_norm(range_norm)
    target_range = gt_range / float(cfg.input_h)
    range_cost = (
        predicted_range[:, None, 0]
        .sub(target_range[None, :, 0])
        .abs()
        + predicted_range[:, None, 1]
        .sub(target_range[None, :, 1])
        .abs()
    )
    line_iou_cost = matcher.compute_line_iou_cost(
        pred_x_rows,
        gt_x,
        gt_mask,
    )
    return {
        "object": object_cost,
        "point": point_cost,
        "range": range_cost,
        "line_iou": line_iou_cost,
    }


def weighted_matcher_cost(
    components: dict[str, torch.Tensor],
    cfg: MatcherConfig,
) -> torch.Tensor:
    return (
        float(cfg.lambda_obj) * components["object"]
        + float(cfg.lambda_point) * components["point"]
        + float(cfg.lambda_range) * components["range"]
        + float(cfg.lambda_line_iou) * components["line_iou"]
    )


def build_counterfactual_configs(
    configured: MatcherConfig,
) -> dict[str, MatcherConfig]:
    return {
        "configured_main": configured,
        "object_weight_1p0": replace(configured, lambda_obj=1.0),
        "object_weight_0p5": replace(configured, lambda_obj=0.5),
        "no_object": replace(configured, lambda_obj=0.0),
        "no_range": replace(configured, lambda_range=0.0),
        "no_line_iou": replace(
            configured,
            lambda_line_iou=0.0,
        ),
        "point_only": replace(
            configured,
            lambda_obj=0.0,
            lambda_range=0.0,
            lambda_line_iou=0.0,
        ),
    }


def assignment_from_cost(
    cost: torch.Tensor,
) -> dict[int, int]:
    if int(cost.shape[0]) == 0 or int(cost.shape[1]) == 0:
        return {}
    candidate_indices, gt_indices = (
        HungarianMatcherS0._linear_sum_assignment(cost)
    )
    return {
        int(gt_index): int(candidate_index)
        for candidate_index, gt_index in zip(
            candidate_indices.tolist(),
            gt_indices.tolist(),
        )
    }


def _official_iou_for_target(
    target_gt: int,
    candidate: int,
    official_iou: torch.Tensor,
    target_to_official: dict[int, int],
) -> float | None:
    official_gt = target_to_official.get(int(target_gt))
    if official_gt is None:
        return None
    if official_gt >= int(official_iou.shape[0]):
        return None
    return float(official_iou[official_gt, int(candidate)])


def _variant_rank_key(
    summary: dict[str, Any],
) -> tuple[int, int, float]:
    return (
        int(summary["threshold_hits"]["0.70"]),
        int(summary["threshold_hits"]["0.50"]),
        float(summary["mean_official_iou"]["mean"]),
    )


def _diagnosis(
    variants: dict[str, dict[str, Any]],
    versus_configured: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trainable_names = [
        name
        for name in variants
        if name != "configured_main"
    ]
    best_name = max(
        trainable_names,
        key=lambda name: _variant_rank_key(variants[name]),
    )
    best_pair = versus_configured[best_name]
    net_050 = (
        best_pair["threshold_transitions"]["0.50"][
            "reference_miss_variant_hit"
        ]
        - best_pair["threshold_transitions"]["0.50"][
            "reference_hit_variant_miss"
        ]
    )
    net_070 = (
        best_pair["threshold_transitions"]["0.70"][
            "reference_miss_variant_hit"
        ]
        - best_pair["threshold_transitions"]["0.70"][
            "reference_hit_variant_miss"
        ]
    )
    mean_delta = float(
        best_pair["official_iou_delta_variant_minus_reference"]["mean"]
    )
    if net_050 <= 0 and net_070 <= 0 and mean_delta <= 0.005:
        diagnosis = "no_matcher_component_has_decisive_positive_signal"
        next_action = (
            "do not launch a matcher fine-tune from this audit; return to "
            "decoder-state or scoring diagnostics"
        )
    elif best_name in {
        "object_weight_1p0",
        "object_weight_0p5",
        "no_object",
    }:
        diagnosis = "object_confidence_cost_is_primary_lock_in"
        next_action = (
            f"run one matched 65k-to70k gate with matcher variant "
            f"{best_name}; use that one assignment for existence, geometry, "
            "DFL, and quality"
        )
    elif best_name == "no_range":
        diagnosis = "separate_range_cost_is_primary_mismatch"
        next_action = (
            "run one matched 5k gate without the separate range term while "
            "retaining range-aware geometry supervision"
        )
    elif best_name == "no_line_iou":
        diagnosis = "line_iou_match_cost_is_primary_mismatch"
        next_action = (
            "run one matched 5k gate without LineIoU in matching; keep the "
            "LineIoU training loss"
        )
    elif best_name == "point_only":
        diagnosis = "compound_match_cost_is_misaligned"
        next_action = (
            "replace the compound assignment cost with the simplest "
            "geometry-only assignment for one matched 5k gate"
        )
    else:
        diagnosis = "range_aware_raster_assignment_is_required"
        next_action = (
            "implement the range-aware raster-IoU surrogate as the shared "
            "training matcher and run one matched 5k gate"
        )
    return {
        "best_counterfactual": best_name,
        "net_hit_gain_050": net_050,
        "net_hit_gain_070": net_070,
        "mean_official_iou_gain": mean_delta,
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
    head.intermediate_supervision = False
    configured_matcher = build_matcher(cfg)
    if str(configured_matcher.cfg.assignment) != "hungarian":
        raise ValueError(
            "This diagnostic requires one-to-one Hungarian assignment; got "
            f"{configured_matcher.cfg.assignment!r}"
        )
    variant_configs = build_counterfactual_configs(
        configured_matcher.cfg
    )
    variant_names = [
        *variant_configs,
        "range_aware_raster",
    ]
    variant_stats = {
        name: VariantStats() for name in variant_names
    }
    versus_configured = {
        name: PairStats()
        for name in variant_names
        if name != "configured_main"
    }
    versus_range_aware = {
        name: PairStats()
        for name in variant_names
        if name != "range_aware_raster"
    }
    range_vs_main_disagreement = defaultdict(ScalarStats)
    mapping_quality = ScalarStats()

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
    images_seen = 0

    for batch_index, (images, targets_cpu, metas) in enumerate(
        tqdm(
            loader,
            total=total_batches,
            desc="matcher-cost counterfactual",
            ncols=100,
        )
    ):
        if int(args.max_batches) > 0 and batch_index >= int(
            args.max_batches
        ):
            break
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
            outputs = head(
                encoded["features"],
                inference_only=True,
            )

        for image_index, (target_cpu, target, meta) in enumerate(
            zip(targets_cpu, targets, metas)
        ):
            components = matcher_cost_components(
                configured_matcher,
                outputs["exist_logits"][image_index],
                outputs["pred_x_rows"][image_index],
                outputs["range_norm"][image_index],
                target,
            )
            assignments = {
                name: assignment_from_cost(
                    weighted_matcher_cost(components, variant_cfg)
                )
                for name, variant_cfg in variant_configs.items()
            }
            stage = _stage_for_image(outputs, image_index)
            surrogate_iou, valid_gt, _candidate_valid = (
                proposal_gt_iou_matrix(
                    stage,
                    target_cpu,
                    input_h=int(meta.get("input_h", 288)),
                    input_w=int(meta.get("input_w", 800)),
                    line_width=float(args.line_width),
                    min_valid_rows=int(args.min_valid_rows),
                )
            )
            assignments["range_aware_raster"] = (
                _selection_assignment_map(
                    surrogate_iou,
                    valid_gt,
                )
            )
            valid_target_ids = torch.where(valid_gt.bool())[0].tolist()
            target_to_surrogate_row = {
                int(target_id): row
                for row, target_id in enumerate(valid_target_ids)
            }
            target_to_official, image_mapping_quality = (
                _target_to_official_mapping(
                    target_cpu,
                    meta,
                    line_width=float(args.line_width),
                )
            )
            for value in image_mapping_quality:
                mapping_quality.update(value)
            official_iou, _official_valid = (
                official_proposal_gt_iou_matrix(
                    {"stages": {"main": stage}, "meta": meta},
                    "main",
                    line_width=float(args.line_width),
                    min_valid_rows=int(args.min_valid_rows),
                )
            )
            exist_probability = torch.softmax(
                outputs["exist_logits"][image_index].detach().float(),
                dim=-1,
            )[:, 0].cpu()
            components_cpu = {
                name: matrix.detach().float().cpu()
                for name, matrix in components.items()
            }

            for variant_name, assignment in assignments.items():
                stats = variant_stats[variant_name]
                variant_cfg = variant_configs.get(
                    variant_name,
                    configured_matcher.cfg,
                )
                for target_gt, candidate in assignment.items():
                    official_value = _official_iou_for_target(
                        target_gt,
                        candidate,
                        official_iou,
                        target_to_official,
                    )
                    if official_value is None:
                        continue
                    stats.assigned += 1
                    stats.official_iou.update(official_value)
                    stats.exist_probability.update(
                        float(exist_probability[candidate])
                    )
                    for component_name, matrix in (
                        components_cpu.items()
                    ):
                        stats.raw_costs[component_name].update(
                            float(matrix[candidate, target_gt])
                        )
                    if variant_name == "range_aware_raster":
                        surrogate_row = target_to_surrogate_row[
                            int(target_gt)
                        ]
                        assignment_cost = 1.0 - float(
                            surrogate_iou[surrogate_row, candidate]
                        )
                    else:
                        weighted = weighted_matcher_cost(
                            components,
                            variant_cfg,
                        ).detach().float().cpu()
                        assignment_cost = float(
                            weighted[candidate, target_gt]
                        )
                    stats.raw_costs["variant_total"].update(
                        assignment_cost
                    )
                    for threshold in (0.5, 0.7):
                        stats.hits[f"{threshold:.2f}"] += int(
                            official_value >= threshold
                        )

            configured_assignment = assignments["configured_main"]
            range_assignment = assignments["range_aware_raster"]
            for variant_name, assignment in assignments.items():
                if variant_name != "configured_main":
                    pair_stats = versus_configured[variant_name]
                    for target_gt in sorted(
                        set(configured_assignment) & set(assignment)
                    ):
                        reference_candidate = configured_assignment[
                            target_gt
                        ]
                        variant_candidate = assignment[target_gt]
                        reference_iou = _official_iou_for_target(
                            target_gt,
                            reference_candidate,
                            official_iou,
                            target_to_official,
                        )
                        variant_iou = _official_iou_for_target(
                            target_gt,
                            variant_candidate,
                            official_iou,
                            target_to_official,
                        )
                        if reference_iou is None or variant_iou is None:
                            continue
                        pair_stats.update(
                            reference_candidate,
                            variant_candidate,
                            reference_iou,
                            variant_iou,
                            win_margin=float(
                                args.official_win_margin
                            ),
                        )
                if variant_name != "range_aware_raster":
                    pair_stats = versus_range_aware[variant_name]
                    for target_gt in sorted(
                        set(range_assignment) & set(assignment)
                    ):
                        reference_candidate = range_assignment[
                            target_gt
                        ]
                        variant_candidate = assignment[target_gt]
                        reference_iou = _official_iou_for_target(
                            target_gt,
                            reference_candidate,
                            official_iou,
                            target_to_official,
                        )
                        variant_iou = _official_iou_for_target(
                            target_gt,
                            variant_candidate,
                            official_iou,
                            target_to_official,
                        )
                        if reference_iou is None or variant_iou is None:
                            continue
                        pair_stats.update(
                            reference_candidate,
                            variant_candidate,
                            reference_iou,
                            variant_iou,
                            win_margin=float(
                                args.official_win_margin
                            ),
                        )

            for target_gt in sorted(
                set(configured_assignment) & set(range_assignment)
            ):
                configured_candidate = configured_assignment[target_gt]
                range_candidate = range_assignment[target_gt]
                if configured_candidate == range_candidate:
                    continue
                for component_name, matrix in components_cpu.items():
                    delta = float(
                        matrix[range_candidate, target_gt]
                        - matrix[configured_candidate, target_gt]
                    )
                    range_vs_main_disagreement[
                        f"raw_{component_name}_range_minus_main"
                    ].update(delta)
                    configured_weight = {
                        "object": configured_matcher.cfg.lambda_obj,
                        "point": configured_matcher.cfg.lambda_point,
                        "range": configured_matcher.cfg.lambda_range,
                        "line_iou": (
                            configured_matcher.cfg.lambda_line_iou
                        ),
                    }[component_name]
                    range_vs_main_disagreement[
                        f"weighted_{component_name}_range_minus_main"
                    ].update(float(configured_weight) * delta)
                range_vs_main_disagreement[
                    "exist_probability_range_minus_main"
                ].update(
                    float(
                        exist_probability[range_candidate]
                        - exist_probability[configured_candidate]
                    )
                )

        images_seen += int(images.shape[0])
        del outputs, encoded, images, targets

    variant_summaries = {
        name: stats.summary()
        for name, stats in variant_stats.items()
    }
    configured_comparisons = {
        name: stats.summary()
        for name, stats in versus_configured.items()
    }
    range_comparisons = {
        name: stats.summary()
        for name, stats in versus_range_aware.items()
    }
    payload = {
        "diagnostic_only": True,
        "warning": (
            "All variants reuse identical frozen predictions and differ only "
            "in the training-time Hungarian cost. These are assignment "
            "counterfactuals, not independently trained benchmark models."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "settings": {
            "configured_matcher": vars(configured_matcher.cfg),
            "counterfactual_weights": {
                name: {
                    "lambda_obj": value.lambda_obj,
                    "lambda_point": value.lambda_point,
                    "lambda_range": value.lambda_range,
                    "lambda_line_iou": value.lambda_line_iou,
                }
                for name, value in variant_configs.items()
            },
            "line_width": float(args.line_width),
            "official_win_margin": float(args.official_win_margin),
            "amp_dtype": args.amp_dtype,
        },
        "target_to_official_mapping_iou": mapping_quality.summary(),
        "variants": variant_summaries,
        "versus_configured_main": configured_comparisons,
        "versus_range_aware_raster": range_comparisons,
        "range_aware_vs_configured_disagreement_cost_deltas": {
            name: stats.summary()
            for name, stats in range_vs_main_disagreement.items()
        },
    }
    payload["verdict"] = _diagnosis(
        variant_summaries,
        configured_comparisons,
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
