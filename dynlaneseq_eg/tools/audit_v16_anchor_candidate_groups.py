from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    ensure_official_iou_cache,
    load_or_collect_cache,
    metadata_for_json,
    write_json,
)
from dynlaneseq_eg.modeling.common import fixed_row_fractions
from dynlaneseq_eg.tools.audit_geometry_proposal_clustering import (
    PairGeometry,
    _pairwise_geometry,
)
from dynlaneseq_eg.tools.audit_v8_route_neighborhood_oracle import (
    _max_binary_matching_hits,
    _official_hungarian_hits,
    _slot_neighborhood_oracle_hits,
)


REFERENCE_INPUT_WIDTH = 1600.0


@dataclass(frozen=True)
class AnchorGroupPolicy:
    """GT-free variable-size candidate grouping around deployed V7 anchors.

    Every eligible proposal is assigned to its nearest active V7 anchor.  The
    primary policy then applies a corridor derived from the nearest other
    anchor, rather than padding every slot to a fixed K.  The diagnostic upper
    policy keeps the entire overlap-valid Voronoi cell.
    """

    name: str
    corridor_fraction: float | None
    min_corridor_px: float
    max_corridor_px: float
    min_common_rows: int
    min_overlap_fraction: float


POLICIES: tuple[AnchorGroupPolicy, ...] = (
    AnchorGroupPolicy(
        name="adaptive_voronoi_060",
        corridor_fraction=0.60,
        min_corridor_px=72.0,
        max_corridor_px=256.0,
        min_common_rows=5,
        min_overlap_fraction=0.25,
    ),
    AnchorGroupPolicy(
        name="overlap_voronoi_upper",
        corridor_fraction=None,
        min_corridor_px=0.0,
        max_corridor_px=math.inf,
        min_common_rows=5,
        min_overlap_fraction=0.25,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free V16 preflight: partition valid proposals into "
            "variable-size, bottom-aware Voronoi groups around the deployed "
            "V7 routes and measure whether each group retains the recoverable "
            "proposal-routing oracle. No averaging or learned model is used."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--sample-strategy", choices=("uniform", "sequential"), default="uniform")
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--official-iou-workers", type=int, default=12)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=(0.50, 0.75))
    parser.add_argument("--near-equivalent-iou-delta", type=float, default=0.02)
    parser.add_argument("--target-support-delta", type=float, default=0.10)
    parser.add_argument("--representable-iou", type=float, default=0.50)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    raw = [float(value) for value in values]
    finite = [value for value in raw if math.isfinite(value)]
    if not finite:
        return {
            "count": len(raw),
            "finite_count": 0,
            "mean": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    tensor = torch.tensor(finite, dtype=torch.float64)
    return {
        "count": len(raw),
        "finite_count": len(finite),
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "median": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "max": float(tensor.max()),
    }


def _curve_distance(
    geometry: PairGeometry | None,
    policy: AnchorGroupPolicy,
) -> float:
    """Bottom-heavy coherent-curve distance in input pixels.

    This scalar is used only for candidate ownership and corridor membership.
    It never averages proposal coordinates and never chooses the emitted lane.
    """

    if geometry is None or not geometry.valid:
        return math.inf
    if int(geometry.common_rows) < int(policy.min_common_rows):
        return math.inf
    if float(geometry.overlap_fraction_min) < float(policy.min_overlap_fraction):
        return math.inf
    lower = (
        float(geometry.lower_median_px)
        if geometry.lower_available and math.isfinite(geometry.lower_median_px)
        else float(geometry.weighted_median_px)
    )
    bottom = (
        float(geometry.bottom_endpoint_px)
        if math.isfinite(geometry.bottom_endpoint_px)
        else float(geometry.weighted_q90_px)
    )
    return (
        0.40 * float(geometry.weighted_median_px)
        + 0.20 * float(geometry.weighted_q90_px)
        + 0.25 * lower
        + 0.15 * bottom
    )


def build_anchor_groups(
    *,
    anchors: torch.Tensor,
    candidate_valid: torch.Tensor,
    pair_geometry: list[list[PairGeometry | None]],
    input_w: int,
    policy: AnchorGroupPolicy,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Return disjoint variable-size proposal groups in active-slot order."""

    anchor_ids = [int(value) for value in anchors.tolist()]
    if len(anchor_ids) != len(set(anchor_ids)):
        raise ValueError("V16 requires unique deployed V7 anchors")
    slots = len(anchor_ids)
    if slots == 0:
        return [], {
            "corridor_px": [],
            "excluded_no_overlap": int(candidate_valid.sum()),
            "excluded_outside_corridor": 0,
            "assigned_candidates": 0,
        }

    distance = torch.full(
        (slots, int(candidate_valid.numel())),
        torch.inf,
        dtype=torch.float64,
    )
    for slot, anchor in enumerate(anchor_ids):
        for candidate in torch.nonzero(candidate_valid.bool(), as_tuple=False).flatten().tolist():
            if int(candidate) == anchor:
                distance[slot, int(candidate)] = 0.0
            else:
                distance[slot, int(candidate)] = _curve_distance(
                    pair_geometry[anchor][int(candidate)],
                    policy,
                )

    scale = float(input_w) / REFERENCE_INPUT_WIDTH
    corridor: list[float] = []
    for slot, anchor in enumerate(anchor_ids):
        other_distances = [
            float(distance[slot, other_anchor])
            for other_slot, other_anchor in enumerate(anchor_ids)
            if other_slot != slot and math.isfinite(float(distance[slot, other_anchor]))
        ]
        if policy.corridor_fraction is None:
            radius = math.inf
        elif other_distances:
            radius = float(policy.corridor_fraction) * min(other_distances)
            radius = max(float(policy.min_corridor_px) * scale, radius)
            radius = min(float(policy.max_corridor_px) * scale, radius)
        else:
            radius = float(policy.max_corridor_px) * scale
        corridor.append(radius)

    groups: list[list[int]] = [[] for _ in range(slots)]
    excluded_no_overlap = 0
    excluded_outside_corridor = 0
    assignments: list[int] = [-1 for _ in range(int(candidate_valid.numel()))]
    assigned_distance: list[float] = [math.inf for _ in assignments]
    for candidate in torch.nonzero(candidate_valid.bool(), as_tuple=False).flatten().tolist():
        candidate = int(candidate)
        if candidate in anchor_ids:
            owner = anchor_ids.index(candidate)
            best_distance = 0.0
        else:
            values = distance[:, candidate]
            best_distance, owner_tensor = values.min(dim=0)
            best_distance = float(best_distance)
            owner = int(owner_tensor)
            if not math.isfinite(best_distance):
                excluded_no_overlap += 1
                continue
        if best_distance > float(corridor[owner]):
            excluded_outside_corridor += 1
            continue
        groups[owner].append(candidate)
        assignments[candidate] = owner
        assigned_distance[candidate] = best_distance

    # The deployed anchor is never removed, even if a malformed cache marked
    # it invalid. This is a safety invariant, not fixed-K padding.
    for slot, anchor in enumerate(anchor_ids):
        if anchor not in groups[slot]:
            groups[slot].append(anchor)
            assignments[anchor] = slot
            assigned_distance[anchor] = 0.0
        groups[slot].sort(key=lambda candidate: (assigned_distance[candidate], candidate))

    flat = [candidate for group in groups for candidate in group]
    if len(flat) != len(set(flat)):
        raise RuntimeError("V16 anchor groups must be disjoint")
    return [torch.tensor(group, dtype=torch.long) for group in groups], {
        "corridor_px": corridor,
        "excluded_no_overlap": excluded_no_overlap,
        "excluded_outside_corridor": excluded_outside_corridor,
        "assigned_candidates": len(flat),
        "candidate_owner": assignments,
        "candidate_anchor_distance_px": assigned_distance,
    }


def _fixed_v7_slot_gt_assignment(
    slot_quality: torch.Tensor,
    active_slots: torch.Tensor,
) -> dict[int, int]:
    if int(slot_quality.shape[0]) == 0 or int(active_slots.numel()) == 0:
        return {}
    local = slot_quality[:, active_slots].float()
    gt_ids, local_slot_ids = linear_sum_assignment(1.0 - local.cpu().numpy())
    return {
        int(local_slot): int(gt)
        for gt, local_slot in zip(gt_ids.tolist(), local_slot_ids.tolist())
    }


def _fixed_assignment_selection(
    proposal_quality: torch.Tensor,
    groups: list[torch.Tensor],
    anchors: torch.Tensor,
    slot_to_gt: dict[int, int],
) -> torch.Tensor:
    selected: list[int] = []
    for slot, group in enumerate(groups):
        if slot not in slot_to_gt or int(group.numel()) == 0:
            selected.append(int(anchors[slot]))
            continue
        gt = int(slot_to_gt[slot])
        local = proposal_quality[gt, group]
        selected.append(int(group[int(local.argmax())]))
    if len(selected) != len(set(selected)):
        raise RuntimeError("disjoint V16 groups produced duplicate oracle selections")
    return torch.tensor(selected, dtype=torch.long)


def _new_metric() -> dict[str, int]:
    return {"tp": 0, "predictions": 0, "gt": 0}


def _add_metric(row: dict[str, int], tp: int, predictions: int, gt: int) -> None:
    row["tp"] += int(tp)
    row["predictions"] += int(predictions)
    row["gt"] += int(gt)


def _finish_metric(row: dict[str, int]) -> dict[str, int | float]:
    tp = int(row["tp"])
    predictions = int(row["predictions"])
    gt = int(row["gt"])
    fp = predictions - tp
    fn = gt - tp
    denominator = predictions + gt
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predictions": predictions,
        "gt": gt,
        "precision": float(tp) / float(max(predictions, 1)),
        "recall": float(tp) / float(max(gt, 1)),
        "f1": 0.0 if denominator <= 0 else float(2 * tp) / float(denominator),
    }


def _resolve_stage(cache: dict[str, Any]) -> str:
    names = sorted(
        {
            name
            for record in cache.get("records", [])
            for name in record.get("stages", {})
        }
    )
    for fallback in ("main", "final", "stage2"):
        if fallback in names:
            return fallback
    raise ValueError(f"no supported V7 cache stage; choices={names}")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    max_batches = (
        0
        if int(args.max_images) <= 0
        else math.ceil(int(args.max_images) / int(args.eval_batch_size))
    )
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        list_path=args.list_path or None,
        dataset_root=args.dataset_root,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=bool(args.reuse_cache or args.require_cache),
        require_cache=bool(args.require_cache),
        max_batches=max_batches,
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        sample_strategy=args.sample_strategy,
        amp_dtype="none",
        desc=f"V16 candidate-group {args.split} cache",
    )
    cache = ensure_official_iou_cache(
        cache,
        line_width=float(args.line_width),
        min_valid_rows=int(args.min_valid_rows),
        row_visibility_thresh=0.0,
        workers=int(args.official_iou_workers),
    )
    if int(args.max_images) > 0:
        cache = dict(cache)
        cache["records"] = cache["records"][: int(args.max_images)]
    stage_name = _resolve_stage(cache)
    input_h = int(cache["metadata"]["input_h"])
    input_w = int(cache["metadata"]["input_w"])
    thresholds = tuple(float(value) for value in args.iou_thresholds)

    metric_rows: dict[str, dict[str, int]] = defaultdict(_new_metric)
    policy_stats: dict[str, dict[str, Any]] = {
        policy.name: {
            "active_slots": 0,
            "valid_candidates": 0,
            "assigned_candidates": 0,
            "excluded_no_overlap": 0,
            "excluded_outside_corridor": 0,
            "anchors_retained": 0,
            "duplicate_memberships": 0,
            "group_sizes": [],
            "corridor_px": [],
            "matched_representable_slots": 0,
            "target_argmax_in_group": 0,
            "target_support_any_in_group": 0,
            "near_equivalent_in_group": 0,
            "global_best_quality": [],
            "local_best_quality": [],
            "local_quality_deficit": [],
            "target_argmax_anchor_distance_px": [],
        }
        for policy in POLICIES
    }
    per_image: list[dict[str, Any]] = []

    for record in tqdm(cache["records"], ncols=90, desc=f"V16 preflight {args.split}"):
        stage = record["stages"][stage_name]
        proposal_quality = stage["official_iou"].float()
        candidate_valid = stage["official_candidate_valid"].bool().clone()
        x, masks, geometric_valid = candidate_row_masks(
            stage,
            input_h=input_h,
            input_w=input_w,
            min_valid_rows=int(args.min_valid_rows),
            row_visibility_thresh=0.0,
        )
        candidate_valid &= geometric_valid.bool()
        y_fraction = fixed_row_fractions(
            int(x.shape[-1]), device=x.device, dtype=x.dtype
        )
        pair_geometry = _pairwise_geometry(x, masks, y_fraction)

        indices = stage["selection_slot_indices"].long()
        active_mask = indices >= 0
        if isinstance(stage.get("selection_slot_active"), torch.Tensor):
            active_mask &= stage["selection_slot_active"].bool()
        if isinstance(stage.get("selection_slot_official_candidate_valid"), torch.Tensor):
            active_mask &= stage["selection_slot_official_candidate_valid"].bool()
        active_slots = torch.nonzero(active_mask, as_tuple=False).flatten()
        anchors = indices[active_slots]
        slot_quality = stage["selection_slot_official_iou"].float()
        gt_count = int(proposal_quality.shape[0])
        prediction_count = int(active_slots.numel())
        slot_to_gt = _fixed_v7_slot_gt_assignment(slot_quality, active_slots)

        for threshold in thresholds:
            current_reference_tp = _official_hungarian_hits(
                proposal_quality[:, anchors], threshold
            ) if prediction_count else 0
            current_refined_tp = _official_hungarian_hits(
                slot_quality[:, active_slots], threshold
            ) if prediction_count else 0
            all32_tp = _max_binary_matching_hits(
                proposal_quality[:, candidate_valid] >= threshold,
                max_predictions=prediction_count,
            )
            _add_metric(
                metric_rows[f"current_reference/{threshold:.2f}"],
                current_reference_tp,
                prediction_count,
                gt_count,
            )
            _add_metric(
                metric_rows[f"current_refined/{threshold:.2f}"],
                current_refined_tp,
                prediction_count,
                gt_count,
            )
            _add_metric(
                metric_rows[f"all32_same_count_oracle/{threshold:.2f}"],
                all32_tp,
                prediction_count,
                gt_count,
            )

        image_payload: dict[str, Any] = {
            "image_id": record["image_id"],
            "gt_count": gt_count,
            "active_slots": active_slots.tolist(),
            "anchors": anchors.tolist(),
            "slot_to_gt": slot_to_gt,
            "policies": {},
        }
        for policy in POLICIES:
            groups, diagnostics = build_anchor_groups(
                anchors=anchors,
                candidate_valid=candidate_valid,
                pair_geometry=pair_geometry,
                input_w=input_w,
                policy=policy,
            )
            stats = policy_stats[policy.name]
            stats["active_slots"] += prediction_count
            stats["valid_candidates"] += int(candidate_valid.sum())
            stats["assigned_candidates"] += int(diagnostics["assigned_candidates"])
            stats["excluded_no_overlap"] += int(diagnostics["excluded_no_overlap"])
            stats["excluded_outside_corridor"] += int(
                diagnostics["excluded_outside_corridor"]
            )
            stats["group_sizes"].extend(int(group.numel()) for group in groups)
            stats["corridor_px"].extend(float(value) for value in diagnostics["corridor_px"])
            flat = [int(value) for group in groups for value in group.tolist()]
            stats["anchors_retained"] += sum(
                int(anchor) in set(group.tolist())
                for anchor, group in zip(anchors.tolist(), groups)
            )
            stats["duplicate_memberships"] += len(flat) - len(set(flat))

            for local_slot, gt in slot_to_gt.items():
                if local_slot >= len(groups) or int(groups[local_slot].numel()) == 0:
                    continue
                valid_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten()
                if int(valid_ids.numel()) == 0:
                    continue
                qualities = proposal_quality[int(gt), valid_ids]
                global_offset = int(qualities.argmax())
                target_id = int(valid_ids[global_offset])
                global_best = float(qualities[global_offset])
                if global_best < float(args.representable_iou):
                    continue
                stats["matched_representable_slots"] += 1
                group = groups[local_slot]
                local_qualities = proposal_quality[int(gt), group]
                local_best = float(local_qualities.max())
                support_floor = global_best - float(args.target_support_delta)
                stats["target_argmax_in_group"] += int(bool((group == target_id).any()))
                stats["target_support_any_in_group"] += int(
                    bool((local_qualities >= support_floor).any())
                )
                stats["near_equivalent_in_group"] += int(
                    local_best >= global_best - float(args.near_equivalent_iou_delta)
                )
                stats["global_best_quality"].append(global_best)
                stats["local_best_quality"].append(local_best)
                stats["local_quality_deficit"].append(global_best - local_best)
                target_distance = diagnostics["candidate_anchor_distance_px"][target_id]
                stats["target_argmax_anchor_distance_px"].append(float(target_distance))

            fixed_selection = _fixed_assignment_selection(
                proposal_quality,
                groups,
                anchors,
                slot_to_gt,
            ) if prediction_count else torch.empty(0, dtype=torch.long)
            image_policy: dict[str, Any] = {
                "groups": [group.tolist() for group in groups],
                "group_sizes": [int(group.numel()) for group in groups],
                "corridor_px": diagnostics["corridor_px"],
                "excluded_no_overlap": diagnostics["excluded_no_overlap"],
                "excluded_outside_corridor": diagnostics[
                    "excluded_outside_corridor"
                ],
                "fixed_assignment_oracle_ids": fixed_selection.tolist(),
            }
            for threshold in thresholds:
                combinatorial_tp = _slot_neighborhood_oracle_hits(
                    proposal_quality,
                    groups,
                    threshold,
                )
                fixed_tp = _official_hungarian_hits(
                    proposal_quality[:, fixed_selection], threshold
                ) if int(fixed_selection.numel()) else 0
                _add_metric(
                    metric_rows[
                        f"{policy.name}/combinatorial_oracle/{threshold:.2f}"
                    ],
                    combinatorial_tp,
                    prediction_count,
                    gt_count,
                )
                _add_metric(
                    metric_rows[
                        f"{policy.name}/fixed_assignment_oracle/{threshold:.2f}"
                    ],
                    fixed_tp,
                    prediction_count,
                    gt_count,
                )
                image_policy[f"fixed_assignment_tp_{threshold:.2f}"] = fixed_tp
                image_policy[f"combinatorial_tp_{threshold:.2f}"] = combinatorial_tp
            image_payload["policies"][policy.name] = image_policy
        per_image.append(image_payload)

    metrics = {key: _finish_metric(row) for key, row in sorted(metric_rows.items())}
    for policy in POLICIES:
        for oracle_name in ("fixed_assignment_oracle", "combinatorial_oracle"):
            for threshold in thresholds:
                current = metrics[f"current_reference/{threshold:.2f}"]
                global_oracle = metrics[f"all32_same_count_oracle/{threshold:.2f}"]
                local = metrics[
                    f"{policy.name}/{oracle_name}/{threshold:.2f}"
                ]
                gap = max(int(global_oracle["tp"]) - int(current["tp"]), 0)
                gain = int(local["tp"]) - int(current["tp"])
                local["gain_tp_over_current_reference"] = gain
                local["global_representative_gap_tp"] = gap
                local["global_gap_closure"] = (
                    float(gain) / float(gap) if gap > 0 else 1.0
                )

    policy_output: dict[str, Any] = {}
    for policy in POLICIES:
        raw = policy_stats[policy.name]
        denominator = max(int(raw["matched_representable_slots"]), 1)
        policy_output[policy.name] = {
            "contract": {
                key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in asdict(policy).items()
            },
            "active_slots": int(raw["active_slots"]),
            "valid_candidates": int(raw["valid_candidates"]),
            "assigned_candidates": int(raw["assigned_candidates"]),
            "assigned_candidate_fraction": float(raw["assigned_candidates"])
            / float(max(int(raw["valid_candidates"]), 1)),
            "excluded_no_overlap": int(raw["excluded_no_overlap"]),
            "excluded_outside_corridor": int(raw["excluded_outside_corridor"]),
            "anchor_retention": float(raw["anchors_retained"])
            / float(max(int(raw["active_slots"]), 1)),
            "duplicate_memberships": int(raw["duplicate_memberships"]),
            "group_size": _summary(raw["group_sizes"]),
            "corridor_px": _summary(raw["corridor_px"]),
            "matched_representable_slots": int(raw["matched_representable_slots"]),
            "target_argmax_coverage": float(raw["target_argmax_in_group"])
            / float(denominator),
            "target_support_any_coverage": float(raw["target_support_any_in_group"])
            / float(denominator),
            "near_equivalent_coverage": float(raw["near_equivalent_in_group"])
            / float(denominator),
            "global_best_quality": _summary(raw["global_best_quality"]),
            "local_best_quality": _summary(raw["local_best_quality"]),
            "local_quality_deficit": _summary(raw["local_quality_deficit"]),
            "target_argmax_anchor_distance_px": _summary(
                raw["target_argmax_anchor_distance_px"]
            ),
        }

    return {
        "experiment": "V16 variable-size anchor candidate-group capacity preflight",
        "scope": {
            "training_performed": False,
            "gt_used_to_build_groups": False,
            "gt_used_for_posthoc_capacity_only": True,
            "proposal_coordinates_averaged": False,
            "fixed_k_or_padding_used": False,
            "test_set_used": bool(args.split == "test"),
            "checkpoint_selection_performed": False,
            "threshold_search_performed": False,
        },
        "primary_policy": "adaptive_voronoi_060",
        "metadata": metadata_for_json(
            cache,
            resolved_stage=stage_name,
            records_analyzed=len(cache["records"]),
            line_width=float(args.line_width),
            min_valid_rows=int(args.min_valid_rows),
            iou_thresholds=list(thresholds),
            near_equivalent_iou_delta=float(args.near_equivalent_iou_delta),
            target_support_delta=float(args.target_support_delta),
            representable_iou=float(args.representable_iou),
        ),
        "distance_contract": {
            "rows": "common visible proposal rows only",
            "perspective_weight": "0.10 + 0.90 * y^3",
            "scalar": (
                "0.40 weighted_median + 0.20 weighted_q90 + "
                "0.25 lower_median + 0.15 bottom_endpoint"
            ),
            "ownership": "nearest active V7 anchor (Voronoi)",
            "primary_corridor": (
                "clamp(0.60 * nearest-other-anchor distance, 72px, 256px) "
                "at 1600px input width"
            ),
            "emission": "none; this is a capacity preflight",
        },
        "policies": policy_output,
        "official_metrics": metrics,
        "per_image": per_image,
    }


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# V16 variable-size anchor candidate-group preflight",
        "",
        (
            "This is training-free. Geometry defines only which coherent proposal "
            "curves may compete; no proposal coordinates are averaged."
        ),
        "",
        "| Policy | Mean group | Assigned | Target argmax | Any support | Near-equivalent |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, policy in report["policies"].items():
        lines.append(
            f"| `{name}` | {policy['group_size']['mean']:.2f} | "
            f"{policy['assigned_candidate_fraction']:.4f} | "
            f"{policy['target_argmax_coverage']:.4f} | "
            f"{policy['target_support_any_coverage']:.4f} | "
            f"{policy['near_equivalent_coverage']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Policy / oracle | Gap closure @.50 | Gap closure @.75 | F1@.50 | F1@.75 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in report["policies"]:
        for oracle in ("fixed_assignment_oracle", "combinatorial_oracle"):
            row50 = report["official_metrics"][f"{policy}/{oracle}/0.50"]
            row75 = report["official_metrics"][f"{policy}/{oracle}/0.75"]
            lines.append(
                f"| `{policy}/{oracle}` | {row50['global_gap_closure']:.4f} | "
                f"{row75['global_gap_closure']:.4f} | {row50['f1']:.4f} | "
                f"{row75['f1']:.4f} |"
            )
    current50 = report["official_metrics"]["current_reference/0.50"]
    current75 = report["official_metrics"]["current_reference/0.75"]
    lines.extend(
        [
            "",
            f"Current reference F1@.50/.75: **{current50['f1']:.4f} / {current75['f1']:.4f}**.",
            "",
            "No V16 training is authorized by one domain alone.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise ValueError("test split is closed for V16")
    report = run_audit(args)
    write_json(args.output_json, report)
    write_markdown(args.output_md, report)
    print(
        json.dumps(
            {
                "output_json": str(Path(args.output_json).resolve()),
                "output_md": str(Path(args.output_md).resolve()),
                "records": report["metadata"]["records_analyzed"],
                "test_set_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
