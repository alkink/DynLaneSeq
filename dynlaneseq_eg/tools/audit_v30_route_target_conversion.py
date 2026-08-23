from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class LaneObservation:
    image_id: str
    gt_index: int
    support_hit: bool
    official_best_iou: float
    official_best_proposal: int
    target_supervised: bool
    target_support_has_good: bool
    target_top1_good: bool
    target_top1_is_official_best: bool
    target_top1_official_iou: float
    target_good_mass: float
    target_policy_hit: bool
    learned_target_slot_good: bool
    learned_target_slot_active: bool
    learned_target_slot_active_good: bool
    learned_target_slot_is_target_top1: bool
    learned_target_slot_official_iou: float
    learned_good_mass: float
    learned_any_slot_good: bool
    learned_any_active_slot_good: bool
    learned_official_match_hit: bool
    learned_official_match_iou: float


def _load_cache(path: Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or not isinstance(cache.get("records"), list):
        raise ValueError(f"invalid diagnostic cache: {path}")
    if not cache["records"]:
        raise ValueError(f"empty diagnostic cache: {path}")
    return cache


def _stage(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    stage = record.get("stages", {}).get("main")
    if not isinstance(stage, dict):
        raise ValueError("cache record has no main stage")
    required = (
        "official_iou",
        "official_candidate_valid",
        "pred_x_rows",
        "range_norm",
        "selection_slot_logits",
        "selection_slot_candidate_valid",
        "selection_slot_indices",
        "selection_slot_active",
    )
    missing = [name for name in required if not isinstance(stage.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"cache stage is missing tensors: {missing}")
    return stage


def build_training_route_targets(
    record: dict[str, Any],
    stage: dict[str, torch.Tensor],
    *,
    input_h: int,
    line_width: float,
    min_valid_rows: int,
    cluster_min: float,
    cluster_delta: float,
    temperature: float,
    num_slots: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct V7's detached all-GT route target from a diagnostic cache.

    Returns ``[G,N]`` target probabilities, ``[G,N]`` support, the
    range-aware row-strip quality ``[G,N]``, and a supervised-GT mask ``[G]``.
    This mirrors
    ``build_four_slot_cluster_targets(..., target_mode='all_gt')`` without
    requiring a model forward pass.
    """

    pred_x = stage["pred_x_rows"].detach().float()
    pred_range = stage["range_norm"].detach().float().sort(dim=-1).values
    gt_x = record["target"]["x_rows"].detach().float()
    gt_valid = (
        record["target"]["valid_mask"].detach().bool()
        & torch.isfinite(gt_x)
    )
    candidates, rows = pred_x.shape
    y_rows = torch.arange(rows, dtype=torch.float32) * (
        float(input_h) / float(rows)
    )
    pred_valid = (
        (y_rows.unsqueeze(0) >= pred_range[:, :1] * float(input_h))
        & (y_rows.unsqueeze(0) <= pred_range[:, 1:] * float(input_h))
        & torch.isfinite(pred_x)
    )
    candidate_valid = pred_valid.sum(dim=-1) >= int(min_valid_rows)
    gt_lane_valid = gt_valid.sum(dim=-1) >= int(min_valid_rows)
    gt_count = int(gt_x.shape[0])
    if gt_count == 0:
        empty = pred_x.new_zeros((0, candidates))
        return empty, empty.bool(), empty, torch.zeros(0, dtype=torch.bool)

    both = pred_valid[:, None, :] & gt_valid[None, :, :]
    either = pred_valid[:, None, :] | gt_valid[None, :, :]
    overlap = (
        float(line_width)
        - (pred_x[:, None, :] - gt_x[None, :, :]).abs()
    ).clamp_min(0.0)
    overlap = torch.where(both, overlap, torch.zeros_like(overlap))
    union = torch.where(
        both,
        2.0 * float(line_width) - overlap,
        torch.where(
            either,
            torch.full_like(overlap, float(line_width)),
            torch.zeros_like(overlap),
        ),
    )
    quality_ng = overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1.0e-6)
    quality_ng = torch.where(
        candidate_valid[:, None] & gt_lane_valid[None, :],
        quality_ng,
        torch.zeros_like(quality_ng),
    )
    target_gn = pred_x.new_zeros((gt_count, candidates))
    support_gn = torch.zeros((gt_count, candidates), dtype=torch.bool)
    supervised_gt = torch.zeros((gt_count,), dtype=torch.bool)
    candidate_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten()
    gt_ids = torch.nonzero(gt_lane_valid, as_tuple=False).flatten()
    if int(candidate_ids.numel()) and int(gt_ids.numel()):
        supervised_ids = [int(value) for value in gt_ids.tolist()]
        if len(supervised_ids) > int(num_slots):
            local_quality = quality_ng[candidate_ids][:, gt_ids]
            local_candidates, local_gt = linear_sum_assignment(
                (1.0 - local_quality).numpy()
            )
            assigned = [
                (
                    int(gt_ids[int(local)]),
                    float(local_quality[int(candidate), int(local)]),
                )
                for candidate, local in zip(
                    local_candidates.tolist(), local_gt.tolist()
                )
            ]
            assigned.sort(key=lambda item: item[1], reverse=True)
            supervised_ids = [gt for gt, _quality in assigned[: int(num_slots)]]
        for gt in supervised_ids:
            gt_quality = quality_ng[:, gt]
            best = gt_quality[candidate_valid].amax()
            effective_floor = best.clamp(max=float(cluster_min))
            cutoff = torch.maximum(
                effective_floor,
                best - float(cluster_delta),
            )
            support = candidate_valid & (gt_quality >= cutoff)
            probability = torch.softmax(
                gt_quality[support] / float(temperature),
                dim=0,
            )
            target_gn[gt, support] = probability
            support_gn[gt] = support
            supervised_gt[gt] = True
    return (
        target_gn,
        support_gn,
        quality_ng.transpose(0, 1).contiguous(),
        supervised_gt,
    )


def _conditional_route(
    stage: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joint = stage["selection_slot_logits"].detach().float()
    candidate_valid = stage["selection_slot_candidate_valid"].detach().bool()
    candidates = int(candidate_valid.numel())
    if joint.ndim != 2 or int(joint.shape[1]) != candidates + 1:
        raise ValueError("selection_slot_logits must have shape [S,N+1]")
    real_joint = joint[:, :candidates]
    real_log_mass = torch.logsumexp(real_joint, dim=-1)
    conditional_log = real_joint - real_log_mass.unsqueeze(-1)
    conditional_log = conditional_log.masked_fill(
        ~candidate_valid.unsqueeze(0),
        float("-inf"),
    )
    active_probability = real_log_mass.exp().clamp(1.0e-7, 1.0 - 1.0e-7)
    active_logits = torch.logit(active_probability)
    return conditional_log, conditional_log.exp(), active_logits


def _target_slot_assignment(
    active_logits: torch.Tensor,
    conditional_log: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, ...]:
    slots = int(active_logits.numel())
    gt_count = int(targets.shape[0])
    if gt_count == 0:
        return ()
    # Invalid candidate columns carry -inf log probability.  Their target
    # mass is zero, but a raw einsum would still form 0 * -inf = NaN.
    invalid_target_mass = targets[:, ~torch.isfinite(conditional_log).all(dim=0)].sum()
    if float(invalid_target_mass) > 1.0e-6:
        raise ValueError("route target places mass on a route-invalid candidate")
    safe_log = torch.where(
        torch.isfinite(conditional_log),
        conditional_log,
        torch.zeros_like(conditional_log),
    )
    candidate_cost = -torch.einsum("sn,gn->sg", safe_log, targets)
    active_cost = F.softplus(-active_logits)
    inactive_cost = F.softplus(active_logits)
    best_value = float("inf")
    best_path: tuple[int, ...] | None = None
    for path in itertools.permutations(range(slots), gt_count):
        assigned = set(path)
        value = sum(
            float(candidate_cost[slot, gt] + active_cost[slot])
            for gt, slot in enumerate(path)
        )
        value += sum(
            float(inactive_cost[slot])
            for slot in range(slots)
            if slot not in assigned
        )
        if value < best_value:
            best_value = value
            best_path = tuple(int(slot) for slot in path)
    if best_path is None:
        raise RuntimeError("could not assign route slots to GT rows")
    return best_path


def _unique_target_policy(targets: torch.Tensor) -> tuple[int, ...]:
    gt_count, candidates = targets.shape
    if gt_count == 0:
        return ()
    if candidates < gt_count:
        raise ValueError("fewer candidates than GT rows")
    gt_ids, candidate_ids = linear_sum_assignment(-targets.numpy())
    result = [-1] * gt_count
    for gt, candidate in zip(gt_ids.tolist(), candidate_ids.tolist()):
        result[int(gt)] = int(candidate)
    if any(candidate < 0 for candidate in result):
        raise RuntimeError("target policy left a GT row unassigned")
    return tuple(result)


def _official_match(
    official_iou: torch.Tensor,
    selected: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    gt_count = int(official_iou.shape[0])
    selected_ids = tuple(int(value) for value in selected if int(value) >= 0)
    matched_iou = official_iou.new_zeros((gt_count,))
    matched_candidate = torch.full((gt_count,), -1, dtype=torch.long)
    if gt_count == 0 or not selected_ids:
        return matched_iou, matched_candidate
    matrix = official_iou[:, torch.tensor(selected_ids, dtype=torch.long)]
    gt_ids, local_ids = linear_sum_assignment(1.0 - matrix.numpy())
    for gt, local in zip(gt_ids.tolist(), local_ids.tolist()):
        matched_iou[int(gt)] = matrix[int(gt), int(local)]
        matched_candidate[int(gt)] = int(selected_ids[int(local)])
    return matched_iou, matched_candidate


def _analyze_record(
    record: dict[str, Any],
    *,
    input_h: int,
    threshold: float,
    line_width: float,
    min_valid_rows: int,
    cluster_min: float,
    cluster_delta: float,
    temperature: float,
) -> list[LaneObservation]:
    stage = _stage(record)
    targets, target_support, _surrogate, target_supervised = build_training_route_targets(
        record,
        stage,
        input_h=input_h,
        line_width=line_width,
        min_valid_rows=min_valid_rows,
        cluster_min=cluster_min,
        cluster_delta=cluster_delta,
        temperature=temperature,
    )
    official = stage["official_iou"].detach().float()
    official_valid = stage["official_candidate_valid"].detach().bool()
    if int(targets.shape[0]) != int(official.shape[0]):
        raise ValueError("training target and official GT counts differ")
    good = official_valid.unsqueeze(0) & (official > float(threshold))
    official_masked = official.masked_fill(
        ~official_valid.unsqueeze(0),
        float("-inf"),
    )
    official_best_iou, official_best = official_masked.max(dim=-1)
    target_top1 = torch.full((int(targets.shape[0]),), -1, dtype=torch.long)
    supervised_ids = torch.nonzero(target_supervised, as_tuple=False).flatten()
    if int(supervised_ids.numel()):
        target_top1[supervised_ids] = targets[supervised_ids].argmax(dim=-1)
        local_policy = _unique_target_policy(targets[supervised_ids])
        target_policy = {
            int(gt): int(candidate)
            for gt, candidate in zip(supervised_ids.tolist(), local_policy)
        }
    else:
        target_policy = {}

    conditional_log, conditional_probability, active_logits = _conditional_route(stage)
    local_slot_assignment = _target_slot_assignment(
        active_logits,
        conditional_log,
        targets[supervised_ids],
    )
    slot_assignment = {
        int(gt): int(slot)
        for gt, slot in zip(supervised_ids.tolist(), local_slot_assignment)
    }
    route_ids = stage["selection_slot_indices"].detach().long()
    active = stage["selection_slot_active"].detach().bool()
    if int(route_ids.numel()) != int(active.numel()):
        raise ValueError("route and active slot counts differ")
    all_route_ids = tuple(int(value) for value in route_ids.tolist() if int(value) >= 0)
    active_route_ids = tuple(
        int(route_ids[slot])
        for slot in range(int(route_ids.numel()))
        if bool(active[slot]) and int(route_ids[slot]) >= 0
    )
    learned_match_iou, _ = _official_match(official, active_route_ids)

    rows: list[LaneObservation] = []
    for gt in range(int(targets.shape[0])):
        target_id = int(target_top1[gt])
        supervised = bool(target_supervised[gt])
        slot = int(slot_assignment[gt]) if supervised else -1
        selected_id = int(route_ids[slot]) if supervised else -1
        selected_valid = supervised and 0 <= selected_id < int(official.shape[1])
        any_slot_good = any(bool(good[gt, candidate]) for candidate in all_route_ids)
        any_active_good = any(
            bool(good[gt, candidate]) for candidate in active_route_ids
        )
        rows.append(
            LaneObservation(
                image_id=str(record["image_id"]),
                gt_index=gt,
                support_hit=bool(good[gt].any()),
                official_best_iou=float(official_best_iou[gt]),
                official_best_proposal=int(official_best[gt]),
                target_supervised=supervised,
                target_support_has_good=(
                    bool((target_support[gt] & good[gt]).any())
                    if supervised
                    else False
                ),
                target_top1_good=(
                    bool(good[gt, target_id]) if supervised else False
                ),
                target_top1_is_official_best=(
                    supervised and target_id == int(official_best[gt])
                ),
                target_top1_official_iou=(
                    float(official[gt, target_id]) if supervised else 0.0
                ),
                target_good_mass=float((targets[gt] * good[gt].float()).sum()),
                target_policy_hit=(
                    bool(good[gt, target_policy[gt]]) if supervised else False
                ),
                learned_target_slot_good=(
                    bool(good[gt, selected_id]) if selected_valid else False
                ),
                learned_target_slot_active=(
                    bool(active[slot]) if supervised else False
                ),
                learned_target_slot_active_good=(
                    bool(active[slot] and good[gt, selected_id])
                    if selected_valid
                    else False
                ),
                learned_target_slot_is_target_top1=(
                    selected_valid and selected_id == target_id
                ),
                learned_target_slot_official_iou=(
                    float(official[gt, selected_id]) if selected_valid else 0.0
                ),
                learned_good_mass=float(
                    (conditional_probability[slot] * good[gt].float()).sum()
                ) if supervised else 0.0,
                learned_any_slot_good=any_slot_good,
                learned_any_active_slot_good=any_active_good,
                learned_official_match_hit=bool(
                    learned_match_iou[gt] > float(threshold)
                ),
                learned_official_match_iou=float(learned_match_iou[gt]),
            )
        )
    return rows


_BOOL_FIELDS = (
    "support_hit",
    "target_supervised",
    "target_support_has_good",
    "target_top1_good",
    "target_top1_is_official_best",
    "target_policy_hit",
    "learned_target_slot_good",
    "learned_target_slot_active",
    "learned_target_slot_active_good",
    "learned_target_slot_is_target_top1",
    "learned_any_slot_good",
    "learned_any_active_slot_good",
    "learned_official_match_hit",
)

_FLOAT_FIELDS = (
    "official_best_iou",
    "target_top1_official_iou",
    "target_good_mass",
    "learned_target_slot_official_iou",
    "learned_good_mass",
    "learned_official_match_iou",
)


def summarize_observations(rows: list[LaneObservation]) -> dict[str, Any]:
    count = len(rows)
    result: dict[str, Any] = {"lanes": count}
    for field in _BOOL_FIELDS:
        positives = sum(bool(getattr(row, field)) for row in rows)
        result[field] = {
            "count": int(positives),
            "fraction": float(positives / count) if count else None,
        }
    for field in _FLOAT_FIELDS:
        values = np.asarray([float(getattr(row, field)) for row in rows], dtype=np.float64)
        result[field] = {
            "mean": float(values.mean()) if count else None,
            "median": float(np.median(values)) if count else None,
        }
    return result


def _sample_rows(rows: list[LaneObservation], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -row.official_best_iou,
            row.learned_official_match_iou,
            row.image_id,
            row.gt_index,
        ),
    )
    return [row.__dict__ for row in ordered[: int(limit)]]


def audit_route_target_conversion(
    control_cache: dict[str, Any],
    treatment_cache: dict[str, Any],
    *,
    threshold: float,
    line_width: float,
    min_valid_rows: int,
    cluster_min: float,
    cluster_delta: float,
    temperature: float,
    sample_limit: int,
) -> dict[str, Any]:
    control_records = {
        str(record["image_id"]): record for record in control_cache["records"]
    }
    treatment_records = {
        str(record["image_id"]): record for record in treatment_cache["records"]
    }
    if set(control_records) != set(treatment_records):
        raise ValueError("control and treatment cache image sets differ")
    control_h = int(control_cache.get("metadata", {}).get("input_h", 640))
    treatment_h = int(treatment_cache.get("metadata", {}).get("input_h", 640))
    if control_h != treatment_h:
        raise ValueError("control and treatment input heights differ")

    control_rows: list[LaneObservation] = []
    treatment_rows: list[LaneObservation] = []
    groups: dict[str, list[LaneObservation]] = {
        "support_gained_treatment": [],
        "support_lost_control": [],
        "support_retained_treatment": [],
        "support_absent_treatment": [],
        "treatment_supported_but_learned_missed": [],
        "treatment_target_support_knows_but_learned_missed": [],
        "treatment_target_top1_knows_but_learned_missed": [],
    }
    target_mismatch_images = 0
    excluded_gt_contract_mismatch: list[dict[str, Any]] = []
    for image_index, image_id in enumerate(sorted(control_records), start=1):
        control_record = control_records[image_id]
        treatment_record = treatment_records[image_id]
        control_target_gt = int(control_record["target"]["x_rows"].shape[0])
        treatment_target_gt = int(treatment_record["target"]["x_rows"].shape[0])
        control_official_gt = int(
            _stage(control_record)["official_iou"].shape[0]
        )
        treatment_official_gt = int(
            _stage(treatment_record)["official_iou"].shape[0]
        )
        if not (
            control_target_gt
            == treatment_target_gt
            == control_official_gt
            == treatment_official_gt
        ):
            excluded_gt_contract_mismatch.append(
                {
                    "image_id": image_id,
                    "control_target_gt": control_target_gt,
                    "treatment_target_gt": treatment_target_gt,
                    "control_official_gt": control_official_gt,
                    "treatment_official_gt": treatment_official_gt,
                }
            )
            continue
        left_target = control_record["target"]
        right_target = treatment_record["target"]
        same_mask = torch.equal(
            left_target["valid_mask"], right_target["valid_mask"]
        )
        finite = (
            left_target["valid_mask"].bool()
            & right_target["valid_mask"].bool()
            & torch.isfinite(left_target["x_rows"])
            & torch.isfinite(right_target["x_rows"])
        )
        same_values = bool(
            torch.allclose(
                left_target["x_rows"][finite],
                right_target["x_rows"][finite],
                atol=0.0,
                rtol=0.0,
            )
        )
        same_target = same_mask and same_values
        if not same_target:
            target_mismatch_images += 1
        left = _analyze_record(
            control_record,
            input_h=control_h,
            threshold=threshold,
            line_width=line_width,
            min_valid_rows=min_valid_rows,
            cluster_min=cluster_min,
            cluster_delta=cluster_delta,
            temperature=temperature,
        )
        right = _analyze_record(
            treatment_record,
            input_h=treatment_h,
            threshold=threshold,
            line_width=line_width,
            min_valid_rows=min_valid_rows,
            cluster_min=cluster_min,
            cluster_delta=cluster_delta,
            temperature=temperature,
        )
        if len(left) != len(right):
            raise ValueError(f"GT count differs for {image_id}")
        control_rows.extend(left)
        treatment_rows.extend(right)
        for before, after in zip(left, right):
            if not before.support_hit and after.support_hit:
                groups["support_gained_treatment"].append(after)
            elif before.support_hit and not after.support_hit:
                groups["support_lost_control"].append(before)
            elif before.support_hit and after.support_hit:
                groups["support_retained_treatment"].append(after)
            else:
                groups["support_absent_treatment"].append(after)
            if after.support_hit and not after.learned_official_match_hit:
                groups["treatment_supported_but_learned_missed"].append(after)
                if after.target_support_has_good:
                    groups[
                        "treatment_target_support_knows_but_learned_missed"
                    ].append(after)
                if after.target_top1_good:
                    groups[
                        "treatment_target_top1_knows_but_learned_missed"
                    ].append(after)
        if image_index % 500 == 0 or image_index == len(control_records):
            print(
                f"route-target audit progress: {image_index}/{len(control_records)} images",
                flush=True,
            )

    gained = groups["support_gained_treatment"]
    missed = groups["treatment_supported_but_learned_missed"]
    gained_target_knows = sum(row.target_support_has_good for row in gained)
    gained_target_top1 = sum(row.target_top1_good for row in gained)
    missed_target_knows = sum(row.target_support_has_good for row in missed)
    missed_target_top1 = sum(row.target_top1_good for row in missed)

    if gained and gained_target_knows / len(gained) >= 0.80:
        gained_diagnosis = "training_target_usually_contains_new_strict_support"
    elif gained and gained_target_knows / len(gained) < 0.50:
        gained_diagnosis = "training_target_usually_misses_new_strict_support"
    else:
        gained_diagnosis = "mixed_target_alignment_on_new_strict_support"
    if missed and missed_target_top1 / len(missed) >= 0.70:
        primary = "learned_route_fails_despite_strong_target_alignment"
    elif missed and missed_target_knows / len(missed) >= 0.70:
        primary = "soft_target_contains_good_proposal_but_route_conversion_fails"
    elif missed and missed_target_knows / len(missed) < 0.50:
        primary = "route_target_mismatch_is_primary"
    else:
        primary = "mixed_target_and_learned_route_failure"

    return {
        "experiment": "V30 exact-paired strict route-target conversion audit",
        "diagnostic_only": True,
        "test_split_used": False,
        "contract": {
            "same_image_set": True,
            "target_mismatch_images": int(target_mismatch_images),
            "images_total": len(control_records),
            "images_analyzed": (
                len(control_records) - len(excluded_gt_contract_mismatch)
            ),
            "excluded_gt_contract_mismatch_images": (
                excluded_gt_contract_mismatch
            ),
            "threshold": float(threshold),
            "target_mode": "all_gt",
            "cluster_min": float(cluster_min),
            "cluster_delta": float(cluster_delta),
            "cluster_temperature": float(temperature),
            "line_width": float(line_width),
            "min_valid_rows": int(min_valid_rows),
        },
        "all_lanes": {
            "control": summarize_observations(control_rows),
            "treatment": summarize_observations(treatment_rows),
        },
        "transition_groups": {
            name: summarize_observations(rows) for name, rows in groups.items()
        },
        "key_counts": {
            "support_gained": len(gained),
            "support_lost": len(groups["support_lost_control"]),
            "net_support_gain": len(gained) - len(groups["support_lost_control"]),
            "gained_target_support_knows": int(gained_target_knows),
            "gained_target_top1_knows": int(gained_target_top1),
            "treatment_supported_but_learned_missed": len(missed),
            "missed_target_support_knows": int(missed_target_knows),
            "missed_target_top1_knows": int(missed_target_top1),
        },
        "verdict": {
            "new_support_target_alignment": gained_diagnosis,
            "primary_failure": primary,
            "interpretation_contract": (
                "Official-IoU support and route targets use validation GT and "
                "are diagnostic only. They are not deployable policies."
            ),
        },
        "samples": {
            "gained_target_knows_learned_misses": _sample_rows(
                [
                    row
                    for row in gained
                    if row.target_support_has_good
                    and not row.learned_official_match_hit
                ],
                sample_limit,
            ),
            "gained_target_misses": _sample_rows(
                [row for row in gained if not row.target_support_has_good],
                sample_limit,
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate V30 strict proposal support, route-target alignment, "
            "learned route identity, and activity conversion."
        )
    )
    parser.add_argument("--control-cache", type=Path, required=True)
    parser.add_argument("--treatment-cache", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.75)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--cluster-min", type=float, default=0.0)
    parser.add_argument("--cluster-delta", type=float, default=0.10)
    parser.add_argument("--cluster-temperature", type=float, default=0.03)
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control = _load_cache(args.control_cache.expanduser().resolve())
    treatment = _load_cache(args.treatment_cache.expanduser().resolve())
    result = audit_route_target_conversion(
        control,
        treatment,
        threshold=float(args.iou_threshold),
        line_width=float(args.line_width),
        min_valid_rows=int(args.min_valid_rows),
        cluster_min=float(args.cluster_min),
        cluster_delta=float(args.cluster_delta),
        temperature=float(args.cluster_temperature),
        sample_limit=int(args.sample_limit),
    )
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
