from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations, product
import json
from pathlib import Path
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment

from dynlaneseq_eg.modeling.common import fixed_row_fractions, sort_range_norm


@dataclass(frozen=True)
class NeighborhoodPolicy:
    name: str
    max_candidates: int
    max_mean_distance_px: float | None
    min_common_fraction: float


POLICIES = (
    NeighborhoodPolicy("local4_24px", 4, 24.0, 0.50),
    NeighborhoodPolicy("local4_48px", 4, 48.0, 0.50),
    NeighborhoodPolicy("local4_96px", 4, 96.0, 0.50),
    NeighborhoodPolicy("local8_48px", 8, 48.0, 0.50),
    NeighborhoodPolicy("knn4", 4, None, 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether a deployment-available geometric neighborhood "
            "around each current V7 route contains enough alternative "
            "proposal geometry to remove the representative bottleneck."
        )
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=(0.5, 0.75))
    return parser.parse_args()


def _pairwise_curve_geometry(
    x_rows: torch.Tensor,
    ranges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deployment-only mean distance and common-visible fraction."""

    if x_rows.ndim != 2 or ranges.shape != (x_rows.shape[0], 2):
        raise ValueError("proposal geometry must be [N,R] and [N,2]")
    candidates, rows = x_rows.shape
    ranges = sort_range_norm(ranges.float())
    y = fixed_row_fractions(
        int(rows),
        device=x_rows.device,
        dtype=torch.float32,
    ).view(1, rows)
    visible = (
        (y >= ranges[:, :1])
        & (y <= ranges[:, 1:])
        & torch.isfinite(x_rows)
    )
    common = visible[:, None, :] & visible[None, :, :]
    common_count = common.sum(dim=-1)
    denominator = common_count.clamp_min(1).float()
    distance = (x_rows[:, None, :] - x_rows[None, :, :]).abs()
    mean_distance = (
        distance.masked_fill(~common, 0.0).sum(dim=-1) / denominator
    )
    mean_distance = mean_distance.masked_fill(common_count == 0, torch.inf)
    shorter_visible = torch.minimum(
        visible.sum(dim=-1)[:, None],
        visible.sum(dim=-1)[None, :],
    ).clamp_min(1)
    common_fraction = common_count.float() / shorter_visible.float()
    if candidates:
        diagonal = torch.arange(candidates, device=x_rows.device)
        mean_distance[diagonal, diagonal] = 0.0
        common_fraction[diagonal, diagonal] = 1.0
    return mean_distance, common_fraction


def _neighbors_for_anchor(
    anchor: int,
    *,
    candidate_valid: torch.Tensor,
    mean_distance: torch.Tensor,
    common_fraction: torch.Tensor,
    policy: NeighborhoodPolicy,
) -> torch.Tensor:
    if anchor < 0 or anchor >= int(candidate_valid.numel()):
        return torch.empty(0, dtype=torch.long)
    valid = candidate_valid.bool().clone()
    valid &= torch.isfinite(mean_distance[anchor])
    valid &= common_fraction[anchor] >= float(policy.min_common_fraction)
    if policy.max_mean_distance_px is not None:
        valid &= mean_distance[anchor] <= float(policy.max_mean_distance_px)
    valid[anchor] = bool(candidate_valid[anchor])
    ids = torch.nonzero(valid, as_tuple=False).flatten()
    if ids.numel() == 0:
        return torch.tensor([anchor], dtype=torch.long)
    order = mean_distance[anchor, ids].argsort(stable=True)
    ids = ids[order[: int(policy.max_candidates)]]
    if not bool((ids == anchor).any()):
        if int(ids.numel()) >= int(policy.max_candidates):
            ids = ids.clone()
            ids[-1] = int(anchor)
        else:
            ids = torch.cat((ids, ids.new_tensor([anchor])))
    return ids.unique(sorted=False)


def _max_binary_matching_hits(edge: torch.Tensor, max_predictions: int | None = None) -> int:
    """Exact maximum thresholded GT/prediction matching for G<=4."""

    if edge.ndim != 2:
        raise ValueError("matching edge matrix must be [G,P]")
    gt_count, prediction_count = edge.shape
    if gt_count == 0 or prediction_count == 0:
        return 0
    capacity = prediction_count
    if max_predictions is not None:
        capacity = min(capacity, int(max_predictions))
    if capacity <= 0:
        return 0

    # Standard augmenting-path bipartite matching.  The full proposal side may
    # contain 32 nodes, so enumerating 32P4 assignments would turn a cheap
    # audit into hundreds of millions of Python iterations.
    matched_gt_for_prediction = [-1 for _ in range(prediction_count)]

    def augment(gt_index: int, seen: list[bool]) -> bool:
        for prediction_index in range(prediction_count):
            if seen[prediction_index] or not bool(edge[gt_index, prediction_index]):
                continue
            seen[prediction_index] = True
            previous_gt = matched_gt_for_prediction[prediction_index]
            if previous_gt < 0 or augment(previous_gt, seen):
                matched_gt_for_prediction[prediction_index] = gt_index
                return True
        return False

    hits = 0
    for gt_index in range(gt_count):
        if augment(gt_index, [False for _ in range(prediction_count)]):
            hits += 1
            if hits >= capacity:
                break
    return hits


def _official_hungarian_hits(quality: torch.Tensor, threshold: float) -> int:
    if quality.ndim != 2:
        raise ValueError("quality must have shape [G,P]")
    if quality.shape[0] == 0 or quality.shape[1] == 0:
        return 0
    gt_ids, pred_ids = linear_sum_assignment(1.0 - quality.numpy())
    return int((quality[gt_ids, pred_ids] >= float(threshold)).sum())


def _slot_neighborhood_oracle_hits(
    quality: torch.Tensor,
    neighborhoods: list[torch.Tensor],
    threshold: float,
) -> int:
    """Choose one unique proposal per active slot and maximize threshold TP."""

    if quality.shape[0] == 0 or not neighborhoods:
        return 0
    choices = [tuple(int(value) for value in ids.tolist()) for ids in neighborhoods]
    best = 0
    for selected in product(*choices):
        if len(set(selected)) != len(selected):
            continue
        local = quality[:, torch.tensor(selected, dtype=torch.long)]
        hits = _max_binary_matching_hits(local >= float(threshold))
        best = max(best, hits)
        if best == min(int(quality.shape[0]), len(selected)):
            return best
    return best


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def audit_cache(cache: dict[str, Any], thresholds: tuple[float, ...]) -> dict[str, Any]:
    records = cache.get("records")
    if not isinstance(records, list):
        raise ValueError("diagnostic cache has no records list")
    aggregates: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        aggregates[f"{threshold:.2f}"] = {
            "gt": 0,
            "current_reference_tp": 0,
            "current_refined_tp": 0,
            "same_count_all32_oracle_tp": 0,
            "top4_all32_oracle_tp": 0,
            "policies": {
                policy.name: {"tp": 0}
                for policy in POLICIES
            },
        }

    support_sizes = {policy.name: [] for policy in POLICIES}
    anchor_neighbor_distances = {policy.name: [] for policy in POLICIES}
    images_with_routes = 0
    active_slots_total = 0

    for record in records:
        stage = record["stages"]["main"]
        proposal_quality = stage["official_iou"].float()
        proposal_valid = stage["official_candidate_valid"].bool()
        slot_quality = stage["selection_slot_official_iou"].float()
        anchors = stage["selection_slot_indices"].long()
        active = anchors >= 0
        active_ids = torch.nonzero(active, as_tuple=False).flatten()
        active_anchors = anchors[active_ids]
        gt_count = int(proposal_quality.shape[0])
        active_count = int(active_ids.numel())
        if active_count:
            images_with_routes += 1
            active_slots_total += active_count

        mean_distance, common_fraction = _pairwise_curve_geometry(
            stage["pred_x_rows"].float(),
            stage["range_norm"].float(),
        )
        neighborhoods_by_policy: dict[str, list[torch.Tensor]] = {}
        for policy in POLICIES:
            neighborhoods: list[torch.Tensor] = []
            for anchor in active_anchors.tolist():
                ids = _neighbors_for_anchor(
                    int(anchor),
                    candidate_valid=proposal_valid,
                    mean_distance=mean_distance,
                    common_fraction=common_fraction,
                    policy=policy,
                )
                neighborhoods.append(ids)
                support_sizes[policy.name].append(int(ids.numel()))
                for candidate in ids.tolist():
                    if int(candidate) != int(anchor):
                        anchor_neighbor_distances[policy.name].append(
                            float(mean_distance[int(anchor), int(candidate)])
                        )
            neighborhoods_by_policy[policy.name] = neighborhoods

        for threshold in thresholds:
            row = aggregates[f"{threshold:.2f}"]
            row["gt"] += gt_count
            if active_count:
                selected_quality = proposal_quality[:, active_anchors]
                refined_quality = slot_quality[:, active_ids]
                row["current_reference_tp"] += _official_hungarian_hits(
                    selected_quality,
                    threshold,
                )
                row["current_refined_tp"] += _official_hungarian_hits(
                    refined_quality,
                    threshold,
                )
            row["same_count_all32_oracle_tp"] += _max_binary_matching_hits(
                proposal_quality[:, proposal_valid] >= float(threshold),
                max_predictions=active_count,
            )
            row["top4_all32_oracle_tp"] += _max_binary_matching_hits(
                proposal_quality[:, proposal_valid] >= float(threshold),
                max_predictions=4,
            )
            for policy in POLICIES:
                row["policies"][policy.name]["tp"] += (
                    _slot_neighborhood_oracle_hits(
                        proposal_quality,
                        neighborhoods_by_policy[policy.name],
                        threshold,
                    )
                )

    for threshold_key, row in aggregates.items():
        current = int(row["current_reference_tp"])
        same_count = int(row["same_count_all32_oracle_tp"])
        gap = max(same_count - current, 0)
        row["representative_gap_tp"] = gap
        for policy in POLICIES:
            policy_row = row["policies"][policy.name]
            gain = int(policy_row["tp"]) - current
            policy_row["gain_tp"] = gain
            policy_row["gap_closure"] = _safe_fraction(gain, gap)

    support_summary: dict[str, dict[str, float | int]] = {}
    for policy in POLICIES:
        sizes = torch.tensor(support_sizes[policy.name], dtype=torch.float32)
        distances = torch.tensor(
            anchor_neighbor_distances[policy.name],
            dtype=torch.float32,
        )
        support_summary[policy.name] = {
            "slots": int(sizes.numel()),
            "mean_support": float(sizes.mean()) if sizes.numel() else 0.0,
            "fraction_with_alternative": (
                float((sizes > 1).float().mean()) if sizes.numel() else 0.0
            ),
            "mean_non_anchor_distance_px": (
                float(distances.mean()) if distances.numel() else 0.0
            ),
            "p90_non_anchor_distance_px": (
                float(torch.quantile(distances, 0.90)) if distances.numel() else 0.0
            ),
        }

    primary_050 = aggregates.get("0.50", {}).get("policies", {}).get(
        "local4_48px", {}
    )
    primary_075 = aggregates.get("0.75", {}).get("policies", {}).get(
        "local4_48px", {}
    )
    gate = (
        float(primary_050.get("gap_closure", 0.0)) >= 0.50
        and float(primary_075.get("gap_closure", 0.0)) >= 0.40
    )
    strong_gate = (
        float(primary_050.get("gap_closure", 0.0)) >= 0.70
        and float(primary_075.get("gap_closure", 0.0)) >= 0.60
    )
    return {
        "experiment": "V8 route-anchored proposal-neighborhood capacity audit",
        "diagnostic_only": True,
        "test_set_used": False,
        "records": len(records),
        "images_with_active_routes": images_with_routes,
        "active_slots": active_slots_total,
        "neighborhood_contract": {
            "distance": "mean absolute x distance on common visible rows",
            "inputs": "detached proposal x/range plus current route anchor",
            "gt_used_to_build_neighborhood": False,
            "gt_used_only_for_oracle_evaluation": True,
            "primary_policy": "local4_48px",
        },
        "support": support_summary,
        "thresholds": aggregates,
        "gate": {
            "definition": (
                "local4_48px closes >=50% of the representative TP gap at "
                "IoU .50 and >=40% at IoU .75"
            ),
            "passed": bool(gate),
            "strong_definition": (
                "local4_48px closes >=70% at IoU .50 and >=60% at IoU .75"
            ),
            "strong_passed": bool(strong_gate),
        },
    }


def main() -> None:
    args = parse_args()
    try:
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(args.cache, map_location="cpu")
    payload = audit_cache(cache, tuple(float(value) for value in args.iou_thresholds))
    payload["cache"] = str(Path(args.cache).resolve())
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
