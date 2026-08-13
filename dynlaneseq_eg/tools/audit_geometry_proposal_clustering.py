from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    cardinality_oracle_assignment,
    ensure_official_iou_cache,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    metadata_for_json,
    official_proposal_gt_iou_matrix,
    stage_scores,
    write_json,
)
from dynlaneseq_eg.modeling.common import fixed_row_fractions, sort_range_norm


REFERENCE_INPUT_WIDTH = 1600.0
PROTOTYPE_MODES = ("medoid", "median", "mean")
PRIMARY_POLICY = "perspective_balanced_48"
PRIMARY_PROTOTYPE = "medoid"
PRIMARY_SELECTION = "slot_cluster_mass_unique"


@dataclass(frozen=True)
class GeometryClusterPolicy:
    """A predeclared, GT-free proposal-pair contract.

    Pixel thresholds are expressed at 1600-pixel input width and are scaled
    linearly for other resolutions.  ``flat_complete_48`` is a V8-like
    reference.  The perspective policies require evidence in the lower image
    and use a hard lower-gap guard so convergence near the horizon cannot by
    itself merge two physical lanes.
    """

    name: str
    perspective_weighted: bool
    min_common_rows: int
    min_overlap_fraction: float
    max_weighted_median_px: float
    max_weighted_q90_px: float
    require_lower_evidence: bool
    lower_y_floor: float
    max_lower_median_px: float
    max_lower_q90_px: float
    max_top_endpoint_px: float
    max_bottom_endpoint_px: float
    max_range_start_gap: float
    max_range_end_gap: float
    max_lower_minus_upper_px: float
    max_polynomial_q90_px: float


POLICIES: tuple[GeometryClusterPolicy, ...] = (
    GeometryClusterPolicy(
        name="flat_complete_48",
        perspective_weighted=False,
        min_common_rows=8,
        min_overlap_fraction=0.50,
        max_weighted_median_px=48.0,
        max_weighted_q90_px=96.0,
        require_lower_evidence=False,
        lower_y_floor=0.0,
        max_lower_median_px=math.inf,
        max_lower_q90_px=math.inf,
        max_top_endpoint_px=math.inf,
        max_bottom_endpoint_px=math.inf,
        max_range_start_gap=math.inf,
        max_range_end_gap=math.inf,
        max_lower_minus_upper_px=math.inf,
        max_polynomial_q90_px=math.inf,
    ),
    GeometryClusterPolicy(
        name="perspective_balanced_48",
        perspective_weighted=True,
        min_common_rows=8,
        min_overlap_fraction=0.40,
        max_weighted_median_px=48.0,
        max_weighted_q90_px=72.0,
        require_lower_evidence=True,
        lower_y_floor=0.58,
        max_lower_median_px=48.0,
        max_lower_q90_px=72.0,
        max_top_endpoint_px=96.0,
        max_bottom_endpoint_px=80.0,
        max_range_start_gap=0.25,
        max_range_end_gap=0.15,
        max_lower_minus_upper_px=40.0,
        max_polynomial_q90_px=80.0,
    ),
    GeometryClusterPolicy(
        name="perspective_conservative_36",
        perspective_weighted=True,
        min_common_rows=10,
        min_overlap_fraction=0.50,
        max_weighted_median_px=36.0,
        max_weighted_q90_px=56.0,
        require_lower_evidence=True,
        lower_y_floor=0.62,
        max_lower_median_px=36.0,
        max_lower_q90_px=56.0,
        max_top_endpoint_px=80.0,
        max_bottom_endpoint_px=64.0,
        max_range_start_gap=0.18,
        max_range_end_gap=0.10,
        max_lower_minus_upper_px=32.0,
        max_polynomial_q90_px=64.0,
    ),
)


@dataclass(frozen=True)
class PairGeometry:
    valid: bool
    common_rows: int
    overlap_fraction_min: float
    min_common_y: float
    max_common_y: float
    unweighted_mean_px: float
    unweighted_q90_px: float
    weighted_median_px: float
    weighted_q90_px: float
    lower_available: bool
    lower_median_px: float
    lower_q90_px: float
    upper_median_px: float
    top_endpoint_px: float
    bottom_endpoint_px: float
    range_start_gap: float
    range_end_gap: float
    lower_minus_upper_px: float
    polynomial_q90_px: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free, perspective-aware geometry clustering audit over "
            "all proposal curves. GT is used only after clustering to measure "
            "purity, fragmentation, and official prototype capacity."
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
    parser.add_argument("--stage", default="main")
    parser.add_argument("--sample-strategy", choices=("uniform", "sequential"), default="uniform")
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--official-iou-workers", type=int, default=12)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--label-iou-threshold", type=float, default=0.50)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=(0.50, 0.75))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    raw = list(float(value) for value in values)
    data = [value for value in raw if math.isfinite(value)]
    if not data:
        return {
            "count": len(raw),
            "finite_count": 0,
            "nonfinite_count": len(raw),
            "mean": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
        }
    tensor = torch.tensor(data, dtype=torch.float64)
    return {
        "count": len(raw),
        "finite_count": int(tensor.numel()),
        "nonfinite_count": len(raw) - int(tensor.numel()),
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "median": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
    }


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, quantile: float) -> float:
    if values.numel() == 0:
        return math.inf
    order = values.argsort()
    sorted_values = values[order]
    sorted_weights = weights[order].clamp_min(0.0)
    total = sorted_weights.sum()
    if float(total) <= 0.0:
        return float(torch.quantile(sorted_values, float(quantile)))
    cumulative = sorted_weights.cumsum(0) / total
    index = int(torch.searchsorted(cumulative, cumulative.new_tensor(float(quantile))).clamp(max=values.numel() - 1))
    return float(sorted_values[index])


def _quadratic_fit(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Stable ridge quadratic on normalized y; used only as a shape guard."""

    design = torch.stack((torch.ones_like(y), y, y.square()), dim=-1).double()
    target = x.double().unsqueeze(-1)
    ridge = torch.eye(3, dtype=torch.float64, device=design.device) * 1.0e-6
    return torch.linalg.solve(design.T @ design + ridge, design.T @ target).squeeze(-1)


def _pair_geometry(
    x_a: torch.Tensor,
    mask_a: torch.Tensor,
    x_b: torch.Tensor,
    mask_b: torch.Tensor,
    y_fraction: torch.Tensor,
    polynomial_a: torch.Tensor | None = None,
    polynomial_b: torch.Tensor | None = None,
) -> PairGeometry:
    common = mask_a.bool() & mask_b.bool() & torch.isfinite(x_a) & torch.isfinite(x_b)
    common_count = int(common.sum())
    valid_a = int(mask_a.bool().sum())
    valid_b = int(mask_b.bool().sum())
    if common_count == 0 or min(valid_a, valid_b) == 0:
        return PairGeometry(
            valid=False,
            common_rows=common_count,
            overlap_fraction_min=0.0,
            min_common_y=0.0,
            max_common_y=0.0,
            unweighted_mean_px=math.inf,
            unweighted_q90_px=math.inf,
            weighted_median_px=math.inf,
            weighted_q90_px=math.inf,
            lower_available=False,
            lower_median_px=math.inf,
            lower_q90_px=math.inf,
            upper_median_px=math.inf,
            top_endpoint_px=math.inf,
            bottom_endpoint_px=math.inf,
            range_start_gap=math.inf,
            range_end_gap=math.inf,
            lower_minus_upper_px=math.inf,
            polynomial_q90_px=math.inf,
        )
    y = y_fraction[common].float()
    gap = (x_a[common].float() - x_b[common].float()).abs()
    # A small nonzero upper weight keeps the full shared curve visible while
    # making the lower road dominate the decision under perspective.
    weights = 0.10 + 0.90 * y.pow(3.0)
    weighted_median = _weighted_quantile(gap, weights, 0.50)
    weighted_q90 = _weighted_quantile(gap, weights, 0.90)
    upper_cut = float(torch.quantile(y, 0.35))
    lower_cut = max(0.55, float(torch.quantile(y, 0.65)))
    upper = gap[y <= upper_cut]
    lower = gap[y >= lower_cut]
    upper_median = float(upper.median()) if upper.numel() else math.inf
    lower_median = float(lower.median()) if lower.numel() else math.inf
    lower_q90 = float(torch.quantile(lower, 0.90)) if lower.numel() else math.inf
    lower_available = bool(int(lower.numel()) >= 3 and float(y.max()) >= 0.55)
    endpoint_count = min(3, int(gap.numel()))
    top_indices = y.argsort()[:endpoint_count]
    bottom_indices = y.argsort(descending=True)[:endpoint_count]
    top_endpoint = float(gap[top_indices].median())
    bottom_endpoint = float(gap[bottom_indices].median())
    valid_a_y = y_fraction[mask_a.bool() & torch.isfinite(x_a)].float()
    valid_b_y = y_fraction[mask_b.bool() & torch.isfinite(x_b)].float()
    range_start_gap = float((valid_a_y.min() - valid_b_y.min()).abs())
    range_end_gap = float((valid_a_y.max() - valid_b_y.max()).abs())

    # The polynomial is never extrapolated beyond the observed common range.
    # Raw rows remain the primary signal; this merely suppresses row noise.
    if polynomial_a is None:
        valid_a_mask = mask_a.bool() & torch.isfinite(x_a)
        ya = y_fraction[valid_a_mask].float()
        polynomial_a = (
            _quadratic_fit(ya, x_a[valid_a_mask].float())
            if int(ya.numel()) >= 3
            else None
        )
    if polynomial_b is None:
        valid_b_mask = mask_b.bool() & torch.isfinite(x_b)
        yb = y_fraction[valid_b_mask].float()
        polynomial_b = (
            _quadratic_fit(yb, x_b[valid_b_mask].float())
            if int(yb.numel()) >= 3
            else None
        )
    if polynomial_a is not None and polynomial_b is not None:
        basis = torch.stack((torch.ones_like(y), y, y.square()), dim=-1).double()
        polynomial_gap = ((basis @ polynomial_a) - (basis @ polynomial_b)).abs().float()
        polynomial_q90 = _weighted_quantile(polynomial_gap, weights, 0.90)
    else:
        polynomial_q90 = math.inf
    return PairGeometry(
        valid=True,
        common_rows=common_count,
        overlap_fraction_min=float(common_count) / float(max(min(valid_a, valid_b), 1)),
        min_common_y=float(y.min()),
        max_common_y=float(y.max()),
        unweighted_mean_px=float(gap.mean()),
        unweighted_q90_px=float(torch.quantile(gap, 0.90)),
        weighted_median_px=weighted_median,
        weighted_q90_px=weighted_q90,
        lower_available=lower_available,
        lower_median_px=lower_median,
        lower_q90_px=lower_q90,
        upper_median_px=upper_median,
        top_endpoint_px=top_endpoint,
        bottom_endpoint_px=bottom_endpoint,
        range_start_gap=range_start_gap,
        range_end_gap=range_end_gap,
        lower_minus_upper_px=max(0.0, lower_median - upper_median),
        polynomial_q90_px=polynomial_q90,
    )


def _policy_distance(
    geometry: PairGeometry,
    policy: GeometryClusterPolicy,
    input_w: int,
) -> tuple[float, str]:
    scale = float(input_w) / REFERENCE_INPUT_WIDTH
    if not geometry.valid:
        return math.inf, "no_common_rows"
    if geometry.common_rows < int(policy.min_common_rows):
        return math.inf, "too_few_common_rows"
    if geometry.overlap_fraction_min < float(policy.min_overlap_fraction):
        return math.inf, "insufficient_overlap"
    if policy.require_lower_evidence and (
        not geometry.lower_available or geometry.max_common_y < float(policy.lower_y_floor)
    ):
        return math.inf, "no_reliable_lower_evidence"

    if policy.perspective_weighted:
        values = [
            geometry.weighted_median_px / (float(policy.max_weighted_median_px) * scale),
            geometry.weighted_q90_px / (float(policy.max_weighted_q90_px) * scale),
        ]
        reasons = ["weighted_median", "weighted_q90"]
    else:
        values = [
            geometry.unweighted_mean_px / (float(policy.max_weighted_median_px) * scale),
            geometry.unweighted_q90_px / (float(policy.max_weighted_q90_px) * scale),
        ]
        reasons = ["unweighted_mean", "unweighted_q90"]
    if policy.require_lower_evidence:
        values.extend(
            (
                geometry.lower_median_px / (float(policy.max_lower_median_px) * scale),
                geometry.lower_q90_px / (float(policy.max_lower_q90_px) * scale),
                geometry.top_endpoint_px / (float(policy.max_top_endpoint_px) * scale),
                geometry.bottom_endpoint_px / (float(policy.max_bottom_endpoint_px) * scale),
                geometry.range_start_gap / float(policy.max_range_start_gap),
                geometry.range_end_gap / float(policy.max_range_end_gap),
                geometry.lower_minus_upper_px / (float(policy.max_lower_minus_upper_px) * scale),
                geometry.polynomial_q90_px / (float(policy.max_polynomial_q90_px) * scale),
            )
        )
        reasons.extend(
            (
                "lower_median",
                "lower_q90",
                "top_endpoint",
                "bottom_endpoint",
                "range_start",
                "range_end",
                "lower_divergence",
                "polynomial_q90",
            )
        )
    index = int(np.argmax(np.asarray(values, dtype=np.float64)))
    distance = float(values[index])
    return distance, "compatible" if distance <= 1.0 else reasons[index]


def _pairwise_geometry(
    x: torch.Tensor,
    masks: torch.Tensor,
    y_fraction: torch.Tensor,
) -> list[list[PairGeometry | None]]:
    count = int(x.shape[0])
    polynomials: list[torch.Tensor | None] = []
    for candidate in range(count):
        valid = masks[candidate].bool() & torch.isfinite(x[candidate])
        y = y_fraction[valid].float()
        polynomials.append(
            _quadratic_fit(y, x[candidate, valid].float())
            if int(y.numel()) >= 3
            else None
        )
    matrix: list[list[PairGeometry | None]] = [[None for _ in range(count)] for _ in range(count)]
    for first, second in combinations(range(count), 2):
        value = _pair_geometry(
            x[first],
            masks[first],
            x[second],
            masks[second],
            y_fraction,
            polynomials[first],
            polynomials[second],
        )
        matrix[first][second] = value
        matrix[second][first] = value
    return matrix


def complete_link_clusters(
    candidate_ids: Iterable[int],
    pair_geometry: list[list[PairGeometry | None]],
    policy: GeometryClusterPolicy,
    input_w: int,
) -> tuple[list[list[int]], dict[tuple[int, int], tuple[float, str]]]:
    """Deterministic complete-link agglomeration.

    Complete link is deliberate: single-link connected components can bridge
    two physical lanes through a chain of ambiguous proposals near the horizon.
    """

    ids = sorted(int(value) for value in candidate_ids)
    pair_decisions: dict[tuple[int, int], tuple[float, str]] = {}
    for first, second in combinations(ids, 2):
        geometry = pair_geometry[first][second]
        if geometry is None:
            raise RuntimeError("missing proposal-pair geometry")
        pair_decisions[(first, second)] = _policy_distance(geometry, policy, input_w)
    clusters = [[value] for value in ids]
    while True:
        best: tuple[float, tuple[int, ...], int, int] | None = None
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                cross = [
                    pair_decisions[tuple(sorted((a, b)))][0]
                    for a in clusters[first]
                    for b in clusters[second]
                ]
                distance = max(cross, default=math.inf)
                members = tuple(sorted((*clusters[first], *clusters[second])))
                candidate = (distance, members, first, second)
                if distance <= 1.0 and (best is None or candidate < best):
                    best = candidate
        if best is None:
            break
        _distance, members, first, second = best
        clusters[first] = list(members)
        del clusters[second]
    clusters.sort(key=lambda values: (min(values), len(values)))
    return clusters, pair_decisions


def _cluster_medoid(
    cluster: list[int],
    pair_decisions: dict[tuple[int, int], tuple[float, str]],
    proposal_scores: torch.Tensor,
) -> int:
    if len(cluster) == 1:
        return int(cluster[0])
    candidates: list[tuple[float, float, int]] = []
    for member in cluster:
        distance = sum(
            pair_decisions[tuple(sorted((member, other)))][0]
            for other in cluster
            if other != member
        )
        candidates.append((distance, -float(proposal_scores[member]), int(member)))
    return min(candidates)[2]


def build_cluster_prototypes(
    stage: dict[str, torch.Tensor],
    clusters: list[list[int]],
    pair_decisions: dict[tuple[int, int], tuple[float, str]],
    *,
    input_h: int,
    input_w: int,
    min_valid_rows: int,
) -> dict[str, dict[str, Any]]:
    x, masks, _valid = candidate_row_masks(
        stage,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=0.0,
    )
    ranges = sort_range_norm(stage["range_norm"].float())
    scores = stage_scores(stage, quality_power=0.0)
    output: dict[str, dict[str, Any]] = {}
    medoids = [_cluster_medoid(cluster, pair_decisions, scores) for cluster in clusters]
    for mode in PROTOTYPE_MODES:
        prototype_x: list[torch.Tensor] = []
        prototype_range: list[torch.Tensor] = []
        prototype_score: list[float] = []
        for cluster, medoid in zip(clusters, medoids):
            if mode == "medoid":
                curve = x[medoid].clone()
                lane_range = ranges[medoid].clone()
            else:
                lane_range = ranges[cluster].median(dim=0).values
                lane_range = sort_range_norm(lane_range)
                rows: list[torch.Tensor] = []
                for row in range(int(x.shape[-1])):
                    visible_members = [member for member in cluster if bool(masks[member, row])]
                    if not visible_members:
                        rows.append(x[medoid, row])
                    else:
                        values = x[visible_members, row]
                        rows.append(values.median() if mode == "median" else values.mean())
                curve = torch.stack(rows).clamp(0.0, float(input_w - 1))
            prototype_x.append(curve)
            prototype_range.append(lane_range)
            prototype_score.append(max(float(scores[member]) for member in cluster))
        output[mode] = {
            "pred_x_rows": torch.stack(prototype_x) if prototype_x else x.new_empty((0, x.shape[-1])),
            "range_norm": torch.stack(prototype_range) if prototype_range else ranges.new_empty((0, 2)),
            "scores": torch.tensor(prototype_score, dtype=torch.float32),
            "medoid_ids": list(medoids),
        }
    return output


def _current_writer_slot_routes(stage: dict[str, torch.Tensor]) -> list[tuple[int, int]]:
    indices = stage.get("selection_slot_indices")
    active = stage.get("selection_slot_active")
    if not isinstance(indices, torch.Tensor):
        return []
    valid = indices >= 0
    if isinstance(active, torch.Tensor):
        valid &= active.bool()
    writer_valid = stage.get("selection_slot_official_candidate_valid")
    if isinstance(writer_valid, torch.Tensor) and tuple(writer_valid.shape) == tuple(valid.shape):
        valid &= writer_valid.bool()
    return [
        (int(slot), int(indices[slot]))
        for slot in torch.nonzero(valid, as_tuple=False).flatten().tolist()
    ]


def _current_writer_routes(stage: dict[str, torch.Tensor]) -> list[int]:
    return [route for _slot, route in _current_writer_slot_routes(stage)]


def _slot_cluster_mass_selection(
    clusters: list[list[int]],
    stage: dict[str, torch.Tensor],
    candidate_valid: torch.Tensor,
) -> dict[str, Any]:
    """Aggregate V7 proposal probability into GT-free geometry clusters.

    V7 performs uniqueness at proposal-ID level.  Near-duplicate proposals can
    therefore split a slot's probability mass and make a physical lane cluster
    look weaker than any single proposal ID.  This policy first sums the real
    conditional route probability inside each deterministic geometry cluster,
    then assigns active writer-valid slots to distinct clusters.  No GT, loss,
    optimizer, or learned clustering parameter is involved.
    """

    writer_slots = _current_writer_slot_routes(stage)
    logits = stage.get("selection_slot_logits")
    candidate_count = int(candidate_valid.numel())
    if not writer_slots:
        return {
            "cluster_ids": [],
            "slot_ids": [],
            "selected_masses": [],
            "available": True,
            "reason": "no_writer_slots",
        }
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        return {
            "cluster_ids": [],
            "slot_ids": [slot for slot, _route in writer_slots],
            "selected_masses": [],
            "available": False,
            "reason": "missing_slot_logits",
        }
    if int(logits.shape[-1]) < candidate_count or len(clusters) < len(writer_slots):
        return {
            "cluster_ids": [],
            "slot_ids": [slot for slot, _route in writer_slots],
            "selected_masses": [],
            "available": False,
            "reason": "insufficient_clusters_or_logits",
        }

    slot_ids = [slot for slot, _route in writer_slots]
    real_logits = logits[slot_ids, :candidate_count].float().clone()
    real_logits[:, ~candidate_valid.bool()] = -1.0e4
    probability = torch.softmax(real_logits, dim=-1)
    mass = torch.stack(
        [probability[:, cluster].sum(dim=-1) for cluster in clusters],
        dim=-1,
    )
    # The tiny deterministic column term only resolves exact floating-point
    # ties; it cannot change a non-tied assignment.
    cost = -mass.clamp_min(1.0e-12).log().cpu().numpy().astype(np.float64)
    cost += np.arange(len(clusters), dtype=np.float64)[None, :] * 1.0e-12
    row_indices, column_indices = linear_sum_assignment(cost)
    assigned = {int(row): int(column) for row, column in zip(row_indices, column_indices)}
    if len(assigned) != len(slot_ids):
        return {
            "cluster_ids": [],
            "slot_ids": slot_ids,
            "selected_masses": [],
            "available": False,
            "reason": "incomplete_linear_assignment",
        }
    cluster_ids = [assigned[row] for row in range(len(slot_ids))]
    selected_masses = [float(mass[row, cluster_ids[row]]) for row in range(len(slot_ids))]
    return {
        "cluster_ids": cluster_ids,
        "slot_ids": slot_ids,
        "selected_masses": selected_masses,
        "available": True,
        "reason": "ok",
    }


def _cluster_selection_ids(
    clusters: list[list[int]],
    scores: torch.Tensor,
    current_routes: list[int],
    count: int,
) -> dict[str, list[int]]:
    count = min(int(count), len(clusters))
    member_to_cluster = {
        int(member): int(cluster_id)
        for cluster_id, cluster in enumerate(clusters)
        for member in cluster
    }
    routed: list[int] = []
    for route in current_routes:
        cluster_id = member_to_cluster.get(int(route))
        if cluster_id is not None and cluster_id not in routed:
            routed.append(cluster_id)
    score_order = sorted(range(len(clusters)), key=lambda idx: (-float(scores[idx]), idx))
    for cluster_id in score_order:
        if len(routed) >= count:
            break
        if cluster_id not in routed:
            routed.append(cluster_id)
    return {
        "routed_consensus": routed[:count],
        "score_topk": score_order[:count],
    }


def _new_metric() -> dict[str, int]:
    return {
        "tp": 0,
        "predictions": 0,
        "gt": 0,
        "images": 0,
        "count_mismatch_images": 0,
    }


def _accumulate_metric(
    row: dict[str, int],
    tp: int,
    predictions: int,
    gt: int,
    *,
    expected_predictions: int | None = None,
) -> None:
    row["tp"] += int(tp)
    row["predictions"] += int(predictions)
    row["gt"] += int(gt)
    row["images"] += 1
    if expected_predictions is not None and int(predictions) != int(expected_predictions):
        row["count_mismatch_images"] += 1


def _finish_metric(row: dict[str, int]) -> dict[str, float | int]:
    tp = int(row["tp"])
    predictions = int(row["predictions"])
    gt = int(row["gt"])
    images = int(row["images"])
    mismatch = int(row["count_mismatch_images"])
    fp = predictions - tp
    fn = gt - tp
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predictions": predictions,
        "gt": gt,
        "precision": float(tp) / float(max(predictions, 1)),
        "recall": float(tp) / float(max(gt, 1)),
        "f1": 0.0 if denominator <= 0 else float(2 * tp) / float(denominator),
        "images": images,
        "count_mismatch_images": mismatch,
        "exact_count_parity": bool(mismatch == 0),
    }


def _proposal_labels(iou: torch.Tensor, candidate_valid: torch.Tensor, threshold: float) -> list[int]:
    if int(iou.shape[0]) == 0:
        return [-1 for _ in range(int(iou.shape[1]))]
    best, labels = iou.max(dim=0)
    return [
        int(labels[index])
        if bool(candidate_valid[index]) and float(best[index]) > float(threshold)
        else -1
        for index in range(int(iou.shape[1]))
    ]


def _evaluate_synthetic(
    item: dict[str, Any],
    line_width: float,
    min_valid_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    synthetic = {
        "meta": item["record"]["meta"],
        "stages": {"combined": item["combined_stage"]},
    }
    return official_proposal_gt_iou_matrix(
        synthetic,
        "combined",
        line_width=line_width,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=0.0,
    )


def _resolve_stage(cache: dict[str, Any], requested: str) -> str:
    names = sorted(
        {
            name
            for record in cache.get("records", [])
            for name in record.get("stages", {})
        }
    )
    if requested in names:
        return requested
    if requested == "main":
        for fallback in ("final", "stage2", "main"):
            if fallback in names:
                return fallback
    raise ValueError(f"stage {requested!r} is unavailable; choices={names}")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    max_batches = 0 if int(args.max_images) <= 0 else math.ceil(int(args.max_images) / int(args.eval_batch_size))
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
        desc=f"geometry clustering {args.split} cache",
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
    stage_name = _resolve_stage(cache, str(args.stage))
    input_h = int(cache["metadata"]["input_h"])
    input_w = int(cache["metadata"]["input_w"])
    thresholds = tuple(float(value) for value in args.iou_thresholds)

    prepared: list[dict[str, Any]] = []
    pair_distributions: dict[str, list[float]] = defaultdict(list)
    structural: dict[str, dict[str, Any]] = {
        policy.name: {
            "valid_candidates": 0,
            "clusters": 0,
            "multi_member_clusters": 0,
            "candidates_in_multi_member_clusters": 0,
            "same_gt_pairs": 0,
            "different_gt_pairs": 0,
            "same_cluster_same_gt_pairs": 0,
            "same_cluster_different_gt_pairs": 0,
            "clusters_with_labeled_candidates": 0,
            "catastrophic_multi_gt_clusters": 0,
            "representable_gt": 0,
            "gt_fragmentation": [],
            "largest_gt_cluster_fraction": [],
            "cluster_sizes": [],
            "pair_rejection_reasons": defaultdict(int),
        }
        for policy in POLICIES
    }
    bottom_guard = {
        "flat_compatible_pairs": 0,
        "removed_by_balanced_perspective": 0,
        "removed_same_gt_pairs": 0,
        "removed_different_gt_pairs": 0,
        "removed_unlabeled_pairs": 0,
        "reasons": defaultdict(int),
    }
    per_image: list[dict[str, Any]] = []

    for record in tqdm(cache["records"], ncols=90, desc=f"cluster construction {args.split}"):
        stage = record["stages"][stage_name]
        x, masks, candidate_valid = candidate_row_masks(
            stage,
            input_h=input_h,
            input_w=input_w,
            min_valid_rows=int(args.min_valid_rows),
            row_visibility_thresh=0.0,
        )
        y_fraction = fixed_row_fractions(int(x.shape[-1]), device=x.device, dtype=x.dtype)
        pair_geometry = _pairwise_geometry(x, masks, y_fraction)
        iou = stage["official_iou"].float()
        official_valid = stage["official_candidate_valid"].bool()
        candidate_valid &= official_valid
        labels = _proposal_labels(iou, candidate_valid, float(args.label_iou_threshold))
        valid_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten().tolist()
        for first, second in combinations(valid_ids, 2):
            geometry = pair_geometry[first][second]
            if geometry is None:
                continue
            if labels[first] >= 0 and labels[second] >= 0:
                prefix = "same_gt" if labels[first] == labels[second] else "different_gt"
                pair_distributions[f"{prefix}/lower_median_px"].append(geometry.lower_median_px)
                pair_distributions[f"{prefix}/top_endpoint_px"].append(geometry.top_endpoint_px)
                pair_distributions[f"{prefix}/bottom_endpoint_px"].append(geometry.bottom_endpoint_px)
                pair_distributions[f"{prefix}/range_start_gap"].append(geometry.range_start_gap)
                pair_distributions[f"{prefix}/range_end_gap"].append(geometry.range_end_gap)
                pair_distributions[f"{prefix}/lower_minus_upper_px"].append(geometry.lower_minus_upper_px)
                pair_distributions[f"{prefix}/weighted_q90_px"].append(geometry.weighted_q90_px)

        proposal_scores = stage_scores(stage, quality_power=0.0)
        policy_payload: dict[str, Any] = {}
        geometry: list[torch.Tensor] = []
        ranges: list[torch.Tensor] = []
        layout: dict[str, tuple[int, int]] = {}
        cursor = 0
        policy_decisions: dict[str, dict[tuple[int, int], tuple[float, str]]] = {}
        for policy in POLICIES:
            clusters, decisions = complete_link_clusters(
                valid_ids,
                pair_geometry,
                policy,
                input_w,
            )
            policy_decisions[policy.name] = decisions
            prototypes = build_cluster_prototypes(
                stage,
                clusters,
                decisions,
                input_h=input_h,
                input_w=input_w,
                min_valid_rows=int(args.min_valid_rows),
            )
            stats = structural[policy.name]
            stats["valid_candidates"] += len(valid_ids)
            stats["clusters"] += len(clusters)
            stats["multi_member_clusters"] += sum(len(cluster) > 1 for cluster in clusters)
            stats["candidates_in_multi_member_clusters"] += sum(
                len(cluster) for cluster in clusters if len(cluster) > 1
            )
            stats["cluster_sizes"].extend(len(cluster) for cluster in clusters)
            cluster_map = {
                member: cluster_id
                for cluster_id, cluster in enumerate(clusters)
                for member in cluster
            }
            for first, second in combinations(valid_ids, 2):
                if labels[first] < 0 or labels[second] < 0:
                    continue
                same_gt = labels[first] == labels[second]
                same_cluster = cluster_map[first] == cluster_map[second]
                stats["same_gt_pairs" if same_gt else "different_gt_pairs"] += 1
                if same_cluster:
                    stats[
                        "same_cluster_same_gt_pairs" if same_gt else "same_cluster_different_gt_pairs"
                    ] += 1
                _distance, reason = decisions[tuple(sorted((first, second)))]
                if reason != "compatible":
                    stats["pair_rejection_reasons"][reason] += 1
            for cluster in clusters:
                labeled = [labels[member] for member in cluster if labels[member] >= 0]
                if labeled:
                    stats["clusters_with_labeled_candidates"] += 1
                    if len(set(labeled)) > 1:
                        stats["catastrophic_multi_gt_clusters"] += 1
            for gt in range(int(iou.shape[0])):
                members = [idx for idx in valid_ids if labels[idx] == gt]
                if not members:
                    continue
                stats["representable_gt"] += 1
                fragments = {cluster_map[member] for member in members}
                stats["gt_fragmentation"].append(float(len(fragments)))
                largest = max(sum(cluster_map[member] == fragment for member in members) for fragment in fragments)
                stats["largest_gt_cluster_fraction"].append(float(largest) / float(len(members)))
            for _pair, (_distance, reason) in decisions.items():
                if reason != "compatible":
                    stats["pair_rejection_reasons"][reason] += 0

            for mode, prototype in prototypes.items():
                count = int(prototype["pred_x_rows"].shape[0])
                geometry.append(prototype["pred_x_rows"])
                ranges.append(prototype["range_norm"])
                layout[f"{policy.name}/{mode}"] = (cursor, cursor + count)
                cursor += count
            current_routes = _current_writer_routes(stage)
            policy_payload[policy.name] = {
                "clusters": clusters,
                "prototypes": prototypes,
                "current_routes": current_routes,
                "labels": labels,
            }

        flat = policy_decisions["flat_complete_48"]
        balanced = policy_decisions["perspective_balanced_48"]
        for pair, (flat_distance, _flat_reason) in flat.items():
            if flat_distance > 1.0:
                continue
            bottom_guard["flat_compatible_pairs"] += 1
            balanced_distance, balanced_reason = balanced[pair]
            if balanced_distance <= 1.0:
                continue
            bottom_guard["removed_by_balanced_perspective"] += 1
            bottom_guard["reasons"][balanced_reason] += 1
            first, second = pair
            if labels[first] < 0 or labels[second] < 0:
                bottom_guard["removed_unlabeled_pairs"] += 1
            elif labels[first] == labels[second]:
                bottom_guard["removed_same_gt_pairs"] += 1
            else:
                bottom_guard["removed_different_gt_pairs"] += 1

        combined_stage = {
            "pred_x_rows": torch.cat(geometry, dim=0) if geometry else x.new_empty((0, x.shape[-1])),
            "range_norm": torch.cat(ranges, dim=0) if ranges else stage["range_norm"].new_empty((0, 2)),
        }
        prepared.append(
            {
                "record": record,
                "combined_stage": combined_stage,
                "layout": layout,
                "policy_payload": policy_payload,
                "candidate_valid": candidate_valid,
                "proposal_iou": iou,
            }
        )

    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        if int(args.official_iou_workers) > 1:
            with ThreadPoolExecutor(max_workers=int(args.official_iou_workers)) as executor:
                evaluated = list(
                    tqdm(
                        executor.map(
                            lambda item: _evaluate_synthetic(
                                item, float(args.line_width), int(args.min_valid_rows)
                            ),
                            prepared,
                        ),
                        total=len(prepared),
                        ncols=90,
                        desc=f"prototype official IoU {args.split}",
                    )
                )
        else:
            evaluated = [
                _evaluate_synthetic(item, float(args.line_width), int(args.min_valid_rows))
                for item in tqdm(prepared, ncols=90, desc=f"prototype official IoU {args.split}")
            ]
    finally:
        cv2.setNumThreads(previous_threads)

    metric_rows: dict[str, dict[str, int]] = defaultdict(_new_metric)
    for item, (combined_iou, combined_valid) in zip(prepared, evaluated):
        record = item["record"]
        stage = record["stages"][stage_name]
        gt_count = int(item["proposal_iou"].shape[0])
        current_routes = _current_writer_routes(stage)
        current_hits: dict[str, int] = {}
        slot_iou = stage.get("selection_slot_official_iou")
        slot_valid = stage.get("selection_slot_official_candidate_valid")
        active = stage.get("selection_slot_active")
        if isinstance(slot_iou, torch.Tensor) and isinstance(slot_valid, torch.Tensor):
            selected_slots = slot_valid.bool()
            if isinstance(active, torch.Tensor):
                selected_slots &= active.bool()
            current_count = int(selected_slots.sum())
            for threshold in thresholds:
                assignment = evaluator_hungarian_assignment(
                    slot_iou[:, selected_slots],
                    range(current_count),
                    threshold,
                )
                current_hits[f"{threshold:.2f}"] = int(assignment.hit_count)
                _accumulate_metric(
                    metric_rows[f"current_v7_refined/{threshold:.2f}"],
                    assignment.hit_count,
                    current_count,
                    gt_count,
                    expected_predictions=current_count,
                )
        else:
            raise RuntimeError(
                "geometry clustering audit requires cached official V7 slot geometry"
            )

        current_count = min(current_count, int(args.top_k))
        for threshold in thresholds:
            all32 = cardinality_oracle_assignment(
                item["proposal_iou"],
                threshold,
                current_count,
                candidate_valid=item["candidate_valid"],
            )
            _accumulate_metric(
                metric_rows[f"all32_oracle_same_count/{threshold:.2f}"],
                all32.hit_count,
                min(current_count, int(item["candidate_valid"].sum())),
                gt_count,
            )

        image_row: dict[str, Any] = {
            "image_id": record["image_id"],
            "gt_count": gt_count,
            "current_writer_count": current_count,
            "current_routes": current_routes,
            "current_v7_refined": {
                f"hits_{threshold:.2f}": current_hits[f"{threshold:.2f}"]
                for threshold in thresholds
            },
            "policies": {},
        }
        for policy in POLICIES:
            payload = item["policy_payload"][policy.name]
            clusters = payload["clusters"]
            image_policy = {"clusters": []}
            labels = payload["labels"]
            for cluster_id, cluster in enumerate(clusters):
                image_policy["clusters"].append(
                    {
                        "cluster_id": cluster_id,
                        "members": cluster,
                        "member_gt_labels": [labels[member] for member in cluster],
                    }
                )
            image_policy["prototypes"] = {}
            for mode in PROTOTYPE_MODES:
                start, stop = item["layout"][f"{policy.name}/{mode}"]
                local_iou = combined_iou[:, start:stop]
                local_valid = combined_valid[start:stop].bool()
                prototype = payload["prototypes"][mode]
                selections = _cluster_selection_ids(
                    clusters,
                    prototype["scores"],
                    current_routes,
                    current_count,
                )
                mass_selection = _slot_cluster_mass_selection(
                    clusters,
                    stage,
                    item["candidate_valid"],
                )
                selections["slot_cluster_mass_unique"] = list(
                    mass_selection["cluster_ids"]
                )
                image_mode: dict[str, Any] = {
                    "medoid_ids": prototype["medoid_ids"],
                    "routed_consensus_cluster_ids": selections["routed_consensus"],
                    "score_topk_cluster_ids": selections["score_topk"],
                    "slot_cluster_mass_unique_cluster_ids": selections[
                        "slot_cluster_mass_unique"
                    ],
                    "slot_cluster_mass_unique_slot_ids": mass_selection["slot_ids"],
                    "slot_cluster_mass_unique_selected_masses": mass_selection[
                        "selected_masses"
                    ],
                    "slot_cluster_mass_unique_available": mass_selection["available"],
                    "slot_cluster_mass_unique_reason": mass_selection["reason"],
                }
                for threshold in thresholds:
                    for selection_name, selected_ids in selections.items():
                        selected_ids = [idx for idx in selected_ids if bool(local_valid[idx])]
                        assignment = evaluator_hungarian_assignment(
                            local_iou,
                            selected_ids,
                            threshold,
                        )
                        _accumulate_metric(
                            metric_rows[
                                f"{policy.name}/{mode}/{selection_name}/{threshold:.2f}"
                            ],
                            assignment.hit_count,
                            len(selected_ids),
                            gt_count,
                            expected_predictions=current_count,
                        )
                        image_mode[
                            f"{selection_name}_hits_{threshold:.2f}"
                        ] = int(assignment.hit_count)
                        image_mode[
                            f"{selection_name}_predictions_{threshold:.2f}"
                        ] = int(len(selected_ids))
                    oracle = cardinality_oracle_assignment(
                        local_iou,
                        threshold,
                        current_count,
                        candidate_valid=local_valid,
                    )
                    _accumulate_metric(
                        metric_rows[
                            f"{policy.name}/{mode}/oracle_same_count/{threshold:.2f}"
                        ],
                        oracle.hit_count,
                        min(current_count, int(local_valid.sum())),
                        gt_count,
                    )
                    image_mode[f"oracle_hits_{threshold:.2f}"] = oracle.hit_count
                image_policy["prototypes"][mode] = image_mode
            image_row["policies"][policy.name] = image_policy
        per_image.append(image_row)

    structural_out: dict[str, Any] = {}
    for policy in POLICIES:
        raw = structural[policy.name]
        tp = int(raw["same_cluster_same_gt_pairs"])
        fp = int(raw["same_cluster_different_gt_pairs"])
        positives = int(raw["same_gt_pairs"])
        structural_out[policy.name] = {
            **{
                key: value
                for key, value in raw.items()
                if key not in {
                    "gt_fragmentation", "largest_gt_cluster_fraction", "cluster_sizes",
                    "pair_rejection_reasons",
                }
            },
            "pair_merge_precision": float(tp) / float(max(tp + fp, 1)),
            "same_gt_pair_recall": float(tp) / float(max(positives, 1)),
            "catastrophic_cluster_fraction": float(raw["catastrophic_multi_gt_clusters"])
            / float(max(raw["clusters_with_labeled_candidates"], 1)),
            "candidate_multi_member_fraction": float(raw["candidates_in_multi_member_clusters"])
            / float(max(raw["valid_candidates"], 1)),
            "cluster_size": _summary(raw["cluster_sizes"]),
            "gt_fragmentation": _summary(raw["gt_fragmentation"]),
            "largest_gt_cluster_fraction": _summary(raw["largest_gt_cluster_fraction"]),
            "pair_rejection_reasons": dict(sorted(raw["pair_rejection_reasons"].items())),
        }

    metrics = {key: _finish_metric(row) for key, row in sorted(metric_rows.items())}
    for policy in POLICIES:
        for mode in PROTOTYPE_MODES:
            for threshold in thresholds:
                oracle_key = f"{policy.name}/{mode}/oracle_same_count/{threshold:.2f}"
                all32_key = f"all32_oracle_same_count/{threshold:.2f}"
                if oracle_key in metrics and all32_key in metrics:
                    metrics[oracle_key]["all32_tp_retention"] = float(metrics[oracle_key]["tp"]) / float(
                        max(int(metrics[all32_key]["tp"]), 1)
                    )

    result = {
        "experiment": "training-free perspective-aware proposal geometry clustering",
        "scope": {
            "gt_used_in_clustering": False,
            "gt_used_for_posthoc_audit_only": True,
            "optimizer_steps": 0,
            "backward_performed": False,
            "checkpoint_weights_changed": False,
            "checkpoint_selection_performed": False,
            "threshold_search_performed": False,
            "test_set_used": bool(args.split == "test"),
            "new_model_version": False,
        },
        "primary_confirmatory_contract": {
            "policy": PRIMARY_POLICY,
            "prototype": PRIMARY_PROTOTYPE,
            "selection": PRIMARY_SELECTION,
            "prediction_count": "exact_current_v7_writer_count",
            "cluster_gt_free": True,
            "slot_probability_aggregated_within_cluster": True,
        },
        "metadata": metadata_for_json(
            cache,
            resolved_stage=stage_name,
            records_analyzed=len(cache["records"]),
            line_width=float(args.line_width),
            min_valid_rows=int(args.min_valid_rows),
            label_iou_threshold=float(args.label_iou_threshold),
            iou_thresholds=list(thresholds),
            top_k=int(args.top_k),
        ),
        "policy_contracts": [
            {
                key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in asdict(policy).items()
            }
            for policy in POLICIES
        ],
        "perspective_pair_evidence": {
            key: _summary(values) for key, values in sorted(pair_distributions.items())
        },
        "bottom_guard_ablation": {
            **{key: value for key, value in bottom_guard.items() if key != "reasons"},
            "reasons": dict(sorted(bottom_guard["reasons"].items())),
            "different_gt_precision_among_labeled_removed": float(
                bottom_guard["removed_different_gt_pairs"]
            )
            / float(
                max(
                    bottom_guard["removed_different_gt_pairs"]
                    + bottom_guard["removed_same_gt_pairs"],
                    1,
                )
            ),
        },
        "structural_clustering": structural_out,
        "official_metrics": metrics,
        "per_image": per_image,
    }
    return result


def _metric_value(report: dict[str, Any], key: str, field: str) -> str:
    row = report.get("official_metrics", {}).get(key)
    if not isinstance(row, dict):
        return "-"
    value = row.get(field)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Perspective-aware proposal geometry clustering audit",
        "",
        (
            "This is a training-free diagnostic. Clusters are built without GT; "
            "GT is used only afterward for purity and official-IoU capacity measurement."
        ),
        "",
        "## Structural result",
        "",
        (
            "| Policy | Pair precision | Same-lane pair recall | Catastrophic "
            "cluster fraction | Candidates in multi-member groups | Median clusters/size |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        row = report["structural_clustering"][policy.name]
        lines.append(
            f"| `{policy.name}` | {row['pair_merge_precision']:.4f} | "
            f"{row['same_gt_pair_recall']:.4f} | {row['catastrophic_cluster_fraction']:.4f} | "
            f"{row['candidate_multi_member_fraction']:.4f} | {row['cluster_size']['median']:.2f} |"
        )
    bottom = report["bottom_guard_ablation"]
    lines.extend(
        [
            "",
            "## Bottom-separation ablation",
            "",
            f"Flat-compatible pairs: **{bottom['flat_compatible_pairs']}**  ",
            f"Removed by the balanced perspective guard: **{bottom['removed_by_balanced_perspective']}**  ",
            (
                "Removed different-GT / same-GT / unlabeled: "
                f"**{bottom['removed_different_gt_pairs']} / "
                f"{bottom['removed_same_gt_pairs']} / "
                f"{bottom['removed_unlabeled_pairs']}**  "
            ),
            (
                "Different-GT precision among labeled removed pairs: "
                f"**{bottom['different_gt_precision_among_labeled_removed']:.4f}**"
            ),
            "",
            "## Same-count official prototype capacity",
            "",
            (
                "| Policy / prototype | Oracle TP@.50 | Retention@.50 | Oracle "
                "TP@.75 | Retention@.75 | Cluster-mass F1@.50 | Cluster-mass "
                "F1@.75 | Routed F1@.50 | Routed F1@.75 |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in POLICIES:
        for mode in PROTOTYPE_MODES:
            prefix = f"{policy.name}/{mode}"
            lines.append(
                f"| `{policy.name}/{mode}` | "
                f"{_metric_value(report, prefix + '/oracle_same_count/0.50', 'tp')} | "
                f"{_metric_value(report, prefix + '/oracle_same_count/0.50', 'all32_tp_retention')} | "
                f"{_metric_value(report, prefix + '/oracle_same_count/0.75', 'tp')} | "
                f"{_metric_value(report, prefix + '/oracle_same_count/0.75', 'all32_tp_retention')} | "
                f"{_metric_value(report, prefix + '/slot_cluster_mass_unique/0.50', 'f1')} | "
                f"{_metric_value(report, prefix + '/slot_cluster_mass_unique/0.75', 'f1')} | "
                f"{_metric_value(report, prefix + '/routed_consensus/0.50', 'f1')} | "
                f"{_metric_value(report, prefix + '/routed_consensus/0.75', 'f1')} |"
            )
    lines.extend(
        [
            "",
            "## Reference metrics",
            "",
            f"Current V7 refined F1@.50: **{_metric_value(report, 'current_v7_refined/0.50', 'f1')}**  ",
            f"Current V7 refined F1@.75: **{_metric_value(report, 'current_v7_refined/0.75', 'f1')}**  ",
            (
                "Primary bottom-aware cluster-mass medoid F1@.50: **"
                f"{_metric_value(report, PRIMARY_POLICY + '/' + PRIMARY_PROTOTYPE + '/' + PRIMARY_SELECTION + '/0.50', 'f1')}**  "
            ),
            (
                "Primary bottom-aware cluster-mass medoid F1@.75: **"
                f"{_metric_value(report, PRIMARY_POLICY + '/' + PRIMARY_PROTOTYPE + '/' + PRIMARY_SELECTION + '/0.75', 'f1')}**  "
            ),
            f"All-32 same-count oracle TP@.50: **{_metric_value(report, 'all32_oracle_same_count/0.50', 'tp')}**  ",
            f"All-32 same-count oracle TP@.75: **{_metric_value(report, 'all32_oracle_same_count/0.75', 'tp')}**",
            "",
            "No architecture or long-training decision is authorized by this report alone.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise ValueError("test split is closed for this diagnostic")
    report = run_audit(args)
    write_json(args.output_json, report)
    write_markdown(args.output_md, report)
    print(json.dumps({
        "output_json": str(Path(args.output_json).resolve()),
        "output_md": str(Path(args.output_md).resolve()),
        "records": report["metadata"]["records_analyzed"],
        "test_set_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
