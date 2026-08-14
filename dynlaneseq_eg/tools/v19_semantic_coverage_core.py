from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    evaluator_hungarian_assignment,
)


@dataclass(frozen=True)
class RouteEvaluation:
    routes: tuple[int, ...]
    active_slots: tuple[int, ...]
    prediction_count: int
    gt_count: int
    hit_count: int
    iou_sum: float
    covered_gt: tuple[int, ...]
    slot_gt_pairs: tuple[tuple[int, int], ...]
    semantic_collision_excess: int


@dataclass(frozen=True)
class ExactInjectiveResult:
    routes: tuple[int, ...]
    hit_count: int
    qualified_iou_sum: float
    official_total_iou: float
    assignments_evaluated: int


@dataclass(frozen=True)
class JointInjectiveResult:
    routes: tuple[int, ...]
    hit_count_50: int
    hit_count_75: int
    official_total_iou: float
    edit_count: int
    assignments_evaluated: int


@dataclass(frozen=True)
class JointInjectiveSearch:
    best: JointInjectiveResult
    by_exact_edits: dict[int, JointInjectiveResult]
    by_max_edits: dict[int, JointInjectiveResult]


def select_unique_by_score(
    score: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Select one distinct proposal ID for every active slot."""

    if score.ndim != 2:
        raise ValueError("score must have shape [S,N]")
    if valid.shape != score.shape:
        raise ValueError("valid must have shape [S,N]")
    if active.shape != score.shape[:1]:
        raise ValueError("active must have shape [S]")
    slots, candidates = score.shape
    result = torch.full((slots,), -1, dtype=torch.long)
    active_slots = torch.nonzero(active.bool(), as_tuple=False).flatten()
    if active_slots.numel() == 0:
        return result
    if int(active_slots.numel()) > int(candidates):
        raise ValueError("more active slots than proposal candidates")

    selected_score = score[active_slots].detach().float().cpu().numpy()
    selected_valid = valid[active_slots].detach().bool().cpu().numpy()
    if np.any(selected_valid.sum(axis=1) == 0):
        raise ValueError("an active slot has no valid proposal candidate")
    cost = -selected_score.astype(np.float64, copy=True)
    cost[~selected_valid] = 1.0e12
    row_ids, candidate_ids = linear_sum_assignment(cost)
    if len(row_ids) != int(active_slots.numel()):
        raise RuntimeError("unique proposal assignment is incomplete")
    for row, candidate in zip(row_ids.tolist(), candidate_ids.tolist()):
        if not bool(selected_valid[row, candidate]):
            raise RuntimeError("unique proposal assignment selected an invalid edge")
        result[int(active_slots[row])] = int(candidate)
    return result


def evaluate_routes(
    quality: torch.Tensor,
    valid: torch.Tensor,
    routes: torch.Tensor,
    active: torch.Tensor,
    *,
    threshold: float,
) -> RouteEvaluation:
    """Evaluate slot-specific counterfactual routes with official matching."""

    if quality.ndim != 3:
        raise ValueError("quality must have shape [G,S,N]")
    gt_count, slots, candidates = quality.shape
    if valid.shape != (slots, candidates):
        raise ValueError("valid must have shape [S,N]")
    if routes.shape != (slots,) or active.shape != (slots,):
        raise ValueError("routes and active must have shape [S]")

    selected_slots: list[int] = []
    selected_candidates: list[int] = []
    for slot in torch.nonzero(active.bool(), as_tuple=False).flatten().tolist():
        candidate = int(routes[slot])
        if 0 <= candidate < candidates and bool(valid[slot, candidate]):
            selected_slots.append(int(slot))
            selected_candidates.append(candidate)

    if selected_slots:
        matrix = torch.stack(
            [
                quality[:, slot, candidate]
                for slot, candidate in zip(selected_slots, selected_candidates)
            ],
            dim=1,
        )
        assignment = evaluator_hungarian_assignment(
            matrix,
            range(len(selected_slots)),
            threshold=float(threshold),
        )
        slot_gt_pairs = tuple(
            (selected_slots[local_prediction], int(gt))
            for gt, local_prediction in assignment.pairs
        )
        hit_count = int(assignment.hit_count)
        iou_sum = float(assignment.iou_sum)
    else:
        slot_gt_pairs = ()
        hit_count = 0
        iou_sum = 0.0

    best_gt: list[int] = []
    for slot, candidate in zip(selected_slots, selected_candidates):
        if gt_count == 0:
            continue
        column = quality[:, slot, candidate]
        value, gt = column.max(dim=0)
        if float(value) > float(threshold):
            best_gt.append(int(gt))
    collision_excess = len(best_gt) - len(set(best_gt))
    full_routes = tuple(int(value) for value in routes.tolist())
    return RouteEvaluation(
        routes=full_routes,
        active_slots=tuple(selected_slots),
        prediction_count=len(selected_slots),
        gt_count=int(gt_count),
        hit_count=hit_count,
        iou_sum=iou_sum,
        covered_gt=tuple(sorted(gt for _slot, gt in slot_gt_pairs)),
        slot_gt_pairs=slot_gt_pairs,
        semantic_collision_excess=int(collision_excess),
    )


def source_slot_gt_ownership(
    quality: torch.Tensor,
    valid: torch.Tensor,
    routes: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Assign source active slots injectively to GT by maximum total IoU."""

    if quality.ndim != 3:
        raise ValueError("quality must have shape [G,S,N]")
    gt_count, slots, candidates = quality.shape
    if valid.shape != (slots, candidates):
        raise ValueError("valid must have shape [S,N]")
    if routes.shape != (slots,) or active.shape != (slots,):
        raise ValueError("routes and active must have shape [S]")
    ownership = torch.full((slots,), -1, dtype=torch.long)
    active_slots: list[int] = []
    selected: list[torch.Tensor] = []
    for slot in torch.nonzero(active.bool(), as_tuple=False).flatten().tolist():
        candidate = int(routes[slot])
        if 0 <= candidate < candidates and bool(valid[slot, candidate]):
            active_slots.append(int(slot))
            selected.append(quality[:, slot, candidate])
    if gt_count == 0 or not selected:
        return ownership
    matrix = torch.stack(selected, dim=1).detach().float().cpu().numpy()
    gt_ids, local_slot_ids = linear_sum_assignment(1.0 - matrix)
    for gt, local_slot in zip(gt_ids.tolist(), local_slot_ids.tolist()):
        ownership[active_slots[local_slot]] = int(gt)
    return ownership


def select_source_slot_conditioned_routes(
    quality: torch.Tensor,
    valid: torch.Tensor,
    source_routes: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the best candidate for each source-owned GT lane.

    Active source slots are first assigned injectively to GT lanes.  Each
    owned slot is then scored only against that fixed GT.  If there are more
    active slots than GT lanes, unowned slots are forced to preserve their
    source route.  Proposal-ID uniqueness remains enforced globally.
    """

    ownership = source_slot_gt_ownership(
        quality, valid, source_routes, active
    )
    gt_count, slots, candidates = quality.shape
    score = torch.zeros((slots, candidates), dtype=torch.float32)
    allowed = valid.detach().bool().cpu().clone()
    quality_cpu = quality.detach().float().cpu()
    source_cpu = source_routes.detach().long().cpu()
    active_cpu = active.detach().bool().cpu()
    for slot in range(slots):
        if not bool(active_cpu[slot]):
            continue
        gt = int(ownership[slot])
        if 0 <= gt < gt_count:
            score[slot] = quality_cpu[gt, slot]
            continue
        candidate = int(source_cpu[slot])
        if 0 <= candidate < candidates and bool(allowed[slot, candidate]):
            allowed[slot].fill_(False)
            allowed[slot, candidate] = True
    routes = select_unique_by_score(score, allowed, active_cpu)
    return routes, ownership


@lru_cache(maxsize=None)
def _unique_candidate_tuples(candidates: int, count: int) -> torch.Tensor:
    if count < 0 or candidates < 0 or count > candidates:
        raise ValueError("invalid distinct candidate assignment dimensions")
    if count == 0:
        return torch.empty((1, 0), dtype=torch.long)
    axis = torch.arange(candidates, dtype=torch.long)
    if count == 1:
        return axis.unsqueeze(1)
    grid = torch.meshgrid(*([axis] * count), indexing="ij")
    values = torch.stack([item.reshape(-1) for item in grid], dim=1)
    distinct = torch.ones(values.shape[0], dtype=torch.bool)
    for left in range(count):
        for right in range(left + 1, count):
            distinct &= values[:, left] != values[:, right]
    return values[distinct].contiguous()


@lru_cache(maxsize=None)
def _official_matching_maps(
    gt_count: int,
    prediction_count: int,
) -> tuple[str, torch.Tensor]:
    """Enumerate every matching considered by a tiny Hungarian problem."""

    if gt_count <= 0 or prediction_count <= 0:
        return "empty", torch.empty((1, 0), dtype=torch.long)
    if gt_count <= prediction_count:
        maps = tuple(permutations(range(prediction_count), gt_count))
        return "gt_to_prediction", torch.tensor(maps, dtype=torch.long)
    maps = tuple(permutations(range(gt_count), prediction_count))
    return "prediction_to_gt", torch.tensor(maps, dtype=torch.long)


def _official_matched_values(quality_sets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact max-total-IoU matching for a batch of prediction sets."""

    if quality_sets.ndim != 3:
        raise ValueError("quality_sets must have shape [M,G,P]")
    batch, gt_count, prediction_count = quality_sets.shape
    if gt_count == 0 or prediction_count == 0:
        empty = quality_sets.new_zeros((batch, 0))
        return empty, quality_sets.new_zeros((batch,))

    mode, maps = _official_matching_maps(gt_count, prediction_count)
    maps = maps.to(quality_sets.device)
    if mode == "gt_to_prediction":
        gt_index = torch.arange(gt_count, device=quality_sets.device).view(1, gt_count)
        values = quality_sets[:, gt_index, maps]
    elif mode == "prediction_to_gt":
        prediction_index = torch.arange(
            prediction_count, device=quality_sets.device
        ).view(1, prediction_count)
        values = quality_sets[:, maps, prediction_index]
    else:
        raise RuntimeError("unexpected empty matching mode")
    totals = values.sum(dim=-1)
    best_map = totals.argmax(dim=-1)
    row = torch.arange(batch, device=quality_sets.device)
    matched = values[row, best_map]
    return matched, totals[row, best_map]


def exact_injective_routes(
    quality: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    *,
    thresholds: Iterable[float] = (0.50, 0.75),
    device: str | torch.device = "cpu",
    chunk_size: int = 262_144,
) -> dict[float, ExactInjectiveResult]:
    """Exhaustively search all distinct slot/proposal assignments.

    Every candidate tuple is evaluated with the same maximum-total-IoU
    Hungarian rule used by the official CULane evaluator.  The oracle then
    maximizes threshold TP cardinality, qualified IoU and total matched IoU,
    in that order.  It is therefore an exact selection upper bound for the
    frozen counterfactual population; no greedy prefix is used.
    """

    if quality.ndim != 3:
        raise ValueError("quality must have shape [G,S,N]")
    gt_count, slots, candidates = quality.shape
    if valid.shape != (slots, candidates):
        raise ValueError("valid must have shape [S,N]")
    if active.shape != (slots,):
        raise ValueError("active must have shape [S]")
    active_slots = torch.nonzero(active.bool(), as_tuple=False).flatten().tolist()
    active_count = len(active_slots)
    threshold_values = tuple(float(value) for value in thresholds)
    if active_count == 0:
        empty_route = tuple(-1 for _ in range(slots))
        return {
            threshold: ExactInjectiveResult(empty_route, 0, 0.0, 0.0, 1)
            for threshold in threshold_values
        }

    tuples = _unique_candidate_tuples(int(candidates), active_count)
    oracle_device = torch.device(device)
    quality_device = quality.detach().float().to(oracle_device)
    valid_cpu = valid.detach().bool().cpu()
    best: dict[float, dict[str, object]] = {
        threshold: {
            "hits": -1,
            "qualified": -1.0,
            "total": -1.0,
            "route": None,
        }
        for threshold in threshold_values
    }
    evaluated = 0
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, int(tuples.shape[0]), chunk_size):
        candidate_tuple = tuples[start : start + chunk_size]
        usable = torch.ones(candidate_tuple.shape[0], dtype=torch.bool)
        for column, slot in enumerate(active_slots):
            usable &= valid_cpu[slot, candidate_tuple[:, column]]
        candidate_tuple = candidate_tuple[usable]
        if candidate_tuple.numel() == 0:
            continue
        evaluated += int(candidate_tuple.shape[0])
        candidate_device = candidate_tuple.to(oracle_device)
        quality_sets = torch.stack(
            [
                quality_device[:, slot, candidate_device[:, column]].transpose(0, 1)
                for column, slot in enumerate(active_slots)
            ],
            dim=-1,
        )
        matched, official_total = _official_matched_values(quality_sets)
        for threshold in threshold_values:
            qualified_mask = matched > threshold
            hits = qualified_mask.sum(dim=-1)
            qualified = torch.where(
                qualified_mask, matched, torch.zeros_like(matched)
            ).sum(dim=-1)
            max_hits = int(hits.max())
            eligible = hits == max_hits
            max_qualified = float(qualified[eligible].max())
            eligible &= qualified >= max_qualified - 1.0e-12
            max_total = float(official_total[eligible].max())
            eligible &= official_total >= max_total - 1.0e-12
            local = int(torch.nonzero(eligible, as_tuple=False)[0])
            current = best[threshold]
            key = (max_hits, max_qualified, max_total)
            current_key = (
                int(current["hits"]),
                float(current["qualified"]),
                float(current["total"]),
            )
            if key > current_key:
                full_route = [-1 for _ in range(slots)]
                selected = candidate_tuple[local].tolist()
                for slot, candidate in zip(active_slots, selected):
                    full_route[slot] = int(candidate)
                current.update(
                    {
                        "hits": max_hits,
                        "qualified": max_qualified,
                        "total": max_total,
                        "route": tuple(full_route),
                    }
                )

    if evaluated == 0:
        raise RuntimeError("no valid distinct proposal assignment exists")
    result: dict[float, ExactInjectiveResult] = {}
    for threshold, value in best.items():
        route = value["route"]
        if not isinstance(route, tuple):
            raise RuntimeError("injective oracle failed to produce a route")
        result[threshold] = ExactInjectiveResult(
            routes=route,
            hit_count=int(value["hits"]),
            qualified_iou_sum=float(value["qualified"]),
            official_total_iou=float(value["total"]),
            assignments_evaluated=evaluated,
        )
    return result


def _joint_key(result: JointInjectiveResult) -> tuple[int, int, float, int]:
    return (
        int(result.hit_count_50),
        int(result.hit_count_75),
        float(result.official_total_iou),
        -int(result.edit_count),
    )


def exact_joint_injective_routes(
    quality: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    source_routes: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 262_144,
) -> JointInjectiveSearch:
    """Search the joint .50 -> .75 -> IoU -> minimum-edit objective.

    The same route is evaluated at both thresholds.  All distinct proposal-ID
    assignments are searched exactly.  Results are returned for each exact
    active-route edit count and for every cumulative maximum-edit budget.
    """

    if quality.ndim != 3:
        raise ValueError("quality must have shape [G,S,N]")
    gt_count, slots, candidates = quality.shape
    if valid.shape != (slots, candidates):
        raise ValueError("valid must have shape [S,N]")
    if active.shape != (slots,) or source_routes.shape != (slots,):
        raise ValueError("active and source_routes must have shape [S]")
    active_slots = torch.nonzero(active.bool(), as_tuple=False).flatten().tolist()
    active_count = len(active_slots)
    if active_count == 0:
        empty = JointInjectiveResult(
            tuple(-1 for _ in range(slots)), 0, 0, 0.0, 0, 1
        )
        return JointInjectiveSearch(empty, {0: empty}, {0: empty})

    tuples = _unique_candidate_tuples(int(candidates), active_count)
    oracle_device = torch.device(device)
    quality_device = quality.detach().float().to(oracle_device)
    valid_cpu = valid.detach().bool().cpu()
    source_active = source_routes.detach().long().cpu()[active_slots]
    best_exact: dict[int, JointInjectiveResult] = {}
    evaluated = 0
    chunk_size = max(int(chunk_size), 1)

    for start in range(0, int(tuples.shape[0]), chunk_size):
        candidate_tuple = tuples[start : start + chunk_size]
        usable = torch.ones(candidate_tuple.shape[0], dtype=torch.bool)
        for column, slot in enumerate(active_slots):
            usable &= valid_cpu[slot, candidate_tuple[:, column]]
        candidate_tuple = candidate_tuple[usable]
        if candidate_tuple.numel() == 0:
            continue
        evaluated += int(candidate_tuple.shape[0])
        edit_count = (candidate_tuple != source_active.view(1, -1)).sum(dim=1)
        candidate_device = candidate_tuple.to(oracle_device)
        quality_sets = torch.stack(
            [
                quality_device[:, slot, candidate_device[:, column]].transpose(0, 1)
                for column, slot in enumerate(active_slots)
            ],
            dim=-1,
        )
        matched, official_total = _official_matched_values(quality_sets)
        hits_50 = (matched > 0.50).sum(dim=-1).cpu()
        hits_75 = (matched > 0.75).sum(dim=-1).cpu()
        official_total_cpu = official_total.detach().float().cpu()

        for edits in torch.unique(edit_count).tolist():
            mask = edit_count == int(edits)
            local_50 = hits_50[mask]
            max_50 = int(local_50.max())
            eligible = mask & (hits_50 == max_50)
            max_75 = int(hits_75[eligible].max())
            eligible &= hits_75 == max_75
            max_total = float(official_total_cpu[eligible].max())
            eligible &= official_total_cpu >= max_total - 1.0e-12
            local = int(torch.nonzero(eligible, as_tuple=False)[0])
            full_route = [-1 for _ in range(slots)]
            for slot, candidate in zip(
                active_slots, candidate_tuple[local].tolist()
            ):
                full_route[slot] = int(candidate)
            candidate_result = JointInjectiveResult(
                routes=tuple(full_route),
                hit_count_50=max_50,
                hit_count_75=max_75,
                official_total_iou=max_total,
                edit_count=int(edits),
                assignments_evaluated=0,
            )
            previous = best_exact.get(int(edits))
            if previous is None or _joint_key(candidate_result) > _joint_key(previous):
                best_exact[int(edits)] = candidate_result

    if evaluated == 0 or not best_exact:
        raise RuntimeError("no valid distinct proposal assignment exists")
    best_exact = {
        edits: JointInjectiveResult(
            routes=result.routes,
            hit_count_50=result.hit_count_50,
            hit_count_75=result.hit_count_75,
            official_total_iou=result.official_total_iou,
            edit_count=result.edit_count,
            assignments_evaluated=evaluated,
        )
        for edits, result in best_exact.items()
    }
    by_max_edits: dict[int, JointInjectiveResult] = {}
    running: JointInjectiveResult | None = None
    for budget in range(active_count + 1):
        candidate = best_exact.get(budget)
        if candidate is not None and (
            running is None or _joint_key(candidate) > _joint_key(running)
        ):
            running = candidate
        if running is None:
            raise RuntimeError(
                f"no valid assignment exists for edit budget <= {budget}"
            )
        by_max_edits[budget] = running
    best = by_max_edits[active_count]
    return JointInjectiveSearch(
        best=best,
        by_exact_edits=best_exact,
        by_max_edits=by_max_edits,
    )
