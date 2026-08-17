from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


SLOTS = 4
ALTERNATIVES = 3
ACTIONS = SLOTS * ALTERNATIVES
THRESHOLDS = (0.50, 0.75)
OUTCOME_NEUTRAL = 0
OUTCOME_BENEFICIAL = 1
OUTCOME_HARMFUL = 2


@dataclass(frozen=True)
class AssignmentSummary:
    tp50: int
    tp75: int
    matched_iou: float
    gt50: frozenset[int]
    gt75: frozenset[int]


def _assignment(matrix: np.ndarray) -> AssignmentSummary:
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return AssignmentSummary(0, 0, 0.0, frozenset(), frozenset())
    prediction_ids, gt_ids = linear_sum_assignment(1.0 - matrix)
    matched = matrix[prediction_ids, gt_ids]
    return AssignmentSummary(
        tp50=int((matched > THRESHOLDS[0]).sum()),
        tp75=int((matched > THRESHOLDS[1]).sum()),
        matched_iou=float(matched.sum()),
        gt50=frozenset(int(gt) for gt, value in zip(gt_ids, matched) if value > THRESHOLDS[0]),
        gt75=frozenset(int(gt) for gt, value in zip(gt_ids, matched) if value > THRESHOLDS[1]),
    )


def _objective(summary: AssignmentSummary, *, edit: bool, path_rank: int) -> tuple:
    # Official threshold crossings dominate. An unchanged official outcome
    # always keeps exact V7, irrespective of small IoU changes.
    return (
        int(summary.tp50),
        int(summary.tp75),
        -int(edit),
        float(summary.matched_iou),
        -int(path_rank),
    )


def score_single_edit_iou_bank(
    iou_bank: np.ndarray,
    *,
    source_active: np.ndarray,
    candidate_valid: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    """Exact KEEP-or-one-edit labels from immutable path/GT IoUs.

    ``iou_bank`` is ``[4,4,G]``: member zero is exact V7 and members 1..3
    are G0 paths. No curve is averaged or modified in this routine.
    """

    iou = np.asarray(iou_bank, dtype=np.float32)
    active = np.asarray(source_active, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    if iou.ndim != 3 or iou.shape[:2] != (SLOTS, 1 + ALTERNATIVES):
        raise ValueError("iou_bank must have shape [4,4,G]")
    if active.shape != (SLOTS,) or valid.shape != (SLOTS, ALTERNATIVES):
        raise ValueError("single-edit validity shapes are invalid")

    def evaluate(replacement: tuple[int, int] | None) -> AssignmentSummary:
        rows: list[np.ndarray] = []
        for slot in range(SLOTS):
            if not active[slot]:
                continue
            member = 0
            if replacement is not None and replacement[0] == slot:
                member = 1 + replacement[1]
            rows.append(iou[slot, member])
        matrix = (
            np.stack(rows, axis=0)
            if rows
            else np.zeros((0, iou.shape[-1]), dtype=np.float32)
        )
        return _assignment(matrix)

    source = evaluate(None)
    action_valid = np.zeros((ACTIONS,), dtype=bool)
    action_outcome = np.full((ACTIONS,), OUTCOME_NEUTRAL, dtype=np.uint8)
    tp50 = np.full((ACTIONS,), source.tp50, dtype=np.int8)
    tp75 = np.full((ACTIONS,), source.tp75, dtype=np.int8)
    lost50 = np.zeros((ACTIONS,), dtype=np.int8)
    lost75 = np.zeros((ACTIONS,), dtype=np.int8)
    matched_iou = np.full((ACTIONS,), source.matched_iou, dtype=np.float32)
    summaries: list[AssignmentSummary | None] = [None] * ACTIONS

    raw_best_action = 0
    raw_best_objective = _objective(source, edit=False, path_rank=0)
    safe_best_action = 0
    safe_best_objective = raw_best_objective
    for slot in range(SLOTS):
        for path in range(ALTERNATIVES):
            index = slot * ALTERNATIVES + path
            if not active[slot] or not valid[slot, path]:
                continue
            action_valid[index] = True
            summary = evaluate((slot, path))
            summaries[index] = summary
            tp50[index] = summary.tp50
            tp75[index] = summary.tp75
            matched_iou[index] = summary.matched_iou
            lost50[index] = len(source.gt50 - summary.gt50)
            lost75[index] = len(source.gt75 - summary.gt75)
            harmful = (
                int(lost50[index]) > 0
                or int(lost75[index]) > 0
                or summary.tp50 < source.tp50
                or summary.tp75 < source.tp75
            )
            improves = (
                summary.tp50 > source.tp50
                or (
                    summary.tp50 == source.tp50
                    and summary.tp75 > source.tp75
                )
            )
            action_outcome[index] = (
                OUTCOME_HARMFUL
                if harmful
                else OUTCOME_BENEFICIAL if improves else OUTCOME_NEUTRAL
            )
            objective = _objective(summary, edit=True, path_rank=path + 1)
            if objective > raw_best_objective:
                raw_best_objective = objective
                raw_best_action = index + 1
            # The deploy target is deliberately more conservative than the
            # raw oracle: it cannot sacrifice a V7-correct GT to win elsewhere.
            if not harmful and objective > safe_best_objective:
                safe_best_objective = objective
                safe_best_action = index + 1

    return {
        "target_action": int(safe_best_action),
        "raw_oracle_action": int(raw_best_action),
        "action_valid": action_valid,
        "action_outcome": action_outcome,
        "tp50": tp50,
        "tp75": tp75,
        "lost50": lost50,
        "lost75": lost75,
        "matched_iou": matched_iou,
        "source_tp50": int(source.tp50),
        "source_tp75": int(source.tp75),
        "source_matched_iou": float(source.matched_iou),
    }

