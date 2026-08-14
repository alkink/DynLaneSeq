from __future__ import annotations

from itertools import permutations

import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    evaluator_hungarian_assignment,
)
from dynlaneseq_eg.tools.v19_semantic_coverage_core import (
    _unique_candidate_tuples,
    evaluate_routes,
    exact_injective_routes,
    select_unique_by_score,
)


def test_unique_score_selection_preserves_proposal_id_uniqueness() -> None:
    score = torch.tensor(
        [
            [9.0, 8.0, 0.0],
            [9.0, 7.0, 6.0],
            [1.0, 5.0, 4.0],
        ]
    )
    route = select_unique_by_score(
        score,
        torch.ones_like(score, dtype=torch.bool),
        torch.tensor([True, True, False]),
    )
    assert route.tolist() == [1, 0, -1]
    assert len(set(route[:2].tolist())) == 2


def test_route_evaluation_reports_semantic_collision() -> None:
    quality = torch.zeros(2, 2, 3)
    quality[0, 0, 0] = 0.90
    quality[0, 1, 1] = 0.85
    result = evaluate_routes(
        quality,
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([0, 1]),
        torch.tensor([True, True]),
        threshold=0.50,
    )
    assert result.hit_count == 1
    assert result.semantic_collision_excess == 1


def _brute_force_best(
    quality: torch.Tensor,
    threshold: float,
) -> tuple[int, float]:
    _gt, slots, candidates = quality.shape
    best = (-1, -1.0)
    for route in permutations(range(candidates), slots):
        matrix = torch.stack(
            [quality[:, slot, candidate] for slot, candidate in enumerate(route)],
            dim=1,
        )
        assignment = evaluator_hungarian_assignment(
            matrix,
            range(slots),
            threshold=threshold,
        )
        best = max(best, (assignment.hit_count, assignment.iou_sum))
    return best


def test_exact_injective_oracle_recovers_missing_physical_lane() -> None:
    # Independent max-any-GT quality prefers candidates 0 and 1, both of
    # which describe GT 0.  Exact set coverage instead chooses candidate 2
    # for slot 1 and recovers GT 1.
    quality = torch.zeros(2, 2, 3)
    quality[:, 0, 0] = torch.tensor([0.92, 0.05])
    quality[:, 0, 1] = torch.tensor([0.80, 0.10])
    quality[:, 0, 2] = torch.tensor([0.20, 0.40])
    quality[:, 1, 0] = torch.tensor([0.75, 0.10])
    quality[:, 1, 1] = torch.tensor([0.90, 0.05])
    quality[:, 1, 2] = torch.tensor([0.10, 0.82])
    valid = torch.ones(2, 3, dtype=torch.bool)
    active = torch.ones(2, dtype=torch.bool)

    independent = select_unique_by_score(quality.max(dim=0).values, valid, active)
    independent_result = evaluate_routes(
        quality, valid, independent, active, threshold=0.50
    )
    assert independent.tolist() == [0, 1]
    assert independent_result.hit_count == 1

    oracle = exact_injective_routes(
        quality,
        valid,
        active,
        thresholds=(0.50,),
        chunk_size=3,
    )[0.50]
    assert oracle.hit_count == 2
    assert oracle.routes == (0, 2)


def test_exact_oracle_matches_bruteforce_official_evaluator() -> None:
    torch.manual_seed(1919)
    quality = torch.rand(3, 3, 5)
    valid = torch.ones(3, 5, dtype=torch.bool)
    valid[1, 4] = False
    active = torch.ones(3, dtype=torch.bool)
    # Brute force uses the same validity restriction.
    expected = (-1, -1.0)
    for route in permutations(range(5), 3):
        if any(not bool(valid[slot, candidate]) for slot, candidate in enumerate(route)):
            continue
        matrix = torch.stack(
            [quality[:, slot, candidate] for slot, candidate in enumerate(route)],
            dim=1,
        )
        assignment = evaluator_hungarian_assignment(
            matrix,
            range(3),
            threshold=0.50,
        )
        expected = max(expected, (assignment.hit_count, assignment.iou_sum))

    oracle = exact_injective_routes(
        quality,
        valid,
        active,
        thresholds=(0.50,),
        chunk_size=7,
    )[0.50]
    assert oracle.hit_count == expected[0]
    assert abs(oracle.qualified_iou_sum - expected[1]) <= 1.0e-6
    assert oracle.assignments_evaluated == int(
        sum(
            all(bool(valid[slot, candidate]) for slot, candidate in enumerate(route))
            for route in permutations(range(5), 3)
        )
    )


def test_unique_tuple_population_is_ordered_without_repeated_ids() -> None:
    values = _unique_candidate_tuples(5, 4)
    assert values.shape == (5 * 4 * 3 * 2, 4)
    assert all(len(set(row.tolist())) == 4 for row in values)
