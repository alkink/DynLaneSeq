from __future__ import annotations

from itertools import permutations, product

import torch

from dynlaneseq_eg.tools.audit_v26_refined_candidate_bank import (
    exact_raw_refined_union_oracle,
    exact_source_tied_oracle,
)
from dynlaneseq_eg.tools.v19_semantic_coverage_core import (
    _official_matched_values,
)


def _summary(quality: torch.Tensor) -> tuple[int, int, float]:
    matched, total = _official_matched_values(quality.unsqueeze(0))
    return (
        int((matched[0] > 0.50).sum()),
        int((matched[0] > 0.75).sum()),
        float(total[0]),
    )


def _brute_union(
    raw: torch.Tensor,
    refined: torch.Tensor,
    active: torch.Tensor,
    source: torch.Tensor,
) -> tuple[tuple[int, int, int, float], tuple[int, ...], tuple[int, ...]]:
    _gt, slots, candidates = raw.shape
    active_slots = torch.nonzero(active, as_tuple=False).flatten().tolist()
    best = None
    best_routes = None
    best_variants = None
    for routes in permutations(range(candidates), len(active_slots)):
        for variants in product((0, 1), repeat=len(active_slots)):
            columns = []
            for slot, candidate, variant in zip(active_slots, routes, variants):
                bank = refined if variant else raw
                columns.append(bank[:, slot, candidate])
            tp50, tp75, total = _summary(torch.stack(columns, dim=-1))
            edits = sum(
                int(candidate != int(source[slot]) or variant != 1)
                for slot, candidate, variant in zip(
                    active_slots, routes, variants
                )
            )
            key = tp50, tp75, -edits, total
            if best is None or key > best:
                best = key
                best_routes = routes
                best_variants = variants
    assert best is not None
    assert best_routes is not None
    assert best_variants is not None
    return best, best_routes, best_variants


def test_source_on_tie_precedes_matched_iou() -> None:
    quality = torch.tensor(
        [
            [[0.60, 0.70, 0.05], [0.05, 0.05, 0.05]],
            [[0.05, 0.05, 0.05], [0.05, 0.60, 0.70]],
        ],
        dtype=torch.float32,
    )
    valid = torch.ones((2, 3), dtype=torch.bool)
    active = torch.ones(2, dtype=torch.bool)
    source = torch.tensor((0, 1), dtype=torch.long)
    result = exact_source_tied_oracle(
        quality,
        valid,
        active,
        source,
        device="cpu",
        chunk_size=8,
    )
    assert result.routes == (0, 1)
    assert result.edit_count == 0
    assert result.tp50 == 2


def test_threshold_gain_precedes_source_retention() -> None:
    quality = torch.tensor(
        [
            [[0.60, 0.80, 0.05], [0.05, 0.05, 0.05]],
            [[0.05, 0.05, 0.05], [0.05, 0.60, 0.80]],
        ],
        dtype=torch.float32,
    )
    valid = torch.ones((2, 3), dtype=torch.bool)
    active = torch.ones(2, dtype=torch.bool)
    source = torch.tensor((0, 1), dtype=torch.long)
    result = exact_source_tied_oracle(
        quality,
        valid,
        active,
        source,
        device="cpu",
        chunk_size=8,
    )
    assert result.routes == (1, 2)
    assert result.tp75 == 2
    assert result.edit_count == 2


def test_raw_refined_union_matches_bruteforce_random_small_cases() -> None:
    generator = torch.Generator().manual_seed(3407)
    for _ in range(12):
        raw = torch.rand((2, 2, 3), generator=generator)
        refined = torch.rand((2, 2, 3), generator=generator)
        valid = torch.ones((2, 3), dtype=torch.bool)
        active = torch.ones(2, dtype=torch.bool)
        source = torch.tensor((0, 1), dtype=torch.long)
        expected, _routes, _variants = _brute_union(
            raw, refined, active, source
        )
        result = exact_raw_refined_union_oracle(
            raw,
            refined,
            valid,
            valid,
            active,
            source,
            device="cpu",
            chunk_size=16,
        )
        observed = (
            result.tp50,
            result.tp75,
            -result.edit_count,
            result.matched_iou,
        )
        assert observed[:3] == expected[:3]
        assert abs(observed[3] - expected[3]) < 1.0e-6


def test_union_never_reuses_proposal_provenance() -> None:
    raw = torch.zeros((2, 2, 2), dtype=torch.float32)
    refined = torch.zeros_like(raw)
    refined[0, 0, 0] = 0.95
    refined[1, 1, 0] = 0.95
    refined[0, 0, 1] = 0.80
    refined[1, 1, 1] = 0.80
    valid = torch.ones((2, 2), dtype=torch.bool)
    result = exact_raw_refined_union_oracle(
        raw,
        refined,
        valid,
        valid,
        torch.ones(2, dtype=torch.bool),
        torch.tensor((0, 1), dtype=torch.long),
        device="cpu",
        chunk_size=16,
    )
    selected = [value for value in result.routes if value >= 0]
    assert len(selected) == len(set(selected))

