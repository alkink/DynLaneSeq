from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.structured_queries import SetAwareLaneSelectionHead
from dynlaneseq_eg.tools.audit_v4_5_pointer_stop_representative import (
    _replacement_sequence,
    prefix_extension_oracle_hit_count,
)


def _selector() -> SetAwareLaneSelectionHead:
    return SetAwareLaneSelectionHead(
        8,
        input_w=64,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        curve_samples=4,
        unified_score=True,
        detach_geometry_features=True,
        candidate_interaction="sequential_pointer",
        relation_hidden_dim=8,
        pointer_max_selections=4,
        pointer_min_valid_rows=2,
        pointer_similarity_prior=0.5,
    ).eval()


def test_forced_stop_continuation_preserves_prefix_and_fills_available_slots() -> None:
    torch.manual_seed(811)
    selector = _selector()
    hidden = torch.randn(1, 4, 16)
    relations = torch.zeros(1, 4, 4, 6)
    unary = torch.tensor([[6.0, 5.0, 4.0, 3.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    original_stop = selector.pointer_stop.forward

    def stop_after_first(inputs: torch.Tensor) -> torch.Tensor:
        # Step embeddings are different, but this deterministic diagnostic
        # only needs STOP to dominate every step.  Force action 0 below to
        # create the prefix whose preservation is under test.
        return torch.full((inputs.shape[0], 1), 50.0, device=inputs.device)

    selector.pointer_stop.forward = stop_after_first
    forced_prefix = torch.tensor([[0, -1, -1, -1]])
    prefix_rollout = selector.decode_pointer(
        hidden,
        relations,
        unary,
        valid,
        forced_actions=forced_prefix,
    )
    assert prefix_rollout["selection_pointer_indices"].tolist() == [
        [0, -1, -1, -1]
    ]
    continued = selector.decode_pointer(
        hidden,
        relations,
        unary,
        valid,
        forced_actions=forced_prefix,
        force_candidate_after_stop=True,
    )
    emitted = continued["selection_pointer_indices"][0].tolist()
    assert emitted[0] == 0
    assert len(set(emitted)) == 4
    assert bool(continued["selection_pointer_stop_would_win"].all())
    selector.pointer_stop.forward = original_stop


def test_forced_action_rejects_an_unavailable_candidate() -> None:
    torch.manual_seed(813)
    selector = _selector()
    hidden = torch.randn(1, 3, 16)
    relations = torch.zeros(1, 3, 3, 6)
    unary = torch.tensor([[6.0, 5.0, 4.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    actions = torch.tensor([[0, 0, -1, -1]])
    try:
        selector.decode_pointer(
            hidden,
            relations,
            unary,
            valid,
            forced_actions=actions,
        )
    except ValueError as error:
        assert "unavailable" in str(error)
    else:
        raise AssertionError("forcing the same candidate twice must fail")


def test_prefix_extension_oracle_keeps_false_positive_prefix_slot_cost() -> None:
    # Candidate 0 is a fixed false positive.  With top_k=2 only one optional
    # candidate may be added, even though two valid GT representatives remain.
    iou = torch.tensor(
        [
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.9],
        ]
    )
    hit_count = prefix_extension_oracle_hit_count(
        iou,
        fixed_ids=[0],
        candidate_valid=torch.ones(3, dtype=torch.bool),
        threshold=0.5,
        top_k=2,
    )
    assert hit_count == 1


def test_prefix_extension_oracle_can_reassign_the_fixed_prefix() -> None:
    # Greedy evaluation might pair candidate 0 to GT0, but the extension can
    # reassign it to GT1 and add candidate 1 for GT0, yielding two hits.
    iou = torch.tensor(
        [
            [0.80, 0.90],
            [0.85, 0.00],
        ]
    )
    hit_count = prefix_extension_oracle_hit_count(
        iou,
        fixed_ids=[0],
        candidate_valid=torch.ones(2, dtype=torch.bool),
        threshold=0.75,
        top_k=2,
    )
    assert hit_count == 2


def test_local_replacement_preserves_count_and_prefers_requested_score() -> None:
    official = torch.tensor(
        [
            [0.70, 0.90, 0.05, 0.00],
            [0.00, 0.05, 0.70, 0.88],
        ]
    )
    row_quality = official.t().contiguous()
    unary = torch.tensor([0.1, 0.9, 0.2, 0.8])
    pointer_logits = torch.zeros(4, 5)
    selected = [0, 2]
    official_replacement = _replacement_sequence(
        official,
        row_quality,
        torch.ones(4, dtype=torch.bool),
        selected,
        unary,
        pointer_logits,
        mode="official",
        representable_min=0.2,
    )
    unary_replacement = _replacement_sequence(
        official,
        row_quality,
        torch.ones(4, dtype=torch.bool),
        selected,
        unary,
        pointer_logits,
        mode="unary",
        representable_min=0.2,
    )
    assert official_replacement == [1, 3]
    assert unary_replacement == [1, 3]
    assert len(official_replacement) == len(selected)
