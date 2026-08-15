from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.v21a_pairwise_visual_verifier import (
    PairwiseVisualLaneVerifier,
    reverse_complete_curve_relation,
)
from dynlaneseq_eg.tools.cache_v21a_pairwise_visual_verification import (
    _gather_topk,
    _lexicographic_masks,
    _same_slot_relation,
)
from dynlaneseq_eg.tools.train_v21a_cached_visual_verifier import verifier_loss


def _verifier() -> PairwiseVisualLaneVerifier:
    torch.manual_seed(19)
    return PairwiseVisualLaneVerifier(
        profile_channels=7,
        state_dim=12,
        scalar_dim=4,
        relation_dim=11,
        rows=6,
        offsets=5,
        curve_dim=16,
        state_hidden_dim=8,
        scalar_hidden_dim=8,
        relation_hidden_dim=8,
        pair_hidden_dim=24,
        num_heads=4,
        ff_dim=32,
        vertical_layers=1,
    ).eval()


def test_pairwise_score_is_exactly_antisymmetric() -> None:
    model = _verifier()
    batch, candidates, rows, offsets, channels = 3, 4, 6, 5, 7
    output = model(
        source_profile=torch.randn(batch, rows, offsets, channels),
        candidate_profile=torch.randn(
            batch, candidates, rows, offsets, channels
        ),
        source_row_weight=torch.rand(batch, rows),
        candidate_row_weight=torch.rand(batch, candidates, rows),
        source_state=torch.randn(batch, 12),
        candidate_state=torch.randn(batch, candidates, 12),
        source_scalar=torch.randn(batch, 4),
        candidate_scalar=torch.randn(batch, candidates, 4),
        candidate_to_source_relation=torch.randn(batch, candidates, 11),
        candidate_valid=torch.ones(batch, candidates, dtype=torch.bool),
        return_swapped=True,
    )
    torch.testing.assert_close(
        output["score"],
        -output["swapped_score"],
        rtol=0.0,
        atol=0.0,
    )


def test_invalid_candidate_is_masked_without_affecting_valid_scores() -> None:
    model = _verifier()
    kwargs = {
        "source_profile": torch.randn(1, 6, 5, 7),
        "candidate_profile": torch.randn(1, 2, 6, 5, 7),
        "source_row_weight": torch.ones(1, 6),
        "candidate_row_weight": torch.ones(1, 2, 6),
        "source_state": torch.randn(1, 12),
        "candidate_state": torch.randn(1, 2, 12),
        "source_scalar": torch.randn(1, 4),
        "candidate_scalar": torch.randn(1, 2, 4),
        "candidate_to_source_relation": torch.randn(1, 2, 11),
    }
    valid = model(**kwargs, candidate_valid=torch.ones(1, 2, dtype=torch.bool))["score"]
    masked = model(
        **kwargs, candidate_valid=torch.tensor([[True, False]])
    )["score"]
    torch.testing.assert_close(valid[:, 0], masked[:, 0])
    assert float(masked[0, 1]) == -1.0e4


def test_relation_reversal_negates_only_directed_fields() -> None:
    relation = torch.arange(1, 12, dtype=torch.float32)
    reversed_relation = reverse_complete_curve_relation(relation)
    directed = {0, 3, 4, 6, 7}
    for index in range(11):
        expected = -relation[index] if index in directed else relation[index]
        assert float(reversed_relation[index]) == float(expected)


def test_cache_topk_and_same_slot_relation_helpers() -> None:
    value = torch.arange(2 * 3 * 5 * 2).reshape(2, 3, 5, 2)
    index = torch.tensor(
        [
            [[4, 1], [3, 0], [2, 4]],
            [[0, 2], [1, 4], [3, 2]],
        ]
    )
    gathered = _gather_topk(value, index)
    assert gathered.shape == (2, 3, 2, 2)
    for batch in range(2):
        for slot in range(3):
            torch.testing.assert_close(
                gathered[batch, slot], value[batch, slot, index[batch, slot]]
            )

    relation = torch.randn(2, 3, 5, 3, 11)
    own = _same_slot_relation(relation)
    assert own.shape == (2, 3, 5, 11)
    for slot in range(3):
        torch.testing.assert_close(own[:, slot], relation[:, slot, :, slot])


def test_lexicographic_masks_prioritize_050_then_075() -> None:
    delta50 = torch.tensor([[2, 1, 1, 0, 1]])
    delta75 = torch.tensor([[0, 2, 1, 2, 0]])
    beneficial, harmful, neutral = _lexicographic_masks(delta50, delta75)
    torch.testing.assert_close(
        beneficial, torch.tensor([[True, True, False, False, False]])
    )
    torch.testing.assert_close(
        harmful, torch.tensor([[False, False, False, True, True]])
    )
    torch.testing.assert_close(
        neutral, torch.tensor([[False, False, True, False, False]])
    )


def test_verifier_loss_trains_top5_beneficial_ranking() -> None:
    score = torch.tensor(
        [[0.2, -0.1, 0.5, -0.4, 0.0], [0.1, 0.3, -0.2, 0.0, -0.5]],
        requires_grad=True,
    )
    batch = {
        "candidate_valid": torch.ones(2, 5, dtype=torch.bool),
        "beneficial": torch.tensor(
            [[False, False, True, False, False], [False, True, False, False, False]]
        ),
    }
    loss, diagnostics = verifier_loss(score, batch)
    assert torch.isfinite(loss)
    assert float(diagnostics["conditional_top1"]) == 1.0
    loss.backward()
    assert score.grad is not None
    assert float(score.grad[0, 2]) < 0.0
    assert float(score.grad[1, 1]) < 0.0
