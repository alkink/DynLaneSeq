from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_four_slot_coverage_selector import (
    FourSlotCoverageProbe,
    _fill_with_source_scores,
    _mmr_ids,
    _unique_pointer_ids,
    build_soft_slot_targets,
    coverage_gate,
    coverage_pointer_loss,
)


def _cache() -> dict[str, object]:
    pred_x = torch.tensor(
        [
            [
                [10.0, 10.0, 10.0, 10.0],
                [12.0, 12.0, 12.0, 12.0],
                [50.0, 50.0, 50.0, 50.0],
                [52.0, 52.0, 52.0, 52.0],
            ]
        ]
    )
    return {
        "features": torch.randn(1, 4, 6),
        "candidate_valid": torch.ones(1, 4, dtype=torch.bool),
        "official_iou": [
            torch.tensor(
                [
                    [0.90, 0.85, 0.05, 0.00],
                    [0.00, 0.05, 0.90, 0.82],
                ]
            )
        ],
        "stage": {
            "pred_x_rows": pred_x,
            "range_norm": torch.tensor([[[0.0, 1.0]] * 4]),
        },
    }


def test_four_slot_pointer_is_candidate_permutation_equivariant() -> None:
    torch.manual_seed(17)
    probe = FourSlotCoverageProbe(
        6,
        hidden_dim=16,
        num_slots=4,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    ).eval()
    features = torch.randn(2, 5, 6)
    valid = torch.ones(2, 5, dtype=torch.bool)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    expected, expected_presence = probe(features, valid)
    actual, actual_presence = probe(features[:, permutation], valid[:, permutation])
    torch.testing.assert_close(actual, expected[:, :, permutation])
    torch.testing.assert_close(actual_presence, expected_presence)


def test_soft_targets_keep_duplicates_positive_but_separate_lanes() -> None:
    result = build_soft_slot_targets(
        _cache(),
        num_slots=4,
        temperature=0.05,
        iou_band=0.10,
        min_iou=0.30,
    )
    active = result["active"]
    targets = result["targets"]
    affinity = result["affinity"]
    assert active[0].tolist() == [True, True, False, False]
    assert float(targets[0, 0, 0]) > 0.0
    assert float(targets[0, 0, 1]) > 0.0
    assert float(targets[0, 0, 2]) == 0.0
    assert float(targets[0, 1, 2]) > 0.0
    assert float(targets[0, 1, 3]) > 0.0
    assert float(affinity[0, 0, 1]) > float(affinity[0, 0, 2])


def test_pointer_loss_backpropagates_through_all_terms() -> None:
    target_bundle = build_soft_slot_targets(
        _cache(),
        num_slots=4,
        temperature=0.05,
        iou_band=0.10,
        min_iou=0.30,
    )
    logits = torch.zeros((1, 4, 4), requires_grad=True)
    presence = torch.zeros((1, 4), requires_grad=True)
    losses = coverage_pointer_loss(
        logits,
        presence,
        target_bundle["targets"],
        target_bundle["active"],
        target_bundle["affinity"],
        presence_weight=0.25,
        diversity_weight=0.25,
    )
    losses["total"].backward()
    assert logits.grad is not None
    assert presence.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0
    assert float(presence.grad.abs().sum()) > 0.0


def test_pointer_assignment_and_mmr_select_unique_candidates() -> None:
    logits = torch.tensor(
        [
            [10.0, 9.0, 1.0, 0.0],
            [10.0, 8.0, 7.0, 0.0],
            [10.0, 1.0, 8.0, 7.0],
            [10.0, 1.0, 2.0, 9.0],
        ]
    )
    valid = torch.ones(4, dtype=torch.bool)
    pointer_ids = _unique_pointer_ids(logits, valid)
    assert len(pointer_ids) == 4
    assert len(set(pointer_ids)) == 4

    filled_ids = _fill_with_source_scores(
        [2],
        torch.tensor([0.95, 0.80, 0.99, 0.70]),
        valid,
        top_k=3,
    )
    assert filled_ids == [2, 0, 1]

    distance = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0],
            [0.0, 0.0, 100.0, 100.0],
            [100.0, 100.0, 0.0, 100.0],
            [100.0, 100.0, 100.0, 0.0],
        ]
    )
    mmr_ids = _mmr_ids(
        torch.tensor([1.0, 0.95, 0.90, 0.85]),
        distance,
        valid,
        penalty=0.50,
        sigma=10.0,
        top_k=2,
    )
    assert mmr_ids == [0, 2]


def test_gate_is_positive_only_when_both_recall_thresholds_pass() -> None:
    source = {"recall_050": 0.54, "recall_070": 0.45}
    pointer = {"recall_050": 0.60, "recall_070": 0.47}
    mmr = {"recall_050": 0.55, "recall_070": 0.49}
    result = coverage_gate(
        pointer,
        mmr,
        source,
        min_gain_050_points=5.0,
        min_gain_070_points=3.0,
    )
    assert result["pointer_positive"] is False
    assert result["mmr_positive"] is False

    pointer["recall_070"] = 0.49
    result = coverage_gate(
        pointer,
        mmr,
        source,
        min_gain_050_points=5.0,
        min_gain_070_points=3.0,
    )
    assert result["pointer_positive"] is True
    assert result["four_slot_hypothesis_positive"] is True
    assert result["any_diversity_signal_positive"] is True
    assert result["interpretation"] == "learned_explicit_coverage_is_supported"
