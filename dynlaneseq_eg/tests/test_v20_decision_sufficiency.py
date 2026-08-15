from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v20_decision_sufficiency import (
    _combined_shortlist_coverage,
    average_precision,
    context_comparison,
    decision_metrics,
)


def test_average_precision_is_one_for_perfect_ranking() -> None:
    scores = torch.tensor((0.9, 0.8, 0.1, -0.2))
    labels = torch.tensor((True, True, False, False))
    assert average_precision(scores, labels) == 1.0


def test_decision_metrics_separate_edit_slot_and_candidate() -> None:
    # Two slots, three candidates and KEEP => seven complete actions.
    batch, slots, candidates = 2, 2, 3
    actions = 1 + slots * candidates
    action_valid = torch.ones(batch, slots, candidates, dtype=torch.bool)
    full_valid = torch.ones(batch, actions, dtype=torch.bool)
    policy_target = torch.zeros(batch, actions)
    delta50_class = torch.ones(batch, actions, dtype=torch.long)
    delta75_class = torch.ones(batch, actions, dtype=torch.long)

    # Image 0 has one useful edit: slot 1, candidate 2 => action 6.
    policy_target[0, 6] = 1.0
    delta50_class[0, 6] = 2
    delta75_class[0, 6] = 2
    # Image 1 has no beneficial edit and therefore targets KEEP.
    policy_target[1, 0] = 1.0
    delta50_class[1, 3] = 0

    raw_scores = torch.tensor(
        (
            (0.0, -0.4, -0.3, -0.2, 0.1, 0.2, 1.5),
            (0.0, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7),
        )
    )
    cache = {
        "action_valid": action_valid,
        "full_action_valid": full_valid,
        "policy_target": policy_target,
        "delta50_class": delta50_class,
        "delta75_class": delta75_class,
        "duplicate": torch.zeros(batch, actions, dtype=torch.bool),
        "abandon": torch.zeros(batch, actions, dtype=torch.bool),
    }
    scored = {
        "raw_action_scores": raw_scores,
        "deployed_action": torch.tensor((6, 0)),
        "expected_delta50": torch.zeros(batch, slots, candidates),
        "expected_delta75": torch.zeros(batch, slots, candidates),
        "duplicate_probability": torch.zeros(batch, slots, candidates),
        "abandon_probability": torch.zeros(batch, slots, candidates),
    }
    metrics = decision_metrics(cache, scored)
    split = metrics["slot_and_candidate"]["any_beneficial_action"]
    outcome = metrics["configured_deployment_outcome"]

    assert metrics["edit_detection"]["average_precision"] == 1.0
    assert split["opportunity_images"] == 1
    assert split["slot_top1_hit"] == 1.0
    assert split["candidate_top1_hit"] == 1.0
    assert split["candidate_top5_recall"] == 1.0
    assert outcome["selected"] == 1
    assert outcome["beneficial"] == 1
    assert outcome["harmful"] == 0


def test_context_comparison_flattens_slot_candidate_outputs() -> None:
    correct = {
        "raw_action_scores": torch.tensor(((0.0, 1.0, 2.0),)),
        "deployed_action": torch.tensor((2,)),
        "expected_delta50": torch.tensor([[[0.1, 0.2]]]),
        "expected_delta75": torch.tensor([[[0.3, 0.4]]]),
        "duplicate_probability": torch.tensor([[[0.5, 0.6]]]),
        "abandon_probability": torch.tensor([[[0.7, 0.8]]]),
    }
    masked = {
        "raw_action_scores": torch.tensor(((0.0, 0.5, 2.0),)),
        "deployed_action": torch.tensor((0,)),
        "expected_delta50": torch.zeros(1, 1, 2),
        "expected_delta75": torch.zeros(1, 1, 2),
        "duplicate_probability": torch.zeros(1, 1, 2),
        "abandon_probability": torch.zeros(1, 1, 2),
    }
    result = context_comparison(
        correct, masked, torch.ones(1, 3, dtype=torch.bool)
    )
    assert result["valid_action_mean_absolute_policy_delta"] == 0.25
    assert result["deployed_action_changed_fraction"] == 1.0


def test_combined_shortlist_separates_top1_and_top2_slot_coverage() -> None:
    scores = torch.tensor(
        [[
            [3.0, 2.0, 1.0],
            [0.3, 0.2, 0.1],
        ]]
    )
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    positive = torch.zeros(1, 7, dtype=torch.bool)
    positive[0, 1 + 1 * 3 + 2] = True
    result = _combined_shortlist_coverage(scores, valid, positive)

    assert result["oracle_positive_slot_top5_coverage"] == 1.0
    assert (
        result["retrieved_slot_shortlists"]["top1_slot_top5"]
        ["coverage_on_opportunity_images"]
        == 0.0
    )
    assert (
        result["retrieved_slot_shortlists"]["top2_slot_top5"]
        ["coverage_on_opportunity_images"]
        == 1.0
    )
