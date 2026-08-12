from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_v9_global_support_observability import (
    ARMS,
    GlobalSupportSetProbe,
    evaluate_logits,
    fixed_rademacher_projection,
    make_decision,
)


def _probe() -> GlobalSupportSetProbe:
    return GlobalSupportSetProbe(
        torch.zeros(8),
        torch.ones(8),
        row_count=3,
        hidden_dim=16,
        row_layers=1,
        candidate_layers=1,
        slot_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        num_slots=2,
    )


def test_fixed_projection_is_deterministic() -> None:
    first = fixed_rademacher_projection(17, 8, 3407)
    second = fixed_rademacher_projection(17, 8, 3407)
    other = fixed_rademacher_projection(17, 8, 3408)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.shape == (17, 8)


def test_common_probe_shapes_and_masks() -> None:
    probe = _probe().eval()
    features = torch.randn(2, 4, 3, 8)
    visible = torch.ones(2, 4, 3, dtype=torch.bool)
    visible[0, 3] = False
    candidate_valid = torch.ones(2, 4, dtype=torch.bool)
    candidate_valid[0, 3] = False
    active, route = probe(features, visible, candidate_valid)
    assert active.shape == (2, 2)
    assert route.shape == (2, 2, 4)
    assert torch.all(route[0, :, 3] == -1.0e4)


def test_evaluate_logits_recovers_two_target_supports() -> None:
    target_rows = torch.zeros(1, 4, 5)
    target_rows[0, 0, 0] = 1.0
    target_rows[0, 1, 2] = 1.0
    cache = {
        "candidate_valid": torch.ones(1, 4, dtype=torch.bool),
        "target_rows": target_rows,
        "target_active": torch.tensor([[True, True, False, False]]),
    }
    active = torch.tensor([[4.0, 4.0]])
    route = torch.tensor(
        [[[8.0, 0.0, -1.0, -2.0], [-1.0, 0.0, 8.0, -2.0]]]
    )
    metrics = evaluate_logits(cache, [0], active, route)
    assert metrics["assigned_gt"] == 2
    assert metrics["route_in_support_fraction"] == 1.0
    assert metrics["decoded_target_id_fraction"] == 1.0
    assert metrics["target_id_rank_top1_fraction"] == 1.0
    assert metrics["target_id_rank_top4_fraction"] == 1.0
    assert metrics["cardinality_exact_fraction"] == 1.0
    assert metrics["mean_target_support_mass"] > 0.99


def test_decision_selects_richer_evidence_only_after_control_reproduction() -> None:
    def summary(mass: float, hit: float, top1: float, fit_hit: float) -> dict:
        split = {
            "mean_target_support_mass": {"mean": mass},
            "route_in_support_fraction": {"mean": hit},
            "decoded_target_id_fraction": {"mean": top1},
            "target_id_rank_top1_fraction": {"mean": top1},
            "selection_score": {"mean": (mass + hit) / 2.0},
        }
        fit = dict(split)
        fit["route_in_support_fraction"] = {"mean": fit_hit}
        return {"fit": fit, "validation": split}

    summaries = {
        "descriptor": summary(0.45, 0.50, 0.40, 0.52),
        "row_tokens": summary(0.52, 0.58, 0.48, 0.64),
        "p2_evidence": summary(0.50, 0.56, 0.46, 0.61),
        "combined": summary(0.66, 0.70, 0.57, 0.78),
    }
    assert set(summaries) == set(ARMS)
    decision = make_decision(
        {
            "mean_target_support_mass": 0.44,
            "route_in_support_fraction": 0.49,
        },
        summaries,
    )
    assert decision["passed"] is True
    assert decision["best_arm"] == "combined"
    assert decision["descriptor_control_valid"] is True
