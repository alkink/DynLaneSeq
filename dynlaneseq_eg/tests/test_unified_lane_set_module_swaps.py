from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_unified_lane_set_module_swaps import (
    _hybrid_state,
    _interpret,
)


def _states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    keys = (
        "encoder.backbone.weight",
        "structured_query_head.row_norm.weight",
        "structured_query_head.row_x.weight",
        "structured_query_head.lane_state_layers.0.weight",
    )
    healthy = {key: torch.ones(1) for key in keys}
    failed = {key: torch.full((1,), 2.0) for key in keys}
    return healthy, failed


def _row(recall: float) -> dict[str, object]:
    return {
        "metrics": {
            "capacity": {
                "0.50": {"all_candidates_recall": float(recall)}
            }
        }
    }


def test_bidirectional_row_norm_swaps_use_the_declared_source() -> None:
    healthy, failed = _states()
    rescued, rescued_keys, _ = _hybrid_state(
        "healthy_row_norm",
        healthy,
        failed,
    )
    damaged, damaged_keys, _ = _hybrid_state(
        "failed_row_norm_on_healthy",
        healthy,
        failed,
    )

    assert rescued_keys == ["structured_query_head.row_norm.weight"]
    assert damaged_keys == ["structured_query_head.row_norm.weight"]
    assert torch.equal(
        rescued["structured_query_head.row_norm.weight"],
        healthy["structured_query_head.row_norm.weight"],
    )
    assert torch.equal(
        rescued["structured_query_head.row_x.weight"],
        failed["structured_query_head.row_x.weight"],
    )
    assert torch.equal(
        damaged["structured_query_head.row_norm.weight"],
        failed["structured_query_head.row_norm.weight"],
    )
    assert torch.equal(
        damaged["structured_query_head.row_x.weight"],
        healthy["structured_query_head.row_x.weight"],
    )


def test_interpretation_uses_damage_as_direct_causal_evidence() -> None:
    verdict = _interpret(
        {
            "control_healthy": _row(0.95),
            "control_failed": _row(0.00),
            "healthy_row_norm": _row(0.10),
            "failed_row_norm_on_healthy": _row(0.02),
        }
    )

    assert (
        verdict["primary_localization"]
        == "final_row_normalization_is_a_direct_causal_bottleneck"
    )
    assert verdict["strong_recovery_variants"] == []
    assert verdict["strong_damage_variants"] == [
        "failed_row_norm_on_healthy"
    ]
