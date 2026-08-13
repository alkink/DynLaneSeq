from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v11_causal_replay import (
    _accumulate_record,
    _factorial_decomposition,
    _factorial_policies,
    _finalize_metric_tree,
    _new_metric_tree,
    _variant_kwargs,
)


def _slot_outputs(offset: float, active: tuple[bool, bool]) -> dict[str, torch.Tensor]:
    x = torch.tensor(
        [
            [10.0 + offset, 11.0 + offset, 12.0 + offset],
            [30.0 + offset, 31.0 + offset, 32.0 + offset],
        ]
    )
    lane_range = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    return {
        "selection_slot_pred_x_rows": x,
        "selection_slot_range_norm": lane_range,
        "selection_slot_active": torch.tensor(active),
        "selection_slot_unified_base_x_rows": x - 0.5,
        "selection_slot_unified_base_range_norm": lane_range * 0.9,
        "selection_slot_row_delta_logits": torch.zeros(2, 3, 3),
        "selection_slot_row_delta_offsets_px": torch.tensor(
            [-10.0, 0.0, 10.0]
        ),
        "selection_slot_range_delta": torch.zeros(2, 2),
    }


def test_factorial_policy_axes_mix_only_the_named_tensors() -> None:
    source = _slot_outputs(0.0, (True, False))
    v11 = _slot_outputs(2.0, (False, True))
    policies = _factorial_policies(source, v11, input_w=100)
    fixed = policies["v11_final_x_v7_range__v7_activity"]
    assert torch.equal(fixed["x"], v11["selection_slot_pred_x_rows"])
    assert torch.equal(fixed["range"], source["selection_slot_range_norm"])
    assert torch.equal(fixed["active"], source["selection_slot_active"])
    joint = policies["v11_final__v11_activity"]
    assert torch.equal(joint["x"], v11["selection_slot_pred_x_rows"])
    assert torch.equal(joint["range"], v11["selection_slot_range_norm"])
    assert torch.equal(joint["active"], v11["selection_slot_active"])


def test_v11_residual_can_be_replayed_on_exact_v7_anchor() -> None:
    source = _slot_outputs(0.0, (True, False))
    v11 = _slot_outputs(2.0, (False, True))
    v11["selection_slot_row_delta_logits"][..., 2] = 8.0
    v11["selection_slot_range_delta"] = torch.tensor(
        [[0.1, -0.1], [0.2, -0.2]]
    )
    policies = _factorial_policies(source, v11, input_w=100)
    anchored = policies["v7_anchor_plus_v11_residual__v7_activity"]
    assert not torch.equal(anchored["x"], source["selection_slot_pred_x_rows"])
    assert not torch.equal(
        anchored["range"], source["selection_slot_range_norm"]
    )
    assert torch.equal(anchored["active"], source["selection_slot_active"])


def test_official_counter_uses_strict_threshold_and_writer_validity() -> None:
    policies = ("v7_geometry__v7_activity",)
    tree = _new_metric_tree(policies, (0.50, 0.75))
    # The first proposal is exactly .50 and must not count as a CULane TP;
    # the second is invalid at writer level despite being neural-active.
    quality = torch.tensor([[0.50, 0.90]])
    valid = torch.tensor([True, False])
    primary = _accumulate_record(
        tree,
        {"quality": quality, "valid": valid},
        {"v7_geometry__v7_activity": (0, 2)},
        {"v7_geometry__v7_activity": torch.tensor([True, True])},
        (0.50, 0.75),
    )
    finalized = _finalize_metric_tree(tree)
    neural = finalized["v7_geometry__v7_activity"]["neural_active"]
    writer = finalized["v7_geometry__v7_activity"]["writer_valid"]
    assert neural["thresholds"]["0.50"]["tp"] == 1
    assert writer["thresholds"]["0.50"]["tp"] == 0
    assert writer["thresholds"]["0.50"]["predictions"] == 1
    assert primary["v7_geometry__v7_activity"]["0.50"]["tp"] == 0


def test_geometry_activity_shapley_closes_to_joint_delta() -> None:
    policies = (
        "v7_geometry__v7_activity",
        "v11_final__v7_activity",
        "v7_geometry__v11_activity",
        "v11_final__v11_activity",
    )
    tree = _new_metric_tree(policies, (0.50,))
    values = {
        "v7_geometry__v7_activity": (10, 12, 15),
        "v11_final__v7_activity": (12, 12, 15),
        "v7_geometry__v11_activity": (9, 11, 15),
        "v11_final__v11_activity": (11, 11, 15),
    }
    for policy, (tp, predictions, gt) in values.items():
        for mode in ("neural_active", "writer_valid"):
            row = tree[policy][mode]
            row["images"] = 1
            row["thresholds"]["0.50"] = {
                "tp": tp,
                "predictions": predictions,
                "gt": gt,
            }
    metrics = _finalize_metric_tree(tree)
    decomposition = _factorial_decomposition(metrics, (0.50,))
    f1 = decomposition["writer_valid"]["0.50"]["f1"]
    assert abs(
        f1["geometry_shapley"]
        + f1["activity_shapley"]
        - f1["joint_delta"]
    ) < 1.0e-12


def test_evidence_interventions_change_only_requested_inputs() -> None:
    captured = {
        "row_value_features": torch.arange(2 * 3 * 4 * 1).view(2, 3, 4, 1),
        "proposal_row_tokens": torch.arange(2 * 2 * 3 * 1).view(2, 2, 3, 1),
        "proposal_x_rows": torch.arange(2 * 2 * 3).view(2, 2, 3),
        "proposal_range_norm": torch.arange(2 * 2 * 2).view(2, 2, 2),
        "legacy_route_logits": torch.arange(2 * 2 * 2).view(2, 2, 2),
        "candidate_valid": torch.ones(2, 2, dtype=torch.bool),
        "slot_states": torch.randn(2, 2, 3),
        "legacy_active_logits": torch.randn(2, 2),
        "route_indices": torch.tensor([[0, 1], [1, 0]]),
    }
    wrong_p2 = _variant_kwargs(captured, "p2_wrong_image")
    assert torch.equal(
        wrong_p2["row_value_features"][0],
        captured["row_value_features"][1],
    )
    assert wrong_p2["proposal_row_tokens"] is captured["proposal_row_tokens"]
    zero_prior = _variant_kwargs(captured, "legacy_route_prior_zero")
    assert torch.count_nonzero(zero_prior["legacy_route_logits"]) == 0
    assert zero_prior["proposal_x_rows"] is captured["proposal_x_rows"]
