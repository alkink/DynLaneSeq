from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s0 import (
    LossConfig,
    S0Criterion,
    build_pointer_cluster_soft_targets,
)
from dynlaneseq_eg.modeling.structured_queries import SetAwareLaneSelectionHead
from dynlaneseq_eg.tools.summarize_v4_5_pointer_trajectory import _trajectory_row


CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_v4_5_cluster_soft_pointer.yaml"
)


def _selector(*, row_grid_mode: str = "fixed_rows") -> SetAwareLaneSelectionHead:
    return SetAwareLaneSelectionHead(
        8,
        input_w=100,
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
        pointer_teacher_mode="cluster_soft_randomized",
        row_grid_mode=row_grid_mode,
    )


def _geometry() -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    rows = 8
    outputs = {
        "pred_x_rows": torch.tensor(
            [
                [
                    [10.0] * rows,
                    [10.5] * rows,
                    [30.0] * rows,
                    [50.0] * rows,
                    [50.4] * rows,
                ]
            ]
        ),
        "range_norm": torch.tensor([[[0.0, 1.0]] * 5]),
    }
    targets = [
        {
            "x_rows": torch.tensor(
                [[10.0] * rows, [50.0] * rows, [90.0] * rows]
            ),
            "valid_mask": torch.ones(3, rows, dtype=torch.bool),
        }
    ]
    return outputs, targets


def _teacher(*, iteration: int = 50000, visit: int = 0):
    outputs, targets = _geometry()
    return build_pointer_cluster_soft_targets(
        outputs,
        targets,
        max_selections=4,
        input_h=32,
        line_width=12.0,
        min_valid_rows=2,
        representable_min=0.20,
        support_quality_delta=0.10,
        temperature=0.03,
        base_seed=3407,
        iteration=iteration,
        visit=visit,
    )


def test_v4_5_config_is_the_calibrated_teacher_only_contract() -> None:
    cfg = load_config(CONFIG)
    selection = cfg["model"]["structured_query"]["set_selection"]
    loss = cfg["loss"]
    assert selection["pointer_teacher_mode"] == "cluster_soft_randomized"
    assert selection["pointer_cluster_representable_min"] == 0.20
    assert selection["pointer_cluster_quality_delta"] == 0.10
    assert selection["pointer_cluster_temperature"] == 0.03
    assert selection["row_grid_mode"] == "fixed_rows"
    assert loss["pointer_unary_target_mode"] == "max_quality"
    assert loss["pointer_quality_weight"] == 0.10


def test_cluster_teacher_is_soft_jointly_representable_and_stops() -> None:
    teacher = _teacher()
    indices = teacher["indices"][0]
    probabilities = teacher["probabilities"][0]
    active = teacher["active"][0]
    candidates = probabilities.shape[-1] - 1

    # The third GT has no candidate above the strict 0.20 representability
    # gate.  Two clusters are taught, followed by STOP and one ignored step.
    assert int(teacher["representable_count"][0]) == 2
    assert active.tolist() == [True, True, True, False]
    assert int(indices[2]) == candidates
    assert int(indices[3]) == -100
    torch.testing.assert_close(
        probabilities[active].sum(dim=-1),
        torch.ones(3),
    )
    assert float(probabilities[~active].abs().sum()) == 0.0
    assert sorted(teacher["support_sizes"][0, :2].tolist()) == [2.0, 2.0]
    assert bool((teacher["target_entropy"][0, :2] > 0.0).all())
    emitted = indices[(indices >= 0) & (indices < candidates)]
    assert emitted.numel() == emitted.unique().numel()


def test_cluster_teacher_randomization_is_reproducible_per_visit() -> None:
    first = _teacher(iteration=50007, visit=2)
    repeat = _teacher(iteration=50007, visit=2)
    torch.testing.assert_close(first["indices"], repeat["indices"])
    torch.testing.assert_close(first["probabilities"], repeat["probabilities"])

    sequences = {
        tuple(_teacher(iteration=50007, visit=visit)["indices"][0].tolist())
        for visit in range(12)
    }
    assert len(sequences) > 1


def test_cluster_teacher_reserves_future_representatives_on_overlap() -> None:
    rows = 8
    outputs = {
        "pred_x_rows": torch.tensor(
            [[[10.0] * rows, [10.5] * rows, [70.0] * rows]]
        ),
        "range_norm": torch.tensor([[[0.0, 1.0]] * 3]),
    }
    targets = [
        {
            "x_rows": torch.tensor([[10.0] * rows, [10.5] * rows]),
            "valid_mask": torch.ones(2, rows, dtype=torch.bool),
        }
    ]
    teacher = build_pointer_cluster_soft_targets(
        outputs,
        targets,
        max_selections=4,
        input_h=32,
        line_width=12.0,
        min_valid_rows=2,
        representable_min=0.20,
        support_quality_delta=0.10,
        temperature=0.03,
        base_seed=3407,
        iteration=50000,
        visit=0,
    )
    emitted = teacher["indices"][0, :2]
    assert sorted(emitted.tolist()) == [0, 1]
    assert emitted.numel() == emitted.unique().numel()
    assert float(teacher["reservation_exclusion_count"].sum()) > 0.0


def test_soft_cluster_reroll_and_loss_are_differentiable() -> None:
    torch.manual_seed(451)
    selector = _selector().train()
    hidden = torch.randn(1, 5, 16, requires_grad=True)
    relations = torch.zeros(1, 5, 5, 6)
    unary = torch.randn(1, 5, requires_grad=True)
    geometry, targets = _geometry()
    outputs = {
        **geometry,
        "_selection_pointer_hidden": hidden,
        "_selection_pointer_relations": relations,
        "_selection_pointer_unary_logits": unary,
        "_selection_pointer_candidate_valid": torch.ones(1, 5, dtype=torch.bool),
        "selection_logits": unary,
    }
    teacher = _teacher()
    selector.reroll_pointer_with_cluster_teacher(outputs, teacher)
    criterion = S0Criterion(
        LossConfig(
            input_h=32,
            input_w=100,
            set_selection_line_width=12.0,
            set_selection_min_valid_rows=2,
            set_selection_focal_beta=0.0,
            w_pointer_selection=1.0,
            pointer_quality_weight=0.10,
            pointer_unary_target_mode="max_quality",
        )
    )
    losses = criterion.compute_pointer_selection_loss(outputs, targets)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())
    assert unary.grad is not None and bool(torch.isfinite(unary.grad).all())
    assert float(losses["mean_cluster_support"]) == 2.0
    assert float(losses["teacher_fallback_count"]) == 0.0


def test_fixed_row_grid_does_not_put_the_last_lane_row_at_one() -> None:
    fixed = _selector(row_grid_mode="fixed_rows")._lane_row_grid(
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    legacy = _selector(row_grid_mode="legacy_linspace")._lane_row_grid(
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert float(fixed[-1]) == 7.0 / 8.0
    assert float(legacy[-1]) == 1.0


def test_v4_5_trajectory_summary_reads_pointer_coverage_schema() -> None:
    def metric(f1: float, recall: float):
        return {
            "f1": f1,
            "precision": f1,
            "recall": recall,
            "mean_selected_per_image": 3.3,
            "false_positive_breakdown": {
                "duplicate_fp": {"fraction_of_fp": 0.1}
            },
        }

    report = {
        "methods": {
            "pointer_greedy": {
                "0.50": metric(0.81, 0.80),
                "0.75": metric(0.61, 0.60),
            }
        },
        "capacity": {
            "0.50": {"all_candidate_oracle": {"recall": 0.95}},
            "0.75": {"all_candidate_oracle": {"recall": 0.82}},
        },
    }
    row = _trajectory_row(57500, report)
    assert row["iteration"] == 57500
    assert row["f1_050"] == 0.81
    assert row["f1_075"] == 0.61
    assert row["duplicate_fp_fraction_050"] == 0.1
