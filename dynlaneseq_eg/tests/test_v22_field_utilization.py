from __future__ import annotations

import os
from pathlib import Path

import torch

from dynlaneseq_eg.modeling.v22_field_utilization import (
    candidate_field_component_scores,
    correct_curves_from_distance,
    equal_rank_ensemble,
    owned_component_centerline_score,
    source_seeded_field_path,
)
from dynlaneseq_eg.tools.audit_v22_lane_field_utilization import (
    METRIC_THREAD_ENV,
    _metric_worker,
    _one_edit_outcomes,
)
from dynlaneseq_eg.modeling.common import fixed_y_rows


def test_metric_worker_blas_threads_are_process_bounded() -> None:
    assert all(os.environ[name] == "1" for name in METRIC_THREAD_ENV)


def _outputs(rows: int, bins: int) -> dict[str, torch.Tensor]:
    return {
        "centerline_logits": torch.full((1, 1, rows, bins), -10.0),
        "distance_raw": torch.zeros((1, 1, rows, bins)),
        "support_logits": torch.full((1, 1, rows, bins), 10.0),
    }


def test_candidate_field_heads_sample_candidates_not_only_source() -> None:
    rows = 8
    output = _outputs(rows, 64)
    output["centerline_logits"][..., 20] = 10.0
    candidate_x = torch.stack(
        (
            torch.full((1, 1, rows), 20.5),
            torch.full((1, 1, rows), 40.5),
        ),
        dim=2,
    )
    candidate_range = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    scored = candidate_field_component_scores(
        output,
        candidate_x=candidate_x,
        candidate_range=candidate_range,
        candidate_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        input_w=64,
    )
    assert int(scored["centerline"].argmax(dim=2)) == 0
    assert scored["valid"].all()


def test_distance_correction_applies_exactly_one_and_two_closed_form_steps() -> None:
    rows, bins = 8, 100
    output = _outputs(rows, bins)
    output["distance_raw"].fill_(torch.atanh(torch.tensor(-0.5)))
    x = torch.full((1, 1, rows), 80.0)
    ranges = torch.tensor([[[0.0, 1.0]]])
    first, second = correct_curves_from_distance(
        output,
        x_rows=x,
        range_norm=ranges,
        input_w=100,
        distance_limit_px=40.0,
        steps=2,
    )
    torch.testing.assert_close(first, torch.full_like(first, 60.0), atol=2e-3, rtol=0)
    torch.testing.assert_close(second, torch.full_like(second, 40.0), atol=4e-3, rtol=0)


def test_equal_rank_ensemble_is_component_scale_invariant() -> None:
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    first = torch.tensor([[[1000.0, 100.0, 0.0]]])
    second = torch.tensor([[[0.0, 2.0, 1.0]]])
    combined = equal_rank_ensemble((first, second), valid)
    # Candidate 1 is second-best in the first component and best in the second.
    assert int(combined.argmax(dim=2)) == 1


def test_owned_component_upper_bound_rejects_adjacent_gt_ridge() -> None:
    rows, bins = 8, 64
    output = _outputs(rows, bins)
    output["centerline_logits"][..., 20] = 8.0
    output["centerline_logits"][..., 50] = 10.0
    candidate_x = torch.stack(
        (
            torch.full((1, 1, rows), 20.5),
            torch.full((1, 1, rows), 50.5),
        ),
        dim=2,
    )
    candidate_range = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    target = {
        "x_rows": torch.stack(
            (torch.full((rows,), 20.5), torch.full((rows,), 50.5))
        ),
        "valid_mask": torch.ones(2, rows, dtype=torch.bool),
    }
    score = owned_component_centerline_score(
        output,
        candidate_x=candidate_x,
        candidate_range=candidate_range,
        candidate_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        targets=[target],
        ownership=torch.tensor([[0]]),
        input_w=64,
    )
    assert int(score.argmax(dim=2)) == 0


def test_source_seeded_path_can_leave_source_for_stronger_global_ridge() -> None:
    rows, bins = 8, 64
    output = _outputs(rows, bins)
    output["centerline_logits"][..., 12] = 12.0
    source = torch.full((1, 1, rows), 20.5)
    ranges = torch.tensor([[[0.0, 1.0]]])
    path = source_seeded_field_path(
        output,
        source_x=source,
        source_range=ranges,
        input_w=64,
        distance_limit_px=16.0,
        offset_step_px=4.0,
        transition_scale_px=8.0,
    )
    torch.testing.assert_close(path, torch.full_like(path, 12.5))


def test_one_edit_outcomes_use_joint_hungarian_threshold_deltas() -> None:
    source = torch.tensor(
        [
            [0.40, 0.10],
            [0.10, 0.60],
        ]
    )
    candidates = torch.tensor(
        [
            [[0.80], [0.10]],
            [[0.10], [0.90]],
        ]
    )
    delta50, delta75 = _one_edit_outcomes(
        source,
        candidates,
        active=torch.tensor([True, True]),
        valid=torch.tensor([[True], [True]]),
    )
    assert int(delta50[0, 0]) == 1
    assert int(delta75[0, 0]) == 1
    assert int(delta50[1, 0]) == 0
    assert int(delta75[1, 0]) == 1


def test_metric_worker_rasterizes_all_fixed_curve_layouts(tmp_path: Path) -> None:
    rows = 8
    y = fixed_y_rows(rows, 64).tolist()
    annotation = tmp_path / "frame.lines.txt"
    annotation.write_text(
        " ".join(f"20.5 {value}" for value in y) + "\n",
        encoding="utf-8",
    )
    source = torch.full((1, rows), 20.5)
    candidate = source.unsqueeze(1)
    source_range = torch.tensor([[0.0, 1.0]])
    candidate_range = source_range.unsqueeze(1)
    payload = {
        "meta": {
            "anno_path": str(annotation),
            "input_h": 64,
            "input_w": 64,
            "orig_h": 64,
            "orig_w": 64,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "crop_x": 0.0,
            "crop_y": 0.0,
        },
        "curves": {
            "source_v7": source,
            "source_distance_step1": source,
            "source_distance_step2": source,
            "source_seeded_path": source,
            "raw": candidate,
            "distance_step1": candidate,
            "distance_step2": candidate,
        },
        "ranges": {
            "source_v7": source_range,
            "source_distance_step1": source_range,
            "source_distance_step2": source_range,
            "source_seeded_path": source_range,
            "raw": candidate_range,
            "distance_step1": candidate_range,
            "distance_step2": candidate_range,
        },
        "source_active": torch.tensor([True]),
        "top_valid": torch.tensor([[True]]),
        "ownership": torch.tensor([0]),
        "positive_slot": torch.tensor([True]),
        "beneficial": torch.tensor([[False]]),
        "harmful": torch.tensor([[False]]),
        "neutral": torch.tensor([[True]]),
        "policy_selection": {"synthetic": torch.tensor([0])},
        "gt_count": 1,
    }
    result = _metric_worker(payload)
    assert result["continuous"]["source_v7"]["tp50"] == 1
    assert result["continuous"]["source_v7"]["tp75"] == 1
    assert result["raw_cache_outcome_mismatch"] == 0
    assert result["semantics"]["synthetic"]["outcome_neutral"] == 1
