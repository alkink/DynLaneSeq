from __future__ import annotations

import torch

from dynlaneseq_eg.tools.evaluate_culane_query_groups import (
    _aggregate_variant_metrics,
    _cpu_stage,
    _slice_candidates,
    _variant_specs,
)


def test_cpu_stage_keeps_only_postprocess_tensors() -> None:
    stage = {
        "pred_x_rows": torch.ones(1, 8, 3),
        "exist_logits": torch.ones(1, 8, 2),
        "range_norm": torch.ones(1, 8, 2),
        "nested_debug": {"large_tensor": torch.ones(4)},
        "debug_scalar": torch.tensor(1.0),
    }
    cpu_stage = _cpu_stage(stage)
    assert set(cpu_stage) == {"pred_x_rows", "exist_logits", "range_norm"}
    assert all(value.device.type == "cpu" for value in cpu_stage.values())


def test_slice_candidates_only_slices_candidate_axis() -> None:
    stage = {
        "pred_x_rows": torch.arange(2 * 8 * 3).view(2, 8, 3),
        "exist_logits": torch.arange(2 * 8 * 2).view(2, 8, 2),
        "debug_scalar": torch.tensor(1.0),
        "batch_only": torch.arange(2),
    }
    sliced = _slice_candidates(stage, [2, 3], num_candidates=8)
    assert sliced["pred_x_rows"].shape == (2, 2, 3)
    assert sliced["exist_logits"].shape == (2, 2, 2)
    assert sliced["debug_scalar"].shape == ()
    assert sliced["batch_only"].shape == (2,)


def test_variant_specs_include_reference_and_both_modes() -> None:
    specs = _variant_specs(2)
    assert [spec["name"] for spec in specs] == [
        "full_nms",
        "group0_no_nms",
        "group0_nms",
        "group1_no_nms",
        "group1_nms",
    ]


def test_metric_aggregation_preserves_precision_recall_f1() -> None:
    variants = [{"name": "full_nms", "group_index": None, "nms_enabled": True}]
    per_image = [
        {"full_nms": {0.5: [2, 1, 0]}},
        {"full_nms": {0.5: [1, 0, 1]}},
    ]
    result = _aggregate_variant_metrics(per_image, variants, [0.5])
    metrics = result["full_nms"]["results"]["0.50"]
    assert metrics["TP"] == 3
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert abs(metrics["Precision"] - 0.75) < 1e-8
    assert abs(metrics["Recall"] - 0.75) < 1e-8
    assert abs(metrics["F1"] - 0.75) < 1e-8
