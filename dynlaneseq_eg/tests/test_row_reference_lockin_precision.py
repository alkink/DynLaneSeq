from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.structured_queries import StructuredLaneQueryHead
from dynlaneseq_eg.tools.analyze_row_reference_lockin import (
    CONDITIONS,
    _global_reacquire_reference,
    _reference_counterfactual,
)
from dynlaneseq_eg.tools.analyze_row_reference_precision import (
    _oracle_hits,
    classify_false_positive,
)


def _head() -> StructuredLaneQueryHead:
    return StructuredLaneQueryHead(
        dim=16,
        num_instances=4,
        num_rows=6,
        x_bins=12,
        input_w=48,
        num_heads=4,
        num_layers=4,
        ff_dim=32,
        dropout=0.0,
        evidence_x_bins=12,
        num_groups=1,
        row_reference={
            "enabled": True,
            "offsets_px": [-12.0, -6.0, 0.0, 6.0, 12.0],
            "initial_prior_sigma_px": 16.0,
            "output_prior_sigma_px": 8.0,
        },
    ).eval()


def test_global_reacquisition_preserves_reference_contract() -> None:
    torch.manual_seed(41)
    head = _head()
    row_tokens = torch.randn(2, 4, 6, 16)
    row_values = torch.randn(2, 6, 12, 16)
    current = torch.rand(2, 4, 6) * 47.0
    with torch.inference_mode():
        rescued = _global_reacquire_reference(
            head,
            row_tokens,
            row_values,
            current,
            prior_sigma_px=16.0,
            prior_strength=1.0,
        )
        wrong = _global_reacquire_reference(
            head,
            row_tokens,
            torch.roll(row_values, shifts=1, dims=0),
            current,
            prior_sigma_px=16.0,
            prior_strength=1.0,
        )
    assert rescued.shape == current.shape
    assert bool(torch.isfinite(rescued).all())
    assert float(rescued.min()) >= 0.0
    assert float(rescued.max()) <= 47.0
    assert not torch.allclose(rescued, wrong)


def test_wide_offset_counterfactual_restores_checkpoint_buffers() -> None:
    head = _head()
    before = [layer.offsets_px.clone() for layer in head.layers]
    with _reference_counterfactual(
        head,
        "wide_offsets_2x",
        rescue_layer=3,
        rescue_prior_sigma_px=16.0,
        rescue_prior_strength=1.0,
    ):
        for layer, expected in zip(head.layers, before):
            torch.testing.assert_close(layer.offsets_px, expected * 2.0)
    for layer, expected in zip(head.layers, before):
        torch.testing.assert_close(layer.offsets_px, expected)


def test_every_lockin_counterfactual_runs_through_real_head() -> None:
    torch.manual_seed(43)
    head = _head()
    features = torch.randn(2, 16, 6, 12)
    outputs = {}
    with torch.inference_mode():
        for condition in CONDITIONS:
            with _reference_counterfactual(
                head,
                condition,
                rescue_layer=3,
                rescue_prior_sigma_px=16.0,
                rescue_prior_strength=1.0,
            ):
                outputs[condition] = head(features, inference_only=True)[
                    "pred_x_rows"
                ]
    assert all(value.shape == (2, 4, 6) for value in outputs.values())
    assert not torch.allclose(outputs["normal"], outputs["wide_offsets_2x"])
    assert not torch.allclose(outputs["normal"], outputs["global_rescue_mid"])


def test_false_positive_taxonomy_distinguishes_failure_modes() -> None:
    best = torch.tensor([0.9, 0.8, 0.4, 0.1])
    assigned = {0}
    common = {
        "assigned_proposals": assigned,
        "best_iou": best,
        "gt_count": 2,
        "iou_threshold": 0.5,
        "near_min_iou": 0.3,
    }
    assert classify_false_positive(proposal_index=0, **common) == "true_positive"
    assert classify_false_positive(proposal_index=1, **common) == "duplicate_fp"
    assert classify_false_positive(proposal_index=2, **common) == "near_miss_fp"
    assert classify_false_positive(proposal_index=3, **common) == "background_fp"
    assert (
        classify_false_positive(
            proposal_index=3,
            assigned_proposals=set(),
            best_iou=best,
            gt_count=0,
            iou_threshold=0.5,
            near_min_iou=0.3,
        )
        == "empty_scene_fp"
    )


def test_oracle_pool_is_duplicate_safe_and_topk_limited() -> None:
    # Two predictions support GT0; the third supports GT1. Top-1 may recover
    # only one lane, while Top-2 can recover both without double counting.
    iou = torch.tensor([[0.9, 0.8, 0.0], [0.0, 0.0, 0.85]])
    assert _oracle_hits(iou, [0, 1, 2], threshold=0.5, top_k=1) == 1
    assert _oracle_hits(iou, [0, 1, 2], threshold=0.5, top_k=2) == 2
