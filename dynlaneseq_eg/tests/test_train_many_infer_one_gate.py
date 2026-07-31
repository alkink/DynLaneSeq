from __future__ import annotations

from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.structured_queries import StructuredLaneQueryHead
from dynlaneseq_eg.tools.summarize_train_many_infer_one_gate import summarize


TRAIN_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_slots32_g4train_b4x4_1600x640_"
    "bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml"
)
DEPLOY_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_slots32_g4train_g1infer_b4x4_"
    "1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml"
)


def test_row_reference_train_many_infer_one_config_contract() -> None:
    train = load_config(TRAIN_CONFIG)
    deploy = load_config(DEPLOY_CONFIG)

    assert train["model"]["structured_query"]["row_reference"]["enabled"] is True
    assert train["model"]["structured_query"]["num_groups"] == 4
    assert train["matcher"]["assignment"] == "grouped_one_to_many"
    assert train["matcher"]["num_groups"] == 4
    assert train["matcher"]["lambda_obj"] == 0.5
    assert train["matcher"]["line_iou_radius"] == 15.0
    assert train["model"]["structured_query"]["intermediate_supervision"] is True
    assert train["loss"]["w_smooth"] == 0.0
    assert train["training"]["seed"] == 3407
    assert train["scheduler"]["total_iters"] == 278000

    assert deploy["model"]["structured_query"]["inference_group_index"] == 0
    assert deploy["postprocess"]["score_mode"] == "quality"
    assert deploy["postprocess"]["quality_score_power"] == 1.0
    assert deploy["postprocess"]["lane_nms_distance_thresh_px"] == 0.0
    assert deploy["postprocess"]["top_k"] == 4


def test_row_reference_full_training_output_matches_group_zero_inference() -> None:
    torch.manual_seed(31)
    head = StructuredLaneQueryHead(
        dim=32,
        num_instances=8,
        num_rows=8,
        x_bins=16,
        input_w=64,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
        dropout=0.0,
        evidence_x_bins=12,
        num_groups=2,
        inference_group_index=0,
        row_reference={
            "enabled": True,
            "offsets_px": [-16.0, 0.0, 16.0],
            "initial_prior_sigma_px": 16.0,
            "output_prior_sigma_px": 8.0,
        },
    ).eval()
    features = torch.randn(2, 32, 8, 12)
    with torch.inference_mode():
        full = head(features)
        group_zero = head(features, inference_only=True)

    for key in ("exist_logits", "pred_x_rows", "range_norm", "quality_logits"):
        torch.testing.assert_close(group_zero[key], full[key][:, :4])


def _metric(f1: float) -> dict[str, Any]:
    return {
        "tp": 10,
        "fp": 2,
        "fn": 3,
        "precision": f1,
        "recall": f1,
        "f1": f1,
    }


def _row(
    strategy: str,
    iou: float,
    *,
    recall: float,
    top_k: int,
    quality_power: float | None = None,
    score_threshold: float | None = None,
    official_f1: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": "main",
        "strategy": strategy,
        "top_k": top_k,
        "iou_threshold": iou,
        "quality_power": quality_power,
        "score_threshold": score_threshold,
        "hits": int(round(100 * recall)),
        "gt": 100,
        "recall": recall,
        "mean_best_iou": recall,
        "images": 16,
    }
    if official_f1 is not None:
        row[f"official_iou_{iou:g}"] = _metric(official_f1)
    return row


def _report(candidate_count: int, arm: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for iou, all_recall, oracle, quality in (
        (0.5, 0.80, 0.76, 0.70),
        (0.75, 0.60, 0.34, 0.30),
    ):
        if arm == "candidate_all":
            all_recall += 0.01
            oracle += 0.01
        elif arm == "candidate_group":
            all_recall -= 0.01
            oracle -= 0.01
            quality = oracle - 0.02
        rows.extend(
            [
                _row("all_raw", iou, recall=all_recall, top_k=0),
                _row("oracle_topk", iou, recall=oracle, top_k=4),
                _row("quality_topk", iou, recall=quality, top_k=4),
            ]
        )
        if arm == "base":
            for threshold, f1 in ((0.10, 0.70), (0.20, 0.72)):
                rows.append(
                    _row(
                        "model_topk_nms",
                        iou,
                        recall=0.68,
                        top_k=4,
                        quality_power=0.25,
                        score_threshold=threshold,
                        official_f1=f1 - (0.08 if iou == 0.75 else 0.0),
                    )
                )
        if arm == "candidate_group":
            for threshold, f1 in ((0.10, 0.73), (0.20, 0.75)):
                rows.append(
                    _row(
                        "quality_topk_nms",
                        iou,
                        recall=quality,
                        top_k=4,
                        quality_power=1.0,
                        score_threshold=threshold,
                        official_f1=f1 - (0.08 if iou == 0.75 else 0.0),
                    )
                )
    return {
        "metadata": {
            "split": "val",
            "list_sha256": "same",
            "max_batches": 64,
            "num_records": 256,
            "iou_space": "official_raster",
            "sample_strategy": "uniform",
            "sampled_dataset_indices": list(range(256)),
            "candidate_counts_by_stage": {"main": candidate_count},
        },
        "rows": rows,
    }


def test_train_many_infer_one_gate_detects_positive_selection_signal() -> None:
    payload = summarize(
        _report(32, "base"),
        _report(32, "candidate_all"),
        _report(8, "candidate_group"),
    )

    assert payload["gate"]["geometry_preserved"] is True
    assert payload["gate"]["selection_positive"] is True
    assert payload["gate"]["verdict"] == "positive_continue_full_validation"
    assert payload["candidate_counts"] == {
        "base": 32,
        "candidate_all": 32,
        "candidate_group": 8,
    }
