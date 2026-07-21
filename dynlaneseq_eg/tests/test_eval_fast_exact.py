from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dynlaneseq_eg.data.culane_dataset import CULaneDataset
from dynlaneseq_eg.evaluation.culane_metric import (
    discrete_cross_iou,
    eval_predictions,
    eval_predictions_with_categories,
)
from dynlaneseq_eg.modeling import DynLaneSeqS0


def _small_structured_cfg() -> dict:
    return {
        "model": {
            "name": "DynLaneSeqS0",
            "input_h": 64,
            "input_w": 128,
            "fpn_channels": 64,
            "dim": 64,
            "pretrained_backbone": False,
            "num_slots": 4,
            "num_rows": 8,
            "x_bins": 32,
            "decoder_layers": 0,
            "num_heads": 4,
            "decoder_ff_dim": 128,
            "dropout": 0.0,
            "seg_aux": {"enabled": True, "dropout": 0.0},
            "centerline_aux": {"enabled": True, "dropout": 0.0},
            "structured_query": {
                "enabled": True,
                "num_instances": 4,
                "num_groups": 1,
                "num_layers": 1,
                "num_heads": 4,
                "ff_dim": 128,
                "dropout": 0.0,
                "evidence_x_bins": 32,
            },
        }
    }


def test_structured_inference_only_is_exact_and_prunable() -> None:
    torch.manual_seed(7)
    model = DynLaneSeqS0(_small_structured_cfg()).eval()
    images = torch.randn(1, 3, 64, 128)
    required = {"exist_logits", "pred_x_rows", "range_norm", "quality_logits"}

    with torch.inference_mode():
        full = model(images)
        fast = model(images, inference_only=True)

    assert set(fast) == required
    assert "seg_logits" in full and "centerline_logits" in full
    assert full["structured_debug"] == {}
    for key in required:
        assert torch.equal(fast[key], full[key])

    model.prepare_for_inference()
    assert model.heads is None
    assert model.encoder.seg_aux_head is None
    assert model.encoder.centerline_aux_head is None
    with torch.inference_mode():
        pruned = model(images, inference_only=True)
    for key in required:
        assert torch.equal(pruned[key], full[key])


def test_inference_dataset_skips_annotations_masks_and_targets(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "driver" / "frame.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (48, 32), color=(64, 96, 128)).save(image_path)
    list_path = tmp_path / "list" / "test.txt"
    list_path.parent.mkdir(parents=True)
    list_path.write_text("/driver/frame.jpg\n", encoding="utf-8")
    dataset = CULaneDataset(
        {
            "root": str(tmp_path),
            "lists": {"test": "list/test.txt"},
            "input_w": 48,
            "input_h": 32,
            "num_rows": 8,
            "x_bins": 16,
            "load_targets": False,
            "infer_seg_labels": False,
        },
        split="test",
        training=False,
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("inference-only dataset must not construct targets")

    monkeypatch.setattr(dataset, "_read_lines_txt", fail)
    monkeypatch.setattr(dataset.target_builder, "build", fail)
    item = dataset[0]
    assert item["image"].shape == (3, 32, 48)
    assert item["targets"] == {}
    assert item["meta"]["num_gt_lanes"] == 0


def test_single_channel_raster_matches_legacy_rgb_exactly() -> None:
    pred = [np.asarray([(100, 500), (130, 350), (170, 200)], dtype=np.float32)]
    anno = [np.asarray([(102, 500), (132, 350), (172, 200)], dtype=np.float32)]
    legacy = discrete_cross_iou(pred, anno, width=30, img_shape=(590, 1640, 3))
    optimized = discrete_cross_iou(pred, anno, width=30, img_shape=(590, 1640))
    assert np.array_equal(optimized, legacy)


def _write_lane(path: Path, points: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = " ".join(f"{value:.3f}" for point in points for value in point)
    path.write_text(values + "\n", encoding="utf-8")


def test_category_metrics_are_aggregated_from_one_exact_pass(tmp_path: Path) -> None:
    pred_dir = tmp_path / "pred"
    anno_dir = tmp_path / "anno"
    main_list = tmp_path / "test.txt"
    normal_list = tmp_path / "normal.txt"
    cross_list = tmp_path / "cross.txt"
    main_list.write_text("/a/one.jpg\n/a/two.jpg\n", encoding="utf-8")
    normal_list.write_text("/a/one.jpg\n", encoding="utf-8")
    cross_list.write_text("/a/two.jpg\n", encoding="utf-8")
    lane = [(100.0, 500.0), (120.0, 350.0), (145.0, 200.0)]
    _write_lane(pred_dir / "a" / "one.lines.txt", lane)
    _write_lane(anno_dir / "a" / "one.lines.txt", lane)
    _write_lane(pred_dir / "a" / "two.lines.txt", lane)

    overall, categories = eval_predictions_with_categories(
        pred_dir,
        anno_dir,
        main_list,
        {"normal": normal_list, "cross": cross_list},
        iou_thresholds=(0.5, 0.75),
        sequential=True,
    )
    expected_overall = eval_predictions(
        pred_dir, anno_dir, main_list, iou_thresholds=(0.5, 0.75), sequential=True
    )
    expected_normal = eval_predictions(
        pred_dir, anno_dir, normal_list, iou_thresholds=(0.5, 0.75), sequential=True
    )
    expected_cross = eval_predictions(
        pred_dir, anno_dir, cross_list, iou_thresholds=(0.5, 0.75), sequential=True
    )
    assert overall == expected_overall
    assert categories["normal"] == expected_normal
    assert categories["cross"] == expected_cross
