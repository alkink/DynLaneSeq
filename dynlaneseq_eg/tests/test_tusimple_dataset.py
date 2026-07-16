from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from dynlaneseq_eg.data.tusimple_dataset import TuSimpleDataset


def _write_annotation(path: Path, raw_file: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "raw_file": raw_file,
        "h_samples": [16, 24, 32, 40, 48, 56, 64, 71],
        "lanes": [[64, 64, 64, 64, 64, 64, 64, 64]],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_tusimple_dataset_builds_fixed_row_targets_and_segmentation(tmp_path: Path) -> None:
    image_path = tmp_path / "train_set" / "clips" / "sample" / "20.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (128, 72), color=(40, 40, 40)).save(image_path)
    annotation_path = tmp_path / "train_set" / "labels.json"
    _write_annotation(annotation_path, "clips/sample/20.jpg")

    cfg = {
        "name": "TuSimple",
        "root": str(tmp_path),
        "splits": {
            "train": {
                "image_root": "train_set",
                "annotations": ["train_set/labels.json"],
            }
        },
        "input_w": 160,
        "input_h": 64,
        "num_rows": 16,
        "x_bins": 80,
        "min_valid_rows": 2,
        "rasterize_segmentation": True,
        "seg_line_width": 4,
        "augmentation": {"cut_height": 16},
    }
    dataset = TuSimpleDataset(cfg, split="train", training=True)
    item = dataset[0]

    assert len(dataset) == 1
    assert tuple(item["image"].shape) == (3, 64, 160)
    assert tuple(item["targets"]["x_rows"].shape) == (1, 16)
    assert int(item["targets"]["valid_mask"].sum()) >= 10
    assert bool(item["targets"]["seg_valid"])
    assert float(item["targets"]["seg_mask"].sum()) > 0
    assert item["meta"]["raw_file"] == "clips/sample/20.jpg"
    assert item["meta"]["crop_y"] == 16.0
    assert item["meta"]["scale_x"] == 1.25


def test_tusimple_eval_item_keeps_official_h_samples(tmp_path: Path) -> None:
    image_path = tmp_path / "test_set" / "clips" / "sample" / "20.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (128, 72)).save(image_path)
    annotation_path = tmp_path / "test_label.json"
    _write_annotation(annotation_path, "clips/sample/20.jpg")
    cfg = {
        "root": str(tmp_path),
        "input_w": 160,
        "input_h": 64,
        "num_rows": 16,
        "x_bins": 80,
        "augmentation": {"cut_height": 16},
    }

    item = TuSimpleDataset(cfg, split="test", training=False)[0]
    assert item["meta"]["h_samples"] == (16, 24, 32, 40, 48, 56, 64, 71)
    assert not bool(item["targets"]["seg_valid"])
