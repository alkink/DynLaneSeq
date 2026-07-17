from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from dynlaneseq_eg.data.curvelanes_dataset import CurveLanesDataset


def _write_label(path: Path, points: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"Lines": [[{"x": str(x), "y": str(y)} for x, y in points]]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_curvelanes_dataset_applies_native_crop_and_preserves_inverse_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "train" / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2560, 1440), color=(40, 40, 40)).save(image_path)
    (tmp_path / "train" / "train.txt").write_text("images/sample.jpg\n", encoding="utf-8")
    _write_label(
        tmp_path / "train" / "labels" / "sample.lines.json",
        [(1280.0, 600.0), (1280.0, 700.0), (1280.0, 900.0), (1280.0, 1200.0), (1280.0, 1439.0)],
    )
    cfg = {
        "name": "CurveLanes",
        "root": str(tmp_path),
        "splits": {"train": {"list": "train/train.txt", "image_root": "train", "labels": True}},
        "input_w": 160,
        "input_h": 64,
        "num_rows": 16,
        "x_bins": 80,
        "min_valid_rows": 2,
        "rasterize_segmentation": True,
        # This tiny smoke input downsamples 2560 px to 160 px; use a line
        # wider than production so nearest-neighbour mask resizing is visible.
        "seg_line_width": 64,
        "augmentation": {},
    }

    item = CurveLanesDataset(cfg, split="train", training=True)[0]

    assert tuple(item["image"].shape) == (3, 64, 160)
    assert tuple(item["targets"]["x_rows"].shape) == (1, 16)
    assert int(item["targets"]["valid_mask"].sum()) >= 10
    assert bool(item["targets"]["seg_valid"])
    assert float(item["targets"]["seg_mask"].sum()) > 0.0
    assert item["meta"]["raw_file"] == "images/sample.jpg"
    assert item["meta"]["crop_y"] == 640.0
    assert item["meta"]["crop_h"] == 800.0
    assert item["meta"]["scale_x"] == 160.0 / 2560.0
    assert item["meta"]["scale_y"] == 64.0 / 800.0


def test_curvelanes_dataset_rejects_unknown_native_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "valid" / "images" / "unknown.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(image_path)
    (tmp_path / "valid" / "valid.txt").write_text("images/unknown.jpg\n", encoding="utf-8")
    _write_label(tmp_path / "valid" / "labels" / "unknown.lines.json", [(20.0, 30.0), (40.0, 70.0)])
    cfg = {
        "root": str(tmp_path),
        "splits": {"val": {"list": "valid/valid.txt", "image_root": "valid", "labels": True}},
        "input_w": 160,
        "input_h": 64,
        "num_rows": 16,
        "x_bins": 80,
    }
    dataset = CurveLanesDataset(cfg, split="val", training=False)

    with pytest.raises(ValueError, match="Unsupported CurveLanes image geometry"):
        _ = dataset[0]


def test_curvelanes_reader_orders_non_monotonic_annotation_points(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.lines.json"
    _write_label(
        annotation,
        [(40.0, 900.0), (20.0, 600.0), (35.0, 800.0), (30.0, 700.0)],
    )

    lanes = CurveLanesDataset._read_and_crop_lanes(
        annotation,
        crop_y=640,
        crop_w=2560,
        crop_h=800,
    )

    assert len(lanes) == 1
    assert [point[1] for point in lanes[0]] == sorted(point[1] for point in lanes[0])


def test_curvelanes_official_unlabelled_test_layout_needs_no_list(tmp_path: Path) -> None:
    first = tmp_path / "test" / "images" / "nested" / "b.jpg"
    second = tmp_path / "test" / "images" / "a.png"
    first.parent.mkdir(parents=True)
    Image.new("RGB", (1280, 720), color=(20, 20, 20)).save(first)
    Image.new("RGB", (1280, 720), color=(30, 30, 30)).save(second)
    cfg = {
        "root": str(tmp_path),
        "input_w": 160,
        "input_h": 64,
        "num_rows": 16,
        "x_bins": 80,
    }

    dataset = CurveLanesDataset(cfg, split="test", training=False)

    assert len(dataset) == 2
    assert [record.raw_file for record in dataset.records] == [
        "images/a.png",
        "images/nested/b.jpg",
    ]
    item = dataset[0]
    assert item["meta"]["raw_file"] == "images/a.png"
    assert item["meta"]["anno_path"] == ""
    assert not bool(item["targets"]["seg_valid"])
