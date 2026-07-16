from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from .lane_target_builder import LaneTargetBuilder, TargetBuilderConfig
from .transforms import LaneTransforms, TransformConfig


@dataclass(frozen=True)
class TuSimpleRecord:
    image_path: Path
    annotation_path: Path
    raw_file: str
    lanes: tuple[tuple[tuple[float, float], ...], ...]
    h_samples: tuple[int, ...]


class TuSimpleDataset(Dataset):
    """TuSimple JSON-lines reader using the DynLaneSeq fixed-row target contract."""

    _DEFAULT_SPLITS: dict[str, dict[str, Any]] = {
        "train": {
            "image_root": "train_set",
            "annotations": [
                "train_set/label_data_0313.json",
                "train_set/label_data_0601.json",
                "train_set/label_data_0531.json",
            ],
        },
        "val": {
            "image_root": "train_set",
            "annotations": ["train_set/label_data_0531.json"],
        },
        "test": {
            "image_root": "test_set",
            "annotations": ["test_label.json"],
        },
    }

    def __init__(self, cfg: dict[str, Any], split: str = "train", training: bool = False):
        self.cfg = cfg
        self.root = Path(cfg.get("root", "dataset/tusimple")).expanduser()
        self.split = str(split)
        self.training = bool(training)
        self.input_w = int(cfg.get("input_w", 800))
        self.input_h = int(cfg.get("input_h", 320))
        self.rasterize_segmentation = bool(cfg.get("rasterize_segmentation", True))
        self.seg_line_width = int(cfg.get("seg_line_width", 16))

        split_cfg = self._resolve_split_cfg(cfg, self.split)
        self.image_root = self._resolve_path(split_cfg["image_root"])
        annotation_values = split_cfg["annotations"]
        if isinstance(annotation_values, (str, Path)):
            annotation_values = [annotation_values]
        self.annotation_paths = [self._resolve_path(value) for value in annotation_values]
        self.records = self._read_records()
        max_samples = cfg.get("num_samples")
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]

        self.target_builder = LaneTargetBuilder(
            TargetBuilderConfig(
                input_w=self.input_w,
                input_h=self.input_h,
                num_rows=int(cfg.get("num_rows", 72)),
                x_bins=int(cfg.get("x_bins", 200)),
                min_valid_rows=int(cfg.get("min_valid_rows", 5)),
                token_ignore_index=int(cfg.get("token_ignore_index", -100)),
            )
        )
        aug = cfg.get("augmentation", {})
        self.transforms = LaneTransforms(
            TransformConfig(
                input_w=self.input_w,
                input_h=self.input_h,
                cut_height=int(aug.get("cut_height", cfg.get("cut_height", 0))),
                horizontal_flip_prob=float(aug.get("horizontal_flip_prob", 0.0)),
                color_jitter=bool(aug.get("color_jitter", False)),
                channel_shuffle_prob=float(aug.get("channel_shuffle_prob", 0.0)),
                hue_saturation_prob=float(aug.get("hue_saturation_prob", 0.0)),
                blur_prob=float(aug.get("blur_prob", 0.0)),
                affine_prob=float(aug.get("affine_prob", 0.0)),
                affine_translate_x=float(aug.get("affine_translate_x", 0.0)),
                affine_translate_y=float(aug.get("affine_translate_y", 0.0)),
                affine_rotate_deg=float(aug.get("affine_rotate_deg", 0.0)),
                affine_scale_min=float(aug.get("affine_scale_min", 1.0)),
                affine_scale_max=float(aug.get("affine_scale_max", 1.0)),
                random_shadow_prob=float(aug.get("random_shadow_prob", 0.0)),
                random_shadow_min_opacity=float(aug.get("random_shadow_min_opacity", 0.25)),
                random_shadow_max_opacity=float(aug.get("random_shadow_max_opacity", 0.55)),
                random_shadow_min_vertices=int(aug.get("random_shadow_min_vertices", 3)),
                random_shadow_max_vertices=int(aug.get("random_shadow_max_vertices", 6)),
                random_shadow_roi_start_y=float(aug.get("random_shadow_roi_start_y", 0.25)),
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        image = Image.open(rec.image_path).convert("RGB")
        orig_w, orig_h = image.size
        lanes = [[(float(x), float(y)) for x, y in lane] for lane in rec.lanes]
        seg_mask = None
        if self.training and self.rasterize_segmentation:
            seg_mask = self._rasterize_segmentation((orig_w, orig_h), lanes)

        image_tensor, lanes_aug, seg_tensor, aug_meta = self.transforms(
            image,
            lanes,
            seg_mask=seg_mask,
            training=self.training,
        )
        crop_w = int(round(float(aug_meta.get("crop_w", orig_w))))
        crop_h = int(round(float(aug_meta.get("crop_h", orig_h))))
        target_np = self.target_builder.build(lanes_aug, orig_w=crop_w, orig_h=crop_h)
        targets = {
            "x_rows": torch.from_numpy(target_np["x_rows"]).float(),
            "x_bins": torch.from_numpy(target_np["x_bins"]).long(),
            "valid_mask": torch.from_numpy(target_np["valid_mask"]).bool(),
            "range_y": torch.from_numpy(target_np["range_y"]).float(),
            "exist": torch.from_numpy(target_np["exist"]).long(),
        }
        seg_valid = seg_tensor is not None
        if seg_tensor is None:
            seg_tensor = torch.zeros((1, self.input_h, self.input_w), dtype=torch.float32)
        targets["seg_mask"] = seg_tensor.float()
        targets["seg_valid"] = torch.tensor(seg_valid, dtype=torch.bool)

        crop_y = float(aug_meta.get("crop_y", 0.0))
        meta = {
            "dataset": "TuSimple",
            "sample_index": int(index),
            "image_path": str(rec.image_path),
            "anno_path": str(rec.annotation_path),
            "raw_file": rec.raw_file,
            "h_samples": rec.h_samples,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "input_h": self.input_h,
            "input_w": self.input_w,
            "scale_x": self.input_w / float(crop_w),
            "scale_y": self.input_h / float(crop_h),
            "crop_x": float(aug_meta.get("crop_x", 0.0)),
            "crop_y": crop_y,
            "num_gt_lanes": int(targets["x_rows"].shape[0]),
            **aug_meta,
        }
        return {"image": image_tensor, "targets": targets, "meta": meta}

    def _resolve_split_cfg(self, cfg: dict[str, Any], split: str) -> dict[str, Any]:
        split_table = cfg.get("splits", {})
        split_cfg = split_table.get(split, self._DEFAULT_SPLITS.get(split))
        if split_cfg is None:
            raise KeyError(f"No TuSimple split configuration for split={split!r}")
        if "image_root" not in split_cfg or "annotations" not in split_cfg:
            raise KeyError(f"TuSimple split {split!r} requires image_root and annotations")
        return dict(split_cfg)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    def _read_records(self) -> list[TuSimpleRecord]:
        records: list[TuSimpleRecord] = []
        for annotation_path in self.annotation_paths:
            if not annotation_path.is_file():
                raise FileNotFoundError(f"TuSimple annotation file not found: {annotation_path}")
            with annotation_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    raw_file = str(data["raw_file"]).lstrip("/")
                    h_samples = tuple(int(y) for y in data["h_samples"])
                    raw_lanes = data["lanes"]
                    if any(len(lane) != len(h_samples) for lane in raw_lanes):
                        raise ValueError(
                            f"Invalid TuSimple lane length in {annotation_path}:{line_number}"
                        )
                    lanes = []
                    for lane_xs in raw_lanes:
                        points = tuple(
                            (float(x), float(y))
                            for x, y in zip(lane_xs, h_samples)
                            if float(x) >= 0.0
                        )
                        if len(points) >= 2:
                            lanes.append(points)
                    image_path = self.image_root / raw_file
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"TuSimple image referenced by {annotation_path}:{line_number} not found: {image_path}"
                        )
                    records.append(
                        TuSimpleRecord(
                            image_path=image_path,
                            annotation_path=annotation_path,
                            raw_file=raw_file,
                            lanes=tuple(lanes),
                            h_samples=h_samples,
                        )
                    )
        return records

    def _rasterize_segmentation(
        self,
        image_size: tuple[int, int],
        lanes: list[list[tuple[float, float]]],
    ) -> Image.Image:
        mask = Image.new("L", image_size, color=0)
        draw = ImageDraw.Draw(mask)
        for lane in lanes:
            if len(lane) >= 2:
                draw.line(lane, fill=1, width=max(self.seg_line_width, 1), joint="curve")
        return mask
