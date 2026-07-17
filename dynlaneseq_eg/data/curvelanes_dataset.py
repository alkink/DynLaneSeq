from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from .lane_target_builder import LaneTargetBuilder, TargetBuilderConfig
from .transforms import LaneTransforms, TransformConfig


@dataclass(frozen=True)
class CurveLanesRecord:
    image_path: Path
    annotation_path: Path | None
    raw_file: str


class CurveLanesDataset(Dataset):
    """Raw CurveLanes reader using the DynLaneSeq fixed-row target contract.

    CurveLanes mixes three native image geometries.  We reproduce the public
    CurveLanes preprocessing used by CondLaneNet: remove the sky portion
    before augmentation and resize the remaining road crop to the model input.
    The original crop offset is retained in ``meta`` so predictions can be
    mapped back to the raw image for the official validation metric.
    """

    _DEFAULT_SPLITS: dict[str, dict[str, str | bool]] = {
        "train": {"list": "train/train.txt", "image_root": "train", "labels": True},
        "val": {"list": "valid/valid.txt", "image_root": "valid", "labels": True},
        "test": {"list": "test/test.txt", "image_root": "test", "labels": False},
    }
    # (width, height) -> number of source rows removed from the top.
    _CROP_Y_BY_SIZE: dict[tuple[int, int], int] = {
        (2560, 1440): 640,
        (1570, 660): 180,
        (1280, 720): 368,
    }

    def __init__(self, cfg: dict[str, Any], split: str = "train", training: bool = False):
        self.cfg = cfg
        self.root = Path(cfg.get("root", "dataset/curvelanes")).expanduser()
        self.split = str(split)
        self.training = bool(training)
        self.input_w = int(cfg.get("input_w", 800))
        self.input_h = int(cfg.get("input_h", 320))
        self.rasterize_segmentation = bool(cfg.get("rasterize_segmentation", True))
        self.seg_line_width = int(cfg.get("seg_line_width", 16))
        self.strict_image_shapes = bool(cfg.get("strict_image_shapes", True))

        split_cfg = self._resolve_split_cfg(cfg, self.split)
        self.list_path = self._resolve_path(str(split_cfg["list"]))
        self.image_root = self._resolve_path(str(split_cfg["image_root"]))
        self.labels_required = bool(split_cfg.get("labels", self.split != "test"))
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
        # The source-specific crop is deliberately performed in __getitem__.
        # A global cut_height cannot express CurveLanes' three raw geometries.
        self.transforms = LaneTransforms(
            TransformConfig(
                input_w=self.input_w,
                input_h=self.input_h,
                cut_height=0,
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
        with Image.open(rec.image_path) as raw_image:
            image = raw_image.convert("RGB")
        orig_w, orig_h = image.size
        crop_y = self._crop_y(orig_w, orig_h, rec.image_path)
        crop_w, crop_h = orig_w, orig_h - crop_y
        image = image.crop((0, crop_y, orig_w, orig_h))

        lanes = self._read_and_crop_lanes(rec.annotation_path, crop_y, crop_w, crop_h)
        seg_mask = None
        if self.training and self.rasterize_segmentation:
            seg_mask = self._rasterize_segmentation((crop_w, crop_h), lanes)

        image_tensor, lanes_aug, seg_tensor, aug_meta = self.transforms(
            image,
            lanes,
            seg_mask=seg_mask,
            training=self.training,
        )
        # LaneTransforms sees an already cropped image.  Preserve the source
        # crop in metadata for a correct inverse mapping at evaluation time.
        aug_meta["crop_x"] = 0.0
        aug_meta["crop_y"] = float(crop_y)
        aug_meta["crop_w"] = float(crop_w)
        aug_meta["crop_h"] = float(crop_h)

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

        meta = {
            "dataset": "CurveLanes",
            "sample_index": int(index),
            "split": self.split,
            "image_path": str(rec.image_path),
            "anno_path": str(rec.annotation_path) if rec.annotation_path else "",
            "raw_file": rec.raw_file,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "input_h": self.input_h,
            "input_w": self.input_w,
            "scale_x": self.input_w / float(crop_w),
            "scale_y": self.input_h / float(crop_h),
            "num_gt_lanes": int(targets["x_rows"].shape[0]),
            **aug_meta,
        }
        return {"image": image_tensor, "targets": targets, "meta": meta}

    def _resolve_split_cfg(self, cfg: dict[str, Any], split: str) -> dict[str, Any]:
        split_cfg = cfg.get("splits", {}).get(split, self._DEFAULT_SPLITS.get(split))
        if split_cfg is None:
            raise KeyError(f"No CurveLanes split configuration for split={split!r}")
        if "list" not in split_cfg or "image_root" not in split_cfg:
            raise KeyError(f"CurveLanes split {split!r} requires list and image_root")
        return dict(split_cfg)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    def _read_records(self) -> list[CurveLanesRecord]:
        if not self.list_path.is_file():
            raise FileNotFoundError(f"CurveLanes split list not found: {self.list_path}")
        records: list[CurveLanesRecord] = []
        with self.list_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw_file = line.strip().lstrip("/")
                if not raw_file:
                    continue
                image_path = self.image_root / raw_file
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    raise ValueError(
                        f"Unexpected CurveLanes image entry at {self.list_path}:{line_number}: {raw_file!r}"
                    )
                annotation_path: Path | None = None
                if self.labels_required:
                    annotation_path = image_path.parent.parent / "labels" / f"{image_path.stem}.lines.json"
                records.append(CurveLanesRecord(image_path, annotation_path, raw_file))
        if not records:
            raise ValueError(f"CurveLanes split list is empty: {self.list_path}")
        return records

    def _crop_y(self, width: int, height: int, image_path: Path) -> int:
        crop_y = self._CROP_Y_BY_SIZE.get((width, height))
        if crop_y is not None:
            return int(crop_y)
        if not self.strict_image_shapes:
            return 0
        known = ", ".join(f"{w}x{h}" for w, h in sorted(self._CROP_Y_BY_SIZE))
        raise ValueError(
            f"Unsupported CurveLanes image geometry {width}x{height} for {image_path}; "
            f"known geometries are {known}. Set strict_image_shapes: false only after defining "
            "an explicit crop policy for the new geometry."
        )

    @staticmethod
    def _read_and_crop_lanes(
        annotation_path: Path | None,
        crop_y: int,
        crop_w: int,
        crop_h: int,
    ) -> list[list[tuple[float, float]]]:
        if annotation_path is None:
            return []
        if not annotation_path.is_file():
            raise FileNotFoundError(f"CurveLanes annotation not found: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw_lines = raw.get("Lines", [])
        if not isinstance(raw_lines, list):
            raise ValueError(f"CurveLanes annotation has invalid Lines field: {annotation_path}")
        lanes: list[list[tuple[float, float]]] = []
        for raw_lane in raw_lines:
            if not isinstance(raw_lane, list):
                continue
            source_points: list[tuple[float, float]] = []
            for point in raw_lane:
                try:
                    x = float(point["x"])
                    y = float(point["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                source_points.append((x, y))
            cropped = CurveLanesDataset._clip_lane_to_crop(
                source_points,
                crop_y=float(crop_y),
                crop_w=float(crop_w),
                crop_h=float(crop_h),
            )
            if len(cropped) >= 2:
                lanes.append(cropped)
        return lanes

    @staticmethod
    def _clip_lane_to_crop(
        points: list[tuple[float, float]],
        crop_y: float,
        crop_w: float,
        crop_h: float,
    ) -> list[tuple[float, float]]:
        """Clip a polyline against the source-image lower crop boundary."""
        if len(points) < 2:
            return []
        out: list[tuple[float, float]] = []

        def append_if_valid(x: float, y_source: float) -> None:
            y = y_source - crop_y
            if not (0.0 <= x < crop_w and 0.0 <= y < crop_h):
                return
            point = (float(x), float(y))
            if not out or abs(out[-1][0] - point[0]) > 1e-5 or abs(out[-1][1] - point[1]) > 1e-5:
                out.append(point)

        for index, (x0, y0) in enumerate(points):
            if index == 0:
                if y0 >= crop_y:
                    append_if_valid(x0, y0)
                continue
            x1, y1 = x0, y0
            xa, ya = points[index - 1]
            a_inside = ya >= crop_y
            b_inside = y1 >= crop_y
            if a_inside != b_inside and abs(y1 - ya) > 1e-6:
                t = (crop_y - ya) / (y1 - ya)
                append_if_valid(xa + t * (x1 - xa), crop_y)
            if b_inside:
                append_if_valid(x1, y1)
        return out

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
