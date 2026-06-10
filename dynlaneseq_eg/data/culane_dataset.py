from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset

from .lane_target_builder import LaneTargetBuilder, TargetBuilderConfig
from .transforms import LaneTransforms, TransformConfig


@dataclass(frozen=True)
class CULaneRecord:
    image_path: Path
    anno_path: Path
    seg_path: Path | None
    lane_flags: tuple[int, ...]


class CULaneDataset(Dataset):
    """CULane dataset reader using PRD fixed-row target output contract."""

    def __init__(self, cfg: dict[str, Any], split: str = "train", training: bool = False):
        self.cfg = cfg
        self.root = Path(cfg.get("root", "dataset")).expanduser()
        self.split = split
        self.training = training
        self.input_w = int(cfg.get("input_w", 800))
        self.input_h = int(cfg.get("input_h", 288))
        self.infer_seg_labels = bool(cfg.get("infer_seg_labels", False))
        self.list_path = self._resolve_list_path(cfg, split)
        self.records = self._read_records(self.list_path)
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
        lanes = self._read_lines_txt(rec.anno_path)
        seg_mask = Image.open(rec.seg_path).convert("L") if rec.seg_path and rec.seg_path.exists() else None
        seg_valid = seg_mask is not None
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
        if seg_tensor is None:
            seg_tensor = torch.zeros((1, self.input_h, self.input_w), dtype=torch.float32)
        targets["seg_mask"] = seg_tensor.float()
        targets["seg_valid"] = torch.tensor(seg_valid, dtype=torch.bool)
        meta = {
            "image_path": str(rec.image_path),
            "anno_path": str(rec.anno_path),
            "seg_path": str(rec.seg_path) if rec.seg_path else "",
            "orig_h": orig_h,
            "orig_w": orig_w,
            "input_h": self.input_h,
            "input_w": self.input_w,
            "scale_x": self.input_w / float(crop_w),
            "scale_y": self.input_h / float(crop_h),
            "num_gt_lanes": int(targets["x_rows"].shape[0]),
            "lane_flags": rec.lane_flags,
            **aug_meta,
        }
        return {"image": image_tensor, "targets": targets, "meta": meta}

    def _resolve_list_path(self, cfg: dict[str, Any], split: str) -> Path:
        list_cfg = cfg.get("lists", {})
        if split in list_cfg:
            candidate = Path(list_cfg[split])
        else:
            default_name = {"train": "list/train_gt.txt", "val": "list/val.txt", "test": "list/test.txt"}.get(
                split, f"list/{split}.txt"
            )
            candidate = Path(default_name)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate

    def _read_records(self, list_path: Path) -> list[CULaneRecord]:
        records: list[CULaneRecord] = []
        with list_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                image_rel = parts[0]
                seg_rel: str | None = None
                flags: tuple[int, ...] = tuple()
                if len(parts) >= 2 and parts[1].endswith((".png", ".jpg", ".jpeg")):
                    seg_rel = parts[1]
                    flags = tuple(int(x) for x in parts[2:] if x.lstrip("-").isdigit())
                elif len(parts) > 1:
                    flags = tuple(int(x) for x in parts[1:] if x.lstrip("-").isdigit())
                image_path = self.root / image_rel.lstrip("/")
                seg_path = self.root / seg_rel.lstrip("/") if seg_rel else None
                if seg_path is None and self.infer_seg_labels:
                    candidate = self.root / "laneseg_label_w16" / Path(image_rel.lstrip("/")).with_suffix(".png")
                    if candidate.exists():
                        seg_path = candidate
                anno_path = image_path.with_suffix(".lines.txt")
                records.append(CULaneRecord(image_path, anno_path, seg_path, flags))
        return records

    @staticmethod
    def _read_lines_txt(path: Path) -> list[list[tuple[float, float]]]:
        lanes: list[list[tuple[float, float]]] = []
        if not path.exists():
            return lanes
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                values = [float(v) for v in line.strip().split()]
                if len(values) < 4:
                    continue
                lane = [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]
                lanes.append(lane)
        return lanes
