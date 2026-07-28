from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import (
    line_iou_against_gt,
    select_candidates,
)
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.analyze_decoder_image_grounding import (
    _group_zero_assignments,
    _head_stages,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether ResNet-34 and DLA-34 miss the same CULane lanes. "
            "This is a raw-proposal diagnostic: scores, Top-K, NMS, and the "
            "official test protocol are intentionally excluded."
        )
    )
    parser.add_argument("--r34-config", required=True)
    parser.add_argument("--r34-checkpoint", required=True)
    parser.add_argument("--dla34-config", required=True)
    parser.add_argument("--dla34-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _prepare_config(
    path: str,
    *,
    dataset_root: str,
    eval_batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg = load_config(path)
    if dataset_root:
        cfg.setdefault("dataset", {})["root"] = dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(eval_batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(num_workers)
    if int(num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False
    model_cfg.setdefault("structured_query", {})["intermediate_supervision"] = True
    return cfg


def _build_frozen_model(
    cfg: dict[str, Any],
    checkpoint: str,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    model = build_model(cfg)
    iteration = load_checkpoint(checkpoint, model, strict=False)
    return model.to(device).eval(), int(iteration)


def _best_iou(
    candidates: torch.Tensor,
    gt_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    line_width: float,
) -> float:
    if int(valid.sum()) < 5 or int(candidates.shape[0]) == 0:
        return 0.0
    values = line_iou_against_gt(
        candidates,
        gt_x,
        valid,
        line_width=float(line_width),
    )
    return float(values.max()) if values.numel() else 0.0


def _query_gt_iou_matrix(
    candidates: torch.Tensor,
    gt_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    line_width: float,
) -> torch.Tensor:
    columns: list[torch.Tensor] = []
    for lane_index in range(int(gt_x.shape[0])):
        if int(valid[lane_index].sum()) < 5:
            columns.append(candidates.new_zeros((int(candidates.shape[0]),)))
            continue
        columns.append(
            line_iou_against_gt(
                candidates,
                gt_x[lane_index],
                valid[lane_index],
                line_width=float(line_width),
            )
        )
    if not columns:
        return candidates.new_zeros((int(candidates.shape[0]), 0))
    return torch.stack(columns, dim=1)


@dataclass
class QuerySpecializationStats:
    group_size: int
    images: int = 0
    lanes: int = 0
    assigned_count: list[int] = field(init=False)
    assigned_iou_total: list[float] = field(init=False)
    assigned_rank_counts: list[Counter[int]] = field(init=False)
    best_iou_total: list[float] = field(init=False)
    useful_030: list[int] = field(init=False)
    useful_050: list[int] = field(init=False)
    best_rank_counts_030: list[Counter[int]] = field(init=False)
    active_queries_030_total: int = 0
    active_queries_050_total: int = 0
    lane_supporters_030_total: int = 0
    lane_supporters_050_total: int = 0
    unassigned_useful_030_total: int = 0
    unassigned_useful_050_total: int = 0

    def __post_init__(self) -> None:
        self.assigned_count = [0 for _ in range(self.group_size)]
        self.assigned_iou_total = [0.0 for _ in range(self.group_size)]
        self.assigned_rank_counts = [Counter() for _ in range(self.group_size)]
        self.best_iou_total = [0.0 for _ in range(self.group_size)]
        self.useful_030 = [0 for _ in range(self.group_size)]
        self.useful_050 = [0 for _ in range(self.group_size)]
        self.best_rank_counts_030 = [Counter() for _ in range(self.group_size)]

    def update(
        self,
        matrix: torch.Tensor,
        *,
        gt_ranks: dict[int, int],
        assignments: dict[int, int],
    ) -> None:
        self.images += 1
        self.lanes += int(matrix.shape[1])
        assigned_queries = set(assignments.values())
        if int(matrix.shape[1]) == 0:
            return
        query_best_iou, query_best_gt = matrix.max(dim=1)
        active_030 = query_best_iou >= 0.3
        active_050 = query_best_iou >= 0.5
        self.active_queries_030_total += int(active_030.sum())
        self.active_queries_050_total += int(active_050.sum())
        self.lane_supporters_030_total += int((matrix >= 0.3).sum())
        self.lane_supporters_050_total += int((matrix >= 0.5).sum())
        for query_index in range(self.group_size):
            best_iou = float(query_best_iou[query_index])
            self.best_iou_total[query_index] += best_iou
            if best_iou >= 0.3:
                self.useful_030[query_index] += 1
                gt_index = int(query_best_gt[query_index])
                self.best_rank_counts_030[query_index][gt_ranks[gt_index]] += 1
                if query_index not in assigned_queries:
                    self.unassigned_useful_030_total += 1
            if best_iou >= 0.5:
                self.useful_050[query_index] += 1
                if query_index not in assigned_queries:
                    self.unassigned_useful_050_total += 1
        for gt_index, query_index in assignments.items():
            if query_index >= self.group_size or gt_index >= int(matrix.shape[1]):
                continue
            self.assigned_count[query_index] += 1
            self.assigned_iou_total[query_index] += float(matrix[query_index, gt_index])
            self.assigned_rank_counts[query_index][gt_ranks[gt_index]] += 1

    @staticmethod
    def _counter_dict(counter: Counter[int]) -> dict[str, int]:
        return {str(key): int(counter[key]) for key in sorted(counter)}

    def summary(self) -> dict[str, Any]:
        image_count = max(self.images, 1)
        lane_count = max(self.lanes, 1)
        per_query = []
        for query_index in range(self.group_size):
            assigned = self.assigned_count[query_index]
            per_query.append(
                {
                    "query_index": query_index,
                    "assignment_rate_per_image": assigned / image_count,
                    "mean_assigned_iou": (
                        self.assigned_iou_total[query_index] / max(assigned, 1)
                    ),
                    "assigned_lane_rank_counts": self._counter_dict(
                        self.assigned_rank_counts[query_index]
                    ),
                    "mean_best_gt_iou": self.best_iou_total[query_index] / image_count,
                    "useful_iou030_image_fraction": (
                        self.useful_030[query_index] / image_count
                    ),
                    "useful_iou050_image_fraction": (
                        self.useful_050[query_index] / image_count
                    ),
                    "best_lane_rank_counts_when_iou030": self._counter_dict(
                        self.best_rank_counts_030[query_index]
                    ),
                }
            )
        return {
            "images": self.images,
            "lanes": self.lanes,
            "mean_active_queries_iou030_per_image": (
                self.active_queries_030_total / image_count
            ),
            "mean_active_queries_iou050_per_image": (
                self.active_queries_050_total / image_count
            ),
            "mean_query_supporters_iou030_per_lane": (
                self.lane_supporters_030_total / lane_count
            ),
            "mean_query_supporters_iou050_per_lane": (
                self.lane_supporters_050_total / lane_count
            ),
            "mean_unassigned_useful_iou030_per_image": (
                self.unassigned_useful_030_total / image_count
            ),
            "mean_unassigned_useful_iou050_per_image": (
                self.unassigned_useful_050_total / image_count
            ),
            "per_query": per_query,
        }


def _lane_shape_stats(
    gt_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    input_w: int,
) -> dict[str, float | int]:
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    valid_rows = int(valid_indices.numel())
    if valid_rows == 0:
        return {
            "valid_rows": 0,
            "row_span": 0,
            "mean_x_norm": 0.0,
            "curvature_px": 0.0,
        }
    values = gt_x[valid_indices].float()
    curvature_parts: list[torch.Tensor] = []
    if valid_rows >= 3:
        consecutive = (
            (valid_indices[1:-1] - valid_indices[:-2] == 1)
            & (valid_indices[2:] - valid_indices[1:-1] == 1)
        )
        second = values[2:] - 2.0 * values[1:-1] + values[:-2]
        if bool(consecutive.any()):
            curvature_parts.append(second[consecutive].abs())
    curvature = (
        float(torch.cat(curvature_parts).mean())
        if curvature_parts
        else 0.0
    )
    return {
        "valid_rows": valid_rows,
        "row_span": int(valid_indices[-1] - valid_indices[0] + 1),
        "mean_x_norm": float(values.mean()) / max(float(input_w), 1.0),
        "curvature_px": curvature,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(centered_left, centered_right))
    left_energy = sum(value * value for value in centered_left)
    right_energy = sum(value * value for value in centered_right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator > 0.0 else 0.0


def _bucket_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "lanes": len(records),
        "mean_valid_rows": _mean([float(row["valid_rows"]) for row in records]),
        "mean_row_span": _mean([float(row["row_span"]) for row in records]),
        "mean_curvature_px": _mean([float(row["curvature_px"]) for row in records]),
        "mean_lanes_in_image": _mean(
            [float(row["lanes_in_image"]) for row in records]
        ),
        "mean_x_norm": _mean([float(row["mean_x_norm"]) for row in records]),
        "mean_r34_iou": _mean([float(row["r34_l4_group0_iou"]) for row in records]),
        "mean_dla34_iou": _mean(
            [float(row["dla34_l4_group0_iou"]) for row in records]
        ),
    }


def summarize_records(
    records: list[dict[str, Any]],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "lanes": len(records),
        "best_iou_pearson_r34_vs_dla34": _pearson(
            [float(row["r34_l4_group0_iou"]) for row in records],
            [float(row["dla34_l4_group0_iou"]) for row in records],
        ),
        "thresholds": {},
    }
    for threshold in thresholds:
        r_key = "r34_l4_group0_iou"
        d_key = "dla34_l4_group0_iou"
        buckets: dict[str, list[dict[str, Any]]] = {
            "common_hit": [],
            "r34_only": [],
            "dla34_only": [],
            "common_miss": [],
        }
        for row in records:
            r_hit = float(row[r_key]) >= threshold
            d_hit = float(row[d_key]) >= threshold
            if r_hit and d_hit:
                bucket = "common_hit"
            elif r_hit:
                bucket = "r34_only"
            elif d_hit:
                bucket = "dla34_only"
            else:
                bucket = "common_miss"
            buckets[bucket].append(row)
        lane_count = max(len(records), 1)
        r_hits = len(buckets["common_hit"]) + len(buckets["r34_only"])
        d_hits = len(buckets["common_hit"]) + len(buckets["dla34_only"])
        union_hits = lane_count - len(buckets["common_miss"])
        threshold_key = f"{threshold:.2f}"
        result["thresholds"][threshold_key] = {
            "r34_recall": r_hits / lane_count,
            "dla34_recall": d_hits / lane_count,
            "cross_backbone_union_recall": union_hits / lane_count,
            "union_gain_over_r34": (union_hits - r_hits) / lane_count,
            "union_gain_over_dla34": (union_hits - d_hits) / lane_count,
            "common_miss_fraction": len(buckets["common_miss"]) / lane_count,
            "status_counts": {
                name: len(values) for name, values in buckets.items()
            },
            "status_difficulty": {
                name: _bucket_summary(values) for name, values in buckets.items()
            },
            "r34_cross_layer_union_recall": (
                sum(
                    float(row["r34_layer_union_group0_iou"]) >= threshold
                    for row in records
                )
                / lane_count
            ),
            "dla34_cross_layer_union_recall": (
                sum(
                    float(row["dla34_layer_union_group0_iou"]) >= threshold
                    for row in records
                )
                / lane_count
            ),
            "cross_backbone_all32_union_recall": (
                sum(
                    float(row["cross_backbone_l4_all32_union_iou"]) >= threshold
                    for row in records
                )
                / lane_count
            ),
            "r34_all32_recall": (
                sum(float(row["r34_l4_all32_iou"]) >= threshold for row in records)
                / lane_count
            ),
            "dla34_all32_recall": (
                sum(float(row["dla34_l4_all32_iou"]) >= threshold for row in records)
                / lane_count
            ),
            "r34_model_top4_recall": (
                sum(float(row["r34_l4_model_top4_iou"]) >= threshold for row in records)
                / lane_count
            ),
            "dla34_model_top4_recall": (
                sum(float(row["dla34_l4_model_top4_iou"]) >= threshold for row in records)
                / lane_count
            ),
        }
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    r_cfg = _prepare_config(
        args.r34_config,
        dataset_root=args.dataset_root,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    d_cfg = _prepare_config(
        args.dla34_config,
        dataset_root=args.dataset_root,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    r_model, r_iteration = _build_frozen_model(
        r_cfg,
        args.r34_checkpoint,
        device,
    )
    d_model, d_iteration = _build_frozen_model(
        d_cfg,
        args.dla34_checkpoint,
        device,
    )
    r_head = r_model.structured_query_head
    d_head = d_model.structured_query_head
    if r_head is None or d_head is None:
        raise ValueError("Both checkpoints must use structured_query heads")
    if int(r_head.num_rows) != int(d_head.num_rows):
        raise ValueError("The two heads must use the same row count")
    if int(r_head.input_w) != int(d_head.input_w):
        raise ValueError("The two heads must use the same input width")
    r_matcher = build_matcher(r_cfg)
    d_matcher = build_matcher(d_cfg)
    r_group_size = int(r_head.num_instances) // int(r_head.num_groups)
    d_group_size = int(d_head.num_instances) // int(d_head.num_groups)
    loader = build_dataloader(r_cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
    input_w = int(r_head.input_w)
    records: list[dict[str, Any]] = []
    query_stats = {
        "r34": QuerySpecializationStats(r_group_size),
        "dla34": QuerySpecializationStats(d_group_size),
    }
    images_seen = 0

    total = len(loader)
    if int(args.max_batches) > 0:
        total = min(total, int(args.max_batches))
    progress = tqdm(
        enumerate(loader),
        total=total,
        desc="cross-backbone error overlap",
        ncols=100,
    )
    for batch_index, (images, targets, metas) in progress:
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        images = images.to(device)
        targets = nested_to_device(targets, device)
        with _amp_context(device, amp_dtype):
            r_p2 = r_model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )["features"]
            d_p2 = d_model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )["features"]
        r_stages = _head_stages(r_head, r_p2, amp_dtype)
        d_stages = _head_stages(d_head, d_p2, amp_dtype)
        r_stage_names = sorted(r_stages, key=lambda name: int(name[1:]))
        d_stage_names = sorted(d_stages, key=lambda name: int(name[1:]))
        r_final = r_stages[r_stage_names[-1]]["pred_x_rows"]
        d_final = d_stages[d_stage_names[-1]]["pred_x_rows"]
        r_top4 = [
            select_candidates(
                r_stages[r_stage_names[-1]],
                image_index,
                top_k=4,
                rank_by="score_quality",
            )
            for image_index in range(int(images.shape[0]))
        ]
        d_top4 = [
            select_candidates(
                d_stages[d_stage_names[-1]],
                image_index,
                top_k=4,
                rank_by="score_quality",
            )
            for image_index in range(int(images.shape[0]))
        ]
        r_matches = r_matcher(r_stages[r_stage_names[-1]], targets)
        d_matches = d_matcher(d_stages[d_stage_names[-1]], targets)

        for image_index, target in enumerate(targets):
            gt_x_all = target["x_rows"].float()
            valid_all = target["valid_mask"].bool()
            lanes_in_image = int(gt_x_all.shape[0])
            lane_mean_x = []
            for gt_index in range(lanes_in_image):
                valid = valid_all[gt_index]
                lane_mean_x.append(
                    float(gt_x_all[gt_index, valid].mean())
                    if bool(valid.any())
                    else float("inf")
                )
            lane_order = sorted(range(lanes_in_image), key=lane_mean_x.__getitem__)
            gt_ranks = {
                gt_index: rank for rank, gt_index in enumerate(lane_order)
            }
            r_matrix = _query_gt_iou_matrix(
                r_final[image_index, :r_group_size],
                gt_x_all,
                valid_all,
                line_width=args.line_width,
            )
            d_matrix = _query_gt_iou_matrix(
                d_final[image_index, :d_group_size],
                gt_x_all,
                valid_all,
                line_width=args.line_width,
            )
            query_stats["r34"].update(
                r_matrix,
                gt_ranks=gt_ranks,
                assignments=_group_zero_assignments(
                    r_matches[image_index],
                    group_size=r_group_size,
                ),
            )
            query_stats["dla34"].update(
                d_matrix,
                gt_ranks=gt_ranks,
                assignments=_group_zero_assignments(
                    d_matches[image_index],
                    group_size=d_group_size,
                ),
            )
            r_layer_union = torch.cat(
                [
                    r_stages[name]["pred_x_rows"][image_index, :r_group_size]
                    for name in r_stage_names
                ],
                dim=0,
            )
            d_layer_union = torch.cat(
                [
                    d_stages[name]["pred_x_rows"][image_index, :d_group_size]
                    for name in d_stage_names
                ],
                dim=0,
            )
            cross_group0 = torch.cat(
                [
                    r_final[image_index, :r_group_size],
                    d_final[image_index, :d_group_size],
                ],
                dim=0,
            )
            cross_all32 = torch.cat(
                [r_final[image_index], d_final[image_index]],
                dim=0,
            )
            for gt_index in range(lanes_in_image):
                valid = valid_all[gt_index]
                if int(valid.sum()) < 5:
                    continue
                gt_x = gt_x_all[gt_index]
                row: dict[str, Any] = {
                    "image_index": images_seen + image_index,
                    "dataset_index": int(
                        sampled_indices[images_seen + image_index]
                    ),
                    "image_path": str(metas[image_index].get("image_path", "")),
                    "gt_index": gt_index,
                    "lane_rank_left_to_right": int(gt_ranks[gt_index]),
                    "lanes_in_image": lanes_in_image,
                    **_lane_shape_stats(
                        gt_x,
                        valid,
                        input_w=input_w,
                    ),
                }
                for prefix, stages, group_size in (
                    ("r34", r_stages, r_group_size),
                    ("dla34", d_stages, d_group_size),
                ):
                    for stage_name in sorted(
                        stages,
                        key=lambda name: int(name[1:]),
                    ):
                        row[f"{prefix}_{stage_name.lower()}_group0_iou"] = _best_iou(
                            stages[stage_name]["pred_x_rows"][
                                image_index, :group_size
                            ],
                            gt_x,
                            valid,
                            line_width=args.line_width,
                        )
                row["r34_l4_all32_iou"] = _best_iou(
                    r_final[image_index],
                    gt_x,
                    valid,
                    line_width=args.line_width,
                )
                row["dla34_l4_all32_iou"] = _best_iou(
                    d_final[image_index],
                    gt_x,
                    valid,
                    line_width=args.line_width,
                )
                row["r34_l4_model_top4_iou"] = _best_iou(
                    r_top4[image_index],
                    gt_x,
                    valid,
                    line_width=args.line_width,
                )
                row["dla34_l4_model_top4_iou"] = _best_iou(
                    d_top4[image_index],
                    gt_x,
                    valid,
                    line_width=args.line_width,
                )
                row["r34_layer_union_group0_iou"] = _best_iou(
                    r_layer_union,
                    gt_x,
                    valid,
                    line_width=args.line_width,
                )
                row["dla34_layer_union_group0_iou"] = _best_iou(
                    d_layer_union,
                    gt_x,
                    valid,
                    line_width=args.line_width,
                )
                row["cross_backbone_l4_group0_union_iou"] = _best_iou(
                    cross_group0,
                    gt_x,
                    valid,
                    line_width=args.line_width,
                )
                row["cross_backbone_l4_all32_union_iou"] = _best_iou(
                    cross_all32,
                    gt_x,
                    valid,
                    line_width=args.line_width,
                )
                records.append(row)
        images_seen += int(images.shape[0])

    thresholds = tuple(float(value) for value in args.iou_thresholds)
    summary = summarize_records(records, thresholds)
    payload = {
        "diagnostic_only": True,
        "warning": (
            "Cross-backbone and cross-layer unions are oracle diagnostics, "
            "not deployable predictions or benchmark results."
        ),
        "r34": {
            "config": args.r34_config,
            "checkpoint": args.r34_checkpoint,
            "iteration": r_iteration,
        },
        "dla34": {
            "config": args.dla34_config,
            "checkpoint": args.dla34_checkpoint,
            "iteration": d_iteration,
        },
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "line_width": float(args.line_width),
        "summary": summary,
        "query_specialization": {
            name: stats.summary() for name, stats in query_stats.items()
        },
        "lane_records": records,
    }
    print(json.dumps({key: value for key, value in payload.items() if key != "lane_records"}, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
