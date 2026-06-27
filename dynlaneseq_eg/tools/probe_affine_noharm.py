"""Frozen mixed-candidate probe for gated affine correction.

The probe sees every deployed Top-K proposal, not only near misses.  It learns
two tasks from frozen S0 representations:

1. gate: should this proposal receive affine correction?
2. correction: bounded affine delta a0 + a1*y for repairable near misses.

Images are separated into train/calibration/test partitions.  The gate threshold
is selected only on calibration data under a maximum action rate on existing
true positives.  Final rescue/kill counts are recomputed in official CULane
raster IoU space on the untouched test partition.  Correction is applied after
NMS/Top-K, deliberately isolating the no-harm question from NMS reorderings.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    _raster_lane_mask,
    candidate_row_masks,
    diagnostic_iou_matrix,
    evaluator_hungarian_assignment,
    lanes_to_original,
    override_eval_list,
    trace_postprocess,
)
from dynlaneseq_eg.evaluation.culane_metric import load_culane_img_data
from dynlaneseq_eg.evaluation.near_miss_oracles import fit_polynomial_delta
from dynlaneseq_eg.evaluation.proposal_recall import collect_prediction_stages
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows
from dynlaneseq_eg.tools.probe_affine_decodability import (
    _canonical_id,
    _chunk_means,
    _geometry_feature,
    _load_torch,
    _stage_name,
)


KIND_TP = 0
KIND_REPAIRABLE = 1
KIND_OTHER_FP = 2


@dataclass(frozen=True)
class SplitSets:
    train: set[str]
    calibration: set[str]
    test: set[str]


class GatedAffineProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Linear(hidden_dim, 1)
        self.coeff = nn.Linear(hidden_dim, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.gate(hidden).squeeze(-1), self.coeff(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage", default="main")
    parser.add_argument("--score-thresh", type=float, default=0.40)
    parser.add_argument("--quality-power", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--near-min-iou", type=float, default=0.30)
    parser.add_argument("--near-max-iou", type=float, default=0.50)
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--max-affine-displacement", type=float, default=64.0)
    parser.add_argument("--row-segments", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--split-group",
        choices=["image", "sequence"],
        default="sequence",
        help="Keep all frames from one CULane sequence in the same partition.",
    )
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument(
        "--feature-kinds",
        nargs="+",
        choices=["geometry", "q_ins", "pooled", "segmented"],
        default=["geometry", "q_ins", "segmented"],
    )
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--max-tp-action-rate", type=float, default=0.005)
    parser.add_argument("--min-gate-precision", type=float, default=0.50)
    parser.add_argument("--threshold-steps", type=int, default=1001)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def split_images(image_ids: list[str], args: argparse.Namespace) -> SplitSets:
    if args.train_fraction <= 0 or args.calibration_fraction <= 0:
        raise ValueError("train/calibration fractions must be positive")
    if args.train_fraction + args.calibration_fraction >= 1:
        raise ValueError("train_fraction + calibration_fraction must be < 1")
    ids = sorted(image_ids)
    rng = random.Random(args.split_seed)
    if args.split_group == "image":
        groups = [[image_id] for image_id in ids]
    else:
        by_sequence: dict[str, list[str]] = defaultdict(list)
        for image_id in ids:
            by_sequence[Path(image_id).parent.as_posix()].append(image_id)
        groups = [sorted(values) for _key, values in sorted(by_sequence.items())]
    rng.shuffle(groups)

    train_target = round(len(ids) * args.train_fraction)
    calibration_target = round(len(ids) * args.calibration_fraction)
    train_ids: list[str] = []
    calibration_ids: list[str] = []
    test_ids: list[str] = []
    for group in groups:
        if len(train_ids) < train_target:
            train_ids.extend(group)
        elif len(calibration_ids) < calibration_target:
            calibration_ids.extend(group)
        else:
            test_ids.extend(group)
    if not train_ids or not calibration_ids or not test_ids:
        raise RuntimeError(
            f"Invalid grouped split: train={len(train_ids)} calibration={len(calibration_ids)} "
            f"test={len(test_ids)} groups={len(groups)}"
        )
    return SplitSets(
        train=set(train_ids),
        calibration=set(calibration_ids),
        test=set(test_ids),
    )


def _split_id(image_id: str, splits: SplitSets) -> int:
    if image_id in splits.train:
        return 0
    if image_id in splits.calibration:
        return 1
    if image_id in splits.test:
        return 2
    raise KeyError(image_id)


def _gt_rasters(record: dict[str, Any], line_width: float) -> tuple[list[Any], list[int]]:
    anno_path = record["meta"].get("anno_path")
    gt_lanes = load_culane_img_data(anno_path) if anno_path else []
    image_h = int(record["meta"].get("orig_h", 590))
    image_w = int(record["meta"].get("orig_w", 1640))
    width = int(round(line_width))
    masks = [_raster_lane_mask(lane, image_h, image_w, width) for lane in gt_lanes]
    return masks, [int(mask.sum()) for mask in masks]


def exact_iou_vector(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    record: dict[str, Any],
    gt_masks: list[Any],
    gt_counts: list[int],
    line_width: float,
) -> torch.Tensor:
    values = torch.zeros(len(gt_masks), dtype=torch.float32)
    if int(pred_mask.sum()) < 2 or not gt_masks:
        return values
    input_h = int(record["meta"].get("input_h", 288))
    y_rows = fixed_y_rows(pred_x.numel(), input_h, device=pred_x.device, dtype=pred_x.dtype)
    lane = [(float(x), float(y)) for x, y in zip(pred_x[pred_mask], y_rows[pred_mask])]
    lane_original = lanes_to_original([lane], record["meta"])[0]
    image_h = int(record["meta"].get("orig_h", 590))
    image_w = int(record["meta"].get("orig_w", 1640))
    pred_raster = _raster_lane_mask(
        lane_original,
        image_h,
        image_w,
        int(round(line_width)),
    )
    pred_count = int(pred_raster.sum())
    if pred_count == 0:
        return values
    for gt_id, gt_raster in enumerate(gt_masks):
        intersection = int(cv2.countNonZero(cv2.bitwise_and(pred_raster, gt_raster)))
        union = pred_count + gt_counts[gt_id] - intersection
        values[gt_id] = 0.0 if union <= 0 else float(intersection) / float(union)
    return values


def _postprocess(
    stage: dict[str, torch.Tensor],
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = cache.get("metadata", {})
    post = metadata.get("postprocess", {})
    return trace_postprocess(
        stage,
        input_h=int(metadata.get("input_h", 288)),
        input_w=int(metadata.get("input_w", 800)),
        score_thresh=args.score_thresh,
        quality_power=args.quality_power,
        min_valid_rows=args.min_valid_rows,
        nms_distance_thresh_px=float(post.get("lane_nms_distance_thresh_px", 20.0)),
        nms_min_overlap_points=int(post.get("lane_nms_min_overlap_points", 5)),
        top_k=args.top_k,
        row_visibility_thresh=args.row_visibility_thresh,
    )


def extract_mixed_features(
    cache: dict[str, Any],
    splits: SplitSets,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = override_eval_list(load_config(args.config), args.split, args.list_path or None)
    input_h = int(cfg.get("model", {}).get("input_h", 288))
    input_w = int(cfg.get("model", {}).get("input_w", 800))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = build_dataloader(cfg, split=args.split, training=False)

    record_by_key: dict[str, tuple[int, dict[str, Any]]] = {}
    for record_index, record in enumerate(cache["records"]):
        key = _canonical_id(record["image_id"])
        if key in record_by_key:
            raise RuntimeError(f"Canonical image ID collision: {key}")
        record_by_key[key] = (record_index, record)

    tensor_lists: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "geometry",
            "q_ins",
            "pooled",
            "segmented",
            "gate_target",
            "kind",
            "coeff_target",
            "fit_mask",
            "pred_x",
            "pred_mask",
            "record_index",
            "proposal_id",
            "split",
        )
    }
    counts = defaultdict(int)
    seen: set[str] = set()
    all_split_ids = splits.train | splits.calibration | splits.test

    for images, _targets, metas in tqdm(loader, ncols=90, desc="extracting mixed no-harm features"):
        keys = [_canonical_id(meta["image_path"]) for meta in metas]
        selected_batch = [index for index, key in enumerate(keys) if key in all_split_ids]
        if not selected_batch:
            continue
        selected_keys = [keys[index] for index in selected_batch]
        with torch.inference_mode():
            outputs = model(images[selected_batch].to(device, non_blocking=True))
        stages = collect_prediction_stages(outputs)
        if args.stage in stages:
            output_stage = stages[args.stage]
        elif args.stage == "main" and len(stages) == 1:
            output_stage = next(iter(stages.values()))
        else:
            raise KeyError(f"Stage {args.stage!r} missing; available={list(stages)}")
        if "queries" not in output_stage or "structured_row_tokens" not in output_stage:
            raise KeyError("Mixed probe requires queries and structured_row_tokens")

        for local_id, image_key in enumerate(selected_keys):
            seen.add(image_key)
            record_index, record = record_by_key[image_key]
            stage_name = _stage_name(record, args.stage)
            cached_stage = record["stages"][stage_name]
            official, _valid_gt, _candidate_valid = diagnostic_iou_matrix(
                record,
                stage_name,
                use_official=True,
                input_h=input_h,
                input_w=input_w,
                line_width=args.line_width,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            trace = _postprocess(cached_stage, cache, args)
            selected_ids = list(trace["selected_ids"])
            assignment = evaluator_hungarian_assignment(official, selected_ids, args.iou_thresh)
            tp_ids = set(assignment.proposal_ids)

            target_x = record["target"]["x_rows"].float()
            target_mask = record["target"]["valid_mask"].bool() & torch.isfinite(target_x)
            valid_gt = target_mask.sum(dim=-1) >= args.min_valid_rows
            target_x = target_x[valid_gt]
            target_mask = target_mask[valid_gt]
            if target_x.shape[0] != official.shape[0]:
                counts["skipped_gt_alignment_images"] += 1
                continue

            pred_x_all, pred_mask_all, _ = candidate_row_masks(
                cached_stage,
                input_h=input_h,
                input_w=input_w,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            queries = output_stage["queries"][local_id].detach().float().cpu()
            row_tokens = output_stage["structured_row_tokens"][local_id].detach().float().cpu()
            gt_masks: list[Any] = []
            gt_counts: list[int] = []

            for proposal_id in selected_ids:
                counts["selected"] += 1
                kind = KIND_OTHER_FP
                gate_target = False
                coeff = torch.zeros(2, dtype=torch.float32)
                fit_mask = torch.zeros(pred_x_all.shape[-1], dtype=torch.bool)
                pred_x = pred_x_all[proposal_id]
                pred_mask = pred_mask_all[proposal_id]

                if proposal_id in tp_ids:
                    kind = KIND_TP
                    counts["tp"] += 1
                elif official.shape[0] > 0:
                    best_iou, gt_id_tensor = official[:, proposal_id].max(dim=0)
                    best_iou_value = float(best_iou)
                    gt_id = int(gt_id_tensor)
                    if args.near_min_iou <= best_iou_value <= args.near_max_iou:
                        counts["near_miss"] += 1
                        fitted = fit_polynomial_delta(
                            pred_x,
                            pred_mask,
                            target_x[gt_id],
                            target_mask[gt_id],
                            degree=1,
                        )
                        if fitted is not None:
                            if not gt_masks:
                                gt_masks, gt_counts = _gt_rasters(record, args.line_width)
                            y = torch.linspace(-1.0, 1.0, pred_x.numel())
                            delta = (fitted[0] + fitted[1] * y).clamp(
                                -args.max_affine_displacement,
                                args.max_affine_displacement,
                            )
                            corrected = (pred_x + delta).clamp(0.0, float(input_w - 1))
                            exact_vector = exact_iou_vector(
                                corrected,
                                pred_mask,
                                record,
                                gt_masks,
                                gt_counts,
                                args.line_width,
                            )
                            if gt_id < exact_vector.numel() and float(exact_vector[gt_id]) > args.iou_thresh:
                                kind = KIND_REPAIRABLE
                                gate_target = True
                                coeff = fitted.float()
                                fit_mask = pred_mask & target_mask[gt_id]
                                counts["repairable"] += 1
                if kind == KIND_OTHER_FP:
                    counts["other_fp"] += 1

                q_ins = queries[proposal_id]
                q_geo = row_tokens[proposal_id]
                tensor_lists["geometry"].append(
                    _geometry_feature(
                        pred_x,
                        pred_mask,
                        cached_stage,
                        proposal_id,
                        input_w,
                        args.row_segments,
                    )
                )
                tensor_lists["q_ins"].append(q_ins)
                tensor_lists["pooled"].append(
                    torch.cat([q_ins, q_geo.mean(dim=0), q_geo.amax(dim=0)])
                )
                tensor_lists["segmented"].append(
                    torch.cat([q_ins, _chunk_means(q_geo, args.row_segments)])
                )
                tensor_lists["gate_target"].append(torch.tensor(float(gate_target)))
                tensor_lists["kind"].append(torch.tensor(kind))
                tensor_lists["coeff_target"].append(coeff)
                tensor_lists["fit_mask"].append(fit_mask)
                tensor_lists["pred_x"].append(pred_x)
                tensor_lists["pred_mask"].append(pred_mask)
                tensor_lists["record_index"].append(torch.tensor(record_index))
                tensor_lists["proposal_id"].append(torch.tensor(proposal_id))
                tensor_lists["split"].append(torch.tensor(_split_id(image_key, splits)))

        if seen >= all_split_ids:
            break

    missing = all_split_ids - seen
    if missing:
        raise RuntimeError(f"Missing {len(missing)} images from dataloader; examples={sorted(missing)[:5]}")
    data: dict[str, Any] = {key: torch.stack(values) for key, values in tensor_lists.items()}
    data["metadata"] = {
        "counts": dict(counts),
        "input_h": input_h,
        "input_w": input_w,
        "split_seed": args.split_seed,
        "train_images": len(splits.train),
        "calibration_images": len(splits.calibration),
        "test_images": len(splits.test),
    }
    return data


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.bool()
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = scores.argsort(descending=True)
    sorted_labels = labels[order].float()
    precision = sorted_labels.cumsum(0) / torch.arange(1, len(labels) + 1, dtype=torch.float32)
    return float((precision * sorted_labels).sum() / positives)


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return 0.0
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    positive_rank_sum = ranks[labels].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def gate_metrics(
    scores: torch.Tensor,
    kind: torch.Tensor,
    threshold: float,
    *,
    compute_ranking: bool = True,
) -> dict[str, float | int]:
    positive = kind == KIND_REPAIRABLE
    tp = kind == KIND_TP
    other = kind == KIND_OTHER_FP
    acted = scores >= threshold
    true_actions = acted & positive
    action_count = int(acted.sum())
    result: dict[str, float | int] = {
        "threshold": float(threshold),
        "count": int(scores.numel()),
        "repairable_count": int(positive.sum()),
        "action_count": action_count,
        "precision": float(true_actions.sum() / max(action_count, 1)),
        "recall": float(true_actions.sum() / max(int(positive.sum()), 1)),
        "tp_action_rate": float((acted & tp).sum() / max(int(tp.sum()), 1)),
        "other_fp_action_rate": float((acted & other).sum() / max(int(other.sum()), 1)),
    }
    if compute_ranking:
        result["ap"] = average_precision(scores, positive)
        result["auroc"] = auroc(scores, positive)
    return result


def choose_threshold(
    scores: torch.Tensor,
    kind: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float | int]]:
    if scores.numel() == 0:
        threshold = 1.000001
        return threshold, gate_metrics(scores, kind, threshold)

    # Sigmoid scores from positive-weighted BCE commonly bunch above 0.99.
    # A fixed 0.001 grid can therefore miss the only safe operating point.
    # Evaluate every distinct score boundary exactly using cumulative counts.
    order = scores.argsort(descending=True)
    sorted_scores = scores[order]
    sorted_kind = kind[order]
    positive = (sorted_kind == KIND_REPAIRABLE).long()
    existing_tp = (sorted_kind == KIND_TP).long()
    cumulative_positive = positive.cumsum(0)
    cumulative_tp = existing_tp.cumsum(0)
    action_count = torch.arange(1, scores.numel() + 1, dtype=torch.long)
    boundary = torch.ones(scores.numel(), dtype=torch.bool)
    if scores.numel() > 1:
        boundary[:-1] = sorted_scores[:-1] > sorted_scores[1:]

    total_positive = max(int(positive.sum()), 1)
    total_tp = max(int(existing_tp.sum()), 1)
    precision = cumulative_positive.float() / action_count.float()
    tp_action_rate = cumulative_tp.float() / float(total_tp)
    feasible = boundary
    feasible &= precision >= float(args.min_gate_precision)
    feasible &= tp_action_rate <= float(args.max_tp_action_rate)
    feasible_ids = feasible.nonzero(as_tuple=False).flatten()
    if feasible_ids.numel() == 0:
        threshold = 1.000001
        return threshold, gate_metrics(scores, kind, threshold)

    best_index = int(feasible_ids[0])
    best_rank = (
        int(cumulative_positive[best_index]),
        float(precision[best_index]),
        float(sorted_scores[best_index]),
    )
    for candidate_index in feasible_ids[1:].tolist():
        rank = (
            int(cumulative_positive[candidate_index]),
            float(precision[candidate_index]),
            float(sorted_scores[candidate_index]),
        )
        if rank > best_rank:
            best_index = int(candidate_index)
            best_rank = rank
    selected_threshold = float(sorted_scores[best_index])
    return selected_threshold, gate_metrics(scores, kind, selected_threshold)


def exact_system_evaluation(
    scores: torch.Tensor,
    coefficients: torch.Tensor,
    data: dict[str, Any],
    cache: dict[str, Any],
    sample_mask: torch.Tensor,
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_ids = sample_mask.nonzero(as_tuple=False).flatten()
    local_by_record: dict[int, list[int]] = defaultdict(list)
    for local_id, sample_id in enumerate(sample_ids.tolist()):
        local_by_record[int(data["record_index"][sample_id])].append(local_id)

    baseline_tp = modified_tp = selected_total = gt_total = 0
    rescued = killed = acted_total = 0
    acted_by_kind = defaultdict(int)
    for record_index, local_ids in tqdm(
        local_by_record.items(), ncols=90, desc="official no-harm evaluation", leave=False
    ):
        record = cache["records"][record_index]
        stage_name = _stage_name(record, args.stage)
        stage = record["stages"][stage_name]
        official, _valid_gt, _candidate_valid = diagnostic_iou_matrix(
            record,
            stage_name,
            use_official=True,
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        selected = list(_postprocess(stage, cache, args)["selected_ids"])
        baseline = evaluator_hungarian_assignment(official, selected, args.iou_thresh)
        modified_matrix = official.clone()
        gt_masks: list[Any] = []
        gt_counts: list[int] = []
        for local_id in local_ids:
            score_value = float(scores[local_id])
            if not bool(torch.isfinite(scores[local_id])) or score_value < threshold:
                continue
            sample_id = int(sample_ids[local_id])
            proposal_id = int(data["proposal_id"][sample_id])
            if proposal_id not in selected:
                continue
            acted_total += 1
            acted_by_kind[int(data["kind"][sample_id])] += 1
            if not gt_masks:
                gt_masks, gt_counts = _gt_rasters(record, args.line_width)
            pred_x = data["pred_x"][sample_id]
            pred_mask = data["pred_mask"][sample_id]
            coeff = coefficients[local_id]
            y = torch.linspace(-1.0, 1.0, pred_x.numel())
            delta = (coeff[0] + coeff[1] * y).clamp(
                -args.max_affine_displacement,
                args.max_affine_displacement,
            )
            corrected = (pred_x + delta).clamp(
                0.0, float(data["metadata"].get("input_w", 800) - 1)
            )
            modified_matrix[:, proposal_id] = exact_iou_vector(
                corrected,
                pred_mask,
                record,
                gt_masks,
                gt_counts,
                args.line_width,
            )
        modified = evaluator_hungarian_assignment(modified_matrix, selected, args.iou_thresh)
        delta_tp = modified.hit_count - baseline.hit_count
        rescued += max(delta_tp, 0)
        killed += max(-delta_tp, 0)
        baseline_tp += baseline.hit_count
        modified_tp += modified.hit_count
        selected_total += len(selected)
        gt_total += int(official.shape[0])

    def summary(tp: int) -> dict[str, float | int]:
        fp = selected_total - tp
        fn = gt_total - tp
        denom = 2 * tp + fp + fn
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(denom, 1),
        }

    baseline_summary = summary(baseline_tp)
    modified_summary = summary(modified_tp)
    return {
        "threshold": threshold,
        "acted": acted_total,
        "acted_existing_tp": acted_by_kind[KIND_TP],
        "acted_repairable": acted_by_kind[KIND_REPAIRABLE],
        "acted_other_fp": acted_by_kind[KIND_OTHER_FP],
        "rescued_tp": rescued,
        "killed_tp": killed,
        "net_tp": modified_tp - baseline_tp,
        "baseline": baseline_summary,
        "modified": modified_summary,
        "f1_delta": float(modified_summary["f1"] - baseline_summary["f1"]),
    }


def train_one_probe(
    feature_name: str,
    features: torch.Tensor,
    data: dict[str, Any],
    cache: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_mask = data["split"] == 0
    calibration_mask = data["split"] == 1
    test_mask = data["split"] == 2
    train_x = features[train_mask].float()
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    normalized = (features.float() - mean) / std

    train_gate = data["gate_target"][train_mask].float()
    positives = int(train_gate.sum())
    negatives = int(train_gate.numel() - positives)
    if positives == 0:
        raise RuntimeError("No repairable training proposals")
    pos_weight = torch.tensor(negatives / max(positives, 1), dtype=torch.float32)
    train_coeff = data["coeff_target"][train_mask].float() / args.max_affine_displacement
    train_fit_mask = data["fit_mask"][train_mask].bool()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(normalized[train_mask], train_gate, train_coeff, train_fit_mask),
        batch_size=args.probe_batch_size,
        shuffle=True,
        generator=generator,
    )
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = GatedAffineProbe(features.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gate_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    regression_loss_fn = nn.SmoothL1Loss(beta=0.1)
    model.train()
    for _epoch in range(args.epochs):
        for batch_x, batch_gate, batch_coeff, batch_fit_mask in loader:
            batch_x = batch_x.to(device)
            batch_gate = batch_gate.to(device)
            batch_coeff = batch_coeff.to(device)
            batch_fit_mask = batch_fit_mask.to(device)
            gate_logits, predicted_coeff = model(batch_x)
            loss = gate_loss_fn(gate_logits, batch_gate)
            positive_rows = batch_fit_mask & (batch_gate[:, None] > 0.5)
            if bool(positive_rows.any()):
                rows = int(batch_fit_mask.shape[1])
                y = torch.linspace(-1.0, 1.0, rows, device=device).view(1, rows)
                predicted_curve = (
                    predicted_coeff[:, 0:1] + predicted_coeff[:, 1:2] * y
                ).clamp(-1.0, 1.0)
                target_curve = (
                    batch_coeff[:, 0:1] + batch_coeff[:, 1:2] * y
                ).clamp(-1.0, 1.0)
                regression_loss = regression_loss_fn(
                    predicted_curve[positive_rows], target_curve[positive_rows]
                )
                loss = loss + args.regression_weight * regression_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.inference_mode():
        gate_logits, coefficient_norm = model(normalized.to(device))
        scores = torch.sigmoid(gate_logits).cpu()
        coefficients = coefficient_norm.cpu() * args.max_affine_displacement

    threshold, calibration_metrics = choose_threshold(
        scores[calibration_mask], data["kind"][calibration_mask], args
    )
    test_gate = gate_metrics(scores[test_mask], data["kind"][test_mask], threshold)
    exact = exact_system_evaluation(
        scores[test_mask],
        coefficients[test_mask],
        data,
        cache,
        test_mask,
        threshold,
        args,
    )
    return {
        "feature": feature_name,
        "seed": seed,
        "input_dim": int(features.shape[1]),
        "train_positive": positives,
        "train_negative": negatives,
        "calibration": calibration_metrics,
        "test_gate": test_gate,
        "test_official": exact,
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["feature"])].append(run)
    result: dict[str, Any] = {}
    for feature, items in grouped.items():
        result[feature] = {
            "seeds": [item["seed"] for item in items],
            "test_gate_precision_mean": float(
                torch.tensor([item["test_gate"]["precision"] for item in items]).mean()
            ),
            "test_gate_recall_mean": float(
                torch.tensor([item["test_gate"]["recall"] for item in items]).mean()
            ),
            "test_tp_action_rate_mean": float(
                torch.tensor([item["test_gate"]["tp_action_rate"] for item in items]).mean()
            ),
            "official_net_tp_mean": float(
                torch.tensor([item["test_official"]["net_tp"] for item in items], dtype=torch.float32).mean()
            ),
            "official_net_tp_min": min(item["test_official"]["net_tp"] for item in items),
            "official_net_tp_max": max(item["test_official"]["net_tp"] for item in items),
            "official_f1_delta_mean": float(
                torch.tensor([item["test_official"]["f1_delta"] for item in items]).mean()
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    cache = _load_torch(args.candidate_cache)
    image_ids = [_canonical_id(record["image_id"]) for record in cache["records"]]
    splits = split_images(image_ids, args)
    feature_path = Path(args.feature_cache)
    if args.reuse_feature_cache and feature_path.exists():
        data = _load_torch(feature_path)
    else:
        data = extract_mixed_features(cache, splits, args)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, feature_path)

    # Split assignment is deliberately recomputed from record IDs so an
    # existing feature cache can be reused when moving from image-level to
    # leakage-safe sequence-level partitions.
    record_split = torch.full((len(cache["records"]),), -1, dtype=torch.long)
    for record_index, record in enumerate(cache["records"]):
        image_id = _canonical_id(record["image_id"])
        record_split[record_index] = _split_id(image_id, splits)
    data["split"] = record_split[data["record_index"].long()]
    data.setdefault("metadata", {})["split_seed"] = args.split_seed
    data["metadata"]["split_group"] = args.split_group
    data["metadata"]["train_images"] = len(splits.train)
    data["metadata"]["calibration_images"] = len(splits.calibration)
    data["metadata"]["test_images"] = len(splits.test)

    runs: list[dict[str, Any]] = []
    for seed in args.probe_seeds:
        for feature_name in args.feature_kinds:
            run = train_one_probe(
                feature_name,
                data[feature_name],
                data,
                cache,
                int(seed),
                args,
            )
            runs.append(run)
            official = run["test_official"]
            gate = run["test_gate"]
            print(
                f"{feature_name:>10} seed={seed}: "
                f"gateP/R={gate['precision']:.3f}/{gate['recall']:.3f} "
                f"tpAction={gate['tp_action_rate']:.4f} "
                f"rescued/killed/net={official['rescued_tp']}/"
                f"{official['killed_tp']}/{official['net_tp']} "
                f"dF1={official['f1_delta']:+.5f}"
            )

    payload = {
        "metadata": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "candidate_cache": args.candidate_cache,
            "feature_cache": str(feature_path),
            "feature_extraction": data.get("metadata", {}),
            "postprocess_order": "correction_after_nms_and_topk",
            "max_tp_action_rate": args.max_tp_action_rate,
            "min_gate_precision": args.min_gate_precision,
            "split_seed": args.split_seed,
            "split_group": args.split_group,
            "probe_seeds": args.probe_seeds,
        },
        "summary": aggregate_runs(runs),
        "runs": runs,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
