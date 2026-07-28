from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import ProposalRecallStats, line_iou_against_gt
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.losses import HungarianMatcherS0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short diagnostic for decoder-layer geometry progression and matcher-cost sensitivity. "
            "It performs one model pass per batch and never writes benchmark predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="none")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _prediction_layers(outputs: dict[str, Any]) -> list[tuple[str, dict[str, torch.Tensor]]]:
    layers: list[tuple[str, dict[str, torch.Tensor]]] = []
    aux = outputs.get("aux_outputs")
    if isinstance(aux, (list, tuple)):
        for index, stage in enumerate(aux, start=1):
            if isinstance(stage, dict) and "pred_x_rows" in stage:
                layers.append((f"L{index}", stage))
    layers.append((f"L{len(layers) + 1}", outputs))
    return layers


def _normalized_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return float((entropy / math.log(max(int(logits.shape[-1]), 2))).mean().cpu())


def _update_recall(
    stats: ProposalRecallStats,
    candidates: torch.Tensor,
    target: dict[str, torch.Tensor],
    line_width: float,
) -> None:
    for value in _best_lane_ious(candidates, target, line_width):
        stats.update(float(value))


def _best_lane_ious(
    candidates: torch.Tensor,
    target: dict[str, torch.Tensor],
    line_width: float,
) -> torch.Tensor:
    gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
    valid = target["valid_mask"].to(device=candidates.device).bool()
    values: list[torch.Tensor] = []
    for lane_index in range(int(gt_x.shape[0])):
        if int(valid[lane_index].sum()) < 5:
            continue
        ious = line_iou_against_gt(candidates, gt_x[lane_index], valid[lane_index], line_width=line_width)
        values.append(ious.max() if ious.numel() else candidates.new_zeros(()))
    if not values:
        return candidates.new_zeros((0,), dtype=torch.float32)
    return torch.stack(values).float()


def _assigned_line_iou(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    line_width: float,
) -> tuple[float, int]:
    total = 0.0
    count = 0
    pred_x = outputs["pred_x_rows"]
    for batch_index, match in enumerate(matches):
        pred_ids = match["pred_indices"].to(pred_x.device)
        gt_ids = match["gt_indices"].to(pred_x.device)
        for pred_id, gt_id in zip(pred_ids.tolist(), gt_ids.tolist()):
            target = targets[batch_index]
            gt_x = target["x_rows"][gt_id].to(device=pred_x.device, dtype=pred_x.dtype)
            valid = target["valid_mask"][gt_id].to(device=pred_x.device).bool()
            iou = line_iou_against_gt(
                pred_x[batch_index, pred_id : pred_id + 1],
                gt_x,
                valid,
                line_width=line_width,
            )
            total += float(iou[0]) if iou.numel() else 0.0
            count += 1
    return total, count


def _assignment_map(
    match: dict[str, torch.Tensor],
    group_size: int,
) -> dict[tuple[int, int], int]:
    pred_ids = match["pred_indices"].tolist()
    gt_ids = match["gt_indices"].tolist()
    return {(int(pred_id) // group_size, int(gt_id)): int(pred_id) for pred_id, gt_id in zip(pred_ids, gt_ids)}


_TRANSITION_BUCKETS = (
    ("all", 0.0, math.inf),
    ("error_0_4px", 0.0, 4.0),
    ("error_4_8px", 4.0, 8.0),
    ("error_8_16px", 8.0, 16.0),
    ("error_16px_plus", 16.0, math.inf),
)


def _empty_transition_bucket() -> dict[str, float]:
    return {
        "rows": 0.0,
        "abs_before_sum": 0.0,
        "abs_after_sum": 0.0,
        "abs_update_sum": 0.0,
        "improved": 0.0,
        "worsened": 0.0,
        "direction_eligible": 0.0,
        "direction_correct": 0.0,
        "outside_4_before": 0.0,
        "inside_4_after_from_outside": 0.0,
        "inside_4_before": 0.0,
        "outside_4_after_from_inside": 0.0,
        "outside_8_before": 0.0,
        "inside_8_after_from_outside": 0.0,
        "inside_8_before": 0.0,
        "outside_8_after_from_inside": 0.0,
    }


class LayerTransitionStats:
    """Accumulate GT-aligned row corrections for one decoder transition."""

    def __init__(self) -> None:
        self.buckets = {name: _empty_transition_bucket() for name, _low, _high in _TRANSITION_BUCKETS}

    def update(
        self,
        current_x: torch.Tensor,
        next_x: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
        *,
        group_size: int,
    ) -> None:
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"].to(current_x.device)
            gt_ids = match["gt_indices"].to(current_x.device)
            keep = pred_ids < int(group_size)
            pred_ids = pred_ids[keep]
            gt_ids = gt_ids[keep]
            for pred_id, gt_id in zip(pred_ids.tolist(), gt_ids.tolist()):
                target = targets[batch_index]
                gt_x = target["x_rows"][gt_id].to(device=current_x.device, dtype=torch.float32)
                valid = target["valid_mask"][gt_id].to(device=current_x.device).bool()
                before_signed = gt_x - current_x[batch_index, pred_id].float()
                after_signed = gt_x - next_x[batch_index, pred_id].float()
                update = next_x[batch_index, pred_id].float() - current_x[batch_index, pred_id].float()
                finite = torch.isfinite(before_signed) & torch.isfinite(after_signed) & torch.isfinite(update)
                valid = valid & finite
                if not bool(valid.any()):
                    continue
                before_signed = before_signed[valid]
                after_signed = after_signed[valid]
                update = update[valid]
                abs_before = before_signed.abs()
                abs_after = after_signed.abs()

                for bucket_name, low, high in _TRANSITION_BUCKETS:
                    selected = abs_before >= float(low)
                    if math.isfinite(high):
                        selected = selected & (abs_before < float(high))
                    if not bool(selected.any()):
                        continue
                    self._update_bucket(
                        self.buckets[bucket_name],
                        before_signed[selected],
                        abs_before[selected],
                        abs_after[selected],
                        update[selected],
                    )

    @staticmethod
    def _update_bucket(
        bucket: dict[str, float],
        before_signed: torch.Tensor,
        abs_before: torch.Tensor,
        abs_after: torch.Tensor,
        update: torch.Tensor,
    ) -> None:
        count = float(abs_before.numel())
        bucket["rows"] += count
        bucket["abs_before_sum"] += float(abs_before.sum())
        bucket["abs_after_sum"] += float(abs_after.sum())
        bucket["abs_update_sum"] += float(update.abs().sum())
        bucket["improved"] += float((abs_after < abs_before).sum())
        bucket["worsened"] += float((abs_after > abs_before).sum())

        direction_eligible = abs_before > 4.0
        bucket["direction_eligible"] += float(direction_eligible.sum())
        bucket["direction_correct"] += float(
            ((update * before_signed > 0.0) & direction_eligible).sum()
        )

        outside_4 = abs_before > 4.0
        inside_4 = ~outside_4
        bucket["outside_4_before"] += float(outside_4.sum())
        bucket["inside_4_after_from_outside"] += float(((abs_after <= 4.0) & outside_4).sum())
        bucket["inside_4_before"] += float(inside_4.sum())
        bucket["outside_4_after_from_inside"] += float(((abs_after > 4.0) & inside_4).sum())

        outside_8 = abs_before > 8.0
        inside_8 = ~outside_8
        bucket["outside_8_before"] += float(outside_8.sum())
        bucket["inside_8_after_from_outside"] += float(((abs_after <= 8.0) & outside_8).sum())
        bucket["inside_8_before"] += float(inside_8.sum())
        bucket["outside_8_after_from_inside"] += float(((abs_after > 8.0) & inside_8).sum())

    def summary(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, bucket in self.buckets.items():
            rows = max(bucket["rows"], 1.0)
            result[name] = {
                "rows": int(bucket["rows"]),
                "mae_before_px": bucket["abs_before_sum"] / rows,
                "mae_after_px": bucket["abs_after_sum"] / rows,
                "mean_error_reduction_px": (
                    bucket["abs_before_sum"] - bucket["abs_after_sum"]
                )
                / rows,
                "mean_abs_update_px": bucket["abs_update_sum"] / rows,
                "improved_fraction": bucket["improved"] / rows,
                "worsened_fraction": bucket["worsened"] / rows,
                "direction_accuracy_over_4px": bucket["direction_correct"]
                / max(bucket["direction_eligible"], 1.0),
                "rescue_to_4px_fraction": bucket["inside_4_after_from_outside"]
                / max(bucket["outside_4_before"], 1.0),
                "harm_from_4px_fraction": bucket["outside_4_after_from_inside"]
                / max(bucket["inside_4_before"], 1.0),
                "rescue_to_8px_fraction": bucket["inside_8_after_from_outside"]
                / max(bucket["outside_8_before"], 1.0),
                "harm_from_8px_fraction": bucket["outside_8_after_from_inside"]
                / max(bucket["inside_8_before"], 1.0),
            }
        return result


class RecallTransitionStats:
    """Track whether lane-level proposal hits are retained or newly discovered."""

    def __init__(self, thresholds: tuple[float, ...]) -> None:
        self.thresholds = thresholds
        self.counts = {
            threshold: {
                "gt": 0,
                "retained": 0,
                "lost": 0,
                "gained": 0,
                "remained_miss": 0,
                "iou_delta_sum": 0.0,
                "iou_improved": 0,
            }
            for threshold in thresholds
        }

    def update(self, before: torch.Tensor, after: torch.Tensor) -> None:
        if before.shape != after.shape:
            raise ValueError(f"Recall-transition shape mismatch: {before.shape} vs {after.shape}")
        for threshold, counts in self.counts.items():
            before_hit = before >= float(threshold)
            after_hit = after >= float(threshold)
            counts["gt"] += int(before.numel())
            counts["retained"] += int((before_hit & after_hit).sum())
            counts["lost"] += int((before_hit & ~after_hit).sum())
            counts["gained"] += int((~before_hit & after_hit).sum())
            counts["remained_miss"] += int((~before_hit & ~after_hit).sum())
            counts["iou_delta_sum"] += float((after - before).sum())
            counts["iou_improved"] += int((after > before).sum())

    def summary(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for threshold, counts in self.counts.items():
            gt = max(int(counts["gt"]), 1)
            before_hits = int(counts["retained"]) + int(counts["lost"])
            before_misses = int(counts["gained"]) + int(counts["remained_miss"])
            result[f"{threshold:.2f}"] = {
                **counts,
                "before_recall": before_hits / gt,
                "after_recall": (int(counts["retained"]) + int(counts["gained"])) / gt,
                "retention_fraction": int(counts["retained"]) / max(before_hits, 1),
                "loss_fraction": int(counts["lost"]) / max(before_hits, 1),
                "recovery_fraction": int(counts["gained"]) / max(before_misses, 1),
                "mean_best_iou_delta": float(counts["iou_delta_sum"]) / gt,
                "iou_improved_fraction": int(counts["iou_improved"]) / gt,
            }
        return result


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg.setdefault("dataloader", {})["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg.setdefault("model", {})["require_pretrained_backbone"] = False
    structured_cfg = cfg.setdefault("model", {}).setdefault("structured_query", {})
    structured_cfg["intermediate_supervision"] = True

    device = torch.device(args.device)
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    if bool(cfg.get("training", {}).get("channels_last", False)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    loader = build_dataloader(cfg, split=args.split, training=False)

    matcher_log = build_matcher(cfg)
    matcher_prob = HungarianMatcherS0(replace(matcher_log.cfg, object_cost_type="neg_probability"))
    num_instances = int(structured_cfg.get("num_instances", cfg.get("model", {}).get("num_slots", 0)))
    num_groups = int(structured_cfg.get("num_groups", 1))
    group_size = num_instances // max(num_groups, 1)
    thresholds = tuple(float(value) for value in args.iou_thresholds)

    recalls: dict[str, ProposalRecallStats] = {}
    entropy_sum: dict[str, float] = {}
    entropy_count: dict[str, int] = {}
    layer_delta_sum: dict[str, float] = {}
    layer_delta_count: dict[str, int] = {}
    assignment_images = 0
    changed_assignment_images = 0
    assignment_keys = 0
    changed_assignment_keys = 0
    assigned_iou = {"neg_log_probability": [0.0, 0], "neg_probability": [0.0, 0]}
    transition_stats: dict[str, LayerTransitionStats] = {}
    recall_transition_stats: dict[str, RecallTransitionStats] = {}
    image_count = 0

    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    autocast_enabled = amp_dtype is not None and device.type == "cuda"

    for batch_index, (images, targets, _metas) in enumerate(
        tqdm(loader, ncols=88, desc="decoder progression")
    ):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last
                if bool(cfg.get("training", {}).get("channels_last", False)) and device.type == "cuda"
                else torch.contiguous_format
            ),
        )
        amp_context = (
            torch.autocast(device_type=device.type, enabled=True, dtype=amp_dtype)
            if autocast_enabled
            else nullcontext()
        )
        with amp_context:
            outputs = model(images)
        layers = _prediction_layers(outputs)
        image_count += int(images.shape[0])

        previous_x = None
        group0_best_by_layer: dict[str, list[torch.Tensor]] = {}
        for layer_name, stage in layers:
            pred_x = stage["pred_x_rows"].float()
            all_key = f"{layer_name}/all{num_instances}"
            group_key = f"{layer_name}/group0_{group_size}"
            recalls.setdefault(all_key, ProposalRecallStats(thresholds=thresholds))
            recalls.setdefault(group_key, ProposalRecallStats(thresholds=thresholds))
            group0_best_by_layer[layer_name] = []
            for sample_index, target in enumerate(targets):
                _update_recall(
                    recalls[all_key],
                    pred_x[sample_index],
                    target,
                    line_width=float(args.line_width),
                )
                _update_recall(
                    recalls[group_key],
                    pred_x[sample_index, :group_size],
                    target,
                    line_width=float(args.line_width),
                )
                group0_best_by_layer[layer_name].append(
                    _best_lane_ious(
                        pred_x[sample_index, :group_size],
                        target,
                        line_width=float(args.line_width),
                    )
                )
            logits = stage.get("row_x_logits")
            if isinstance(logits, torch.Tensor):
                entropy_sum[layer_name] = entropy_sum.get(layer_name, 0.0) + _normalized_entropy(logits)
                entropy_count[layer_name] = entropy_count.get(layer_name, 0) + 1
            if previous_x is not None:
                delta_key = f"{previous_x[0]}->{layer_name}"
                layer_delta_sum[delta_key] = layer_delta_sum.get(delta_key, 0.0) + float(
                    (pred_x - previous_x[1]).abs().mean().cpu()
                )
                layer_delta_count[delta_key] = layer_delta_count.get(delta_key, 0) + 1
            previous_x = (layer_name, pred_x)

        for (current_name, _current), (next_name, _next) in zip(layers[:-1], layers[1:]):
            transition_name = f"{current_name}->{next_name}"
            accumulator = recall_transition_stats.setdefault(
                transition_name,
                RecallTransitionStats(thresholds),
            )
            for before, after in zip(
                group0_best_by_layer[current_name],
                group0_best_by_layer[next_name],
            ):
                accumulator.update(before, after)

        final = layers[-1][1]
        log_matches = matcher_log(final, targets)
        prob_matches = matcher_prob(final, targets)
        log_iou, log_count = _assigned_line_iou(final, targets, log_matches, float(args.line_width))
        prob_iou, prob_count = _assigned_line_iou(final, targets, prob_matches, float(args.line_width))
        assigned_iou["neg_log_probability"][0] += log_iou
        assigned_iou["neg_log_probability"][1] += log_count
        assigned_iou["neg_probability"][0] += prob_iou
        assigned_iou["neg_probability"][1] += prob_count
        for (current_name, current), (next_name, next_stage) in zip(layers[:-1], layers[1:]):
            transition_name = f"{current_name}->{next_name}"
            transition_stats.setdefault(transition_name, LayerTransitionStats()).update(
                current["pred_x_rows"],
                next_stage["pred_x_rows"],
                targets,
                log_matches,
                group_size=group_size,
            )
        for log_match, prob_match in zip(log_matches, prob_matches):
            assignment_images += 1
            log_map = _assignment_map(log_match, group_size)
            prob_map = _assignment_map(prob_match, group_size)
            keys = set(log_map) | set(prob_map)
            changed = sum(log_map.get(key) != prob_map.get(key) for key in keys)
            assignment_keys += len(keys)
            changed_assignment_keys += changed
            changed_assignment_images += int(changed > 0)

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "images": image_count,
        "num_workers": int(args.num_workers),
        "line_width": float(args.line_width),
        "num_instances": num_instances,
        "num_groups": num_groups,
        "group_size": group_size,
        "proposal_recall": {name: stat.summary() for name, stat in sorted(recalls.items())},
        "normalized_distribution_entropy": {
            name: entropy_sum[name] / max(entropy_count[name], 1) for name in sorted(entropy_sum)
        },
        "mean_abs_x_change_px": {
            name: layer_delta_sum[name] / max(layer_delta_count[name], 1) for name in sorted(layer_delta_sum)
        },
        "group0_gt_aligned_layer_transitions": {
            name: transition_stats[name].summary() for name in sorted(transition_stats)
        },
        "group0_lane_recall_transitions": {
            name: recall_transition_stats[name].summary() for name in sorted(recall_transition_stats)
        },
        "matcher_cost_sensitivity": {
            "images_with_changed_assignment_fraction": changed_assignment_images / max(assignment_images, 1),
            "changed_assignment_fraction": changed_assignment_keys / max(assignment_keys, 1),
            "mean_assigned_line_iou": {
                name: total / max(count, 1) for name, (total, count) in assigned_iou.items()
            },
        },
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
