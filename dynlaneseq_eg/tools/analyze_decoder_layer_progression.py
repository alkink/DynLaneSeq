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
    gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
    valid = target["valid_mask"].to(device=candidates.device).bool()
    for lane_index in range(int(gt_x.shape[0])):
        if int(valid[lane_index].sum()) < 5:
            continue
        ious = line_iou_against_gt(candidates, gt_x[lane_index], valid[lane_index], line_width=line_width)
        stats.update(float(ious.max()) if ious.numel() else 0.0)


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
        for layer_name, stage in layers:
            pred_x = stage["pred_x_rows"].float()
            all_key = f"{layer_name}/all{num_instances}"
            group_key = f"{layer_name}/group0_{group_size}"
            recalls.setdefault(all_key, ProposalRecallStats(thresholds=thresholds))
            recalls.setdefault(group_key, ProposalRecallStats(thresholds=thresholds))
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

        final = layers[-1][1]
        log_matches = matcher_log(final, targets)
        prob_matches = matcher_prob(final, targets)
        log_iou, log_count = _assigned_line_iou(final, targets, log_matches, float(args.line_width))
        prob_iou, prob_count = _assigned_line_iou(final, targets, prob_matches, float(args.line_width))
        assigned_iou["neg_log_probability"][0] += log_iou
        assigned_iou["neg_log_probability"][1] += log_count
        assigned_iou["neg_probability"][0] += prob_iou
        assigned_iou["neg_probability"][1] += prob_count
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
