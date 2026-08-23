from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device


GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "backbone_all": ("encoder.backbone.",),
    "backbone_early_p2": (
        "encoder.backbone.base_layer.",
        "encoder.backbone.level0.",
        "encoder.backbone.level1.",
        "encoder.backbone.level2.",
    ),
    "backbone_mid_p3": ("encoder.backbone.level3.",),
    "backbone_deep_p4_p5": (
        "encoder.backbone.level4.",
        "encoder.backbone.level5.",
    ),
    "fpn_all": ("encoder.fpn.",),
    "fpn_p2": (
        "encoder.fpn.lateral.c2.",
        "encoder.fpn.output.",
    ),
    "fpn_p3_p5": (
        "encoder.fpn.lateral.c3.",
        "encoder.fpn.lateral.c4.",
        "encoder.fpn.lateral.c5.",
        "encoder.fpn.pyramid_outputs.",
    ),
    "row_image_projection": ("structured_query_head.feature_proj.",),
    "proposal_row_transformer": (
        "structured_query_head.layers.",
        "structured_query_head.reference_",
        "structured_query_head.lane_state_layers.",
        "structured_query_head.ownership_",
    ),
    "slot_decoder": (
        "structured_query_head.set_selection_head.slot_decoder.",
        "structured_query_head.set_selection_head.slot_tokens.",
        "structured_query_head.set_selection_head.slot_norm.",
    ),
    "slot_refiner": (
        "structured_query_head.set_selection_head.slot_refinement.",
    ),
    "joint_field_private": (
        "structured_query_head.set_selection_head.joint_slot_field.",
    ),
    "coordinate_readouts": (
        "structured_query_head.row_delta_heads.",
        "structured_query_head.range.",
        "structured_query_head.exist.",
        "structured_query_head.quality.",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether V30 field gradients agree with or oppose the "
            "mature proposal/geometry objectives at a fixed checkpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _weighted_proposal_loss(
    losses: dict[str, torch.Tensor], cfg: dict[str, Any]
) -> torch.Tensor:
    loss_cfg = cfg.get("loss", {})
    iteration = int(losses.get("iteration", 0)) if "iteration" in losses else 0
    row_warmup = int(loss_cfg.get("row_dfl_warmup_iters", 0))
    row_weight = float(loss_cfg.get("w_row_dfl", 0.0))
    if row_warmup > 0 and iteration > 0:
        row_weight *= min(1.0, iteration / row_warmup)
    terms = (
        ("loss_exist", "w_exist"),
        ("loss_point", "w_point"),
        ("loss_range", "w_range"),
        ("loss_smooth", "w_smooth"),
        ("loss_line_iou", "w_line_iou"),
        ("loss_seg", "w_seg"),
        ("loss_centerline", "w_centerline"),
    )
    total = losses["loss_total"] * 0.0
    for loss_name, weight_name in terms:
        if loss_name in losses:
            total = total + float(loss_cfg.get(weight_name, 0.0)) * losses[loss_name]
    if "loss_row_dfl" in losses:
        total = total + row_weight * losses["loss_row_dfl"]
    return total


def _group_metrics(
    names: list[str],
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for group, prefixes in GROUP_PREFIXES.items():
        dot = left_sq = right_sq = 0.0
        overlap_tensors = left_tensors = right_tensors = 0
        for name, left_grad, right_grad in zip(names, left, right):
            if not name.startswith(prefixes):
                continue
            if left_grad is not None:
                value = left_grad.detach().float()
                left_sq += float(value.square().sum())
                left_tensors += 1
            if right_grad is not None:
                value = right_grad.detach().float()
                right_sq += float(value.square().sum())
                right_tensors += 1
            if left_grad is not None and right_grad is not None:
                dot += float(
                    (left_grad.detach().float() * right_grad.detach().float()).sum()
                )
                overlap_tensors += 1
        left_norm = math.sqrt(left_sq)
        right_norm = math.sqrt(right_sq)
        denominator = left_norm * right_norm
        output[group] = {
            "cosine": dot / denominator if denominator else 0.0,
            "left_norm": left_norm,
            "right_norm": right_norm,
            "left_over_right_norm": left_norm / max(right_norm, 1.0e-30),
            "overlap_parameter_tensors": overlap_tensors,
            "left_parameter_tensors": left_tensors,
            "right_parameter_tensors": right_tensors,
        }
    return output


def _summarize(per_batch: list[dict[str, Any]]) -> dict[str, Any]:
    pair_names = sorted(per_batch[0]["pairs"])
    result: dict[str, Any] = {"pairs": {}, "losses": {}}
    for pair in pair_names:
        result["pairs"][pair] = {}
        for group in GROUP_PREFIXES:
            rows = [batch["pairs"][pair][group] for batch in per_batch]
            cosines = np.asarray([row["cosine"] for row in rows], dtype=np.float64)
            ratios = np.asarray(
                [row["left_over_right_norm"] for row in rows], dtype=np.float64
            )
            result["pairs"][pair][group] = {
                "cosine_mean": float(cosines.mean()),
                "cosine_median": float(np.median(cosines)),
                "cosine_p10": float(np.quantile(cosines, 0.10)),
                "negative_batch_fraction": float((cosines < 0.0).mean()),
                "left_over_right_norm_mean": float(ratios.mean()),
                "left_over_right_norm_median": float(np.median(ratios)),
                "overlap_parameter_tensors": int(
                    rows[0]["overlap_parameter_tensors"]
                ),
            }
    for loss_name in per_batch[0]["losses"]:
        values = np.asarray(
            [batch["losses"][loss_name] for batch in per_batch], dtype=np.float64
        )
        result["losses"][loss_name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
        }
    return result


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(Path(args.dataset_root))
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg["dataloader"]["pin_memory"] = False

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.train()
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    if hasattr(criterion, "set_iteration"):
        criterion.set_iteration(checkpoint_iteration)
    loader = build_dataloader(cfg, split="train", training=True)

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(name.startswith(prefixes) for prefixes in GROUP_PREFIXES.values())
    ]
    if not named_parameters:
        raise RuntimeError("no parameters matched the gradient-audit groups")
    names = [name for name, _parameter in named_parameters]
    parameters = [parameter for _name, parameter in named_parameters]
    field_weight = float(cfg.get("loss", {}).get("w_four_slot_joint_field", 0.0))
    geometry_weight = float(cfg.get("loss", {}).get("w_four_slot_geometry", 0.0))
    if field_weight <= 0.0:
        raise ValueError("field-gradient audit requires a positive field loss weight")

    per_batch: list[dict[str, Any]] = []
    iterator = tqdm(loader, desc="field gradient conflict", ncols=90)
    for batch_index, (images, targets, _metas) in enumerate(iterator):
        if batch_index >= int(args.max_batches):
            break
        images = images.to(device)
        targets = nested_to_device(targets, device)
        outputs, matches = forward_with_matches(
            model, images, targets, matcher, cfg, checkpoint_iteration
        )
        losses = criterion(outputs, targets, matches)
        weighted_field = field_weight * losses["loss_four_slot_joint_field"]
        legacy = losses["loss_total"] - weighted_field
        four_slot_geometry = geometry_weight * losses["loss_four_slot_geometry"]
        proposal = _weighted_proposal_loss(losses, cfg)

        field_grad = torch.autograd.grad(
            weighted_field, parameters, retain_graph=True, allow_unused=True
        )
        legacy_grad = torch.autograd.grad(
            legacy, parameters, retain_graph=True, allow_unused=True
        )
        geometry_grad = torch.autograd.grad(
            four_slot_geometry, parameters, retain_graph=True, allow_unused=True
        )
        proposal_grad = torch.autograd.grad(
            proposal, parameters, retain_graph=False, allow_unused=True
        )
        per_batch.append(
            {
                "batch_index": batch_index,
                "losses": {
                    "weighted_field": float(weighted_field.detach().cpu()),
                    "legacy": float(legacy.detach().cpu()),
                    "four_slot_geometry": float(four_slot_geometry.detach().cpu()),
                    "proposal": float(proposal.detach().cpu()),
                },
                "pairs": {
                    "field_vs_legacy": _group_metrics(
                        names, field_grad, legacy_grad
                    ),
                    "field_vs_four_slot_geometry": _group_metrics(
                        names, field_grad, geometry_grad
                    ),
                    "field_vs_proposal": _group_metrics(
                        names, field_grad, proposal_grad
                    ),
                },
            }
        )
        model.zero_grad(set_to_none=True)
    if not per_batch:
        raise RuntimeError("gradient audit processed no batches")

    payload = {
        "experiment": "V30 field-gradient conflict audit",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "batches": len(per_batch),
        "field_loss_weight": field_weight,
        "per_batch": per_batch,
        "summary": _summarize(per_batch),
        "interpretation": {
            "negative_cosine": "field and comparison objective locally oppose each other",
            "positive_cosine": "field and comparison objective locally agree",
            "near_zero_cosine": "objectives are locally close to orthogonal",
            "scope": "local checkpoint/batch evidence, not a whole-training proof",
            "test_split_used": False,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
