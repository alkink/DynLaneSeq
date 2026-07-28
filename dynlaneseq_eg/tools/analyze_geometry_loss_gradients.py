from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device, soft_expected_x
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure gradient scale, alignment, and lane-length weighting of "
            "the final row-geometry losses at a frozen checkpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
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
    return cfg


def lane_balanced_point_loss(
    pred_x: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    *,
    input_w: int,
    beta: float,
) -> torch.Tensor:
    lane_losses: list[torch.Tensor] = []
    for batch_index, match in enumerate(matches):
        pred_indices = match["pred_indices"].to(pred_x.device)
        gt_indices = match["gt_indices"].to(pred_x.device)
        for pred_index, gt_index in zip(pred_indices, gt_indices):
            gt = targets[batch_index]["x_rows"].to(pred_x.device)[gt_index]
            valid = targets[batch_index]["valid_mask"].to(pred_x.device)[
                gt_index
            ].bool()
            if not bool(valid.any()):
                continue
            loss = F.smooth_l1_loss(
                pred_x[batch_index, pred_index][valid] / float(input_w),
                gt[valid] / float(input_w),
                beta=float(beta),
                reduction="mean",
            )
            lane_losses.append(loss)
    if not lane_losses:
        return pred_x.sum() * 0.0
    return torch.stack(lane_losses).mean()


def lane_balanced_dfl_loss(
    logits: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    *,
    input_w: int,
) -> torch.Tensor:
    lane_losses: list[torch.Tensor] = []
    x_bins = int(logits.shape[-1])
    num_rows = int(logits.shape[-2])
    bin_width = float(input_w) / float(x_bins)
    for batch_index, match in enumerate(matches):
        pred_indices = match["pred_indices"].to(logits.device)
        gt_indices = match["gt_indices"].to(logits.device)
        for pred_index, gt_index in zip(pred_indices, gt_indices):
            gt = targets[batch_index]["x_rows"].to(
                logits.device,
                dtype=logits.dtype,
            )[gt_index, :num_rows]
            valid = targets[batch_index]["valid_mask"].to(logits.device)[
                gt_index, :num_rows
            ].bool()
            valid = (
                valid
                & torch.isfinite(gt)
                & (gt >= 0.0)
                & (gt <= float(input_w))
            )
            if not bool(valid.any()):
                continue
            target_bin = (gt / bin_width).clamp(0.0, float(x_bins - 1))
            left = target_bin.floor().long()
            right = (left + 1).clamp(max=x_bins - 1)
            right_weight = target_bin - left.to(dtype=target_bin.dtype)
            left_weight = 1.0 - right_weight
            same = right == left
            left_weight = torch.where(
                same,
                torch.ones_like(left_weight),
                left_weight,
            )
            right_weight = torch.where(
                same,
                torch.zeros_like(right_weight),
                right_weight,
            )
            log_probs = F.log_softmax(
                logits[batch_index, pred_index],
                dim=-1,
            )
            left_lp = log_probs.gather(-1, left.unsqueeze(-1)).squeeze(-1)
            right_lp = log_probs.gather(-1, right.unsqueeze(-1)).squeeze(-1)
            loss = -(left_weight * left_lp + right_weight * right_lp)
            lane_losses.append(loss[valid].mean())
    if not lane_losses:
        return logits.sum() * 0.0
    return torch.stack(lane_losses).mean()


def _gradient(
    loss: torch.Tensor,
    logits: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    result = torch.autograd.grad(
        loss,
        logits,
        retain_graph=retain_graph,
        allow_unused=False,
    )[0]
    return result.detach()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().flatten()
    right = right.double().flatten()
    denominator = float(left.norm() * right.norm())
    if denominator <= 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _lane_ranks(target: dict[str, torch.Tensor]) -> dict[int, int]:
    gt_x = target["x_rows"].float()
    valid = target["valid_mask"].bool()
    mean_x: list[float] = []
    for lane_index in range(int(gt_x.shape[0])):
        lane_valid = valid[lane_index]
        mean_x.append(
            float(gt_x[lane_index, lane_valid].mean())
            if bool(lane_valid.any())
            else float("inf")
        )
    order = sorted(range(len(mean_x)), key=mean_x.__getitem__)
    return {lane_index: rank for rank, lane_index in enumerate(order)}


@dataclass
class ScalarStats:
    values: list[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        if math.isfinite(value):
            self.values.append(float(value))

    def summary(self) -> dict[str, float | int]:
        if not self.values:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(self.values),
            "mean": sum(self.values) / len(self.values),
            "min": min(self.values),
            "max": max(self.values),
        }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    cfg = _prepare_config(
        args.config,
        dataset_root=args.dataset_root,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(cfg)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model = model.to(device).eval()
    head = model.structured_query_head
    if head is None:
        raise ValueError("Checkpoint must use a structured query head")
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    criterion.set_iteration(max(iteration, 1_000_000))
    loss_cfg = criterion.cfg
    input_w = int(head.input_w)
    x_bins = int(head.x_bins)

    component_norms = defaultdict(ScalarStats)
    component_cosines = defaultdict(ScalarStats)
    total_comparisons = defaultdict(ScalarStats)
    current_lane_norm_by_rank = defaultdict(ScalarStats)
    balanced_lane_norm_by_rank = defaultdict(ScalarStats)
    balance_multiplier_by_rank = defaultdict(ScalarStats)
    balance_multiplier_by_length = defaultdict(ScalarStats)
    loss_values = defaultdict(ScalarStats)

    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
    total_batches = (
        min(len(loader), int(args.max_batches))
        if int(args.max_batches) > 0
        else len(loader)
    )
    images_seen = 0
    for batch_index, (images, targets, _metas) in enumerate(
        tqdm(loader, total=total_batches, desc="geometry gradient audit", ncols=100)
    ):
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        images = images.to(device)
        targets = nested_to_device(targets, device)
        with torch.no_grad(), _amp_context(device, amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
            frozen_outputs = head(encoded["features"], inference_only=False)
            matches = matcher(frozen_outputs, targets)

        logits = (
            frozen_outputs["row_x_logits"]
            .detach()
            .float()
            .requires_grad_(True)
        )
        pred_x = soft_expected_x(
            logits,
            input_w=input_w,
            x_bins=x_bins,
        )
        geometry_outputs = {
            "row_x_logits": logits,
            "pred_x_rows": pred_x,
        }
        losses = {
            "point": (
                float(loss_cfg.w_point)
                * criterion.compute_point_loss(
                    geometry_outputs,
                    targets,
                    matches,
                )
            ),
            "line_iou": (
                float(loss_cfg.w_line_iou)
                * criterion.compute_line_iou_loss(
                    geometry_outputs,
                    targets,
                    matches,
                )
            ),
            "dfl": (
                float(loss_cfg.w_row_dfl)
                * criterion.compute_row_dfl_loss(
                    geometry_outputs,
                    targets,
                    matches,
                )
            ),
            "smooth": (
                float(loss_cfg.w_smooth)
                * criterion.compute_smoothness_loss(
                    geometry_outputs,
                    targets,
                    matches,
                )
            ),
        }
        balanced_point = float(loss_cfg.w_point) * lane_balanced_point_loss(
            pred_x,
            targets,
            matches,
            input_w=input_w,
            beta=float(loss_cfg.smooth_l1_beta),
        )
        balanced_dfl = float(loss_cfg.w_row_dfl) * lane_balanced_dfl_loss(
            logits,
            targets,
            matches,
            input_w=input_w,
        )
        current_total = sum(losses.values())
        balanced_total = (
            balanced_point
            + balanced_dfl
            + losses["line_iou"]
            + losses["smooth"]
        )

        gradients: dict[str, torch.Tensor] = {}
        for name, loss in losses.items():
            gradients[name] = _gradient(loss, logits, retain_graph=True)
            component_norms[name].update(float(gradients[name].double().norm()))
            loss_values[name].update(float(loss.detach()))
        current_gradient = _gradient(current_total, logits, retain_graph=True)
        balanced_gradient = _gradient(
            balanced_total,
            logits,
            retain_graph=False,
        )
        component_norms["current_total"].update(
            float(current_gradient.double().norm())
        )
        component_norms["lane_balanced_total"].update(
            float(balanced_gradient.double().norm())
        )
        loss_values["current_total"].update(float(current_total.detach()))
        loss_values["lane_balanced_total"].update(float(balanced_total.detach()))
        for left_index, left_name in enumerate(gradients):
            for right_name in list(gradients)[left_index + 1 :]:
                component_cosines[f"{left_name}_vs_{right_name}"].update(
                    _cosine(gradients[left_name], gradients[right_name])
                )
        total_comparisons["current_vs_lane_balanced_cosine"].update(
            _cosine(current_gradient, balanced_gradient)
        )
        total_comparisons["balanced_to_current_norm_ratio"].update(
            float(
                balanced_gradient.double().norm()
                / current_gradient.double().norm().clamp_min(1e-12)
            )
        )

        for image_index, (target, match) in enumerate(zip(targets, matches)):
            ranks = _lane_ranks(target)
            gt_valid = target["valid_mask"].bool()
            pred_indices = match["pred_indices"].tolist()
            gt_indices = match["gt_indices"].tolist()
            valid_lengths = [
                int(gt_valid[gt_index].sum()) for gt_index in gt_indices
            ]
            total_rows = sum(valid_lengths)
            lane_count = max(len(valid_lengths), 1)
            for pred_index, gt_index, valid_length in zip(
                pred_indices,
                gt_indices,
                valid_lengths,
            ):
                rank = int(ranks[int(gt_index)])
                current_lane_norm_by_rank[rank].update(
                    float(
                        current_gradient[image_index, pred_index]
                        .double()
                        .norm()
                    )
                )
                balanced_lane_norm_by_rank[rank].update(
                    float(
                        balanced_gradient[image_index, pred_index]
                        .double()
                        .norm()
                    )
                )
                multiplier = (
                    float(total_rows)
                    / max(float(lane_count * valid_length), 1.0)
                )
                balance_multiplier_by_rank[rank].update(multiplier)
                if valid_length < 80:
                    length_bucket = "lt80"
                elif valid_length < 120:
                    length_bucket = "80to119"
                elif valid_length < 150:
                    length_bucket = "120to149"
                else:
                    length_bucket = "ge150"
                balance_multiplier_by_length[length_bucket].update(multiplier)
        images_seen += int(images.shape[0])

    rank_effects: dict[str, Any] = {}
    for rank in sorted(
        set(current_lane_norm_by_rank) | set(balanced_lane_norm_by_rank)
    ):
        current = current_lane_norm_by_rank[rank].summary()
        balanced = balanced_lane_norm_by_rank[rank].summary()
        rank_effects[str(rank)] = {
            "current_gradient_norm": current,
            "lane_balanced_gradient_norm": balanced,
            "balanced_to_current_mean_ratio": (
                float(balanced["mean"])
                / max(float(current["mean"]), 1e-12)
            ),
            "analytical_row_weight_multiplier": (
                balance_multiplier_by_rank[rank].summary()
            ),
        }

    payload = {
        "diagnostic_only": True,
        "warning": (
            "Gradients are measured with respect to detached final row logits "
            "at a frozen checkpoint. They diagnose the objective, not the "
            "performance of a retrained lane-balanced model."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "loss_weights": {
            "point": float(loss_cfg.w_point),
            "line_iou": float(loss_cfg.w_line_iou),
            "dfl": float(loss_cfg.w_row_dfl),
            "smooth": float(loss_cfg.w_smooth),
        },
        "weighted_loss_values": {
            key: value.summary() for key, value in loss_values.items()
        },
        "gradient_norms": {
            key: value.summary() for key, value in component_norms.items()
        },
        "gradient_cosines": {
            key: value.summary() for key, value in component_cosines.items()
        },
        "lane_balanced_comparison": {
            key: value.summary() for key, value in total_comparisons.items()
        },
        "per_lane_rank_effect": rank_effects,
        "analytical_row_weight_multiplier_by_length": {
            key: value.summary()
            for key, value in sorted(balance_multiplier_by_length.items())
        },
    }
    compact = dict(payload)
    compact.pop("sampled_dataset_indices")
    print(json.dumps(compact, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
