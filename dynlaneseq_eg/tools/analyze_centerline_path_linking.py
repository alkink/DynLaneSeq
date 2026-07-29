from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_frozen_p2_centerline_separability import (
    FrozenP2CenterlineProbe,
)
from dynlaneseq_eg.tools.probe_query_conditioned_dense_curve import (
    _best_iou_per_gt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Link frozen-P2 centerline evidence into complete, GT-free lane "
            "paths with dynamic programming. This separates fragmented visual "
            "evidence from the production query decoder."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--max-step-bins", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument(
        "--transition-penalties",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1],
    )
    parser.add_argument("--suppression-radius-bins", type=int, default=8)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def link_best_path(
    score: torch.Tensor,
    *,
    max_step_bins: int,
    transition_penalty: float,
) -> torch.Tensor:
    """Return the globally best bottom-to-top path for each batch item.

    ``score`` has shape ``[B,R,X]``. The transition is bounded by
    ``max_step_bins`` and can optionally penalize horizontal movement.
    """

    if score.ndim != 3:
        raise ValueError("score must have shape [B,R,X]")
    batch, rows, width = score.shape
    radius = int(max_step_bins)
    if radius < 0:
        raise ValueError("max_step_bins must be non-negative")
    ordered = score.flip(1)
    dp = ordered[:, 0]
    parents: list[torch.Tensor] = []
    offsets = torch.arange(
        -radius,
        radius + 1,
        device=score.device,
        dtype=score.dtype,
    )
    movement_cost = offsets.abs() * float(transition_penalty)
    for row_index in range(1, rows):
        padded = F.pad(
            dp,
            (radius, radius),
            mode="constant",
            value=-1e4,
        )
        windows = padded.unfold(-1, 2 * radius + 1, 1)
        candidates = windows - movement_cost.view(1, 1, -1)
        best_previous, parent_offset_index = candidates.max(dim=-1)
        dp = ordered[:, row_index] + best_previous
        parents.append(parent_offset_index.to(torch.int16))

    current = dp.argmax(dim=-1)
    reverse_path = torch.empty(
        (batch, rows),
        device=score.device,
        dtype=torch.long,
    )
    reverse_path[:, -1] = current
    batch_indices = torch.arange(batch, device=score.device)
    for row_index in range(rows - 1, 0, -1):
        offset_index = parents[row_index - 1][batch_indices, current].long()
        current = (current + offset_index - radius).clamp(0, width - 1)
        reverse_path[:, row_index - 1] = current
    return reverse_path.flip(1)


def extract_paths(
    logits: torch.Tensor,
    *,
    num_paths: int,
    max_step_bins: int,
    transition_penalty: float,
    suppression_radius_bins: int,
    input_w: float,
) -> torch.Tensor:
    """Extract multiple smooth paths from one union centerline map."""

    if logits.ndim == 4:
        if int(logits.shape[1]) != 1:
            raise ValueError("4D logits must have one centerline channel")
        logits = logits[:, 0]
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,R,X] or [B,1,R,X]")
    working = F.logsigmoid(logits.float())
    batch, rows, width = working.shape
    grid = torch.arange(width, device=working.device).view(1, 1, width)
    paths: list[torch.Tensor] = []
    for _ in range(int(num_paths)):
        path = link_best_path(
            working,
            max_step_bins=int(max_step_bins),
            transition_penalty=float(transition_penalty),
        )
        paths.append(path)
        corridor = (
            grid - path.unsqueeze(-1)
        ).abs() <= int(suppression_radius_bins)
        working = working.masked_fill(corridor, -20.0)
    path_bins = torch.stack(paths, dim=1).float()
    return (path_bins + 0.5) * (float(input_w) / float(width))


class PathMetrics:
    def __init__(self) -> None:
        self.lanes = 0
        self.path_hits_050 = 0
        self.path_hits_070 = 0
        self.wrong_hits_050 = 0
        self.wrong_hits_070 = 0
        self.base_hits_050 = 0
        self.base_hits_070 = 0
        self.union_hits_050 = 0
        self.union_hits_070 = 0
        self.recovered_050 = 0
        self.recovered_070 = 0
        self.lost_050 = 0
        self.lost_070 = 0
        self.path_iou_sum = 0.0
        self.wrong_iou_sum = 0.0

    def update(
        self,
        base: torch.Tensor,
        path: torch.Tensor,
        wrong: torch.Tensor,
    ) -> None:
        count = int(base.numel())
        if path.numel() != count or wrong.numel() != count:
            raise ValueError("per-GT IoU vectors must have equal length")
        self.lanes += count
        self.path_hits_050 += int((path >= 0.5).sum())
        self.path_hits_070 += int((path >= 0.7).sum())
        self.wrong_hits_050 += int((wrong >= 0.5).sum())
        self.wrong_hits_070 += int((wrong >= 0.7).sum())
        self.base_hits_050 += int((base >= 0.5).sum())
        self.base_hits_070 += int((base >= 0.7).sum())
        union = torch.maximum(base, path)
        self.union_hits_050 += int((union >= 0.5).sum())
        self.union_hits_070 += int((union >= 0.7).sum())
        self.recovered_050 += int(((base < 0.5) & (path >= 0.5)).sum())
        self.recovered_070 += int(((base < 0.7) & (path >= 0.7)).sum())
        self.lost_050 += int(((base >= 0.5) & (path < 0.5)).sum())
        self.lost_070 += int(((base >= 0.7) & (path < 0.7)).sum())
        self.path_iou_sum += float(path.sum())
        self.wrong_iou_sum += float(wrong.sum())

    def summary(self) -> dict[str, float | int]:
        lanes = max(self.lanes, 1)
        base_050 = self.base_hits_050 / lanes
        base_070 = self.base_hits_070 / lanes
        union_050 = self.union_hits_050 / lanes
        union_070 = self.union_hits_070 / lanes
        return {
            "lanes": self.lanes,
            "base_recall_050": base_050,
            "base_recall_070": base_070,
            "path_recall_050": self.path_hits_050 / lanes,
            "path_recall_070": self.path_hits_070 / lanes,
            "wrong_image_path_recall_050": self.wrong_hits_050 / lanes,
            "wrong_image_path_recall_070": self.wrong_hits_070 / lanes,
            "union_recall_050": union_050,
            "union_recall_070": union_070,
            "union_gain_050_points": 100.0 * (union_050 - base_050),
            "union_gain_070_points": 100.0 * (union_070 - base_070),
            "recovered_base_misses_050": self.recovered_050,
            "recovered_base_misses_070": self.recovered_070,
            "path_lacks_base_hits_050": self.lost_050,
            "path_lacks_base_hits_070": self.lost_070,
            "path_mean_best_iou": self.path_iou_sum / lanes,
            "wrong_image_path_mean_best_iou": self.wrong_iou_sum / lanes,
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    probe: nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    group_size: int,
    input_w: float,
    line_width: float,
    combinations: list[tuple[int, float]],
    num_paths: int,
    suppression_radius_bins: int,
) -> tuple[dict[str, dict[str, float | int]], int]:
    model.eval()
    probe.eval()
    metrics = {
        f"step{step}_penalty{penalty:g}": PathMetrics()
        for step, penalty in combinations
    }
    images_seen = 0
    for images, targets, _metas in tqdm(
        loader,
        desc="centerline path-linking diagnostic",
        ncols=108,
    ):
        if channels_last:
            images = images.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        with _autocast_context(device, amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
            outputs = model.structured_query_head(
                encoded["features"],
                inference_only=False,
            )
            logits = probe(encoded["features"].float())
            wrong_logits = probe(encoded["features"].float().roll(1, dims=0))
        base_candidates = outputs["pred_x_rows"][:, :group_size].float()
        for step, penalty in combinations:
            key = f"step{step}_penalty{penalty:g}"
            paths = extract_paths(
                logits,
                num_paths=num_paths,
                max_step_bins=step,
                transition_penalty=penalty,
                suppression_radius_bins=suppression_radius_bins,
                input_w=input_w,
            )
            wrong_paths = extract_paths(
                wrong_logits,
                num_paths=num_paths,
                max_step_bins=step,
                transition_penalty=penalty,
                suppression_radius_bins=suppression_radius_bins,
                input_w=input_w,
            )
            for batch_index, target in enumerate(targets):
                base_iou = _best_iou_per_gt(
                    base_candidates[batch_index],
                    target,
                    line_width=line_width,
                )
                path_iou = _best_iou_per_gt(
                    paths[batch_index],
                    target,
                    line_width=line_width,
                )
                wrong_iou = _best_iou_per_gt(
                    wrong_paths[batch_index],
                    target,
                    line_width=line_width,
                )
                metrics[key].update(base_iou, path_iou, wrong_iou)
        images_seen += int(images.shape[0])
    return {
        key: value.summary()
        for key, value in metrics.items()
    }, images_seen


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["persistent_workers"] = bool(int(args.num_workers) > 0)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    if (
        amp_dtype == torch.bfloat16
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.float16
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )

    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    probe = FrozenP2CenterlineProbe(
        in_dim=int(model_cfg.get("dim", 256)),
        hidden_dim=int(model_cfg.get("dim", 256)) // 2,
        num_rows=int(model_cfg.get("num_rows", 72)),
        x_bins=int(model_cfg.get("x_bins", 200)),
    )
    probe_payload = torch.load(args.probe_checkpoint, map_location="cpu")
    probe.load_state_dict(probe_payload["discovery_probe"], strict=True)
    probe.requires_grad_(False)
    probe = probe.to(device).eval()

    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(
        structured_cfg.get("num_instances", model_cfg.get("num_slots", 0))
    )
    num_groups = int(structured_cfg.get("num_groups", 1))
    group_size = num_instances // num_groups
    combinations = [
        (int(step), float(penalty))
        for step in args.max_step_bins
        for penalty in args.transition_penalties
    ]
    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy="uniform",
        max_batches=int(args.max_batches),
        num_workers=int(args.num_workers),
    )
    results, images_seen = evaluate(
        model,
        probe,
        loader,
        device=device,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        group_size=group_size,
        input_w=float(model_cfg.get("input_w", 800)),
        line_width=float(args.line_width),
        combinations=combinations,
        num_paths=int(args.num_paths),
        suppression_radius_bins=int(args.suppression_radius_bins),
    )
    best_key = max(
        results,
        key=lambda key: (
            float(results[key]["union_gain_050_points"]),
            float(results[key]["union_gain_070_points"]),
            float(results[key]["path_mean_best_iou"]),
        ),
    )
    best = results[best_key]
    positive_gate = bool(
        float(best["union_gain_050_points"]) >= 5.0
        and float(best["path_recall_050"])
        >= float(best["wrong_image_path_recall_050"]) + 0.03
    )
    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "warning": (
            "GT is never used to generate paths. The parameter grid and oracle "
            "union are diagnostic analyses on validation data, not a benchmark "
            "post-processing result."
        ),
        "question": (
            "Can a GT-free sequence linker assemble fragmented frozen-P2 "
            "centerline evidence into lanes missed by the production decoder?"
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "probe_checkpoint": args.probe_checkpoint,
        "images": int(images_seen),
        "sample_strategy": "uniform",
        "sampled_dataset_indices": sampled_indices,
        "num_paths": int(args.num_paths),
        "suppression_radius_bins": int(args.suppression_radius_bins),
        "combinations": results,
        "best_combination": best_key,
        "best_result": best,
        "positive_gate_definition": (
            "best union gain at IoU 0.50 >= 5.0 points AND correct-image "
            "path recall >= wrong-image control by 3.0 points"
        ),
        "positive_gate": positive_gate,
    }
    compact = dict(payload)
    compact.pop("sampled_dataset_indices")
    print(json.dumps(compact, indent=2))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
