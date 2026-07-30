from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train two small quality scorers over one frozen LaneRowNet "
            "checkpoint. The query-only arm tests whether corrected quality "
            "supervision is sufficient; the geometry-aware arm additionally "
            "uses visible-range pooling and row-distribution confidence."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-batches", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-strategy", choices=("uniform", "sequential"), default="uniform")
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quality-focal-beta", type=float, default=2.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-target-margin", type=float, default=0.10)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quality-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--min-gain-050-points", type=float, default=1.0)
    parser.add_argument("--min-gain-070-points", type=float, default=0.5)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-probe", default="")
    return parser.parse_args()


class QualityProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def quality_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float = 2.0,
) -> torch.Tensor:
    """Generalized-focal-style loss for continuous IoU targets."""

    targets = targets.to(dtype=logits.dtype)
    probabilities = torch.sigmoid(logits)
    modulation = (targets - probabilities).abs().pow(float(beta))
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (modulation * bce).mean()


def pairwise_quality_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_margin: float = 0.10,
) -> torch.Tensor:
    """Encourage the within-image ordering used by Top-K inference."""

    target_delta = targets.unsqueeze(-1) - targets.unsqueeze(-2)
    valid_pairs = target_delta > float(target_margin)
    if not bool(valid_pairs.any()):
        return logits.sum() * 0.0
    logit_delta = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    weights = target_delta[valid_pairs].detach()
    return (F.softplus(-logit_delta[valid_pairs]) * weights).sum() / weights.sum().clamp_min(1e-6)


def pairwise_lane_quality(
    pred_x_rows: torch.Tensor,
    range_norm: torch.Tensor,
    gt_x_rows: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    *,
    input_h: int,
    line_width: float,
) -> torch.Tensor:
    """Raster-IoU surrogate for every GT/candidate pair.

    Unlike the historical quality target, this includes predicted visible
    range and is computed for every candidate, not only the Hungarian match.
    """

    num_candidates, num_rows = pred_x_rows.shape
    num_gt = int(gt_x_rows.shape[0])
    if num_gt == 0:
        return pred_x_rows.new_zeros((0, num_candidates), dtype=torch.float32)

    pred_x = pred_x_rows.float()
    gt_x = gt_x_rows.to(device=pred_x.device, dtype=torch.float32)
    gt_valid = gt_valid_mask.to(device=pred_x.device).bool()
    ranges = sort_range_norm(range_norm.float())
    y_rows = fixed_y_rows(
        int(num_rows),
        int(input_h),
        device=pred_x.device,
        dtype=torch.float32,
    )
    pred_valid = (
        (y_rows.view(1, -1) >= ranges[:, :1] * float(input_h))
        & (y_rows.view(1, -1) <= ranges[:, 1:] * float(input_h))
        & torch.isfinite(pred_x)
    )
    gt_valid = gt_valid & torch.isfinite(gt_x)

    gt = gt_x[:, None, :]
    pred = pred_x[None, :, :]
    both = gt_valid[:, None, :] & pred_valid[None, :, :]
    either = gt_valid[:, None, :] | pred_valid[None, :, :]
    overlap = (float(line_width) - (gt - pred).abs()).clamp(min=0.0)
    overlap = torch.where(both, overlap, torch.zeros_like(overlap))
    union = torch.where(
        both,
        2.0 * float(line_width) - overlap,
        torch.where(
            either,
            torch.full_like(overlap, float(line_width)),
            torch.zeros_like(overlap),
        ),
    )
    return overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1e-6)


def all_proposal_quality_targets(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    input_h: int,
    line_width: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    pred_x_rows = outputs["pred_x_rows"].detach()
    ranges = outputs["range_norm"].detach()
    target_rows = []
    pairwise_rows = []
    for batch_index, target in enumerate(targets):
        pairwise = pairwise_lane_quality(
            pred_x_rows[batch_index],
            ranges[batch_index],
            target["x_rows"],
            target["valid_mask"],
            input_h=input_h,
            line_width=line_width,
        )
        pairwise_rows.append(pairwise)
        if int(pairwise.shape[0]) == 0:
            target_rows.append(pred_x_rows.new_zeros(pred_x_rows.shape[1], dtype=torch.float32))
        else:
            target_rows.append(pairwise.max(dim=0).values)
    return torch.stack(target_rows, dim=0), pairwise_rows


def query_features(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["queries"].detach().float()


def geometry_aware_features(
    outputs: dict[str, torch.Tensor],
    *,
    input_w: int,
    range_temperature: float = 0.02,
) -> torch.Tensor:
    row_tokens = outputs["structured_row_tokens"].detach().float()
    queries = outputs["queries"].detach().float()
    ranges = sort_range_norm(outputs["range_norm"].detach().float())
    pred_x = outputs["pred_x_rows"].detach().float()
    row_logits = outputs["row_x_logits"].detach().float()
    batch, candidates, rows, _channels = row_tokens.shape

    y_norm = (
        torch.arange(
            rows,
            device=row_tokens.device,
            dtype=row_tokens.dtype,
        )
        / float(max(rows, 1))
    ).view(1, 1, rows)
    temperature = max(float(range_temperature), 1e-4)
    row_weight = torch.sigmoid((y_norm - ranges[..., :1]) / temperature)
    row_weight = row_weight * torch.sigmoid((ranges[..., 1:] - y_norm) / temperature)
    denominator = row_weight.sum(dim=-1, keepdim=True).clamp_min(1e-4)
    masked_mean = (row_tokens * row_weight.unsqueeze(-1)).sum(dim=2) / denominator
    centered = row_tokens - masked_mean.unsqueeze(2)
    state_variance = (
        centered.square().mean(dim=-1) * row_weight
    ).sum(dim=-1, keepdim=True) / denominator

    log_max_probability = row_logits.amax(dim=-1) - torch.logsumexp(row_logits, dim=-1)
    row_confidence = log_max_probability.exp()
    confidence_mean = (row_confidence * row_weight).sum(dim=-1, keepdim=True) / denominator
    confidence_max = row_confidence.amax(dim=-1, keepdim=True)

    pred_x_norm = pred_x / float(max(int(input_w) - 1, 1))
    first_difference = (pred_x_norm[..., 1:] - pred_x_norm[..., :-1]).abs()
    first_weight = row_weight[..., 1:] * row_weight[..., :-1]
    slope = (first_difference * first_weight).sum(dim=-1, keepdim=True)
    slope = slope / first_weight.sum(dim=-1, keepdim=True).clamp_min(1e-4)
    second_difference = (
        pred_x_norm[..., 2:]
        - 2.0 * pred_x_norm[..., 1:-1]
        + pred_x_norm[..., :-2]
    ).abs()
    second_weight = row_weight[..., 2:] * row_weight[..., 1:-1] * row_weight[..., :-2]
    curvature = (second_difference * second_weight).sum(dim=-1, keepdim=True)
    curvature = curvature / second_weight.sum(dim=-1, keepdim=True).clamp_min(1e-4)

    reference = outputs.get("input_reference_x_rows")
    if reference is None:
        reference_mean = pred_x.new_zeros((batch, candidates, 1))
        reference_max = pred_x.new_zeros((batch, candidates, 1))
    else:
        reference_delta = (
            pred_x - reference.detach().float()
        ).abs() / float(max(int(input_w) - 1, 1))
        reference_mean = (reference_delta * row_weight).sum(dim=-1, keepdim=True)
        reference_mean = reference_mean / denominator
        reference_max = reference_delta.amax(dim=-1, keepdim=True)

    scalar_features = torch.cat(
        (
            ranges,
            ranges[..., 1:] - ranges[..., :1],
            confidence_mean,
            confidence_max,
            slope,
            curvature,
            reference_mean,
            reference_max,
            state_variance,
        ),
        dim=-1,
    )
    return torch.cat((queries, masked_mean, scalar_features), dim=-1)


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _prepare_config(
    path: str,
    *,
    dataset_root: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg = load_config(path)
    if dataset_root:
        cfg.setdefault("dataset", {})["root"] = dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(batch_size)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(eval_batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(num_workers)
    if int(num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _frozen_outputs(
    model: nn.Module,
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if model.structured_query_head is None:
        raise ValueError("quality rescoring probe requires a structured query head")
    encoder_outputs = model.encoder.forward_features(
        images,
        inference_only=True,
        structured_only=True,
    )
    return model.structured_query_head(
        encoder_outputs["features"],
        inference_only=False,
    )


def _average_precision(scores: list[float], labels: list[int]) -> float:
    positives = int(sum(labels))
    if positives == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if int(labels[index]) == 0:
            continue
        true_positives += 1
        precision_sum += float(true_positives) / float(rank)
    return precision_sum / float(positives)


def _new_strategy_stats() -> dict[str, Any]:
    return {
        "hits": {0.5: 0, 0.7: 0},
        "gt": 0,
        "selected": 0,
        "score_values": [],
        "labels_050": [],
        "labels_070": [],
        "zero_gt_top4_score_sum": 0.0,
        "zero_gt_images": 0,
    }


def _finish_strategy(stats: dict[str, Any]) -> dict[str, Any]:
    gt = max(int(stats["gt"]), 1)
    return {
        "gt_lanes": int(stats["gt"]),
        "top4_recall_050": float(stats["hits"][0.5]) / float(gt),
        "top4_recall_070": float(stats["hits"][0.7]) / float(gt),
        "candidate_ap_050": _average_precision(stats["score_values"], stats["labels_050"]),
        "candidate_ap_070": _average_precision(stats["score_values"], stats["labels_070"]),
        "mean_zero_gt_top4_score": (
            float(stats["zero_gt_top4_score_sum"]) / float(stats["zero_gt_images"])
            if int(stats["zero_gt_images"]) > 0
            else None
        ),
        "zero_gt_images": int(stats["zero_gt_images"]),
    }


@torch.no_grad()
def evaluate_probes(
    model: nn.Module,
    query_probe: nn.Module,
    geometry_probe: nn.Module,
    loader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    input_h: int,
    input_w: int,
    line_width: float,
    top_k: int,
    quality_power: float,
    quality_focal_beta: float,
) -> dict[str, Any]:
    model.eval()
    query_probe.eval()
    geometry_probe.eval()
    strategy_stats = {
        "current_exist_quality": _new_strategy_stats(),
        "exist_only": _new_strategy_stats(),
        "query_probe": _new_strategy_stats(),
        "geometry_probe": _new_strategy_stats(),
        "oracle_top4": _new_strategy_stats(),
    }
    eval_losses = {"query_probe": 0.0, "geometry_probe": 0.0, "batches": 0}
    for images, targets, _metas in tqdm(loader, desc="quality probe eval", ncols=80):
        if channels_last:
            images = images.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images.to(device, non_blocking=True)
        with _amp_context(device, amp_dtype):
            outputs = _frozen_outputs(model, images)
        quality_targets, pairwise_rows = all_proposal_quality_targets(
            outputs,
            targets,
            input_h=input_h,
            line_width=line_width,
        )
        q_features = query_features(outputs)
        g_features = geometry_aware_features(outputs, input_w=input_w)
        query_logits = query_probe(q_features)
        geometry_logits = geometry_probe(g_features)
        eval_losses["query_probe"] += float(
            quality_focal_loss(
                query_logits,
                quality_targets,
                beta=quality_focal_beta,
            ).item()
        )
        eval_losses["geometry_probe"] += float(
            quality_focal_loss(
                geometry_logits,
                quality_targets,
                beta=quality_focal_beta,
            ).item()
        )
        eval_losses["batches"] += 1

        exist_score = torch.softmax(outputs["exist_logits"].float(), dim=-1)[..., 0]
        old_quality = torch.sigmoid(outputs["quality_logits"].float()).clamp_min(1e-6)
        scores_by_strategy = {
            "current_exist_quality": exist_score * old_quality.pow(float(quality_power)),
            "exist_only": exist_score,
            "query_probe": exist_score * torch.sigmoid(query_logits).pow(float(quality_power)),
            "geometry_probe": exist_score * torch.sigmoid(geometry_logits).pow(float(quality_power)),
        }
        for batch_index, pairwise in enumerate(pairwise_rows):
            gt_count = int(pairwise.shape[0])
            candidate_count = int(pairwise.shape[1])
            for strategy_name, scores in scores_by_strategy.items():
                stats = strategy_stats[strategy_name]
                score_row = scores[batch_index]
                selected_ids = torch.argsort(score_row, descending=True)[: int(top_k)].tolist()
                stats["gt"] += gt_count
                stats["selected"] += len(selected_ids)
                if gt_count == 0:
                    stats["zero_gt_images"] += 1
                    stats["zero_gt_top4_score_sum"] += float(
                        score_row[selected_ids].mean().item()
                    ) if selected_ids else 0.0
                for threshold in (0.5, 0.7):
                    assignment = evaluator_hungarian_assignment(
                        pairwise,
                        selected_ids,
                        threshold=threshold,
                    )
                    stats["hits"][threshold] += int(assignment.hit_count)
                candidate_best = (
                    pairwise.max(dim=0).values
                    if gt_count > 0
                    else score_row.new_zeros(candidate_count)
                )
                stats["score_values"].extend(float(value) for value in score_row.tolist())
                stats["labels_050"].extend(
                    int(value > 0.5) for value in candidate_best.tolist()
                )
                stats["labels_070"].extend(
                    int(value > 0.7) for value in candidate_best.tolist()
                )

            oracle_stats = strategy_stats["oracle_top4"]
            oracle_stats["gt"] += gt_count
            if gt_count > 0:
                candidate_valid = torch.ones(candidate_count, dtype=torch.bool)
                for threshold in (0.5, 0.7):
                    assignment = cardinality_oracle_assignment(
                        pairwise,
                        threshold=threshold,
                        top_k=int(top_k),
                        candidate_valid=candidate_valid,
                    )
                    oracle_stats["hits"][threshold] += int(assignment.hit_count)

    batches = max(int(eval_losses.pop("batches")), 1)
    return {
        "strategies": {
            name: _finish_strategy(stats)
            for name, stats in strategy_stats.items()
        },
        "quality_focal_loss": {
            name: float(value) / float(batches)
            for name, value in eval_losses.items()
        },
    }


def _probe_verdict(
    evaluation: dict[str, Any],
    *,
    min_gain_050_points: float,
    min_gain_070_points: float,
) -> dict[str, Any]:
    strategies = evaluation["strategies"]
    baseline = strategies["current_exist_quality"]
    arms = {}
    for name in ("query_probe", "geometry_probe"):
        row = strategies[name]
        gain_050 = 100.0 * (
            float(row["top4_recall_050"]) - float(baseline["top4_recall_050"])
        )
        gain_070 = 100.0 * (
            float(row["top4_recall_070"]) - float(baseline["top4_recall_070"])
        )
        arms[name] = {
            "gain_050_points": gain_050,
            "gain_070_points": gain_070,
            "positive": bool(
                gain_050 >= float(min_gain_050_points)
                and gain_070 >= float(min_gain_070_points)
            ),
        }
    if arms["query_probe"]["positive"]:
        recommendation = "quality_target_and_loss_are_sufficient"
    elif arms["geometry_probe"]["positive"]:
        recommendation = "add_geometry_aware_quality_scorer"
    else:
        recommendation = "no_frozen_scorer_signal_revisit_decoder_or_assignment"
    return {
        "gate": {
            "min_gain_050_points": float(min_gain_050_points),
            "min_gain_070_points": float(min_gain_070_points),
        },
        "arms": arms,
        "recommendation": recommendation,
    }


def main() -> None:
    args = parse_args()
    if int(args.train_steps) < 1:
        raise ValueError("train_steps must be positive")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    cfg = _prepare_config(
        args.config,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    model_cfg = cfg.get("model", {})
    input_h = int(model_cfg.get("input_h", 288))
    input_w = int(model_cfg.get("input_w", 800))
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if model.structured_query_head is None:
        raise ValueError("quality rescoring probe requires a structured query head")
    # The probe consumes only the final decoder state. Avoid exporting unused
    # auxiliary predictions while preserving the frozen final predictions.
    model.structured_query_head.intermediate_supervision = False
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    train_loader = build_dataloader(cfg, split="train", training=True)
    eval_loader = build_dataloader(cfg, split="val", training=False)
    eval_loader, eval_indices = select_diagnostic_loader(
        eval_loader,
        strategy=args.sample_strategy,
        max_batches=args.eval_max_batches,
        num_workers=args.num_workers,
    )

    first_images, first_targets, _first_metas = next(iter(train_loader))
    if channels_last:
        first_images = first_images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    else:
        first_images = first_images.to(device, non_blocking=True)
    with torch.no_grad(), _amp_context(device, amp_dtype):
        first_outputs = _frozen_outputs(model, first_images)
    first_query_features = query_features(first_outputs)
    first_geometry_features = geometry_aware_features(first_outputs, input_w=input_w)
    query_probe = QualityProbe(int(first_query_features.shape[-1])).to(device)
    geometry_probe = QualityProbe(int(first_geometry_features.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(
        list(query_probe.parameters()) + list(geometry_probe.parameters()),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    del first_images, first_targets, first_outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_iterator = iter(train_loader)
    running = {
        "query": 0.0,
        "geometry": 0.0,
        "rank_query": 0.0,
        "rank_geometry": 0.0,
        "target_mean": 0.0,
        "target_ge_050": 0.0,
    }
    training_totals = {key: 0.0 for key in running}
    for step in tqdm(range(1, int(args.train_steps) + 1), desc="quality probe train", ncols=80):
        try:
            images, targets, _metas = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, targets, _metas = next(train_iterator)
        if channels_last:
            images = images.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images.to(device, non_blocking=True)
        with torch.no_grad(), _amp_context(device, amp_dtype):
            outputs = _frozen_outputs(model, images)
        targets_quality, _pairwise = all_proposal_quality_targets(
            outputs,
            targets,
            input_h=input_h,
            line_width=args.line_width,
        )
        q_features = query_features(outputs)
        g_features = geometry_aware_features(outputs, input_w=input_w)
        query_logits = query_probe(q_features)
        geometry_logits = geometry_probe(g_features)
        query_loss = quality_focal_loss(
            query_logits,
            targets_quality,
            beta=args.quality_focal_beta,
        )
        geometry_loss = quality_focal_loss(
            geometry_logits,
            targets_quality,
            beta=args.quality_focal_beta,
        )
        query_rank = pairwise_quality_ranking_loss(
            query_logits,
            targets_quality,
            target_margin=args.rank_target_margin,
        )
        geometry_rank = pairwise_quality_ranking_loss(
            geometry_logits,
            targets_quality,
            target_margin=args.rank_target_margin,
        )
        total_loss = (
            query_loss
            + geometry_loss
            + float(args.rank_loss_weight) * (query_rank + geometry_rank)
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(query_probe.parameters()) + list(geometry_probe.parameters()),
            max_norm=5.0,
        )
        optimizer.step()

        running["query"] += float(query_loss.detach())
        running["geometry"] += float(geometry_loss.detach())
        running["rank_query"] += float(query_rank.detach())
        running["rank_geometry"] += float(geometry_rank.detach())
        running["target_mean"] += float(targets_quality.mean())
        running["target_ge_050"] += float((targets_quality > 0.5).float().mean())
        training_totals["query"] += float(query_loss.detach())
        training_totals["geometry"] += float(geometry_loss.detach())
        training_totals["rank_query"] += float(query_rank.detach())
        training_totals["rank_geometry"] += float(geometry_rank.detach())
        training_totals["target_mean"] += float(targets_quality.mean())
        training_totals["target_ge_050"] += float(
            (targets_quality > 0.5).float().mean()
        )
        if int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
            denominator = float(args.log_interval)
            tqdm.write(
                f"step {step:05d}/{int(args.train_steps):05d} "
                f"q={running['query'] / denominator:.4f} "
                f"g={running['geometry'] / denominator:.4f} "
                f"rank_q={running['rank_query'] / denominator:.4f} "
                f"rank_g={running['rank_geometry'] / denominator:.4f} "
                f"target={running['target_mean'] / denominator:.4f} "
                f"target>=.5={running['target_ge_050'] / denominator:.4f}"
            )
            for key in running:
                running[key] = 0.0

    evaluation = evaluate_probes(
        model,
        query_probe,
        geometry_probe,
        eval_loader,
        device=device,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        input_h=input_h,
        input_w=input_w,
        line_width=args.line_width,
        top_k=args.top_k,
        quality_power=args.quality_power,
        quality_focal_beta=args.quality_focal_beta,
    )
    verdict = _probe_verdict(
        evaluation,
        min_gain_050_points=args.min_gain_050_points,
        min_gain_070_points=args.min_gain_070_points,
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "The base detector is frozen. A positive result localizes the "
            "ceiling to quality supervision or lane-level scoring; it is not "
            "an official CULane benchmark result."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "train_steps": int(args.train_steps),
        "train_split": "train",
        "eval_split": "val",
        "eval_sample_strategy": args.sample_strategy,
        "eval_images": len(eval_indices),
        "eval_dataset_indices": eval_indices,
        "seed": int(args.seed),
        "line_width": float(args.line_width),
        "top_k": int(args.top_k),
        "quality_power": float(args.quality_power),
        "quality_target": (
            "maximum range-aware row-space raster-IoU surrogate to any "
            "ground-truth lane, for every proposal"
        ),
        "quality_loss": {
            "name": "quality_focal_plus_pairwise_ranking",
            "focal_beta": float(args.quality_focal_beta),
            "rank_loss_weight": float(args.rank_loss_weight),
            "rank_target_margin": float(args.rank_target_margin),
        },
        "training_summary": {
            key: float(value) / float(args.train_steps)
            for key, value in training_totals.items()
        },
        "query_probe_parameters": sum(parameter.numel() for parameter in query_probe.parameters()),
        "geometry_probe_parameters": sum(
            parameter.numel() for parameter in geometry_probe.parameters()
        ),
        "query_feature_dim": int(first_query_features.shape[-1]),
        "geometry_feature_dim": int(first_geometry_features.shape[-1]),
        "evaluation": evaluation,
        "verdict": verdict,
    }
    if args.save_probe:
        save_path = Path(args.save_probe)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "query_probe": query_probe.state_dict(),
                "geometry_probe": geometry_probe.state_dict(),
                "query_feature_dim": int(first_query_features.shape[-1]),
                "geometry_feature_dim": int(first_geometry_features.shape[-1]),
                "source_checkpoint": args.checkpoint,
                "source_iteration": int(checkpoint_iteration),
                "train_steps": int(args.train_steps),
            },
            save_path,
        )
        payload["probe_checkpoint"] = str(save_path)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")
    if args.save_probe:
        print(f"probe_checkpoint: {args.save_probe}")


if __name__ == "__main__":
    main()
