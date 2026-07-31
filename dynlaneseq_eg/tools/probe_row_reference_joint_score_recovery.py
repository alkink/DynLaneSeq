from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
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
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    geometry_aware_features,
    pairwise_lane_quality,
    pairwise_quality_ranking_loss,
    query_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a row-reference detector and train fresh joint existence/quality "
            "scorers on its current Hungarian targets. Historical, class-balanced, "
            "and geometry-aware arms distinguish stale head weights, score-loss "
            "imbalance, and an insufficient lane-query representation."
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
    parser.add_argument(
        "--sample-strategy", choices=("uniform", "sequential"), default="uniform"
    )
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced-rank-weight", type=float, default=0.25)
    parser.add_argument("--rank-target-margin", type=float, default=0.10)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quality-power", type=float, default=0.25)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=(0.50, 0.75))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-probe", default="")
    return parser.parse_args()


class ScalarProbe(nn.Module):
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


class JointScoreProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Keep the two heads independent, matching LaneRowNet's deployed heads.
        self.exist = ScalarProbe(int(input_dim))
        self.quality = ScalarProbe(int(input_dim))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.exist(features), self.quality(features)


def binary_focal_elements(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    return alpha_t * (1.0 - p_t).pow(float(gamma)) * bce


def historical_joint_loss(
    exist_logits: torch.Tensor,
    quality_logits: torch.Tensor,
    exist_targets: torch.Tensor,
    quality_targets: torch.Tensor,
    *,
    focal_alpha: float,
    focal_gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reproduce the current existence and quality reductions."""

    exist = binary_focal_elements(
        exist_logits,
        exist_targets,
        alpha=focal_alpha,
        gamma=focal_gamma,
    ).mean()
    quality = F.binary_cross_entropy_with_logits(
        quality_logits,
        quality_targets.to(dtype=quality_logits.dtype),
    )
    return exist + quality, {"exist": exist, "quality": quality, "rank": exist * 0.0}


def _two_class_mean(values: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    positive_mask = positive_mask.bool()
    terms = []
    if bool(positive_mask.any()):
        terms.append(values[positive_mask].mean())
    negative_mask = ~positive_mask
    if bool(negative_mask.any()):
        terms.append(values[negative_mask].mean())
    if not terms:
        return values.sum() * 0.0
    return torch.stack(terms).mean()


def balanced_joint_loss(
    exist_logits: torch.Tensor,
    quality_logits: torch.Tensor,
    exist_targets: torch.Tensor,
    quality_targets: torch.Tensor,
    *,
    focal_gamma: float,
    rank_weight: float,
    rank_target_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Give matched and unmatched candidates equal aggregate score weight."""

    positive = exist_targets > 0.5
    # Alpha is deliberately removed here: class balancing is explicit and
    # independent of the 4-positive/28-negative proposal ratio.
    exist_elements = binary_focal_elements(
        exist_logits,
        exist_targets,
        alpha=0.5,
        gamma=focal_gamma,
    )
    exist = _two_class_mean(exist_elements, positive)
    quality_elements = F.binary_cross_entropy_with_logits(
        quality_logits,
        quality_targets.to(dtype=quality_logits.dtype),
        reduction="none",
    )
    quality = _two_class_mean(quality_elements, positive)
    rank = pairwise_quality_ranking_loss(
        quality_logits,
        quality_targets,
        target_margin=float(rank_target_margin),
    )
    total = exist + quality + float(rank_weight) * rank
    return total, {"exist": exist, "quality": quality, "rank": rank}


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


def _frozen_outputs(model: nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    if model.structured_query_head is None:
        raise ValueError("joint score probe requires a structured query head")
    encoder_outputs = model.encoder.forward_features(
        images,
        inference_only=True,
        structured_only=True,
    )
    return model.structured_query_head(
        encoder_outputs["features"],
        inference_only=False,
    )


def training_targets(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    *,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce final-layer existence and historical quality targets."""

    pred_x = outputs.get("quality_pred_x_rows", outputs["pred_x_rows"]).detach().float()
    batch, candidates, _rows = pred_x.shape
    exist_targets = pred_x.new_zeros((batch, candidates))
    quality_targets = pred_x.new_zeros((batch, candidates))
    for batch_index, match in enumerate(matches):
        pred_indices = match["pred_indices"].to(pred_x.device)
        gt_indices = match["gt_indices"].to(pred_x.device)
        if int(pred_indices.numel()) == 0:
            continue
        exist_targets[batch_index, pred_indices] = 1.0
        gt_x = targets[batch_index]["x_rows"].to(
            pred_x.device,
            dtype=pred_x.dtype,
        )[gt_indices]
        valid = targets[batch_index]["valid_mask"].to(pred_x.device)[gt_indices].bool()
        valid = valid & torch.isfinite(gt_x)
        prediction = pred_x[batch_index, pred_indices]
        overlap = (
            torch.minimum(prediction + float(radius), gt_x + float(radius))
            - torch.maximum(prediction - float(radius), gt_x - float(radius))
        ).clamp(min=0.0)
        union = (4.0 * float(radius) - overlap).clamp(min=1e-6)
        valid_float = valid.to(dtype=prediction.dtype)
        valid_count = valid_float.sum(dim=-1)
        quality = ((overlap / union) * valid_float).sum(dim=-1)
        quality = quality / valid_count.clamp_min(1.0)
        quality = quality * (valid_count > 0).to(dtype=quality.dtype)
        quality_targets[batch_index, pred_indices] = quality.detach()
    return exist_targets, quality_targets


def _average_precision(scores: list[float], labels: list[int]) -> float:
    positive_count = int(sum(labels))
    if positive_count == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    found = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if int(labels[index]) == 0:
            continue
        found += 1
        precision_sum += float(found) / float(rank)
    return precision_sum / float(positive_count)


def _new_stats(thresholds: tuple[float, ...]) -> dict[str, Any]:
    return {
        "gt": 0,
        "selected": 0,
        "hits": {threshold: 0 for threshold in thresholds},
        "scores": [],
        "match_labels": [],
        "row_iou_labels": {threshold: [] for threshold in thresholds},
        "exist_positive_sum": 0.0,
        "exist_positive_count": 0,
        "exist_negative_sum": 0.0,
        "exist_negative_count": 0,
        "quality_positive_sum": 0.0,
        "quality_positive_count": 0,
        "quality_negative_sum": 0.0,
        "quality_negative_count": 0,
    }


def _finish_stats(stats: dict[str, Any], thresholds: tuple[float, ...]) -> dict[str, Any]:
    gt = max(int(stats["gt"]), 1)
    result = {
        "gt_lanes": int(stats["gt"]),
        "selected_candidates": int(stats["selected"]),
        "assignment_target_ap": _average_precision(stats["scores"], stats["match_labels"]),
        "mean_exist_probability_matched": float(stats["exist_positive_sum"])
        / float(max(int(stats["exist_positive_count"]), 1)),
        "mean_exist_probability_unmatched": float(stats["exist_negative_sum"])
        / float(max(int(stats["exist_negative_count"]), 1)),
        "mean_quality_probability_matched": float(stats["quality_positive_sum"])
        / float(max(int(stats["quality_positive_count"]), 1)),
        "mean_quality_probability_unmatched": float(stats["quality_negative_sum"])
        / float(max(int(stats["quality_negative_count"]), 1)),
    }
    for threshold in thresholds:
        suffix = f"{int(round(100 * threshold)):03d}"
        result[f"top4_row_recall_{suffix}"] = float(stats["hits"][threshold]) / float(gt)
        result[f"candidate_row_iou_ap_{suffix}"] = _average_precision(
            stats["scores"],
            stats["row_iou_labels"][threshold],
        )
    return result


def _update_probability_groups(
    stats: dict[str, Any],
    exist_probability: torch.Tensor,
    quality_probability: torch.Tensor,
    match_target: torch.Tensor,
) -> None:
    positive = match_target.bool()
    negative = ~positive
    stats["exist_positive_sum"] += float(exist_probability[positive].sum())
    stats["exist_positive_count"] += int(positive.sum())
    stats["exist_negative_sum"] += float(exist_probability[negative].sum())
    stats["exist_negative_count"] += int(negative.sum())
    stats["quality_positive_sum"] += float(quality_probability[positive].sum())
    stats["quality_positive_count"] += int(positive.sum())
    stats["quality_negative_sum"] += float(quality_probability[negative].sum())
    stats["quality_negative_count"] += int(negative.sum())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    matcher,
    probes: dict[str, JointScoreProbe],
    loader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    input_h: int,
    input_w: int,
    radius: float,
    line_width: float,
    top_k: int,
    quality_power: float,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    model.eval()
    for probe in probes.values():
        probe.eval()
    stats = {"current": _new_stats(thresholds)}
    stats.update({name: _new_stats(thresholds) for name in probes})
    oracle_hits = {threshold: 0 for threshold in thresholds}
    oracle_gt = 0
    for images, targets, _metas in tqdm(loader, desc="joint score probe eval", ncols=88):
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
        matches = matcher(outputs, targets)
        exist_targets, _quality_targets = training_targets(
            outputs,
            targets,
            matches,
            radius=radius,
        )
        q_features = query_features(outputs)
        g_features = geometry_aware_features(outputs, input_w=input_w)
        probe_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, probe in probes.items():
            features = g_features if name == "balanced_geometry" else q_features
            probe_outputs[name] = probe(features)

        current_exist = torch.softmax(outputs["exist_logits"].float(), dim=-1)[..., 0]
        current_quality = torch.sigmoid(outputs["quality_logits"].float())
        probabilities = {"current": (current_exist, current_quality)}
        probabilities.update(
            {
                name: (torch.sigmoid(pair[0]), torch.sigmoid(pair[1]))
                for name, pair in probe_outputs.items()
            }
        )
        score_rows = {
            name: exist * quality.clamp_min(1e-8).pow(float(quality_power))
            for name, (exist, quality) in probabilities.items()
        }

        for batch_index, target in enumerate(targets):
            pairwise = pairwise_lane_quality(
                outputs["pred_x_rows"][batch_index],
                outputs["range_norm"][batch_index],
                target["x_rows"],
                target["valid_mask"],
                input_h=input_h,
                line_width=line_width,
            )
            gt_count = int(pairwise.shape[0])
            candidate_count = int(pairwise.shape[1])
            oracle_gt += gt_count
            for threshold in thresholds:
                assignment = cardinality_oracle_assignment(
                    pairwise,
                    threshold=threshold,
                    top_k=int(top_k),
                    candidate_valid=torch.ones(candidate_count, dtype=torch.bool),
                )
                oracle_hits[threshold] += int(assignment.hit_count)
            best_row_iou = (
                pairwise.max(dim=0).values
                if gt_count > 0
                else pairwise.new_zeros(candidate_count)
            )
            match_row = exist_targets[batch_index].bool()
            for name, score_tensor in score_rows.items():
                row_stats = stats[name]
                score = score_tensor[batch_index].float()
                selected = torch.argsort(score, descending=True)[: int(top_k)].tolist()
                row_stats["gt"] += gt_count
                row_stats["selected"] += len(selected)
                for threshold in thresholds:
                    assignment = evaluator_hungarian_assignment(
                        pairwise,
                        selected,
                        threshold=threshold,
                    )
                    row_stats["hits"][threshold] += int(assignment.hit_count)
                row_stats["scores"].extend(float(value) for value in score.tolist())
                row_stats["match_labels"].extend(int(value) for value in match_row.tolist())
                for threshold in thresholds:
                    row_stats["row_iou_labels"][threshold].extend(
                        int(float(value) > threshold) for value in best_row_iou.tolist()
                    )
                exist_probability, quality_probability = probabilities[name]
                _update_probability_groups(
                    row_stats,
                    exist_probability[batch_index],
                    quality_probability[batch_index],
                    match_row,
                )
    return {
        "strategies": {
            name: _finish_stats(row, thresholds) for name, row in stats.items()
        },
        "oracle_topk_recall": {
            f"{int(round(100 * threshold)):03d}": float(oracle_hits[threshold])
            / float(max(oracle_gt, 1))
            for threshold in thresholds
        },
    }


def verdict(evaluation: dict[str, Any]) -> dict[str, Any]:
    rows = evaluation["strategies"]
    baseline = rows["current"]

    def gains(name: str) -> dict[str, float]:
        row = rows[name]
        return {
            "assignment_ap": float(row["assignment_target_ap"])
            - float(baseline["assignment_target_ap"]),
            "row_iou_ap_050": float(row["candidate_row_iou_ap_050"])
            - float(baseline["candidate_row_iou_ap_050"]),
            "row_iou_ap_075": float(row["candidate_row_iou_ap_075"])
            - float(baseline["candidate_row_iou_ap_075"]),
            "top4_recall_050_points": 100.0
            * (
                float(row["top4_row_recall_050"])
                - float(baseline["top4_row_recall_050"])
            ),
            "top4_recall_075_points": 100.0
            * (
                float(row["top4_row_recall_075"])
                - float(baseline["top4_row_recall_075"])
            ),
        }

    arm_gains = {name: gains(name) for name in ("historical_query", "balanced_query", "balanced_geometry")}
    historical_positive = (
        arm_gains["historical_query"]["assignment_ap"] >= 0.03
        and arm_gains["historical_query"]["row_iou_ap_050"] >= 0.01
    )
    balanced_query_positive = (
        arm_gains["balanced_query"]["assignment_ap"] >= 0.03
        and arm_gains["balanced_query"]["row_iou_ap_050"] >= 0.01
    )
    balanced_geometry_positive = (
        arm_gains["balanced_geometry"]["assignment_ap"] >= 0.03
        and arm_gains["balanced_geometry"]["row_iou_ap_050"] >= 0.01
    )
    geometry_increment = (
        float(rows["balanced_geometry"]["candidate_row_iou_ap_050"])
        - float(rows["balanced_query"]["candidate_row_iou_ap_050"])
    )
    if historical_positive:
        diagnosis = "stale_or_underoptimized_score_heads"
    elif balanced_query_positive:
        diagnosis = "historical_score_loss_imbalance"
    elif balanced_geometry_positive and geometry_increment >= 0.02:
        diagnosis = "lane_query_score_representation_limit"
    else:
        diagnosis = "frozen_score_features_not_sufficiently_separable"
    return {
        "diagnosis": diagnosis,
        "arm_gains_over_current": arm_gains,
        "geometry_ap050_gain_over_balanced_query": geometry_increment,
        "interpretation": {
            "historical_query_positive": (
                "Fresh heads recover ranking under the unchanged loss; old scorer state/history "
                "is the main bottleneck."
            ),
            "balanced_query_positive_only": (
                "The same lane query is sufficient, but the historical positive/negative "
                "reduction suppresses rare useful lanes."
            ),
            "balanced_geometry_positive_only": (
                "Final lane queries omit score-critical row geometry; use a geometry/evidence-aware scorer."
            ),
            "no_arm_positive": (
                "Do not start a full training run yet; the frozen decoder representation does not "
                "make useful and false candidates separable to these small scorers."
            ),
        },
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
    loss_cfg = cfg.get("loss", {})
    input_h = int(model_cfg.get("input_h", 288))
    input_w = int(model_cfg.get("input_w", 800))
    radius = float(loss_cfg.get("line_iou_radius", 15.0))
    focal_alpha = float(loss_cfg.get("focal_alpha", 0.25))
    focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if model.structured_query_head is None:
        raise ValueError("joint score probe requires a structured query head")
    model.structured_query_head.intermediate_supervision = False
    matcher = build_matcher(cfg)
    channels_last = bool(cfg.get("training", {}).get("channels_last", False) and device.type == "cuda")
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
    first_images, _first_targets, _first_metas = next(iter(train_loader))
    if channels_last:
        first_images = first_images.to(device, non_blocking=True, memory_format=torch.channels_last)
    else:
        first_images = first_images.to(device, non_blocking=True)
    with torch.no_grad(), _amp_context(device, amp_dtype):
        first_outputs = _frozen_outputs(model, first_images)
    query_dim = int(query_features(first_outputs).shape[-1])
    geometry_dim = int(geometry_aware_features(first_outputs, input_w=input_w).shape[-1])
    probes = {
        "historical_query": JointScoreProbe(query_dim).to(device),
        "balanced_query": JointScoreProbe(query_dim).to(device),
        "balanced_geometry": JointScoreProbe(geometry_dim).to(device),
    }
    optimizer = torch.optim.AdamW(
        [parameter for probe in probes.values() for parameter in probe.parameters()],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    del first_images, first_outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    running = {
        name: {"total": 0.0, "exist": 0.0, "quality": 0.0, "rank": 0.0}
        for name in probes
    }
    totals = {
        name: {"total": 0.0, "exist": 0.0, "quality": 0.0, "rank": 0.0}
        for name in probes
    }
    train_iterator = iter(train_loader)
    for step in tqdm(range(1, int(args.train_steps) + 1), desc="joint score probe train", ncols=88):
        try:
            images, targets, _metas = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, targets, _metas = next(train_iterator)
        if channels_last:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        with torch.no_grad(), _amp_context(device, amp_dtype):
            outputs = _frozen_outputs(model, images)
        matches = matcher(outputs, targets)
        exist_targets, quality_targets = training_targets(
            outputs,
            targets,
            matches,
            radius=radius,
        )
        q_features = query_features(outputs)
        g_features = geometry_aware_features(outputs, input_w=input_w)
        losses = {}
        parts = {}
        historical_logits = probes["historical_query"](q_features)
        losses["historical_query"], parts["historical_query"] = historical_joint_loss(
            *historical_logits,
            exist_targets,
            quality_targets,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        for name, features in (
            ("balanced_query", q_features),
            ("balanced_geometry", g_features),
        ):
            logits = probes[name](features)
            losses[name], parts[name] = balanced_joint_loss(
                *logits,
                exist_targets,
                quality_targets,
                focal_gamma=focal_gamma,
                rank_weight=args.balanced_rank_weight,
                rank_target_margin=args.rank_target_margin,
            )
        total_loss = torch.stack(tuple(losses.values())).sum()
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for probe in probes.values() for parameter in probe.parameters()],
            max_norm=5.0,
        )
        optimizer.step()
        for name in probes:
            values = {"total": losses[name], **parts[name]}
            for key, value in values.items():
                scalar = float(value.detach())
                running[name][key] += scalar
                totals[name][key] += scalar
        if int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
            denominator = float(args.log_interval)
            tqdm.write(
                f"step {step:05d}/{int(args.train_steps):05d} "
                + " ".join(
                    f"{name}={running[name]['total'] / denominator:.4f}"
                    for name in probes
                )
            )
            for name in probes:
                for key in running[name]:
                    running[name][key] = 0.0

    thresholds = tuple(float(value) for value in args.iou_thresholds)
    evaluation = evaluate(
        model,
        matcher,
        probes,
        eval_loader,
        device=device,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        input_h=input_h,
        input_w=input_w,
        radius=radius,
        line_width=args.line_width,
        top_k=args.top_k,
        quality_power=args.quality_power,
        thresholds=thresholds,
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "The detector and its geometry are frozen. Probe scores localize the "
            "precision bottleneck. Row-IoU probe metrics are differentiable-surrogate "
            "diagnostics, not official CULane test results."
        ),
        "purpose": (
            "Distinguish stale score heads, historical class-imbalanced score loss, "
            "and an insufficient lane-query score representation after lambda_obj=0.5 "
            "has repaired final-layer assignment coverage."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "matcher": {
            "lambda_obj": float(matcher.cfg.lambda_obj),
            "object_cost_type": str(matcher.cfg.object_cost_type),
            "assignment": str(matcher.cfg.assignment),
            "num_groups": int(matcher.cfg.num_groups),
        },
        "train_steps": int(args.train_steps),
        "eval_images": len(eval_indices),
        "eval_dataset_indices": eval_indices,
        "quality_power": float(args.quality_power),
        "iou_thresholds": list(thresholds),
        "query_feature_dim": query_dim,
        "geometry_feature_dim": geometry_dim,
        "loss_arms": {
            "historical_query": (
                "fresh query-only heads with the deployed focal-existence and all-candidate BCE quality reductions"
            ),
            "balanced_query": (
                "fresh query-only heads with equal aggregate matched/unmatched weight and quality ranking"
            ),
            "balanced_geometry": (
                "the balanced objective over lane query, visible row-state pooling, and geometry confidence features"
            ),
        },
        "training_loss_mean": {
            name: {
                key: float(value) / float(args.train_steps)
                for key, value in row.items()
            }
            for name, row in totals.items()
        },
        "probe_parameters": {
            name: sum(parameter.numel() for parameter in probe.parameters())
            for name, probe in probes.items()
        },
        "evaluation": evaluation,
        "verdict": verdict(evaluation),
    }
    if args.save_probe:
        save_path = Path(args.save_probe)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "probes": {name: probe.state_dict() for name, probe in probes.items()},
                "query_feature_dim": query_dim,
                "geometry_feature_dim": geometry_dim,
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
