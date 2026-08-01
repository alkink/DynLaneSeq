from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import math
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
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    official_proposal_gt_iou_matrix,
    trace_postprocess,
)
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.modeling.structured_queries import SetAwareLaneSelectionHead
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_official_set_selection import (
    official_unique_quality_targets,
    training_index_schedule,
)
from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    _frozen_outputs,
)


CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a trained unified LaneRowNet detector, cache the exact "
            "descriptors consumed by its selector, and train only that "
            "selector against stationary matcher and official-IoU one-to-one "
            "targets."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--train-cache-images", type=int, default=2048)
    parser.add_argument("--val-cache-images", type=int, default=256)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--train-steps", type=int, default=2500)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--continued-learning-rate", type=float, default=3e-4)
    parser.add_argument("--fresh-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--quality-focal-beta", type=float, default=2.0)
    parser.add_argument("--negative-weight", type=float, default=0.25)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-target-margin", type=float, default=0.10)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--min-gain-050-points", type=float, default=5.0)
    parser.add_argument("--min-gain-070-points", type=float, default=3.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-probe", required=True)
    parser.add_argument("--save-best-checkpoint", default="")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _amp_context(device: torch.device, name: str):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(name)
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _prepare_config(
    path: str,
    *,
    dataset_root: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg = load_config(path)
    if dataset_root:
        cfg.setdefault("dataset", {})["root"] = dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(num_workers)
    if int(num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg.setdefault("model", {}).setdefault("structured_query", {})[
        "intermediate_supervision"
    ] = False
    return cfg


def _stage_for_image(
    outputs: dict[str, torch.Tensor], batch_index: int
) -> dict[str, torch.Tensor]:
    return {
        name: outputs[name][batch_index].detach().float().cpu()
        for name in (
            "pred_x_rows",
            "range_norm",
            "exist_logits",
            "quality_logits",
        )
        if isinstance(outputs.get(name), torch.Tensor)
    }


def _cache_metadata(
    *,
    config: str,
    checkpoint: str,
    split: str,
    sample_count: int,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    stat = checkpoint_path.stat()
    return {
        "cache_version": CACHE_VERSION,
        "config": str(Path(config).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_size": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "split": str(split),
        "sample_strategy": "uniform",
        "sample_count": int(sample_count),
    }


@torch.no_grad()
def _collect_cache(
    model: nn.Module,
    selector: SetAwareLaneSelectionHead,
    matcher: Any,
    criterion: Any,
    cfg: dict[str, Any],
    *,
    split: str,
    sample_count: int,
    cache_batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_dtype: str,
    channels_last: bool,
    line_width: float,
    min_valid_rows: int,
    row_visibility_thresh: float,
    metadata: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    local_cfg = deepcopy(cfg)
    local_cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        cache_batch_size
    )
    local_cfg["dataloader"]["num_workers"] = int(num_workers)
    if int(num_workers) == 0:
        local_cfg["dataloader"]["persistent_workers"] = False
    base_loader = build_dataloader(local_cfg, split=split, training=False)
    max_batches = math.ceil(int(sample_count) / int(cache_batch_size))
    loader, dataset_indices = select_diagnostic_loader(
        base_loader,
        strategy="uniform",
        max_batches=max_batches,
        num_workers=num_workers,
    )

    features: list[torch.Tensor] = []
    matcher_targets: list[torch.Tensor] = []
    official_targets: list[torch.Tensor] = []
    candidate_valid_rows: list[torch.Tensor] = []
    official_iou_rows: list[torch.Tensor] = []
    stages: dict[str, list[torch.Tensor]] = {
        "pred_x_rows": [],
        "range_norm": [],
        "exist_logits": [],
        "quality_logits": [],
    }
    image_paths: list[str] = []
    collected = 0
    for images, batch_targets, metas in tqdm(
        loader,
        desc=f"frozen selector cache ({split})",
        ncols=90,
    ):
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
            batch_features = selector.build_selection_features(outputs)
        device_targets = nested_to_device(batch_targets, device)
        matches = matcher(outputs, device_targets)
        batch_matcher_targets = criterion.compute_set_selection_targets(
            outputs,
            device_targets,
            matches,
        )
        for batch_index, meta in enumerate(metas):
            if collected >= int(sample_count):
                break
            stage = _stage_for_image(outputs, batch_index)
            record = {"stages": {"main": stage}, "meta": meta}
            official_iou, candidate_valid = official_proposal_gt_iou_matrix(
                record,
                "main",
                line_width=float(line_width),
                min_valid_rows=int(min_valid_rows),
                row_visibility_thresh=float(row_visibility_thresh),
            )
            unique_target = official_unique_quality_targets(
                official_iou,
                candidate_valid,
            )
            features.append(batch_features[batch_index].detach().half().cpu())
            matcher_targets.append(
                batch_matcher_targets[batch_index].detach().float().cpu()
            )
            official_targets.append(unique_target.float().cpu())
            candidate_valid_rows.append(candidate_valid.bool().cpu())
            official_iou_rows.append(official_iou.float().cpu())
            for name, value in stage.items():
                stages[name].append(value)
            image_paths.append(str(meta.get("image_path", "")))
            collected += 1
        if collected >= int(sample_count):
            break
    if collected != int(sample_count):
        raise ValueError(f"requested {sample_count} {split} images, got {collected}")
    cache = {
        "metadata": {
            **metadata,
            "dataset_indices": dataset_indices[:collected],
            "image_paths": image_paths,
            "feature_dim": int(features[0].shape[-1]),
            "num_candidates": int(features[0].shape[0]),
        },
        "features": torch.stack(features),
        "matcher_targets": torch.stack(matcher_targets),
        "official_targets": torch.stack(official_targets),
        "candidate_valid": torch.stack(candidate_valid_rows),
        "official_iou": official_iou_rows,
        "stage": {
            name: torch.stack(rows) for name, rows in stages.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


def _load_or_collect_cache(
    model: nn.Module,
    selector: SetAwareLaneSelectionHead,
    matcher: Any,
    criterion: Any,
    cfg: dict[str, Any],
    *,
    split: str,
    sample_count: int,
    cache_batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_dtype: str,
    channels_last: bool,
    line_width: float,
    min_valid_rows: int,
    row_visibility_thresh: float,
    metadata: dict[str, Any],
    output_path: Path,
    reuse_cache: bool,
) -> dict[str, Any]:
    if reuse_cache and output_path.exists():
        try:
            cache = torch.load(output_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(output_path, map_location="cpu")
        actual = cache.get("metadata", {})
        mismatches = {
            key: (actual.get(key), value)
            for key, value in metadata.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise ValueError(f"cache signature mismatch: {mismatches}")
        return cache
    return _collect_cache(
        model,
        selector,
        matcher,
        criterion,
        cfg,
        split=split,
        sample_count=sample_count,
        cache_batch_size=cache_batch_size,
        num_workers=num_workers,
        device=device,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        line_width=line_width,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
        metadata=metadata,
        output_path=output_path,
    )


def _selection_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    focal_beta: float,
    negative_weight: float,
    rank_weight: float,
    rank_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = targets.to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    modulation = (targets - probability).abs().pow(float(focal_beta))
    per_candidate = modulation * F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    weights = torch.where(
        targets > 0.0,
        torch.ones_like(targets),
        torch.full_like(targets, float(negative_weight)),
    )
    quality = (per_candidate * weights).sum() / weights.sum().clamp_min(1.0)
    target_delta = targets.unsqueeze(-1) - targets.unsqueeze(-2)
    pair_weight = (target_delta - float(rank_margin)).clamp_min(0.0)
    logit_delta = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    ranking = (
        F.softplus(-logit_delta) * pair_weight.detach()
    ).sum() / pair_weight.sum().clamp_min(1e-6)
    total = quality + float(rank_weight) * ranking
    return total, quality, ranking


def _new_counts() -> dict[str, Any]:
    return {
        "gt": 0,
        "selected": 0,
        "hits": {0.5: 0, 0.7: 0},
    }


def _update_counts(
    counts: dict[str, Any],
    official_iou: torch.Tensor,
    selected_ids: Iterable[int],
) -> None:
    selected = [int(value) for value in selected_ids]
    counts["gt"] += int(official_iou.shape[0])
    counts["selected"] += len(selected)
    for threshold in (0.5, 0.7):
        assignment = evaluator_hungarian_assignment(
            official_iou,
            selected,
            threshold=threshold,
        )
        counts["hits"][threshold] += int(assignment.hit_count)


def _finish_counts(counts: dict[str, Any]) -> dict[str, Any]:
    gt = int(counts["gt"])
    selected = int(counts["selected"])
    result: dict[str, Any] = {
        "gt_lanes": gt,
        "selected_predictions": selected,
    }
    for threshold in (0.5, 0.7):
        suffix = f"{int(round(100 * threshold)):03d}"
        tp = int(counts["hits"][threshold])
        precision = tp / float(max(selected, 1))
        recall = tp / float(max(gt, 1))
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        result[f"tp_{suffix}"] = tp
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"f1_{suffix}"] = f1
    return result


def _topk_ids(
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    top_k: int,
) -> list[int]:
    ids = [
        index
        for index in range(int(scores.shape[0]))
        if bool(candidate_valid[index])
    ]
    ids.sort(key=lambda index: float(scores[index]), reverse=True)
    return ids[: int(top_k)]


@torch.no_grad()
def _evaluate(
    head: SetAwareLaneSelectionHead,
    cache: dict[str, Any],
    *,
    device: torch.device,
    probe_batch_size: int,
    top_k: int,
    input_h: int,
    input_w: int,
    min_valid_rows: int,
    row_visibility_thresh: float,
    nms_distance: float,
    nms_min_overlap_points: int,
) -> dict[str, Any]:
    head.eval()
    feature_tensor = cache["features"]
    logits_rows: list[torch.Tensor] = []
    for start in range(0, int(feature_tensor.shape[0]), int(probe_batch_size)):
        features = feature_tensor[start : start + int(probe_batch_size)].to(
            device=device,
            dtype=torch.float32,
        )
        logits_rows.append(head.score_selection_features(features).cpu())
    logits = torch.cat(logits_rows, dim=0)
    raw_counts = _new_counts()
    nms_counts = _new_counts()
    oracle_counts = {0.5: 0, 0.7: 0}
    oracle_gt = 0
    for image_index, official_iou in enumerate(cache["official_iou"]):
        scores = torch.sigmoid(logits[image_index])
        valid = cache["candidate_valid"][image_index].bool()
        raw_ids = _topk_ids(scores, valid, top_k)
        stage = {
            name: value[image_index]
            for name, value in cache["stage"].items()
        }
        trace = trace_postprocess(
            stage,
            input_h=input_h,
            input_w=input_w,
            score_thresh=-1.0,
            quality_power=0.0,
            min_valid_rows=min_valid_rows,
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_min_overlap_points,
            top_k=top_k,
            row_visibility_thresh=row_visibility_thresh,
            score_override={
                index: float(scores[index])
                for index in range(int(scores.shape[0]))
            },
        )
        _update_counts(raw_counts, official_iou, raw_ids)
        _update_counts(nms_counts, official_iou, trace["selected_ids"])
        oracle_gt += int(official_iou.shape[0])
        for threshold in (0.5, 0.7):
            oracle_counts[threshold] += int(
                cardinality_oracle_assignment(
                    official_iou,
                    threshold=threshold,
                    top_k=top_k,
                    candidate_valid=valid,
                ).hit_count
            )
    return {
        "raw_top4": _finish_counts(raw_counts),
        "nms_top4": _finish_counts(nms_counts),
        "oracle_top4": {
            f"{threshold:.2f}": {
                "gt_lanes": int(oracle_gt),
                "tp": int(oracle_counts[threshold]),
                "recall": oracle_counts[threshold] / float(max(oracle_gt, 1)),
            }
            for threshold in (0.5, 0.7)
        },
    }


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _objective(metrics: dict[str, Any]) -> float:
    row = metrics["raw_top4"]
    return float(row["recall_050"]) + float(row["recall_070"])


def _export_checkpoint(
    source_path: str,
    output_path: str,
    selector_state: dict[str, torch.Tensor],
    *,
    arm: str,
) -> None:
    try:
        source = torch.load(source_path, map_location="cpu", weights_only=False)
    except TypeError:
        source = torch.load(source_path, map_location="cpu")
    model_state = dict(source["model"])
    prefix = "structured_query_head.set_selection_head."
    for name, value in selector_state.items():
        key = prefix + name
        if key not in model_state:
            raise KeyError(f"selector parameter is absent from checkpoint: {key}")
        model_state[key] = value.detach().cpu()
    payload = {
        "model": model_state,
        "iteration": int(source.get("iteration", 0)),
        "cfg": source.get("cfg", {}),
        "diagnostic": {
            "source_checkpoint": str(source_path),
            "frozen_selector_arm": str(arm),
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "cache_batch_size",
        "train_cache_images",
        "val_cache_images",
        "train_steps",
        "probe_batch_size",
        "eval_interval",
        "top_k",
        "min_valid_rows",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    if float(args.negative_weight) <= 0.0:
        raise ValueError("negative_weight must be positive")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _set_seed(args.seed)
    device = torch.device(args.device)
    cfg = _prepare_config(
        args.config,
        dataset_root=args.dataset_root,
        batch_size=args.cache_batch_size,
        num_workers=args.num_workers,
    )
    input_h = int(cfg.get("model", {}).get("input_h", 640))
    input_w = int(cfg.get("model", {}).get("input_w", 1600))

    model = build_model(cfg)
    if model.structured_query_head is None:
        raise ValueError("frozen selector probe requires a structured query head")
    initial_selector = model.structured_query_head.set_selection_head
    if initial_selector is None or not initial_selector.unified_score:
        raise ValueError("probe requires set_selection.unified_score=true")
    fresh_matcher_head = deepcopy(initial_selector)
    fresh_official_head = deepcopy(initial_selector)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    loaded_selector = model.structured_query_head.set_selection_head
    if loaded_selector is None or not loaded_selector.unified_score:
        raise ValueError("checkpoint did not load a unified selector")
    source_head = deepcopy(loaded_selector)
    continued_head = deepcopy(loaded_selector)

    model.requires_grad_(False)
    model = model.to(device).eval()
    model.structured_query_head.intermediate_supervision = False
    loaded_selector = model.structured_query_head.set_selection_head
    if loaded_selector is None:
        raise RuntimeError("selector disappeared after device transfer")
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)

    caches: dict[str, dict[str, Any]] = {}
    for split, sample_count, path_value in (
        ("train", args.train_cache_images, args.train_cache),
        ("val", args.val_cache_images, args.val_cache),
    ):
        metadata = _cache_metadata(
            config=args.config,
            checkpoint=args.checkpoint,
            split=split,
            sample_count=sample_count,
        )
        caches[split] = _load_or_collect_cache(
            model,
            loaded_selector,
            matcher,
            criterion,
            cfg,
            split=split,
            sample_count=sample_count,
            cache_batch_size=args.cache_batch_size,
            num_workers=args.num_workers,
            device=device,
            amp_dtype=args.amp_dtype,
            channels_last=channels_last,
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
            metadata=metadata,
            output_path=Path(path_value),
            reuse_cache=args.reuse_cache,
        )
    overlap = set(caches["train"]["metadata"]["image_paths"]) & set(
        caches["val"]["metadata"]["image_paths"]
    )
    if overlap:
        raise ValueError(f"train/val cache overlap: {sorted(overlap)[0]}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    source_head = source_head.float().to(device).eval()
    heads = {
        "continued_matcher": continued_head.float().to(device).train(),
        "fresh_matcher": fresh_matcher_head.float().to(device).train(),
        "fresh_official": fresh_official_head.float().to(device).train(),
    }
    optimizers = {
        "continued_matcher": torch.optim.AdamW(
            heads["continued_matcher"].parameters(),
            lr=float(args.continued_learning_rate),
            weight_decay=float(args.weight_decay),
        ),
        "fresh_matcher": torch.optim.AdamW(
            heads["fresh_matcher"].parameters(),
            lr=float(args.fresh_learning_rate),
            weight_decay=float(args.weight_decay),
        ),
        "fresh_official": torch.optim.AdamW(
            heads["fresh_official"].parameters(),
            lr=float(args.fresh_learning_rate),
            weight_decay=float(args.weight_decay),
        ),
    }
    source_metrics = _evaluate(
        source_head,
        caches["val"],
        device=device,
        probe_batch_size=args.probe_batch_size,
        top_k=args.top_k,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        nms_distance=args.nms_distance,
        nms_min_overlap_points=args.nms_min_overlap_points,
    )
    best = {
        name: {
            "objective": float("-inf"),
            "step": 0,
            "state": _cpu_state(head),
            "metrics": None,
        }
        for name, head in heads.items()
    }
    trajectory: list[dict[str, Any]] = []
    schedule = training_index_schedule(
        num_examples=int(caches["train"]["features"].shape[0]),
        batch_size=args.probe_batch_size,
        steps=args.train_steps,
        seed=args.seed,
    )
    running = {
        name: {"total": 0.0, "quality": 0.0, "ranking": 0.0}
        for name in heads
    }
    for step in range(1, int(args.train_steps) + 1):
        indices = schedule[step - 1]
        features = caches["train"]["features"][indices].to(
            device=device,
            dtype=torch.float32,
        )
        for name, head in heads.items():
            target_key = (
                "official_targets"
                if name == "fresh_official"
                else "matcher_targets"
            )
            targets = caches["train"][target_key][indices].to(
                device=device,
                dtype=torch.float32,
            )
            optimizer = optimizers[name]
            head.train()
            optimizer.zero_grad(set_to_none=True)
            logits = head.score_selection_features(features)
            total, quality, ranking = _selection_loss(
                logits,
                targets,
                focal_beta=args.quality_focal_beta,
                negative_weight=args.negative_weight,
                rank_weight=args.rank_loss_weight,
                rank_margin=args.rank_target_margin,
            )
            total.backward()
            optimizer.step()
            running[name]["total"] += float(total.detach())
            running[name]["quality"] += float(quality.detach())
            running[name]["ranking"] += float(ranking.detach())

        should_evaluate = (
            step == 1
            or step == int(args.train_steps)
            or step % int(args.eval_interval) == 0
        )
        if should_evaluate:
            row: dict[str, Any] = {"step": int(step), "arms": {}}
            for name, head in heads.items():
                metrics = _evaluate(
                    head,
                    caches["val"],
                    device=device,
                    probe_batch_size=args.probe_batch_size,
                    top_k=args.top_k,
                    input_h=input_h,
                    input_w=input_w,
                    min_valid_rows=args.min_valid_rows,
                    row_visibility_thresh=args.row_visibility_thresh,
                    nms_distance=args.nms_distance,
                    nms_min_overlap_points=args.nms_min_overlap_points,
                )
                objective = _objective(metrics)
                row["arms"][name] = metrics
                if objective > float(best[name]["objective"]):
                    best[name] = {
                        "objective": objective,
                        "step": int(step),
                        "state": _cpu_state(head),
                        "metrics": metrics,
                    }
                head.train()
            trajectory.append(row)
        if step % int(args.log_interval) == 0 or step == int(args.train_steps):
            denominator = float(args.log_interval)
            if step < int(args.log_interval):
                denominator = float(step)
            description = ", ".join(
                f"{name}={running[name]['total'] / max(denominator, 1.0):.4f}"
                for name in heads
            )
            print(f"selector-only step {step:05d}/{args.train_steps:05d} | {description}")
            running = {
                name: {"total": 0.0, "quality": 0.0, "ranking": 0.0}
                for name in heads
            }

    for name, head in heads.items():
        head.load_state_dict(best[name]["state"])
        head.eval()
    baseline_050 = float(source_metrics["raw_top4"]["recall_050"])
    baseline_070 = float(source_metrics["raw_top4"]["recall_070"])
    arms: dict[str, Any] = {}
    for name in heads:
        metrics = best[name]["metrics"]
        if metrics is None:
            raise RuntimeError(f"arm {name} was never evaluated")
        gain_050 = 100.0 * (
            float(metrics["raw_top4"]["recall_050"]) - baseline_050
        )
        gain_070 = 100.0 * (
            float(metrics["raw_top4"]["recall_070"]) - baseline_070
        )
        arms[name] = {
            "best_step": int(best[name]["step"]),
            "gain_raw_top4_recall_050_points": gain_050,
            "gain_raw_top4_recall_070_points": gain_070,
            "positive": bool(
                gain_050 >= float(args.min_gain_050_points)
                and gain_070 >= float(args.min_gain_070_points)
            ),
            "metrics": metrics,
        }
    best_name = max(heads, key=lambda name: float(best[name]["objective"]))
    matcher_positive = any(
        bool(arms[name]["positive"])
        for name in ("continued_matcher", "fresh_matcher")
    )
    official_positive = bool(arms["fresh_official"]["positive"])
    stationary_positive = bool(matcher_positive or official_positive)
    if matcher_positive:
        interpretation = "selector_capacity_present_joint_target_or_optimization_is_primary"
    elif official_positive:
        interpretation = "training_target_is_misaligned_with_official_set_selection"
    else:
        interpretation = "stationary_scalar_set_selector_still_insufficient"
    result = {
        "diagnostic_only": True,
        "warning": (
            "The detector and curve-aligned evidence are frozen. A positive "
            "result proves that stationary one-to-one supervision can recover "
            "selection; it is not a full jointly trained benchmark result."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "train_steps": int(args.train_steps),
        "train_cache_images": int(args.train_cache_images),
        "val_cache_images": int(args.val_cache_images),
        "targets": {
            "matcher_arms": "stationary_original_range_aware_matcher_assignment",
            "official_arm": "stationary_official_raster_iou_unique_hungarian",
        },
        "source": source_metrics,
        "arms": arms,
        "trajectory": trajectory,
        "gate": {
            "min_gain_050_points": float(args.min_gain_050_points),
            "min_gain_070_points": float(args.min_gain_070_points),
            "stationary_frozen_selector_positive": stationary_positive,
            "stationary_original_matcher_target_positive": bool(matcher_positive),
            "stationary_official_target_positive": bool(official_positive),
            "best_arm": best_name,
            "interpretation": interpretation,
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    probe_path = Path(args.save_probe)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_checkpoint": args.checkpoint,
            "checkpoint_iteration": int(checkpoint_iteration),
            "best_arm": best_name,
            "selector_states": {
                name: best[name]["state"] for name in heads
            },
            "best_steps": {name: int(best[name]["step"]) for name in heads},
        },
        probe_path,
    )
    if args.save_best_checkpoint:
        _export_checkpoint(
            args.checkpoint,
            args.save_best_checkpoint,
            best[best_name]["state"],
            arm=best_name,
        )
    print(json.dumps(result, indent=2))
    print(f"output_json: {output_path}")
    print(f"probe_checkpoint: {probe_path}")
    if args.save_best_checkpoint:
        print(f"patched_checkpoint: {args.save_best_checkpoint}")


if __name__ == "__main__":
    main()
