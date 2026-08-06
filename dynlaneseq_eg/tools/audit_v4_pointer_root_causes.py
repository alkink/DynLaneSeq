from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.losses.loss_s0 import build_pointer_cluster_soft_targets
from dynlaneseq_eg.losses.range_aware_iou import (
    pairwise_range_aware_row_strip_iou,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.train import seed_everything


CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run three frozen-geometry root-cause audits for the V4 pointer: "
            "fixed-set memorization, frozen-descriptor quality probing, and "
            "a parallel Hungarian selector probe."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--memorize-images", type=int, default=64)
    parser.add_argument("--memorize-steps", type=int, default=3000)
    parser.add_argument("--memorize-batch-size", type=int, default=16)
    parser.add_argument("--memorize-lr", type=float, default=3e-4)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--parallel-steps", type=int, default=3000)
    parser.add_argument("--parallel-batch-size", type=int, default=32)
    parser.add_argument("--parallel-lr", type=float, default=1e-3)
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--representable-min", type=float, default=0.20)
    parser.add_argument("--max-selections", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _autocast_kwargs(device: torch.device, amp_dtype: str) -> dict[str, Any]:
    enabled = device.type == "cuda" and amp_dtype != "none"
    kwargs: dict[str, Any] = {"device_type": device.type, "enabled": enabled}
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        amp_dtype
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def _root_selector(model: nn.Module) -> nn.Module:
    root = getattr(model, "_orig_mod", model)
    structured = getattr(root, "structured_query_head", None)
    selector = getattr(structured, "set_selection_head", None)
    if selector is None:
        raise ValueError("model has no set-selection head")
    if getattr(selector, "candidate_interaction", "") != "sequential_pointer":
        raise ValueError("root-cause audit requires a sequential pointer head")
    return selector


def _selection_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return (
        cfg.get("model", {})
        .get("structured_query", {})
        .get("set_selection", {})
    )


def _teacher_target_mode(selector: nn.Module) -> str:
    mode = str(getattr(selector, "pointer_teacher_mode", ""))
    if mode == "cluster_soft_remaining_mixture":
        return "remaining_cluster_mixture"
    if mode == "cluster_soft_randomized":
        return "sampled_cluster"
    raise ValueError(
        "root-cause audit requires cluster_soft_randomized or "
        "cluster_soft_remaining_mixture"
    )


def _cache_signature(
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_stat = checkpoint_path.stat()
    dataset_cfg = cfg.get("dataset", {})
    raw_list = dataset_cfg.get("lists", {}).get(
        str(args.split),
        f"list/{args.split}.txt",
    )
    list_path = Path(raw_list).expanduser()
    if not list_path.is_absolute():
        list_path = Path(args.dataset_root).expanduser() / list_path
    list_path = list_path.resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f"missing dataset list: {list_path}")
    return {
        "version": CACHE_VERSION,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "split": str(args.split),
        "list_path": str(list_path),
        "list_sha256": _sha256(list_path),
        "sample_count": int(args.sample_count),
        "feature_batch_size": int(args.feature_batch_size),
        "sample_strategy": str(args.sample_strategy),
        "teacher_seed": int(args.seed),
        "max_selections": int(args.max_selections),
    }


def _slice_batch_tensor(value: torch.Tensor, index: int) -> torch.Tensor:
    return value[index : index + 1]


@torch.no_grad()
def _collect_frozen_cache(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    model: nn.Module,
    selector: nn.Module,
    signature: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    cache_path = Path(args.cache_path)
    if args.reuse_cache and cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if payload.get("signature") != signature:
            raise ValueError(
                "frozen-feature cache signature mismatch; remove the cache or "
                "omit --reuse-cache"
            )
        return payload

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("dataset", {})["root"] = str(args.dataset_root)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.feature_batch_size
    )
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    loader = build_dataloader(cfg, split=args.split, training=False)
    max_batches = math.ceil(int(args.sample_count) / int(args.feature_batch_size))
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=max_batches,
        num_workers=int(args.num_workers),
    )
    selection_cfg = _selection_cfg(cfg)
    loss_cfg = cfg.get("loss", {})
    input_h = int(loss_cfg.get("input_h", cfg.get("model", {}).get("input_h", 288)))
    line_width = float(loss_cfg.get("set_selection_line_width", 30.0))
    min_valid_rows = int(loss_cfg.get("set_selection_min_valid_rows", 5))
    target_mode = _teacher_target_mode(selector)
    autocast_kwargs = _autocast_kwargs(device, args.amp_dtype)
    records: list[dict[str, Any]] = []

    model.eval()
    for images, targets, metas in tqdm(
        loader,
        desc="frozen descriptor cache",
        ncols=88,
    ):
        if len(records) >= int(args.sample_count):
            break
        images = images.to(device, non_blocking=True)
        targets_device = nested_to_device(targets, device)
        with torch.autocast(**autocast_kwargs):
            outputs = model(images)
            features = selector.build_selection_features(outputs)
            relations = selector.build_pairwise_relations(outputs)
            hidden = selector.encode_selection_features(features, relations)
        candidate_valid = selector.build_pointer_candidate_valid(outputs)
        pointer_indices = outputs["selection_pointer_indices"]
        unary_logits = outputs["selection_logits"]
        batch_size = int(images.shape[0])
        for local_index in range(batch_size):
            if len(records) >= int(args.sample_count):
                break
            global_index = len(records)
            single_outputs = {
                "pred_x_rows": _slice_batch_tensor(
                    outputs["pred_x_rows"], local_index
                ),
                "range_norm": _slice_batch_tensor(
                    outputs["range_norm"], local_index
                ),
            }
            single_target = [targets_device[local_index]]
            teacher = build_pointer_cluster_soft_targets(
                single_outputs,
                single_target,
                max_selections=int(args.max_selections),
                input_h=input_h,
                line_width=line_width,
                min_valid_rows=min_valid_rows,
                representable_min=float(
                    selection_cfg.get(
                        "pointer_cluster_representable_min",
                        args.representable_min,
                    )
                ),
                support_quality_delta=float(
                    selection_cfg.get("pointer_cluster_quality_delta", 0.10)
                ),
                temperature=float(
                    selection_cfg.get("pointer_cluster_temperature", 0.03)
                ),
                base_seed=int(args.seed),
                iteration=0,
                visit=global_index,
                target_mode=target_mode,
            )
            target = targets_device[local_index]
            quality, quality_candidate_valid, valid_gt = (
                pairwise_range_aware_row_strip_iou(
                    single_outputs["pred_x_rows"][0].float(),
                    single_outputs["range_norm"][0].float(),
                    target["x_rows"].to(device=device).float(),
                    target["valid_mask"].to(device=device).bool(),
                    input_h=input_h,
                    line_width=line_width,
                    min_valid_rows=min_valid_rows,
                )
            )
            quality = quality[:, valid_gt]
            valid = candidate_valid[local_index].bool() & quality_candidate_valid
            records.append(
                {
                    "image_id": str(metas[local_index].get("image_path", global_index)),
                    "features": features[local_index].detach().float().cpu(),
                    "relations": relations[local_index].detach().float().cpu(),
                    "hidden": hidden[local_index].detach().float().cpu(),
                    "candidate_valid": valid.detach().cpu(),
                    "quality": quality.detach().float().cpu(),
                    "unary_logits": unary_logits[local_index].detach().float().cpu(),
                    "pointer_indices": pointer_indices[local_index]
                    .detach()
                    .long()
                    .cpu(),
                    "teacher": {
                        key: value[0].detach().cpu()
                        for key, value in teacher.items()
                        if isinstance(value, torch.Tensor)
                    },
                }
            )

    if len(records) != int(args.sample_count):
        raise RuntimeError(
            f"requested {args.sample_count} records but collected {len(records)}"
        )
    payload = {
        "signature": signature,
        "metadata": {
            "sampled_dataset_indices": sampled_indices[: len(records)],
            "augmentations_enabled": False,
            "fixed_teacher": True,
            "teacher_target_mode": target_mode,
            "input_h": input_h,
            "line_width": line_width,
            "min_valid_rows": min_valid_rows,
        },
        "records": records,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def _stack(
    records: list[dict[str, Any]],
    indices: Iterable[int],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack([records[index][key] for index in indices]).to(device)


def _stack_teacher(
    records: list[dict[str, Any]],
    indices: Iterable[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    ids = list(indices)
    keys = records[ids[0]]["teacher"].keys()
    return {
        key: torch.stack([records[index]["teacher"][key] for index in ids]).to(
            device
        )
        for key in keys
    }


def _quality_max(record: dict[str, Any]) -> torch.Tensor:
    quality = record["quality"].float()
    if int(quality.shape[1]) == 0:
        return torch.zeros(int(quality.shape[0]), dtype=torch.float32)
    return quality.amax(dim=-1)


def _selector_components(
    selector: nn.Module,
    features: torch.Tensor,
    relations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    hidden = selector.encode_selection_features(features, relations)
    if selector.pointer_quality_policy_mode == "decoupled":
        quality_input = hidden.detach()
        quality_hidden = quality_input + selector.pointer_quality_adapter(
            quality_input
        )
        unary = selector.output(selector.output_norm(quality_hidden)).squeeze(-1)
        policy = selector.pointer_policy_output(
            selector.pointer_policy_output_norm(hidden)
        ).squeeze(-1)
    else:
        unary = selector.output(selector.output_norm(hidden)).squeeze(-1)
        policy = None
    return hidden, unary.float(), None if policy is None else policy.float()


def _pointer_losses(
    logits: torch.Tensor,
    unary_logits: torch.Tensor,
    teacher: dict[str, torch.Tensor],
    quality_target: torch.Tensor,
    *,
    stop_weight: float,
    quality_weight: float,
    focal_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probabilities = teacher["probabilities"].to(logits.dtype)
    active = teacher["active"].bool()
    log_probability = F.log_softmax(logits, dim=-1)
    per_step = -(probabilities * log_probability).sum(dim=-1)
    stop = (probabilities[..., -1] > 0.5) & active
    step_weight = torch.where(
        stop,
        torch.full_like(per_step, float(stop_weight)),
        torch.ones_like(per_step),
    )
    denominator = (step_weight * active).sum().clamp_min(1.0)
    sequence = (per_step * step_weight * active).sum() / denominator
    unary_probability = torch.sigmoid(unary_logits)
    modulation = (quality_target - unary_probability).abs().pow(float(focal_beta))
    quality = (
        modulation
        * F.binary_cross_entropy_with_logits(
            unary_logits,
            quality_target,
            reduction="none",
        )
    ).mean()
    total = sequence + float(quality_weight) * quality
    return total, {"sequence": sequence, "quality": quality}


def _hungarian_hits(
    quality: torch.Tensor,
    selected: Iterable[int],
    threshold: float,
) -> int:
    selected_ids = [int(value) for value in selected if int(value) >= 0]
    gt_count = int(quality.shape[1])
    if not selected_ids or gt_count == 0:
        return 0
    local = quality[selected_ids].float().cpu()
    qualified = local >= float(threshold)
    assignment_size = min(int(local.shape[0]), int(local.shape[1]))
    reward = qualified.float() * float(assignment_size + 1) + local
    row, col = linear_sum_assignment((-reward).numpy())
    return sum(bool(qualified[r, c]) for r, c in zip(row, col))


def selection_metrics(
    records: list[dict[str, Any]],
    selections: list[list[int]],
    thresholds: tuple[float, ...] = (0.50, 0.75),
) -> dict[str, Any]:
    if len(records) != len(selections):
        raise ValueError("record/selection length mismatch")
    result: dict[str, Any] = {}
    for threshold in thresholds:
        tp = fp = fn = 0
        for record, selected in zip(records, selections):
            valid_selected = [
                index
                for index in selected
                if 0 <= int(index) < int(record["quality"].shape[0])
                and bool(record["candidate_valid"][int(index)])
            ]
            hits = _hungarian_hits(record["quality"], valid_selected, threshold)
            gt_count = int(record["quality"].shape[1])
            tp += hits
            fp += max(len(valid_selected) - hits, 0)
            fn += max(gt_count - hits, 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        result[f"iou_{threshold:.2f}"] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    result["mean_selected_per_image"] = sum(map(len, selections)) / max(
        len(selections), 1
    )
    return result


def _pointer_selections(indices: torch.Tensor) -> list[list[int]]:
    rows: list[list[int]] = []
    for row in indices.detach().cpu().long():
        selected: list[int] = []
        for value in row.tolist():
            if int(value) < 0:
                break
            selected.append(int(value))
        rows.append(selected)
    return rows


@torch.no_grad()
def _evaluate_memorization(
    selector: nn.Module,
    records: list[dict[str, Any]],
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    selector.eval()
    target_mass_sum = 0.0
    support_hits = 0
    active_count = 0
    candidate_hits = 0
    candidate_count = 0
    stop_hits = 0
    stop_count = 0
    ce_sum = 0.0
    entropy_sum = 0.0
    greedy_selections: list[list[int]] = []
    selected_records: list[dict[str, Any]] = []
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        features = _stack(records, batch_ids, "features", device)
        relations = _stack(records, batch_ids, "relations", device)
        valid = _stack(records, batch_ids, "candidate_valid", device).bool()
        teacher = _stack_teacher(records, batch_ids, device)
        hidden, unary, policy = _selector_components(selector, features, relations)
        rollout = selector.decode_pointer(
            hidden,
            relations,
            unary,
            valid,
            policy_logits=policy,
            teacher_indices=teacher["indices"].long(),
        )
        logits = rollout["selection_pointer_logits"].float()
        predicted_probability = torch.softmax(logits, dim=-1)
        target = teacher["probabilities"].float()
        active = teacher["active"].bool()
        support = target > 0.0
        predicted_class = logits.argmax(dim=-1)
        hit = support.gather(-1, predicted_class.unsqueeze(-1)).squeeze(-1)
        target_mass = (predicted_probability * support).sum(dim=-1)
        entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1)
        ce = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        candidate_steps = active & (target[..., :-1].sum(dim=-1) > 0.5)
        stop_steps = active & (target[..., -1] > 0.5)
        target_mass_sum += float(target_mass[active].sum().cpu())
        support_hits += int(hit[active].sum().cpu())
        active_count += int(active.sum().cpu())
        candidate_hits += int(hit[candidate_steps].sum().cpu())
        candidate_count += int(candidate_steps.sum().cpu())
        stop_hits += int(hit[stop_steps].sum().cpu())
        stop_count += int(stop_steps.sum().cpu())
        ce_sum += float(ce[active].sum().cpu())
        entropy_sum += float(entropy[active].sum().cpu())

        greedy = selector.decode_pointer(
            hidden,
            relations,
            unary,
            valid,
            policy_logits=policy,
        )
        greedy_selections.extend(
            _pointer_selections(greedy["selection_pointer_indices"])
        )
        selected_records.extend(records[index] for index in batch_ids)
    mean_ce = ce_sum / max(active_count, 1)
    mean_entropy = entropy_sum / max(active_count, 1)
    return {
        "teacher_prefix": {
            "active_steps": active_count,
            "mean_target_support_mass": target_mass_sum / max(active_count, 1),
            "support_top1_rate": support_hits / max(active_count, 1),
            "candidate_support_top1_rate": candidate_hits
            / max(candidate_count, 1),
            "stop_top1_rate": stop_hits / max(stop_count, 1),
            "soft_cross_entropy": mean_ce,
            "target_entropy_lower_bound": mean_entropy,
            "excess_cross_entropy": mean_ce - mean_entropy,
        },
        "free_greedy_row_strip": selection_metrics(
            selected_records,
            greedy_selections,
        ),
    }


def run_memorization_audit(
    selector_template: nn.Module,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    count = min(int(args.memorize_images), len(records))
    indices = list(range(count))
    selector = copy.deepcopy(selector_template).to(device)
    # Gradients are valid in eval mode.  Keeping dropout disabled makes this a
    # strict fixed-input/fixed-teacher capacity test rather than an
    # augmentation or stochastic-regularization experiment.
    selector.eval()
    for parameter in selector.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=float(args.memorize_lr),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.memorize_steps), 1),
        eta_min=float(args.memorize_lr) * 0.10,
    )
    loss_cfg = cfg.get("loss", {})
    stop_weight = float(loss_cfg.get("pointer_stop_weight", 1.0))
    quality_weight = float(loss_cfg.get("pointer_quality_weight", 0.10))
    focal_beta = float(loss_cfg.get("set_selection_focal_beta", 0.0))
    checkpoints = sorted(
        {
            0,
            10,
            50,
            100,
            250,
            500,
            1000,
            int(args.memorize_steps),
        }
    )
    checkpoints = [value for value in checkpoints if value <= args.memorize_steps]
    trajectory: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 11)
    order = torch.randperm(count, generator=generator).tolist()
    cursor = 0

    def snapshot(step: int) -> None:
        row = _evaluate_memorization(
            selector,
            records,
            indices,
            device,
            int(args.memorize_batch_size),
        )
        row["step"] = int(step)
        trajectory.append(row)

    snapshot(0)
    for step in tqdm(
        range(1, int(args.memorize_steps) + 1),
        desc="64-image pointer memorization",
        ncols=88,
    ):
        if cursor + int(args.memorize_batch_size) > count:
            order = torch.randperm(count, generator=generator).tolist()
            cursor = 0
        batch_ids = order[cursor : cursor + int(args.memorize_batch_size)]
        cursor += int(args.memorize_batch_size)
        features = _stack(records, batch_ids, "features", device)
        relations = _stack(records, batch_ids, "relations", device)
        valid = _stack(records, batch_ids, "candidate_valid", device).bool()
        teacher = _stack_teacher(records, batch_ids, device)
        quality_target = torch.stack(
            [_quality_max(records[index]) for index in batch_ids]
        ).to(device)
        hidden, unary, policy = _selector_components(selector, features, relations)
        rollout = selector.decode_pointer(
            hidden,
            relations,
            unary,
            valid,
            policy_logits=policy,
            teacher_indices=teacher["indices"].long(),
        )
        total, _parts = _pointer_losses(
            rollout["selection_pointer_logits"],
            unary,
            teacher,
            quality_target,
            stop_weight=stop_weight,
            quality_weight=quality_weight,
            focal_beta=focal_beta,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step in checkpoints:
            snapshot(step)
    def capacity_score(row: dict[str, Any]) -> float:
        prefix = row["teacher_prefix"]
        return min(
            float(prefix["mean_target_support_mass"]),
            float(prefix["candidate_support_top1_rate"]),
            float(prefix["stop_top1_rate"]),
        )

    best = max(trajectory, key=capacity_score)
    best_prefix = best["teacher_prefix"]
    passed = all(
        (
            float(best_prefix["mean_target_support_mass"]) >= 0.95,
            float(best_prefix["candidate_support_top1_rate"]) >= 0.95,
            float(best_prefix["stop_top1_rate"]) >= 0.95,
        )
    )
    return {
        "contract": {
            "images": count,
            "augmentation": "disabled",
            "teacher": "fixed_per_image",
            "geometry": "cached_and_frozen",
            "selector_initialization": "warm_start_from_checkpoint",
            "dropout": "disabled",
            "steps": int(args.memorize_steps),
            "learning_rate": float(args.memorize_lr),
            "schedule": "cosine_to_0.1x",
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in selector.parameters()
                if parameter.requires_grad
            ),
        },
        "trajectory": trajectory,
        "best_step": int(best["step"]),
        "best_teacher_prefix": best_prefix,
        "pass_threshold": {
            "mean_target_support_mass": 0.95,
            "candidate_support_top1_rate": 0.95,
            "stop_top1_rate": 0.95,
        },
        "passed": bool(passed),
    }


class CandidateQualityProbe(nn.Module):
    def __init__(self, hidden_dim: int, nonlinear: bool) -> None:
        super().__init__()
        if nonlinear:
            inner = max(hidden_dim // 2, 32)
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, inner),
                nn.GELU(),
                nn.Linear(inner, 1),
            )
        else:
            self.net = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


class ParallelHungarianProbe(nn.Module):
    def __init__(self, hidden_dim: int, max_selections: int) -> None:
        super().__init__()
        inner = max(hidden_dim // 2, 32)
        self.candidate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, 1),
        )
        self.count = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, max_selections + 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.candidate(hidden).squeeze(-1)
        weight = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return logits, self.count(pooled)


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().flatten()
    right = right.double().flatten()
    if left.numel() < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1e-12:
        return 0.0
    return float((left * right).sum() / denominator)


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.flatten())
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float64)
    return ranks


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores = scores.double().flatten()
    labels = labels.bool().flatten()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = _rank(scores) + 1.0
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)


def _cluster_ranking_metrics(
    records: list[dict[str, Any]],
    scores: list[torch.Tensor],
    representable_threshold: float = 0.50,
) -> dict[str, Any]:
    if len(records) != len(scores):
        raise ValueError("record/score length mismatch")
    ranks: list[int] = []
    regrets: list[float] = []
    for record, candidate_score in zip(records, scores):
        quality = record["quality"].float()
        valid = record["candidate_valid"].bool()
        owner = (
            quality.argmax(dim=-1)
            if int(quality.shape[1]) > 0
            else torch.full(
                (int(quality.shape[0]),),
                -1,
                dtype=torch.long,
            )
        )
        for gt_index in range(int(quality.shape[1])):
            q = quality[:, gt_index]
            # Candidate-local quality is meaningful only inside its natural
            # lane cluster.  Without this ownership guard, the globally best
            # score could be (incorrectly) reused as the representative of
            # every GT in an image.
            support = valid & (owner == gt_index) & (q > 0.0)
            if not bool(support.any()) or float(q[support].max()) < representable_threshold:
                continue
            ids = torch.nonzero(support, as_tuple=False).flatten()
            selected = int(ids[candidate_score[ids].argmax()])
            selected_quality = float(q[selected])
            best = float(q[ids].max())
            rank = 1 + int((q[ids] > selected_quality + 1e-7).sum())
            ranks.append(rank)
            regrets.append(best - selected_quality)
    if not ranks:
        return {
            "representable_gt": 0,
            "top1_rate": 0.0,
            "top2_rate": 0.0,
            "mean_rank": 0.0,
            "mean_regret": 0.0,
            "p90_regret": 0.0,
        }
    regret_tensor = torch.tensor(regrets)
    return {
        "representable_gt": len(ranks),
        "top1_rate": sum(value == 1 for value in ranks) / len(ranks),
        "top2_rate": sum(value <= 2 for value in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "mean_regret": sum(regrets) / len(regrets),
        "p90_regret": float(torch.quantile(regret_tensor, 0.90)),
    }


@torch.no_grad()
def _evaluate_quality_scores(
    records: list[dict[str, Any]],
    scores: list[torch.Tensor],
) -> dict[str, Any]:
    if len(records) != len(scores):
        raise ValueError("record/score length mismatch")
    predicted: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for record, score in zip(records, scores):
        valid = record["candidate_valid"].bool()
        predicted.append(score.float()[valid])
        targets.append(_quality_max(record)[valid])
    pred = torch.cat(predicted) if predicted else torch.empty(0)
    target = torch.cat(targets) if targets else torch.empty(0)
    return {
        "candidate_count": int(target.numel()),
        "pearson": _pearson(pred, target),
        "spearman": _pearson(_rank(pred), _rank(target)),
        "auc_quality_ge_050": _auc(pred, target >= 0.50),
        "mae_after_sigmoid": float((torch.sigmoid(pred) - target).abs().mean())
        if target.numel()
        else 0.0,
        "per_gt_independent_cluster_ranking": _cluster_ranking_metrics(
            records,
            scores,
        ),
    }


def _record_split(
    record_count: int,
    holdout_stride: int,
) -> tuple[list[int], list[int]]:
    if holdout_stride < 2:
        raise ValueError("holdout stride must be at least two")
    holdout = [index for index in range(record_count) if index % holdout_stride == 0]
    train = [index for index in range(record_count) if index % holdout_stride != 0]
    if not train or not holdout:
        raise ValueError("probe split produced an empty partition")
    return train, holdout


@torch.no_grad()
def _probe_scores(
    probe: nn.Module,
    records: list[dict[str, Any]],
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    probe.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start : start + batch_size]
        hidden = _stack(records, ids, "hidden", device)
        rows.extend(probe(hidden).detach().float().cpu())
    return rows


def _fit_quality_probe(
    nonlinear: bool,
    records: list[dict[str, Any]],
    train_indices: list[int],
    holdout_indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    hidden_dim = int(records[0]["hidden"].shape[-1])
    probe = CandidateQualityProbe(hidden_dim, nonlinear=nonlinear).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.probe_lr),
        weight_decay=1e-4,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + (101 if nonlinear else 79))
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    cursor = 0
    probe.train()
    for _step in tqdm(
        range(int(args.probe_steps)),
        desc="nonlinear quality probe" if nonlinear else "linear quality probe",
        ncols=88,
    ):
        if cursor + int(args.probe_batch_size) > len(order):
            order = torch.randperm(len(train_indices), generator=generator).tolist()
            cursor = 0
        local = order[cursor : cursor + int(args.probe_batch_size)]
        cursor += int(args.probe_batch_size)
        ids = [train_indices[index] for index in local]
        hidden = _stack(records, ids, "hidden", device)
        valid = _stack(records, ids, "candidate_valid", device).bool()
        target = torch.stack([_quality_max(records[index]) for index in ids]).to(
            device
        )
        logits = probe(hidden)
        loss = F.binary_cross_entropy_with_logits(
            logits[valid],
            target[valid],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_records = [records[index] for index in train_indices]
    holdout_records = [records[index] for index in holdout_indices]
    train_scores = _probe_scores(
        probe,
        records,
        train_indices,
        device,
        int(args.probe_batch_size),
    )
    holdout_scores = _probe_scores(
        probe,
        records,
        holdout_indices,
        device,
        int(args.probe_batch_size),
    )
    return {
        "nonlinear": nonlinear,
        "steps": int(args.probe_steps),
        "train": _evaluate_quality_scores(train_records, train_scores),
        "holdout": _evaluate_quality_scores(holdout_records, holdout_scores),
    }


def run_quality_probe(
    records: list[dict[str, Any]],
    train_indices: list[int],
    holdout_indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    train_records = [records[index] for index in train_indices]
    holdout_records = [records[index] for index in holdout_indices]
    baseline_train = [record["unary_logits"].float() for record in train_records]
    baseline_holdout = [record["unary_logits"].float() for record in holdout_records]
    return {
        "contract": {
            "input": "frozen relation-encoded candidate hidden [N,256]",
            "target": "candidate-local max row-strip IoU",
            "image_disjoint_holdout": True,
            "train_images": len(train_indices),
            "holdout_images": len(holdout_indices),
        },
        "checkpoint_unary": {
            "train": _evaluate_quality_scores(train_records, baseline_train),
            "holdout": _evaluate_quality_scores(
                holdout_records,
                baseline_holdout,
            ),
        },
        "linear_probe": _fit_quality_probe(
            False,
            records,
            train_indices,
            holdout_indices,
            args,
            device,
        ),
        "nonlinear_probe": _fit_quality_probe(
            True,
            records,
            train_indices,
            holdout_indices,
            args,
            device,
        ),
    }


def unique_hungarian_target(
    quality: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    representable_min: float,
    max_selections: int,
) -> tuple[torch.Tensor, int]:
    candidates = int(quality.shape[0])
    target = torch.zeros(candidates, dtype=torch.bool)
    valid_ids = torch.nonzero(candidate_valid.bool(), as_tuple=False).flatten()
    if valid_ids.numel() == 0 or int(quality.shape[1]) == 0:
        return target, 0
    local = quality[valid_ids].float()
    qualified = local > float(representable_min)
    assignment_size = min(int(local.shape[0]), int(local.shape[1]))
    reward = qualified.float() * float(assignment_size + 1) + local
    rows, cols = linear_sum_assignment((-reward).numpy())
    pairs = [
        (int(valid_ids[row]), float(local[row, col]))
        for row, col in zip(rows, cols)
        if bool(qualified[row, col])
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    for candidate, _quality in pairs[: int(max_selections)]:
        target[candidate] = True
    return target, min(len(pairs), int(max_selections))


def _parallel_targets(
    records: list[dict[str, Any]],
    indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        unique_hungarian_target(
            records[index]["quality"],
            records[index]["candidate_valid"],
            representable_min=float(args.representable_min),
            max_selections=int(args.max_selections),
        )
        for index in indices
    ]
    target = torch.stack([row[0] for row in rows]).to(device)
    count = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
    return target, count


def _balanced_unique_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    raw = F.binary_cross_entropy_with_logits(
        logits,
        target.to(logits.dtype),
        reduction="none",
    )
    for row in range(int(logits.shape[0])):
        positive = valid[row] & target[row]
        negative = valid[row] & ~target[row]
        if bool(positive.any()):
            losses.append(
                0.5 * raw[row][positive].mean()
                + 0.5 * raw[row][negative].mean()
            )
        elif bool(negative.any()):
            losses.append(raw[row][negative].mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


@torch.no_grad()
def _evaluate_parallel_probe(
    probe: ParallelHungarianProbe,
    records: list[dict[str, Any]],
    indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    probe.eval()
    learned_selections: list[list[int]] = []
    oracle_count_selections: list[list[int]] = []
    top4_selections: list[list[int]] = []
    selected_records: list[dict[str, Any]] = []
    exact_count = 0
    overlap = total_positive = 0
    for start in range(0, len(indices), int(args.parallel_batch_size)):
        ids = indices[start : start + int(args.parallel_batch_size)]
        hidden = _stack(records, ids, "hidden", device)
        valid = _stack(records, ids, "candidate_valid", device).bool()
        target, target_count = _parallel_targets(records, ids, args, device)
        logits, count_logits = probe(hidden, valid)
        logits = logits.masked_fill(~valid, -1e4)
        predicted_count = count_logits.argmax(dim=-1)
        exact_count += int((predicted_count == target_count).sum().cpu())
        for row, record_index in enumerate(ids):
            valid_count = int(valid[row].sum())
            learned_k = min(int(predicted_count[row]), valid_count)
            oracle_k = min(int(target_count[row]), valid_count)
            top_k = min(int(args.max_selections), valid_count)
            order = logits[row].argsort(descending=True)
            learned = order[:learned_k].detach().cpu().tolist()
            oracle_count = order[:oracle_k].detach().cpu().tolist()
            fixed = order[:top_k].detach().cpu().tolist()
            learned_selections.append([int(value) for value in learned])
            oracle_count_selections.append(
                [int(value) for value in oracle_count]
            )
            top4_selections.append([int(value) for value in fixed])
            chosen_target = set(oracle_count) & set(
                torch.nonzero(target[row], as_tuple=False).flatten().tolist()
            )
            overlap += len(chosen_target)
            total_positive += int(target[row].sum())
            selected_records.append(records[record_index])
    return {
        "count_accuracy": exact_count / max(len(indices), 1),
        "oracle_count_target_overlap": overlap / max(total_positive, 1),
        "learned_count": selection_metrics(selected_records, learned_selections),
        "oracle_count": selection_metrics(
            selected_records,
            oracle_count_selections,
        ),
        "fixed_top4": selection_metrics(selected_records, top4_selections),
    }


def run_parallel_probe(
    records: list[dict[str, Any]],
    train_indices: list[int],
    holdout_indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    hidden_dim = int(records[0]["hidden"].shape[-1])
    probe = ParallelHungarianProbe(hidden_dim, int(args.max_selections)).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.parallel_lr),
        weight_decay=1e-4,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 211)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    cursor = 0
    milestones = sorted(
        {
            0,
            100,
            500,
            1000,
            int(args.parallel_steps),
        }
    )
    milestones = [value for value in milestones if value <= args.parallel_steps]
    trajectory: list[dict[str, Any]] = []

    def snapshot(step: int) -> None:
        trajectory.append(
            {
                "step": int(step),
                "train": _evaluate_parallel_probe(
                    probe,
                    records,
                    train_indices,
                    args,
                    device,
                ),
                "holdout": _evaluate_parallel_probe(
                    probe,
                    records,
                    holdout_indices,
                    args,
                    device,
                ),
            }
        )

    snapshot(0)
    probe.train()
    for step in tqdm(
        range(1, int(args.parallel_steps) + 1),
        desc="parallel Hungarian selector probe",
        ncols=88,
    ):
        if cursor + int(args.parallel_batch_size) > len(order):
            order = torch.randperm(len(train_indices), generator=generator).tolist()
            cursor = 0
        local = order[cursor : cursor + int(args.parallel_batch_size)]
        cursor += int(args.parallel_batch_size)
        ids = [train_indices[index] for index in local]
        hidden = _stack(records, ids, "hidden", device)
        valid = _stack(records, ids, "candidate_valid", device).bool()
        target, count_target = _parallel_targets(records, ids, args, device)
        candidate_logits, count_logits = probe(hidden, valid)
        selection_loss = _balanced_unique_bce(candidate_logits, target, valid)
        count_loss = F.cross_entropy(count_logits, count_target)
        loss = selection_loss + count_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
        optimizer.step()
        if step in milestones:
            snapshot(step)
            probe.train()

    def pointer_baseline(indices: list[int]) -> dict[str, Any]:
        subset = [records[index] for index in indices]
        selections = [
            _pointer_selections(record["pointer_indices"].unsqueeze(0))[0]
            for record in subset
        ]
        return selection_metrics(subset, selections)

    return {
        "contract": {
            "input": "same frozen relation-encoded candidate hidden as pointer",
            "candidate_target": "cardinality-first one-to-one Hungarian",
            "count_target": "jointly representable GT count, capped at four",
            "autoregression": False,
            "train_images": len(train_indices),
            "holdout_images": len(holdout_indices),
        },
        "pointer_baseline": {
            "train": pointer_baseline(train_indices),
            "holdout": pointer_baseline(holdout_indices),
        },
        "trajectory": trajectory,
    }


def _diagnosis(
    memorization: dict[str, Any],
    quality: dict[str, Any],
    parallel: dict[str, Any],
) -> dict[str, Any]:
    memory_pass = bool(memorization["passed"])
    nonlinear = quality["nonlinear_probe"]
    train_rank = nonlinear["train"]["per_gt_independent_cluster_ranking"]
    holdout_rank = nonlinear["holdout"]["per_gt_independent_cluster_ranking"]
    representation_train_fit = (
        float(train_rank["top2_rate"]) >= 0.80
        and float(train_rank["mean_regret"]) <= 0.08
    )
    representation_generalizes = (
        float(holdout_rank["top2_rate"]) >= 0.70
        and float(holdout_rank["mean_regret"]) <= 0.10
    )
    final_parallel = parallel["trajectory"][-1]["holdout"]
    pointer_f1 = float(
        parallel["pointer_baseline"]["holdout"]["iou_0.50"]["f1"]
    )
    learned_f1 = float(final_parallel["learned_count"]["iou_0.50"]["f1"])
    oracle_count_f1 = float(final_parallel["oracle_count"]["iou_0.50"]["f1"])
    parallel_gain = max(learned_f1, oracle_count_f1) - pointer_f1
    parallel_advantage = parallel_gain >= 0.02

    if not memory_pass:
        root = "intrinsic_pointer_fit_or_credit_assignment_failure"
        next_step = (
            "Do not tune long schedules. The current autoregressive pointer "
            "cannot even fit fixed descriptors and a fixed teacher; replace "
            "the objective/state transition before any full training."
        )
    elif not representation_train_fit:
        root = "frozen_descriptor_observability_failure"
        next_step = (
            "The frozen candidate hidden does not expose representative "
            "quality even to a nonlinear probe. A pointer-only fine-tune "
            "cannot solve this; restore joint query supervision or add an "
            "upstream quality representation path."
        )
    elif not representation_generalizes:
        root = "descriptor_quality_generalization_failure"
        next_step = (
            "Quality is recoverable by memorization but not on held-out "
            "images. Improve shared query supervision/data regularization; "
            "more pointer iterations alone are unlikely to help."
        )
    elif parallel_advantage:
        root = "autoregressive_pointer_objective_failure"
        next_step = (
            "The same frozen hidden works with a parallel Hungarian head. "
            "The sequential teacher/STOP policy is the bottleneck; prototype "
            "a parallel no-object set head before changing geometry."
        )
    else:
        root = "shared_representation_and_selection_bottleneck"
        next_step = (
            "Neither the autoregressive nor the simple parallel readout "
            "converts the frozen hidden cleanly. The next controlled test "
            "must train objectness and geometry jointly on the same query, "
            "with the bounded-delta stability guard retained."
        )
    return {
        "root_cause_classification": root,
        "recommended_next_experiment": next_step,
        "signals": {
            "fixed_set_memorization_pass": memory_pass,
            "nonlinear_quality_probe_train_fit": representation_train_fit,
            "nonlinear_quality_probe_holdout_generalization": representation_generalizes,
            "parallel_holdout_gain_over_pointer_f1_050": parallel_gain,
            "parallel_advantage_threshold": 0.02,
            "parallel_advantage": parallel_advantage,
        },
        "scope_warning": (
            "Probe F1 uses the detached range-aware row-strip IoU surrogate, "
            "not the official raster evaluator. A winning architecture must "
            "still be confirmed on full validation with official IoU."
        ),
    }


def main() -> None:
    args = parse_args()
    if int(args.sample_count) < int(args.memorize_images):
        raise ValueError("sample-count must be at least memorize-images")
    if int(args.max_selections) != 4:
        raise ValueError("CULane V4 root-cause audit is predeclared for four selections")
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = str(args.dataset_root)
    cfg.setdefault("training", {})["seed"] = int(args.seed)
    model = build_model(cfg).to(device)
    checkpoint_iteration = load_checkpoint(
        checkpoint_path,
        model,
        strict=False,
    )
    selector = _root_selector(model)
    selector_template = copy.deepcopy(selector).cpu()
    signature = _cache_signature(args, config_path, checkpoint_path, cfg)
    cache = _collect_frozen_cache(
        args,
        cfg,
        model,
        selector,
        signature,
        device,
    )
    records = cache["records"]
    del model, selector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_indices, holdout_indices = _record_split(
        len(records),
        int(args.holdout_stride),
    )
    memorization = run_memorization_audit(
        selector_template,
        records,
        args,
        cfg,
        device,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    quality = run_quality_probe(
        records,
        train_indices,
        holdout_indices,
        args,
        device,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    parallel = run_parallel_probe(
        records,
        train_indices,
        holdout_indices,
        args,
        device,
    )
    result = {
        "experiment": "V4 pointer root-cause audit suite",
        "provenance": {
            "git_commit": _git_commit(),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_iteration": int(checkpoint_iteration),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "split": str(args.split),
            "sample_count": len(records),
            "sample_strategy": str(args.sample_strategy),
            "sampled_dataset_indices": cache["metadata"][
                "sampled_dataset_indices"
            ],
            "seed": int(args.seed),
            "geometry_gradient_enabled": False,
            "augmentation_enabled": False,
        },
        "audit_1_fixed_64_memorization": memorization,
        "audit_2_frozen_descriptor_quality": quality,
        "audit_3_parallel_hungarian_selector": parallel,
    }
    result["decision"] = _diagnosis(memorization, quality, parallel)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
