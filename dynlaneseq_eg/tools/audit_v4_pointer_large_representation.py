from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.losses.range_aware_iou import (
    pairwise_range_aware_row_strip_iou,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.audit_v4_pointer_root_causes import (
    _auc,
    _autocast_kwargs,
    _git_commit,
    _pearson,
    _rank,
    _root_selector,
    _sha256,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.train import seed_everything


CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether V4 raw candidate descriptors or relation-encoded "
            "hidden states contain a stable, image-generalizable signal for "
            "ranking duplicate representatives of the same GT lane."
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
    parser.add_argument("--development-images", type=int, default=4096)
    parser.add_argument("--holdout-images", type=int, default=1024)
    parser.add_argument("--early-stop-images", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3407, 5419, 7823],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--linear-lr", type=float, default=3e-3)
    parser.add_argument("--nonlinear-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--nonlinear-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--representable-min", type=float, default=0.50)
    parser.add_argument("--cluster-support-min", type=float, default=0.20)
    parser.add_argument("--pair-support-delta", type=float, default=0.20)
    parser.add_argument("--pair-min-quality-gap", type=float, default=0.02)
    parser.add_argument("--quality-aux-weight", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=1907)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _resolve_list_path(cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    raw = cfg.get("dataset", {}).get("lists", {}).get(
        str(args.split),
        f"list/{args.split}.txt",
    )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(args.dataset_root).expanduser() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset list: {path}")
    return path


def _cache_signature(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint_stat = checkpoint_path.stat()
    list_path = _resolve_list_path(cfg, args)
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
        "sample_count": int(args.development_images + args.holdout_images),
        "feature_batch_size": int(args.feature_batch_size),
        "sample_strategy": str(args.sample_strategy),
        "representable_min": float(args.representable_min),
        "cluster_support_min": float(args.cluster_support_min),
        "pair_support_delta": float(args.pair_support_delta),
        "pair_min_quality_gap": float(args.pair_min_quality_gap),
    }


def build_cluster_pairs(
    quality: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    representable_min: float,
    support_min: float,
    support_delta: float,
    min_quality_gap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ordered (better, worse) pairs inside natural GT clusters."""

    quality = quality.detach().float().cpu()
    valid = candidate_valid.detach().bool().cpu()
    if quality.ndim != 2 or valid.shape != (int(quality.shape[0]),):
        raise ValueError("cluster-pair quality/validity shape mismatch")
    if int(quality.shape[1]) == 0:
        return torch.empty((0, 2), dtype=torch.long), torch.empty(0)
    owner = quality.argmax(dim=-1)
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    for gt_index in range(int(quality.shape[1])):
        q = quality[:, gt_index]
        natural = valid & (owner == gt_index)
        if not bool(natural.any()):
            continue
        best = float(q[natural].max())
        if best < float(representable_min):
            continue
        cutoff = max(float(support_min), best - float(support_delta))
        ids = torch.nonzero(
            natural & (q >= cutoff),
            as_tuple=False,
        ).flatten()
        for left_position in range(int(ids.numel())):
            for right_position in range(left_position + 1, int(ids.numel())):
                left = int(ids[left_position])
                right = int(ids[right_position])
                delta = float(q[left] - q[right])
                if abs(delta) < float(min_quality_gap):
                    continue
                better, worse = (left, right) if delta > 0.0 else (right, left)
                pairs.append((better, worse))
                # Preserve fine pairs while preventing the few easiest pairs
                # from dominating an image's complete duplicate cluster.
                weights.append(min(max(abs(delta) / 0.10, 0.20), 2.0))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long), torch.empty(0)
    return torch.tensor(pairs, dtype=torch.long), torch.tensor(
        weights,
        dtype=torch.float32,
    )


@torch.no_grad()
def _collect_cache(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    model: nn.Module,
    selector: nn.Module,
    signature: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    cache_path = Path(args.cache_path)
    if args.reuse_cache and cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if payload.get("signature") != signature:
            raise ValueError(
                "large representation cache signature mismatch; use a new "
                "cache path or remove --reuse-cache"
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
    total_images = int(args.development_images + args.holdout_images)
    max_batches = math.ceil(total_images / int(args.feature_batch_size))
    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=max_batches,
        num_workers=int(args.num_workers),
    )
    loss_cfg = cfg.get("loss", {})
    input_h = int(loss_cfg.get("input_h", cfg.get("model", {}).get("input_h", 288)))
    line_width = float(loss_cfg.get("set_selection_line_width", 30.0))
    min_valid_rows = int(loss_cfg.get("set_selection_min_valid_rows", 5))
    autocast_kwargs = _autocast_kwargs(device, args.amp_dtype)
    raw_batches: list[torch.Tensor] = []
    hidden_batches: list[torch.Tensor] = []
    unary_batches: list[torch.Tensor] = []
    valid_batches: list[torch.Tensor] = []
    qmax_batches: list[torch.Tensor] = []
    qualities: list[torch.Tensor] = []
    pairs: list[torch.Tensor] = []
    pair_weights: list[torch.Tensor] = []
    image_ids: list[str] = []
    collected = 0

    model.eval()
    for images, targets, metas in tqdm(
        loader,
        desc="large frozen representation cache",
        ncols=92,
    ):
        if collected >= total_images:
            break
        images = images.to(device, non_blocking=True)
        targets_device = nested_to_device(targets, device)
        with torch.autocast(**autocast_kwargs):
            outputs = model(images)
            raw = selector.build_selection_features(outputs)
            relations = selector.build_pairwise_relations(outputs)
            hidden = outputs.get("_selection_pointer_hidden")
            if not isinstance(hidden, torch.Tensor):
                hidden = selector.encode_selection_features(raw, relations)
        candidate_valid = selector.build_pointer_candidate_valid(outputs).bool()
        unary = outputs["selection_logits"].float()
        keep = min(int(images.shape[0]), total_images - collected)
        raw_batches.append(raw[:keep].detach().to(device="cpu", dtype=torch.float16))
        hidden_batches.append(
            hidden[:keep].detach().to(device="cpu", dtype=torch.float16)
        )
        unary_batches.append(
            unary[:keep].detach().to(device="cpu", dtype=torch.float16)
        )
        valid_rows: list[torch.Tensor] = []
        qmax_rows: list[torch.Tensor] = []
        for local_index in range(keep):
            target = targets_device[local_index]
            quality, quality_valid, valid_gt = pairwise_range_aware_row_strip_iou(
                outputs["pred_x_rows"][local_index].detach().float(),
                outputs["range_norm"][local_index].detach().float(),
                target["x_rows"].to(device=device).float(),
                target["valid_mask"].to(device=device).bool(),
                input_h=input_h,
                line_width=line_width,
                min_valid_rows=min_valid_rows,
            )
            quality = quality[:, valid_gt].detach().float().cpu()
            valid = (
                candidate_valid[local_index].detach().cpu() & quality_valid.cpu()
            )
            qmax = (
                quality.amax(dim=-1)
                if int(quality.shape[1]) > 0
                else torch.zeros(int(quality.shape[0]))
            )
            cluster_pairs, cluster_weights = build_cluster_pairs(
                quality,
                valid,
                representable_min=float(args.representable_min),
                support_min=float(args.cluster_support_min),
                support_delta=float(args.pair_support_delta),
                min_quality_gap=float(args.pair_min_quality_gap),
            )
            valid_rows.append(valid)
            qmax_rows.append(qmax)
            qualities.append(quality.to(dtype=torch.float16))
            pairs.append(cluster_pairs)
            pair_weights.append(cluster_weights)
            image_ids.append(str(metas[local_index].get("image_path", collected)))
            collected += 1
        valid_batches.append(torch.stack(valid_rows))
        qmax_batches.append(torch.stack(qmax_rows).to(dtype=torch.float16))

    if collected != total_images:
        raise RuntimeError(
            f"requested {total_images} images but collected {collected}"
        )
    payload = {
        "signature": signature,
        "metadata": {
            "sampled_dataset_indices": sampled_indices[:collected],
            "image_ids": image_ids,
            "augmentation_enabled": False,
            "input_h": input_h,
            "line_width": line_width,
            "min_valid_rows": min_valid_rows,
            "raw_dtype": "float16",
            "hidden_dtype": "float16",
        },
        "raw_descriptor": torch.cat(raw_batches, dim=0),
        "relation_hidden": torch.cat(hidden_batches, dim=0),
        "checkpoint_unary": torch.cat(unary_batches, dim=0),
        "candidate_valid": torch.cat(valid_batches, dim=0),
        "quality_max": torch.cat(qmax_batches, dim=0),
        "quality": qualities,
        "pairs": pairs,
        "pair_weights": pair_weights,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def make_image_disjoint_splits(
    total_images: int,
    development_images: int,
    holdout_images: int,
    early_stop_images: int,
    split_seed: int,
) -> dict[str, list[int]]:
    if total_images != development_images + holdout_images:
        raise ValueError("development/holdout counts do not cover the cache")
    if not 0 < early_stop_images < development_images:
        raise ValueError("early-stop count must lie inside development count")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(split_seed))
    order = torch.randperm(total_images, generator=generator).tolist()
    holdout = order[:holdout_images]
    development = order[holdout_images:]
    early_stop = development[:early_stop_images]
    fit = development[early_stop_images:]
    if set(fit) & set(early_stop) or set(development) & set(holdout):
        raise RuntimeError("large representation split leaked image identities")
    return {
        "development": development,
        "fit": fit,
        "early_stop": early_stop,
        "holdout": holdout,
    }


def _feature_moments(
    features: torch.Tensor,
    valid: torch.Tensor,
    indices: list[int],
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = int(features.shape[-1])
    total = torch.zeros(dimension, dtype=torch.float64)
    total_square = torch.zeros(dimension, dtype=torch.float64)
    count = 0
    for start in range(0, len(indices), chunk_size):
        ids = indices[start : start + chunk_size]
        values = features[ids].float()
        selected = values[valid[ids].bool()]
        if selected.numel() == 0:
            continue
        total += selected.double().sum(dim=0)
        total_square += selected.double().square().sum(dim=0)
        count += int(selected.shape[0])
    if count < 2:
        raise ValueError("not enough valid candidates to normalize probe inputs")
    mean = total / float(count)
    variance = total_square / float(count) - mean.square()
    std = variance.clamp_min(1e-6).sqrt()
    return mean.float(), std.float()


class NormalizedQualityProbe(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        nonlinear: bool,
        nonlinear_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-3))
        dimension = int(mean.numel())
        if nonlinear:
            self.readout = nn.Sequential(
                nn.Linear(dimension, int(nonlinear_hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(nonlinear_hidden), 1),
            )
        else:
            self.readout = nn.Linear(dimension, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.mean) / self.std
        return self.readout(normalized).squeeze(-1)


def _pairwise_loss(
    scores: torch.Tensor,
    batch_indices: list[int],
    cache: dict[str, Any],
) -> torch.Tensor:
    image_losses: list[torch.Tensor] = []
    for row, record_index in enumerate(batch_indices):
        pairs = cache["pairs"][record_index]
        if int(pairs.shape[0]) == 0:
            continue
        pair_ids = pairs.to(device=scores.device)
        weights = cache["pair_weights"][record_index].to(
            device=scores.device,
            dtype=scores.dtype,
        )
        margin = scores[row, pair_ids[:, 0]] - scores[row, pair_ids[:, 1]]
        image_losses.append((F.softplus(-margin) * weights).mean())
    return (
        torch.stack(image_losses).mean()
        if image_losses
        else scores.sum() * 0.0
    )


def probe_training_loss(
    scores: torch.Tensor,
    batch_indices: list[int],
    cache: dict[str, Any],
    *,
    quality_aux_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairwise = _pairwise_loss(scores, batch_indices, cache)
    valid = cache["candidate_valid"][batch_indices].to(scores.device).bool()
    quality = cache["quality_max"][batch_indices].to(
        device=scores.device,
        dtype=scores.dtype,
    )
    auxiliary = F.binary_cross_entropy_with_logits(scores[valid], quality[valid])
    total = pairwise + float(quality_aux_weight) * auxiliary
    return total, pairwise, auxiliary


@torch.no_grad()
def _score_probe(
    probe: nn.Module,
    features: torch.Tensor,
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    probe.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start : start + batch_size]
        rows.append(probe(features[ids].to(device)).detach().float().cpu())
    return torch.cat(rows, dim=0)


def _cluster_metrics(
    cache: dict[str, Any],
    indices: list[int],
    scores: torch.Tensor,
    representable_min: float,
) -> dict[str, Any]:
    ranks: list[int] = []
    regrets: list[float] = []
    pair_correct = pair_total = 0
    weighted_correct = weighted_total = 0.0
    for row, record_index in enumerate(indices):
        quality = cache["quality"][record_index].float()
        valid = cache["candidate_valid"][record_index].bool()
        if int(quality.shape[1]) > 0:
            owner = quality.argmax(dim=-1)
            for gt_index in range(int(quality.shape[1])):
                q = quality[:, gt_index]
                support = valid & (owner == gt_index) & (q > 0.0)
                if not bool(support.any()):
                    continue
                best = float(q[support].max())
                if best < float(representable_min):
                    continue
                ids = torch.nonzero(support, as_tuple=False).flatten()
                selected = int(ids[scores[row, ids].argmax()])
                selected_quality = float(q[selected])
                rank = 1 + int((q[ids] > selected_quality + 1e-7).sum())
                ranks.append(rank)
                regrets.append(best - selected_quality)
        pairs = cache["pairs"][record_index]
        if int(pairs.shape[0]) > 0:
            better = scores[row, pairs[:, 0]]
            worse = scores[row, pairs[:, 1]]
            correct = better > worse
            weights = cache["pair_weights"][record_index].float()
            pair_correct += int(correct.sum())
            pair_total += int(correct.numel())
            weighted_correct += float((correct.float() * weights).sum())
            weighted_total += float(weights.sum())
    if ranks:
        regret = torch.tensor(regrets)
        top1 = sum(value == 1 for value in ranks) / len(ranks)
        top2 = sum(value <= 2 for value in ranks) / len(ranks)
        mean_regret = sum(regrets) / len(regrets)
        p90 = float(torch.quantile(regret, 0.90))
        mean_rank = sum(ranks) / len(ranks)
    else:
        top1 = top2 = mean_regret = p90 = mean_rank = 0.0
    return {
        "representable_gt": len(ranks),
        "top1_rate": top1,
        "top2_rate": top2,
        "mean_rank": mean_rank,
        "mean_regret": mean_regret,
        "p90_regret": p90,
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
        "weighted_pairwise_accuracy": weighted_correct
        / max(weighted_total, 1e-12),
        "pair_count": pair_total,
        "selection_score": top1 + 0.5 * top2 - mean_regret,
    }


def evaluate_scores(
    cache: dict[str, Any],
    indices: list[int],
    scores: torch.Tensor,
    representable_min: float,
) -> dict[str, Any]:
    if scores.shape[:2] != (
        len(indices),
        int(cache["candidate_valid"].shape[1]),
    ):
        raise ValueError("probe score shape mismatch")
    valid = cache["candidate_valid"][indices].bool()
    quality = cache["quality_max"][indices].float()
    flat_score = scores[valid]
    flat_quality = quality[valid]
    return {
        "images": len(indices),
        "valid_candidates": int(flat_quality.numel()),
        "candidate_quality": {
            "auc_ge_050": _auc(flat_score, flat_quality >= 0.50),
            "pearson": _pearson(flat_score, flat_quality),
            "spearman": _pearson(_rank(flat_score), _rank(flat_quality)),
            "mae_after_sigmoid": float(
                (torch.sigmoid(flat_score) - flat_quality).abs().mean()
            ),
        },
        "cluster_ranking": _cluster_metrics(
            cache,
            indices,
            scores,
            representable_min,
        ),
    }


def _state_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def train_probe(
    cache: dict[str, Any],
    source: str,
    nonlinear: bool,
    splits: dict[str, list[int]],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    moments: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    seed_everything(int(seed))
    features = cache[source]
    mean, std = moments
    probe = NormalizedQualityProbe(
        mean,
        std,
        nonlinear=nonlinear,
        nonlinear_hidden=int(args.nonlinear_hidden),
        dropout=float(args.dropout),
    ).to(device)
    learning_rate = (
        float(args.nonlinear_lr) if nonlinear else float(args.linear_lr)
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=float(args.weight_decay) if nonlinear else 1e-4,
    )
    fit = splits["fit"]
    early = splits["early_stop"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 37)
    best_score = -float("inf")
    best_epoch = 0
    best_state = _state_to_cpu(probe)
    stale_epochs = 0
    trajectory: list[dict[str, Any]] = []

    for epoch in range(1, int(args.max_epochs) + 1):
        order = torch.randperm(len(fit), generator=generator).tolist()
        probe.train()
        total_loss = pair_loss = quality_loss = 0.0
        batches = 0
        for start in range(0, len(order), int(args.batch_size)):
            local = order[start : start + int(args.batch_size)]
            ids = [fit[position] for position in local]
            values = features[ids].to(device)
            scores = probe(values)
            loss, pairwise, auxiliary = probe_training_loss(
                scores,
                ids,
                cache,
                quality_aux_weight=float(args.quality_aux_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            pair_loss += float(pairwise.detach().cpu())
            quality_loss += float(auxiliary.detach().cpu())
            batches += 1
        early_scores = _score_probe(
            probe,
            features,
            early,
            device,
            int(args.batch_size),
        )
        early_metrics = evaluate_scores(
            cache,
            early,
            early_scores,
            float(args.representable_min),
        )
        selection_score = float(
            early_metrics["cluster_ranking"]["selection_score"]
        )
        trajectory.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(batches, 1),
                "train_pairwise_loss": pair_loss / max(batches, 1),
                "train_quality_aux_loss": quality_loss / max(batches, 1),
                "early_stop": early_metrics,
            }
        )
        if selection_score > best_score + 1e-4:
            best_score = selection_score
            best_epoch = epoch
            best_state = _state_to_cpu(probe)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(args.patience):
            break

    probe.load_state_dict(best_state)
    result: dict[str, Any] = {
        "seed": int(seed),
        "source": source,
        "architecture": "nonlinear" if nonlinear else "linear",
        "best_epoch": int(best_epoch),
        "epochs_run": len(trajectory),
        "early_stop_best_selection_score": float(best_score),
        "trajectory": trajectory,
    }
    for split_name in ("fit", "early_stop", "holdout"):
        indices = splits[split_name]
        score = _score_probe(
            probe,
            features,
            indices,
            device,
            int(args.batch_size),
        )
        result[split_name] = evaluate_scores(
            cache,
            indices,
            score,
            float(args.representable_min),
        )
    return result


def _metric(run: dict[str, Any], split: str, name: str) -> float:
    return float(run[split]["cluster_ranking"][name])


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"seeds": [int(run["seed"]) for run in runs]}
    for split in ("fit", "early_stop", "holdout"):
        rows: dict[str, Any] = {}
        for metric in (
            "top1_rate",
            "top2_rate",
            "mean_regret",
            "p90_regret",
            "pairwise_accuracy",
            "weighted_pairwise_accuracy",
            "selection_score",
        ):
            values = [_metric(run, split, metric) for run in runs]
            rows[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
                "values": values,
            }
        summary[split] = rows
    summary["best_epochs"] = [int(run["best_epoch"]) for run in runs]
    return summary


def _split_digest(indices: list[int]) -> str:
    payload = ",".join(str(value) for value in sorted(indices)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mean(summary: dict[str, Any], split: str, metric: str) -> float:
    return float(summary[split][metric]["mean"])


def make_decision(
    baseline: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    thresholds = {
        "holdout_top1_rate": 0.40,
        "holdout_top2_rate": 0.70,
        "holdout_mean_regret": 0.10,
        "holdout_weighted_pairwise_accuracy": 0.70,
        "max_top1_generalization_gap": 0.15,
    }

    def passes(summary: dict[str, Any]) -> bool:
        gap = _mean(summary, "fit", "top1_rate") - _mean(
            summary,
            "holdout",
            "top1_rate",
        )
        return (
            _mean(summary, "holdout", "top1_rate")
            >= thresholds["holdout_top1_rate"]
            and _mean(summary, "holdout", "top2_rate")
            >= thresholds["holdout_top2_rate"]
            and _mean(summary, "holdout", "mean_regret")
            <= thresholds["holdout_mean_regret"]
            and _mean(summary, "holdout", "weighted_pairwise_accuracy")
            >= thresholds["holdout_weighted_pairwise_accuracy"]
            and gap <= thresholds["max_top1_generalization_gap"]
        )

    raw = summaries["raw_descriptor_nonlinear"]
    hidden = summaries["relation_hidden_nonlinear"]
    raw_pass = passes(raw)
    hidden_pass = passes(hidden)
    hidden_fit = _mean(hidden, "fit", "top1_rate")
    hidden_holdout = _mean(hidden, "holdout", "top1_rate")
    if hidden_pass:
        root = "pointer_objective_or_rollout_bottleneck"
        action = (
            "The relation-hidden representation generalizes representative "
            "quality. Keep geometry frozen and replace the pointer objective "
            "with the simplest full-data selector that exploits this signal."
        )
    elif raw_pass:
        root = "relation_encoder_erases_representative_quality"
        action = (
            "Raw descriptors generalize but relation-hidden states do not. "
            "Bypass or redesign the candidate relation encoder before any "
            "long pointer schedule."
        )
    elif hidden_fit >= 0.60 and hidden_fit - hidden_holdout > 0.20:
        root = "fine_representative_signal_is_not_image_invariant"
        action = (
            "The frozen representation supports memorization but not stable "
            "winner prediction. Train query ownership/objectness jointly with "
            "geometry from the start; more post-hoc pointer fine-tuning is "
            "not authorized."
        )
    else:
        root = "fine_representative_signal_is_not_observable"
        action = (
            "Even the large probe cannot reliably recover duplicate winner "
            "quality. Add joint one-to-one query ownership supervision upstream "
            "while retaining bounded-delta coordinate stability."
        )
    return {
        "root_cause_classification": root,
        "recommended_next_experiment": action,
        "predeclared_pass_thresholds": thresholds,
        "signals": {
            "raw_descriptor_nonlinear_pass": raw_pass,
            "relation_hidden_nonlinear_pass": hidden_pass,
            "checkpoint_unary_holdout_top1": float(
                baseline["holdout"]["cluster_ranking"]["top1_rate"]
            ),
            "raw_nonlinear_holdout_top1_mean": _mean(
                raw,
                "holdout",
                "top1_rate",
            ),
            "hidden_nonlinear_fit_top1_mean": hidden_fit,
            "hidden_nonlinear_holdout_top1_mean": hidden_holdout,
            "hidden_nonlinear_top1_generalization_gap": hidden_fit
            - hidden_holdout,
        },
        "scope_warning": (
            "All quality and ranking targets use detached range-aware row-strip "
            "IoU. This audit diagnoses representation; it is not an official "
            "CULane validation or test score."
        ),
    }


def main() -> None:
    args = parse_args()
    total_images = int(args.development_images + args.holdout_images)
    if total_images < 4:
        raise ValueError("large representation audit needs at least four images")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("probe seeds must be unique")
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = str(args.dataset_root)
    device = torch.device(args.device)
    seed_everything(int(args.split_seed))
    model = build_model(cfg).to(device)
    checkpoint_iteration = load_checkpoint(checkpoint_path, model, strict=False)
    selector = _root_selector(model)
    signature = _cache_signature(args, cfg, config_path, checkpoint_path)
    cache = _collect_cache(
        args,
        cfg,
        model,
        selector,
        signature,
        device,
    )
    del model, selector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    splits = make_image_disjoint_splits(
        total_images,
        int(args.development_images),
        int(args.holdout_images),
        int(args.early_stop_images),
        int(args.split_seed),
    )
    baseline: dict[str, Any] = {}
    for split_name in ("fit", "early_stop", "holdout"):
        ids = splits[split_name]
        baseline[split_name] = evaluate_scores(
            cache,
            ids,
            cache["checkpoint_unary"][ids].float(),
            float(args.representable_min),
        )

    runs: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for source in ("raw_descriptor", "relation_hidden"):
        moments = _feature_moments(
            cache[source],
            cache["candidate_valid"],
            splits["fit"],
        )
        for nonlinear in (False, True):
            architecture = "nonlinear" if nonlinear else "linear"
            key = f"{source}_{architecture}"
            print(f"probe arm: {key}")
            arm_runs = [
                train_probe(
                    cache,
                    source,
                    nonlinear,
                    splits,
                    seed,
                    args,
                    device,
                    moments,
                )
                for seed in args.seeds
            ]
            runs[key] = arm_runs
            summaries[key] = summarize_runs(arm_runs)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = {
        "experiment": "V4 large-scale frozen representative-quality probe",
        "provenance": {
            "git_commit": _git_commit(),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_iteration": int(checkpoint_iteration),
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "split": str(args.split),
            "sample_strategy": str(args.sample_strategy),
            "augmentation_enabled": False,
            "geometry_gradient_enabled": False,
            "cache_path": str(Path(args.cache_path).expanduser().resolve()),
        },
        "contract": {
            "development_images": int(args.development_images),
            "fit_images": len(splits["fit"]),
            "early_stop_images": len(splits["early_stop"]),
            "holdout_images": len(splits["holdout"]),
            "image_disjoint": True,
            "split_seed": int(args.split_seed),
            "probe_seeds": [int(value) for value in args.seeds],
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "pairwise_target": {
                "representable_min": float(args.representable_min),
                "cluster_support_min": float(args.cluster_support_min),
                "support_delta_from_best": float(args.pair_support_delta),
                "minimum_quality_gap": float(args.pair_min_quality_gap),
                "quality_aux_weight": float(args.quality_aux_weight),
            },
            "split_digests": {
                name: _split_digest(indices)
                for name, indices in splits.items()
            },
        },
        "checkpoint_unary_baseline": baseline,
        "runs": runs,
        "summaries": summaries,
    }
    result["decision"] = make_decision(baseline, summaries)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
