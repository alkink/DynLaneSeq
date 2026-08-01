from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    evaluator_hungarian_assignment,
)
from dynlaneseq_eg.factory import build_model
from dynlaneseq_eg.modeling.common import sort_range_norm
from dynlaneseq_eg.modeling.structured_queries import SetAwareLaneSelectionHead
from dynlaneseq_eg.tools.probe_frozen_unified_selector import _evaluate
from dynlaneseq_eg.tools.probe_official_set_selection import (
    training_index_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the explicit-coverage hypothesis on frozen unified-selector "
            "features. A four-slot pointer is trained with duplicate-tolerant "
            "GT targets and compared with scalar Top-K, NMS, MMR, and oracle."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--target-iou-band", type=float, default=0.10)
    parser.add_argument("--target-min-iou", type=float, default=0.30)
    parser.add_argument("--presence-weight", type=float, default=0.25)
    parser.add_argument("--diversity-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--input-h", type=int, default=640)
    parser.add_argument("--input-w", type=int, default=1600)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument(
        "--mmr-penalties",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
    )
    parser.add_argument(
        "--mmr-sigmas",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 30.0],
    )
    parser.add_argument("--min-gain-050-points", type=float, default=5.0)
    parser.add_argument("--min-gain-070-points", type=float, default=3.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-probe", required=True)
    return parser.parse_args()


class FourSlotCoverageProbe(nn.Module):
    """Four image-conditioned pointer slots over one frozen candidate set."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        num_slots: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_slots = int(num_slots)
        self.scale = float(hidden_dim) ** -0.5
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.candidate_projection = nn.Linear(int(input_dim), int(hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(num_layers)),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=max(1, int(num_layers)),
        )
        self.slot_embedding = nn.Parameter(
            torch.randn(self.num_slots, int(hidden_dim)) * 0.02
        )
        self.candidate_key = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.slot_query = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.presence = nn.Linear(int(hidden_dim), 1)

    def forward(
        self,
        features: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        invalid = ~candidate_valid.bool()
        candidates = self.candidate_projection(self.input_norm(features))
        candidates = self.candidate_encoder(
            candidates,
            src_key_padding_mask=invalid,
        )
        slots = self.slot_embedding.unsqueeze(0).expand(
            int(features.shape[0]),
            -1,
            -1,
        )
        slots = self.slot_decoder(
            slots,
            candidates,
            memory_key_padding_mask=invalid,
        )
        query = self.slot_query(slots)
        key = self.candidate_key(candidates)
        logits = torch.einsum("bsh,bnh->bsn", query, key) * self.scale
        logits = logits.masked_fill(invalid.unsqueeze(1), -1e4)
        presence_logits = self.presence(slots).squeeze(-1)
        return logits.float(), presence_logits.float()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_cache(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_cache(cache: dict[str, Any], name: str) -> None:
    required = (
        "features",
        "candidate_valid",
        "official_iou",
        "stage",
    )
    missing = [key for key in required if key not in cache]
    if missing:
        raise ValueError(f"{name} cache is missing fields: {missing}")
    images, candidates, _feature_dim = cache["features"].shape
    if tuple(cache["candidate_valid"].shape) != (images, candidates):
        raise ValueError(f"{name} candidate_valid shape mismatch")
    if len(cache["official_iou"]) != int(images):
        raise ValueError(f"{name} official_iou length mismatch")
    metadata = cache.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{name} cache has no metadata")
    image_paths = metadata.get("image_paths")
    if not isinstance(image_paths, list) or len(image_paths) != int(images):
        raise ValueError(f"{name} cache image_paths mismatch")


def _load_source_selector(
    config_path: str,
    checkpoint_path: str,
) -> tuple[SetAwareLaneSelectionHead, int]:
    cfg = load_config(config_path)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    model = build_model(cfg)
    iteration = load_checkpoint(checkpoint_path, model, strict=False)
    if model.structured_query_head is None:
        raise ValueError("source checkpoint has no structured query head")
    selector = model.structured_query_head.set_selection_head
    if selector is None or not selector.unified_score:
        raise ValueError("source checkpoint has no unified selector")
    return deepcopy(selector).float().cpu(), int(iteration)


def _estimate_gt_x(
    quality: torch.Tensor,
    pred_x: torch.Tensor,
    ranges: torch.Tensor,
) -> torch.Tensor:
    gt_count = int(quality.shape[0])
    if gt_count == 0:
        return quality.new_zeros((0,))
    best_candidate = quality.argmax(dim=-1)
    sorted_ranges = sort_range_norm(ranges.float())
    rows = int(pred_x.shape[-1])
    y_norm = torch.linspace(0.0, 1.0, rows)
    positions: list[torch.Tensor] = []
    for gt_index in range(gt_count):
        candidate = int(best_candidate[gt_index])
        visible = (
            (y_norm >= sorted_ranges[candidate, 0])
            & (y_norm <= sorted_ranges[candidate, 1])
            & torch.isfinite(pred_x[candidate])
        )
        ids = torch.nonzero(visible, as_tuple=False).flatten()
        if ids.numel() == 0:
            positions.append(pred_x[candidate].nan_to_num().median())
            continue
        tail = ids[-min(10, int(ids.numel())) :]
        positions.append(pred_x[candidate, tail].median())
    return torch.stack(positions)


def build_soft_slot_targets(
    cache: dict[str, Any],
    *,
    num_slots: int,
    temperature: float,
    iou_band: float,
    min_iou: float,
) -> dict[str, torch.Tensor]:
    images, candidates, _ = cache["features"].shape
    targets = torch.zeros((images, num_slots, candidates), dtype=torch.float32)
    active = torch.zeros((images, num_slots), dtype=torch.bool)
    affinity = torch.zeros((images, candidates, candidates), dtype=torch.float32)
    for image_index, quality_value in enumerate(cache["official_iou"]):
        quality = quality_value.float()
        valid = cache["candidate_valid"][image_index].bool()
        gt_count = int(quality.shape[0])
        if gt_count == 0:
            affinity[image_index].fill_diagonal_(1.0)
            continue
        pred_x = cache["stage"]["pred_x_rows"][image_index].float()
        ranges = cache["stage"]["range_norm"][image_index].float()
        gt_x = _estimate_gt_x(quality, pred_x, ranges)
        max_quality = quality.max(dim=-1).values
        if gt_count > int(num_slots):
            chosen = torch.topk(max_quality, k=int(num_slots)).indices
        else:
            chosen = torch.arange(gt_count)
        chosen = chosen[torch.argsort(gt_x[chosen])]
        for slot_index, gt_index_value in enumerate(chosen.tolist()):
            row = quality[int(gt_index_value)].clone()
            maximum = float(row.max())
            threshold = max(float(min_iou), maximum - float(iou_band))
            acceptable = (row >= threshold) & valid
            if not bool(acceptable.any()):
                acceptable[row.argmax()] = True
            weights = torch.zeros_like(row)
            weights[acceptable] = torch.exp(
                (row[acceptable] - maximum) / max(float(temperature), 1e-4)
            )
            targets[image_index, slot_index] = weights / weights.sum().clamp_min(1e-8)
            active[image_index, slot_index] = True
        # Candidates are duplicates when they both explain the same GT lane.
        pair = torch.sqrt(
            quality.clamp_min(0.0).unsqueeze(-1)
            * quality.clamp_min(0.0).unsqueeze(-2)
        ).amax(dim=0)
        pair = pair * valid.float().unsqueeze(0) * valid.float().unsqueeze(1)
        pair.fill_diagonal_(1.0)
        affinity[image_index] = pair
    return {"targets": targets, "active": active, "affinity": affinity}


def coverage_pointer_loss(
    logits: torch.Tensor,
    presence_logits: torch.Tensor,
    targets: torch.Tensor,
    active: torch.Tensor,
    affinity: torch.Tensor,
    *,
    presence_weight: float,
    diversity_weight: float,
) -> dict[str, torch.Tensor]:
    log_probability = F.log_softmax(logits, dim=-1)
    per_slot = -(targets.to(logits.dtype) * log_probability).sum(dim=-1)
    active_float = active.to(logits.dtype)
    pointer = (per_slot * active_float).sum() / active_float.sum().clamp_min(1.0)
    presence = F.binary_cross_entropy_with_logits(
        presence_logits,
        active_float,
    )
    probability = torch.softmax(logits, dim=-1)
    expected_affinity = torch.einsum(
        "bsn,bnm,btm->bst",
        probability,
        affinity.to(probability.dtype),
        probability,
    )
    slot_pair_valid = active_float.unsqueeze(-1) * active_float.unsqueeze(-2)
    upper = torch.triu(
        torch.ones_like(expected_affinity),
        diagonal=1,
    )
    pair_weight = slot_pair_valid * upper
    diversity = (expected_affinity * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)
    total = pointer + float(presence_weight) * presence + float(diversity_weight) * diversity
    return {
        "total": total,
        "pointer": pointer,
        "presence": presence,
        "diversity": diversity,
    }


def _new_counts() -> dict[str, Any]:
    return {"gt": 0, "selected": 0, "hits": {0.5: 0, 0.7: 0}}


def _update_counts(
    counts: dict[str, Any],
    official_iou: torch.Tensor,
    selected_ids: Iterable[int],
) -> None:
    selected = [int(value) for value in selected_ids]
    counts["gt"] += int(official_iou.shape[0])
    counts["selected"] += len(selected)
    for threshold in (0.5, 0.7):
        counts["hits"][threshold] += int(
            evaluator_hungarian_assignment(
                official_iou,
                selected,
                threshold=threshold,
            ).hit_count
        )


def _finish_counts(counts: dict[str, Any]) -> dict[str, Any]:
    gt = int(counts["gt"])
    selected = int(counts["selected"])
    result: dict[str, Any] = {
        "gt_lanes": gt,
        "selected_predictions": selected,
    }
    for threshold in (0.5, 0.7):
        suffix = f"{int(100 * threshold):03d}"
        tp = int(counts["hits"][threshold])
        precision = tp / float(max(selected, 1))
        recall = tp / float(max(gt, 1))
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        result[f"tp_{suffix}"] = tp
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"f1_{suffix}"] = f1
    return result


def _unique_pointer_ids(
    logits: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> list[int]:
    valid_ids = torch.nonzero(candidate_valid.bool(), as_tuple=False).flatten()
    if valid_ids.numel() == 0:
        return []
    matrix = logits[:, valid_ids].detach().float().cpu().numpy()
    slot_ids, local_candidate_ids = linear_sum_assignment(-matrix)
    pairs = sorted(
        zip(slot_ids.tolist(), local_candidate_ids.tolist()),
        key=lambda pair: int(pair[0]),
    )
    return [int(valid_ids[int(local_id)]) for _, local_id in pairs]


def _fill_with_source_scores(
    selected_ids: Iterable[int],
    source_score: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    top_k: int,
) -> list[int]:
    selected = [int(value) for value in selected_ids]
    selected_set = set(selected)
    fallback = [
        index
        for index in range(int(source_score.shape[0]))
        if bool(candidate_valid[index]) and index not in selected_set
    ]
    fallback.sort(key=lambda index: float(source_score[index]), reverse=True)
    selected.extend(fallback[: max(0, int(top_k) - len(selected))])
    return selected[: int(top_k)]


@torch.no_grad()
def evaluate_pointer(
    model: FourSlotCoverageProbe,
    cache: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    fallback_scores: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    counts = _new_counts()
    all_slots_counts = _new_counts()
    all_logits: list[torch.Tensor] = []
    all_presence: list[torch.Tensor] = []
    features = cache["features"]
    valid = cache["candidate_valid"]
    for start in range(0, int(features.shape[0]), int(batch_size)):
        batch_features = features[start : start + int(batch_size)].to(
            device=device,
            dtype=torch.float32,
        )
        batch_valid = valid[start : start + int(batch_size)].to(device)
        logits, presence = model(batch_features, batch_valid)
        all_logits.append(logits.cpu())
        all_presence.append(presence.cpu())
    logits = torch.cat(all_logits)
    presence = torch.cat(all_presence)
    for image_index, official_iou in enumerate(cache["official_iou"]):
        all_slots_selected = _unique_pointer_ids(
            logits[image_index],
            valid[image_index],
        )
        active_slot_ids = torch.nonzero(
            torch.sigmoid(presence[image_index]) >= 0.5,
            as_tuple=False,
        ).flatten()
        if active_slot_ids.numel() > 0:
            selected = _unique_pointer_ids(
                logits[image_index, active_slot_ids],
                valid[image_index],
            )
        else:
            selected = []
        selected = _fill_with_source_scores(
            selected,
            fallback_scores[image_index],
            valid[image_index],
            top_k=model.num_slots,
        )
        _update_counts(counts, official_iou, selected)
        _update_counts(all_slots_counts, official_iou, all_slots_selected)
    result = _finish_counts(counts)
    result["selection_mode"] = "presence_slots_then_source_fill"
    result["all_slots_unique"] = _finish_counts(all_slots_counts)
    result["mean_slot_presence_probability"] = (
        torch.sigmoid(presence).mean(dim=0).tolist()
    )
    return result


@torch.no_grad()
def source_scores(
    selector: SetAwareLaneSelectionHead,
    cache: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    selector = selector.to(device).eval()
    rows: list[torch.Tensor] = []
    features = cache["features"]
    for start in range(0, int(features.shape[0]), int(batch_size)):
        value = features[start : start + int(batch_size)].to(
            device=device,
            dtype=torch.float32,
        )
        rows.append(selector.score_selection_features(value).cpu())
    return torch.sigmoid(torch.cat(rows))


def pairwise_curve_distance(cache: dict[str, Any]) -> torch.Tensor:
    pred_x = cache["stage"]["pred_x_rows"].float()
    ranges = sort_range_norm(cache["stage"]["range_norm"].float())
    images, candidates, rows = pred_x.shape
    y_norm = torch.linspace(0.0, 1.0, int(rows)).view(1, 1, rows)
    visible = (
        (y_norm >= ranges[..., :1])
        & (y_norm <= ranges[..., 1:])
        & torch.isfinite(pred_x)
    )
    output = torch.full((images, candidates, candidates), 1e4)
    for image_index in range(int(images)):
        overlap = visible[image_index].unsqueeze(1) & visible[image_index].unsqueeze(0)
        count = overlap.sum(dim=-1)
        distance = (
            (pred_x[image_index].unsqueeze(1) - pred_x[image_index].unsqueeze(0)).abs()
            * overlap.float()
        ).sum(dim=-1) / count.clamp_min(1).float()
        distance = torch.where(count >= 5, distance, torch.full_like(distance, 1e4))
        output[image_index] = distance
    return output


def _mmr_ids(
    scores: torch.Tensor,
    distance: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    penalty: float,
    sigma: float,
    top_k: int = 4,
) -> list[int]:
    available = candidate_valid.bool().clone()
    selected: list[int] = []
    similarity = torch.exp(-distance / max(float(sigma), 1e-4))
    for _ in range(min(int(top_k), int(available.sum()))):
        utility = scores.clone()
        if selected:
            redundancy = similarity[:, selected].amax(dim=-1)
            utility = utility - float(penalty) * redundancy
        utility = utility.masked_fill(~available, float("-inf"))
        index = int(utility.argmax())
        selected.append(index)
        available[index] = False
    return selected


def evaluate_mmr(
    cache: dict[str, Any],
    scores: torch.Tensor,
    distance: torch.Tensor,
    *,
    penalty: float,
    sigma: float,
) -> dict[str, Any]:
    counts = _new_counts()
    for image_index, official_iou in enumerate(cache["official_iou"]):
        ids = _mmr_ids(
            scores[image_index],
            distance[image_index],
            cache["candidate_valid"][image_index],
            penalty=penalty,
            sigma=sigma,
        )
        _update_counts(counts, official_iou, ids)
    return _finish_counts(counts)


def _objective(metrics: dict[str, Any]) -> float:
    return float(metrics["recall_050"]) + float(metrics["recall_070"])


def coverage_gate(
    pointer_metrics: dict[str, Any],
    mmr_metrics: dict[str, Any],
    source_raw: dict[str, Any],
    *,
    min_gain_050_points: float,
    min_gain_070_points: float,
) -> dict[str, Any]:
    def gains(metrics: dict[str, Any]) -> dict[str, float]:
        return {
            "gain_recall_050_points": 100.0
            * (float(metrics["recall_050"]) - float(source_raw["recall_050"])),
            "gain_recall_070_points": 100.0
            * (float(metrics["recall_070"]) - float(source_raw["recall_070"])),
        }

    pointer_gain = gains(pointer_metrics)
    mmr_gain = gains(mmr_metrics)

    def positive(gain: dict[str, float]) -> bool:
        return bool(
            gain["gain_recall_050_points"] >= float(min_gain_050_points)
            and gain["gain_recall_070_points"] >= float(min_gain_070_points)
        )

    pointer_positive = positive(pointer_gain)
    mmr_positive = positive(mmr_gain)
    if pointer_positive:
        interpretation = "learned_explicit_coverage_is_supported"
    elif mmr_positive:
        interpretation = (
            "geometry_conditioned_coverage_works_but_pointer_probe_does_not"
        )
    else:
        interpretation = "explicit_coverage_hypothesis_not_supported_by_frozen_probe"
    return {
        "pointer_gain": pointer_gain,
        "mmr_gain": mmr_gain,
        "pointer_positive": pointer_positive,
        "mmr_positive": mmr_positive,
        "four_slot_hypothesis_positive": pointer_positive,
        "any_diversity_signal_positive": bool(pointer_positive or mmr_positive),
        "interpretation": interpretation,
    }


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    if int(args.num_slots) != 4:
        raise ValueError("this diagnostic is matched to the deployed Top-4 contract")
    for name in ("train_steps", "batch_size", "eval_interval", "hidden_dim"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    _set_seed(args.seed)
    device = torch.device(args.device)
    train_cache = _load_cache(args.train_cache)
    val_cache = _load_cache(args.val_cache)
    _validate_cache(train_cache, "train")
    _validate_cache(val_cache, "val")
    if int(train_cache["features"].shape[-1]) != int(val_cache["features"].shape[-1]):
        raise ValueError("train/validation feature dimensions differ")
    overlap = set(train_cache["metadata"]["image_paths"]) & set(
        val_cache["metadata"]["image_paths"]
    )
    if overlap:
        raise ValueError(f"train/validation cache overlap: {sorted(overlap)[0]}")

    selector, source_iteration = _load_source_selector(
        args.config,
        args.source_checkpoint,
    )
    expected_dim = int(selector.input_norm.normalized_shape[0])
    feature_dim = int(train_cache["features"].shape[-1])
    if expected_dim != feature_dim:
        raise ValueError(
            f"selector/cache feature mismatch: selector={expected_dim}, cache={feature_dim}"
        )
    selector = selector.to(device).eval()
    source_val_metrics = _evaluate(
        selector,
        val_cache,
        device=device,
        probe_batch_size=args.batch_size,
        top_k=4,
        input_h=args.input_h,
        input_w=args.input_w,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        nms_distance=args.nms_distance,
        nms_min_overlap_points=args.nms_min_overlap_points,
    )
    train_scores = source_scores(
        deepcopy(selector),
        train_cache,
        device=device,
        batch_size=args.batch_size,
    )
    val_scores = source_scores(
        deepcopy(selector),
        val_cache,
        device=device,
        batch_size=args.batch_size,
    )
    train_distance = pairwise_curve_distance(train_cache)
    val_distance = pairwise_curve_distance(val_cache)
    mmr_trials: list[dict[str, Any]] = []
    best_mmr: dict[str, Any] | None = None
    for sigma in args.mmr_sigmas:
        for penalty in args.mmr_penalties:
            metrics = evaluate_mmr(
                train_cache,
                train_scores,
                train_distance,
                penalty=penalty,
                sigma=sigma,
            )
            row = {
                "sigma": float(sigma),
                "penalty": float(penalty),
                "train_metrics": metrics,
            }
            mmr_trials.append(row)
            if best_mmr is None or _objective(metrics) > _objective(
                best_mmr["train_metrics"]
            ):
                best_mmr = row
    if best_mmr is None:
        raise RuntimeError("MMR parameter grid is empty")
    best_mmr = dict(best_mmr)
    best_mmr["val_metrics"] = evaluate_mmr(
        val_cache,
        val_scores,
        val_distance,
        penalty=float(best_mmr["penalty"]),
        sigma=float(best_mmr["sigma"]),
    )

    train_targets = build_soft_slot_targets(
        train_cache,
        num_slots=args.num_slots,
        temperature=args.target_temperature,
        iou_band=args.target_iou_band,
        min_iou=args.target_min_iou,
    )
    probe = FourSlotCoverageProbe(
        feature_dim,
        hidden_dim=args.hidden_dim,
        num_slots=args.num_slots,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    schedule = training_index_schedule(
        num_examples=int(train_cache["features"].shape[0]),
        batch_size=args.batch_size,
        steps=args.train_steps,
        seed=args.seed,
    )
    best = {
        "step": 0,
        "objective": float("-inf"),
        "state": _cpu_state(probe),
        "metrics": None,
    }
    trajectory: list[dict[str, Any]] = []
    running = {"total": 0.0, "pointer": 0.0, "presence": 0.0, "diversity": 0.0}
    for step in range(1, int(args.train_steps) + 1):
        indices = schedule[step - 1]
        features = train_cache["features"][indices].to(device, dtype=torch.float32)
        valid = train_cache["candidate_valid"][indices].to(device)
        target = train_targets["targets"][indices].to(device)
        active = train_targets["active"][indices].to(device)
        affinity = train_targets["affinity"][indices].to(device)
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        logits, presence = probe(features, valid)
        losses = coverage_pointer_loss(
            logits,
            presence,
            target,
            active,
            affinity,
            presence_weight=args.presence_weight,
            diversity_weight=args.diversity_weight,
        )
        losses["total"].backward()
        optimizer.step()
        for name in running:
            running[name] += float(losses[name].detach())
        should_eval = (
            step == 1
            or step == int(args.train_steps)
            or step % int(args.eval_interval) == 0
        )
        if should_eval:
            metrics = evaluate_pointer(
                probe,
                val_cache,
                device=device,
                batch_size=args.batch_size,
                fallback_scores=val_scores,
            )
            trajectory.append({"step": int(step), "metrics": metrics})
            objective = _objective(metrics)
            if objective > float(best["objective"]):
                best = {
                    "step": int(step),
                    "objective": objective,
                    "state": _cpu_state(probe),
                    "metrics": metrics,
                }
        if step % int(args.log_interval) == 0 or step == int(args.train_steps):
            denominator = float(args.log_interval if step >= args.log_interval else step)
            print(
                f"four-slot step {step:05d}/{args.train_steps:05d} | "
                + " | ".join(
                    f"{name}={value / max(denominator, 1.0):.4f}"
                    for name, value in running.items()
                )
            )
            running = {name: 0.0 for name in running}
    if best["metrics"] is None:
        raise RuntimeError("four-slot probe was never evaluated")
    probe.load_state_dict(best["state"])
    source_raw = source_val_metrics["raw_top4"]
    source_nms = source_val_metrics["nms_top4"]
    pointer_metrics = best["metrics"]
    mmr_metrics = best_mmr["val_metrics"]

    gate = coverage_gate(
        pointer_metrics,
        mmr_metrics,
        source_raw,
        min_gain_050_points=args.min_gain_050_points,
        min_gain_070_points=args.min_gain_070_points,
    )
    result = {
        "diagnostic_only": True,
        "warning": (
            "This is a frozen-feature falsification test. A positive gate "
            "supports a full architecture experiment but is not benchmark F1."
        ),
        "config": args.config,
        "source_checkpoint": args.source_checkpoint,
        "source_iteration": int(source_iteration),
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "train_images": int(train_cache["features"].shape[0]),
        "val_images": int(val_cache["features"].shape[0]),
        "probe_config": {
            "seed": int(args.seed),
            "train_steps": int(args.train_steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "num_slots": int(args.num_slots),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "ff_dim": int(args.ff_dim),
            "target_temperature": float(args.target_temperature),
            "target_iou_band": float(args.target_iou_band),
            "target_min_iou": float(args.target_min_iou),
            "presence_weight": float(args.presence_weight),
            "diversity_weight": float(args.diversity_weight),
        },
        "source": source_val_metrics,
        "mmr": {
            "selected_sigma": float(best_mmr["sigma"]),
            "selected_penalty": float(best_mmr["penalty"]),
            "val_metrics": mmr_metrics,
            **gate["mmr_gain"],
            "positive": gate["mmr_positive"],
            "train_grid": mmr_trials,
        },
        "four_slot_pointer": {
            "best_step": int(best["step"]),
            "metrics": pointer_metrics,
            **gate["pointer_gain"],
            "positive": gate["pointer_positive"],
            "trajectory": trajectory,
        },
        "gate": {
            "min_gain_050_points": float(args.min_gain_050_points),
            "min_gain_070_points": float(args.min_gain_070_points),
            "pointer_positive": gate["pointer_positive"],
            "mmr_positive": gate["mmr_positive"],
            "four_slot_hypothesis_positive": gate[
                "four_slot_hypothesis_positive"
            ],
            "any_diversity_signal_positive": gate[
                "any_diversity_signal_positive"
            ],
            "interpretation": gate["interpretation"],
            "source_nms_recall_050": float(source_nms["recall_050"]),
            "source_nms_recall_070": float(source_nms["recall_070"]),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    probe_path = Path(args.save_probe)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best["state"],
            "best_step": int(best["step"]),
            "config": vars(args),
        },
        probe_path,
    )
    print(json.dumps(result, indent=2))
    print(f"output_json: {output_path}")
    print(f"probe_checkpoint: {probe_path}")


if __name__ == "__main__":
    main()
