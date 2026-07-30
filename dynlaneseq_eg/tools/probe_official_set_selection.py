from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    official_proposal_gt_iou_matrix,
    resolve_list_path,
    sha256_file,
    trace_postprocess,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    QualityProbe,
    _amp_context,
    _average_precision,
    _frozen_outputs,
    _prepare_config,
    geometry_aware_features,
    pairwise_quality_ranking_loss,
    quality_focal_loss,
)


CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a LaneRowNet checkpoint and compare an independent scalar "
            "quality MLP with a proposal-set-aware scorer. Both probes use "
            "unique targets derived from the official CULane raster IoU."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--train-cache-images", type=int, default=4096)
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
    parser.add_argument("--curve-samples", type=int, default=20)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quality-focal-beta", type=float, default=2.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-target-margin", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--set-heads", type=int, default=8)
    parser.add_argument("--set-ff-dim", type=int, default=512)
    parser.add_argument("--set-dropout", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quality-power", type=float, default=0.5)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--min-gain-050-points", type=float, default=1.0)
    parser.add_argument("--min-gain-070-points", type=float, default=0.5)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-probe", required=True)
    return parser.parse_args()


class SetAwareQualityProbe(nn.Module):
    """Permutation-equivariant scorer over the complete proposal set."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.input_projection = nn.Linear(int(input_dim), int(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(self.input_norm(features))
        hidden = self.encoder(hidden)
        return self.output(self.output_norm(hidden)).squeeze(-1)


def official_unique_quality_targets(
    official_iou: torch.Tensor,
    candidate_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Maximum-IoU one-to-one target in the official raster-IoU space."""

    gt_count, candidate_count = official_iou.shape
    targets = official_iou.new_zeros(candidate_count, dtype=torch.float32)
    if gt_count == 0 or candidate_count == 0:
        return targets
    if candidate_valid is None:
        candidate_valid = torch.ones(
            candidate_count,
            dtype=torch.bool,
            device=official_iou.device,
        )
    valid_ids = [
        index
        for index in range(candidate_count)
        if bool(candidate_valid[index])
    ]
    if not valid_ids:
        return targets
    matrix = official_iou[:, valid_ids].detach().cpu().numpy()
    gt_indices, local_candidate_indices = linear_sum_assignment(1.0 - matrix)
    if len(gt_indices) == 0:
        return targets
    gt_tensor = torch.as_tensor(gt_indices, dtype=torch.long)
    local_tensor = torch.as_tensor(local_candidate_indices, dtype=torch.long)
    candidate_tensor = torch.as_tensor(
        [valid_ids[index] for index in local_candidate_indices],
        dtype=torch.long,
    )
    assigned = official_iou.detach().cpu()[gt_tensor, candidate_tensor]
    targets[candidate_tensor.to(targets.device)] = assigned.to(targets.device)
    return targets


def selection_features(
    outputs: dict[str, torch.Tensor],
    *,
    input_w: int,
    curve_samples: int,
) -> torch.Tensor:
    """Shared per-proposal features for the independent and set-aware arms."""

    base = geometry_aware_features(outputs, input_w=input_w)
    pred_x = outputs["pred_x_rows"].detach().float()
    row_logits = outputs["row_x_logits"].detach().float()
    rows = int(pred_x.shape[-1])
    sample_count = max(1, min(int(curve_samples), rows))
    sample_ids = torch.linspace(
        0,
        rows - 1,
        sample_count,
        device=pred_x.device,
    ).round().long()
    sampled_x = pred_x.index_select(-1, sample_ids)
    sampled_x = sampled_x / float(max(int(input_w) - 1, 1))
    sampled_confidence = (
        row_logits.amax(dim=-1) - torch.logsumexp(row_logits, dim=-1)
    ).exp()
    sampled_confidence = sampled_confidence.index_select(-1, sample_ids)
    return torch.cat((base, sampled_x, sampled_confidence), dim=-1)


def training_index_schedule(
    *,
    num_examples: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> torch.Tensor:
    if int(num_examples) < 1:
        raise ValueError("num_examples must be positive")
    if int(batch_size) < 1 or int(steps) < 1:
        raise ValueError("batch_size and steps must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randint(
        0,
        int(num_examples),
        (int(steps), int(batch_size)),
        generator=generator,
    )


def assert_disjoint_image_paths(
    train_paths: Iterable[str],
    val_paths: Iterable[str],
) -> None:
    overlap = set(str(value) for value in train_paths) & set(
        str(value) for value in val_paths
    )
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(f"train/validation cache overlap detected: {example}")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_signature(
    *,
    cfg: dict[str, Any],
    config_path: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    split: str,
    list_path: str,
    list_sha256: str,
    sample_count: int,
    curve_samples: int,
    line_width: float,
    min_valid_rows: int,
    row_visibility_thresh: float,
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "resolved_config_sha256": _sha256_json(cfg),
        "config_path": str(Path(config_path)),
        "checkpoint_path": str(Path(checkpoint_path)),
        "checkpoint_sha256": checkpoint_sha256,
        "split": str(split),
        "list_path": str(list_path),
        "list_sha256": str(list_sha256),
        "sample_strategy": "uniform",
        "sample_count": int(sample_count),
        "curve_samples": int(curve_samples),
        "line_width": float(line_width),
        "min_valid_rows": int(min_valid_rows),
        "row_visibility_thresh": float(row_visibility_thresh),
    }


def _stage_for_image(
    outputs: dict[str, torch.Tensor],
    batch_index: int,
) -> dict[str, torch.Tensor]:
    fields = (
        "pred_x_rows",
        "range_norm",
        "exist_logits",
        "quality_logits",
    )
    return {
        name: outputs[name][batch_index].detach().float().cpu()
        for name in fields
        if isinstance(outputs.get(name), torch.Tensor)
    }


@torch.no_grad()
def collect_selection_cache(
    model: nn.Module,
    cfg: dict[str, Any],
    *,
    split: str,
    sample_count: int,
    cache_batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    input_w: int,
    curve_samples: int,
    line_width: float,
    min_valid_rows: int,
    row_visibility_thresh: float,
    signature: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(cache_batch_size)
    cfg["dataloader"]["num_workers"] = int(num_workers)
    if int(num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    base_loader = build_dataloader(cfg, split=split, training=False)
    max_batches = math.ceil(int(sample_count) / int(cache_batch_size))
    loader, dataset_indices = select_diagnostic_loader(
        base_loader,
        strategy="uniform",
        max_batches=max_batches,
        num_workers=num_workers,
    )
    if len(dataset_indices) != int(sample_count):
        raise ValueError(
            f"requested {sample_count} {split} samples, got {len(dataset_indices)}"
        )

    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    official_rows: list[torch.Tensor] = []
    stage_rows: dict[str, list[torch.Tensor]] = {
        "pred_x_rows": [],
        "range_norm": [],
        "exist_logits": [],
        "quality_logits": [],
    }
    image_paths: list[str] = []

    for images, _targets, metas in tqdm(
        loader,
        desc=f"official selection cache ({split})",
        ncols=80,
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
        features = selection_features(
            outputs,
            input_w=input_w,
            curve_samples=curve_samples,
        )
        for batch_index, meta in enumerate(metas):
            stage = _stage_for_image(outputs, batch_index)
            record = {"stages": {"main": stage}, "meta": meta}
            official_iou, candidate_valid = official_proposal_gt_iou_matrix(
                record,
                "main",
                line_width=line_width,
                min_valid_rows=min_valid_rows,
                row_visibility_thresh=row_visibility_thresh,
            )
            targets = official_unique_quality_targets(
                official_iou,
                candidate_valid,
            )
            feature_rows.append(features[batch_index].detach().half().cpu())
            target_rows.append(targets.float().cpu())
            valid_rows.append(candidate_valid.bool().cpu())
            official_rows.append(official_iou.float().cpu())
            for name, tensor in stage.items():
                stage_rows[name].append(tensor)
            image_paths.append(str(meta.get("image_path", "")))

    if not feature_rows:
        raise ValueError(f"no samples were collected for split={split}")
    cache = {
        "metadata": {
            **signature,
            "dataset_indices": dataset_indices,
            "image_paths": image_paths,
            "feature_dim": int(feature_rows[0].shape[-1]),
            "num_candidates": int(feature_rows[0].shape[0]),
            "num_images": len(feature_rows),
        },
        "features": torch.stack(feature_rows, dim=0),
        "targets": torch.stack(target_rows, dim=0),
        "candidate_valid": torch.stack(valid_rows, dim=0),
        "official_iou": official_rows,
        "stage": {
            name: torch.stack(rows, dim=0)
            for name, rows in stage_rows.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


def _load_cache(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_or_collect_selection_cache(
    model: nn.Module,
    cfg: dict[str, Any],
    *,
    split: str,
    sample_count: int,
    cache_batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    input_w: int,
    curve_samples: int,
    line_width: float,
    min_valid_rows: int,
    row_visibility_thresh: float,
    signature: dict[str, Any],
    output_path: Path,
    reuse_cache: bool,
) -> dict[str, Any]:
    if reuse_cache and output_path.exists():
        cache = _load_cache(output_path)
        metadata = cache.get("metadata", {})
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in signature.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"cache signature mismatch for {output_path}: {mismatches}"
            )
        return cache
    return collect_selection_cache(
        model,
        cfg,
        split=split,
        sample_count=sample_count,
        cache_batch_size=cache_batch_size,
        num_workers=num_workers,
        device=device,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        input_w=input_w,
        curve_samples=curve_samples,
        line_width=line_width,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
        signature=signature,
        output_path=output_path,
    )


def _new_counts() -> dict[str, Any]:
    return {
        "gt": 0,
        "selected": 0,
        "hits": {0.5: 0, 0.7: 0},
        "scores": [],
        "labels": {0.5: [], 0.7: []},
    }


def _update_counts(
    counts: dict[str, Any],
    official_iou: torch.Tensor,
    selected_ids: Iterable[int],
    *,
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
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
    best = (
        official_iou.max(dim=0).values
        if int(official_iou.shape[0]) > 0
        else official_iou.new_zeros(int(official_iou.shape[1]))
    )
    for candidate_index in range(int(scores.shape[0])):
        if not bool(candidate_valid[candidate_index]):
            continue
        counts["scores"].append(float(scores[candidate_index]))
        for threshold in (0.5, 0.7):
            counts["labels"][threshold].append(
                int(float(best[candidate_index]) > threshold)
            )


def _finish_counts(counts: dict[str, Any]) -> dict[str, Any]:
    gt = int(counts["gt"])
    selected = int(counts["selected"])
    output: dict[str, Any] = {
        "gt_lanes": gt,
        "selected_predictions": selected,
    }
    for threshold in (0.5, 0.7):
        suffix = f"{int(round(100 * threshold)):03d}"
        hits = int(counts["hits"][threshold])
        precision = float(hits) / float(max(selected, 1))
        recall = float(hits) / float(max(gt, 1))
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        output[f"tp_{suffix}"] = hits
        output[f"precision_{suffix}"] = precision
        output[f"recall_{suffix}"] = recall
        output[f"f1_{suffix}"] = f1
        output[f"candidate_ap_{suffix}"] = _average_precision(
            counts["scores"],
            counts["labels"][threshold],
        )
    return output


def _topk_ids(
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    top_k: int,
) -> list[int]:
    valid_ids = [
        index
        for index in range(int(scores.shape[0]))
        if bool(candidate_valid[index])
    ]
    valid_ids.sort(key=lambda index: float(scores[index]), reverse=True)
    return valid_ids[: int(top_k)]


@torch.no_grad()
def evaluate_selection_probes(
    cache: dict[str, Any],
    scalar_probe: nn.Module,
    set_probe: nn.Module,
    *,
    device: torch.device,
    probe_batch_size: int,
    quality_power: float,
    top_k: int,
    input_h: int,
    input_w: int,
    min_valid_rows: int,
    row_visibility_thresh: float,
    nms_distance: float,
    nms_min_overlap_points: int,
) -> dict[str, Any]:
    scalar_probe.eval()
    set_probe.eval()
    features = cache["features"]
    logits_by_strategy: dict[str, list[torch.Tensor]] = {
        "official_scalar_quality": [],
        "official_set_quality": [],
    }
    for start in range(0, int(features.shape[0]), int(probe_batch_size)):
        batch = features[start : start + int(probe_batch_size)].to(
            device=device,
            dtype=torch.float32,
        )
        logits_by_strategy["official_scalar_quality"].append(
            scalar_probe(batch).float().cpu()
        )
        logits_by_strategy["official_set_quality"].append(
            set_probe(batch).float().cpu()
        )
    scalar_logits = torch.cat(
        logits_by_strategy["official_scalar_quality"],
        dim=0,
    )
    set_logits = torch.cat(
        logits_by_strategy["official_set_quality"],
        dim=0,
    )

    exist_probability = torch.softmax(
        cache["stage"]["exist_logits"].float(),
        dim=-1,
    )[..., 0]
    current_quality = torch.sigmoid(
        cache["stage"]["quality_logits"].float()
    ).clamp_min(1e-6)
    scalar_quality = torch.sigmoid(scalar_logits).clamp_min(1e-6)
    set_quality = torch.sigmoid(set_logits).clamp_min(1e-6)
    scores_by_strategy = {
        "current_exist_quality": (
            exist_probability * current_quality.pow(float(quality_power))
        ),
        "exist_only": exist_probability,
        "official_scalar_quality": scalar_quality,
        "official_set_quality": set_quality,
        "official_scalar_exist_fusion": (
            exist_probability * scalar_quality.pow(float(quality_power))
        ),
        "official_set_exist_fusion": (
            exist_probability * set_quality.pow(float(quality_power))
        ),
    }
    modes = {
        "raw_top4": {
            name: _new_counts()
            for name in scores_by_strategy
        },
        "nms_top4": {
            name: _new_counts()
            for name in scores_by_strategy
        },
    }
    oracle_hits = {
        0.5: 0,
        0.7: 0,
    }
    oracle_gt = 0

    for image_index, official_iou in enumerate(cache["official_iou"]):
        candidate_valid = cache["candidate_valid"][image_index].bool()
        oracle_gt += int(official_iou.shape[0])
        for threshold in (0.5, 0.7):
            oracle = cardinality_oracle_assignment(
                official_iou,
                threshold=threshold,
                top_k=top_k,
                candidate_valid=candidate_valid,
            )
            oracle_hits[threshold] += int(oracle.hit_count)
        stage = {
            name: value[image_index]
            for name, value in cache["stage"].items()
        }
        for strategy_name, all_scores in scores_by_strategy.items():
            scores = all_scores[image_index]
            raw_ids = _topk_ids(scores, candidate_valid, top_k)
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
            _update_counts(
                modes["raw_top4"][strategy_name],
                official_iou,
                raw_ids,
                scores=scores,
                candidate_valid=candidate_valid,
            )
            _update_counts(
                modes["nms_top4"][strategy_name],
                official_iou,
                trace["selected_ids"],
                scores=scores,
                candidate_valid=candidate_valid,
            )

    return {
        "modes": {
            mode: {
                name: _finish_counts(counts)
                for name, counts in strategies.items()
            }
            for mode, strategies in modes.items()
        },
        "oracle_top4": {
            f"{threshold:.2f}": {
                "gt_lanes": int(oracle_gt),
                "tp": int(oracle_hits[threshold]),
                "recall": float(oracle_hits[threshold]) / float(max(oracle_gt, 1)),
            }
            for threshold in (0.5, 0.7)
        },
    }


def selection_verdict(
    evaluation: dict[str, Any],
    *,
    min_gain_050_points: float,
    min_gain_070_points: float,
) -> dict[str, Any]:
    rows = evaluation["modes"]["nms_top4"]
    baseline = rows["current_exist_quality"]
    arms: dict[str, Any] = {}
    for name in ("official_scalar_quality", "official_set_quality"):
        row = rows[name]
        gain_050 = 100.0 * (
            float(row["f1_050"]) - float(baseline["f1_050"])
        )
        gain_070 = 100.0 * (
            float(row["f1_070"]) - float(baseline["f1_070"])
        )
        arms[name] = {
            "gain_f1_050_points": gain_050,
            "gain_f1_070_points": gain_070,
            "positive": bool(
                gain_050 >= float(min_gain_050_points)
                and gain_070 >= float(min_gain_070_points)
            ),
        }
    scalar_positive = bool(arms["official_scalar_quality"]["positive"])
    set_positive = bool(arms["official_set_quality"]["positive"])
    if scalar_positive:
        recommendation = "official_iou_quality_target_is_sufficient"
    elif set_positive:
        recommendation = "add_set_aware_selection_head"
    else:
        recommendation = "selection_probe_negative_revisit_joint_proposal_decoder"
    return {
        "gate": {
            "min_gain_f1_050_points": float(min_gain_050_points),
            "min_gain_f1_070_points": float(min_gain_070_points),
        },
        "arms": arms,
        "recommendation": recommendation,
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "cache_batch_size",
        "train_cache_images",
        "val_cache_images",
        "num_workers",
        "curve_samples",
        "min_valid_rows",
        "train_steps",
        "probe_batch_size",
        "hidden_dim",
        "set_layers",
        "set_heads",
        "set_ff_dim",
        "top_k",
        "nms_min_overlap_points",
    )
    for name in positive_integer_fields:
        value = int(getattr(args, name))
        if name == "num_workers":
            if value < 0:
                raise ValueError("num_workers must be non-negative")
        elif value < 1:
            raise ValueError(f"{name} must be positive")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _set_seed(args.seed)

    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    cfg = _prepare_config(
        args.config,
        dataset_root=args.dataset_root,
        batch_size=args.cache_batch_size,
        eval_batch_size=args.cache_batch_size,
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
        raise ValueError("official selection probe requires a structured query head")
    model.structured_query_head.intermediate_supervision = False
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    checkpoint_sha256 = sha256_file(args.checkpoint)
    list_metadata = {
        split: {
            "path": str(resolve_list_path(cfg, split).resolve()),
            "sha256": sha256_file(resolve_list_path(cfg, split)),
        }
        for split in ("train", "val")
    }
    signatures = {
        split: _cache_signature(
            cfg=cfg,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            split=split,
            list_path=list_metadata[split]["path"],
            list_sha256=list_metadata[split]["sha256"],
            sample_count=sample_count,
            curve_samples=args.curve_samples,
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        for split, sample_count in (
            ("train", args.train_cache_images),
            ("val", args.val_cache_images),
        )
    }
    train_cache = load_or_collect_selection_cache(
        model,
        cfg,
        split="train",
        sample_count=args.train_cache_images,
        cache_batch_size=args.cache_batch_size,
        num_workers=args.num_workers,
        device=device,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        input_w=input_w,
        curve_samples=args.curve_samples,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        signature=signatures["train"],
        output_path=Path(args.train_cache),
        reuse_cache=args.reuse_cache,
    )
    val_cache = load_or_collect_selection_cache(
        model,
        cfg,
        split="val",
        sample_count=args.val_cache_images,
        cache_batch_size=args.cache_batch_size,
        num_workers=args.num_workers,
        device=device,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        input_w=input_w,
        curve_samples=args.curve_samples,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        signature=signatures["val"],
        output_path=Path(args.val_cache),
        reuse_cache=args.reuse_cache,
    )
    assert_disjoint_image_paths(
        train_cache["metadata"]["image_paths"],
        val_cache["metadata"]["image_paths"],
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Cache collection creates DataLoader iterators and therefore advances the
    # global torch RNG even though the detector itself is frozen. Reset here so
    # a fresh-cache run and a --reuse-cache run initialize and train identical
    # probes under the same declared seed.
    _set_seed(args.seed)
    feature_dim = int(train_cache["features"].shape[-1])
    if int(val_cache["features"].shape[-1]) != feature_dim:
        raise ValueError("train and validation feature dimensions differ")
    scalar_probe = QualityProbe(
        feature_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    set_probe = SetAwareQualityProbe(
        feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.set_layers,
        num_heads=args.set_heads,
        ff_dim=args.set_ff_dim,
        dropout=args.set_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(scalar_probe.parameters()) + list(set_probe.parameters()),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    schedule = training_index_schedule(
        num_examples=int(train_cache["features"].shape[0]),
        batch_size=args.probe_batch_size,
        steps=args.train_steps,
        seed=args.seed,
    )
    training_totals = {
        "scalar_quality": 0.0,
        "scalar_rank": 0.0,
        "set_quality": 0.0,
        "set_rank": 0.0,
        "target_mean": 0.0,
        "target_ge_050": 0.0,
    }
    running = {key: 0.0 for key in training_totals}

    scalar_probe.train()
    set_probe.train()
    for step_index in tqdm(
        range(int(args.train_steps)),
        desc="official scalar/set probe train",
        ncols=80,
    ):
        indices = schedule[step_index]
        features = train_cache["features"][indices].to(
            device=device,
            dtype=torch.float32,
        )
        targets = train_cache["targets"][indices].to(
            device=device,
            dtype=torch.float32,
        )
        scalar_logits = scalar_probe(features)
        set_logits = set_probe(features)
        scalar_quality = quality_focal_loss(
            scalar_logits,
            targets,
            beta=args.quality_focal_beta,
        )
        set_quality = quality_focal_loss(
            set_logits,
            targets,
            beta=args.quality_focal_beta,
        )
        scalar_rank = pairwise_quality_ranking_loss(
            scalar_logits,
            targets,
            target_margin=args.rank_target_margin,
        )
        set_rank = pairwise_quality_ranking_loss(
            set_logits,
            targets,
            target_margin=args.rank_target_margin,
        )
        total_loss = (
            scalar_quality
            + set_quality
            + float(args.rank_loss_weight) * (scalar_rank + set_rank)
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(scalar_probe.parameters()) + list(set_probe.parameters()),
            max_norm=5.0,
        )
        optimizer.step()

        values = {
            "scalar_quality": float(scalar_quality.detach()),
            "scalar_rank": float(scalar_rank.detach()),
            "set_quality": float(set_quality.detach()),
            "set_rank": float(set_rank.detach()),
            "target_mean": float(targets.mean()),
            "target_ge_050": float((targets > 0.5).float().mean()),
        }
        for name, value in values.items():
            running[name] += value
            training_totals[name] += value
        step = step_index + 1
        if int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
            denominator = float(args.log_interval)
            tqdm.write(
                f"step {step:05d}/{int(args.train_steps):05d} "
                f"scalar={running['scalar_quality'] / denominator:.4f} "
                f"set={running['set_quality'] / denominator:.4f} "
                f"rank_scalar={running['scalar_rank'] / denominator:.4f} "
                f"rank_set={running['set_rank'] / denominator:.4f} "
                f"target={running['target_mean'] / denominator:.4f}"
            )
            for name in running:
                running[name] = 0.0

    evaluation = evaluate_selection_probes(
        val_cache,
        scalar_probe,
        set_probe,
        device=device,
        probe_batch_size=args.probe_batch_size,
        quality_power=args.quality_power,
        top_k=args.top_k,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        nms_distance=args.nms_distance,
        nms_min_overlap_points=args.nms_min_overlap_points,
    )
    verdict = selection_verdict(
        evaluation,
        min_gain_050_points=args.min_gain_050_points,
        min_gain_070_points=args.min_gain_070_points,
    )
    output_payload = {
        "diagnostic_only": True,
        "warning": (
            "The detector is frozen and official CULane ground truth is used "
            "only to train diagnostic scoring heads on the train split. This "
            "is not a benchmark result or a deployable model."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "checkpoint_sha256": checkpoint_sha256,
        "target": {
            "name": "official_raster_iou_unique_hungarian",
            "line_width": float(args.line_width),
            "description": (
                "maximum-IoU one-to-one Hungarian target in official CULane "
                "raster space; unmatched duplicates/background receive zero"
            ),
        },
        "feature_contract": {
            "feature_dim": feature_dim,
            "curve_samples": int(args.curve_samples),
            "shared_between_arms": True,
        },
        "training": {
            "steps": int(args.train_steps),
            "batch_size": int(args.probe_batch_size),
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "quality_focal_beta": float(args.quality_focal_beta),
            "rank_loss_weight": float(args.rank_loss_weight),
            "rank_target_margin": float(args.rank_target_margin),
            "mean_losses": {
                name: value / float(args.train_steps)
                for name, value in training_totals.items()
            },
        },
        "probes": {
            "scalar_parameters": sum(
                parameter.numel()
                for parameter in scalar_probe.parameters()
            ),
            "set_parameters": sum(
                parameter.numel()
                for parameter in set_probe.parameters()
            ),
            "set_layers": int(args.set_layers),
            "set_heads": int(args.set_heads),
            "set_ff_dim": int(args.set_ff_dim),
            "set_dropout": float(args.set_dropout),
            "permutation_equivariant": True,
        },
        "caches": {
            "train": {
                "path": args.train_cache,
                **train_cache["metadata"],
            },
            "val": {
                "path": args.val_cache,
                **val_cache["metadata"],
            },
            "split_paths_disjoint": True,
        },
        "evaluation_settings": {
            "split": "val",
            "score_threshold": None,
            "top_k": int(args.top_k),
            "quality_power": float(args.quality_power),
            "nms_distance": float(args.nms_distance),
            "nms_min_overlap_points": int(args.nms_min_overlap_points),
            "min_valid_rows": int(args.min_valid_rows),
            "row_visibility_thresh": float(args.row_visibility_thresh),
        },
        "evaluation": evaluation,
        "verdict": verdict,
    }

    save_path = Path(args.save_probe)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scalar_probe": scalar_probe.state_dict(),
            "set_probe": set_probe.state_dict(),
            "feature_dim": feature_dim,
            "source_checkpoint": args.checkpoint,
            "source_iteration": int(checkpoint_iteration),
            "source_checkpoint_sha256": checkpoint_sha256,
            "target": "official_raster_iou_unique_hungarian",
            "curve_samples": int(args.curve_samples),
            "training_steps": int(args.train_steps),
            "training_seed": int(args.seed),
            "set_config": {
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.set_layers),
                "num_heads": int(args.set_heads),
                "ff_dim": int(args.set_ff_dim),
                "dropout": float(args.set_dropout),
            },
        },
        save_path,
    )
    output_payload["probe_checkpoint"] = str(save_path)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output_payload, indent=2))
    print(f"output_json: {output_path}")
    print(f"probe_checkpoint: {save_path}")


if __name__ == "__main__":
    main()
