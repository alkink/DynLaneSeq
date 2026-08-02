from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    official_proposal_gt_iou_matrix,
    recall_from_ids,
)
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


StateDict = Dict[str, torch.Tensor]
KeyPredicate = Callable[[str], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Localize a unified lane-set collapse with zero-training module "
            "swaps between a healthy and a failed checkpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--healthy-checkpoint", required=True)
    parser.add_argument("--failed-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--score-thresholds",
        type=float,
        nargs="+",
        default=(0.20, 0.30),
    )
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=(0.50, 0.75),
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=(
            "control_healthy",
            "control_failed",
            "healthy_encoder_failed_decoder",
            "failed_encoder_healthy_decoder",
            "healthy_row_readout",
            "failed_row_readout_on_healthy",
            "healthy_row_norm",
            "failed_row_norm_on_healthy",
            "healthy_row_x",
            "failed_row_x_on_healthy",
            "healthy_lane_state_core",
            "failed_lane_state_core_on_healthy",
            "healthy_fpn_bn_buffers",
            "failed_fpn_bn_buffers_on_healthy",
        ),
    )
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Strictly construct every hybrid but do not read the dataset.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_signature(cfg: dict[str, Any]) -> str:
    contract = {
        "model": cfg.get("model"),
        "matcher": cfg.get("matcher"),
        "loss": cfg.get("loss"),
        "optimizer": cfg.get("optimizer"),
        "scheduler": cfg.get("scheduler"),
        "seed": cfg.get("training", {}).get("seed"),
    }
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint(path: str | Path) -> tuple[StateDict, int, str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    state = dict(payload["model"])
    iteration = int(payload.get("iteration", -1))
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"checkpoint has no expanded config: {path}")
    metadata = {
        "has_optimizer": isinstance(payload.get("optimizer"), dict),
        "has_scheduler": isinstance(payload.get("scheduler"), dict),
        "optimizer_groups": len(payload.get("optimizer", {}).get("param_groups", [])),
        "scheduler_last_epoch": payload.get("scheduler", {}).get("last_epoch"),
        "scheduler_step_count": payload.get("scheduler", {}).get("_step_count"),
    }
    return state, iteration, _config_signature(cfg), metadata


def _assert_state_compatible(
    label: str,
    state: StateDict,
    target: StateDict,
) -> None:
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    shape_mismatch = sorted(
        key
        for key in set(state) & set(target)
        if tuple(state[key].shape) != tuple(target[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            f"{label} is not strict-load compatible: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}, shape_mismatch={shape_mismatch[:5]}"
        )


def _prefixes(*values: str) -> KeyPredicate:
    return lambda key: key.startswith(values)


def _fpn_bn_buffer(key: str) -> bool:
    return key.startswith("encoder.fpn.") and key.endswith(
        ("running_mean", "running_var", "num_batches_tracked")
    )


def _row_reference_path(key: str) -> bool:
    prefixes = (
        "structured_query_head.layers.",
        "structured_query_head.feature_proj.",
        "structured_query_head.reference_",
        "structured_query_head.instance_tokens.",
        "structured_query_head.row_tokens.",
        "structured_query_head.x_tokens.",
        "structured_query_head.row_norm.",
        "structured_query_head.row_x.",
    )
    return key.startswith(prefixes)


VARIANTS: dict[str, tuple[str, str | None, KeyPredicate | None, str]] = {
    "control_healthy": (
        "healthy",
        None,
        None,
        "Unmodified healthy checkpoint.",
    ),
    "control_failed": (
        "failed",
        None,
        None,
        "Unmodified failed checkpoint.",
    ),
    "healthy_encoder_failed_decoder": (
        "failed",
        "healthy",
        _prefixes("encoder."),
        "Healthy encoder with the failed structured decoder.",
    ),
    "failed_encoder_healthy_decoder": (
        "failed",
        "healthy",
        _prefixes("structured_query_head."),
        "Failed encoder with the complete healthy structured decoder.",
    ),
    "healthy_row_readout": (
        "failed",
        "healthy",
        _prefixes(
            "structured_query_head.row_norm.",
            "structured_query_head.row_x.",
        ),
        "Failed model with only healthy row normalization and x readout.",
    ),
    "failed_row_readout_on_healthy": (
        "healthy",
        "failed",
        _prefixes(
            "structured_query_head.row_norm.",
            "structured_query_head.row_x.",
        ),
        "Healthy model with only the failed row normalization and x readout.",
    ),
    "healthy_row_norm": (
        "failed",
        "healthy",
        _prefixes("structured_query_head.row_norm."),
        "Failed model with only the healthy final row normalization.",
    ),
    "failed_row_norm_on_healthy": (
        "healthy",
        "failed",
        _prefixes("structured_query_head.row_norm."),
        "Healthy model with only the failed final row normalization.",
    ),
    "healthy_row_x": (
        "failed",
        "healthy",
        _prefixes("structured_query_head.row_x."),
        "Failed model with only the healthy x-distribution projection.",
    ),
    "failed_row_x_on_healthy": (
        "healthy",
        "failed",
        _prefixes("structured_query_head.row_x."),
        "Healthy model with only the failed x-distribution projection.",
    ),
    "healthy_lane_state_core": (
        "failed",
        "healthy",
        _prefixes("structured_query_head.lane_state_layers."),
        "Failed model with the healthy unified lane-state stack.",
    ),
    "failed_lane_state_core_on_healthy": (
        "healthy",
        "failed",
        _prefixes("structured_query_head.lane_state_layers."),
        "Healthy model with the failed unified lane-state stack.",
    ),
    "healthy_fpn_bn_buffers": (
        "failed",
        "healthy",
        _fpn_bn_buffer,
        "Failed model with only healthy FPN BatchNorm running buffers.",
    ),
    "failed_fpn_bn_buffers_on_healthy": (
        "healthy",
        "failed",
        _fpn_bn_buffer,
        "Healthy model with only failed FPN BatchNorm running buffers.",
    ),
    "healthy_full_fpn": (
        "failed",
        "healthy",
        _prefixes("encoder.fpn.", "encoder.proj."),
        "Failed model with the healthy FPN and P2 projection.",
    ),
    "failed_full_fpn_on_healthy": (
        "healthy",
        "failed",
        _prefixes("encoder.fpn.", "encoder.proj."),
        "Healthy model with the failed FPN and P2 projection.",
    ),
    "healthy_row_reference_path": (
        "failed",
        "healthy",
        _row_reference_path,
        "Failed model with the broad healthy row-reference geometry path.",
    ),
    "failed_row_reference_path_on_healthy": (
        "healthy",
        "failed",
        _row_reference_path,
        "Healthy model with the broad failed row-reference geometry path.",
    ),
}


def _hybrid_state(
    name: str,
    healthy: StateDict,
    failed: StateDict,
) -> tuple[StateDict, list[str], str]:
    if name not in VARIANTS:
        raise ValueError(
            f"unknown variant {name!r}; choices: {', '.join(VARIANTS)}"
        )
    base, source, predicate, description = VARIANTS[name]
    state = dict(healthy if base == "healthy" else failed)
    source_state = healthy if source == "healthy" else failed
    replaced: list[str] = []
    if predicate is not None:
        for key in state:
            if predicate(key):
                state[key] = source_state[key]
                replaced.append(key)
    if predicate is not None and not replaced:
        raise ValueError(f"variant {name!r} replaced no tensors")
    return state, replaced, description


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
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _stage_for_image(
    outputs: dict[str, torch.Tensor],
    image_index: int,
) -> dict[str, torch.Tensor]:
    return {
        name: outputs[name][image_index].detach().float().cpu()
        for name in ("pred_x_rows", "range_norm", "exist_logits", "quality_logits")
        if isinstance(outputs.get(name), torch.Tensor)
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _metric(counter: dict[str, int]) -> dict[str, float | int]:
    tp = int(counter["tp"])
    fp = int(counter["fp"])
    fn = int(counter["fn"])
    precision = float(tp) / float(max(tp + fp, 1))
    recall = float(tp) / float(max(tp + fn, 1))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _evaluate(
    model: torch.nn.Module,
    matcher: Any,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[dict[str, Any], list[int]]:
    from tqdm import tqdm

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
    capacity = {
        float(threshold): {"gt": 0, "all": 0, "direct": 0, "oracle": 0}
        for threshold in args.iou_thresholds
    }
    deployed = {
        float(score): {
            float(iou): {"tp": 0, "fp": 0, "fn": 0}
            for iou in args.iou_thresholds
        }
        for score in args.score_thresholds
    }
    all_scores: list[float] = []
    all_best_iou: list[float] = []
    matched_scores: list[float] = []
    unmatched_scores: list[float] = []
    matched_ious: list[float] = []
    unmatched_ious: list[float] = []
    probability_mass: list[float] = []
    target_counts: list[float] = []
    candidate_curve_spread: list[float] = []
    candidate_curve_pair_distance: list[float] = []
    candidate_score_std: list[float] = []
    images_seen = 0

    iterator = tqdm(
        loader,
        total=total_batches,
        desc="module-swap audit",
        ncols=100,
    )
    for batch_index, (images_cpu, targets_cpu, metas) in enumerate(iterator):
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        if device.type == "cuda":
            images = images_cpu.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images_cpu.to(device)
        targets = nested_to_device(targets_cpu, device)
        with torch.no_grad(), _amp_context(device, amp_dtype):
            outputs = model(images, inference_only=True)
            matches = matcher(outputs, targets)

        for image_index, (target_cpu, meta, match) in enumerate(
            zip(targets_cpu, metas, matches)
        ):
            stage = _stage_for_image(outputs, image_index)
            record = {"stages": {"main": stage}, "meta": meta}
            official_iou, candidate_valid = official_proposal_gt_iou_matrix(
                record,
                "main",
                line_width=float(args.line_width),
                min_valid_rows=int(args.min_valid_rows),
                row_visibility_thresh=float(args.row_visibility_thresh),
            )
            scores = torch.softmax(stage["exist_logits"], dim=-1)[..., 0]
            pred_x = stage["pred_x_rows"]
            valid_ids = torch.nonzero(
                candidate_valid,
                as_tuple=False,
            ).flatten().tolist()
            if int(official_iou.shape[0]) > 0:
                best_iou = official_iou.max(dim=0).values
            else:
                best_iou = torch.zeros_like(scores)
            matched_ids = set(
                int(value)
                for value in match["pred_indices"].detach().cpu().tolist()
            )
            for candidate_index in valid_ids:
                score = float(scores[candidate_index])
                iou = float(best_iou[candidate_index])
                all_scores.append(score)
                all_best_iou.append(iou)
                if candidate_index in matched_ids:
                    matched_scores.append(score)
                    matched_ious.append(iou)
                else:
                    unmatched_scores.append(score)
                    unmatched_ious.append(iou)

            valid_index_tensor = torch.as_tensor(valid_ids, dtype=torch.long)
            if valid_ids:
                valid_scores = scores.index_select(0, valid_index_tensor)
                valid_curves = pred_x.index_select(0, valid_index_tensor)
                probability_mass.append(float(valid_scores.sum()))
                candidate_score_std.append(float(valid_scores.std(unbiased=False)))
                candidate_curve_spread.append(
                    float(valid_curves.std(dim=0, unbiased=False).mean())
                )
                if int(valid_curves.shape[0]) > 1:
                    normalized_distance = torch.pdist(
                        valid_curves,
                        p=2,
                    ) / math.sqrt(float(valid_curves.shape[1]))
                    candidate_curve_pair_distance.append(
                        float(normalized_distance.mean())
                    )
            else:
                probability_mass.append(0.0)
                candidate_score_std.append(0.0)
                candidate_curve_spread.append(0.0)
            target_counts.append(float(int(target_cpu["x_rows"].shape[0])))

            direct_ids = sorted(
                valid_ids,
                key=lambda index: float(scores[index]),
                reverse=True,
            )[: int(args.top_k)]
            by_score = {
                float(threshold): [
                    index
                    for index in direct_ids
                    if float(scores[index]) >= float(threshold)
                ]
                for threshold in args.score_thresholds
            }
            for threshold in args.iou_thresholds:
                threshold = float(threshold)
                all_hits, gt_count, _ = recall_from_ids(
                    official_iou,
                    valid_ids,
                    threshold,
                )
                direct_hits, _gt, _ = recall_from_ids(
                    official_iou,
                    direct_ids,
                    threshold,
                )
                oracle = cardinality_oracle_assignment(
                    official_iou,
                    threshold,
                    int(args.top_k),
                    candidate_valid,
                )
                capacity[threshold]["gt"] += int(gt_count)
                capacity[threshold]["all"] += int(all_hits)
                capacity[threshold]["direct"] += int(direct_hits)
                capacity[threshold]["oracle"] += int(oracle.hit_count)
                for score_threshold, selected_ids in by_score.items():
                    hits, selected_gt, _ = recall_from_ids(
                        official_iou,
                        selected_ids,
                        threshold,
                    )
                    counter = deployed[score_threshold][threshold]
                    counter["tp"] += int(hits)
                    counter["fp"] += int(len(selected_ids) - hits)
                    counter["fn"] += int(selected_gt - hits)

        images_seen += int(images.shape[0])
        del outputs, matches, images, targets

    capacity_payload: dict[str, Any] = {}
    for threshold, counts in capacity.items():
        denominator = max(int(counts["gt"]), 1)
        capacity_payload[f"{threshold:.2f}"] = {
            "gt_lanes": int(counts["gt"]),
            "all_candidates_recall": float(counts["all"]) / denominator,
            "direct_topk_recall": float(counts["direct"]) / denominator,
            "oracle_topk_recall": float(counts["oracle"]) / denominator,
        }
    deployed_payload = {
        f"score_{score:.2f}": {
            f"{iou:.2f}": _metric(counter)
            for iou, counter in by_iou.items()
        }
        for score, by_iou in deployed.items()
    }
    return (
        {
            "images": images_seen,
            "capacity": capacity_payload,
            "deployed_operating_points": deployed_payload,
            "score_geometry_alignment": {
                "pearson_score_vs_best_official_iou": _correlation(
                    all_scores,
                    all_best_iou,
                ),
                "matched_mean_score": _mean(matched_scores),
                "unmatched_mean_score": _mean(unmatched_scores),
                "matched_mean_best_official_iou": _mean(matched_ious),
                "unmatched_mean_best_official_iou": _mean(unmatched_ious),
            },
            "candidate_diversity": {
                "mean_curve_candidate_std_px": _mean(candidate_curve_spread),
                "mean_pairwise_curve_rms_distance_px": _mean(
                    candidate_curve_pair_distance
                ),
                "mean_score_candidate_std": _mean(candidate_score_std),
            },
            "count_calibration": {
                "mean_foreground_probability_mass": _mean(probability_mass),
                "mean_target_lane_count": _mean(target_counts),
            },
        },
        sampled_indices,
    )


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _recovery_fraction(
    value: float | None,
    healthy: float | None,
    failed: float | None,
) -> float | None:
    if value is None or healthy is None or failed is None:
        return None
    denominator = float(healthy) - float(failed)
    if abs(denominator) <= 1e-12:
        return None
    return (float(value) - float(failed)) / denominator


def _interpret(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "control_healthy" not in results or "control_failed" not in results:
        return {
            "primary_localization": "controls_missing",
            "warning": "Both controls are required for automatic interpretation.",
        }
    metric_path = ("metrics", "capacity", "0.50", "all_candidates_recall")
    healthy = _nested(results["control_healthy"], *metric_path)
    failed = _nested(results["control_failed"], *metric_path)
    recoveries = {
        name: _recovery_fraction(
            _nested(result, *metric_path),
            healthy,
            failed,
        )
        for name, result in results.items()
        if name not in {"control_healthy", "control_failed"}
        and not name.endswith("_on_healthy")
    }
    damage_fractions = {
        name: _recovery_fraction(
            _nested(result, *metric_path),
            failed,
            healthy,
        )
        for name, result in results.items()
        if name.endswith("_on_healthy")
    }
    strong = [
        name
        for name, value in recoveries.items()
        if value is not None and value >= 0.50
    ]
    strong_damage = [
        name
        for name, value in damage_fractions.items()
        if value is not None and value >= 0.50
    ]
    if (
        "healthy_row_norm" in strong
        or "failed_row_norm_on_healthy" in strong_damage
    ):
        primary = "final_row_normalization_is_a_direct_causal_bottleneck"
    elif (
        "healthy_row_x" in strong
        or "failed_row_x_on_healthy" in strong_damage
    ):
        primary = "row_x_projection_is_a_direct_causal_bottleneck"
    elif (
        "healthy_row_readout" in strong
        or "failed_row_readout_on_healthy" in strong_damage
    ):
        primary = "row_readout_is_a_direct_causal_bottleneck"
    elif (
        "healthy_lane_state_core" in strong
        or "failed_lane_state_core_on_healthy" in strong_damage
    ):
        primary = "unified_lane_state_core_is_a_direct_causal_bottleneck"
    elif (
        "healthy_fpn_bn_buffers" in strong
        or "failed_fpn_bn_buffers_on_healthy" in strong_damage
    ):
        primary = "fpn_batchnorm_buffers_are_a_direct_causal_bottleneck"
    elif "failed_encoder_healthy_decoder" in strong:
        primary = "collapse_is_localized_to_the_structured_decoder"
    elif "healthy_encoder_failed_decoder" in strong:
        primary = "collapse_is_localized_to_the_encoder"
    elif strong:
        primary = "a_broad_swapped_subsystem_recovers_capacity"
    else:
        primary = "no_single_swapped_subsystem_recovers_half_the_lost_capacity"
    return {
        "primary_localization": primary,
        "all_candidate_recall_050_recovery_fraction": recoveries,
        "all_candidate_recall_050_damage_fraction": damage_fractions,
        "strong_recovery_variants": strong,
        "strong_damage_variants": strong_damage,
        "warning": (
            "A hybrid can fail because independently trained modules co-adapt. "
            "A negative swap is not proof that the swapped subsystem is healthy."
        ),
    }


def main() -> None:
    args = parse_args()
    if int(args.eval_batch_size) < 1 or int(args.max_batches) < 0:
        raise ValueError("eval-batch-size must be positive and max-batches non-negative")
    if int(args.top_k) < 1:
        raise ValueError("top-k must be positive")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("variants must not repeat")

    raw_cfg = load_config(args.config)
    current_signature = _config_signature(raw_cfg)
    cfg = _prepare_config(
        args.config,
        dataset_root=args.dataset_root,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    healthy_state, healthy_iteration, healthy_signature, healthy_metadata = (
        _load_checkpoint(args.healthy_checkpoint)
    )
    failed_state, failed_iteration, failed_signature, failed_metadata = (
        _load_checkpoint(args.failed_checkpoint)
    )
    if healthy_iteration >= failed_iteration:
        raise ValueError(
            "healthy checkpoint iteration must precede failed checkpoint iteration"
        )
    if not (
        healthy_signature == failed_signature == current_signature
    ):
        raise ValueError(
            "config/checkpoint contract signatures differ; refusing module swaps"
        )

    model = build_model(cfg)
    target_state = model.state_dict()
    _assert_state_compatible("healthy checkpoint", healthy_state, target_state)
    _assert_state_compatible("failed checkpoint", failed_state, target_state)
    device = torch.device("cpu" if args.integrity_only else args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    model = model.to(device).eval()
    matcher = None if args.integrity_only else build_matcher(cfg)

    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "warning": (
            "Module swaps are causal localization probes, not deployable "
            "models and not estimates of retrained F1."
        ),
        "config": args.config,
        "config_contract_sha256": current_signature,
        "healthy_checkpoint": {
            "path": args.healthy_checkpoint,
            "sha256": _sha256(args.healthy_checkpoint),
            "iteration": healthy_iteration,
            **healthy_metadata,
        },
        "failed_checkpoint": {
            "path": args.failed_checkpoint,
            "sha256": _sha256(args.failed_checkpoint),
            "iteration": failed_iteration,
            **failed_metadata,
        },
        "settings": {
            "split": args.split,
            "sample_strategy": args.sample_strategy,
            "eval_batch_size": int(args.eval_batch_size),
            "max_batches": int(args.max_batches),
            "amp_dtype": args.amp_dtype,
            "top_k": int(args.top_k),
            "score_thresholds": [float(value) for value in args.score_thresholds],
            "iou_thresholds": [float(value) for value in args.iou_thresholds],
            "integrity_only": bool(args.integrity_only),
        },
        "variants": {},
    }
    reference_indices: list[int] | None = None
    for name in args.variants:
        state, replaced, description = _hybrid_state(
            name,
            healthy_state,
            failed_state,
        )
        model.load_state_dict(state, strict=True)
        row: dict[str, Any] = {
            "description": description,
            "replaced_tensors": len(replaced),
            "replaced_elements": int(sum(state[key].numel() for key in replaced)),
        }
        if not args.integrity_only:
            metrics, indices = _evaluate(
                model,
                matcher,
                cfg,
                args,
                device,
                amp_dtype,
            )
            if reference_indices is None:
                reference_indices = indices
            elif indices != reference_indices:
                raise RuntimeError("module-swap variants did not use identical samples")
            row["metrics"] = metrics
        payload["variants"][name] = row

    payload["sampled_dataset_indices"] = reference_indices or []
    payload["interpretation"] = (
        {
            "primary_localization": "integrity_only",
            "warning": "Run without --integrity-only to localize the collapse.",
        }
        if args.integrity_only
        else _interpret(payload["variants"])
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    compact = dict(payload)
    compact.pop("sampled_dataset_indices")
    print(json.dumps(compact, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
