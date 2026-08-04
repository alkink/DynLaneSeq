from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    proposal_gt_iou_matrix,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether V4.5 GT-cluster soft targets are safe and jointly "
            "realizable before starting any pointer training."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--representable-thresholds", type=float, nargs="+", default=[0.50]
    )
    parser.add_argument(
        "--support-mins", type=float, nargs="+", default=[0.45, 0.50]
    )
    parser.add_argument(
        "--quality-deltas", type=float, nargs="+", default=[0.03, 0.05, 0.10]
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.03, 0.05, 0.10]
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(max(float(denominator), 1.0))


def value_summary(values: Iterable[float]) -> dict[str, int | float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    def percentile(fraction: float) -> float:
        index = int(round(float(fraction) * float(len(ordered) - 1)))
        return ordered[min(max(index, 0), len(ordered) - 1)]

    return {
        "count": len(ordered),
        "mean": sum(ordered) / float(len(ordered)),
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def maximum_cardinality_support_assignment(
    quality: torch.Tensor,
    support: torch.Tensor,
    *,
    top_k: int,
) -> tuple[tuple[int, int], ...]:
    """Return a lexicographic maximum-cardinality matching on a support graph.

    A normal maximum-quality Hungarian assignment followed by filtering can
    lose a valid GT because a high-quality edge may consume the only candidate
    available to a second GT.  The cardinality bonus makes support coverage the
    primary objective and detached quality only the tie-breaker.
    """

    q = quality.detach().cpu().float()
    mask = support.detach().cpu().bool()
    if q.ndim != 2 or mask.shape != q.shape:
        raise ValueError("quality and support must have the same 2-D shape")
    gt_count, candidate_count = q.shape
    if gt_count == 0 or candidate_count == 0 or int(top_k) <= 0:
        return ()

    assignment_size = min(int(gt_count), int(candidate_count))
    reward = (
        mask.numpy().astype(np.float64) * float(assignment_size + 1)
        + q.numpy().astype(np.float64)
    )
    gt_ids, candidate_ids = linear_sum_assignment(-reward)
    pairs = [
        (int(gt_id), int(candidate_id))
        for gt_id, candidate_id in zip(gt_ids.tolist(), candidate_ids.tolist())
        if bool(mask[gt_id, candidate_id])
    ]
    if len(pairs) > int(top_k):
        pairs.sort(key=lambda pair: float(q[pair[0], pair[1]]), reverse=True)
        pairs = pairs[: int(top_k)]
    return tuple(pairs)


def cluster_soft_distribution(
    quality: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    support_min: float,
    quality_delta: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build one GT cluster's support and normalized soft representative target."""

    q = quality.detach().cpu().float().flatten()
    valid = candidate_valid.detach().cpu().bool().flatten()
    if q.shape != valid.shape:
        raise ValueError("quality and candidate-valid vectors must align")
    if not bool(valid.any()):
        raise ValueError("a cluster distribution requires a valid candidate")
    if float(temperature) <= 0.0:
        raise ValueError("cluster temperature must be positive")
    q_best = float(q[valid].max())
    cutoff = max(float(support_min), q_best - float(quality_delta))
    support = valid & (q >= cutoff)
    if not bool(support.any()):
        raise RuntimeError("cluster support unexpectedly excluded its best candidate")
    probabilities = torch.zeros_like(q)
    probabilities[support] = torch.softmax(q[support] / float(temperature), dim=0)
    return support, probabilities, cutoff


def _grid_key(
    representable_threshold: float,
    support_min: float,
    quality_delta: float,
    temperature: float,
) -> str:
    return (
        f"repr={representable_threshold:.3f}|support={support_min:.3f}|"
        f"delta={quality_delta:.3f}|temp={temperature:.3f}"
    )


def _second_best(values: torch.Tensor) -> float:
    if values.numel() < 2:
        return 0.0
    return float(torch.topk(values, k=2).values[1])


def analyze_cluster_support_image(
    quality: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    representable_thresholds: tuple[float, ...],
    support_mins: tuple[float, ...],
    quality_deltas: tuple[float, ...],
    temperatures: tuple[float, ...],
    top_k: int,
) -> dict[str, Any]:
    q = quality.detach().cpu().float()
    valid = candidate_valid.detach().cpu().bool()
    if q.ndim != 2 or valid.ndim != 1 or int(q.shape[1]) != int(valid.numel()):
        raise ValueError("quality must be [GT,N] and candidate_valid must be [N]")

    valid_ids = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    result: dict[str, Any] = {
        "gt_count": int(q.shape[0]),
        "valid_candidate_count": int(valid.sum()),
        "thresholds": {},
    }
    for representable_threshold in representable_thresholds:
        threshold = float(representable_threshold)
        if q.shape[0] == 0 or not valid_ids:
            best = q.new_zeros((int(q.shape[0]),))
        else:
            best = q[:, valid].max(dim=1).values
        individual_count = int((best > threshold).sum())
        deployable_individual_count = min(individual_count, int(top_k))
        joint = cardinality_oracle_assignment(
            q,
            threshold=threshold,
            top_k=int(top_k),
            candidate_valid=valid,
        )
        vanilla = evaluator_hungarian_assignment(q, valid_ids, threshold=threshold)
        vanilla_count = min(int(vanilla.hit_count), int(top_k))
        joint_gt_ids = [int(gt_id) for gt_id, _candidate_id in joint.pairs]

        q_best_values: list[float] = []
        q_second_values: list[float] = []
        q_gap_values: list[float] = []
        for gt_id in joint_gt_ids:
            values = q[gt_id, valid]
            q_best = float(values.max())
            q_second = _second_best(values)
            q_best_values.append(q_best)
            q_second_values.append(q_second)
            q_gap_values.append(q_best - q_second)

        threshold_result: dict[str, Any] = {
            "individual_representable_count": individual_count,
            "deployable_individual_count": deployable_individual_count,
            "joint_representable_count": int(joint.hit_count),
            "vanilla_hungarian_count": vanilla_count,
            "individual_to_joint_collision_loss": max(
                deployable_individual_count - int(joint.hit_count), 0
            ),
            "vanilla_to_cardinality_first_loss": max(
                int(joint.hit_count) - vanilla_count, 0
            ),
            "joint_q_best": q_best_values,
            "joint_q_second": q_second_values,
            "joint_q_best_minus_second": q_gap_values,
            "grids": {},
        }

        joint_quality = (
            q[joint_gt_ids]
            if joint_gt_ids
            else q.new_zeros((0, int(q.shape[1])))
        )
        for support_min in support_mins:
            for quality_delta in quality_deltas:
                for temperature in temperatures:
                    key = _grid_key(
                        threshold,
                        float(support_min),
                        float(quality_delta),
                        float(temperature),
                    )
                    supports: list[torch.Tensor] = []
                    probabilities: list[torch.Tensor] = []
                    support_sizes: list[float] = []
                    entropy_values: list[float] = []
                    normalized_entropy_values: list[float] = []
                    expected_quality_values: list[float] = []
                    expected_regret_values: list[float] = []
                    below_repr_mass_values: list[float] = []
                    cutoffs: list[float] = []
                    for cluster_quality in joint_quality:
                        support, probability, cutoff = cluster_soft_distribution(
                            cluster_quality,
                            valid,
                            support_min=float(support_min),
                            quality_delta=float(quality_delta),
                            temperature=float(temperature),
                        )
                        supports.append(support)
                        probabilities.append(probability)
                        support_size = int(support.sum())
                        entropy = float(
                            -(
                                probability[support]
                                * probability[support].clamp_min(1e-12).log()
                            ).sum()
                        )
                        normalized_entropy = (
                            entropy / math.log(float(support_size))
                            if support_size > 1
                            else 0.0
                        )
                        expected_quality = float((probability * cluster_quality).sum())
                        q_best = float(cluster_quality[valid].max())
                        below_mass = float(
                            probability[cluster_quality <= threshold].sum()
                        )
                        support_sizes.append(float(support_size))
                        entropy_values.append(entropy)
                        normalized_entropy_values.append(normalized_entropy)
                        expected_quality_values.append(expected_quality)
                        expected_regret_values.append(q_best - expected_quality)
                        below_repr_mass_values.append(below_mass)
                        cutoffs.append(float(cutoff))

                    cluster_count = len(supports)
                    if supports:
                        support_matrix = torch.stack(supports, dim=0)
                        probability_matrix = torch.stack(probabilities, dim=0)
                        support_pairs = maximum_cardinality_support_assignment(
                            joint_quality,
                            support_matrix,
                            top_k=int(top_k),
                        )
                        memberships = support_matrix.sum(dim=0)
                        supported = memberships > 0
                        multi_cluster = memberships > 1
                        intersecting_pairs = 0
                        expected_pair_collisions = 0.0
                        sole_support_conflicts = 0
                        pair_count = 0
                        for left in range(cluster_count):
                            for right in range(left + 1, cluster_count):
                                pair_count += 1
                                shared = support_matrix[left] & support_matrix[right]
                                intersecting_pairs += int(bool(shared.any()))
                                expected_pair_collisions += float(
                                    (
                                        probability_matrix[left]
                                        * probability_matrix[right]
                                    ).sum()
                                )
                                sole_support_conflicts += int(
                                    int(support_matrix[left].sum()) == 1
                                    and int(support_matrix[right].sum()) == 1
                                    and bool(shared.any())
                                )
                        supported_candidate_count = int(supported.sum())
                        multi_cluster_candidate_count = int(multi_cluster.sum())
                    else:
                        support_pairs = ()
                        pair_count = 0
                        intersecting_pairs = 0
                        expected_pair_collisions = 0.0
                        sole_support_conflicts = 0
                        supported_candidate_count = 0
                        multi_cluster_candidate_count = 0

                    threshold_result["grids"][key] = {
                        "representable_threshold": threshold,
                        "support_min": float(support_min),
                        "quality_delta": float(quality_delta),
                        "temperature": float(temperature),
                        "cluster_count": cluster_count,
                        "joint_support_matched_count": len(support_pairs),
                        "full_joint_support_coverage": len(support_pairs)
                        == cluster_count,
                        "support_sizes": support_sizes,
                        "target_entropy": entropy_values,
                        "target_normalized_entropy": normalized_entropy_values,
                        "target_expected_quality": expected_quality_values,
                        "target_expected_regret": expected_regret_values,
                        "target_mass_at_or_below_representable_threshold": (
                            below_repr_mass_values
                        ),
                        "support_cutoffs": cutoffs,
                        "supported_candidate_count": supported_candidate_count,
                        "multi_cluster_candidate_count": multi_cluster_candidate_count,
                        "gt_pair_count": pair_count,
                        "intersecting_gt_pair_count": intersecting_pairs,
                        "expected_pair_collision_sum": expected_pair_collisions,
                        "sole_support_conflict_count": sole_support_conflicts,
                    }
        result["thresholds"][f"{threshold:.3f}"] = threshold_result
    return result


def summarize_cluster_support(
    rows: list[dict[str, Any]],
    *,
    representable_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "images": len(rows),
        "gt_lanes": sum(int(row["gt_count"]) for row in rows),
        "valid_candidates_per_image": value_summary(
            float(row["valid_candidate_count"]) for row in rows
        ),
        "thresholds": {},
    }
    eligible_configs: list[dict[str, Any]] = []
    for threshold in representable_thresholds:
        threshold_key = f"{float(threshold):.3f}"
        values = [row["thresholds"][threshold_key] for row in rows]
        joint_total = sum(int(value["joint_representable_count"]) for value in values)
        relevant_images = sum(
            int(value["joint_representable_count"] > 0) for value in values
        )
        threshold_summary: dict[str, Any] = {
            "individual_representable_gt": sum(
                int(value["individual_representable_count"]) for value in values
            ),
            "deployable_individual_gt": sum(
                int(value["deployable_individual_count"]) for value in values
            ),
            "jointly_representable_gt": joint_total,
            "vanilla_hungarian_gt": sum(
                int(value["vanilla_hungarian_count"]) for value in values
            ),
            "individual_to_joint_collision_loss": sum(
                int(value["individual_to_joint_collision_loss"]) for value in values
            ),
            "images_with_individual_to_joint_collision_loss": sum(
                int(value["individual_to_joint_collision_loss"] > 0)
                for value in values
            ),
            "vanilla_to_cardinality_first_loss": sum(
                int(value["vanilla_to_cardinality_first_loss"]) for value in values
            ),
            "images_where_vanilla_loses_cardinality": sum(
                int(value["vanilla_to_cardinality_first_loss"] > 0)
                for value in values
            ),
            "joint_q_best": value_summary(
                number for value in values for number in value["joint_q_best"]
            ),
            "joint_q_second": value_summary(
                number for value in values for number in value["joint_q_second"]
            ),
            "joint_q_best_minus_second": value_summary(
                number
                for value in values
                for number in value["joint_q_best_minus_second"]
            ),
            "grids": {},
        }
        grid_keys = sorted(values[0]["grids"]) if values else []
        for grid_key in grid_keys:
            grid_rows = [value["grids"][grid_key] for value in values]
            template = grid_rows[0]
            cluster_total = sum(int(row["cluster_count"]) for row in grid_rows)
            support_matches = sum(
                int(row["joint_support_matched_count"]) for row in grid_rows
            )
            relevant_grid_rows = [row for row in grid_rows if row["cluster_count"] > 0]
            full_coverage_images = sum(
                int(row["full_joint_support_coverage"])
                for row in relevant_grid_rows
            )
            supported_candidates = sum(
                int(row["supported_candidate_count"]) for row in grid_rows
            )
            multi_cluster_candidates = sum(
                int(row["multi_cluster_candidate_count"]) for row in grid_rows
            )
            gt_pairs = sum(int(row["gt_pair_count"]) for row in grid_rows)
            intersecting_pairs = sum(
                int(row["intersecting_gt_pair_count"]) for row in grid_rows
            )
            expected_collision_sum = sum(
                float(row["expected_pair_collision_sum"]) for row in grid_rows
            )
            support_sizes = [
                number for row in grid_rows for number in row["support_sizes"]
            ]
            below_mass = [
                number
                for row in grid_rows
                for number in row[
                    "target_mass_at_or_below_representable_threshold"
                ]
            ]
            expected_quality = [
                number
                for row in grid_rows
                for number in row["target_expected_quality"]
            ]
            expected_regret = [
                number
                for row in grid_rows
                for number in row["target_expected_regret"]
            ]
            grid_summary = {
                "parameters": {
                    "representable_threshold": template[
                        "representable_threshold"
                    ],
                    "support_min": template["support_min"],
                    "quality_delta": template["quality_delta"],
                    "temperature": template["temperature"],
                },
                "clusters": cluster_total,
                "support_size": value_summary(support_sizes),
                "clusters_with_multiple_candidates_fraction": _safe_ratio(
                    sum(int(size > 1) for size in support_sizes), cluster_total
                ),
                "target_entropy": value_summary(
                    number for row in grid_rows for number in row["target_entropy"]
                ),
                "target_normalized_entropy": value_summary(
                    number
                    for row in grid_rows
                    for number in row["target_normalized_entropy"]
                ),
                "target_expected_quality": value_summary(expected_quality),
                "target_expected_regret": value_summary(expected_regret),
                "target_mass_at_or_below_representable_threshold": value_summary(
                    below_mass
                ),
                "clusters_with_subthreshold_target_mass_fraction": _safe_ratio(
                    sum(int(mass > 1e-8) for mass in below_mass), cluster_total
                ),
                "support_cutoff": value_summary(
                    number for row in grid_rows for number in row["support_cutoffs"]
                ),
                "joint_support_matched_fraction": _safe_ratio(
                    support_matches, cluster_total
                ),
                "full_joint_support_coverage_rate_on_relevant_images": _safe_ratio(
                    full_coverage_images, relevant_images
                ),
                "images_with_multi_cluster_candidate_overlap": sum(
                    int(row["multi_cluster_candidate_count"] > 0)
                    for row in relevant_grid_rows
                ),
                "multi_cluster_candidate_fraction_among_supported": _safe_ratio(
                    multi_cluster_candidates, supported_candidates
                ),
                "intersecting_gt_pair_fraction": _safe_ratio(
                    intersecting_pairs, gt_pairs
                ),
                "expected_same_candidate_collision_probability_per_gt_pair": (
                    _safe_ratio(expected_collision_sum, gt_pairs)
                ),
                "sole_support_conflicts": sum(
                    int(row["sole_support_conflict_count"]) for row in grid_rows
                ),
            }
            checks = {
                "preserves_every_joint_target": (
                    support_matches == cluster_total
                ),
                "no_subthreshold_target_mass": (
                    grid_summary[
                        "target_mass_at_or_below_representable_threshold"
                    ]["max"]
                    <= 1e-8
                ),
                "mean_expected_quality_above_representable_threshold": (
                    grid_summary["target_expected_quality"]["mean"]
                    > float(threshold)
                ),
                "p90_support_no_more_than_8": (
                    grid_summary["support_size"]["p90"] <= 8.0
                ),
                "median_support_at_least_2": (
                    grid_summary["support_size"]["p50"] >= 2.0
                ),
            }
            grid_summary["preflight_checks"] = checks
            grid_summary["eligible_for_teacher_only_gate"] = all(checks.values())
            if grid_summary["eligible_for_teacher_only_gate"]:
                eligible_configs.append(
                    {
                        "key": grid_key,
                        "parameters": grid_summary["parameters"],
                        "soft_cluster_fraction": grid_summary[
                            "clusters_with_multiple_candidates_fraction"
                        ],
                        "mean_expected_regret": grid_summary[
                            "target_expected_regret"
                        ]["mean"],
                        "expected_pair_collision_probability": grid_summary[
                            "expected_same_candidate_collision_probability_per_gt_pair"
                        ],
                        "requires_collision_safe_teacher_sampling": (
                            grid_summary[
                                "multi_cluster_candidate_fraction_among_supported"
                            ]
                            > 0.0
                        ),
                    }
                )
            threshold_summary["grids"][grid_key] = grid_summary
        summary["thresholds"][threshold_key] = threshold_summary

    eligible_configs.sort(
        key=lambda row: (
            -float(row["soft_cluster_fraction"]),
            float(row["mean_expected_regret"]),
            float(row["expected_pair_collision_probability"]),
        )
    )
    summary["teacher_only_gate_screening"] = {
        "eligible_config_count": len(eligible_configs),
        "eligible_configs": eligible_configs,
        "provisional_preferred": eligible_configs[0] if eligible_configs else None,
        "warning": (
            "The preferred row is a mechanical preflight shortlist, not an "
            "authorization to train. Inspect support/collision statistics first."
        ),
    }
    return summary


def _stage_name(record: dict[str, Any]) -> str:
    stages = record["stages"]
    for name in ("final", "stage2", "main"):
        if name in stages:
            return name
    raise KeyError("candidate cache record has no final/stage2/main stage")


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"iter_(\d+)", path.stem)
    return int(match.group(1)) if match else 0


@torch.no_grad()
def main() -> None:
    args = parse_args()
    representable_thresholds = tuple(
        sorted(set(float(value) for value in args.representable_thresholds))
    )
    support_mins = tuple(sorted(set(float(value) for value in args.support_mins)))
    quality_deltas = tuple(sorted(set(float(value) for value in args.quality_deltas)))
    temperatures = tuple(sorted(set(float(value) for value in args.temperatures)))
    if not representable_thresholds:
        raise ValueError("at least one representable threshold is required")
    if not support_mins or not quality_deltas or not temperatures:
        raise ValueError("support-min, quality-delta and temperature grids are required")
    if any(value <= 0.0 for value in temperatures):
        raise ValueError("all cluster temperatures must be positive")
    if int(args.top_k) < 1:
        raise ValueError("top-k must be positive")

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    cache = load_or_collect_cache(
        config_path,
        checkpoint_path,
        split=str(args.split),
        dataset_root=args.dataset_root or None,
        device=str(args.device),
        cache_dir=args.cache_dir,
        reuse_cache=True,
        require_cache=bool(args.require_cache),
        max_batches=int(args.max_batches),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        sample_strategy=str(args.sample_strategy),
        desc="V4.5 cluster-support preflight",
    )
    rows: list[dict[str, Any]] = []
    for record in cache["records"]:
        stage_name = _stage_name(record)
        stage = record["stages"][stage_name]
        quality, _valid_gt, candidate_valid = proposal_gt_iou_matrix(
            stage,
            record["target"],
            input_h=int(cache["metadata"].get("input_h", 288)),
            input_w=int(cache["metadata"].get("input_w", 800)),
            line_width=float(args.line_width),
            min_valid_rows=int(args.min_valid_rows),
            row_visibility_thresh=0.0,
        )
        rows.append(
            analyze_cluster_support_image(
                quality,
                candidate_valid,
                representable_thresholds=representable_thresholds,
                support_mins=support_mins,
                quality_deltas=quality_deltas,
                temperatures=temperatures,
                top_k=int(args.top_k),
            )
        )

    report = {
        "experiment": "V4.5 GT-cluster soft-target support preflight",
        "diagnostic_only": True,
        "training_started": False,
        "provenance": {
            "git_commit": _git_head(),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_iteration_from_filename": _checkpoint_iteration(
                checkpoint_path
            ),
            "cache": cache["metadata"].get("cache_path", ""),
            "cache_version": cache.get("cache_version"),
        },
        "sample": {
            "split": str(args.split),
            "records": len(rows),
            "max_batches": int(args.max_batches),
            "eval_batch_size": int(args.eval_batch_size),
            "sample_strategy": str(args.sample_strategy),
            "sampled_dataset_indices": cache["metadata"].get(
                "sampled_dataset_indices", []
            ),
        },
        "contract_grid": {
            "representable_thresholds": list(representable_thresholds),
            "support_mins": list(support_mins),
            "quality_deltas": list(quality_deltas),
            "temperatures": list(temperatures),
            "top_k": int(args.top_k),
            "strict_evaluator_boundary": "IoU > representable_threshold",
            "support_boundary": "IoU >= max(support_min, q_best - quality_delta)",
            "joint_representability": (
                "lexicographic maximum cardinality first, detached IoU second"
            ),
        },
        "analysis": summarize_cluster_support(
            rows,
            representable_thresholds=representable_thresholds,
        ),
        "decision_rule": (
            "Do not start V4.5 from this report alone. Prefer support with no "
            "sub-threshold mass and preserved joint cardinality; if supports "
            "overlap across GTs, use collision-safe one-to-one teacher sampling "
            "instead of independent categorical samples."
        ),
        "warning": (
            "This uniform subset is a training-contract diagnostic, not a CULane "
            "benchmark result and not a license to tune the observed test split."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
