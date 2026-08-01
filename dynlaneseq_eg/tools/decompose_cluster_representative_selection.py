from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    trace_postprocess,
)
from dynlaneseq_eg.tools.probe_four_slot_coverage_selector import (
    _load_cache,
    _load_source_selector,
    _validate_cache,
    source_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose frozen Top-4 selection loss into NMS-cluster ranking, "
            "within-cluster representative ranking, and clustering loss."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--reference-report", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--input-h", type=int, default=640)
    parser.add_argument("--input-w", type=int, default=1600)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def nms_clusters_from_trace(trace: dict[str, Any]) -> dict[int, list[int]]:
    """Return the exact greedy-NMS partition keyed by kept representative."""

    clusters = {
        int(keeper): [int(keeper)]
        for keeper in trace["nms_kept_ids"]
    }
    for candidate, keeper in trace["suppressed_by"].items():
        keeper = int(keeper)
        if keeper not in clusters:
            raise ValueError(f"suppression keeper {keeper} was not retained")
        clusters[keeper].append(int(candidate))
    eligible = {int(value) for value in trace["eligible_ids"]}
    partitioned = {
        candidate
        for members in clusters.values()
        for candidate in members
    }
    if eligible != partitioned:
        missing = sorted(eligible - partitioned)
        extra = sorted(partitioned - eligible)
        raise ValueError(
            f"NMS partition mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    return clusters


def _cluster_iou_matrices(
    official_iou: torch.Tensor,
    clusters: dict[int, list[int]],
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    keepers = list(clusters)
    gt_count = int(official_iou.shape[0])
    if not keepers:
        empty = official_iou.new_zeros((gt_count, 0))
        return keepers, empty, empty
    keeper_iou = official_iou[:, keepers]
    cluster_best = torch.stack(
        [
            official_iou[:, members].max(dim=1).values
            for members in clusters.values()
        ],
        dim=1,
    )
    return keepers, keeper_iou, cluster_best


def decompose_one_image(
    official_iou: torch.Tensor,
    *,
    candidate_valid: torch.Tensor,
    clusters: dict[int, list[int]],
    selected_keepers: list[int],
    raw_topk_ids: list[int],
    threshold: float,
    top_k: int,
) -> dict[str, int]:
    """Compute parallel oracle counterfactuals for one image.

    `selected_cluster_oracle_representative` changes only the representative
    inside each source-selected NMS cluster. `oracle_cluster_source_rep`
    changes only which cluster keepers are selected. The full cluster oracle
    changes both while retaining the one-output-per-cluster restriction.
    """

    keepers, keeper_iou, cluster_best = _cluster_iou_matrices(
        official_iou,
        clusters,
    )
    keeper_to_cluster = {
        keeper: cluster_index
        for cluster_index, keeper in enumerate(keepers)
    }
    selected_cluster_indices = [
        keeper_to_cluster[int(keeper)]
        for keeper in selected_keepers
    ]
    if selected_cluster_indices:
        selected_cluster_iou = cluster_best[:, selected_cluster_indices]
    else:
        selected_cluster_iou = cluster_best[:, :0]

    actual_nms = evaluator_hungarian_assignment(
        official_iou,
        selected_keepers,
        threshold=float(threshold),
    ).hit_count
    raw_topk = evaluator_hungarian_assignment(
        official_iou,
        raw_topk_ids,
        threshold=float(threshold),
    ).hit_count
    selected_cluster_oracle = cardinality_oracle_assignment(
        selected_cluster_iou,
        threshold=float(threshold),
        top_k=min(int(top_k), len(selected_cluster_indices)),
    ).hit_count
    oracle_cluster_source_rep = cardinality_oracle_assignment(
        keeper_iou,
        threshold=float(threshold),
        top_k=int(top_k),
    ).hit_count
    oracle_cluster_and_rep = cardinality_oracle_assignment(
        cluster_best,
        threshold=float(threshold),
        top_k=int(top_k),
    ).hit_count
    candidate_oracle = cardinality_oracle_assignment(
        official_iou,
        threshold=float(threshold),
        top_k=int(top_k),
        candidate_valid=candidate_valid,
    ).hit_count
    return {
        "gt_lanes": int(official_iou.shape[0]),
        "raw_topk": int(raw_topk),
        "actual_nms": int(actual_nms),
        "selected_cluster_oracle_representative": int(
            selected_cluster_oracle
        ),
        "oracle_cluster_source_representative": int(
            oracle_cluster_source_rep
        ),
        "oracle_cluster_and_representative": int(oracle_cluster_and_rep),
        "candidate_oracle": int(candidate_oracle),
    }


def _new_counter() -> dict[str, int]:
    return {
        "gt_lanes": 0,
        "raw_topk": 0,
        "actual_nms": 0,
        "selected_cluster_oracle_representative": 0,
        "oracle_cluster_source_representative": 0,
        "oracle_cluster_and_representative": 0,
        "candidate_oracle": 0,
    }


def _add_counter(target: dict[str, int], row: dict[str, int]) -> None:
    for name in target:
        target[name] += int(row[name])


def _finish(counter: dict[str, int]) -> dict[str, Any]:
    gt = int(counter["gt_lanes"])

    def recall(name: str) -> float:
        return float(counter[name]) / float(max(gt, 1))

    modes = {
        name: {
            "tp": int(counter[name]),
            "recall": recall(name),
        }
        for name in counter
        if name != "gt_lanes"
    }
    actual = recall("actual_nms")
    representative_gain = 100.0 * (
        recall("selected_cluster_oracle_representative") - actual
    )
    cluster_ranking_gain = 100.0 * (
        recall("oracle_cluster_source_representative") - actual
    )
    joint_selection_gain = 100.0 * (
        recall("oracle_cluster_and_representative") - actual
    )
    joint_interaction = max(
        0.0,
        joint_selection_gain - max(representative_gain, cluster_ranking_gain),
    )
    clustering_loss = 100.0 * (
        recall("candidate_oracle")
        - recall("oracle_cluster_and_representative")
    )
    residual_candidate_gap = 100.0 * (
        recall("candidate_oracle") - actual
    )
    return {
        "gt_lanes": gt,
        "modes": modes,
        "headroom_points": {
            "within_selected_cluster_representative": representative_gain,
            "cluster_ranking_with_source_representatives": cluster_ranking_gain,
            "joint_cluster_and_representative": joint_selection_gain,
            "joint_interaction_beyond_best_single": joint_interaction,
            "nms_partition_loss": clustering_loss,
            "total_to_candidate_oracle": residual_candidate_gap,
        },
    }


def _dominant_diagnosis(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = {
        "representative": "within_selected_cluster_representative",
        "cluster_ranking": "cluster_ranking_with_source_representatives",
        "joint_interaction": "joint_interaction_beyond_best_single",
        "nms_partition": "nms_partition_loss",
    }
    mean_headroom = {
        label: sum(
            float(row["headroom_points"][metric])
            for row in results.values()
        )
        / float(max(len(results), 1))
        for label, metric in names.items()
    }
    ordered = sorted(mean_headroom, key=mean_headroom.get, reverse=True)
    lead = float(mean_headroom[ordered[0]]) - float(mean_headroom[ordered[1]])
    if lead >= 2.0:
        diagnosis = f"{ordered[0]}_is_primary"
        confidence = "strong"
    else:
        diagnosis = "mixed_cluster_and_representative_failure"
        confidence = "moderate"
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "mean_headroom_points": mean_headroom,
        "note": (
            "Representative and cluster-ranking counterfactuals are parallel "
            "oracles and must not be added together."
        ),
    }


def verify_reference_report(
    threshold_results: dict[str, dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for threshold in (0.5, 0.7):
        key = f"{threshold:.2f}"
        suffix = f"{int(threshold * 100):03d}"
        actual_raw = float(
            threshold_results[key]["modes"]["raw_topk"]["recall"]
        )
        actual_nms = float(
            threshold_results[key]["modes"]["actual_nms"]["recall"]
        )
        expected_raw = float(
            reference["source"]["raw_top4"][f"recall_{suffix}"]
        )
        expected_nms = float(
            reference["source"]["nms_top4"][f"recall_{suffix}"]
        )
        raw_delta = actual_raw - expected_raw
        nms_delta = actual_nms - expected_nms
        comparisons[key] = {
            "raw_recall_delta": raw_delta,
            "nms_recall_delta": nms_delta,
        }
        if abs(raw_delta) > 1e-12 or abs(nms_delta) > 1e-12:
            raise ValueError(
                "decomposition does not reproduce the reference report at "
                f"IoU {key}: raw_delta={raw_delta}, nms_delta={nms_delta}"
            )
    return {
        "matched": True,
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 1 or int(args.top_k) < 1:
        raise ValueError("batch-size and top-k must be positive")
    if float(args.nms_distance) <= 0.0:
        raise ValueError("this decomposition requires positive NMS distance")
    cache = _load_cache(args.val_cache)
    _validate_cache(cache, "validation")
    selector, source_iteration = _load_source_selector(
        args.config,
        args.source_checkpoint,
    )
    feature_dim = int(cache["features"].shape[-1])
    expected_dim = int(selector.input_norm.normalized_shape[0])
    if feature_dim != expected_dim:
        raise ValueError(
            f"selector/cache feature mismatch: {expected_dim} != {feature_dim}"
        )
    device = torch.device(args.device)
    scores = source_scores(
        selector,
        cache,
        device=device,
        batch_size=args.batch_size,
    )
    thresholds = (0.5, 0.7)
    counters = {threshold: _new_counter() for threshold in thresholds}
    cluster_sizes: list[int] = []
    selected_cluster_sizes: list[int] = []
    image_rows: list[dict[str, Any]] = []
    candidate_valid_rows = cache["candidate_valid"]
    for image_index, official_iou in enumerate(cache["official_iou"]):
        stage = {
            name: values[image_index]
            for name, values in cache["stage"].items()
        }
        score = scores[image_index]
        valid = candidate_valid_rows[image_index].bool()
        trace = trace_postprocess(
            stage,
            input_h=args.input_h,
            input_w=args.input_w,
            score_thresh=-1.0,
            quality_power=0.0,
            min_valid_rows=args.min_valid_rows,
            nms_distance_thresh_px=args.nms_distance,
            nms_min_overlap_points=args.nms_min_overlap_points,
            top_k=args.top_k,
            row_visibility_thresh=args.row_visibility_thresh,
            score_override={
                candidate: float(score[candidate])
                for candidate in range(int(score.shape[0]))
            },
        )
        clusters = nms_clusters_from_trace(trace)
        selected = [int(value) for value in trace["selected_ids"]]
        raw_ids = [
            int(value)
            for value in torch.argsort(score, descending=True).tolist()
            if bool(valid[int(value)])
        ][: int(args.top_k)]
        cluster_sizes.extend(len(members) for members in clusters.values())
        selected_cluster_sizes.extend(len(clusters[keeper]) for keeper in selected)
        per_threshold: dict[str, Any] = {}
        for threshold in thresholds:
            row = decompose_one_image(
                official_iou,
                candidate_valid=valid,
                clusters=clusters,
                selected_keepers=selected,
                raw_topk_ids=raw_ids,
                threshold=threshold,
                top_k=args.top_k,
            )
            _add_counter(counters[threshold], row)
            per_threshold[f"{threshold:.2f}"] = row
        image_rows.append(
            {
                "dataset_index": int(
                    cache["metadata"]["dataset_indices"][image_index]
                ),
                "image_path": cache["metadata"]["image_paths"][image_index],
                "valid_candidates": int(valid.sum()),
                "nms_clusters": len(clusters),
                "selected_clusters": len(selected),
                "thresholds": per_threshold,
            }
        )
    threshold_results = {
        f"{threshold:.2f}": _finish(counters[threshold])
        for threshold in thresholds
    }
    consistency: dict[str, Any] | None = None
    if args.reference_report:
        reference_path = Path(args.reference_report)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        consistency = {
            "reference_report": str(reference_path),
            **verify_reference_report(threshold_results, reference),
        }
    result = {
        "diagnostic_only": True,
        "warning": (
            "All oracle modes use GT and are non-deployable. Results localize "
            "selection loss only for this frozen checkpoint and cache."
        ),
        "config": args.config,
        "source_checkpoint": args.source_checkpoint,
        "source_iteration": int(source_iteration),
        "val_cache": args.val_cache,
        "images": int(cache["features"].shape[0]),
        "nms": {
            "distance_px": float(args.nms_distance),
            "min_overlap_points": int(args.nms_min_overlap_points),
            "top_k": int(args.top_k),
            "mean_cluster_size": sum(cluster_sizes)
            / float(max(len(cluster_sizes), 1)),
            "mean_selected_cluster_size": sum(selected_cluster_sizes)
            / float(max(len(selected_cluster_sizes), 1)),
        },
        "consistency": consistency,
        "thresholds": threshold_results,
        "decision": _dominant_diagnosis(threshold_results),
        "per_image": image_rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
