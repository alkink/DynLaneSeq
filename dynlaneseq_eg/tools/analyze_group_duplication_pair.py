from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    cardinality_oracle_assignment,
    diagnostic_iou_matrix,
    recall_from_ids,
    stage_scores,
    trace_postprocess,
)
from dynlaneseq_eg.tools.summarize_nms_ranking_pair import _comparability, _common_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether Top-K duplication occurs within or across contiguous "
            "query blocks, reusing caches from analyze_oracle_topk."
        )
    )
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--base-cache", default="")
    parser.add_argument("--candidate-cache", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-query-blocks", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quality-power", type=float, default=0.5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_cache(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_cache_path(report: dict[str, Any], override: str, report_path: str) -> Path:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    metadata_path = report.get("metadata", {}).get("cache_path")
    if metadata_path:
        candidates.append(Path(str(metadata_path)).expanduser())
        candidates.append(Path(report_path).resolve().parent / "cache" / Path(str(metadata_path)).name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
    raise FileNotFoundError(
        "Could not locate the cached model outputs. Checked:\n"
        f"{rendered}\n"
        "Pass --base-cache/--candidate-cache explicitly if the output directory moved."
    )


def _rank_ids(
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    allowed_ids: list[int],
    top_k: int,
) -> list[int]:
    ids = [idx for idx in allowed_ids if bool(candidate_valid[idx])]
    ids.sort(key=lambda idx: float(scores[idx]), reverse=True)
    return ids[: int(top_k)] if top_k > 0 else ids


def _block_ids(num_candidates: int, num_blocks: int) -> list[list[int]]:
    if num_blocks < 1:
        raise ValueError("num_query_blocks must be >= 1")
    if num_candidates % num_blocks != 0:
        raise ValueError(
            f"num_candidates={num_candidates} is not divisible by num_query_blocks={num_blocks}"
        )
    block_size = num_candidates // num_blocks
    return [
        list(range(block_index * block_size, (block_index + 1) * block_size))
        for block_index in range(num_blocks)
    ]


def _empty_recall_counter() -> dict[str, int]:
    return {"hits": 0, "gt": 0}


def _update_recall(
    counter: dict[str, int],
    iou: torch.Tensor,
    proposal_ids: list[int],
    threshold: float,
) -> None:
    hits, gt, _best = recall_from_ids(iou, proposal_ids, threshold)
    counter["hits"] += int(hits)
    counter["gt"] += int(gt)


def _finish_recall(counter: dict[str, int]) -> dict[str, int | float]:
    gt = int(counter["gt"])
    return {
        "hits": int(counter["hits"]),
        "gt": gt,
        "recall": 0.0 if gt == 0 else float(counter["hits"]) / float(gt),
    }


def _add_matrix(target: list[list[int]], source: list[list[int]]) -> None:
    for row_index, row in enumerate(source):
        for column_index, value in enumerate(row):
            target[row_index][column_index] += int(value)


def _pairwise_close_counts(
    stage: dict[str, torch.Tensor],
    *,
    input_h: int,
    input_w: int,
    block_ids: list[list[int]],
    min_valid_rows: int,
    row_visibility_thresh: float,
    distance_threshold: float,
    min_overlap_points: int,
) -> dict[str, Any]:
    pred_x, masks, candidate_valid = candidate_row_masks(
        stage,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
    )
    num_blocks = len(block_ids)
    candidate_to_block = {
        candidate_id: block_index
        for block_index, ids in enumerate(block_ids)
        for candidate_id in ids
    }
    comparable = [[0 for _ in range(num_blocks)] for _ in range(num_blocks)]
    close = [[0 for _ in range(num_blocks)] for _ in range(num_blocks)]
    same_comparable = 0
    same_close = 0
    cross_comparable = 0
    cross_close = 0

    valid_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten().tolist()
    for offset, left in enumerate(valid_ids):
        for right in valid_ids[offset + 1 :]:
            overlap_mask = masks[left] & masks[right]
            overlap = int(overlap_mask.sum())
            if overlap < int(min_overlap_points):
                continue
            distance = float((pred_x[left, overlap_mask] - pred_x[right, overlap_mask]).abs().mean())
            left_block = candidate_to_block[left]
            right_block = candidate_to_block[right]
            comparable[left_block][right_block] += 1
            if left_block != right_block:
                comparable[right_block][left_block] += 1
                cross_comparable += 1
            else:
                same_comparable += 1
            if distance < float(distance_threshold):
                close[left_block][right_block] += 1
                if left_block != right_block:
                    close[right_block][left_block] += 1
                    cross_close += 1
                else:
                    same_close += 1
    return {
        "comparable_pair_matrix": comparable,
        "close_pair_matrix": close,
        "same_block_comparable_pairs": same_comparable,
        "same_block_close_pairs": same_close,
        "cross_block_comparable_pairs": cross_comparable,
        "cross_block_close_pairs": cross_close,
    }


def _config_group_semantics(report: dict[str, Any], num_query_blocks: int) -> dict[str, Any]:
    config_path = Path(str(report.get("metadata", {}).get("config", ""))).expanduser()
    if not config_path.is_file():
        return {
            "config_available": False,
            "model_num_groups": None,
            "matcher_num_groups": None,
            "query_block_interpretation": "contiguous_query_blocks",
        }
    cfg = load_config(config_path)
    model_groups = int(
        cfg.get("model", {}).get("structured_query", {}).get("num_groups", 1)
    )
    matcher_groups = int(cfg.get("matcher", {}).get("num_groups", model_groups))
    exact_assignment_groups = (
        model_groups == int(num_query_blocks)
        and matcher_groups == int(num_query_blocks)
    )
    return {
        "config_available": True,
        "model_num_groups": model_groups,
        "matcher_num_groups": matcher_groups,
        "query_block_interpretation": (
            "exact_assignment_groups"
            if exact_assignment_groups
            else "historical_contiguous_query_blocks_not_current_assignment_groups"
        ),
    }


def _analyze_arm(
    report: dict[str, Any],
    cache: dict[str, Any],
    *,
    report_path: str,
    cache_path: Path,
    stage_name: str,
    num_query_blocks: int,
    top_k: int,
    quality_power: float,
    iou_thresholds: list[float],
    line_width: float,
    min_valid_rows: int,
    row_visibility_thresh: float,
    nms_distance_thresh_px: float,
    nms_min_overlap_points: int,
) -> dict[str, Any]:
    records = cache.get("records", [])
    if not records:
        raise ValueError(f"Cache contains no records: {cache_path}")
    first_stage = records[0].get("stages", {}).get(stage_name)
    if first_stage is None:
        raise KeyError(f"Stage {stage_name!r} is missing from cache {cache_path}")
    num_candidates = int(first_stage["pred_x_rows"].shape[0])
    blocks = _block_ids(num_candidates, num_query_blocks)
    block_size = len(blocks[0])
    input_h = int(cache["metadata"]["input_h"])
    input_w = int(cache["metadata"]["input_w"])

    all_capacity = {threshold: _empty_recall_counter() for threshold in iou_thresholds}
    oracle_topk = {threshold: _empty_recall_counter() for threshold in iou_thresholds}
    raw_topk = {threshold: _empty_recall_counter() for threshold in iou_thresholds}
    nms_topk = {threshold: _empty_recall_counter() for threshold in iou_thresholds}
    block_capacity = {
        threshold: [_empty_recall_counter() for _ in blocks]
        for threshold in iou_thresholds
    }
    block_oracle_topk = {
        threshold: [_empty_recall_counter() for _ in blocks]
        for threshold in iou_thresholds
    }
    block_raw_topk = {
        threshold: [_empty_recall_counter() for _ in blocks]
        for threshold in iou_thresholds
    }
    block_nms_topk = {
        threshold: [_empty_recall_counter() for _ in blocks]
        for threshold in iou_thresholds
    }
    support_histograms = {
        threshold: [0 for _ in range(num_query_blocks + 1)]
        for threshold in iou_thresholds
    }
    exclusive_geometry_hits = {
        threshold: [0 for _ in blocks] for threshold in iou_thresholds
    }

    raw_origin_counts = [0 for _ in blocks]
    nms_origin_counts = [0 for _ in blocks]
    raw_selected = 0
    nms_selected = 0
    raw_distinct_blocks = 0
    nms_distinct_blocks = 0
    raw_distinct_block_histogram = [0 for _ in range(num_query_blocks + 1)]
    nms_distinct_block_histogram = [0 for _ in range(num_query_blocks + 1)]
    raw_ids_removed_by_nms = 0
    eligible_candidates = 0
    nms_kept_candidates = 0
    nms_removed_candidates = 0
    suppression_matrix = [[0 for _ in blocks] for _ in blocks]
    pair_comparable = [[0 for _ in blocks] for _ in blocks]
    pair_close = [[0 for _ in blocks] for _ in blocks]
    same_comparable = 0
    same_close = 0
    cross_comparable = 0
    cross_close = 0

    for record in records:
        stage = record["stages"].get(stage_name)
        if stage is None:
            continue
        if int(stage["pred_x_rows"].shape[0]) != num_candidates:
            raise ValueError("Candidate count changes across cached records")
        iou, _valid_gt, candidate_valid = diagnostic_iou_matrix(
            record,
            stage_name,
            use_official=False,
            input_h=input_h,
            input_w=input_w,
            line_width=line_width,
            min_valid_rows=min_valid_rows,
            row_visibility_thresh=row_visibility_thresh,
        )
        scores = stage_scores(stage, quality_power=quality_power)
        all_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten().tolist()
        raw_ids = _rank_ids(scores, candidate_valid, all_ids, top_k)
        global_trace = trace_postprocess(
            stage,
            input_h=input_h,
            input_w=input_w,
            score_thresh=-1.0,
            quality_power=quality_power,
            min_valid_rows=min_valid_rows,
            nms_distance_thresh_px=nms_distance_thresh_px,
            nms_min_overlap_points=nms_min_overlap_points,
            top_k=top_k,
            row_visibility_thresh=row_visibility_thresh,
        )
        selected_ids = list(global_trace["selected_ids"])
        candidate_to_block = {
            candidate_id: block_index
            for block_index, ids in enumerate(blocks)
            for candidate_id in ids
        }
        raw_blocks = [candidate_to_block[index] for index in raw_ids]
        selected_blocks = [candidate_to_block[index] for index in selected_ids]
        for block_index in raw_blocks:
            raw_origin_counts[block_index] += 1
        for block_index in selected_blocks:
            nms_origin_counts[block_index] += 1
        raw_selected += len(raw_ids)
        nms_selected += len(selected_ids)
        raw_distinct = len(set(raw_blocks))
        nms_distinct = len(set(selected_blocks))
        raw_distinct_blocks += raw_distinct
        nms_distinct_blocks += nms_distinct
        raw_distinct_block_histogram[raw_distinct] += 1
        nms_distinct_block_histogram[nms_distinct] += 1
        raw_ids_removed_by_nms += sum(
            1 for proposal_id in raw_ids if global_trace["status"][proposal_id] == "nms_removed"
        )
        eligible_candidates += len(global_trace["eligible_ids"])
        nms_kept_candidates += len(global_trace["nms_kept_ids"])
        nms_removed_candidates += len(global_trace["suppressed_by"])
        for removed, keeper in global_trace["suppressed_by"].items():
            suppression_matrix[candidate_to_block[int(removed)]][
                candidate_to_block[int(keeper)]
            ] += 1

        per_block_raw: list[list[int]] = []
        per_block_nms: list[list[int]] = []
        for block in blocks:
            per_block_raw.append(_rank_ids(scores, candidate_valid, block, top_k))
            block_trace = trace_postprocess(
                stage,
                input_h=input_h,
                input_w=input_w,
                score_thresh=-1.0,
                quality_power=quality_power,
                min_valid_rows=min_valid_rows,
                nms_distance_thresh_px=nms_distance_thresh_px,
                nms_min_overlap_points=nms_min_overlap_points,
                top_k=top_k,
                row_visibility_thresh=row_visibility_thresh,
                allowed_ids=block,
            )
            per_block_nms.append(list(block_trace["selected_ids"]))

        for threshold in iou_thresholds:
            all_selection = cardinality_oracle_assignment(
                iou, threshold, max(num_candidates, int(iou.shape[0])), candidate_valid
            )
            topk_selection = cardinality_oracle_assignment(
                iou, threshold, top_k, candidate_valid
            )
            _update_recall(all_capacity[threshold], iou, list(all_selection.proposal_ids), threshold)
            _update_recall(oracle_topk[threshold], iou, list(topk_selection.proposal_ids), threshold)
            _update_recall(raw_topk[threshold], iou, raw_ids, threshold)
            _update_recall(nms_topk[threshold], iou, selected_ids, threshold)

            group_geometry_sets: list[set[int]] = []
            for block_index, block in enumerate(blocks):
                block_mask = torch.zeros_like(candidate_valid)
                block_mask[block] = candidate_valid[block]
                capacity_selection = cardinality_oracle_assignment(
                    iou, threshold, block_size, block_mask
                )
                topk_block_selection = cardinality_oracle_assignment(
                    iou, threshold, top_k, block_mask
                )
                _update_recall(
                    block_capacity[threshold][block_index],
                    iou,
                    list(capacity_selection.proposal_ids),
                    threshold,
                )
                _update_recall(
                    block_oracle_topk[threshold][block_index],
                    iou,
                    list(topk_block_selection.proposal_ids),
                    threshold,
                )
                _update_recall(
                    block_raw_topk[threshold][block_index],
                    iou,
                    per_block_raw[block_index],
                    threshold,
                )
                _update_recall(
                    block_nms_topk[threshold][block_index],
                    iou,
                    per_block_nms[block_index],
                    threshold,
                )
                if iou.shape[0] == 0:
                    group_geometry_sets.append(set())
                else:
                    group_geometry_sets.append(
                        {
                            gt_index
                            for gt_index in range(int(iou.shape[0]))
                            if bool((iou[gt_index, block] > float(threshold)).any())
                        }
                    )
            for gt_index in range(int(iou.shape[0])):
                supporters = [
                    block_index
                    for block_index, hit_set in enumerate(group_geometry_sets)
                    if gt_index in hit_set
                ]
                support_histograms[threshold][len(supporters)] += 1
                if len(supporters) == 1:
                    exclusive_geometry_hits[threshold][supporters[0]] += 1

        pair_counts = _pairwise_close_counts(
            stage,
            input_h=input_h,
            input_w=input_w,
            block_ids=blocks,
            min_valid_rows=min_valid_rows,
            row_visibility_thresh=row_visibility_thresh,
            distance_threshold=nms_distance_thresh_px,
            min_overlap_points=nms_min_overlap_points,
        )
        _add_matrix(pair_comparable, pair_counts["comparable_pair_matrix"])
        _add_matrix(pair_close, pair_counts["close_pair_matrix"])
        same_comparable += int(pair_counts["same_block_comparable_pairs"])
        same_close += int(pair_counts["same_block_close_pairs"])
        cross_comparable += int(pair_counts["cross_block_comparable_pairs"])
        cross_close += int(pair_counts["cross_block_close_pairs"])

    images = len(records)
    threshold_output: dict[str, Any] = {}
    for threshold in iou_thresholds:
        support_hist = support_histograms[threshold]
        recoverable = sum(support_hist[1:])
        total_supports = sum(index * count for index, count in enumerate(support_hist))
        threshold_output[f"{threshold:.2f}"] = {
            "all_candidates_capacity": _finish_recall(all_capacity[threshold]),
            "oracle_topk_capacity": _finish_recall(oracle_topk[threshold]),
            "raw_model_topk": _finish_recall(raw_topk[threshold]),
            "nms_model_topk": _finish_recall(nms_topk[threshold]),
            "per_query_block": [
                {
                    "query_block": block_index,
                    "candidate_ids": blocks[block_index],
                    "all_block_candidates": _finish_recall(
                        block_capacity[threshold][block_index]
                    ),
                    "oracle_topk": _finish_recall(
                        block_oracle_topk[threshold][block_index]
                    ),
                    "raw_model_topk": _finish_recall(
                        block_raw_topk[threshold][block_index]
                    ),
                    "nms_model_topk": _finish_recall(
                        block_nms_topk[threshold][block_index]
                    ),
                    "exclusive_geometry_hits": int(
                        exclusive_geometry_hits[threshold][block_index]
                    ),
                }
                for block_index in range(num_query_blocks)
            ],
            "geometry_supporting_block_histogram": {
                str(index): int(count) for index, count in enumerate(support_hist)
            },
            "geometry_recoverable_lanes": int(recoverable),
            "mean_supporting_blocks_per_recoverable_lane": (
                0.0 if recoverable == 0 else float(total_supports) / float(recoverable)
            ),
        }

    suppression_same = sum(
        suppression_matrix[index][index] for index in range(num_query_blocks)
    )
    suppression_total = sum(sum(row) for row in suppression_matrix)
    suppression_cross = suppression_total - suppression_same
    return {
        "report": str(Path(report_path)),
        "cache": str(cache_path),
        "config": report.get("metadata", {}).get("config"),
        "checkpoint": report.get("metadata", {}).get("checkpoint"),
        "images": images,
        "num_candidates": num_candidates,
        "num_query_blocks": num_query_blocks,
        "query_block_size": block_size,
        "group_semantics": _config_group_semantics(report, num_query_blocks),
        "thresholds": threshold_output,
        "selection_and_nms": {
            "quality_power": float(quality_power),
            "top_k": int(top_k),
            "raw_topk_origin_counts": raw_origin_counts,
            "nms_topk_origin_counts": nms_origin_counts,
            "raw_topk_origin_fractions": [
                0.0 if raw_selected == 0 else float(count) / float(raw_selected)
                for count in raw_origin_counts
            ],
            "nms_topk_origin_fractions": [
                0.0 if nms_selected == 0 else float(count) / float(nms_selected)
                for count in nms_origin_counts
            ],
            "raw_topk_selected_total": raw_selected,
            "nms_topk_selected_total": nms_selected,
            "raw_topk_distinct_block_histogram": {
                str(index): int(count)
                for index, count in enumerate(raw_distinct_block_histogram)
            },
            "nms_topk_distinct_block_histogram": {
                str(index): int(count)
                for index, count in enumerate(nms_distinct_block_histogram)
            },
            "mean_distinct_query_blocks_raw_topk": (
                0.0 if images == 0 else float(raw_distinct_blocks) / float(images)
            ),
            "mean_distinct_query_blocks_nms_topk": (
                0.0 if images == 0 else float(nms_distinct_blocks) / float(images)
            ),
            "raw_topk_ids_removed_by_global_nms": raw_ids_removed_by_nms,
            "raw_topk_nms_removal_fraction": (
                0.0 if raw_selected == 0 else float(raw_ids_removed_by_nms) / float(raw_selected)
            ),
            "eligible_candidates": eligible_candidates,
            "nms_kept_candidates": nms_kept_candidates,
            "nms_removed_candidates": nms_removed_candidates,
            "nms_suppression_removed_block_by_keeper_block": suppression_matrix,
            "same_block_nms_suppressions": suppression_same,
            "cross_block_nms_suppressions": suppression_cross,
            "cross_block_nms_suppression_fraction": (
                0.0
                if suppression_total == 0
                else float(suppression_cross) / float(suppression_total)
            ),
        },
        "pairwise_geometry": {
            "definition": (
                f"mean row distance < {nms_distance_thresh_px:g}px with at least "
                f"{nms_min_overlap_points} overlapping rows"
            ),
            "comparable_pair_matrix": pair_comparable,
            "close_pair_matrix": pair_close,
            "same_block_comparable_pairs": same_comparable,
            "same_block_close_pairs": same_close,
            "same_block_close_rate": (
                0.0 if same_comparable == 0 else float(same_close) / float(same_comparable)
            ),
            "cross_block_comparable_pairs": cross_comparable,
            "cross_block_close_pairs": cross_close,
            "cross_block_close_rate": (
                0.0 if cross_comparable == 0 else float(cross_close) / float(cross_comparable)
            ),
            "cross_block_share_of_all_close_pairs": (
                0.0
                if same_close + cross_close == 0
                else float(cross_close) / float(same_close + cross_close)
            ),
        },
    }


def _paired_summary(
    base: dict[str, Any],
    candidate: dict[str, Any],
    iou_thresholds: list[float],
) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    for threshold in iou_thresholds:
        key = f"{threshold:.2f}"
        base_threshold = base["thresholds"][key]
        candidate_threshold = candidate["thresholds"][key]
        thresholds[key] = {
            metric: {
                "base_recall": float(base_threshold[metric]["recall"]),
                "candidate_recall": float(candidate_threshold[metric]["recall"]),
                "delta_recall_points": 100.0
                * (
                    float(candidate_threshold[metric]["recall"])
                    - float(base_threshold[metric]["recall"])
                ),
            }
            for metric in (
                "all_candidates_capacity",
                "oracle_topk_capacity",
                "raw_model_topk",
                "nms_model_topk",
            )
        }
    return {
        "thresholds": thresholds,
        "base_cross_block_nms_suppression_fraction": base["selection_and_nms"][
            "cross_block_nms_suppression_fraction"
        ],
        "candidate_cross_block_nms_suppression_fraction": candidate["selection_and_nms"][
            "cross_block_nms_suppression_fraction"
        ],
        "base_mean_distinct_blocks_raw_topk": base["selection_and_nms"][
            "mean_distinct_query_blocks_raw_topk"
        ],
        "base_mean_distinct_blocks_nms_topk": base["selection_and_nms"][
            "mean_distinct_query_blocks_nms_topk"
        ],
        "candidate_mean_distinct_blocks_raw_topk": candidate["selection_and_nms"][
            "mean_distinct_query_blocks_raw_topk"
        ],
        "candidate_mean_distinct_blocks_nms_topk": candidate["selection_and_nms"][
            "mean_distinct_query_blocks_nms_topk"
        ],
    }


def _print_arm(name: str, arm: dict[str, Any]) -> None:
    semantics = arm["group_semantics"]["query_block_interpretation"]
    print(f"\n{name}: {semantics}")
    selection = arm["selection_and_nms"]
    print(
        "  distinct blocks in Top-K: "
        f"raw={selection['mean_distinct_query_blocks_raw_topk']:.2f}, "
        f"after_nms={selection['mean_distinct_query_blocks_nms_topk']:.2f}"
    )
    print(
        "  NMS suppressions: "
        f"same-block={selection['same_block_nms_suppressions']}, "
        f"cross-block={selection['cross_block_nms_suppressions']} "
        f"({100.0 * selection['cross_block_nms_suppression_fraction']:.1f}% cross)"
    )
    for threshold, values in arm["thresholds"].items():
        print(
            f"  IoU {threshold}: all={100.0 * values['all_candidates_capacity']['recall']:.2f}, "
            f"oracleK={100.0 * values['oracle_topk_capacity']['recall']:.2f}, "
            f"rawK={100.0 * values['raw_model_topk']['recall']:.2f}, "
            f"nmsK={100.0 * values['nms_model_topk']['recall']:.2f}"
        )
        block_values = ", ".join(
            f"b{row['query_block']}={100.0 * row['all_block_candidates']['recall']:.2f}"
            for row in values["per_query_block"]
        )
        print(f"    block capacities: {block_values}")


def main() -> None:
    args = parse_args()
    base_report = _load_json(args.base_report)
    candidate_report = _load_json(args.candidate_report)
    comparability = _comparability(base_report, candidate_report)
    if not comparability["all_checks_pass"]:
        raise ValueError(f"Input reports are not directly comparable: {comparability}")
    stage_name = _common_stage(base_report, candidate_report)
    base_cache_path = _resolve_cache_path(
        base_report, args.base_cache, args.base_report
    )
    candidate_cache_path = _resolve_cache_path(
        candidate_report, args.candidate_cache, args.candidate_report
    )
    base_cache = _load_cache(base_cache_path)
    candidate_cache = _load_cache(candidate_cache_path)
    cache_checks = {
        "list_sha256": base_cache.get("metadata", {}).get("list_sha256")
        == candidate_cache.get("metadata", {}).get("list_sha256"),
        "sample_strategy": base_cache.get("metadata", {}).get("sample_strategy")
        == candidate_cache.get("metadata", {}).get("sample_strategy"),
        "sampled_dataset_indices": base_cache.get("metadata", {}).get(
            "sampled_dataset_indices"
        )
        == candidate_cache.get("metadata", {}).get("sampled_dataset_indices"),
        "num_records": len(base_cache.get("records", []))
        == len(candidate_cache.get("records", [])),
    }
    if not all(cache_checks.values()):
        raise ValueError(f"Cached records are not paired: {cache_checks}")

    common = {
        "stage_name": stage_name,
        "num_query_blocks": int(args.num_query_blocks),
        "top_k": int(args.top_k),
        "quality_power": float(args.quality_power),
        "iou_thresholds": [float(value) for value in args.iou_thresholds],
        "line_width": float(args.line_width),
        "min_valid_rows": int(args.min_valid_rows),
        "row_visibility_thresh": float(args.row_visibility_thresh),
        "nms_distance_thresh_px": float(args.nms_distance_thresh_px),
        "nms_min_overlap_points": int(args.nms_min_overlap_points),
    }
    base = _analyze_arm(
        base_report,
        base_cache,
        report_path=args.base_report,
        cache_path=base_cache_path,
        **common,
    )
    candidate = _analyze_arm(
        candidate_report,
        candidate_cache,
        report_path=args.candidate_report,
        cache_path=candidate_cache_path,
        **common,
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "Row-space query-block diagnostic. Query blocks are exact assignment "
            "groups only when the resolved model and matcher num_groups equal "
            "num_query_blocks."
        ),
        "comparability": {
            "report_checks": comparability,
            "cache_checks": cache_checks,
        },
        "settings": common,
        "base": base,
        "candidate": candidate,
        "paired_summary": _paired_summary(
            base, candidate, [float(value) for value in args.iou_thresholds]
        ),
    }
    _print_arm("base", base)
    _print_arm("candidate", candidate)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\noutput_json: {output_path}")


if __name__ == "__main__":
    main()
