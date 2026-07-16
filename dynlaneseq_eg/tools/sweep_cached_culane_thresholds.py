from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    ensure_official_iou_cache,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    metadata_for_json,
    trace_postprocess,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast official-CULane validation sweep from cached candidates. "
            "The model is forwarded once per checkpoint; all score/quality "
            "combinations reuse the exact same predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default="outputs/diagnostic_cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--no-pretrained-init", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--cache-num-workers", type=int, default=0)
    parser.add_argument("--stage", default="main")
    parser.add_argument(
        "--score-thresholds",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    )
    parser.add_argument(
        "--quality-powers",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
    )
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    )
    parser.add_argument("--selection-iou", type=float, default=0.50)
    parser.add_argument("--tie-break-iou", type=float, default=0.70)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=None)
    parser.add_argument("--nms-min-overlap-points", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", default="")
    return parser.parse_args()


def _metric_from_counts(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = float(tp) / float(tp + fp) if tp + fp > 0 else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def _format_metric(metric: Dict[str, Any]) -> str:
    return f"P={metric['Precision']:.4f} R={metric['Recall']:.4f} F1={metric['F1']:.4f}"


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.2f}"


def _resolve_stage(cache: Dict[str, Any], requested: str) -> str:
    names = sorted({name for record in cache.get("records", []) for name in record.get("stages", {})})
    if not names:
        raise RuntimeError("No prediction stages found in candidate cache.")
    if requested in names:
        return requested
    raise KeyError(f"Requested stage={requested!r} not found. Available stages: {names}")


def _counts_for_selection(
    official_iou: torch.Tensor,
    selected_ids: Sequence[int],
    threshold: float,
) -> Tuple[int, int, int]:
    selected = [int(index) for index in selected_ids]
    assignment = evaluator_hungarian_assignment(official_iou, selected, threshold=float(threshold))
    tp = int(assignment.hit_count)
    fp = max(0, len(selected) - tp)
    fn = max(0, int(official_iou.shape[0]) - tp)
    return tp, fp, fn


def _row_sort_key(
    row: Dict[str, Any],
    selection_iou: float,
    tie_break_iou: float,
) -> Tuple[float, float, float, float]:
    primary = row["metrics"][_threshold_key(selection_iou)]["F1"]
    tie_metric = row["metrics"].get(_threshold_key(tie_break_iou), {"F1": 0.0})["F1"]
    return (
        float(primary),
        float(row["mF1"]),
        float(tie_metric),
        -float(row["score_threshold"]),
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        list_path=args.list_path or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=args.reuse_cache,
        max_batches=args.max_batches,
        eval_batch_size=args.eval_batch_size,
        cache_num_workers=args.cache_num_workers,
        no_pretrained_init=args.no_pretrained_init,
        desc="candidate cache",
    )
    cache = ensure_official_iou_cache(
        cache,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
    )

    metadata = cache["metadata"]
    input_h = int(metadata["input_h"])
    input_w = int(metadata["input_w"])
    post = metadata.get("postprocess", {})
    nms_distance = float(
        args.nms_distance_thresh_px
        if args.nms_distance_thresh_px is not None
        else post.get("lane_nms_distance_thresh_px", 20.0)
    )
    nms_overlap = int(
        args.nms_min_overlap_points
        if args.nms_min_overlap_points is not None
        else post.get("lane_nms_min_overlap_points", 5)
    )
    top_k = int(args.top_k if args.top_k is not None else post.get("top_k", 4))
    stage_name = _resolve_stage(cache, args.stage)
    iou_thresholds = [float(value) for value in args.iou_thresholds]
    score_thresholds = [float(value) for value in args.score_thresholds]
    quality_powers = [float(value) for value in args.quality_powers]

    rows: List[Dict[str, Any]] = []
    for quality_power in quality_powers:
        for score_threshold in score_thresholds:
            counts: Dict[float, List[int]] = {
                threshold: [0, 0, 0] for threshold in iou_thresholds
            }
            image_count = 0
            selected_count = 0
            for record in tqdm(
                cache["records"],
                ncols=80,
                desc=f"{stage_name} q={quality_power:g} thr={score_threshold:g}",
                leave=False,
            ):
                stage = record["stages"].get(stage_name)
                if stage is None:
                    continue
                trace = trace_postprocess(
                    stage,
                    input_h=input_h,
                    input_w=input_w,
                    score_thresh=score_threshold,
                    quality_power=quality_power,
                    min_valid_rows=args.min_valid_rows,
                    nms_distance_thresh_px=nms_distance,
                    nms_min_overlap_points=nms_overlap,
                    top_k=top_k,
                    row_visibility_thresh=args.row_visibility_thresh,
                )
                selected_ids = list(trace["selected_ids"])
                selected_count += len(selected_ids)
                official_iou = stage.get("official_iou")
                if official_iou is None:
                    raise RuntimeError("Official IoU cache is missing.")
                for threshold in iou_thresholds:
                    tp, fp, fn = _counts_for_selection(
                        official_iou.float(), selected_ids, threshold
                    )
                    acc = counts[threshold]
                    acc[0] += tp
                    acc[1] += fp
                    acc[2] += fn
                image_count += 1

            metrics = {
                _threshold_key(threshold): _metric_from_counts(*counts[threshold])
                for threshold in iou_thresholds
            }
            mean_f1 = sum(metric["F1"] for metric in metrics.values()) / max(len(metrics), 1)
            row = {
                "stage": stage_name,
                "score_threshold": score_threshold,
                "quality_power": quality_power,
                "top_k": top_k,
                "nms_distance_thresh_px": nms_distance,
                "nms_min_overlap_points": nms_overlap,
                "row_visibility_thresh": float(args.row_visibility_thresh),
                "images": image_count,
                "selected_lanes": int(selected_count),
                "avg_selected_lanes_per_image": float(selected_count) / max(image_count, 1),
                "mF1": float(mean_f1),
                "metrics": metrics,
            }
            rows.append(row)
            print(
                f"{stage_name:>8} q={quality_power:>4.2f} thr={score_threshold:>4.2f} "
                f"F1@0.50={metrics['0.50']['F1']:.4f} "
                f"F1@0.70={metrics['0.70']['F1']:.4f} "
                f"mF1={mean_f1:.4f}"
            )

    rows.sort(
        key=lambda row: _row_sort_key(row, args.selection_iou, args.tie_break_iou),
        reverse=True,
    )
    best = rows[0] if rows else None
    output = {
        "metadata": metadata_for_json(
            cache,
            tool="sweep_cached_culane_thresholds",
            stage=stage_name,
            score_thresholds=score_thresholds,
            quality_powers=quality_powers,
            iou_thresholds=iou_thresholds,
            selection_iou=float(args.selection_iou),
            tie_break_iou=float(args.tie_break_iou),
            line_width=float(args.line_width),
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_overlap,
            top_k=top_k,
            row_visibility_thresh=float(args.row_visibility_thresh),
            iou_space="official_raster_cached",
        ),
        "best": best,
        "rows": rows,
    }
    write_json(args.output_json, output)

    report_lines = [
        f"config: {args.config}",
        f"checkpoint: {args.checkpoint}",
        f"split: {args.split}",
        f"list_path: {metadata.get('list_path')}",
        f"selection: max F1@{args.selection_iou:.2f}, tie mF1 then F1@{args.tie_break_iou:.2f}",
        f"score_thresholds: {' '.join(f'{v:.2f}' for v in score_thresholds)}",
        f"quality_powers: {' '.join(f'{v:.2f}' for v in quality_powers)}",
    ]
    if best is not None:
        report_lines.extend(
            [
                "best:",
                f"  score_threshold: {best['score_threshold']:.2f}",
                f"  quality_power: {best['quality_power']:.2f}",
                f"  mF1: {best['mF1']:.4f}",
            ]
        )
        for key in sorted(best["metrics"]):
            report_lines.append(f"  IoU {key}: {_format_metric(best['metrics'][key])}")
    report_lines.append("rows_sorted:")
    for row in rows:
        report_lines.append(
            f"  q={row['quality_power']:.2f} thr={row['score_threshold']:.2f} "
            f"F1@0.50={row['metrics']['0.50']['F1']:.4f} "
            f"F1@0.70={row['metrics']['0.70']['F1']:.4f} mF1={row['mF1']:.4f}"
        )
    report = "\n".join(report_lines) + "\n"
    print(report)
    if args.output_txt:
        output_txt = Path(args.output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text(report, encoding="utf-8")
        print(f"output_txt: {output_txt}")


if __name__ == "__main__":
    main()
