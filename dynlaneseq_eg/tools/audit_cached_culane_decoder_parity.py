from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
import re
import subprocess
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    evaluator_hungarian_assignment,
    trace_postprocess,
)
from dynlaneseq_eg.evaluation.culane_metric import culane_metric, load_culane_img_data
from dynlaneseq_eg.evaluation.decoder_parity import (
    DECODER_VARIANTS,
    DecoderVariant,
    lane_to_original,
    prediction_relative_path,
    select_candidate_ids,
    write_prediction_file,
)


THRESHOLDS = (0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed A-E decoder/writer parity treatments on a cached full CULane validation population."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--official-evaluator", default="")
    parser.add_argument("--data-root", default="/home/alki/projects/CULane")
    parser.add_argument("--list-path", default="/home/alki/projects/CULane/list/val.txt")
    parser.add_argument("--workers", type=int, default=min(12, cpu_count()))
    parser.add_argument("--score-thresh", type=float, default=0.30)
    parser.add_argument("--quality-power", type=float, default=0.50)
    parser.add_argument("--current-nms-distance-px", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--expected-images", type=int, default=9675)
    parser.add_argument("--expected-python-tp50", type=int, default=25484)
    parser.add_argument("--expected-python-fp50", type=int, default=3777)
    parser.add_argument("--expected-python-fn50", type=int, default=7198)
    parser.add_argument("--reuse-written-predictions", action="store_true")
    parser.add_argument("--reuse-official-results", action="store_true")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = float(tp) / float(tp + fp) if tp + fp else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def cached_baseline_counts(
    records: list[dict[str, Any]],
    *,
    input_h: int,
    input_w: int,
    score_thresh: float,
    quality_power: float,
    nms_distance: float,
    nms_overlap: int,
    top_k: int,
) -> tuple[dict[float, list[int]], dict[str, list[int]]]:
    counts = {threshold: [0, 0, 0] for threshold in THRESHOLDS}
    selections: dict[str, list[int]] = {}
    for record in tqdm(records, desc="cached baseline parity", ncols=90):
        stage = record["stages"]["main"]
        trace = trace_postprocess(
            stage,
            input_h=input_h,
            input_w=input_w,
            score_thresh=score_thresh,
            quality_power=quality_power,
            min_valid_rows=5,
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_overlap,
            top_k=top_k,
        )
        selected = [int(value) for value in trace["selected_ids"]]
        selections[str(record["image_id"])] = selected
        iou = stage["official_iou"].float()
        for threshold in THRESHOLDS:
            assignment = evaluator_hungarian_assignment(iou, selected, threshold)
            tp = int(assignment.hit_count)
            counts[threshold][0] += tp
            counts[threshold][1] += len(selected) - tp
            counts[threshold][2] += int(iou.shape[0]) - tp
    return counts, selections


def write_all_predictions(
    records: list[dict[str, Any]],
    variants: tuple[DecoderVariant, ...],
    output_root: Path,
    *,
    input_h: int,
    input_w: int,
    score_thresh: float,
    quality_power: float,
    current_nms_distance: float,
    nms_overlap: int,
    min_valid_rows: int,
    top_k: int,
    cached_current_selections: dict[str, list[int]],
) -> dict[str, Any]:
    selected_counts = defaultdict(int)
    selection_changes = defaultdict(int)
    selection_parity_mismatches = 0
    for record in tqdm(records, desc="write A-E predictions", ncols=90):
        stage = record["stages"]["main"]
        relative_path = prediction_relative_path(record["meta"])
        baseline_ids = cached_current_selections[str(record["image_id"])]
        for variant in variants:
            selected, candidate_lanes = select_candidate_ids(
                stage,
                input_h=input_h,
                input_w=input_w,
                variant=variant,
                score_thresh=score_thresh,
                quality_power=quality_power,
                current_nms_distance_px=current_nms_distance,
                nms_min_overlap_points=nms_overlap,
                min_valid_rows=min_valid_rows,
                top_k=top_k,
            )
            if variant.name == "A_current" and selected != baseline_ids:
                selection_parity_mismatches += 1
            if selected != baseline_ids:
                selection_changes[variant.name] += 1
            selected_counts[variant.name] += len(selected)
            lanes = [
                lane_to_original(candidate_lanes[proposal_idx], record["meta"], variant.writer_mode)
                for proposal_idx in selected
            ]
            write_prediction_file(
                output_root / "predictions" / variant.name / relative_path,
                lanes,
                variant.writer_mode,
            )
    if selection_parity_mismatches:
        raise RuntimeError(f"A_current selection parity failed on {selection_parity_mismatches} images")
    return {
        "selected_lanes": dict(selected_counts),
        "images_with_selection_change_vs_A": dict(selection_changes),
        "A_selection_parity_mismatches": int(selection_parity_mismatches),
    }


def parse_official_output(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    counts = re.search(r"tp:\s*(\d+)\s+fp:\s*(\d+)\s+fn:\s*(\d+)", text)
    fmeasure = re.search(r"Fmeasure:\s*([-+0-9.eE]+)", text)
    if counts is None or fmeasure is None:
        raise RuntimeError(f"Could not parse official evaluator output: {path}\n{text}")
    return metric(int(counts.group(1)), int(counts.group(2)), int(counts.group(3)))


def run_official_evaluator(
    evaluator: Path,
    variants: tuple[DecoderVariant, ...],
    output_root: Path,
    data_root: Path,
    list_path: Path,
    reuse_existing: bool = False,
) -> dict[str, dict[str, dict[str, float | int]]]:
    results: dict[str, dict[str, dict[str, float | int]]] = {}
    for variant in variants:
        variant_results: dict[str, dict[str, float | int]] = {}
        for threshold in THRESHOLDS:
            result_path = output_root / "official" / f"{variant.name}_iou{threshold:.2f}.txt"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = result_path.with_suffix(".log")
            if reuse_existing and result_path.is_file() and log_path.is_file():
                log_text = log_path.read_text(encoding="utf-8")
                if "list images num: 9675" not in log_text:
                    raise RuntimeError(f"Existing official result did not evaluate all 9675 images: {log_path}")
                variant_results[f"{threshold:.2f}"] = parse_official_output(result_path)
                continue
            command = [
                str(evaluator),
                "-a",
                str(data_root) + "/",
                "-d",
                str(output_root / "predictions" / variant.name) + "/",
                "-i",
                str(data_root) + "/",
                "-l",
                str(list_path),
                "-w",
                "30",
                "-t",
                f"{threshold:.2f}",
                "-c",
                "1640",
                "-r",
                "590",
                "-f",
                "1",
                "-p",
                "1",
                "-o",
                str(result_path),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"Official evaluator failed ({completed.returncode}); see {log_path}")
            variant_results[f"{threshold:.2f}"] = parse_official_output(result_path)
        results[variant.name] = variant_results
    return results


def _paired_worker(task: tuple[str, str, dict[str, str]]) -> tuple[str, dict[str, dict[float, list[int]]]]:
    image_id, anno_path, prediction_paths = task
    anno = load_culane_img_data(anno_path)
    output: dict[str, dict[float, list[int]]] = {}
    for name, prediction_path in prediction_paths.items():
        pred = load_culane_img_data(prediction_path)
        output[name] = culane_metric(pred, anno, width=30, iou_thresholds=THRESHOLDS, official=True)
    return image_id, output


def run_paired_python_metrics(
    records: list[dict[str, Any]],
    variants: tuple[DecoderVariant, ...],
    output_root: Path,
    workers: int,
) -> tuple[dict[str, dict[str, dict[str, float | int]]], dict[str, Any]]:
    names = [variant.name for variant in variants]

    def tasks():
        for record in records:
            relative = prediction_relative_path(record["meta"])
            yield (
                str(record["image_id"]),
                str(record["meta"]["anno_path"]),
                {name: str(output_root / "predictions" / name / relative) for name in names},
            )

    aggregate = {name: {threshold: [0, 0, 0] for threshold in THRESHOLDS} for name in names}
    paired = {
        name: {
            threshold: {"improved_images": 0, "worsened_images": 0, "same_images": 0, "tp_recovered": 0, "tp_lost": 0}
            for threshold in THRESHOLDS
        }
        for name in names
        if name != "A_current"
    }
    with Pool(processes=max(1, int(workers))) as pool:
        iterator = pool.imap(_paired_worker, tasks(), chunksize=8)
        for _image_id, image_results in tqdm(iterator, total=len(records), desc="paired Python raster", ncols=90):
            for name in names:
                for threshold in THRESHOLDS:
                    row = image_results[name][threshold]
                    for index in range(3):
                        aggregate[name][threshold][index] += int(row[index])
            baseline = image_results["A_current"]
            for name in paired:
                for threshold in THRESHOLDS:
                    delta = int(image_results[name][threshold][0]) - int(baseline[threshold][0])
                    if delta > 0:
                        paired[name][threshold]["improved_images"] += 1
                        paired[name][threshold]["tp_recovered"] += delta
                    elif delta < 0:
                        paired[name][threshold]["worsened_images"] += 1
                        paired[name][threshold]["tp_lost"] += -delta
                    else:
                        paired[name][threshold]["same_images"] += 1

    metrics = {
        name: {f"{threshold:.2f}": metric(*aggregate[name][threshold]) for threshold in THRESHOLDS}
        for name in names
    }
    paired_json = {
        name: {f"{threshold:.2f}": values for threshold, values in thresholds.items()}
        for name, thresholds in paired.items()
    }
    return metrics, paired_json


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# CULane Decoder/Writer Parity Audit",
        "",
        f"- Images: {payload['protocol']['images']}",
        f"- List SHA-256: `{payload['protocol']['list_sha256']}`",
        "- No image exclusion, deduplication, checkpoint selection, or threshold sweep.",
        "- Official C evaluator was forced to `-p 1`; its local 20-thread fork drops remainder images.",
        f"- Current-writer serialization ΔTP/FP/FN @.50: "
        f"`{payload['serialization_effect']['delta_tp50']:+d}/"
        f"{payload['serialization_effect']['delta_fp50']:+d}/"
        f"{payload['serialization_effect']['delta_fn50']:+d}`.",
        "",
        "| Variant | Official F1@.50 | Δ | Official F1@.75 | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline = payload["official_c"]["A_current"]
    for variant in DECODER_VARIANTS:
        row = payload["official_c"][variant.name]
        f50 = float(row["0.50"]["F1"])
        f75 = float(row["0.75"]["F1"])
        lines.append(
            f"| {variant.name} | {100*f50:.4f} | {100*(f50-float(baseline['0.50']['F1'])):+.4f} "
            f"| {100*f75:.4f} | {100*(f75-float(baseline['0.75']['F1'])):+.4f} |"
        )
    lines.extend(["", "## Fixed treatments", ""])
    for variant in DECODER_VARIANTS:
        lines.append(f"- `{variant.name}`: {variant.description}")
    lines.extend(["", "## Paired image-level Python-raster changes", ""])
    for name, thresholds in payload["paired"].items():
        for threshold, values in thresholds.items():
            lines.append(
                f"- `{name}` @ {threshold}: improved={values['improved_images']}, "
                f"worsened={values['worsened_images']}, recovered={values['tp_recovered']}, lost={values['tp_lost']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache["records"]
    if len(records) != int(args.expected_images):
        raise RuntimeError(f"Expected {args.expected_images} official val images, found {len(records)}")
    list_path = Path(args.list_path).expanduser().resolve()
    list_count = sum(1 for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if list_count != len(records):
        raise RuntimeError(f"List/cache population mismatch: list={list_count}, cache={len(records)}")

    input_h = int(cache["metadata"]["input_h"])
    input_w = int(cache["metadata"]["input_w"])
    cached_counts, cached_selections = cached_baseline_counts(
        records,
        input_h=input_h,
        input_w=input_w,
        score_thresh=args.score_thresh,
        quality_power=args.quality_power,
        nms_distance=args.current_nms_distance_px,
        nms_overlap=args.nms_min_overlap_points,
        top_k=args.top_k,
    )
    expected = (args.expected_python_tp50, args.expected_python_fp50, args.expected_python_fn50)
    observed = tuple(cached_counts[0.50])
    if observed != expected:
        raise RuntimeError(f"Cached baseline parity failed: expected={expected}, observed={observed}")

    if args.reuse_written_predictions:
        expected_paths = [prediction_relative_path(record["meta"]) for record in records]
        for variant in DECODER_VARIANTS:
            missing = [
                str(relative)
                for relative in expected_paths
                if not (output_root / "predictions" / variant.name / relative).is_file()
            ]
            if missing:
                raise RuntimeError(f"Cannot reuse {variant.name}: {len(missing)} prediction files are missing")
        selection_summary = {
            "reused_existing_predictions": True,
            "validated_prediction_files_per_variant": len(expected_paths),
        }
    else:
        selection_summary = write_all_predictions(
            records,
            DECODER_VARIANTS,
            output_root,
            input_h=input_h,
            input_w=input_w,
            score_thresh=args.score_thresh,
            quality_power=args.quality_power,
            current_nms_distance=args.current_nms_distance_px,
            nms_overlap=args.nms_min_overlap_points,
            min_valid_rows=args.min_valid_rows,
            top_k=args.top_k,
            cached_current_selections=cached_selections,
        )

    if not args.official_evaluator:
        raise RuntimeError("--official-evaluator is required for this audit")
    evaluator = Path(args.official_evaluator).expanduser().resolve()
    official = run_official_evaluator(
        evaluator,
        DECODER_VARIANTS,
        output_root,
        Path(args.data_root).expanduser().resolve(),
        list_path,
        reuse_existing=args.reuse_official_results,
    )
    python_metrics, paired = run_paired_python_metrics(records, DECODER_VARIANTS, output_root, args.workers)
    python_a = python_metrics["A_current"]["0.50"]
    written = (int(python_a["TP"]), int(python_a["FP"]), int(python_a["FN"]))
    serialization_effect = {
        "cached_in_memory_tp50": expected[0],
        "cached_in_memory_fp50": expected[1],
        "cached_in_memory_fn50": expected[2],
        "written_reloaded_tp50": written[0],
        "written_reloaded_fp50": written[1],
        "written_reloaded_fn50": written[2],
        "delta_tp50": written[0] - expected[0],
        "delta_fp50": written[1] - expected[1],
        "delta_fn50": written[2] - expected[2],
        "exact_parity": written == expected,
    }

    payload = {
        "protocol": {
            "cache": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "checkpoint": cache["metadata"].get("checkpoint"),
            "config": cache["metadata"].get("config"),
            "images": len(records),
            "list_path": str(list_path),
            "list_sha256": sha256_file(list_path),
            "score_thresh": float(args.score_thresh),
            "quality_power": float(args.quality_power),
            "top_k": int(args.top_k),
            "current_nms_distance_px": float(args.current_nms_distance_px),
            "clr_nms_distance_px_at_input_width": 50.0 * float(input_w) / 800.0,
            "official_evaluator": str(evaluator),
            "official_evaluator_sha256": sha256_file(evaluator),
            "official_evaluator_processes": 1,
        },
        "variants": {variant.name: variant.__dict__ for variant in DECODER_VARIANTS},
        "cached_baseline": {f"{threshold:.2f}": metric(*values) for threshold, values in cached_counts.items()},
        "selection": selection_summary,
        "official_c": official,
        "python_raster": python_metrics,
        "paired": paired,
        "serialization_effect": serialization_effect,
    }
    json_path = output_root / "decoder_parity_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = markdown_report(payload)
    report_path = output_root / "decoder_parity_audit.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"json: {json_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
