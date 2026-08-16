from __future__ import annotations

import argparse
import copy
from itertools import repeat
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_metric import (
    list_image_rel_paths,
    load_culane_img_data,
)
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_row_fractions
from dynlaneseq_eg.modeling.dynlaneseq_v23 import DynLaneSeqV23
from dynlaneseq_eg.tools.evaluate_v23_official import (
    FIXED_MIN_POINTS,
    FIXED_SCORE_THRESHOLD,
    FIXED_THRESHOLDS,
    FIXED_TOP_K,
    _checkpoint_contract,
    _matched_gt,
    _move_images,
    _validate_wrong_image_control,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


FIXED_SEED = 3407
POLICIES = (
    "source_v7",
    "learned_gate_v23",
    "raw_student_correct_image",
    "raw_student_wrong_image",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free full-validation V23 learned-gate versus raw-student audit."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--official-val-report", required=True)
    parser.add_argument("--official-val-prediction-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--wrong-image-list", required=True)
    parser.add_argument("--wrong-image-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--metric-workers", type=int, default=20)
    parser.add_argument("--metric-chunksize", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def raw_student_public_output(
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expose the fixed gate=1 student path in the public V7 slot order."""

    raw = output["student_x_rows"]
    source_slot_indices = output["source_slot_indices"].long()
    inverse = source_slot_indices.argsort(dim=1, stable=True)
    gather = inverse.unsqueeze(-1).expand(-1, -1, int(raw.shape[-1]))
    public_x = raw.gather(1, gather)
    return {
        "exist_logits": output["exist_logits"],
        "pred_x_rows": public_x,
        "range_norm": output["range_norm"],
        "quality_logits": output["quality_logits"],
    }


def _configured(
    base: dict[str, Any],
    *,
    dataset_root: Path,
    list_path: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("dataset", {})["root"] = str(dataset_root)
    cfg["dataset"].setdefault("lists", {})["val"] = str(list_path)
    cfg["dataset"]["load_targets"] = False
    cfg["dataset"]["infer_seg_labels"] = False
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _empty_displacement() -> dict[str, float]:
    return {
        "rows": 0.0,
        "raw_abs_sum_px": 0.0,
        "deployed_abs_sum_px": 0.0,
        "wrong_raw_abs_sum_px": 0.0,
    }


def _duplicate_teacher_batch(
    teacher: dict[str, torch.Tensor], batch_size: int
) -> dict[str, torch.Tensor]:
    """Repeat source-image teacher tensors for a correct/wrong image pair."""

    return {
        name: (
            torch.cat((value, value), dim=0)
            if value.ndim > 0 and int(value.shape[0]) == int(batch_size)
            else value
        )
        for name, value in teacher.items()
    }


def _split_student_batch(
    output: dict[str, torch.Tensor], batch_size: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    expected = 2 * int(batch_size)
    paired = {
        name: value
        for name, value in output.items()
        if value.ndim > 0 and int(value.shape[0]) == expected
    }
    if "student_x_rows" not in paired:
        raise ValueError("combined V23 output does not have a paired batch")
    return (
        {name: value[:batch_size] for name, value in paired.items()},
        {name: value[batch_size:] for name, value in paired.items()},
    )


def _accumulate_displacement(
    aggregate: dict[str, float],
    correct: dict[str, torch.Tensor],
    wrong: dict[str, torch.Tensor],
) -> None:
    source_x = correct["source_x_rows"].float()
    source_range = correct["source_range_norm"].float()
    source_active = correct["source_active"].bool()
    rows = int(source_x.shape[-1])
    row_fraction = fixed_row_fractions(
        rows, device=source_x.device, dtype=torch.float32
    ).view(1, 1, rows)
    valid = (
        source_active.unsqueeze(-1)
        & (row_fraction >= source_range[..., :1])
        & (row_fraction <= source_range[..., 1:])
    )
    count = int(valid.sum())
    aggregate["rows"] += float(count)
    aggregate["raw_abs_sum_px"] += float(
        (correct["student_x_rows"].float() - source_x).abs()[valid].sum()
    )
    # Public and canonical order have the same multiset, but use the exact
    # canonical deployed expression to avoid any ordering assumption.
    gate = correct["geometry_gate"].float()
    deployed = source_x + gate * (
        correct["student_x_rows"].float() - source_x
    )
    aggregate["deployed_abs_sum_px"] += float(
        (deployed - source_x).abs()[valid].sum()
    )
    aggregate["wrong_raw_abs_sum_px"] += float(
        (wrong["student_x_rows"].float() - source_x).abs()[valid].sum()
    )


@torch.inference_mode()
def _write_raw_predictions(
    model: DynLaneSeqV23,
    source_loader,
    wrong_loader,
    *,
    device: torch.device,
    correct_dir: Path,
    wrong_dir: Path,
    channels_last: bool,
    log_interval: int,
) -> dict[str, Any]:
    model.requires_grad_(False).eval()
    model.prepare_for_inference()
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)
    images_written = 0
    same_image = 0
    same_clip = 0
    displacement = _empty_displacement()
    started = time.perf_counter()
    for batch_index, (source_batch, wrong_batch) in enumerate(
        zip(source_loader, wrong_loader), start=1
    ):
        images, _targets, metas = source_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        for meta, wrong_meta in zip(metas, wrong_metas):
            source_path = str(meta.get("image_path", ""))
            wrong_path = str(wrong_meta.get("image_path", ""))
            same_image += int(source_path == wrong_path)
            same_clip += int(
                str(Path(source_path).parent) == str(Path(wrong_path).parent)
            )
        images = _move_images(
            images, device=device, channels_last=channels_last
        )
        wrong_images = _move_images(
            wrong_images, device=device, channels_last=channels_last
        )
        with torch.autocast(device_type=device.type, enabled=False):
            teacher = model.teacher(images.float(), inference_only=True)
            # One batch-2B student call avoids a second 319-step Viterbi
            # launch chain. A fixed-shape A/B on the deployment RTX 5080
            # measured a 1.47x student-forward speedup at 8.19 GiB peak.
            paired_images = torch.cat((images, wrong_images), dim=0)
            if channels_last:
                paired_images = paired_images.contiguous(
                    memory_format=torch.channels_last
                )
            paired_teacher = _duplicate_teacher_batch(
                teacher, int(images.shape[0])
            )
            paired_output = model.student(
                paired_images.float(),
                paired_teacher,
                include_proposal_scores=False,
            )
            correct, wrong = _split_student_batch(
                paired_output, int(images.shape[0])
            )
        _accumulate_displacement(displacement, correct, wrong)
        write_culane_predictions(
            raw_student_public_output(correct),
            metas,
            correct_dir,
            score_thresh=FIXED_SCORE_THRESHOLD,
            min_pred_points=FIXED_MIN_POINTS,
            nms_distance_thresh_px=0.0,
            top_k=FIXED_TOP_K,
            quality_score_power=0.0,
            score_mode="exist",
        )
        write_culane_predictions(
            raw_student_public_output(wrong),
            metas,
            wrong_dir,
            score_thresh=FIXED_SCORE_THRESHOLD,
            min_pred_points=FIXED_MIN_POINTS,
            nms_distance_thresh_px=0.0,
            top_k=FIXED_TOP_K,
            quality_score_power=0.0,
            score_mode="exist",
        )
        images_written += int(images.shape[0])
        if batch_index == 1 or (
            int(log_interval) > 0 and batch_index % int(log_interval) == 0
        ):
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            print(
                json.dumps(
                    {
                        "phase": "write_raw_student_predictions",
                        "batches": batch_index,
                        "images": images_written,
                        "images_per_second": images_written / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    count = max(displacement["rows"], 1.0)
    return {
        "images_written": images_written,
        "runtime_same_image_wrong_partner": same_image,
        "runtime_same_clip_wrong_partner": same_clip,
        "mean_abs_displacement_px": {
            "learned_gate_correct_image": displacement[
                "deployed_abs_sum_px"
            ]
            / count,
            "raw_student_correct_image": displacement["raw_abs_sum_px"]
            / count,
            "raw_student_wrong_image": displacement["wrong_raw_abs_sum_px"]
            / count,
        },
        "valid_active_rows": int(displacement["rows"]),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _evaluate_image(
    rel: str,
    dataset_root: str,
    prediction_roots: dict[str, str],
) -> dict[str, Any]:
    annotation = load_culane_img_data(
        Path(dataset_root) / rel.replace(".jpg", ".lines.txt")
    )
    predictions = {
        policy: load_culane_img_data(
            Path(root) / rel.replace(".jpg", ".lines.txt")
        )
        for policy, root in prediction_roots.items()
    }
    return {
        "prediction_count": {
            policy: len(lanes) for policy, lanes in predictions.items()
        },
        "thresholds": {
            f"{threshold:.2f}": {
                policy: _matched_gt(lanes, annotation, threshold)
                for policy, lanes in predictions.items()
            }
            for threshold in FIXED_THRESHOLDS
        },
    }


def _finish_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def summarize_gate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        policy: {
            f"{threshold:.2f}": {"TP": 0, "FP": 0, "FN": 0}
            for threshold in FIXED_THRESHOLDS
        }
        for policy in POLICIES
    }
    paired = {
        policy: {
            f"{threshold:.2f}": {"improved": 0, "worsened": 0, "tied": 0}
            for threshold in FIXED_THRESHOLDS
        }
        for policy in POLICIES
        if policy != "source_v7"
    }
    source_correct = {
        f"{threshold:.2f}": 0 for threshold in FIXED_THRESHOLDS
    }
    source_lost = {
        policy: {f"{threshold:.2f}": 0 for threshold in FIXED_THRESHOLDS}
        for policy in POLICIES
        if policy != "source_v7"
    }
    cardinality_exact = 0
    for row in rows:
        cardinality_exact += int(
            len(set(row["prediction_count"].values())) == 1
        )
        for threshold in FIXED_THRESHOLDS:
            key = f"{threshold:.2f}"
            threshold_row = row["thresholds"][key]
            source_tp = int(threshold_row["source_v7"][0])
            source_gt = threshold_row["source_v7"][3]
            source_correct[key] += len(source_gt)
            for policy in POLICIES:
                tp, fp, fn, matched_gt = threshold_row[policy]
                counts[policy][key]["TP"] += int(tp)
                counts[policy][key]["FP"] += int(fp)
                counts[policy][key]["FN"] += int(fn)
                if policy == "source_v7":
                    continue
                label = (
                    "improved"
                    if int(tp) > source_tp
                    else "worsened"
                    if int(tp) < source_tp
                    else "tied"
                )
                paired[policy][key][label] += 1
                source_lost[policy][key] += len(source_gt - matched_gt)
    metrics = {
        policy: {
            key: _finish_counts(value["TP"], value["FP"], value["FN"])
            for key, value in thresholds.items()
        }
        for policy, thresholds in counts.items()
    }
    degradation = {
        policy: {
            key: {
                "source_correct": source_correct[key],
                "lost": lost,
                "fraction": lost / max(source_correct[key], 1),
            }
            for key, lost in thresholds.items()
        }
        for policy, thresholds in source_lost.items()
    }
    return {
        "metrics": metrics,
        "paired_image_effects_vs_source": paired,
        "source_correct_degradation": degradation,
        "cardinality": {
            "exact_all_policy_count_images": cardinality_exact,
            "images": len(rows),
            "exact_fraction": cardinality_exact / max(len(rows), 1),
        },
    }


def _interpret(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    source = metrics["source_v7"]
    learned = metrics["learned_gate_v23"]
    raw = metrics["raw_student_correct_image"]
    wrong = metrics["raw_student_wrong_image"]

    def f1_delta(left, right, threshold):
        return 100.0 * (
            float(left[threshold]["F1"]) - float(right[threshold]["F1"])
        )

    values = {
        "learned_minus_source_f1_50_points": f1_delta(
            learned, source, "0.50"
        ),
        "learned_minus_source_f1_75_points": f1_delta(
            learned, source, "0.75"
        ),
        "raw_minus_source_f1_50_points": f1_delta(raw, source, "0.50"),
        "raw_minus_source_f1_75_points": f1_delta(raw, source, "0.75"),
        "raw_minus_learned_f1_50_points": f1_delta(raw, learned, "0.50"),
        "raw_minus_learned_f1_75_points": f1_delta(raw, learned, "0.75"),
        "raw_correct_minus_wrong_tp_50": int(raw["0.50"]["TP"])
        - int(wrong["0.50"]["TP"]),
        "raw_correct_minus_wrong_tp_75": int(raw["0.75"]["TP"])
        - int(wrong["0.75"]["TP"]),
    }
    checks = {
        "raw_student_beats_source_f1_50": values[
            "raw_minus_source_f1_50_points"
        ]
        > 0.0,
        "raw_student_nonregresses_source_f1_75": values[
            "raw_minus_source_f1_75_points"
        ]
        >= 0.0,
        "raw_correct_image_beats_wrong_tp_50": values[
            "raw_correct_minus_wrong_tp_50"
        ]
        > 0,
        "raw_correct_image_beats_wrong_tp_75": values[
            "raw_correct_minus_wrong_tp_75"
        ]
        > 0,
    }
    if checks["raw_student_beats_source_f1_50"] and checks[
        "raw_student_nonregresses_source_f1_75"
    ]:
        verdict = "geometry_gate_bottleneck_supported"
    elif (
        values["raw_minus_source_f1_50_points"] < 0.0
        and values["raw_minus_source_f1_75_points"] < 0.0
    ):
        verdict = "geometry_gate_protected_source_from_worse_raw_student"
    else:
        verdict = "mixed_raw_student_tradeoff"
    return {"values": values, "checks": checks, "verdict": verdict}


def main() -> None:
    args = parse_args()
    seed_everything(FIXED_SEED)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(dataset_root, split="val")
    source_list = Path(population["list_path"])
    wrong_list = Path(args.wrong_image_list).expanduser().resolve()
    wrong_contract = _validate_wrong_image_control(
        source_list,
        wrong_list,
        Path(args.wrong_image_report).expanduser().resolve(),
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    training_report = Path(args.training_report).expanduser().resolve()
    endpoint_contract = _checkpoint_contract(checkpoint, training_report)
    official_report_path = Path(args.official_val_report).expanduser().resolve()
    official_report = json.loads(official_report_path.read_text(encoding="utf-8"))
    if (
        official_report.get("checkpoint_iteration") != 8_000
        or official_report.get("contract", {}).get("official_val_rows") != 9_675
        or official_report.get("contract", {}).get("test_set_used") is not False
    ):
        raise ValueError("V23 learned-gate official report contract mismatch")

    base_cfg: dict[str, Any] = load_config(args.config)
    source_cfg = _configured(
        base_cfg,
        dataset_root=dataset_root,
        list_path=source_list,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    wrong_cfg = _configured(
        base_cfg,
        dataset_root=dataset_root,
        list_path=wrong_list,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    source_loader = build_dataloader(source_cfg, split="val", training=False)
    wrong_loader = build_dataloader(wrong_cfg, split="val", training=False)
    if len(source_loader.dataset) != 9_675 or len(wrong_loader.dataset) != 9_675:
        raise ValueError("V23 gate audit requires all 9,675 validation images")

    device = torch.device(args.device)
    model = build_model(source_cfg)
    if not isinstance(model, DynLaneSeqV23):
        raise TypeError("factory did not construct DynLaneSeqV23")
    iteration = int(load_checkpoint(checkpoint, model, strict=True))
    if iteration != 8_000:
        raise ValueError("V23 gate audit requires the fixed step-8000 endpoint")
    channels_last = bool(
        source_cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)

    output_dir = Path(args.output_dir).expanduser().resolve()
    correct_dir = output_dir / "predictions/raw_student_correct_image"
    wrong_dir = output_dir / "predictions/raw_student_wrong_image"
    writer = _write_raw_predictions(
        model,
        source_loader,
        wrong_loader,
        device=device,
        correct_dir=correct_dir,
        wrong_dir=wrong_dir,
        channels_last=channels_last,
        log_interval=int(args.log_interval),
    )
    if (
        writer["runtime_same_image_wrong_partner"] != 0
        or writer["runtime_same_clip_wrong_partner"] != 0
        or writer["images_written"] != 9_675
    ):
        raise ValueError("V23 raw-student writer contract failed")
    del model, source_loader, wrong_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    learned_prediction_root = Path(
        args.official_val_prediction_root
    ).expanduser().resolve()
    prediction_roots = {
        "source_v7": str(learned_prediction_root / "source_v7"),
        "learned_gate_v23": str(learned_prediction_root / "v23"),
        "raw_student_correct_image": str(correct_dir),
        "raw_student_wrong_image": str(wrong_dir),
    }
    rels = list_image_rel_paths(source_list)
    tasks = zip(rels, repeat(str(dataset_root)), repeat(prediction_roots))
    workers = (
        int(args.metric_workers)
        if int(args.metric_workers) > 0
        else cpu_count()
    )
    with Pool(workers) as pool:
        rows = list(
            tqdm(
                pool.starmap(
                    _evaluate_image,
                    tasks,
                    chunksize=max(int(args.metric_chunksize), 1),
                ),
                total=len(rels),
                desc="V23 raw gate official raster",
                ncols=80,
            )
        )
    summary = summarize_gate_audit(rows)
    interpretation = _interpret(summary)
    report = {
        "experiment": "V23 learned geometry gate versus fixed raw student audit",
        "diagnostic_only": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_iteration": iteration,
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "official_val_report": str(official_report_path),
        "official_val_report_sha256": sha256_file(official_report_path),
        "endpoint_contract": endpoint_contract,
        "wrong_image_contract": wrong_contract,
        "official_validation_population_contract": population,
        "writer_contract": writer,
        **summary,
        "interpretation": interpretation,
        "contract": {
            "raw_student_definition": "fixed geometry_gate=1 equivalent student_x_rows",
            "paired_correct_wrong_student_forward": True,
            "paired_forward_batch_size": 2 * int(args.eval_batch_size),
            "paired_forward_roundoff_max_px_preflight": 0.1160888671875,
            "raw_gate_values_tested": [1.0],
            "interpolation_or_gate_sweep_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "score_threshold": FIXED_SCORE_THRESHOLD,
            "lane_nms_enabled": False,
            "validation_subset_used": False,
            "official_val_rows": 9_675,
            "test_set_used": False,
            "optimizer_steps": 0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "raw_student_gate_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
