from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import (
    line_iou_against_gt,
    select_candidates,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired short-run gate for the learned lane-row reference decoder. "
            "The primary metric is raw all-query proposal recall; score, Top-K, "
            "NMS, and the official test protocol cannot hide or create a pass."
        )
    )
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--min-recovered-lanes", type=int, default=5)
    parser.add_argument("--max-lost-lanes", type=int, default=2)
    parser.add_argument("--min-image-specificity-points", type=float, default=5.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


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


def _best_iou(
    candidates: torch.Tensor,
    gt_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    line_width: float,
) -> float:
    if int(valid.sum()) < 5 or int(candidates.shape[0]) == 0:
        return 0.0
    iou = line_iou_against_gt(
        candidates,
        gt_x,
        valid,
        line_width=float(line_width),
    )
    return float(iou.max()) if iou.numel() else 0.0


def _model_outputs(
    model,
    images: torch.Tensor,
    *,
    wrong_image_control: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    if model.structured_query_head is None:
        raise ValueError("row-reference gate requires a structured query head")
    enc = model.encoder.forward_features(
        images,
        inference_only=True,
        structured_only=True,
    )
    features = enc["features"]
    correct = model.structured_query_head(features, inference_only=True)
    wrong = None
    if wrong_image_control:
        wrong_features = torch.roll(features, shifts=1, dims=0)
        wrong = model.structured_query_head(wrong_features, inference_only=True)
    return correct, wrong


def _evaluate(
    config_path: str,
    checkpoint_path: str,
    *,
    dataset_root: str,
    split: str,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    eval_batch_size: int,
    num_workers: int,
    max_batches: int,
    sample_strategy: str,
    line_width: float,
    top_k: int,
    wrong_image_control: bool,
) -> dict[str, Any]:
    cfg = _prepare_config(
        config_path,
        dataset_root=dataset_root,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    loader = build_dataloader(cfg, split=split, training=False)
    loader, indices = select_diagnostic_loader(
        loader,
        strategy=sample_strategy,
        max_batches=max_batches,
        num_workers=num_workers,
    )
    model = build_model(cfg)
    iteration = load_checkpoint(checkpoint_path, model, strict=False)
    model = model.to(device).eval()
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False) and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    records: list[dict[str, float | int]] = []
    image_offset = 0
    with torch.inference_mode():
        for images, targets, _metas in tqdm(loader, desc=f"gate {Path(config_path).stem}"):
            if channels_last:
                images = images.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                images = images.to(device, non_blocking=True)
            with _amp_context(device, amp_dtype):
                correct, wrong = _model_outputs(
                    model,
                    images,
                    wrong_image_control=wrong_image_control,
                )
            for batch_index, target in enumerate(targets):
                all_candidates = correct["pred_x_rows"][batch_index]
                top_candidates = select_candidates(
                    correct,
                    batch_index,
                    top_k=int(top_k),
                    rank_by="score_quality",
                )
                wrong_candidates = (
                    wrong["pred_x_rows"][batch_index]
                    if wrong is not None
                    else all_candidates.new_zeros((0, all_candidates.shape[-1]))
                )
                gt_x = target["x_rows"].to(
                    device=all_candidates.device,
                    dtype=all_candidates.dtype,
                )
                valid = target["valid_mask"].to(device=all_candidates.device).bool()
                dataset_index = int(indices[image_offset + batch_index])
                for lane_index in range(int(gt_x.shape[0])):
                    if int(valid[lane_index].sum()) < 5:
                        continue
                    records.append(
                        {
                            "dataset_index": dataset_index,
                            "lane_index": int(lane_index),
                            "all_iou": _best_iou(
                                all_candidates,
                                gt_x[lane_index],
                                valid[lane_index],
                                line_width=line_width,
                            ),
                            "top_iou": _best_iou(
                                top_candidates,
                                gt_x[lane_index],
                                valid[lane_index],
                                line_width=line_width,
                            ),
                            "wrong_image_all_iou": _best_iou(
                                wrong_candidates,
                                gt_x[lane_index],
                                valid[lane_index],
                                line_width=line_width,
                            ),
                        }
                    )
            image_offset += int(images.shape[0])

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "iteration": int(iteration),
        "sampled_dataset_indices": indices,
        "records": records,
        "parameters": int(parameter_count),
        "peak_memory_bytes": peak_memory,
    }


def _recall(values: list[float], threshold: float) -> float:
    return sum(value >= threshold for value in values) / float(max(len(values), 1))


def _summarize_pair(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_recovered_lanes: int,
    max_lost_lanes: int,
    min_image_specificity_points: float,
) -> dict[str, Any]:
    control_records = control["records"]
    candidate_records = candidate["records"]
    control_ids = [
        (int(record["dataset_index"]), int(record["lane_index"]))
        for record in control_records
    ]
    candidate_ids = [
        (int(record["dataset_index"]), int(record["lane_index"]))
        for record in candidate_records
    ]
    if control_ids != candidate_ids:
        raise ValueError("control and candidate lane identities are not aligned")

    summary: dict[str, Any] = {
        "lanes": len(control_records),
        "thresholds": {},
    }
    for threshold in (0.5, 0.7):
        control_all = [float(record["all_iou"]) for record in control_records]
        candidate_all = [float(record["all_iou"]) for record in candidate_records]
        candidate_wrong = [
            float(record["wrong_image_all_iou"]) for record in candidate_records
        ]
        control_top = [float(record["top_iou"]) for record in control_records]
        candidate_top = [float(record["top_iou"]) for record in candidate_records]
        recovered = sum(
            base < threshold <= new
            for base, new in zip(control_all, candidate_all)
        )
        lost = sum(
            new < threshold <= base
            for base, new in zip(control_all, candidate_all)
        )
        control_recall = _recall(control_all, threshold)
        candidate_recall = _recall(candidate_all, threshold)
        wrong_recall = _recall(candidate_wrong, threshold)
        summary["thresholds"][f"{threshold:.2f}"] = {
            "control_all_recall": control_recall,
            "candidate_all_recall": candidate_recall,
            "candidate_wrong_image_all_recall": wrong_recall,
            "candidate_gain_points": 100.0 * (candidate_recall - control_recall),
            "candidate_image_specificity_points": 100.0 * (
                candidate_recall - wrong_recall
            ),
            "unique_control_misses_recovered": int(recovered),
            "control_hits_lost": int(lost),
            "net_unique_hits": int(recovered - lost),
            "control_scored_topk_recall": _recall(control_top, threshold),
            "candidate_scored_topk_recall": _recall(candidate_top, threshold),
        }
    control_mean = sum(float(record["all_iou"]) for record in control_records) / float(
        max(len(control_records), 1)
    )
    candidate_mean = sum(
        float(record["all_iou"]) for record in candidate_records
    ) / float(max(len(candidate_records), 1))
    primary = summary["thresholds"]["0.50"]
    positive_gate = bool(
        int(primary["unique_control_misses_recovered"]) >= int(min_recovered_lanes)
        and int(primary["control_hits_lost"]) <= int(max_lost_lanes)
        and float(primary["candidate_gain_points"]) > 0.0
        and float(primary["candidate_image_specificity_points"])
        >= float(min_image_specificity_points)
        and candidate_mean > control_mean
    )
    summary.update(
        {
            "control_mean_best_iou": control_mean,
            "candidate_mean_best_iou": candidate_mean,
            "mean_best_iou_gain": candidate_mean - control_mean,
            "positive_gate": positive_gate,
            "gate_definition": {
                "min_unique_control_misses_recovered_at_050": int(min_recovered_lanes),
                "max_control_hits_lost_at_050": int(max_lost_lanes),
                "candidate_gain_points_at_050": "> 0",
                "min_correct_vs_wrong_image_gap_points_at_050": float(
                    min_image_specificity_points
                ),
                "mean_best_iou_gain": "> 0",
            },
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    if int(args.eval_batch_size) < 2:
        raise ValueError(
            "eval_batch_size must be >= 2 so the batch-rolled wrong-image "
            "control cannot reuse the same image"
        )
    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    control = _evaluate(
        args.control_config,
        args.control_checkpoint,
        dataset_root=args.dataset_root,
        split=args.split,
        device=device,
        amp_dtype=amp_dtype,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        sample_strategy=args.sample_strategy,
        line_width=args.line_width,
        top_k=args.top_k,
        wrong_image_control=False,
    )
    candidate = _evaluate(
        args.candidate_config,
        args.candidate_checkpoint,
        dataset_root=args.dataset_root,
        split=args.split,
        device=device,
        amp_dtype=amp_dtype,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        sample_strategy=args.sample_strategy,
        line_width=args.line_width,
        top_k=args.top_k,
        wrong_image_control=True,
    )
    summary = _summarize_pair(
        control,
        candidate,
        min_recovered_lanes=args.min_recovered_lanes,
        max_lost_lanes=args.max_lost_lanes,
        min_image_specificity_points=args.min_image_specificity_points,
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "This is a predeclared short-run raw-proposal gate, not an official "
            "CULane test result. A pass authorizes a full run; it does not "
            "establish final F1."
        ),
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "images": len(control["sampled_dataset_indices"]),
        "control": {
            key: value for key, value in control.items() if key != "records"
        },
        "candidate": {
            key: value for key, value in candidate.items() if key != "records"
        },
        "summary": summary,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
