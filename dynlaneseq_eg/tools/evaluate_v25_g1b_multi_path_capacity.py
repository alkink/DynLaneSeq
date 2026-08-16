from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import cv2
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    _crop_intersection_count,
    _raster_lane_crop,
    sha256_file,
)
from dynlaneseq_eg.evaluation.culane_metric import eval_predictions, load_culane_img_data
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.modeling.v25_dual_energy_multi_path import (
    diverse_viterbi_paths,
    exact_small_path_set_decode,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import official_v23_culane_list_contract


POLICIES = (
    "single_map_predicted_count",
    "single_map_oracle_count",
    "top3_owned_official_oracle",
    "top3_deterministic_set",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-free V25 G1B diverse coherent-path capacity gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--metric-workers", type=int, default=20)
    parser.add_argument("--metric-chunksize", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _configured(path: str, root: Path, *, batch_size: int, workers: int) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(path))
    cfg.setdefault("dataset", {})["root"] = str(root)
    cfg["dataset"].setdefault("lists", {})["val"] = str(root / "list/val.txt")
    cfg["dataset"]["load_targets"] = False
    cfg["dataset"]["infer_seg_labels"] = False
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(workers)
    cfg["dataloader"]["persistent_workers"] = workers > 0
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _bottom_x(lane: list[tuple[float, float]]) -> float:
    return float(max(lane, key=lambda point: point[1])[0]) if lane else float("inf")


def _model_lane_to_original(
    x_rows: torch.Tensor,
    range_norm: torch.Tensor,
    meta: dict[str, Any],
) -> list[tuple[float, float]]:
    input_h = int(meta.get("input_h", 640))
    rows = int(x_rows.numel())
    y = fixed_y_rows(rows, input_h, device=x_rows.device, dtype=torch.float32)
    lane_range = sort_range_norm(range_norm.float().view(1, 2))[0]
    mask = (y >= lane_range[0] * input_h) & (y <= lane_range[1] * input_h)
    if int(mask.sum()) < 5:
        return []
    sx = float(meta.get("scale_x", 1.0))
    sy = float(meta.get("scale_y", 1.0))
    crop_x = float(meta.get("crop_x", 0.0))
    crop_y = float(meta.get("crop_y", 0.0))
    return [
        (float(x_value) / sx + crop_x, float(y_value) / sy + crop_y)
        for x_value, y_value in zip(x_rows[mask].cpu(), y[mask].cpu())
    ]


def _official_iou(
    prediction: list[tuple[float, float]],
    target: list[tuple[float, float]],
    *,
    height: int,
    width: int,
    line_width: int = 30,
) -> float:
    if len(prediction) < 2 or len(target) < 2:
        return 0.0
    pred_mask = _raster_lane_crop(prediction, height, width, line_width)
    target_mask = _raster_lane_crop(target, height, width, line_width)
    pred_count = int(cv2.countNonZero(pred_mask[0]))
    target_count = int(cv2.countNonZero(target_mask[0]))
    intersection = _crop_intersection_count(pred_mask, target_mask)
    union = pred_count + target_count - intersection
    return 0.0 if union <= 0 else float(intersection) / float(union)


def _exist_logits(active: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    positive = torch.full(active.shape, 20.0, device=active.device, dtype=dtype)
    negative = torch.full(active.shape, -20.0, device=active.device, dtype=dtype)
    return torch.stack(
        (torch.where(active, positive, negative), torch.where(active, negative, positive)),
        dim=-1,
    )


def _policy_output(
    x: torch.Tensor,
    active: torch.Tensor,
    source: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "pred_x_rows": x,
        "exist_logits": _exist_logits(active, dtype=x.dtype),
        "range_norm": source["range_norm"],
        "quality_logits": source["quality_logits"],
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    seed_everything(3407)
    root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(root, split="val")
    cfg = _configured(
        args.config,
        root,
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
    )
    loader = build_dataloader(cfg, split="val", training=False)
    expected = int(population["expected_nonempty_rows"])
    if len(loader.dataset) != expected:
        raise ValueError("G1B altered official validation population")
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("factory did not construct DynLaneSeqV25")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    iteration = int(load_checkpoint(checkpoint, model, strict=True))
    device = torch.device(args.device)
    model.requires_grad_(False).eval().to(device)
    channels_last = bool(cfg.get("training", {}).get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    output_dir = Path(args.output_dir).expanduser().resolve()
    directories = {name: output_dir / "predictions" / name for name in POLICIES}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    images_seen = 0
    distinct_second_sum = 0.0
    distinct_count = 0
    started = time.perf_counter()
    for batch_index, (images, _targets, metas) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=False):
            source = model(images.float())
        diverse = diverse_viterbi_paths(
            source["unary_logits"],
            num_hypotheses=3,
            transition_radius_bins=model.detector.transition_radius_bins,
            transition_penalty=model.detector.transition_penalty,
            suppression_radius_bins=5,
            suppression_penalty=8.0,
        )
        bin_width = float(model.detector.input_w) / float(model.detector.x_bins)
        hypotheses = (diverse.indices.float() + 0.5) * bin_width
        deterministic = exact_small_path_set_decode(
            hypotheses,
            diverse.scores,
            source["exist_logits"],
            input_w=model.detector.input_w,
            minimum_spacing_px=12.0,
        )
        distinct_second_sum += float(
            (hypotheses[:, :, 1] - hypotheses[:, :, 0]).abs().mean().item()
        ) * int(images.shape[0])
        distinct_count += int(images.shape[0])

        gt_counts: list[int] = []
        oracle_choice = torch.zeros(
            (images.shape[0], 4), device=device, dtype=torch.long
        )
        for image_index, meta in enumerate(metas):
            gt_lanes = sorted(
                load_culane_img_data(meta["anno_path"]), key=_bottom_x
            )[:4]
            gt_counts.append(len(gt_lanes))
            height = int(meta.get("orig_h", 590))
            width = int(meta.get("orig_w", 1640))
            for slot, gt_lane in enumerate(gt_lanes):
                values = []
                for hypothesis in range(3):
                    lane = _model_lane_to_original(
                        hypotheses[image_index, slot, hypothesis],
                        source["range_norm"][image_index, slot],
                        meta,
                    )
                    iou = _official_iou(
                        lane, gt_lane, height=height, width=width
                    )
                    values.append((iou >= 0.50, iou >= 0.75, iou, -hypothesis))
                oracle_choice[image_index, slot] = max(
                    range(3), key=lambda index: values[index]
                )
        gt_count = torch.tensor(gt_counts, device=device, dtype=torch.long)
        oracle_active = (
            torch.arange(4, device=device).view(1, 4) < gt_count.view(-1, 1)
        )
        oracle = hypotheses.gather(
            2,
            oracle_choice.view(-1, 4, 1, 1).expand(-1, -1, 1, hypotheses.shape[-1]),
        ).squeeze(2)
        single = hypotheses[:, :, 0]

        variants = {
            "single_map_predicted_count": _policy_output(
                single,
                torch.softmax(source["exist_logits"].float(), dim=-1)[..., 0]
                >= 0.5,
                source,
            ),
            "single_map_oracle_count": _policy_output(single, oracle_active, source),
            "top3_owned_official_oracle": _policy_output(oracle, oracle_active, source),
            "top3_deterministic_set": _policy_output(
                deterministic["selected_paths"],
                deterministic["selected_active"],
                source,
            ),
        }
        for name, output in variants.items():
            write_culane_predictions(
                output,
                metas,
                directories[name],
                score_thresh=0.5,
                min_pred_points=5,
                nms_distance_thresh_px=0.0,
                top_k=4,
                quality_score_power=0.0,
                score_mode="exist",
            )
        images_seen += int(images.shape[0])
        if batch_index == 1 or (
            args.log_interval > 0 and batch_index % args.log_interval == 0
        ):
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            print(
                json.dumps(
                    {
                        "phase": "v25_g1b_multi_path",
                        "images": images_seen,
                        "images_per_second": images_seen / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if images_seen != expected:
        raise RuntimeError("G1B did not consume full official validation")
    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = {
        name: {
            str(key): value
            for key, value in eval_predictions(
                pred_dir=directory,
                anno_dir=root,
                list_path=root / "list/val.txt",
                iou_thresholds=(0.50, 0.75),
                width=30,
                official=True,
                sequential=False,
                num_workers=args.metric_workers,
                chunksize=args.metric_chunksize,
            ).items()
        }
        for name, directory in directories.items()
    }
    single50 = float(metrics["single_map_oracle_count"]["0.5"]["F1"])
    oracle50 = float(metrics["top3_owned_official_oracle"]["0.5"]["F1"])
    deterministic50 = float(metrics["top3_deterministic_set"]["0.5"]["F1"])
    oracle_gain = 100.0 * (oracle50 - single50)
    predicted_single50 = float(
        metrics["single_map_predicted_count"]["0.5"]["F1"]
    )
    deterministic_delta = 100.0 * (deterministic50 - predicted_single50)
    checks = {
        "alternatives_spatially_distinct": distinct_second_sum / max(distinct_count, 1) >= 4.0,
        "deterministic_set_not_worse_by_0p10": deterministic_delta >= -0.10,
        "top3_oracle_adds_at_least_0p30": oracle_gain >= 0.30,
    }
    report = {
        "experiment": "V25 G1B diverse coherent-path capacity gate",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_iteration": iteration,
        "official_validation_population_contract": population,
        "metrics": metrics,
        "mean_abs_path1_minus_path0_px": distinct_second_sum / max(distinct_count, 1),
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "top3_owned_oracle_minus_single_map_f1_50_points": oracle_gain,
            "deterministic_set_minus_single_map_f1_50_points": deterministic_delta,
            "learned_selector_authorized": oracle_gain >= 0.80,
        },
        "contract": {
            "training_performed": False,
            "oracle_is_deployable": False,
            "official_val_rows": expected,
            "validation_subset_used": False,
            "validation_deduplication_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "test_set_used": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "g1b_multi_path_capacity_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
