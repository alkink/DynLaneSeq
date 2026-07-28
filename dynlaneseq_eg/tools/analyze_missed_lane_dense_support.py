from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import line_iou_against_gt
from dynlaneseq_eg.factory import build_dataloader, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether GT lanes missed by the structured candidates are still "
            "visible in the frozen dense centerline auxiliary output. This is an "
            "oracle association diagnostic, not a benchmark metric."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--peak-top-k", type=int, default=8)
    parser.add_argument("--peak-nms-radius-bins", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="none")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _best_iou(
    candidates: torch.Tensor,
    gt_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    line_width: float,
) -> float:
    ious = line_iou_against_gt(candidates, gt_x, valid, line_width=line_width)
    return float(ious.max().cpu()) if ious.numel() else 0.0


def _row_peaks(
    probabilities: torch.Tensor,
    *,
    top_k: int,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return NMS-filtered peak indices and scores for every image row."""
    if probabilities.ndim != 2:
        raise ValueError(f"Expected [rows, bins], got {tuple(probabilities.shape)}")
    kernel = 2 * int(radius) + 1
    pooled = F.max_pool1d(
        probabilities.unsqueeze(1),
        kernel_size=kernel,
        stride=1,
        padding=int(radius),
    ).squeeze(1)
    local = probabilities.masked_fill(probabilities < pooled, -1.0)
    k = min(int(top_k), int(probabilities.shape[-1]))
    scores, indices = torch.topk(local, k=k, dim=-1)
    return indices, scores


def _empty_bucket() -> dict[str, Any]:
    return {
        "lanes": 0,
        "best_group_iou_sum": 0.0,
        "best_all_iou_sum": 0.0,
        "gt_probability_sum": 0.0,
        "gt_probability_rows": 0,
        "peak_distance_sum_px": 0.0,
        "peak_distance_rows": 0,
        "rows_with_peak_8px": 0,
        "rows_with_peak_15px": 0,
        "rows_with_peak_30px": 0,
        "lane_peak_fraction_15px_sum": 0.0,
        "lane_peak_fraction_30px_sum": 0.0,
        "dense_peak_oracle_iou_sum": 0.0,
        "dense_peak_oracle_hits_050": 0,
        "dense_peak_oracle_hits_070": 0,
        "endpoint_peak_15px": 0,
        "endpoint_peak_30px": 0,
    }


def _update_bucket(
    bucket: dict[str, Any],
    *,
    best_group_iou: float,
    best_all_iou: float,
    gt_probabilities: torch.Tensor,
    peak_distances_px: torch.Tensor,
    dense_peak_oracle_iou: float,
) -> None:
    rows = int(peak_distances_px.numel())
    bucket["lanes"] += 1
    bucket["best_group_iou_sum"] += float(best_group_iou)
    bucket["best_all_iou_sum"] += float(best_all_iou)
    bucket["gt_probability_sum"] += float(gt_probabilities.sum().cpu())
    bucket["gt_probability_rows"] += int(gt_probabilities.numel())
    bucket["peak_distance_sum_px"] += float(peak_distances_px.sum().cpu())
    bucket["peak_distance_rows"] += rows
    bucket["rows_with_peak_8px"] += int((peak_distances_px <= 8.0).sum().cpu())
    bucket["rows_with_peak_15px"] += int((peak_distances_px <= 15.0).sum().cpu())
    bucket["rows_with_peak_30px"] += int((peak_distances_px <= 30.0).sum().cpu())
    bucket["lane_peak_fraction_15px_sum"] += float((peak_distances_px <= 15.0).float().mean().cpu())
    bucket["lane_peak_fraction_30px_sum"] += float((peak_distances_px <= 30.0).float().mean().cpu())
    bucket["dense_peak_oracle_iou_sum"] += float(dense_peak_oracle_iou)
    bucket["dense_peak_oracle_hits_050"] += int(dense_peak_oracle_iou >= 0.5)
    bucket["dense_peak_oracle_hits_070"] += int(dense_peak_oracle_iou >= 0.7)
    bucket["endpoint_peak_15px"] += int(float(peak_distances_px[-1].cpu()) <= 15.0)
    bucket["endpoint_peak_30px"] += int(float(peak_distances_px[-1].cpu()) <= 30.0)


def _summarize(bucket: dict[str, Any]) -> dict[str, float | int]:
    lanes = max(int(bucket["lanes"]), 1)
    rows = max(int(bucket["peak_distance_rows"]), 1)
    probability_rows = max(int(bucket["gt_probability_rows"]), 1)
    return {
        "lanes": int(bucket["lanes"]),
        "mean_best_group_iou": float(bucket["best_group_iou_sum"]) / lanes,
        "mean_best_all_iou": float(bucket["best_all_iou_sum"]) / lanes,
        "mean_centerline_probability_at_gt": float(bucket["gt_probability_sum"]) / probability_rows,
        "mean_nearest_topk_peak_distance_px": float(bucket["peak_distance_sum_px"]) / rows,
        "row_peak_recall_8px": int(bucket["rows_with_peak_8px"]) / rows,
        "row_peak_recall_15px": int(bucket["rows_with_peak_15px"]) / rows,
        "row_peak_recall_30px": int(bucket["rows_with_peak_30px"]) / rows,
        "mean_lane_peak_fraction_15px": float(bucket["lane_peak_fraction_15px_sum"]) / lanes,
        "mean_lane_peak_fraction_30px": float(bucket["lane_peak_fraction_30px_sum"]) / lanes,
        "mean_dense_peak_oracle_iou": float(bucket["dense_peak_oracle_iou_sum"]) / lanes,
        "dense_peak_oracle_recall_050": int(bucket["dense_peak_oracle_hits_050"]) / lanes,
        "dense_peak_oracle_recall_070": int(bucket["dense_peak_oracle_hits_070"]) / lanes,
        "endpoint_peak_recall_15px": int(bucket["endpoint_peak_15px"]) / lanes,
        "endpoint_peak_recall_30px": int(bucket["endpoint_peak_30px"]) / lanes,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    dataloader_cfg = cfg.setdefault("dataloader", {})
    dataloader_cfg["eval_batch_size"] = int(args.eval_batch_size)
    dataloader_cfg["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        dataloader_cfg["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg.setdefault("model", {})["require_pretrained_backbone"] = False

    model_cfg = cfg.get("model", {})
    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(structured_cfg.get("num_instances", model_cfg.get("num_slots", 0)))
    num_groups = int(structured_cfg.get("num_groups", 1))
    group_size = num_instances // max(num_groups, 1)
    input_w = float(model_cfg.get("input_w", 800))

    device = torch.device(args.device)
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    channels_last = bool(cfg.get("training", {}).get("channels_last", False)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    loader = build_dataloader(cfg, split=args.split, training=False)

    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    autocast_enabled = amp_dtype is not None and device.type == "cuda"

    buckets = {
        "all": _empty_bucket(),
        "group_hit_050": _empty_bucket(),
        "group_miss_050": _empty_bucket(),
        "group_hit_070": _empty_bucket(),
        "group_miss_070": _empty_bucket(),
    }
    images_seen = 0

    for batch_index, (images, targets, _metas) in enumerate(
        tqdm(loader, ncols=88, desc="missed-lane dense support")
    ):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format,
        )
        amp_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if autocast_enabled
            else nullcontext()
        )
        with amp_context:
            outputs = model(images)
        centerline_logits = outputs.get("centerline_logits")
        if not isinstance(centerline_logits, torch.Tensor):
            raise RuntimeError("Checkpoint/config does not expose centerline_logits")
        centerline_prob = torch.sigmoid(centerline_logits[:, 0].float())
        pred_x = outputs["pred_x_rows"].float()
        images_seen += int(images.shape[0])

        for sample_index, target in enumerate(targets):
            row_prob = centerline_prob[sample_index]
            peak_indices, _peak_scores = _row_peaks(
                row_prob,
                top_k=int(args.peak_top_k),
                radius=int(args.peak_nms_radius_bins),
            )
            bins = int(row_prob.shape[-1])
            bin_scale = input_w / float(bins)
            candidates_all = pred_x[sample_index]
            candidates_group = candidates_all[:group_size]
            gt_rows = target["x_rows"].to(device=device, dtype=torch.float32)
            gt_valid = target["valid_mask"].to(device=device).bool()

            for lane_index in range(int(gt_rows.shape[0])):
                valid = gt_valid[lane_index]
                if int(valid.sum()) < 5:
                    continue
                gt_x = gt_rows[lane_index]
                best_group_iou = _best_iou(
                    candidates_group,
                    gt_x,
                    valid,
                    line_width=float(args.line_width),
                )
                best_all_iou = _best_iou(
                    candidates_all,
                    gt_x,
                    valid,
                    line_width=float(args.line_width),
                )
                valid_rows = valid.nonzero(as_tuple=False).flatten()
                gt_bin_float = (gt_x[valid_rows] / bin_scale).clamp(0.0, float(bins - 1))
                gt_bin_nearest = gt_bin_float.round().long()
                gt_probabilities = row_prob[valid_rows, gt_bin_nearest]
                selected_peaks = peak_indices[valid_rows]
                peak_distances_bins = (selected_peaks.float() - gt_bin_float[:, None]).abs()
                nearest_peak_position = selected_peaks.gather(
                    1,
                    peak_distances_bins.argmin(dim=1, keepdim=True),
                ).squeeze(1)
                peak_distances_px = (
                    nearest_peak_position.float() - gt_bin_float
                ).abs() * bin_scale

                dense_curve = gt_x.new_zeros(gt_x.shape)
                dense_curve[valid_rows] = (nearest_peak_position.float() + 0.5) * bin_scale
                dense_iou = _best_iou(
                    dense_curve.unsqueeze(0),
                    gt_x,
                    valid,
                    line_width=float(args.line_width),
                )
                values = {
                    "best_group_iou": best_group_iou,
                    "best_all_iou": best_all_iou,
                    "gt_probabilities": gt_probabilities,
                    "peak_distances_px": peak_distances_px,
                    "dense_peak_oracle_iou": dense_iou,
                }
                _update_bucket(buckets["all"], **values)
                _update_bucket(
                    buckets["group_hit_050" if best_group_iou >= 0.5 else "group_miss_050"],
                    **values,
                )
                _update_bucket(
                    buckets["group_hit_070" if best_group_iou >= 0.7 else "group_miss_070"],
                    **values,
                )

    payload = {
        "diagnostic_only": True,
        "oracle_warning": (
            "The dense peak curve selects the nearest top-k centerline peak independently "
            "at each GT-valid row. It measures evidence availability under oracle association; "
            "it is not an achievable detector result."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "images": images_seen,
        "num_instances": num_instances,
        "num_groups": num_groups,
        "group_size": group_size,
        "line_width": float(args.line_width),
        "peak_top_k_per_row": int(args.peak_top_k),
        "peak_nms_radius_bins": int(args.peak_nms_radius_bins),
        "buckets": {name: _summarize(bucket) for name, bucket in buckets.items()},
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
