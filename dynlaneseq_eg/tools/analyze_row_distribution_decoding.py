from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import line_iou_against_gt
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device, soft_expected_x
from dynlaneseq_eg.tools.analyze_decoder_image_grounding import (
    _group_zero_assignments,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether soft expected-x decoding of the final row-bin "
            "distribution limits raw lane proposal geometry. This diagnostic "
            "does not apply scores, Top-K, NMS, or the official test protocol."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.5],
    )
    parser.add_argument(
        "--local-mode-radii",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Renormalize probability inside +/- this many bins around argmax.",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7],
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


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
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False
    return cfg


def decode_argmax(
    logits: torch.Tensor,
    *,
    input_w: int,
    x_bins: int,
) -> torch.Tensor:
    return logits.argmax(dim=-1).to(dtype=logits.dtype) * (
        float(input_w) / float(x_bins)
    )


def decode_local_mode_expectation(
    logits: torch.Tensor,
    *,
    radius_bins: int,
    input_w: int,
    x_bins: int,
) -> torch.Tensor:
    """Expected coordinate after retaining one window around the global mode."""

    radius_bins = int(radius_bins)
    if radius_bins < 0:
        raise ValueError("radius_bins must be non-negative")
    probs = torch.softmax(logits, dim=-1)
    mode = probs.argmax(dim=-1, keepdim=True)
    bins = torch.arange(
        x_bins,
        device=logits.device,
        dtype=torch.long,
    )
    view_shape = [1] * (logits.ndim - 1) + [x_bins]
    bins = bins.view(*view_shape)
    keep = (bins - mode).abs() <= radius_bins
    local = probs * keep.to(dtype=probs.dtype)
    local = local / local.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    centers = torch.arange(
        x_bins,
        device=logits.device,
        dtype=logits.dtype,
    )
    expected = (local * centers).sum(dim=-1)
    return expected * (float(input_w) / float(x_bins))


@dataclass
class DecodeStats:
    thresholds: tuple[float, ...]
    best_ious: list[float] = field(default_factory=list)
    assigned_row_abs_error_total: float = 0.0
    assigned_row_count: int = 0

    def update_lane(
        self,
        candidates: torch.Tensor,
        *,
        gt_x: torch.Tensor,
        valid: torch.Tensor,
        assigned_candidate: torch.Tensor | None,
        line_width: float,
    ) -> None:
        values = line_iou_against_gt(
            candidates,
            gt_x,
            valid,
            line_width=float(line_width),
        )
        self.best_ious.append(float(values.max()) if values.numel() else 0.0)
        if assigned_candidate is None:
            return
        error = (
            assigned_candidate[valid].float() - gt_x[valid].float()
        ).abs()
        finite = torch.isfinite(error)
        if bool(finite.any()):
            self.assigned_row_abs_error_total += float(error[finite].sum())
            self.assigned_row_count += int(finite.sum())

    def summary(self) -> dict[str, float | int]:
        count = max(len(self.best_ious), 1)
        sorted_ious = sorted(self.best_ious)
        return {
            "gt_lanes": len(self.best_ious),
            **{
                f"raw_recall@{threshold:.2f}": (
                    sum(value >= threshold for value in self.best_ious) / count
                )
                for threshold in self.thresholds
            },
            "mean_best_iou": sum(self.best_ious) / count,
            "median_best_iou": (
                sorted_ious[len(sorted_ious) // 2] if sorted_ious else 0.0
            ),
            "assigned_row_mae_px": (
                self.assigned_row_abs_error_total
                / max(self.assigned_row_count, 1)
            ),
        }


@dataclass
class DistributionStats:
    count: int = 0
    entropy_total: float = 0.0
    top1_total: float = 0.0
    top2_total: float = 0.0
    top1_top2_distance_total: float = 0.0
    expected_mode_distance_total: float = 0.0
    gt_count: int = 0
    gt_nearest_probability_total: float = 0.0
    gt_dfl_probability_total: float = 0.0
    gt_mode_error_total: float = 0.0
    gt_expected_error_total: float = 0.0
    gt_rank_histogram: list[int] = field(default_factory=list)
    gt_window_mass_totals: dict[int, float] = field(
        default_factory=lambda: {1: 0.0, 2: 0.0, 4: 0.0, 8: 0.0}
    )

    def update(
        self,
        logits: torch.Tensor,
        *,
        valid_rows: torch.Tensor,
        input_w: int,
        x_bins: int,
        gt_x: torch.Tensor | None = None,
    ) -> None:
        selected = logits[valid_rows].float()
        if selected.numel() == 0:
            return
        probs = torch.softmax(selected, dim=-1)
        entropy = -(
            probs * probs.clamp_min(1e-12).log()
        ).sum(dim=-1) / math.log(max(x_bins, 2))
        top_values, top_indices = probs.topk(k=2, dim=-1)
        expected = soft_expected_x(
            selected,
            input_w=input_w,
            x_bins=x_bins,
        )
        mode = top_indices[:, 0].float() * (float(input_w) / float(x_bins))
        self.count += int(selected.shape[0])
        self.entropy_total += float(entropy.sum())
        self.top1_total += float(top_values[:, 0].sum())
        self.top2_total += float(top_values[:, 1].sum())
        self.top1_top2_distance_total += float(
            (top_indices[:, 0] - top_indices[:, 1]).abs().float().sum()
            * (float(input_w) / float(x_bins))
        )
        self.expected_mode_distance_total += float((expected - mode).abs().sum())
        if gt_x is None:
            return

        selected_gt = gt_x[valid_rows].float()
        finite = torch.isfinite(selected_gt)
        if not bool(finite.any()):
            return
        selected_gt = selected_gt[finite]
        probs = probs[finite]
        expected = expected[finite]
        top_indices = top_indices[finite]
        bin_width = float(input_w) / float(x_bins)
        target_bin = (selected_gt / bin_width).clamp(0.0, float(x_bins - 1))
        nearest = target_bin.round().long()
        nearest_probability = probs.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
        rank = (probs > nearest_probability.unsqueeze(-1)).sum(dim=-1) + 1

        if not self.gt_rank_histogram:
            self.gt_rank_histogram = [0] * (x_bins + 1)
        rank_counts = torch.bincount(rank, minlength=x_bins + 1).cpu().tolist()
        for index, value in enumerate(rank_counts[: x_bins + 1]):
            self.gt_rank_histogram[index] += int(value)

        left = target_bin.floor().long()
        right = (left + 1).clamp(max=x_bins - 1)
        right_weight = target_bin - left.to(dtype=target_bin.dtype)
        left_weight = 1.0 - right_weight
        same = right == left
        left_weight = torch.where(same, torch.ones_like(left_weight), left_weight)
        right_weight = torch.where(same, torch.zeros_like(right_weight), right_weight)
        left_probability = probs.gather(-1, left.unsqueeze(-1)).squeeze(-1)
        right_probability = probs.gather(-1, right.unsqueeze(-1)).squeeze(-1)
        dfl_probability = (
            left_weight * left_probability + right_weight * right_probability
        )

        bins = torch.arange(
            x_bins,
            device=probs.device,
            dtype=target_bin.dtype,
        ).view(1, -1)
        for radius_bins in self.gt_window_mass_totals:
            inside = (bins - target_bin.unsqueeze(-1)).abs() <= float(radius_bins)
            self.gt_window_mass_totals[radius_bins] += float(
                (probs * inside.to(dtype=probs.dtype)).sum()
            )

        mode = top_indices[:, 0].float() * bin_width
        self.gt_count += int(selected_gt.numel())
        self.gt_nearest_probability_total += float(nearest_probability.sum())
        self.gt_dfl_probability_total += float(dfl_probability.sum())
        self.gt_mode_error_total += float((mode - selected_gt).abs().sum())
        self.gt_expected_error_total += float((expected - selected_gt).abs().sum())

    def _rank_quantile(self, quantile: float) -> int:
        if self.gt_count <= 0 or not self.gt_rank_histogram:
            return 0
        target = max(1, int(math.ceil(float(quantile) * self.gt_count)))
        running = 0
        for rank, count in enumerate(self.gt_rank_histogram):
            running += int(count)
            if running >= target:
                return int(rank)
        return len(self.gt_rank_histogram) - 1

    def summary(self) -> dict[str, float | int]:
        count = max(self.count, 1)
        gt_count = max(self.gt_count, 1)
        return {
            "valid_rows": self.count,
            # Backward-compatible name from the first version of this audit.
            "valid_assigned_rows": self.count,
            "mean_normalized_entropy": self.entropy_total / count,
            "mean_top1_probability": self.top1_total / count,
            "mean_top2_probability": self.top2_total / count,
            "mean_top1_top2_distance_px": (
                self.top1_top2_distance_total / count
            ),
            "mean_expected_to_mode_distance_px": (
                self.expected_mode_distance_total / count
            ),
            "valid_gt_rows": self.gt_count,
            "mean_probability_at_nearest_gt_bin": (
                self.gt_nearest_probability_total / gt_count
            ),
            "mean_probability_at_dfl_target": (
                self.gt_dfl_probability_total / gt_count
            ),
            "mean_mode_abs_error_px": self.gt_mode_error_total / gt_count,
            "mean_expected_abs_error_px": self.gt_expected_error_total / gt_count,
            "mean_gt_bin_rank": (
                sum(
                    rank * count
                    for rank, count in enumerate(self.gt_rank_histogram)
                )
                / gt_count
            ),
            "median_gt_bin_rank": self._rank_quantile(0.5),
            "p90_gt_bin_rank": self._rank_quantile(0.9),
            **{
                f"fraction_gt_bin_in_top{k}": (
                    sum(self.gt_rank_histogram[1 : k + 1]) / gt_count
                    if self.gt_rank_histogram
                    else 0.0
                )
                for k in (1, 5, 10, 20)
            },
            **{
                f"mean_gt_probability_mass_within_{radius}_bins": (
                    total / gt_count
                )
                for radius, total in self.gt_window_mass_totals.items()
            },
        }


def _make_decoders(
    temperatures: tuple[float, ...],
    local_mode_radii: tuple[int, ...],
    *,
    input_w: int,
    x_bins: int,
):
    decoders: dict[str, Any] = {
        "argmax": lambda logits: decode_argmax(
            logits,
            input_w=input_w,
            x_bins=x_bins,
        ),
    }
    for temperature in temperatures:
        name = f"expected_t{temperature:g}"
        decoders[name] = lambda logits, value=temperature: soft_expected_x(
            logits,
            input_w=input_w,
            x_bins=x_bins,
            temperature=float(value),
        )
    for radius in local_mode_radii:
        name = f"local_mode_r{radius}"
        decoders[name] = lambda logits, value=radius: (
            decode_local_mode_expectation(
                logits,
                radius_bins=int(value),
                input_w=input_w,
                x_bins=x_bins,
            )
        )
    return decoders


def _prediction_stages(
    outputs: dict[str, Any],
) -> list[tuple[str, dict[str, torch.Tensor]]]:
    stages: list[tuple[str, dict[str, torch.Tensor]]] = []
    auxiliary = outputs.get("aux_outputs")
    if isinstance(auxiliary, (list, tuple)):
        for index, stage in enumerate(auxiliary, start=1):
            if isinstance(stage, dict) and "row_x_logits" in stage:
                stages.append((f"L{index}", stage))
    stages.append((f"L{len(stages) + 1}", outputs))
    return stages


def _new_distribution_buckets() -> dict[str, DistributionStats]:
    buckets: dict[str, DistributionStats] = {}
    for source in ("matched_candidate", "best_candidate"):
        buckets[f"{source}/all"] = DistributionStats()
        for threshold in (0.5, 0.7):
            buckets[f"{source}/baseline_hit@{threshold:.2f}"] = DistributionStats()
            buckets[f"{source}/baseline_miss@{threshold:.2f}"] = DistributionStats()
    return buckets


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    cfg = _prepare_config(
        args.config,
        dataset_root=args.dataset_root,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(cfg)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model = model.to(device).eval()
    head = model.structured_query_head
    if head is None:
        raise ValueError("Checkpoint must use a structured query head")
    matcher = build_matcher(cfg)
    group_size = int(head.num_instances) // int(head.num_groups)
    input_w = int(head.input_w)
    x_bins = int(head.x_bins)
    thresholds = tuple(float(value) for value in args.iou_thresholds)
    decoders = _make_decoders(
        tuple(float(value) for value in args.temperatures),
        tuple(int(value) for value in args.local_mode_radii),
        input_w=input_w,
        x_bins=x_bins,
    )
    stats_by_stage: dict[str, dict[str, DecodeStats]] = {}
    distribution_by_stage: dict[str, dict[str, DistributionStats]] = {}
    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
    total = min(len(loader), int(args.max_batches)) if args.max_batches > 0 else len(loader)
    images_seen = 0
    for batch_index, (images, targets, _metas) in enumerate(
        tqdm(loader, total=total, desc="row distribution decoding", ncols=100)
    ):
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        images = images.to(device)
        targets = nested_to_device(targets, device)
        with torch.no_grad(), _amp_context(device, amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
            previous_intermediate_supervision = bool(head.intermediate_supervision)
            head.intermediate_supervision = True
            try:
                outputs = head(
                    encoded["features"],
                    multi_scale_features=encoded.get("multi_scale_features"),
                    inference_only=False,
                )
            finally:
                head.intermediate_supervision = previous_intermediate_supervision

        for stage_name, stage in _prediction_stages(outputs):
            stage_stats = stats_by_stage.setdefault(
                stage_name,
                {
                    name: DecodeStats(thresholds=thresholds)
                    for name in decoders
                },
            )
            stage_distributions = distribution_by_stage.setdefault(
                stage_name,
                _new_distribution_buckets(),
            )
            logits = stage["row_x_logits"].detach().float()
            decoded = {
                name: decoder(logits) for name, decoder in decoders.items()
            }
            matches = matcher(stage, targets)
            baseline = decoded["expected_t1"]
            for image_index, (target, match) in enumerate(zip(targets, matches)):
                gt_x_all = target["x_rows"].float()
                valid_all = target["valid_mask"].bool()
                assignments = _group_zero_assignments(
                    match,
                    group_size=group_size,
                )
                for gt_index in range(int(gt_x_all.shape[0])):
                    valid = valid_all[gt_index]
                    if int(valid.sum()) < 5:
                        continue
                    gt_x = gt_x_all[gt_index]
                    assigned_index = assignments.get(gt_index)
                    baseline_iou = 0.0
                    best_index: int | None = None
                    baseline_values = line_iou_against_gt(
                        baseline[image_index, :group_size],
                        gt_x,
                        valid,
                        line_width=float(args.line_width),
                    )
                    if baseline_values.numel():
                        baseline_iou = float(baseline_values.max())
                        best_index = int(baseline_values.argmax())
                    for name, candidates in decoded.items():
                        assigned_candidate = (
                            candidates[image_index, assigned_index]
                            if assigned_index is not None
                            else None
                        )
                        stage_stats[name].update_lane(
                            candidates[image_index, :group_size],
                            gt_x=gt_x,
                            valid=valid,
                            assigned_candidate=assigned_candidate,
                            line_width=float(args.line_width),
                        )

                    distribution_sources = {
                        "matched_candidate": assigned_index,
                        "best_candidate": best_index,
                    }
                    for source_name, candidate_index in distribution_sources.items():
                        if candidate_index is None:
                            continue
                        candidate_logits = logits[image_index, candidate_index]
                        base_key = f"{source_name}/all"
                        stage_distributions[base_key].update(
                            candidate_logits,
                            valid_rows=valid,
                            input_w=input_w,
                            x_bins=x_bins,
                            gt_x=gt_x,
                        )
                        for threshold in (0.5, 0.7):
                            status = "hit" if baseline_iou >= threshold else "miss"
                            bucket = (
                                f"{source_name}/baseline_{status}@{threshold:.2f}"
                            )
                            stage_distributions[bucket].update(
                                candidate_logits,
                                valid_rows=valid,
                                input_w=input_w,
                                x_bins=x_bins,
                                gt_x=gt_x,
                            )
        images_seen += int(images.shape[0])

    stage_payload: dict[str, dict[str, Any]] = {}
    for stage_name, stage_stats in stats_by_stage.items():
        summaries = {
            name: value.summary() for name, value in stage_stats.items()
        }
        baseline_summary = summaries["expected_t1"]
        baseline_recall = {
            threshold: float(
                baseline_summary[f"raw_recall@{threshold:.2f}"]
            )
            for threshold in thresholds
        }
        comparisons: dict[str, dict[str, float]] = {}
        for name, summary in summaries.items():
            comparisons[name] = {
                **{
                    f"recall_gain@{threshold:.2f}": (
                        float(summary[f"raw_recall@{threshold:.2f}"])
                        - baseline_recall[threshold]
                    )
                    for threshold in thresholds
                },
                "assigned_row_mae_gain_px": (
                    float(baseline_summary["assigned_row_mae_px"])
                    - float(summary["assigned_row_mae_px"])
                ),
            }
        stage_payload[stage_name] = {
            "decode_metrics": summaries,
            "comparisons_to_expected_t1": comparisons,
            "distribution_shape": {
                name: value.summary()
                for name, value in distribution_by_stage[stage_name].items()
            },
        }

    if not stage_payload:
        raise RuntimeError("No decoder stages were produced")
    final_layer = list(stage_payload)[-1]
    final_payload = stage_payload[final_layer]
    payload = {
        "diagnostic_only": True,
        "warning": (
            "Decode variants and GT-bin statistics are frozen-checkpoint "
            "raw-proposal diagnostics, not validation-selected benchmark "
            "results. GT-bin metrics inspect existing logits and do not "
            "inject GT into decoded predictions."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "group_size": group_size,
        "input_w": input_w,
        "x_bins": x_bins,
        "line_width": float(args.line_width),
        "decoder_layers": stage_payload,
        "final_layer": final_layer,
        # Retain the original top-level keys for scripts that consumed the
        # first version of this diagnostic.
        "decode_metrics": final_payload["decode_metrics"],
        "comparisons_to_expected_t1": (
            final_payload["comparisons_to_expected_t1"]
        ),
        "distribution_shape": final_payload["distribution_shape"],
    }
    console_layers: dict[str, Any] = {}
    for stage_name, stage in stage_payload.items():
        baseline = stage["decode_metrics"]["expected_t1"]
        best_050_name, best_050 = max(
            stage["comparisons_to_expected_t1"].items(),
            key=lambda item: float(item[1]["recall_gain@0.50"]),
        )
        best_070_name, best_070 = max(
            stage["comparisons_to_expected_t1"].items(),
            key=lambda item: float(item[1]["recall_gain@0.70"]),
        )
        miss_050 = stage["distribution_shape"][
            "best_candidate/baseline_miss@0.50"
        ]
        miss_070 = stage["distribution_shape"][
            "best_candidate/baseline_miss@0.70"
        ]
        console_layers[stage_name] = {
            "expected_t1": {
                "raw_recall@0.50": baseline["raw_recall@0.50"],
                "raw_recall@0.70": baseline["raw_recall@0.70"],
                "assigned_row_mae_px": baseline["assigned_row_mae_px"],
            },
            "best_decode_gain@0.50": {
                "decoder": best_050_name,
                **best_050,
            },
            "best_decode_gain@0.70": {
                "decoder": best_070_name,
                **best_070,
            },
            "best_candidate_baseline_miss@0.50": {
                key: miss_050[key]
                for key in (
                    "valid_gt_rows",
                    "mean_expected_abs_error_px",
                    "mean_mode_abs_error_px",
                    "median_gt_bin_rank",
                    "fraction_gt_bin_in_top5",
                    "mean_gt_probability_mass_within_4_bins",
                )
            },
            "best_candidate_baseline_miss@0.70": {
                key: miss_070[key]
                for key in (
                    "valid_gt_rows",
                    "mean_expected_abs_error_px",
                    "mean_mode_abs_error_px",
                    "median_gt_bin_rank",
                    "fraction_gt_bin_in_top5",
                    "mean_gt_probability_mass_within_4_bins",
                )
            },
        }
    print(
        json.dumps(
            {
                "checkpoint_iteration": iteration,
                "images": images_seen,
                "group_size": group_size,
                "decoder_layers": console_layers,
            },
            indent=2,
        )
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
