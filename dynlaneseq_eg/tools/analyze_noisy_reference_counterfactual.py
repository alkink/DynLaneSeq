from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_gt_curve_aligned_features import (
    BucketStats,
    CurveAlignedSequenceProbe,
    _base_iou_buckets,
    _build_lane_examples,
    _extract_frozen_sources,
    _paired_line_iou,
    _profiles_for_source,
    _selected_row_indices,
    seed_everything,
)


CONDITION_NAMES = (
    "reference_only",
    "correct_p2",
    "wrong_image_p2",
    "zero_p2",
    "horizontal_mean_p2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained GT-shifted curve probe under correct, wrong, "
            "zero, and horizontally collapsed P2. The noisy GT reference is "
            "identical in every condition; only image evidence changes."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=16)
    parser.add_argument(
        "--eval-sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--base-hit-threshold", type=float, default=0.5)
    parser.add_argument("--eval-shift-px", type=float, default=32.0)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _counterfactual_p2(p2: torch.Tensor) -> dict[str, torch.Tensor]:
    """Keep geometry fixed while replacing only image-derived P2 content."""

    if p2.ndim != 4:
        raise ValueError("p2 must have shape [batch,channels,height,width]")
    if int(p2.shape[0]) < 2:
        raise ValueError("wrong-image control requires eval_batch_size >= 2")
    return {
        "correct_p2": p2,
        "wrong_image_p2": torch.roll(p2, shifts=1, dims=0),
        "zero_p2": torch.zeros_like(p2),
        "horizontal_mean_p2": p2.mean(dim=-1, keepdim=True).expand_as(p2),
    }


def _condition_gate(
    summaries: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    correct = summaries["correct_p2"]["base_miss"]
    controls = {
        name: summaries[name]["base_miss"]
        for name in ("reference_only", "wrong_image_p2", "zero_p2", "horizontal_mean_p2")
    }
    largest_control_recall = max(
        float(summary["corrected_recall_050"]) for summary in controls.values()
    )
    largest_control_iou_gain = max(
        float(summary["mean_iou_gain"]) for summary in controls.values()
    )
    largest_control_direction = max(
        float(summary["direction_accuracy"]) for summary in controls.values()
    )
    recall_increment = (
        float(correct["corrected_recall_050"]) - largest_control_recall
    )
    iou_increment = float(correct["mean_iou_gain"]) - largest_control_iou_gain
    direction_increment = (
        float(correct["direction_accuracy"]) - largest_control_direction
    )
    positive = (
        recall_increment >= 0.10
        and iou_increment >= 0.05
        and direction_increment >= 0.10
    )
    return {
        "positive_gate": bool(positive),
        "correct_p2_base_miss_recall_050": float(
            correct["corrected_recall_050"]
        ),
        "largest_control_base_miss_recall_050": largest_control_recall,
        "correct_minus_largest_control_recall_050_points": (
            100.0 * recall_increment
        ),
        "correct_minus_largest_control_mean_iou_gain": iou_increment,
        "correct_minus_largest_control_direction_accuracy": direction_increment,
        "gate_definition": (
            "On detector-missed lanes, correct-image P2 must exceed every "
            "reference/image-removed control by at least 10 recall@0.50 points, "
            "0.05 mean-IoU gain, and 0.10 residual-direction accuracy."
        ),
    }


def _bootstrap_mean_ci(
    deltas: torch.Tensor,
    *,
    seed: int,
    samples: int = 4000,
) -> tuple[float, float]:
    values = deltas.detach().float().cpu().flatten()
    if int(values.numel()) == 0:
        return 0.0, 0.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        int(values.numel()),
        (int(samples), int(values.numel())),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    return (
        float(torch.quantile(means, 0.025)),
        float(torch.quantile(means, 0.975)),
    )


def _paired_lane_visual_summary(
    lane_values: dict[tuple[int, int, int], dict[str, list[float]]],
    *,
    seed: int,
) -> dict[str, Any]:
    controls = (
        "reference_only",
        "wrong_image_p2",
        "zero_p2",
        "horizontal_mean_p2",
    )
    result: dict[str, Any] = {"base_miss_lanes": len(lane_values), "controls": {}}
    for control_index, control in enumerate(controls):
        correct_means: list[float] = []
        control_means: list[float] = []
        correct_best: list[float] = []
        control_best: list[float] = []
        correct_only_patterns = 0
        control_only_patterns = 0
        for condition_values in lane_values.values():
            correct = condition_values["correct_p2"]
            compared = condition_values[control]
            if len(correct) != len(compared):
                raise RuntimeError("paired counterfactual pattern counts differ")
            correct_tensor = torch.tensor(correct, dtype=torch.float32)
            control_tensor = torch.tensor(compared, dtype=torch.float32)
            correct_means.append(float(correct_tensor.mean()))
            control_means.append(float(control_tensor.mean()))
            correct_best.append(float(correct_tensor.max()))
            control_best.append(float(control_tensor.max()))
            correct_hit = correct_tensor >= 0.5
            control_hit = control_tensor >= 0.5
            correct_only_patterns += int((correct_hit & ~control_hit).sum())
            control_only_patterns += int((control_hit & ~correct_hit).sum())
        correct_mean = torch.tensor(correct_means, dtype=torch.float32)
        control_mean = torch.tensor(control_means, dtype=torch.float32)
        delta = correct_mean - control_mean
        ci_low, ci_high = _bootstrap_mean_ci(
            delta,
            seed=int(seed) + control_index,
        )
        correct_best_tensor = torch.tensor(correct_best, dtype=torch.float32)
        control_best_tensor = torch.tensor(control_best, dtype=torch.float32)
        result["controls"][control] = {
            "correct_p2_lane_mean_iou": float(correct_mean.mean())
            if int(correct_mean.numel())
            else 0.0,
            "control_lane_mean_iou": float(control_mean.mean())
            if int(control_mean.numel())
            else 0.0,
            "paired_lane_mean_iou_delta": float(delta.mean())
            if int(delta.numel())
            else 0.0,
            "paired_lane_mean_iou_delta_bootstrap_95ci": [ci_low, ci_high],
            "correct_p2_lane_win_fraction": float((delta > 0.0).float().mean())
            if int(delta.numel())
            else 0.0,
            "correct_p2_best_of_four_recall_050": float(
                (correct_best_tensor >= 0.5).float().mean()
            )
            if int(correct_best_tensor.numel())
            else 0.0,
            "control_best_of_four_recall_050": float(
                (control_best_tensor >= 0.5).float().mean()
            )
            if int(control_best_tensor.numel())
            else 0.0,
            "correct_minus_control_best_of_four_recall_050_points": (
                100.0
                * float(
                    (
                        (correct_best_tensor >= 0.5).float()
                        - (control_best_tensor >= 0.5).float()
                    ).mean()
                )
                if int(correct_best_tensor.numel())
                else 0.0
            ),
            "correct_only_pattern_hits_050": correct_only_patterns,
            "control_only_pattern_hits_050": control_only_patterns,
        }
    return result


def _load_p2_probe(
    *,
    path: str,
    common_channels: int,
    device: torch.device,
) -> tuple[CurveAlignedSequenceProbe, dict[str, Any], int]:
    payload = torch.load(path, map_location="cpu")
    saved_args = payload.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError("probe checkpoint is missing saved args")
    offsets = [float(value) for value in saved_args["offsets_px"]]
    probe = CurveAlignedSequenceProbe(
        common_channels=int(common_channels),
        hidden_dim=int(saved_args["hidden_dim"]),
        num_rows=int(saved_args["probe_rows"]),
        offsets_px=offsets,
        max_scales=3,
        num_layers=int(saved_args["probe_layers"]),
    )
    states = payload.get("probes")
    if not isinstance(states, dict) or "p2" not in states:
        raise ValueError("probe checkpoint is missing the trained P2 probe")
    probe.load_state_dict(states["p2"], strict=True)
    probe = probe.to(device).eval()
    return probe, saved_args, sum(parameter.numel() for parameter in probe.parameters())


@torch.inference_mode()
def _evaluate(
    *,
    model: nn.Module,
    probe: CurveAlignedSequenceProbe,
    loader,
    device: torch.device,
    channels_last: bool,
    amp_dtype: torch.dtype | None,
    row_indices: torch.Tensor,
    offsets: torch.Tensor,
    common_channels: int,
    input_w: int,
    input_h: int,
    group_size: int,
    max_batches: int,
    line_width: float,
    base_hit_threshold: float,
    eval_shift_px: float,
    seed: int,
) -> dict[str, Any]:
    stats = {
        condition: {
            "all": BucketStats(),
            "base_hit": BucketStats(),
            "base_miss": BucketStats(),
        }
        for condition in CONDITION_NAMES
    }
    images_seen = 0
    gt_lanes = 0
    base_hits = 0
    base_miss_lane_values: dict[
        tuple[int, int, int],
        dict[str, list[float]],
    ] = {}
    displayed_batches = len(loader)
    if int(max_batches) > 0:
        displayed_batches = min(displayed_batches, int(max_batches))
    progress = tqdm(
        enumerate(loader),
        total=displayed_batches,
        desc="noisy-reference P2 counterfactual",
        ncols=100,
    )
    for batch_index, (images, targets, _metas) in progress:
        if int(max_batches) > 0 and batch_index >= int(max_batches):
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last if channels_last else torch.contiguous_format
            ),
        )
        targets = nested_to_device(targets, device)
        with _amp_context(device, amp_dtype):
            _sources, p2 = _extract_frozen_sources(model, images)
            predictions = model.structured_query_head(
                p2,
                inference_only=True,
            )["pred_x_rows"]
        base_iou = _base_iou_buckets(
            predictions,
            targets,
            group_size=group_size,
            line_width=float(line_width),
        )
        gt_lanes += len(base_iou)
        base_hits += sum(
            value >= float(base_hit_threshold) for value in base_iou.values()
        )
        examples = _build_lane_examples(
            targets,
            row_indices=row_indices,
            input_w=input_w,
            max_offset=float(offsets.abs().max()),
            training=False,
            max_train_shift=0.0,
            eval_shift_px=float(eval_shift_px),
        )
        if examples is None:
            images_seen += int(images.shape[0])
            continue

        base_values = torch.tensor(
            [
                base_iou.get((int(image_index), int(lane_index)), 0.0)
                for image_index, lane_index in zip(
                    examples.image_indices.tolist(),
                    examples.lane_indices.tolist(),
                )
            ],
            device=device,
            dtype=torch.float32,
        )
        bucket_masks = {
            "all": torch.ones(examples.count, device=device, dtype=torch.bool),
            "base_hit": base_values >= float(base_hit_threshold),
            "base_miss": base_values < float(base_hit_threshold),
        }
        anchor_iou = _paired_line_iou(
            examples.anchor_x,
            examples.gt_x,
            examples.valid,
            line_width=float(line_width),
        )
        counterfactuals = _counterfactual_p2(p2)
        outputs_by_condition: dict[str, dict[str, torch.Tensor]] = {}
        for condition, feature in counterfactuals.items():
            profiles, scale_mask = _profiles_for_source(
                [feature],
                examples,
                offsets,
                common_channels=common_channels,
                max_scales=3,
                input_w=input_w,
                input_h=input_h,
            )
            outputs_by_condition[condition] = probe(
                profiles,
                scale_mask,
                examples.valid,
            )

        # The balanced +/-32 global and linear perturbations make an unchanged
        # noisy reference the leakage-free no-image baseline.
        zeros = torch.zeros_like(examples.anchor_x)
        outputs_by_condition["reference_only"] = {
            "residual": zeros,
            "scale_weights": torch.zeros(
                examples.count,
                int(examples.anchor_x.shape[1]),
                int(offsets.numel()),
                3,
                device=device,
                dtype=torch.float32,
            ),
        }
        outputs_by_condition["reference_only"]["scale_weights"][..., 0] = 1.0

        corrected_iou_by_condition: dict[str, torch.Tensor] = {}
        for condition in CONDITION_NAMES:
            outputs = outputs_by_condition[condition]
            corrected = examples.anchor_x + outputs["residual"]
            corrected_iou = _paired_line_iou(
                corrected,
                examples.gt_x,
                examples.valid,
                line_width=float(line_width),
            )
            corrected_iou_by_condition[condition] = corrected_iou
            values = {
                "anchor": examples.anchor_x,
                "corrected": corrected,
                "gt": examples.gt_x,
                "valid": examples.valid,
                "target_residual": examples.target_residual,
                "predicted_residual": outputs["residual"],
                "scale_weights": outputs["scale_weights"],
                "anchor_iou": anchor_iou,
                "corrected_iou": corrected_iou,
            }
            for bucket_name, selected in bucket_masks.items():
                stats[condition][bucket_name].update_batch(
                    **values,
                    selected=selected,
                )
        image_values = examples.image_indices.detach().cpu().tolist()
        lane_indices = examples.lane_indices.detach().cpu().tolist()
        miss_values = bucket_masks["base_miss"].detach().cpu().tolist()
        condition_cpu = {
            condition: values.detach().cpu().tolist()
            for condition, values in corrected_iou_by_condition.items()
        }
        for example_index, (local_image, lane_index, is_base_miss) in enumerate(
            zip(image_values, lane_indices, miss_values)
        ):
            if not bool(is_base_miss):
                continue
            key = (batch_index, int(local_image), int(lane_index))
            entry = base_miss_lane_values.setdefault(
                key,
                {condition: [] for condition in CONDITION_NAMES},
            )
            for condition in CONDITION_NAMES:
                entry[condition].append(
                    float(condition_cpu[condition][example_index])
                )
        images_seen += int(images.shape[0])

    summaries = {
        condition: {
            bucket: bucket_stats.summary()
            for bucket, bucket_stats in buckets.items()
        }
        for condition, buckets in stats.items()
    }
    return {
        "images": images_seen,
        "gt_lanes": gt_lanes,
        "base_group0_recall_050": base_hits / max(gt_lanes, 1),
        "conditions": summaries,
        "decision": _condition_gate(summaries),
        "paired_lane_analysis": _paired_lane_visual_summary(
            base_miss_lane_values,
            seed=int(seed),
        ),
    }


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    channels_last = (
        bool(cfg.get("training", {}).get("channels_last", False))
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    head = model.structured_query_head
    if head is None:
        raise ValueError("counterfactual requires structured queries")

    input_w = int(model_cfg.get("input_w", head.input_w))
    input_h = int(model_cfg.get("input_h", 288))
    fpn_channels = int(model_cfg.get("fpn_channels", 128))
    dim = int(model_cfg.get("dim", fpn_channels))
    common_channels = max(
        int(model.encoder.backbone.out_channels["c2"]),
        fpn_channels,
        dim,
    )
    probe, saved_args, parameter_count = _load_p2_probe(
        path=args.probe_checkpoint,
        common_channels=common_channels,
        device=device,
    )
    offsets = torch.tensor(
        saved_args["offsets_px"],
        device=device,
        dtype=torch.float32,
    )
    row_indices = _selected_row_indices(
        int(head.num_rows),
        int(saved_args["probe_rows"]),
        device,
    )
    group_size = int(head.num_instances) // max(int(head.num_groups), 1)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)

    loader = build_dataloader(cfg, split="val", training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=str(args.eval_sample_strategy),
        max_batches=int(args.eval_max_batches),
        num_workers=int(args.num_workers),
    )
    evaluation = _evaluate(
        model=model,
        probe=probe,
        loader=loader,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        row_indices=row_indices,
        offsets=offsets,
        common_channels=common_channels,
        input_w=input_w,
        input_h=input_h,
        group_size=group_size,
        max_batches=int(args.eval_max_batches),
        line_width=float(args.line_width),
        base_hit_threshold=float(args.base_hit_threshold),
        eval_shift_px=float(args.eval_shift_px),
        seed=int(args.seed),
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "GT plus balanced synthetic residuals define the noisy reference. "
            "This test only asks whether correct P2 contributes visual correction "
            "beyond reference and image-removed controls."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "probe_checkpoint": args.probe_checkpoint,
        "probe_training_steps": int(saved_args["train_steps"]),
        "probe_parameter_count": int(parameter_count),
        "eval_split": "val",
        "eval_sample_strategy": str(args.eval_sample_strategy),
        "sampled_dataset_indices": sampled_indices,
        "noisy_reference_shift_px": float(args.eval_shift_px),
        "noisy_reference_patterns_px": [
            f"global {-float(args.eval_shift_px):g}",
            f"global +{float(args.eval_shift_px):g}",
            (
                f"linear {-float(args.eval_shift_px):g} to "
                f"+{float(args.eval_shift_px):g}"
            ),
            (
                f"linear +{float(args.eval_shift_px):g} to "
                f"{-float(args.eval_shift_px):g}"
            ),
        ],
        "conditions": {
            "reference_only": "unchanged noisy reference; no learned correction",
            "correct_p2": "trained P2 probe with the matching image",
            "wrong_image_p2": "same probe and reference with batch-rolled P2",
            "zero_p2": "same probe and reference with zero P2",
            "horizontal_mean_p2": "same probe with horizontal position removed",
        },
        "evaluation": evaluation,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
