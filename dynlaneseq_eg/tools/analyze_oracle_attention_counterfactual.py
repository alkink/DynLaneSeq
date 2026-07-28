from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.analyze_attention_acquisition import (
    _build_corridor_indicator,
    _structured_forward_with_attention,
)
from dynlaneseq_eg.tools.analyze_attention_alternative_proposals import (
    AlternativeStats,
    _attention_curves,
    _maps_for_candidates,
    _model_score,
    _select_curves,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate image-grounded evidence from direct coordinate copying "
            "in GT-corridor-biased attention alternatives."
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
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--corridor-radius-px", type=float, default=16.0)
    parser.add_argument("--target-layer", type=int, default=3)
    parser.add_argument(
        "--layer-mode",
        choices=("single", "cascade"),
        default="cascade",
    )
    parser.add_argument("--bias-strength", type=float, default=2.0)
    parser.add_argument("--active-top-k", type=int, default=4)
    parser.add_argument("--quality-power", type=float, default=0.5)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _counterfactual_features(
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return controls that preserve tensor shape but remove correct imagery."""

    return {
        "correct_image": features,
        "wrong_image": torch.roll(features, shifts=1, dims=0),
        "zero_image": torch.zeros_like(features),
        "horizontal_mean": features.mean(dim=-1, keepdim=True).expand_as(
            features
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg.setdefault("dataloader", {})["persistent_workers"] = False
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
        raise ValueError("counterfactual audit requires structured_query")

    layer_count = len(head.layers)
    target_layer_index = int(args.target_layer) - 1
    if not 0 <= target_layer_index < layer_count:
        raise ValueError("--target-layer is outside the decoder")
    if args.layer_mode == "single":
        active_layers = (target_layer_index,)
    else:
        active_layers = tuple(range(target_layer_index, layer_count))
    attention_layer_index = int(active_layers[-1])

    matcher = build_matcher(cfg)
    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    num_instances = int(head.num_instances)
    group_size = num_instances // max(int(head.num_groups), 1)
    active_top_k = min(max(int(args.active_top_k), 1), group_size)
    input_w = float(head.input_w)
    methods = ("attention_peak_v5", "attention_peak_v11")

    stats: dict[str, AlternativeStats] = {
        f"{source}_{method}": AlternativeStats.create()
        for source in (
            "natural_correct_image",
            "oracle_correct_image",
            "oracle_wrong_image",
            "oracle_zero_image",
            "oracle_horizontal_mean",
        )
        for method in methods
    }
    images_seen = 0
    total = len(loader)
    if int(args.max_batches) > 0:
        total = min(total, int(args.max_batches))
    progress = tqdm(
        enumerate(loader),
        total=total,
        desc="oracle attention counterfactual",
        ncols=100,
    )
    for batch_index, (images, targets, _metas) in progress:
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last
                if channels_last
                else torch.contiguous_format
            ),
        )
        targets = nested_to_device(targets, device)
        with _amp_context(device, amp_dtype):
            features = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )["features"]
            baseline, _baseline_stages, natural_attention = (
                _structured_forward_with_attention(head, features)
            )
        matches = matcher(baseline, targets)
        corridor = _build_corridor_indicator(
            targets=targets,
            matches=matches,
            batch=int(images.shape[0]),
            instances=num_instances,
            rows=int(head.num_rows),
            x_bins=int(head.evidence_x_bins),
            group_size=group_size,
            input_w=input_w,
            radius_px=float(args.corridor_radius_px),
            device=features.device,
            dtype=features.dtype,
        )
        bias_by_layer = {
            layer_index: corridor * float(args.bias_strength)
            for layer_index in active_layers
        }

        baseline_x = baseline["pred_x_rows"][:, :group_size].float()
        baseline_maps = _maps_for_candidates(
            baseline_x,
            targets,
            line_width=float(args.line_width),
        )
        score = _model_score(
            baseline,
            quality_power=float(args.quality_power),
        )[:, :group_size]
        active_indices = score.topk(active_top_k, dim=-1).indices
        active_baseline = _select_curves(baseline_x, active_indices)

        attention_sources: dict[str, torch.Tensor] = {
            "natural_correct_image": natural_attention[attention_layer_index]
        }
        for source_name, source_features in _counterfactual_features(
            features
        ).items():
            with _amp_context(device, amp_dtype):
                _output, _stages, attention = (
                    _structured_forward_with_attention(
                        head,
                        source_features,
                        cross_bias_by_layer=bias_by_layer,
                    )
                )
            attention_sources[f"oracle_{source_name}"] = attention[
                attention_layer_index
            ]

        for source_name, attention in attention_sources.items():
            curves = _attention_curves(attention, input_w=input_w)
            for method_name in methods:
                selected_alternative = _select_curves(
                    curves[method_name][:, :group_size].float(),
                    active_indices,
                )
                fixed_eight = torch.cat(
                    [active_baseline, selected_alternative],
                    dim=1,
                )
                after_maps = _maps_for_candidates(
                    fixed_eight,
                    targets,
                    line_width=float(args.line_width),
                )
                stats[f"{source_name}_{method_name}"].update(
                    before_maps=baseline_maps,
                    after_maps=after_maps,
                    candidates_per_image=int(fixed_eight.shape[1]),
                )
        images_seen += int(images.shape[0])

    summaries = {key: value.summary() for key, value in stats.items()}
    correct = summaries["oracle_correct_image_attention_peak_v5"]
    controls = {
        name: summaries[f"oracle_{name}_attention_peak_v5"]
        for name in ("wrong_image", "zero_image", "horizontal_mean")
    }
    correct_gain = float(correct["net_gain_050_points"])
    largest_control_gain = max(
        float(value["net_gain_050_points"]) for value in controls.values()
    )
    visual_increment = correct_gain - largest_control_gain
    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "warning": (
            "The GT corridor is an oracle coordinate intervention. A gain "
            "that survives wrong/zero-image controls is direct coordinate "
            "copying, not evidence that P2 visual content supplied the lane."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "group_size": group_size,
        "active_top_k": active_top_k,
        "conditions": summaries,
        "decision_summary": {
            "oracle_correct_gain_050_points": correct_gain,
            "largest_image_removed_control_gain_050_points": (
                largest_control_gain
            ),
            "visual_increment_over_largest_control_points": visual_increment,
            "interpretation": (
                "If wrong/zero-image controls retain most of the oracle gain, "
                "the large attention-peak upper bound mainly re-reads injected "
                "GT coordinates. A substantial correct-image-only increment "
                "would instead support image-grounded acquisition."
            ),
        },
    }
    print(json.dumps(payload, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
