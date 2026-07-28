from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, input_to_grid, nested_to_device


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _prediction_layers(outputs: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    layers: dict[str, dict[str, torch.Tensor]] = {}
    auxiliary = outputs.get("aux_outputs")
    if isinstance(auxiliary, (list, tuple)):
        for index, stage in enumerate(auxiliary, start=1):
            if isinstance(stage, dict) and "pred_x_rows" in stage:
                layers[f"L{index}"] = stage
    layers[f"L{len(layers) + 1}"] = outputs
    return layers


def _sample_points(
    feature: torch.Tensor,
    x: torch.Tensor,
    *,
    input_w: int,
    input_h: int,
) -> torch.Tensor:
    """Sample [C,H,W] at x=[lanes,rows,offsets].

    The result is [lanes,rows,offsets,C]. Sampling is deliberately performed in
    FP32 because CUDA grid_sample does not support BF16 and because sub-pixel
    feature comparisons should not depend on the model autocast format.
    """
    if feature.ndim != 3 or x.ndim != 3:
        raise ValueError("Expected feature=[C,H,W] and x=[lanes,rows,offsets]")
    feature = feature.float()
    x = x.float()
    lanes, rows, offsets = x.shape
    y = fixed_y_rows(rows, input_h, device=x.device, dtype=x.dtype)
    y = y.view(1, rows, 1).expand(lanes, rows, offsets)
    grid = input_to_grid(x.clamp(0.0, float(input_w - 1)), y, input_w, input_h)
    sampled = F.grid_sample(
        feature.unsqueeze(0),
        grid.reshape(1, lanes * rows * offsets, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).reshape(
        lanes, rows, offsets, feature.shape[0]
    )


def _canonical_profile_input(
    profiles: torch.Tensor,
    *,
    center_index: int,
    common_channels: int,
) -> torch.Tensor:
    """Create an equal-width probe input without discarding source information.

    C2 commonly has 64 channels while projected P2 has 256. Each sampled vector
    is L2-normalized and lower-dimensional sources are zero-padded to the common
    width. Consequently both probes have exactly the same trainable parameter
    count, while P2 is never compressed to make the comparison easier for C2.
    """
    if profiles.ndim != 3:
        raise ValueError("profiles must have shape [rows, offsets, channels]")
    channels = int(profiles.shape[-1])
    if channels > int(common_channels):
        raise ValueError(
            f"source channels ({channels}) exceed common_channels ({common_channels}); "
            "increase --common-channels rather than silently compressing the source"
        )
    normalized = F.normalize(profiles.float(), p=2.0, dim=-1, eps=1e-6)
    if channels < int(common_channels):
        normalized = F.pad(normalized, (0, int(common_channels) - channels))
    relative = normalized - normalized[:, center_index : center_index + 1]
    return torch.cat((normalized, relative), dim=-1).flatten(1).contiguous()


class ResidualOffsetProbe(nn.Module):
    """One shared trunk with discrete and continuous residual readouts."""

    def __init__(self, input_dim: int, hidden_dim: int, classes: int, max_offset: float):
        super().__init__()
        self.max_offset = float(max_offset)
        self.trunk = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.classifier = nn.Linear(int(hidden_dim), int(classes))
        self.regressor = nn.Linear(int(hidden_dim), 1)
        # A residual probe must begin as a literal no-op. This prevents a
        # randomly initialized regressor from appearing worse than the anchor
        # merely because it starts by applying a large arbitrary correction.
        nn.init.zeros_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        logits = self.classifier(hidden)
        residual = torch.tanh(self.regressor(hidden).squeeze(-1)) * self.max_offset
        return logits, residual


def _balanced_accuracy(prediction: torch.Tensor, target: torch.Tensor, classes: int) -> float:
    recalls = []
    for class_index in range(int(classes)):
        selected = target == class_index
        if bool(selected.any()):
            recalls.append((prediction[selected] == target[selected]).float().mean())
    return float(torch.stack(recalls).mean()) if recalls else 0.0


def _summarize_predictions(
    class_prediction: torch.Tensor,
    residual_prediction: torch.Tensor,
    labels: torch.Tensor,
    residual_target: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {"rows": 0}
    class_prediction = class_prediction[mask]
    residual_prediction = residual_prediction[mask]
    labels = labels[mask]
    residual_target = residual_target[mask]
    class_offsets = offsets[class_prediction]
    anchor_error = residual_target.abs()
    class_error = (residual_target - class_offsets).abs()
    regression_error = (residual_target - residual_prediction).abs()
    oracle_error = (residual_target - offsets[labels]).abs()
    target_direction = torch.sign(offsets[labels])
    predicted_direction = torch.sign(class_offsets)
    return {
        "rows": count,
        "class_accuracy": float((class_prediction == labels).float().mean()),
        "class_balanced_accuracy": _balanced_accuracy(
            class_prediction, labels, int(offsets.numel())
        ),
        "class_direction_accuracy": float(
            (predicted_direction == target_direction).float().mean()
        ),
        "anchor_mae_px": float(anchor_error.mean()),
        "class_corrected_mae_px": float(class_error.mean()),
        "regression_corrected_mae_px": float(regression_error.mean()),
        "oracle_discrete_mae_px": float(oracle_error.mean()),
        "regression_mae_gain_px": float(anchor_error.mean() - regression_error.mean()),
        "anchor_within_4px": float((anchor_error <= 4.0).float().mean()),
        "regression_within_4px": float((regression_error <= 4.0).float().mean()),
        "anchor_within_8px": float((anchor_error <= 8.0).float().mean()),
        "regression_within_8px": float((regression_error <= 8.0).float().mean()),
    }


def _probe_metrics(
    class_prediction: torch.Tensor,
    residual_prediction: torch.Tensor,
    labels: torch.Tensor,
    residual_target: torch.Tensor,
    contrast: torch.Tensor,
    offsets: torch.Tensor,
    low_contrast_cut: float,
    high_contrast_cut: float,
) -> dict[str, dict[str, float | int]]:
    absolute_error = residual_target.abs()
    masks = {
        "all": torch.ones_like(labels, dtype=torch.bool),
        "error_0_4px": absolute_error < 4.0,
        "error_4_8px": (absolute_error >= 4.0) & (absolute_error < 8.0),
        "error_8_16px": (absolute_error >= 8.0) & (absolute_error < 16.0),
        "error_16px_plus": absolute_error >= 16.0,
        "low_contrast": contrast <= float(low_contrast_cut),
        "mid_contrast": (contrast > float(low_contrast_cut))
        & (contrast < float(high_contrast_cut)),
        "high_contrast": contrast >= float(high_contrast_cut),
    }
    return {
        name: _summarize_predictions(
            class_prediction,
            residual_prediction,
            labels,
            residual_target,
            offsets,
            mask,
        )
        for name, mask in masks.items()
    }


def _train_probe(
    profiles: torch.Tensor,
    labels: torch.Tensor,
    residual_target: torch.Tensor,
    groups: torch.Tensor,
    contrast: torch.Tensor,
    offsets: torch.Tensor,
    *,
    common_channels: int,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    unique_groups = groups.unique(sorted=True)
    if unique_groups.numel() < 5:
        raise RuntimeError("The probe requires at least five images; increase --max-batches")
    test_groups = unique_groups[torch.arange(unique_groups.numel()) % 5 == 0]
    is_test = (groups[:, None] == test_groups[None, :]).any(dim=1)
    is_train = ~is_test
    center_index = int(offsets.abs().argmin())
    inputs = _canonical_profile_input(
        profiles,
        center_index=center_index,
        common_channels=int(common_channels),
    )
    train_x = inputs[is_train]
    train_y = labels[is_train]
    train_residual = residual_target[is_train]
    test_x = inputs[is_test]
    test_y = labels[is_test]
    test_residual = residual_target[is_test]
    test_contrast = contrast[is_test]
    if train_y.numel() == 0 or test_y.numel() == 0:
        raise RuntimeError("Empty image-disjoint probe train/test split")

    torch.manual_seed(int(seed))
    probe = ResidualOffsetProbe(
        input_dim=int(train_x.shape[-1]),
        hidden_dim=int(hidden_dim),
        classes=int(offsets.numel()),
        max_offset=float(offsets.abs().max()),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    counts = torch.bincount(train_y, minlength=int(offsets.numel())).float()
    class_weights = torch.sqrt(counts.sum() / counts.clamp_min(1.0))
    class_weights = (class_weights / class_weights.mean()).to(device)
    scale = float(offsets.abs().max())
    generator = torch.Generator().manual_seed(int(seed))

    probe.train()
    last_loss = 0.0
    for _ in range(max(int(steps), 1)):
        take = torch.randint(
            0,
            train_y.numel(),
            (min(int(batch_size), train_y.numel()),),
            generator=generator,
        )
        x = train_x[take].to(device, non_blocking=True)
        y = train_y[take].to(device, non_blocking=True)
        residual = train_residual[take].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, residual_prediction = probe(x)
        class_loss = F.cross_entropy(logits, y, weight=class_weights)
        regression_loss = F.smooth_l1_loss(
            residual_prediction / scale,
            residual / scale,
            beta=0.125,
        )
        loss = class_loss + 2.0 * regression_loss
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())

    class_predictions = []
    residual_predictions = []
    probe.eval()
    with torch.inference_mode():
        for start in range(0, test_y.numel(), 2048):
            logits, residual = probe(test_x[start : start + 2048].to(device))
            class_predictions.append(logits.argmax(dim=-1).cpu())
            residual_predictions.append(residual.cpu())
    class_prediction = torch.cat(class_predictions)
    residual_prediction = torch.cat(residual_predictions)
    low_cut = float(torch.quantile(contrast[is_train], 1.0 / 3.0))
    high_cut = float(torch.quantile(contrast[is_train], 2.0 / 3.0))
    test_class_counts = torch.bincount(test_y, minlength=int(offsets.numel()))
    center_class = int(offsets.abs().argmin())
    always_center_prediction = torch.full_like(test_y, center_class)
    return {
        "source_channels": int(profiles.shape[-1]),
        "canonical_channels": int(common_channels),
        "probe_input_dim": int(train_x.shape[-1]),
        "probe_hidden_dim": int(hidden_dim),
        "probe_parameters": int(parameter_count),
        "probe_steps": int(steps),
        "final_train_loss": last_loss,
        "train_rows": int(train_y.numel()),
        "test_rows": int(test_y.numel()),
        "train_images": int(unique_groups.numel() - test_groups.numel()),
        "test_images": int(test_groups.numel()),
        "contrast_tertiles_from_train": [low_cut, high_cut],
        "test_class_counts": {
            f"{float(offset):+g}px": int(test_class_counts[index])
            for index, offset in enumerate(offsets)
        },
        "baselines": {
            "uniform_random_expected_accuracy": 1.0 / float(offsets.numel()),
            "always_center_accuracy": float((test_y == center_class).float().mean()),
            "always_center_balanced_accuracy": _balanced_accuracy(
                always_center_prediction, test_y, int(offsets.numel())
            ),
        },
        "metrics": _probe_metrics(
            class_prediction,
            residual_prediction,
            test_y,
            test_residual,
            test_contrast,
            offsets,
            low_cut,
            high_cut,
        ),
    }


@torch.inference_mode()
def _collect(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    cfg = load_config(args.config)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg["model"].setdefault("structured_query", {})["intermediate_supervision"] = True
    if args.data_root:
        cfg.setdefault("dataset", {})["root"] = args.data_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["pin_memory"] = bool(
        args.num_workers > 0 and str(args.device).startswith("cuda")
    )
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False) and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()
    matcher = build_matcher(cfg)
    loader = build_dataloader(cfg, split=args.split, training=False)
    input_w = int(cfg["model"].get("input_w", 800))
    input_h = int(cfg["model"].get("input_h", 288))
    structured_cfg = cfg["model"]["structured_query"]
    num_instances = int(structured_cfg["num_instances"])
    num_groups = int(structured_cfg.get("num_groups", 1))
    group_size = num_instances // max(num_groups, 1)
    group_index = int(args.group_index)
    if not 0 <= group_index < num_groups:
        raise ValueError(f"group-index must be in [0,{num_groups - 1}]")
    group_start = group_index * group_size
    group_end = group_start + group_size
    offsets = torch.tensor(args.offsets_px, device=device, dtype=torch.float32)
    requested_layers = tuple(str(name).upper() for name in args.anchor_layers)
    photometric_offsets = torch.tensor(
        (-12, -8, -4, 0, 4, 8, 12), device=device, dtype=torch.float32
    )

    captured: dict[str, Any] = {}

    def capture_backbone(_module, _inputs, output):
        captured["backbone"] = output

    hook = model.encoder.backbone.register_forward_hook(capture_backbone)
    stores: dict[str, dict[str, list[torch.Tensor]]] = {
        layer: defaultdict(list) for layer in requested_layers
    }
    valid_rows = {layer: 0 for layer in requested_layers}
    corridor_rows = {layer: 0 for layer in requested_layers}
    matched_lanes = {layer: 0 for layer in requested_layers}
    image_counter = 0
    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(args.amp_dtype)
    try:
        for batch_index, (images, targets, _metas) in enumerate(
            tqdm(loader, desc="frozen residual probe", ncols=92)
        ):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = images.to(
                device,
                non_blocking=True,
                memory_format=(
                    torch.channels_last if channels_last else torch.contiguous_format
                ),
            )
            targets = nested_to_device(targets, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                outputs = model(images, return_features=True)
            layers = _prediction_layers(outputs)
            missing = sorted(set(requested_layers) - set(layers))
            if missing:
                raise RuntimeError(
                    f"Requested decoder layers {missing} are unavailable; found {sorted(layers)}"
                )
            final_matches = matcher(layers[max(layers, key=lambda name: int(name[1:]))], targets)
            c2_batch = captured["backbone"]["c2"]
            p2_batch = outputs["features"]
            mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
            rgb = (images.float() * std.float() + mean.float()).clamp(0.0, 1.0)
            gray = (
                0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
            ).unsqueeze(1)

            for batch_item, match in enumerate(final_matches):
                pred_indices = match["pred_indices"].to(device)
                gt_indices = match["gt_indices"].to(device)
                in_group = (pred_indices >= group_start) & (pred_indices < group_end)
                pred_indices = pred_indices[in_group]
                gt_indices = gt_indices[in_group]
                if pred_indices.numel() == 0:
                    image_counter += 1
                    continue
                target_x = targets[batch_item]["x_rows"][gt_indices].float()
                valid = targets[batch_item]["valid_mask"][gt_indices].bool()
                group_ids = torch.full(
                    target_x.shape,
                    image_counter,
                    dtype=torch.long,
                    device=device,
                )
                photo_x = target_x.unsqueeze(-1) + photometric_offsets.view(1, 1, -1)
                contrast = _sample_points(
                    gray[batch_item],
                    photo_x,
                    input_w=input_w,
                    input_h=input_h,
                ).squeeze(-1).std(dim=-1)

                for layer_name in requested_layers:
                    anchor_x = layers[layer_name]["pred_x_rows"][
                        batch_item, pred_indices
                    ].float()
                    residual = target_x - anchor_x
                    inside = valid & (residual >= float(offsets.min())) & (
                        residual <= float(offsets.max())
                    )
                    valid_rows[layer_name] += int(valid.sum())
                    corridor_rows[layer_name] += int(inside.sum())
                    matched_lanes[layer_name] += int(pred_indices.numel())
                    selected = inside.flatten().nonzero(as_tuple=False).flatten()
                    if selected.numel() == 0:
                        continue
                    if (
                        args.max_rows_per_image > 0
                        and selected.numel() > args.max_rows_per_image
                    ):
                        selected = selected[
                            torch.randperm(selected.numel(), device=device)[
                                : args.max_rows_per_image
                            ]
                        ]
                    sample_x = anchor_x.unsqueeze(-1) + offsets.view(1, 1, -1)
                    label = (
                        residual.unsqueeze(-1) - offsets.view(1, 1, -1)
                    ).abs().argmin(dim=-1)
                    store = stores[layer_name]
                    store["labels"].append(label.flatten()[selected].cpu())
                    store["residual"].append(residual.flatten()[selected].cpu())
                    store["groups"].append(group_ids.flatten()[selected].cpu())
                    store["contrast"].append(contrast.flatten()[selected].cpu())
                    for source_name, feature in (
                        ("c2", c2_batch[batch_item]),
                        ("p2", p2_batch[batch_item]),
                    ):
                        profile = _sample_points(
                            feature,
                            sample_x,
                            input_w=input_w,
                            input_h=input_h,
                        )
                        profile = profile.reshape(
                            -1, profile.shape[-2], profile.shape[-1]
                        )
                        store[source_name].append(profile[selected].cpu())
                image_counter += 1
    finally:
        hook.remove()

    collected: dict[str, dict[str, torch.Tensor]] = {}
    for layer_offset, layer_name in enumerate(requested_layers):
        if not stores[layer_name]:
            raise RuntimeError(
                f"No residual rows collected for {layer_name}; increase the offset corridor"
            )
        layer = {
            name: torch.cat(parts, dim=0) for name, parts in stores[layer_name].items()
        }
        if args.max_probe_rows > 0 and layer["labels"].numel() > args.max_probe_rows:
            generator = torch.Generator().manual_seed(args.seed + layer_offset)
            keep = torch.randperm(
                layer["labels"].numel(), generator=generator
            )[: args.max_probe_rows]
            layer = {name: value[keep] for name, value in layer.items()}
        collected[layer_name] = layer

    metadata = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(iteration),
        "split": args.split,
        "images": int(image_counter),
        "group_index": int(group_index),
        "group_size": int(group_size),
        "anchor_assignment": "final-layer Hungarian assignment restricted to one training group",
        "anchor_layers": list(requested_layers),
        "offsets_px": [float(value) for value in offsets.cpu()],
        "feature_sources": {
            "c2": "raw stride-4 backbone C2",
            "p2": "projected fused P2 consumed by the structured decoder",
        },
        "fairness": (
            "identical image-disjoint rows, offsets, optimizer draws, probe architecture, "
            "and trainable parameter count; C2 is zero-padded, P2 is not compressed"
        ),
        "layers": {
            name: {
                "matched_lanes": int(matched_lanes[name]),
                "valid_rows": int(valid_rows[name]),
                "rows_inside_offset_corridor": int(corridor_rows[name]),
                "corridor_coverage": corridor_rows[name] / max(valid_rows[name], 1),
                "retained_probe_rows": int(collected[name]["labels"].numel()),
            }
            for name in requested_layers
        },
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata, collected


def _source_delta(c2: dict[str, Any], p2: dict[str, Any]) -> dict[str, float]:
    c2_all = c2["metrics"]["all"]
    p2_all = p2["metrics"]["all"]
    keys = (
        "class_balanced_accuracy",
        "class_direction_accuracy",
        "regression_corrected_mae_px",
        "regression_mae_gain_px",
        "regression_within_4px",
        "regression_within_8px",
    )
    return {f"c2_minus_p2/{key}": float(c2_all[key] - p2_all[key]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train equal-capacity probes on frozen raw C2 and decoder P2 profiles "
            "to test whether intermediate lane residuals are visually recoverable."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--max-probe-rows", type=int, default=12000)
    parser.add_argument("--max-rows-per-image", type=int, default=256)
    parser.add_argument("--anchor-layers", nargs="+", default=["L2", "L3"])
    parser.add_argument(
        "--offsets-px",
        type=float,
        nargs="+",
        default=[-32, -16, -8, -4, 0, 4, 8, 16, 32],
    )
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--common-channels", type=int, default=256)
    parser.add_argument("--probe-hidden-dim", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=160)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    metadata, collected = _collect(args)
    device = torch.device(args.device)
    offsets = torch.tensor(args.offsets_px, dtype=torch.float32)
    payload: dict[str, Any] = {**metadata, "probe_results": {}}
    for layer_index, layer_name in enumerate(metadata["anchor_layers"]):
        values = collected[layer_name]
        layer_result: dict[str, Any] = {}
        for source_name in ("c2", "p2"):
            layer_result[source_name] = _train_probe(
                values[source_name],
                values["labels"],
                values["residual"],
                values["groups"],
                values["contrast"],
                offsets,
                common_channels=int(args.common_channels),
                hidden_dim=int(args.probe_hidden_dim),
                steps=int(args.probe_steps),
                batch_size=int(args.probe_batch_size),
                seed=int(args.seed + 100 * layer_index),
                device=device,
            )
        if (
            layer_result["c2"]["probe_parameters"]
            != layer_result["p2"]["probe_parameters"]
        ):
            raise RuntimeError("C2 and P2 probes do not have equal parameter counts")
        layer_result["comparison"] = _source_delta(
            layer_result["c2"], layer_result["p2"]
        )
        payload["probe_results"][layer_name] = layer_result

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"checkpoint_iteration={payload['checkpoint_iteration']} "
        f"images={payload['images']} output={output_path}"
    )
    for layer_name, layer_result in payload["probe_results"].items():
        coverage = payload["layers"][layer_name]["corridor_coverage"]
        for source_name in ("c2", "p2"):
            metrics = layer_result[source_name]["metrics"]["all"]
            print(
                f"{layer_name} {source_name.upper()} coverage={coverage:.4f} "
                f"bal_acc={metrics['class_balanced_accuracy']:.4f} "
                f"direction={metrics['class_direction_accuracy']:.4f} "
                f"anchor_mae={metrics['anchor_mae_px']:.3f} "
                f"class_mae={metrics['class_corrected_mae_px']:.3f} "
                f"reg_mae={metrics['regression_corrected_mae_px']:.3f}"
            )
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
