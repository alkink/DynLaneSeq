from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.v22_lane_field import (
    V22LaneFieldStageA,
    lane_field_loss,
    model_contract,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v22_official_protocol import official_culane_list_contract


FIXED_STEPS = 10_000
FIXED_PHYSICAL_BATCH = 8
FIXED_GRAD_ACCUMULATION = 2
FIXED_SEED = 3407


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the V22 Stage-A global lane field. No slot decoder or "
            "deployable detector is constructed."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--v7-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--steps", type=int, default=FIXED_STEPS)
    parser.add_argument("--batch-size", type=int, default=FIXED_PHYSICAL_BATCH)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=FIXED_GRAD_ACCUMULATION,
    )
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def _configured(
    args: argparse.Namespace, *, official_train_list: str
) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})["train"] = str(
        Path(official_train_list).expanduser().resolve()
    )
    cfg.setdefault("training", {})["seed"] = int(args.seed)
    cfg["training"]["batch_size"] = int(args.batch_size)
    cfg["training"]["gradient_accumulation_steps"] = int(
        args.gradient_accumulation_steps
    )
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    # V7 is restored from its exact checkpoint; construction must never start
    # a network download or silently substitute random pretrained weights.
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _make_field(cfg: dict[str, Any]) -> V22LaneFieldStageA:
    model_cfg = cfg["model"]
    stage_cfg = cfg["v22_stage_a"]
    return V22LaneFieldStageA(
        input_h=int(model_cfg["input_h"]),
        input_w=int(model_cfg["input_w"]),
        num_rows=int(model_cfg["num_rows"]),
        x_bins=int(model_cfg["x_bins"]),
        fpn_channels=int(model_cfg["fpn_channels"]),
        hidden_dim=int(stage_cfg["hidden_dim"]),
        distance_limit_px=float(stage_cfg["distance_limit_px"]),
        freeze_batch_norm_stats=bool(stage_cfg["freeze_batch_norm_stats"]),
    )


def _optimizer(model: V22LaneFieldStageA, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = cfg["v22_stage_a"]["optimizer"]
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(value) for value in backbone}
    field = [value for value in model.parameters() if id(value) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": float(opt_cfg["backbone_lr"]),
                "initial_lr": float(opt_cfg["backbone_lr"]),
                "name": "new_lane_encoder_backbone",
            },
            {
                "params": field,
                "lr": float(opt_cfg["field_lr"]),
                "initial_lr": float(opt_cfg["field_lr"]),
                "name": "new_lane_field",
            },
        ],
        betas=(0.9, 0.999),
        weight_decay=float(opt_cfg["weight_decay"]),
    )


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    if step <= warmup_steps:
        ratio = float(step) / float(max(warmup_steps, 1))
    else:
        progress = float(step - warmup_steps) / float(
            max(total_steps - warmup_steps, 1)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        ratio = float(minimum_ratio) + (1.0 - float(minimum_ratio)) * cosine
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * ratio
    return ratio


def _next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _gate_zero(
    model: V22LaneFieldStageA,
    *,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    model_cfg = cfg["model"]
    stage_cfg = cfg["v22_stage_a"]
    image = torch.zeros(
        (1, 3, int(model_cfg["input_h"]), int(model_cfg["input_w"])),
        device=device,
    )
    if bool(cfg["training"].get("channels_last", False)):
        image = image.contiguous(memory_format=torch.channels_last)
    rows = int(model_cfg["num_rows"])
    x = torch.linspace(
        0.35 * float(model_cfg["input_w"]),
        0.45 * float(model_cfg["input_w"]),
        rows,
        device=device,
    ).view(1, rows)
    target = {
        "x_rows": x,
        "valid_mask": torch.ones_like(x, dtype=torch.bool),
    }
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        output = model(image)
        total, diagnostics = lane_field_loss(
            output,
            [target],
            input_w=int(model_cfg["input_w"]),
            centerline_sigma_px=float(stage_cfg["centerline_sigma_px"]),
            distance_limit_px=float(stage_cfg["distance_limit_px"]),
            centerline_positive_weight=float(
                stage_cfg["centerline_positive_weight"]
            ),
        )
    total.backward()
    groups = {
        "backbone": [
            parameter.grad
            for parameter in model.backbone.parameters()
            if parameter.grad is not None
        ],
        "fpn": [
            parameter.grad
            for parameter in model.fpn.parameters()
            if parameter.grad is not None
        ],
        "field_heads": [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(("field_trunk.", "centerline_head.", "distance_head.", "support_head."))
            and parameter.grad is not None
        ],
    }
    gradient_norms = {
        name: float(
            torch.sqrt(
                sum(gradient.detach().float().pow(2).sum() for gradient in values)
            )
        )
        if values
        else 0.0
        for name, values in groups.items()
    }
    expected_shape = (
        1,
        1,
        int(model_cfg["num_rows"]),
        int(model_cfg["x_bins"]),
    )
    shapes_exact = all(
        tuple(output[name].shape) == expected_shape
        for name in ("centerline_logits", "distance_raw", "support_logits")
    )
    finite = bool(torch.isfinite(total).all()) and all(
        math.isfinite(float(value)) for value in diagnostics.values()
    )
    passed = bool(
        shapes_exact
        and finite
        and all(value > 0.0 and math.isfinite(value) for value in gradient_norms.values())
    )
    model.zero_grad(set_to_none=True)
    del image, output, total
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "passed": passed,
        "output_shapes_exact": shapes_exact,
        "losses_finite": finite,
        "gradient_norms": gradient_norms,
        "expected_shape": list(expected_shape),
    }


def main() -> None:
    args = parse_args()
    if int(args.steps) != FIXED_STEPS:
        raise ValueError(f"V22 Stage-A fixed gate requires {FIXED_STEPS} steps")
    if int(args.batch_size) != FIXED_PHYSICAL_BATCH:
        raise ValueError(
            f"V22 Stage-A fixed gate requires physical batch {FIXED_PHYSICAL_BATCH}"
        )
    if int(args.gradient_accumulation_steps) != FIXED_GRAD_ACCUMULATION:
        raise ValueError(
            "V22 Stage-A fixed gate requires gradient accumulation "
            f"{FIXED_GRAD_ACCUMULATION}"
        )
    if int(args.seed) != FIXED_SEED:
        raise ValueError(f"V22 Stage-A fixed gate requires seed {FIXED_SEED}")
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    train_population = official_culane_list_contract(
        args.dataset_root, split="train"
    )
    cfg = _configured(
        args, official_train_list=str(train_population["list_path"])
    )

    # Construct and restore V7 only on CPU; it is an initialization source and
    # never participates in Stage-A optimization or inference.
    v7 = build_model(cfg)
    v7_iteration = int(load_checkpoint(args.v7_checkpoint, v7, strict=False))
    field_model = _make_field(cfg)
    warm_start = field_model.copy_v7_encoder_(v7)
    del v7
    field_model.to(device)
    if bool(cfg["training"].get("channels_last", False)):
        field_model.to(memory_format=torch.channels_last)
    gate_zero = _gate_zero(field_model, cfg=cfg, device=device)
    if not gate_zero["passed"] or warm_start["maximum_copy_difference"] != 0.0:
        raise RuntimeError("V22 Stage-A Gate 0 failed")

    loader = build_dataloader(cfg, split="train", training=True)
    official_train_rows = int(train_population["expected_nonempty_rows"])
    if len(loader.dataset) != official_train_rows:
        raise ValueError(
            "V22 Stage-A loader did not retain the complete official CULane "
            f"training population: expected {official_train_rows}, "
            f"found {len(loader.dataset)}"
        )
    iterator = iter(loader)
    optimizer = _optimizer(field_model, cfg)
    stage_cfg = cfg["v22_stage_a"]
    opt_cfg = stage_cfg["optimizer"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    metrics_handle = metrics_path.open("w", encoding="utf-8")
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    channels_last = bool(cfg["training"].get("channels_last", False))
    start_time = time.perf_counter()
    images_seen = 0
    final_diagnostics: dict[str, float] = {}
    field_model.train()
    for step in range(1, int(args.steps) + 1):
        lr_ratio = _set_learning_rate(
            optimizer,
            step=step,
            total_steps=int(args.steps),
            warmup_steps=int(opt_cfg["warmup_steps"]),
            minimum_ratio=float(opt_cfg["minimum_lr_ratio"]),
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for _micro in range(int(args.gradient_accumulation_steps)):
            (images, targets, _metas), iterator = _next_batch(loader, iterator)
            images_seen += int(images.shape[0])
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = field_model(images)
                loss, diagnostics = lane_field_loss(
                    output,
                    targets,
                    input_w=int(cfg["model"]["input_w"]),
                    centerline_sigma_px=float(stage_cfg["centerline_sigma_px"]),
                    distance_limit_px=float(stage_cfg["distance_limit_px"]),
                    centerline_positive_weight=float(
                        stage_cfg["centerline_positive_weight"]
                    ),
                )
                scaled_loss = loss / float(args.gradient_accumulation_steps)
            if not bool(torch.isfinite(scaled_loss)):
                raise FloatingPointError(f"non-finite V22 Stage-A loss at step {step}")
            scaled_loss.backward()
            for name, value in diagnostics.items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value) / float(
                    args.gradient_accumulation_steps
                )
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(field_model.parameters(), max_norm=1.0)
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite V22 Stage-A gradient at step {step}"
            )
        optimizer.step()
        final_diagnostics = accumulated
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            elapsed = max(time.perf_counter() - start_time, 1.0e-6)
            row = {
                "step": step,
                "images_seen": images_seen,
                "images_per_second": float(images_seen) / elapsed,
                "gradient_norm": gradient_norm,
                "learning_rate_ratio": lr_ratio,
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "field_lr": float(optimizer.param_groups[1]["lr"]),
                **accumulated,
            }
            text = json.dumps(row, sort_keys=True)
            print(text, flush=True)
            metrics_handle.write(text + "\n")
            metrics_handle.flush()
    metrics_handle.close()

    checkpoint_path = output_dir / "lane_field_endpoint.pt"
    checkpoint = {
        "experiment": "V22 Stage-A trainable global lane field",
        "iteration": int(args.steps),
        "model": field_model.state_dict(),
        "model_contract": model_contract(field_model),
        "v7_checkpoint": str(Path(args.v7_checkpoint).expanduser().resolve()),
        "v7_checkpoint_sha256": sha256_file(args.v7_checkpoint),
        "v7_iteration": v7_iteration,
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "train_list": str(train_population["list_path"]),
        "train_list_sha256": str(train_population["list_sha256"]),
        "official_train_population_contract": train_population,
        "warm_start": warm_start,
        "gate_zero": gate_zero,
        "images_seen": images_seen,
        "complete_official_train_epochs_seen": (
            float(images_seen) / float(official_train_rows)
        ),
        "elapsed_seconds": time.perf_counter() - start_time,
        "final_training_diagnostics": final_diagnostics,
        "checkpoint_selection_performed": False,
        "full_validation_executed": False,
        "test_set_used": False,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        key: value for key, value in checkpoint.items() if key not in {"model"}
    }
    report["checkpoint"] = str(checkpoint_path)
    report["checkpoint_sha256"] = sha256_file(checkpoint_path)
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
