from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint, save_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v23 import DynLaneSeqV23
from dynlaneseq_eg.modeling.v22_lane_field import V22LaneFieldStageA
from dynlaneseq_eg.modeling.v23_ordered_slot_cost_volume import (
    V23LossWeights,
    v23_model_contract,
    v23_ordered_cost_volume_loss,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


FIXED_GATE_STEPS = 8_000
FIXED_PHYSICAL_BATCH = 1
FIXED_GRAD_ACCUMULATION = 8
FIXED_SEED = 3407


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V23 ordered slot-conditioned lane cost volume."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--v7-checkpoint", required=True)
    parser.add_argument("--v22-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument(
        "--mode",
        choices=("gate", "smoke"),
        default="gate",
        help="Only gate mode is a scientific result; smoke is mechanical only.",
    )
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-interval", type=int, default=500)
    return parser.parse_args()


def _configured(
    args: argparse.Namespace,
    *,
    official_train_list: str,
    official_val_list: str,
) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    dataset_root = str(Path(args.dataset_root).expanduser().resolve())
    cfg.setdefault("dataset", {})["root"] = dataset_root
    cfg["dataset"].setdefault("lists", {})["train"] = official_train_list
    cfg["dataset"]["lists"]["val"] = official_val_list
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _make_v22(cfg: dict[str, Any]) -> V22LaneFieldStageA:
    model_cfg = cfg["model"]
    # The V22 endpoint contract is fixed and independent of the removed V22
    # output heads. Only backbone/FPN tensors are copied into V23.
    return V22LaneFieldStageA(
        input_h=int(model_cfg["input_h"]),
        input_w=int(model_cfg["input_w"]),
        num_rows=int(model_cfg["num_rows"]),
        x_bins=int(model_cfg["x_bins"]),
        fpn_channels=int(model_cfg["fpn_channels"]),
        hidden_dim=128,
        distance_limit_px=96.0,
        freeze_batch_norm_stats=True,
    )


def _loss_weights(cfg: dict[str, Any]) -> V23LossWeights:
    raw = cfg["v23"]["loss"]
    return V23LossWeights(
        row_distribution=float(raw["row_distribution"]),
        point=float(raw["point"]),
        strip_iou=float(raw["strip_iou"]),
        quality50=float(raw["quality50"]),
        quality75=float(raw["quality75"]),
        smoothness=float(raw["smoothness"]),
        order=float(raw["order"]),
        proposal_path=float(raw["proposal_path"]),
        nondegradation=float(raw["nondegradation"]),
        gate_regularization=float(raw["gate_regularization"]),
        tail_scale=float(raw["tail_scale"]),
        proposal_temperature=float(raw["proposal_temperature"]),
        line_width=float(raw["line_width"]),
        minimum_valid_rows=int(raw["minimum_valid_rows"]),
    )


def _optimizer(model: DynLaneSeqV23, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    raw = cfg["v23"]["optimizer"]
    backbone = list(model.student.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    rest = [
        parameter
        for parameter in model.student.parameters()
        if id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        (
            {
                "params": backbone,
                "lr": float(raw["backbone_lr"]),
                "initial_lr": float(raw["backbone_lr"]),
                "name": "student_backbone",
            },
            {
                "params": rest,
                "lr": float(raw["student_lr"]),
                "initial_lr": float(raw["student_lr"]),
                "name": "cost_volume_path_decoder",
            },
        ),
        betas=(0.9, 0.999),
        weight_decay=float(raw["weight_decay"]),
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


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _gradient_norm(parameters) -> float:
    values = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.sqrt(sum(values))) if values else 0.0


def _gate_zero(
    model: DynLaneSeqV23,
    batch,
    *,
    device: torch.device,
    cfg: dict[str, Any],
    weights: V23LossWeights,
) -> dict[str, Any]:
    images, targets, _metas = batch
    images = images[:1].to(device)
    targets = targets[:1]
    if bool(cfg["training"].get("channels_last", False)):
        images = images.contiguous(memory_format=torch.channels_last)
    model.train()
    model.zero_grad(set_to_none=True)
    teacher_before = _state_digest(model.teacher)
    with torch.no_grad():
        teacher_output = model.teacher(images.float(), inference_only=True)
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        output = model(images)
        total, diagnostics = v23_ordered_cost_volume_loss(
            output,
            targets,
            input_h=int(cfg["model"]["input_h"]),
            input_w=int(cfg["model"]["input_w"]),
            weights=weights,
        )
    total.backward()
    teacher_after = _state_digest(model.teacher)
    parity = {
        "pred_x_rows": float(
            (
                output["pred_x_rows"].float()
                - teacher_output["selection_slot_pred_x_rows"].float()
            ).abs().max().detach()
        ),
        "range_norm": float(
            (
                output["range_norm"].float()
                - teacher_output["selection_slot_range_norm"].float()
            ).abs().max().detach()
        ),
    }
    teacher_gradient_tensors = sum(
        parameter.grad is not None for parameter in model.teacher.parameters()
    )
    gradient_norms = {
        "backbone": _gradient_norm(model.student.backbone.parameters()),
        "fpn": _gradient_norm(model.student.fpn.parameters()),
        "fine_stem": _gradient_norm(model.student.fine_stem.parameters()),
        "cost_volume": _gradient_norm(
            list(model.student.geometry_projection.parameters())
            + list(model.student.key_projection.parameters())
        ),
        "vertical_path": _gradient_norm(model.student.vertical_encoder.parameters()),
        "four_slot_inter": _gradient_norm(model.student.inter_attention.parameters()),
    }
    finite = bool(torch.isfinite(total)) and all(
        math.isfinite(float(value)) for value in diagnostics.values()
    )
    passed = bool(
        finite
        and parity["pred_x_rows"] == 0.0
        and parity["range_norm"] == 0.0
        and teacher_before == teacher_after
        and teacher_gradient_tensors == 0
        and all(value > 0.0 and math.isfinite(value) for value in gradient_norms.values())
    )
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "passed": passed,
        "zero_step_public_geometry_parity": parity,
        "teacher_state_sha256_before": teacher_before,
        "teacher_state_sha256_after": teacher_after,
        "teacher_gradient_tensors": teacher_gradient_tensors,
        "student_gradient_norms": gradient_norms,
        "losses_finite": finite,
    }


def main() -> None:
    args = parse_args()
    seed_everything(FIXED_SEED)
    train_population = official_v23_culane_list_contract(
        args.dataset_root, split="train"
    )
    val_population = official_v23_culane_list_contract(
        args.dataset_root, split="val"
    )
    cfg = _configured(
        args,
        official_train_list=str(train_population["list_path"]),
        official_val_list=str(val_population["list_path"]),
    )
    if str(cfg["model"].get("name")) != "DynLaneSeqV23":
        raise ValueError("V23 trainer requires model.name=DynLaneSeqV23")
    if int(cfg["training"]["batch_size"]) != FIXED_PHYSICAL_BATCH:
        raise ValueError(f"V23 fixed physical batch must be {FIXED_PHYSICAL_BATCH}")
    if int(cfg["training"]["gradient_accumulation_steps"]) != FIXED_GRAD_ACCUMULATION:
        raise ValueError(
            f"V23 fixed gradient accumulation must be {FIXED_GRAD_ACCUMULATION}"
        )
    if int(cfg["training"]["seed"]) != FIXED_SEED:
        raise ValueError(f"V23 fixed seed must be {FIXED_SEED}")
    steps = (
        FIXED_GATE_STEPS
        if args.mode == "gate"
        else int(args.smoke_steps)
    )
    if args.mode == "gate" and int(cfg["training"]["max_iters"]) != FIXED_GATE_STEPS:
        raise ValueError(f"V23 gate is fixed at {FIXED_GATE_STEPS} steps")
    if args.mode == "smoke" and not 1 <= steps <= 10:
        raise ValueError("V23 smoke mode is limited to 1..10 mechanical steps")

    device = torch.device(args.device)
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV23):
        raise TypeError("factory did not construct DynLaneSeqV23")
    v7_iteration = int(load_checkpoint(args.v7_checkpoint, model.teacher, strict=True))
    v7_teacher_digest = _state_digest(model.teacher)
    v22 = _make_v22(cfg)
    v22_iteration = int(load_checkpoint(args.v22_checkpoint, v22, strict=True))
    warm_start = model.student.copy_v22_encoder_(v22)
    del v22
    if warm_start["maximum_copy_difference"] != 0.0:
        raise RuntimeError("V23 did not copy the V22 encoder exactly")
    model.to(device)
    channels_last = bool(cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)

    weights = _loss_weights(cfg)
    optimizer = _optimizer(model, cfg)
    start_step = 0
    resumed_from = ""
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        start_step = int(
            load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                strict=True,
                restore_rng_state=True,
            )
        )
        if not 0 < start_step < steps:
            raise ValueError(
                f"resume iteration must be in (0,{steps}), got {start_step}"
            )
        resumed_from = str(resume_path)
        if _state_digest(model.teacher) != v7_teacher_digest:
            raise RuntimeError("resume checkpoint changed the exact V7 teacher state")
    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=start_step
    )
    if len(loader.dataset) != int(train_population["expected_nonempty_rows"]):
        raise ValueError(
            "V23 loader altered the official train.txt population: "
            f"expected {train_population['expected_nonempty_rows']}, "
            f"found {len(loader.dataset)}"
        )
    iterator = iter(loader)
    if start_step == 0:
        gate_batch, iterator = _next_batch(loader, iterator)
        gate_zero = _gate_zero(
            model,
            gate_batch,
            device=device,
            cfg=cfg,
            weights=weights,
        )
        if not gate_zero["passed"]:
            raise RuntimeError(
                "V23 Gate 0 failed: " + json.dumps(gate_zero, sort_keys=True)
            )
        # Gate 0 is diagnostic only. Rewind the resume-safe stream so no
        # official training image is consumed without an optimizer update.
        iterator = iter(loader)
    else:
        gate_zero = {
            "passed": True,
            "reused_from_resume": resumed_from,
            "zero_step_contract_was_executed_before_optimization": True,
        }

    opt_cfg = cfg["v23"]["optimizer"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    metrics_handle = metrics_path.open(
        "a" if start_step > 0 else "w", encoding="utf-8"
    )
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    accumulation_steps = int(cfg["training"]["gradient_accumulation_steps"])
    start_time = time.perf_counter()
    images_seen = start_step * int(cfg["training"]["batch_size"]) * int(
        cfg["training"]["gradient_accumulation_steps"]
    )
    final_diagnostics: dict[str, float] = {}
    model.train()
    resume_checkpoint = output_dir / "resume_latest.pt"
    for step in range(start_step + 1, steps + 1):
        lr_ratio = _set_learning_rate(
            optimizer,
            step=step,
            total_steps=steps,
            warmup_steps=min(int(opt_cfg["warmup_steps"]), max(steps // 4, 1)),
            minimum_ratio=float(opt_cfg["minimum_lr_ratio"]),
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for _micro in range(accumulation_steps):
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
                output = model(images)
                loss, diagnostics = v23_ordered_cost_volume_loss(
                    output,
                    targets,
                    input_h=int(cfg["model"]["input_h"]),
                    input_w=int(cfg["model"]["input_w"]),
                    weights=weights,
                )
                scaled_loss = loss / float(accumulation_steps)
            if not bool(torch.isfinite(scaled_loss)):
                raise FloatingPointError(f"non-finite V23 loss at step {step}")
            scaled_loss.backward()
            for name, value in diagnostics.items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value) / float(
                    accumulation_steps
                )
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.student.parameters(),
                max_norm=float(cfg["training"]["clip_grad_norm"]),
            )
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite V23 gradient at step {step}")
        optimizer.step()
        final_diagnostics = accumulated
        if step == 1 or step % int(args.log_interval) == 0 or step == steps:
            elapsed = max(time.perf_counter() - start_time, 1.0e-6)
            row = {
                "step": step,
                "mode": args.mode,
                "images_seen": images_seen,
                "images_per_second": float(images_seen) / elapsed,
                "gradient_norm": gradient_norm,
                "learning_rate_ratio": lr_ratio,
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "student_lr": float(optimizer.param_groups[1]["lr"]),
                **accumulated,
            }
            text = json.dumps(row, sort_keys=True)
            print(text, flush=True)
            metrics_handle.write(text + "\n")
            metrics_handle.flush()
        if (
            args.mode == "gate"
            and int(args.resume_interval) > 0
            and step < steps
            and step % int(args.resume_interval) == 0
        ):
            save_checkpoint(
                resume_checkpoint,
                model,
                optimizer=optimizer,
                iteration=step,
                cfg=cfg,
                include_rng_state=True,
            )
    metrics_handle.close()

    checkpoint_path = output_dir / (
        "v23_gate_endpoint.pt" if args.mode == "gate" else "v23_smoke_endpoint.pt"
    )
    save_checkpoint(
        checkpoint_path,
        model,
        iteration=steps,
        cfg=cfg,
    )
    report = {
        "experiment": "V23 ordered slot-conditioned lane cost volume",
        "scientific_gate": args.mode == "gate",
        "iteration": steps,
        "images_seen": images_seen,
        "start_iteration": start_step,
        "resumed_from": resumed_from,
        "complete_official_train_epochs_seen": float(images_seen)
        / float(train_population["expected_nonempty_rows"]),
        "elapsed_seconds": time.perf_counter() - start_time,
        "maximum_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "v7_checkpoint": str(Path(args.v7_checkpoint).expanduser().resolve()),
        "v7_checkpoint_sha256": sha256_file(args.v7_checkpoint),
        "v7_iteration": v7_iteration,
        "v22_initialization_checkpoint": str(
            Path(args.v22_checkpoint).expanduser().resolve()
        ),
        "v22_initialization_checkpoint_sha256": sha256_file(args.v22_checkpoint),
        "v22_iteration": v22_iteration,
        "model_contract": v23_model_contract(model.student),
        "warm_start": warm_start,
        "gate_zero": gate_zero,
        "teacher_state_sha256_at_endpoint": _state_digest(model.teacher),
        "teacher_state_still_exact": _state_digest(model.teacher)
        == v7_teacher_digest,
        "official_train_population_contract": train_population,
        "official_val_population_contract": val_population,
        "final_training_diagnostics": final_diagnostics,
        "checkpoint_selection_performed": False,
        "threshold_selection_performed": False,
        "full_validation_executed": False,
        "test_set_used": False,
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
