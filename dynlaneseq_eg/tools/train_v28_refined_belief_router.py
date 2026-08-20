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
from dynlaneseq_eg.modeling.dynlaneseq_v28 import DynLaneSeqV28
from dynlaneseq_eg.modeling.v28_refined_belief_router import (
    V28BeliefLossWeights,
    gather_slot_candidates,
    v28_model_contract,
    v28_refined_belief_loss,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


FIXED_SEED = 3407


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train V28 immutable-refined-bank belief arm B (route only) or "
            "arm C (the same route objective plus dense slot field)."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--v7-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", choices=("B", "C"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--mode", choices=("gate", "smoke"), default="gate")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-interval", type=int, default=500)
    return parser.parse_args()


def _configured(
    args: argparse.Namespace,
    *,
    train_list: str,
    val_list: str,
) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})["train"] = train_list
    cfg["dataset"]["lists"]["val"] = val_list
    if args.num_workers is not None:
        cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    workers = int(cfg.setdefault("dataloader", {}).get("num_workers", 0))
    cfg["dataloader"]["persistent_workers"] = workers > 0
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg.setdefault("v28", {})["arm"] = str(args.arm).upper()
    return cfg


def _configure_runtime(cfg: dict[str, Any], device: torch.device) -> None:
    training = cfg.get("training", {})
    if training.get("cpu_threads") is not None:
        torch.set_num_threads(max(int(training["cpu_threads"]), 1))
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", False))
    if bool(training.get("tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _move_images(
    images: torch.Tensor,
    *,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        return images.contiguous(memory_format=torch.channels_last)
    return images


def _weights(cfg: dict[str, Any]) -> V28BeliefLossWeights:
    raw = cfg["v28"]["loss"]
    return V28BeliefLossWeights(
        route=float(raw["route"]),
        field=float(raw["field"]),
        target_temperature=float(raw["target_temperature"]),
        target_delta=float(raw["target_delta"]),
        target_floor=float(raw["target_floor"]),
        line_width=float(raw["line_width"]),
        minimum_valid_rows=int(raw["minimum_valid_rows"]),
    )


def _optimizer(
    model: DynLaneSeqV28,
    cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    raw = cfg["v28"]["optimizer"]
    backbone = list(model.router.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    rest = [
        parameter
        for parameter in model.router.parameters()
        if id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        (
            {
                "params": backbone,
                "lr": float(raw["backbone_lr"]),
                "initial_lr": float(raw["backbone_lr"]),
                "name": "belief_backbone",
            },
            {
                "params": rest,
                "lr": float(raw["router_lr"]),
                "initial_lr": float(raw["router_lr"]),
                "name": "slot_belief_router",
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


def _state_versions(module: torch.nn.Module) -> dict[str, int]:
    return {
        name: int(value._version)
        for name, value in module.state_dict(keep_vars=True).items()
    }


def _gradient_norm(parameters) -> float:
    values = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.sqrt(sum(values))) if values else 0.0


def _loss(
    model_output: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    cfg: dict[str, Any],
    arm: str,
    weights: V28BeliefLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return v28_refined_belief_loss(
        model_output,
        targets,
        candidate_x=model_output["candidate_x"],
        candidate_range=model_output["candidate_range"],
        candidate_valid=model_output["candidate_valid"],
        source_x=model_output["source_x"],
        source_range=model_output["source_range"],
        source_active=model_output["source_active"],
        source_route=model_output["source_route"],
        input_h=int(cfg["model"]["input_h"]),
        input_w=int(cfg["model"]["input_w"]),
        arm=arm,
        weights=weights,
    )


def _gate_zero(
    model: DynLaneSeqV28,
    batch,
    *,
    device: torch.device,
    cfg: dict[str, Any],
    arm: str,
    weights: V28BeliefLossWeights,
) -> dict[str, Any]:
    images, targets, _metas = batch
    images = _move_images(
        images,
        device=device,
        channels_last=bool(cfg["training"].get("channels_last", False)),
    )
    model.train()
    model.zero_grad(set_to_none=True)
    teacher_versions_before = _state_versions(model.teacher)
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp,
    ):
        output = model(images, deployment_policy="source")
        total, diagnostics = _loss(
            output, targets, cfg=cfg, arm=arm, weights=weights
        )
        route_probe, _ = _loss(
            output, targets, cfg=cfg, arm="B", weights=weights
        )
        combined_probe, _ = _loss(
            output, targets, cfg=cfg, arm="C", weights=weights
        )
        field_probe = (combined_probe - route_probe) / max(
            float(weights.field), 1.0e-12
        )
    trainable = tuple(model.router.parameters())
    route_gradients = torch.autograd.grad(
        route_probe,
        trainable,
        retain_graph=True,
        allow_unused=True,
    )
    field_gradients = torch.autograd.grad(
        field_probe,
        trainable,
        retain_graph=True,
        allow_unused=True,
    )
    route_squared = route_probe.new_zeros(())
    field_squared = route_probe.new_zeros(())
    gradient_dot = route_probe.new_zeros(())
    for route_gradient, field_gradient in zip(
        route_gradients, field_gradients
    ):
        if route_gradient is not None:
            route_squared = route_squared + route_gradient.float().pow(2).sum()
        if field_gradient is not None:
            field_squared = field_squared + field_gradient.float().pow(2).sum()
        if route_gradient is not None and field_gradient is not None:
            gradient_dot = gradient_dot + (
                route_gradient.float() * field_gradient.float()
            ).sum()
    route_norm = route_squared.sqrt()
    field_norm = field_squared.sqrt()
    gradient_cosine = gradient_dot / (
        route_norm * field_norm
    ).clamp_min(1.0e-12)
    total.backward()
    teacher_versions_after = _state_versions(model.teacher)
    source_routes = output["source_route"].clamp_min(0)
    replay_x = gather_slot_candidates(output["candidate_x"], source_routes)
    replay_range = gather_slot_candidates(
        output["candidate_range"], source_routes
    )
    active = output["source_active"]
    counterfactual_parity = {
        "x_rows": float(
            (replay_x[active] - output["source_x"][active]).abs().max()
        )
        if bool(active.any())
        else 0.0,
        "range_norm": float(
            (
                replay_range[active] - output["source_range"][active]
            ).abs().max()
        )
        if bool(active.any())
        else 0.0,
    }
    public_parity = {
        "pred_x_rows": float(
            (
                output["pred_x_rows"]
                - output["teacher_source_x_rows"]
            ).abs().max()
        ),
        "range_norm": float(
            (
                output["range_norm"]
                - output["teacher_source_range_norm"]
            ).abs().max()
        ),
    }
    gradient_norms = {
        "backbone": _gradient_norm(model.router.backbone.parameters()),
        "fpn": _gradient_norm(model.router.fpn.parameters()),
        "field_key": _gradient_norm(model.router.key_projection.parameters()),
        "vertical": _gradient_norm(model.router.vertical_encoder.parameters()),
    }
    teacher_gradient_tensors = sum(
        parameter.grad is not None for parameter in model.teacher.parameters()
    )
    finite = bool(torch.isfinite(total)) and all(
        bool(torch.isfinite(value)) for value in diagnostics.values()
    )
    passed = bool(
        finite
        and public_parity["pred_x_rows"] == 0.0
        and public_parity["range_norm"] == 0.0
        and counterfactual_parity["x_rows"] <= 1.0e-4
        and counterfactual_parity["range_norm"] <= 1.0e-6
        and teacher_versions_before == teacher_versions_after
        and teacher_gradient_tensors == 0
        and all(value > 0.0 and math.isfinite(value) for value in gradient_norms.values())
    )
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "passed": passed,
        "source_writer_parity": public_parity,
        "counterfactual_source_parity": counterfactual_parity,
        "teacher_state_versions_unchanged": teacher_versions_before
        == teacher_versions_after,
        "teacher_gradient_tensors": teacher_gradient_tensors,
        "belief_gradient_norms": gradient_norms,
        "objective_gradient_geometry": {
            "route_norm": float(route_norm),
            "unweighted_field_norm": float(field_norm),
            "configured_weighted_field_norm": float(field_norm)
            * float(weights.field),
            "route_field_cosine": float(gradient_cosine),
        },
        "losses_finite": finite,
    }


def _host_scalars(values: dict[str, torch.Tensor]) -> dict[str, float]:
    names = tuple(values)
    packed = torch.stack(
        [values[name].detach().float().reshape(()) for name in names]
    ).cpu()
    return dict(zip(names, (float(value) for value in packed.tolist())))


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
        train_list=str(train_population["list_path"]),
        val_list=str(val_population["list_path"]),
    )
    if str(cfg["model"].get("name")) != "DynLaneSeqV28":
        raise ValueError("V28 trainer requires model.name=DynLaneSeqV28")
    if int(cfg["training"].get("seed", -1)) != FIXED_SEED:
        raise ValueError(f"V28 seed must remain {FIXED_SEED}")
    steps = (
        int(cfg["training"]["max_iters"])
        if args.mode == "gate"
        else int(args.smoke_steps)
    )
    if args.mode == "smoke" and not 1 <= steps <= 10:
        raise ValueError("V28 smoke is limited to 1..10 steps")

    device = torch.device(args.device)
    _configure_runtime(cfg, device)
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV28):
        raise TypeError("factory did not construct DynLaneSeqV28")
    v7_iteration = int(load_checkpoint(args.v7_checkpoint, model.teacher, strict=True))
    warm_start = model.router.copy_v7_encoder_(model.teacher)
    if warm_start["maximum_copy_difference"] != 0.0:
        raise RuntimeError("V28 failed exact V7 encoder warm start")
    teacher_digest = _state_digest(model.teacher)
    model.to(device)
    channels_last = bool(cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)

    weights = _weights(cfg)
    optimizer = _optimizer(model, cfg)
    start_step = 0
    if args.resume:
        start_step = int(
            load_checkpoint(
                args.resume,
                model.router,
                optimizer=optimizer,
                strict=True,
                restore_rng_state=True,
            )
        )
        if not 0 < start_step < steps:
            raise ValueError(f"resume step must lie in (0,{steps})")
    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=start_step
    )
    if len(loader.dataset) != int(train_population["expected_nonempty_rows"]):
        raise ValueError("V28 altered the official train.txt population")
    iterator = iter(loader)
    if start_step == 0:
        gate_batch, iterator = _next_batch(loader, iterator)
        gate_zero = _gate_zero(
            model,
            gate_batch,
            device=device,
            cfg=cfg,
            arm=args.arm,
            weights=weights,
        )
        if not gate_zero["passed"]:
            raise RuntimeError(
                "V28 Gate 0 failed: " + json.dumps(gate_zero, sort_keys=True)
            )
        iterator = iter(loader)
    else:
        gate_zero = {
            "passed": True,
            "reused_from_resume": str(Path(args.resume).resolve()),
        }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    metrics_handle = metrics_path.open(
        "a" if start_step else "w", encoding="utf-8"
    )
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 1))
    optimizer_cfg = cfg["v28"]["optimizer"]
    start_time = time.perf_counter()
    images_seen = start_step * int(cfg["training"]["batch_size"]) * accumulation
    run_images_seen = 0
    final_diagnostics: dict[str, float] = {}
    model.train()
    resume_path = output_dir / "resume_latest.pt"
    for step in range(start_step + 1, steps + 1):
        lr_ratio = _set_learning_rate(
            optimizer,
            step=step,
            total_steps=steps,
            warmup_steps=min(
                int(optimizer_cfg["warmup_steps"]), max(steps // 4, 1)
            ),
            minimum_ratio=float(optimizer_cfg["minimum_lr_ratio"]),
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, torch.Tensor] = {}
        for _ in range(accumulation):
            (images, targets, _metas), iterator = _next_batch(loader, iterator)
            images_seen += int(images.shape[0])
            run_images_seen += int(images.shape[0])
            images = _move_images(
                images, device=device, channels_last=channels_last
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                output = model(images, deployment_policy="source")
                loss, diagnostics = _loss(
                    output,
                    targets,
                    cfg=cfg,
                    arm=args.arm,
                    weights=weights,
                )
                scaled_loss = loss / float(accumulation)
            if not bool(torch.isfinite(scaled_loss)):
                raise FloatingPointError(f"non-finite V28 loss at step {step}")
            scaled_loss.backward()
            for name, value in diagnostics.items():
                accumulated[name] = accumulated.get(name, 0.0) + (
                    value.detach() / float(accumulation)
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.router.parameters(),
            max_norm=float(cfg["training"]["clip_grad_norm"]),
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite V28 gradient at step {step}")
        optimizer.step()
        if step == 1 or step % int(args.log_interval) == 0 or step == steps:
            scalars = _host_scalars(
                {**accumulated, "gradient_norm": gradient_norm}
            )
            final_diagnostics = dict(scalars)
            elapsed = max(time.perf_counter() - start_time, 1.0e-6)
            row = {
                "step": step,
                "arm": str(args.arm).upper(),
                "mode": args.mode,
                "images_seen": images_seen,
                "run_images_seen": run_images_seen,
                "images_per_second": float(run_images_seen) / elapsed,
                "learning_rate_ratio": lr_ratio,
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "router_lr": float(optimizer.param_groups[1]["lr"]),
                **scalars,
            }
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            metrics_handle.write(line + "\n")
            metrics_handle.flush()
        if (
            args.mode == "gate"
            and int(args.resume_interval) > 0
            and step < steps
            and step % int(args.resume_interval) == 0
        ):
            save_checkpoint(
                resume_path,
                model.router,
                optimizer=optimizer,
                iteration=step,
                cfg=cfg,
                include_rng_state=True,
            )
    metrics_handle.close()

    endpoint = output_dir / (
        "v28_gate_endpoint.pt" if args.mode == "gate" else "v28_smoke_endpoint.pt"
    )
    # Full endpoint is intentionally compatible with the repository's
    # optimized generic inference/evaluation path. Resume checkpoints remain
    # router-only so the immutable V7 payload is not duplicated every 500
    # steps.
    save_checkpoint(endpoint, model, iteration=steps, cfg=cfg)
    router_endpoint = output_dir / (
        "v28_gate_router_only.pt"
        if args.mode == "gate"
        else "v28_smoke_router_only.pt"
    )
    save_checkpoint(router_endpoint, model.router, iteration=steps, cfg=cfg)
    endpoint_teacher_digest = _state_digest(model.teacher)
    report = {
        "experiment": "V28 refined-bank slot-belief causal gate",
        "arm": str(args.arm).upper(),
        "scientific_gate": args.mode == "gate",
        "iteration": steps,
        "images_seen": images_seen,
        "official_train_epochs_seen": float(images_seen)
        / float(train_population["expected_nonempty_rows"]),
        "elapsed_seconds": time.perf_counter() - start_time,
        "maximum_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "checkpoint": str(endpoint),
        "checkpoint_sha256": sha256_file(endpoint),
        "router_only_checkpoint": str(router_endpoint),
        "router_only_checkpoint_sha256": sha256_file(router_endpoint),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "v7_checkpoint": str(Path(args.v7_checkpoint).expanduser().resolve()),
        "v7_checkpoint_sha256": sha256_file(args.v7_checkpoint),
        "v7_iteration": v7_iteration,
        "model_contract": v28_model_contract(model.router),
        "warm_start": warm_start,
        "gate_zero": gate_zero,
        "teacher_state_sha256_at_start": teacher_digest,
        "teacher_state_sha256_at_endpoint": endpoint_teacher_digest,
        "teacher_state_still_exact": endpoint_teacher_digest == teacher_digest,
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
