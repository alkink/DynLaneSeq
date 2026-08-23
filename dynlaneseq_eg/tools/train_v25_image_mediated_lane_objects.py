from __future__ import annotations

import argparse
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
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import (
    V25LossWeights,
    v25_lane_object_loss,
    v25_model_contract,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V25 image-mediated four-lane object detector."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mode", choices=("gate", "smoke"), default="gate")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-interval", type=int, default=500)
    parser.add_argument("--oof-fold-manifest", default="")
    parser.add_argument("--oof-fold-index", type=int, default=-1)
    parser.add_argument("--train-list", default="")
    return parser.parse_args()


def _configured(
    args: argparse.Namespace,
    *,
    official_train_list: str,
    official_val_list: str,
) -> dict[str, Any]:
    cfg = load_config(args.config)
    root = str(Path(args.dataset_root).expanduser().resolve())
    cfg.setdefault("dataset", {})["root"] = root
    cfg["dataset"].setdefault("lists", {})["train"] = official_train_list
    cfg["dataset"]["lists"]["val"] = official_val_list
    workers = (
        int(cfg.setdefault("dataloader", {}).get("num_workers", 0))
        if args.num_workers is None
        else int(args.num_workers)
    )
    cfg["dataloader"]["num_workers"] = workers
    cfg["dataloader"]["persistent_workers"] = workers > 0
    return cfg


def _configure_runtime(cfg: dict[str, Any], device: torch.device) -> None:
    training = cfg.get("training", {})
    threads = training.get("cpu_threads")
    if threads is not None:
        torch.set_num_threads(max(int(threads), 1))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
        if bool(training.get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")


def _move_images(
    images: torch.Tensor, *, device: torch.device, channels_last: bool
) -> torch.Tensor:
    value = images.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        value = value.contiguous(memory_format=torch.channels_last)
    return value


def _loss_weights(cfg: dict[str, Any]) -> V25LossWeights:
    raw = cfg["v25"]["loss"]
    return V25LossWeights(
        existence=float(raw["existence"]),
        row_distribution=float(raw["row_distribution"]),
        point=float(raw["point"]),
        strip_iou=float(raw["strip_iou"]),
        range=float(raw["range"]),
        quality50=float(raw["quality50"]),
        quality75=float(raw["quality75"]),
        smoothness=float(raw["smoothness"]),
        order=float(raw["order"]),
        duplicate=float(raw["duplicate"]),
        visibility=float(raw.get("visibility", 0.0)),
        proposal_coverage=float(raw.get("proposal_coverage", 0.0)),
        proposal_groups=int(raw.get("proposal_groups", 4)),
        tail_emphasis=float(raw.get("tail_emphasis", 0.0)),
        tail_iou_threshold=float(raw.get("tail_iou_threshold", 0.60)),
        tail_max_weight=float(raw.get("tail_max_weight", 2.0)),
        line_width=float(raw["line_width"]),
        minimum_valid_rows=int(raw["minimum_valid_rows"]),
        minimum_spacing_px=float(raw["minimum_spacing_px"]),
    )


def _optimizer(
    model: DynLaneSeqV25, cfg: dict[str, Any]
) -> torch.optim.Optimizer:
    raw = cfg["v25"]["optimizer"]
    backbone = list(model.detector.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    rest = [
        parameter
        for parameter in model.detector.parameters()
        if id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        (
            {
                "params": backbone,
                "lr": float(raw["backbone_lr"]),
                "initial_lr": float(raw["backbone_lr"]),
                "name": "image_backbone",
            },
            {
                "params": rest,
                "lr": float(raw["detector_lr"]),
                "initial_lr": float(raw["detector_lr"]),
                "name": "four_lane_object_decoder",
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


def _assert_finite(value: torch.Tensor, message: str) -> None:
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise FloatingPointError(message)


def _to_host(
    values: dict[str, torch.Tensor], *, extra: dict[str, torch.Tensor] | None = None
) -> dict[str, float]:
    merged = dict(values)
    if extra:
        merged.update(extra)
    names = list(merged)
    tensor = torch.stack([merged[name].detach().float().reshape(()) for name in names])
    return dict(zip(names, tensor.cpu().tolist()))


def _gradient_contract(
    model: DynLaneSeqV25,
    *,
    weights: V25LossWeights,
) -> dict[str, Any]:
    prefixes = [
        "detector.backbone",
        "detector.fpn",
        "detector.fine_stem",
        "detector.decoder",
        "detector.exist_head",
        "detector.range_head",
    ]
    detector = model.detector
    if hasattr(detector, "reliability_head"):
        fusion_enabled = bool(getattr(detector, "enable_proposal_fusion", False))
        if float(weights.quality50) > 0.0 or float(weights.quality75) > 0.0:
            prefixes.append("detector.reliability_head")
        if float(weights.visibility) > 0.0:
            prefixes.append("detector.row_visibility_head")
        if float(weights.proposal_coverage) > 0.0 or fusion_enabled:
            prefixes.append("detector.proposal_memory")
        if fusion_enabled:
            prefixes.extend(
                (
                    "detector.energy_mixture",
                    "detector.slot_proposal_query",
                    "detector.proposal_key",
                )
            )
    elif float(weights.quality50) > 0.0 or float(weights.quality75) > 0.0:
        prefixes.append("detector.quality_head")
    result: dict[str, Any] = {}
    named = list(model.named_parameters())
    for prefix in prefixes:
        gradients = [
            parameter.grad
            for name, parameter in named
            if name.startswith(prefix) and parameter.requires_grad
        ]
        finite_nonzero = any(
            gradient is not None
            and bool(torch.isfinite(gradient).all().item())
            and float(gradient.detach().abs().sum().item()) > 0.0
            for gradient in gradients
        )
        result[prefix] = {
            "parameter_tensors": len(gradients),
            "finite_nonzero_gradient": finite_nonzero,
        }
    result["passed"] = all(
        value["finite_nonzero_gradient"]
        for key, value in result.items()
        if key != "passed"
    )
    return result


def _gate_zero(
    model: DynLaneSeqV25,
    batch,
    *,
    device: torch.device,
    cfg: dict[str, Any],
    weights: V25LossWeights,
    channels_last: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    images, targets, _metas = batch
    images = _move_images(images, device=device, channels_last=channels_last)
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        output = model(images)
        loss, diagnostics = v25_lane_object_loss(
            output,
            targets,
            input_w=int(cfg["model"]["input_w"]),
            weights=weights,
        )
    _assert_finite(loss, "V25 Gate 0 produced a non-finite loss")
    loss.backward()
    gradients = _gradient_contract(model, weights=weights)
    model.zero_grad(set_to_none=True)
    contract = v25_model_contract(model.detector)
    forbidden_names = [
        name
        for name, _parameter in model.named_parameters()
        if any(token in name for token in ("teacher", "geometry_gate", "router"))
    ]
    passed = bool(gradients["passed"] and not forbidden_names)
    return {
        "passed": passed,
        "loss": float(loss.detach().cpu()),
        "diagnostics": _to_host(diagnostics),
        "gradient_contract": gradients,
        "model_contract": contract,
        "forbidden_deployment_parameters": forbidden_names,
    }


def main() -> None:
    args = parse_args()
    seed_everything(3407)
    root = Path(args.dataset_root).expanduser().resolve()
    official_train_population = official_v23_culane_list_contract(root, split="train")
    if args.oof_fold_manifest:
        if not args.train_list or args.oof_fold_index not in (0, 1):
            raise ValueError(
                "OOF training requires --train-list and --oof-fold-index 0/1"
            )
        from dynlaneseq_eg.tools.build_v25_s1_oof_folds import (
            validate_fold_training_population,
        )

        train_population = validate_fold_training_population(
            args.oof_fold_manifest,
            fold=args.oof_fold_index,
            supplied_train_list=args.train_list,
        )
    else:
        if args.train_list or args.oof_fold_index >= 0:
            raise ValueError("custom train lists are allowed only by the OOF contract")
        train_population = official_train_population
    val_population = official_v23_culane_list_contract(root, split="val")
    cfg = _configured(
        args,
        official_train_list=str(train_population["list_path"]),
        official_val_list=str(val_population["list_path"]),
    )
    if str(cfg["model"].get("name")) != "DynLaneSeqV25":
        raise ValueError("V25 trainer requires model.name=DynLaneSeqV25")
    if int(cfg["training"].get("seed", -1)) != 3407:
        raise ValueError("V25 component gates use fixed seed 3407")
    batch_size = int(cfg["training"]["batch_size"])
    accumulation_steps = int(
        cfg["training"].get("gradient_accumulation_steps", 1)
    )
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    effective_batch = batch_size * accumulation_steps
    expected_steps = math.ceil(
        int(train_population["expected_nonempty_rows"]) / effective_batch
    )
    steps = expected_steps if args.mode == "gate" else int(args.smoke_steps)
    if (
        args.mode == "gate"
        and not args.oof_fold_manifest
        and int(cfg["training"]["max_iters"]) != expected_steps
    ):
        raise ValueError(
            "V25 G0 must consume exactly one official train-list epoch: "
            f"expected max_iters={expected_steps}"
        )
    if args.mode == "smoke" and not 1 <= steps <= 10:
        raise ValueError("V25 smoke mode is limited to 1..10 steps")

    device = torch.device(args.device)
    _configure_runtime(cfg, device)
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("factory did not construct DynLaneSeqV25")
    model.to(device)
    channels_last = bool(cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    weights = _loss_weights(cfg)
    optimizer = _optimizer(model, cfg)
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    amp_dtype_name = str(cfg["training"].get("amp_dtype", "bfloat16"))
    amp_torch_dtype = (
        torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    )
    start_step = 0
    resumed_from = ""
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
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
            raise ValueError(f"resume iteration must be in (0,{steps})")
        resumed_from = str(resume_path)
    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=start_step
    )
    if len(loader.dataset) != int(train_population["expected_nonempty_rows"]):
        raise ValueError("V25 loader altered the official train.txt population")
    iterator = iter(loader)
    if start_step == 0:
        first_batch, iterator = _next_batch(loader, iterator)
        gate_zero = _gate_zero(
            model,
            first_batch,
            device=device,
            cfg=cfg,
            weights=weights,
            channels_last=channels_last,
            amp_enabled=amp_enabled,
            amp_dtype=amp_torch_dtype,
        )
        if not gate_zero["passed"]:
            raise RuntimeError("V25 Gate 0 failed: " + json.dumps(gate_zero))
        iterator = iter(loader)
    else:
        gate_zero = {"passed": True, "reused_from_resume": resumed_from}

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    metrics = metrics_path.open("a" if start_step else "w", encoding="utf-8")
    clip_norm = float(cfg["training"].get("clip_grad_norm", 5.0))
    opt_cfg = cfg["v25"]["optimizer"]
    start_time = time.perf_counter()
    images_seen = start_step * effective_batch
    run_images_seen = 0
    final_diagnostics: dict[str, float] = {}
    clipped_log_steps = 0
    logged_steps = 0
    resume_checkpoint = output_dir / "resume_latest.pt"
    model.train()
    for step in range(start_step + 1, steps + 1):
        ratio = _set_learning_rate(
            optimizer,
            step=step,
            total_steps=steps,
            warmup_steps=min(int(opt_cfg["warmup_steps"]), max(steps // 10, 1)),
            minimum_ratio=float(opt_cfg["minimum_lr_ratio"]),
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, torch.Tensor] = {}
        for _micro_step in range(accumulation_steps):
            (images, targets, _metas), iterator = _next_batch(loader, iterator)
            images_seen += int(images.shape[0])
            run_images_seen += int(images.shape[0])
            images = _move_images(
                images, device=device, channels_last=channels_last
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_torch_dtype,
                enabled=amp_enabled,
            ):
                output = model(images)
                loss, diagnostics = v25_lane_object_loss(
                    output,
                    targets,
                    input_w=int(cfg["model"]["input_w"]),
                    weights=weights,
                )
                scaled_loss = loss / float(accumulation_steps)
            _assert_finite(
                scaled_loss, f"non-finite V25 loss at step {step}"
            )
            scaled_loss.backward()
            for name, value in diagnostics.items():
                detached = value.detach() / float(accumulation_steps)
                accumulated[name] = accumulated.get(name, 0.0) + detached
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_norm
        )
        _assert_finite(gradient_norm, f"non-finite V25 gradient at step {step}")
        optimizer.step()
        should_log = step == 1 or step % int(args.log_interval) == 0 or step == steps
        if should_log:
            host = _to_host(accumulated, extra={"gradient_norm": gradient_norm})
            gradient_value = host.pop("gradient_norm")
            clipped = gradient_value > clip_norm
            clipped_log_steps += int(clipped)
            logged_steps += 1
            final_diagnostics = dict(host)
            elapsed = max(time.perf_counter() - start_time, 1.0e-6)
            row = {
                "step": step,
                "mode": args.mode,
                "images_seen": images_seen,
                "run_images_seen": run_images_seen,
                "images_per_second": float(run_images_seen) / elapsed,
                "gradient_norm_pre_clip": gradient_value,
                "gradient_was_clipped": clipped,
                "learning_rate_ratio": ratio,
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "detector_lr": float(optimizer.param_groups[1]["lr"]),
                **host,
            }
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            metrics.write(line + "\n")
            metrics.flush()
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
    metrics.close()
    checkpoint = output_dir / (
        "v25_g0_endpoint.pt" if args.mode == "gate" else "v25_smoke_endpoint.pt"
    )
    save_checkpoint(checkpoint, model, iteration=steps, cfg=cfg)
    elapsed = time.perf_counter() - start_time
    report = {
        "experiment": "V25 G0 direct image-mediated four-lane objects",
        "scientific_gate": args.mode == "gate",
        "iteration": steps,
        "images_seen": images_seen,
        "physical_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch,
        "complete_official_train_epochs_seen": float(images_seen)
        / float(official_train_population["expected_nonempty_rows"]),
        "complete_loader_population_epochs_seen": float(images_seen)
        / float(train_population["expected_nonempty_rows"]),
        "elapsed_seconds": elapsed,
        "images_per_second": float(run_images_seen) / max(elapsed, 1.0e-6),
        "maximum_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "model_contract": v25_model_contract(model.detector),
        "gate_zero": gate_zero,
        "official_train_population_contract": train_population,
        "untouched_official_train_population_contract": official_train_population,
        "oof_fold_index": int(args.oof_fold_index),
        "oof_fold_training": bool(args.oof_fold_manifest),
        "official_val_population_contract": val_population,
        "final_training_diagnostics": final_diagnostics,
        "logged_gradient_clip_fraction": float(clipped_log_steps)
        / float(max(logged_steps, 1)),
        "start_iteration": start_step,
        "resumed_from": resumed_from,
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
