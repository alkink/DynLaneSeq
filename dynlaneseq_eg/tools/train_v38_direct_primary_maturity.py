from __future__ import annotations

import argparse
import json
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
    v25_lane_object_loss,
    v25_model_contract,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v25_image_mediated_lane_objects import (
    _assert_finite,
    _configure_runtime,
    _gate_zero,
    _loss_weights,
    _move_images,
    _next_batch,
    _optimizer,
    _set_learning_rate,
    _to_host,
)
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


FIXED_SEED = 3407
FIXED_ENDPOINT = 50_000
FIXED_SCHEDULE = 278_000
FIXED_EFFECTIVE_BATCH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fixed V38 direct-primary 50K maturity gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mode", choices=("scientific", "smoke"), default="scientific")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def _configured(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(args.dataset_root).expanduser().resolve()
    train_population = official_v23_culane_list_contract(root, split="train")
    val_population = official_v23_culane_list_contract(root, split="val")
    cfg = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(root)
    cfg["dataset"].setdefault("lists", {})["train"] = str(
        train_population["list_path"]
    )
    cfg["dataset"]["lists"]["val"] = str(val_population["list_path"])
    workers = (
        int(cfg.setdefault("dataloader", {}).get("num_workers", 0))
        if args.num_workers is None
        else int(args.num_workers)
    )
    cfg["dataloader"]["num_workers"] = workers
    cfg["dataloader"]["persistent_workers"] = workers > 0
    return cfg, train_population, val_population


def validate_v38_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    training = cfg["training"]
    v38 = cfg["v38"]
    v25 = cfg["v25"]
    effective_batch = int(training["batch_size"]) * int(
        training.get("gradient_accumulation_steps", 1)
    )
    checks = {
        "model_is_v25": str(cfg["model"].get("name")) == "DynLaneSeqV25",
        "seed_exact": int(training.get("seed", -1)) == FIXED_SEED,
        "endpoint_exact": int(training.get("max_iters", -1)) == FIXED_ENDPOINT,
        "v38_endpoint_exact": int(v38.get("endpoint_iteration", -1))
        == FIXED_ENDPOINT,
        "schedule_exact": int(v38.get("scheduler_total_iters", -1))
        == FIXED_SCHEDULE,
        "effective_batch_exact": effective_batch == FIXED_EFFECTIVE_BATCH,
        "reference_effective_batch_exact": int(
            v38.get("reference_effective_batch", -1)
        )
        == FIXED_EFFECTIVE_BATCH,
        "reference_v7_iteration_exact": int(
            v38.get("reference_v7_iteration", -1)
        )
        == FIXED_ENDPOINT,
        "resume_safe_loader": bool(cfg.get("dataloader", {}).get("resume_safe")),
        "competition_disabled": not bool(v25.get("enable_competition", False)),
        "slot_interaction_disabled": not bool(
            v25.get("enable_slot_interaction", False)
        ),
        "dropout_zero": float(v25.get("dropout", -1.0)) == 0.0,
        "expectation_decode": str(v25.get("decode_mode")) == "expectation",
        "quality50_disabled": float(v25["loss"].get("quality50", -1.0)) == 0.0,
        "quality75_disabled": float(v25["loss"].get("quality75", -1.0)) == 0.0,
        "pretrained_backbone": bool(cfg["model"].get("pretrained_backbone")),
        "no_nms": float(cfg["postprocess"].get("lane_nms_distance_thresh_px", -1.0))
        == 0.0,
        "top4": int(cfg["postprocess"].get("top_k", -1)) == 4,
        "exist_score": str(cfg["postprocess"].get("score_mode")) == "exist",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "effective_batch": effective_batch,
        "endpoint_iteration": FIXED_ENDPOINT,
        "scheduler_total_iters": FIXED_SCHEDULE,
    }


def main() -> None:
    args = parse_args()
    seed_everything(FIXED_SEED)
    cfg, train_population, val_population = _configured(args)
    contract = validate_v38_contract(cfg)
    if not contract["passed"]:
        raise ValueError("V38 config contract failed: " + json.dumps(contract))
    if args.mode == "smoke" and not 1 <= int(args.smoke_steps) <= 10:
        raise ValueError("V38 smoke mode is limited to 1..10 steps")

    device = torch.device(args.device)
    _configure_runtime(cfg, device)
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("V38 factory did not construct DynLaneSeqV25")
    model.to(device)
    channels_last = bool(cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    weights = _loss_weights(cfg)
    optimizer = _optimizer(model, cfg)

    endpoint = FIXED_ENDPOINT if args.mode == "scientific" else int(args.smoke_steps)
    start_iteration = 0
    resumed_from = ""
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        start_iteration = int(
            load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                strict=True,
                restore_rng_state=True,
            )
        )
        if not 0 < start_iteration < endpoint:
            raise ValueError("V38 resume iteration is outside the active run")
        resumed_from = str(resume_path)

    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=start_iteration
    )
    if len(loader.dataset) != int(train_population["expected_nonempty_rows"]):
        raise ValueError("V38 loader altered official train.txt")
    iterator = iter(loader)
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    amp_name = str(cfg["training"].get("amp_dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    if start_iteration == 0:
        first_batch, iterator = _next_batch(loader, iterator)
        gate_zero = _gate_zero(
            model,
            first_batch,
            device=device,
            cfg=cfg,
            weights=weights,
            channels_last=channels_last,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        if not gate_zero["passed"]:
            raise RuntimeError("V38 Gate 0 failed: " + json.dumps(gate_zero))
        iterator = iter(loader)
    else:
        gate_zero = {"passed": True, "reused_from_resume": resumed_from}

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    metrics = metrics_path.open("a" if start_iteration else "w", encoding="utf-8")
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 1))
    effective_batch = int(cfg["training"]["batch_size"]) * accumulation
    clip_norm = float(cfg["training"].get("clip_grad_norm", 50.0))
    opt_cfg = cfg["v25"]["optimizer"]
    warmup_steps = int(opt_cfg["warmup_steps"])
    minimum_ratio = float(opt_cfg["minimum_lr_ratio"])
    log_interval = int(
        cfg["training"].get("log_interval", 25)
        if args.log_interval is None
        else args.log_interval
    )
    checkpoint_interval = int(
        cfg["training"].get("checkpoint_interval", 2500)
        if args.checkpoint_interval is None
        else args.checkpoint_interval
    )
    started = time.perf_counter()
    run_images = 0
    logged_steps = 0
    clipped_steps = 0
    final_diagnostics: dict[str, float] = {}
    model.train()
    for global_step in range(start_iteration + 1, endpoint + 1):
        ratio = _set_learning_rate(
            optimizer,
            step=global_step,
            total_steps=FIXED_SCHEDULE,
            warmup_steps=warmup_steps,
            minimum_ratio=minimum_ratio,
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, torch.Tensor] = {}
        for _micro_step in range(accumulation):
            (images, targets, _metas), iterator = _next_batch(loader, iterator)
            run_images += int(images.shape[0])
            images = _move_images(images, device=device, channels_last=channels_last)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                output = model(images)
                loss, diagnostics = v25_lane_object_loss(
                    output,
                    targets,
                    input_w=int(cfg["model"]["input_w"]),
                    weights=weights,
                )
                scaled = loss / float(accumulation)
            _assert_finite(scaled, f"non-finite V38 loss at {global_step}")
            scaled.backward()
            for name, value in diagnostics.items():
                accumulated[name] = (
                    accumulated.get(name, 0.0)
                    + value.detach() / float(accumulation)
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_norm
        )
        _assert_finite(gradient_norm, f"non-finite V38 gradient at {global_step}")
        optimizer.step()

        should_log = (
            global_step == 1
            or global_step % log_interval == 0
            or global_step == endpoint
        )
        if should_log:
            host = _to_host(accumulated, extra={"gradient_norm": gradient_norm})
            gradient_value = host.pop("gradient_norm")
            final_diagnostics = dict(host)
            clipped = gradient_value > clip_norm
            logged_steps += 1
            clipped_steps += int(clipped)
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            row = {
                "global_step": global_step,
                "run_images_seen": run_images,
                "total_effective_images_seen": global_step * effective_batch,
                "images_per_second": run_images / elapsed,
                "learning_rate_ratio": ratio,
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "detector_lr": float(optimizer.param_groups[1]["lr"]),
                "gradient_norm_pre_clip": gradient_value,
                "gradient_was_clipped": clipped,
                **host,
            }
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            metrics.write(line + "\n")
            metrics.flush()

        if (
            args.mode == "scientific"
            and checkpoint_interval > 0
            and global_step < endpoint
            and global_step % checkpoint_interval == 0
        ):
            save_checkpoint(
                output_dir / f"iter_{global_step:07d}.pt",
                model,
                optimizer=optimizer,
                iteration=global_step,
                cfg=cfg,
                include_rng_state=True,
            )
            save_checkpoint(
                output_dir / "resume_latest.pt",
                model,
                optimizer=optimizer,
                iteration=global_step,
                cfg=cfg,
                include_rng_state=True,
            )

    metrics.close()
    checkpoint = output_dir / (
        f"iter_{endpoint:07d}.pt" if args.mode == "scientific" else "smoke_endpoint.pt"
    )
    save_checkpoint(
        checkpoint,
        model,
        optimizer=optimizer,
        iteration=endpoint,
        cfg=cfg,
        include_rng_state=True,
    )
    elapsed = time.perf_counter() - started
    report = {
        "experiment": "V38 direct-primary matched-50K maturity gate",
        "scientific_gate": args.mode == "scientific",
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "iteration": endpoint,
        "start_iteration": start_iteration,
        "resumed_from": resumed_from,
        "run_effective_images": run_images,
        "total_effective_images": endpoint * effective_batch,
        "complete_official_train_epochs_seen": float(endpoint * effective_batch)
        / float(train_population["expected_nonempty_rows"]),
        "effective_batch_size": effective_batch,
        "scheduler_total_iters": FIXED_SCHEDULE,
        "elapsed_seconds": elapsed,
        "images_per_second": run_images / max(elapsed, 1.0e-9),
        "maximum_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "logged_gradient_clip_fraction": float(clipped_steps)
        / float(max(logged_steps, 1)),
        "final_training_diagnostics": final_diagnostics,
        "config_contract": contract,
        "model_contract": v25_model_contract(model.detector),
        "official_train_population_contract": train_population,
        "official_val_population_contract": val_population,
        "checkpoint_selection_performed": False,
        "threshold_selection_performed": False,
        "test_set_used": False,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
