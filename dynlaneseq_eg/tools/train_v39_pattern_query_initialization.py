from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import (
    _torch_load,
    load_checkpoint,
    remap_optimizer_state_by_parameter,
    restore_checkpoint_rng_state,
    save_checkpoint,
)
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import (
    ImageConditionedPatternQueryInitializer,
    v25_lane_object_loss,
    v25_model_contract,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v25_image_mediated_lane_objects import (
    _assert_finite,
    _configure_runtime,
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
SOURCE_ITERATION = 50_000
FIXED_ENDPOINT = 65_000
FIXED_SCHEDULE = 278_000
FIXED_EFFECTIVE_BATCH = 16
PRIMARY_OUTPUTS = (
    "exist_logits",
    "pred_x_rows",
    "range_norm",
    "quality_logits",
    "unary_logits",
    "path_logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one exact-paired V39 pattern-query arm from V38 50K."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern-bank", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mode", choices=("scientific", "smoke"), default="scientific")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def _configured(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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


def validate_v39_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    arm = str(cfg.get("v39", {}).get("arm", ""))
    enabled = bool(
        cfg.get("v25", {}).get("pattern_query_initializer", {}).get(
            "enabled", False
        )
    )
    effective_batch = int(cfg["training"]["batch_size"]) * int(
        cfg["training"].get("gradient_accumulation_steps", 1)
    )
    checks = {
        "arm_valid": arm in {"control", "treatment"},
        "treatment_flag_exact": enabled == (arm == "treatment"),
        "model_is_v25": str(cfg["model"].get("name")) == "DynLaneSeqV25",
        "seed_exact": int(cfg["training"].get("seed", -1)) == FIXED_SEED,
        "endpoint_exact": int(cfg["training"].get("max_iters", -1))
        == FIXED_ENDPOINT,
        "v39_source_exact": int(cfg["v39"].get("source_iteration", -1))
        == SOURCE_ITERATION,
        "v39_endpoint_exact": int(cfg["v39"].get("endpoint_iteration", -1))
        == FIXED_ENDPOINT,
        "schedule_exact": int(cfg["v39"].get("scheduler_total_iters", -1))
        == FIXED_SCHEDULE,
        "effective_batch_exact": effective_batch == FIXED_EFFECTIVE_BATCH,
        "resume_safe_loader": bool(cfg.get("dataloader", {}).get("resume_safe")),
        "dropout_zero": float(cfg["v25"].get("dropout", -1.0)) == 0.0,
        "expectation_decode": str(cfg["v25"].get("decode_mode")) == "expectation",
        "quality50_disabled": float(cfg["v25"]["loss"].get("quality50", -1.0))
        == 0.0,
        "quality75_disabled": float(cfg["v25"]["loss"].get("quality75", -1.0))
        == 0.0,
        "competition_disabled": not bool(
            cfg["v25"].get("enable_competition", False)
        ),
        "slot_interaction_disabled": not bool(
            cfg["v25"].get("enable_slot_interaction", False)
        ),
        "no_nms": float(
            cfg["postprocess"].get("lane_nms_distance_thresh_px", -1.0)
        )
        == 0.0,
        "top4": int(cfg["postprocess"].get("top_k", -1)) == 4,
        "exist_score": str(cfg["postprocess"].get("score_mode")) == "exist",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "arm": arm,
        "effective_batch": effective_batch,
    }


def _load_pattern_bank(path: str) -> tuple[torch.Tensor, dict[str, Any]]:
    pattern_path = Path(path).expanduser().resolve()
    payload = json.loads(pattern_path.read_text(encoding="utf-8"))
    patterns = torch.tensor(payload.get("patterns"), dtype=torch.float32)
    expected = (4, 16, 160)
    if tuple(patterns.shape) != expected or payload.get("test_set_used") is not False:
        raise ValueError("V39 pattern bank contract failed")
    if payload.get("validation_set_used") is not False:
        raise ValueError("V39 pattern bank must be train-only")
    return patterns, {
        "path": str(pattern_path),
        "sha256": sha256_file(pattern_path),
        "metadata": {key: value for key, value in payload.items() if key != "patterns"},
    }


def _load_source_state(
    model: DynLaneSeqV25,
    optimizer: torch.optim.Optimizer,
    *,
    source_checkpoint: Path,
    cfg: dict[str, Any],
    patterns: torch.Tensor | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _torch_load(source_checkpoint)
    if int(payload.get("iteration", -1)) != SOURCE_ITERATION:
        raise ValueError("V39 requires the fixed V38 50K source checkpoint")
    initializer = model.detector.pattern_query_initializer
    if initializer is None:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        remap = {
            "parameters": sum(len(group["params"]) for group in optimizer.param_groups),
            "state_entries": len(optimizer.state),
            "new_parameters": 0,
        }
    else:
        if not isinstance(initializer, ImageConditionedPatternQueryInitializer):
            raise TypeError("V39 treatment constructed an unsupported initializer")
        incompatible = model.load_state_dict(payload["model"], strict=False)
        allowed_prefix = "detector.pattern_query_initializer."
        if incompatible.unexpected_keys or any(
            not name.startswith(allowed_prefix) for name in incompatible.missing_keys
        ):
            raise ValueError(
                "V39 source checkpoint has unexpected incompatibilities: "
                + repr(incompatible)
            )
        if patterns is None:
            raise ValueError("V39 treatment requires a train-only pattern bank")
        initializer.set_pattern_bank(patterns)
        new_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name not in payload["model"]
        ]
        original = {parameter: parameter.requires_grad for _, parameter in new_parameters}
        try:
            for _, parameter in new_parameters:
                parameter.requires_grad_(False)
            source_optimizer = _optimizer(model, payload.get("cfg", cfg))
        finally:
            for _, parameter in new_parameters:
                parameter.requires_grad_(original[parameter])
        source_optimizer.load_state_dict(payload["optimizer"])
        remap = remap_optimizer_state_by_parameter(
            source_optimizer, optimizer, allow_target_superset=True
        )
        remap["new_parameter_names"] = [name for name, _ in new_parameters]
        del source_optimizer
    if not restore_checkpoint_rng_state(payload):
        raise ValueError("V39 exact pair requires RNG state in the V38 source")
    return payload, remap


def _batch_manifest(metas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": int(meta["sample_index"]),
            "augmentation_seed": int(meta["augmentation_seed"]),
            "image_path": str(meta["image_path"]),
        }
        for meta in metas
    ]


def _zero_parity_and_gradient_gate(
    model: DynLaneSeqV25,
    batch,
    *,
    device: torch.device,
    cfg: dict[str, Any],
    channels_last: bool,
) -> dict[str, Any]:
    initializer = model.detector.pattern_query_initializer
    if initializer is None:
        return {"passed": True, "applicable": False}
    images, targets, _metas = batch
    # The parity gate performs three forwards (source, zero-init treatment,
    # gradient population).  A single deterministic example is sufficient for
    # this mathematical contract and avoids tripling V38's already large
    # full-batch activation footprint on a 32 GB GPU.
    images = _move_images(images[:1], device=device, channels_last=channels_last)
    targets = targets[:1]
    canonical = model.detector.canonical_slot_centres.view(1, 4, 1).expand(
        images.shape[0], 4, model.detector.num_rows
    )
    model.eval()
    with torch.no_grad():
        source = model(images, query_anchor_x_rows=canonical)
        source_cpu = {name: source[name].detach().cpu() for name in PRIMARY_OUTPUTS}
        del source
        if device.type == "cuda":
            torch.cuda.empty_cache()
        treatment = model(images)
        exact = {
            name: bool(torch.equal(treatment[name].detach().cpu(), source_cpu[name]))
            for name in PRIMARY_OUTPUTS
        }
        anchor_exact = bool(
            torch.equal(
                treatment["query_anchor_x_rows"].detach().cpu(), canonical.cpu()
            )
        )
        del treatment, source_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(images)
    loss, _diagnostics = v25_lane_object_loss(
        output,
        targets,
        input_w=int(cfg["model"]["input_w"]),
        weights=_loss_weights(cfg),
    )
    loss.backward()
    gradient = initializer.pattern_and_gate.weight.grad
    gradient_finite = gradient is not None and bool(torch.isfinite(gradient).all())
    gradient_nonzero = gradient_finite and float(gradient.abs().sum()) > 0.0
    model.zero_grad(set_to_none=True)
    result = {
        "applicable": True,
        "examples_checked": int(images.shape[0]),
        "primary_outputs_bit_exact": exact,
        "query_anchor_bit_exact": anchor_exact,
        "initializer_output_gradient_finite": gradient_finite,
        "initializer_output_gradient_nonzero": gradient_nonzero,
    }
    result["passed"] = all(exact.values()) and anchor_exact and gradient_nonzero
    return result


def main() -> None:
    args = parse_args()
    seed_everything(FIXED_SEED)
    cfg, train_population, val_population = _configured(args)
    contract = validate_v39_contract(cfg)
    if not contract["passed"]:
        raise ValueError("V39 config contract failed: " + json.dumps(contract))
    if args.mode == "smoke" and not 1 <= int(args.smoke_steps) <= 10:
        raise ValueError("V39 smoke mode is limited to 1..10 continuation steps")
    arm = str(contract["arm"])
    pattern_tensor = None
    pattern_contract: dict[str, Any] | None = None
    if arm == "treatment":
        if not args.pattern_bank:
            raise ValueError("V39 treatment requires --pattern-bank")
        pattern_tensor, pattern_contract = _load_pattern_bank(args.pattern_bank)
    elif args.pattern_bank:
        raise ValueError("V39 control must not consume the treatment pattern bank")

    device = torch.device(args.device)
    _configure_runtime(cfg, device)
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("V39 factory did not construct DynLaneSeqV25")
    model.to(device)
    channels_last = bool(cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = _optimizer(model, cfg)
    source_checkpoint = Path(args.source_checkpoint).expanduser().resolve()
    source_payload: dict[str, Any] | None = None
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
        if not SOURCE_ITERATION < start_iteration < FIXED_ENDPOINT:
            raise ValueError("V39 resume iteration is outside the continuation")
        optimizer_remap = {
            "resumed_existing_v39_checkpoint": True,
            "iteration": start_iteration,
        }
        resumed_from = str(resume_path)
    else:
        start_iteration = SOURCE_ITERATION
        source_payload, optimizer_remap = _load_source_state(
            model,
            optimizer,
            source_checkpoint=source_checkpoint,
            cfg=cfg,
            patterns=pattern_tensor,
        )

    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=start_iteration
    )
    if len(loader.dataset) != int(train_population["expected_nonempty_rows"]):
        raise ValueError("V39 loader altered official train.txt")
    iterator = iter(loader)
    first_batch, iterator = _next_batch(loader, iterator)
    current_manifest = _batch_manifest(first_batch[2])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_contract_path = output_dir / "pair_start_contract.json"
    if start_iteration == SOURCE_ITERATION:
        first_manifest = current_manifest
        gate = _zero_parity_and_gradient_gate(
            model,
            first_batch,
            device=device,
            cfg=cfg,
            channels_last=channels_last,
        )
        if not gate["passed"]:
            raise RuntimeError("V39 zero/gradient gate failed: " + json.dumps(gate))
        if source_payload is None or not restore_checkpoint_rng_state(source_payload):
            raise ValueError("V39 failed to restore paired RNG after its safety gate")
        start_contract_path.write_text(
            json.dumps(
                {
                    "arm": arm,
                    "source_iteration": SOURCE_ITERATION,
                    "first_batch_manifest": first_manifest,
                    "zero_and_gradient_gate": gate,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        if not start_contract_path.is_file():
            raise ValueError("V39 resume is missing pair_start_contract.json")
        start_contract = json.loads(start_contract_path.read_text(encoding="utf-8"))
        first_manifest = start_contract["first_batch_manifest"]
        gate = start_contract["zero_and_gradient_gate"]
    iterator = iter(loader)

    endpoint = (
        FIXED_ENDPOINT
        if args.mode == "scientific"
        else SOURCE_ITERATION + int(args.smoke_steps)
    )
    metrics_path = output_dir / "train_metrics.jsonl"
    metrics = metrics_path.open("a" if start_iteration > SOURCE_ITERATION else "w", encoding="utf-8")
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 1))
    effective_batch = int(cfg["training"]["batch_size"]) * accumulation
    clip_norm = float(cfg["training"].get("clip_grad_norm", 50.0))
    opt_cfg = cfg["v25"]["optimizer"]
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
    final_diagnostics: dict[str, float] = {}
    model.train()
    for global_step in range(start_iteration + 1, endpoint + 1):
        ratio = _set_learning_rate(
            optimizer,
            step=global_step,
            total_steps=FIXED_SCHEDULE,
            warmup_steps=int(opt_cfg["warmup_steps"]),
            minimum_ratio=float(opt_cfg["minimum_lr_ratio"]),
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, torch.Tensor] = {}
        for _micro_step in range(accumulation):
            (images, targets, _metas), iterator = _next_batch(loader, iterator)
            run_images += int(images.shape[0])
            images = _move_images(images, device=device, channels_last=channels_last)
            amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
            amp_dtype = (
                torch.bfloat16
                if str(cfg["training"].get("amp_dtype", "bfloat16")) == "bfloat16"
                else torch.float16
            )
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                output = model(images)
                loss, diagnostics = v25_lane_object_loss(
                    output,
                    targets,
                    input_w=int(cfg["model"]["input_w"]),
                    weights=_loss_weights(cfg),
                )
                scaled = loss / float(accumulation)
            _assert_finite(scaled, f"non-finite V39 loss at {global_step}")
            scaled.backward()
            if "pattern_gate" in output:
                diagnostics = dict(diagnostics)
                diagnostics["v39_pattern_gate_abs_mean"] = output[
                    "pattern_gate"
                ].detach().abs().mean()
                canonical = model.detector.canonical_slot_centres.view(1, 4, 1)
                diagnostics["v39_anchor_delta_abs_mean"] = (
                    output["query_anchor_x_rows"].detach() - canonical
                ).abs().mean()
            for name, value in diagnostics.items():
                accumulated[name] = accumulated.get(name, 0.0) + value.detach() / float(
                    accumulation
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_norm
        )
        _assert_finite(gradient_norm, f"non-finite V39 gradient at {global_step}")
        optimizer.step()

        should_log = (
            global_step == SOURCE_ITERATION + 1
            or global_step % log_interval == 0
            or global_step == endpoint
        )
        if should_log:
            host = _to_host(accumulated, extra={"gradient_norm": gradient_norm})
            gradient_value = host.pop("gradient_norm")
            final_diagnostics = dict(host)
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            row = {
                "arm": arm,
                "global_step": global_step,
                "run_images_seen": run_images,
                "images_per_second": run_images / elapsed,
                "learning_rate_ratio": ratio,
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "detector_lr": float(optimizer.param_groups[1]["lr"]),
                "gradient_norm_pre_clip": gradient_value,
                "gradient_was_clipped": gradient_value > clip_norm,
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
            for name in (f"iter_{global_step:07d}.pt", "resume_latest.pt"):
                save_checkpoint(
                    output_dir / name,
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
        "experiment": "V39 exact-paired image-conditioned pattern query initialization",
        "arm": arm,
        "scientific_gate": args.mode == "scientific",
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_iteration": SOURCE_ITERATION,
        "process_start_iteration": start_iteration,
        "resumed_from": resumed_from,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "iteration": endpoint,
        "effective_batch_size": effective_batch,
        "scheduler_total_iters": FIXED_SCHEDULE,
        "run_effective_images": run_images,
        "elapsed_seconds": elapsed,
        "images_per_second": run_images / max(elapsed, 1.0e-9),
        "maximum_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "first_batch_manifest": first_manifest,
        "optimizer_state_remap": optimizer_remap,
        "zero_and_gradient_gate": gate,
        "pattern_bank_contract": pattern_contract,
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
