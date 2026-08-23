from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch

from dynlaneseq_eg.engine.checkpoint import load_checkpoint, save_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.modeling.v25_denoising import (
    build_denoising_query_anchors,
)
from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import (
    build_ordered_lane_targets,
    v25_lane_object_loss,
    v25_model_contract,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v25_image_mediated_lane_objects import (
    _assert_finite,
    _configure_runtime,
    _configured,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired quarter-epoch continuation for one V25 component."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument(
        "--allow-advanced-init",
        action="store_true",
        help="Load every shared G0/G2 tensor exactly and initialize only new dual-energy modules.",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--component-name", required=True)
    parser.add_argument(
        "--expected-init-iteration",
        type=int,
        default=0,
        help="Exact parent iteration; 0 means the one-epoch G0 endpoint.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mode", choices=("gate", "smoke"), default="gate")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-interval", type=int, default=250)
    parser.add_argument(
        "--reset-runtime-rng-after-init",
        action="store_true",
        help=(
            "Reset Python/NumPy/PyTorch RNG after model/checkpoint/optimizer "
            "construction. This makes paired arms with different private "
            "module sets start the dataloader and stochastic runtime from the "
            "same state."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(FIXED_SEED)
    root = Path(args.dataset_root).expanduser().resolve()
    train_population = official_v23_culane_list_contract(root, split="train")
    val_population = official_v23_culane_list_contract(root, split="val")
    cfg = _configured(
        args,
        official_train_list=str(train_population["list_path"]),
        official_val_list=str(val_population["list_path"]),
    )
    if str(cfg["model"].get("name")) != "DynLaneSeqV25":
        raise ValueError("V25 component trainer requires DynLaneSeqV25")
    if int(cfg["training"].get("seed", -1)) != FIXED_SEED:
        raise ValueError("V25 component gates use fixed seed 3407")
    physical_batch = int(cfg["training"]["batch_size"])
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 1))
    effective_batch = physical_batch * accumulation
    full_epoch_steps = math.ceil(
        int(train_population["expected_nonempty_rows"]) / effective_batch
    )
    fixed_gate_steps = math.ceil(full_epoch_steps / 4.0)
    component_steps = (
        fixed_gate_steps if args.mode == "gate" else int(args.smoke_steps)
    )
    if args.mode == "smoke" and not 1 <= component_steps <= 10:
        raise ValueError("component smoke mode is limited to 1..10 steps")

    device = torch.device(args.device)
    _configure_runtime(cfg, device)
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("factory did not construct DynLaneSeqV25")
    init_checkpoint = Path(args.init_checkpoint).expanduser().resolve()
    if args.allow_advanced_init:
        base_cfg = copy.deepcopy(cfg)
        base_cfg.setdefault("v25", {})["enable_dual_energy_multi_path"] = False
        base_model = build_model(base_cfg)
        base_iteration = int(load_checkpoint(init_checkpoint, base_model, strict=True))
        init_iteration = int(load_checkpoint(init_checkpoint, model, strict=False))
        advanced_state = model.state_dict()
        shared_mismatch = [
            name
            for name, value in base_model.state_dict().items()
            if name not in advanced_state
            or tuple(advanced_state[name].shape) != tuple(value.shape)
            or not torch.equal(advanced_state[name].cpu(), value.cpu())
        ]
        if base_iteration != init_iteration or shared_mismatch:
            raise ValueError(
                "advanced V25 initialization did not preserve every shared tensor: "
                + json.dumps(
                    {
                        "base_iteration": base_iteration,
                        "advanced_iteration": init_iteration,
                        "mismatch_count": len(shared_mismatch),
                        "first_mismatches": shared_mismatch[:10],
                    }
                )
            )
        del base_model
    else:
        init_iteration = int(load_checkpoint(init_checkpoint, model, strict=True))
    expected_init_iteration = (
        int(args.expected_init_iteration)
        if int(args.expected_init_iteration) > 0
        else full_epoch_steps
    )
    if init_iteration != expected_init_iteration:
        raise ValueError(
            f"V25 component init must be iteration {expected_init_iteration}, "
            f"got {init_iteration}"
        )
    model.to(device)
    channels_last = bool(cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    weights = _loss_weights(cfg)
    denoising_cfg = cfg["v25"].get("denoising", {})
    denoising_enabled = bool(denoising_cfg.get("enabled", False))
    denoising_weight = float(denoising_cfg.get("loss_weight", 1.0))
    if denoising_enabled and denoising_weight <= 0.0:
        raise ValueError("enabled denoising requires a positive loss_weight")
    optimizer = _optimizer(model, cfg)
    runtime_seed = FIXED_SEED
    if args.reset_runtime_rng_after_init:
        # Model construction legitimately consumes a different number of RNG
        # draws when a treatment owns private auxiliary modules.  Those draws
        # must not change the paired augmentation/loader stream.  Dropout is
        # disabled in the V33 causal configs, so resetting here gives all arms
        # the same remaining stochastic contract without coupling parameters.
        runtime_seed = FIXED_SEED * 1_000_003 + int(init_iteration)
        seed_everything(runtime_seed)
    local_start = 0
    global_start = init_iteration
    resumed_from = ""
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        resumed_iteration = int(
            load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                strict=True,
                restore_rng_state=True,
            )
        )
        local_start = resumed_iteration - init_iteration
        global_start = resumed_iteration
        if not 0 < local_start < component_steps:
            raise ValueError("component resume iteration is outside gate")
        resumed_from = str(resume_path)

    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=global_start
    )
    if len(loader.dataset) != int(train_population["expected_nonempty_rows"]):
        raise ValueError("V25 component loader altered official train.txt")
    iterator = iter(loader)
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    amp_name = str(cfg["training"].get("amp_dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    if local_start == 0:
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
            raise RuntimeError("V25 component Gate 0 failed: " + json.dumps(gate_zero))
        iterator = iter(loader)
    else:
        gate_zero = {"passed": True, "reused_from_resume": resumed_from}

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    metrics = metrics_path.open("a" if local_start else "w", encoding="utf-8")
    clip_norm = float(cfg["training"].get("clip_grad_norm", 50.0))
    opt_cfg = cfg["v25"]["optimizer"]
    started = time.perf_counter()
    run_images = 0
    final_diagnostics: dict[str, float] = {}
    resume_checkpoint = output_dir / "resume_latest.pt"
    model.train()
    for local_step in range(local_start + 1, component_steps + 1):
        global_step = init_iteration + local_step
        ratio = _set_learning_rate(
            optimizer,
            step=local_step,
            total_steps=component_steps,
            warmup_steps=min(int(opt_cfg["warmup_steps"]), max(component_steps // 10, 1)),
            minimum_ratio=float(opt_cfg["minimum_lr_ratio"]),
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, torch.Tensor] = {}
        for micro_index in range(accumulation):
            (images, targets, _metas), iterator = _next_batch(loader, iterator)
            run_images += int(images.shape[0])
            images = _move_images(
                images, device=device, channels_last=channels_last
            )
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
            _assert_finite(scaled, f"non-finite V25 component loss at {local_step}")
            scaled.backward()
            for name, value in diagnostics.items():
                accumulated[name] = accumulated.get(name, 0.0) + value.detach() / float(accumulation)
            if denoising_enabled:
                ordered = build_ordered_lane_targets(
                    targets,
                    device=device,
                    slots=model.detector.num_slots,
                    rows=model.detector.num_rows,
                    input_w=int(cfg["model"]["input_w"]),
                    minimum_valid_rows=weights.minimum_valid_rows,
                )
                denoising_seed = (
                    FIXED_SEED * 1_000_003
                    + global_step * 97
                    + micro_index
                )
                anchors = build_denoising_query_anchors(
                    ordered,
                    input_w=int(cfg["model"]["input_w"]),
                    seed=denoising_seed,
                    held_out=False,
                )
                cuda_devices = []
                if device.type == "cuda":
                    cuda_devices = [
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    ]
                # The auxiliary stochastic forward must not alter the RNG
                # sequence seen by the next clean paired-control minibatch.
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(denoising_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(denoising_seed)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        denoising_output = model(
                            images,
                            query_anchor_x_rows=anchors.anchors_normalized,
                        )
                        denoising_loss, denoising_diagnostics = (
                            v25_lane_object_loss(
                                denoising_output,
                                targets,
                                input_w=int(cfg["model"]["input_w"]),
                                weights=weights,
                            )
                        )
                        scaled_denoising = (
                            denoising_weight
                            * denoising_loss
                            / float(accumulation)
                        )
                    _assert_finite(
                        scaled_denoising,
                        f"non-finite V25 denoising loss at {local_step}",
                    )
                    scaled_denoising.backward()
                accumulated["loss_denoising"] = (
                    accumulated.get("loss_denoising", 0.0)
                    + denoising_loss.detach() / float(accumulation)
                )
                accumulated["denoising_anchor_mean_abs_px"] = (
                    accumulated.get("denoising_anchor_mean_abs_px", 0.0)
                    + anchors.mean_absolute_perturbation_px.detach()
                    / float(accumulation)
                )
                accumulated["denoising_mean_soft_iou"] = (
                    accumulated.get("denoising_mean_soft_iou", 0.0)
                    + denoising_diagnostics["mean_soft_iou"].detach()
                    / float(accumulation)
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_norm
        )
        _assert_finite(gradient_norm, "non-finite component gradient")
        optimizer.step()
        if (
            local_step == 1
            or local_step % int(args.log_interval) == 0
            or local_step == component_steps
        ):
            host = _to_host(accumulated, extra={"gradient_norm": gradient_norm})
            gradient_value = host.pop("gradient_norm")
            final_diagnostics = dict(host)
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            row = {
                "component": args.component_name,
                "local_step": local_step,
                "global_step": global_step,
                "run_images_seen": run_images,
                "images_per_second": run_images / elapsed,
                "gradient_norm_pre_clip": gradient_value,
                "gradient_was_clipped": gradient_value > clip_norm,
                "learning_rate_ratio": ratio,
                **host,
            }
            text = json.dumps(row, sort_keys=True)
            print(text, flush=True)
            metrics.write(text + "\n")
            metrics.flush()
        if (
            args.mode == "gate"
            and args.resume_interval > 0
            and local_step < component_steps
            and local_step % args.resume_interval == 0
        ):
            save_checkpoint(
                resume_checkpoint,
                model,
                optimizer=optimizer,
                iteration=global_step,
                cfg=cfg,
                include_rng_state=True,
            )
    metrics.close()
    endpoint = output_dir / (
        "component_endpoint.pt" if args.mode == "gate" else "component_smoke.pt"
    )
    final_iteration = init_iteration + component_steps
    save_checkpoint(endpoint, model, iteration=final_iteration, cfg=cfg)
    elapsed = time.perf_counter() - started
    report = {
        "experiment": "V25 paired component continuation",
        "component": args.component_name,
        "scientific_gate": args.mode == "gate",
        "initial_checkpoint": str(init_checkpoint),
        "initial_checkpoint_sha256": sha256_file(init_checkpoint),
        "advanced_partial_init": bool(args.allow_advanced_init),
        "runtime_rng_reset_after_init": bool(
            args.reset_runtime_rng_after_init
        ),
        "runtime_seed": int(runtime_seed),
        "initial_iteration": init_iteration,
        "component_steps": component_steps,
        "final_iteration": final_iteration,
        "component_official_epoch_fraction": float(component_steps * effective_batch)
        / float(train_population["expected_nonempty_rows"]),
        "endpoint": str(endpoint),
        "endpoint_sha256": sha256_file(endpoint),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "model_contract": v25_model_contract(model.detector),
        "denoising_contract": {
            "enabled": denoising_enabled,
            "loss_weight": denoising_weight,
            "training_shifts_px": [8, 16, 32, 64],
            "row_dropout_probability": 0.15,
            "occlusion_rows": 14,
            "inference_queries_present": False,
        },
        "gate_zero": gate_zero,
        "official_train_population_contract": train_population,
        "official_val_population_contract": val_population,
        "final_training_diagnostics": final_diagnostics,
        "images_per_second": run_images / max(elapsed, 1.0e-9),
        "maximum_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "resumed_from": resumed_from,
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
