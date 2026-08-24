from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from dynlaneseq_eg.engine.checkpoint import load_checkpoint, save_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader
from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import (
    V25LossWeights,
    v25_lane_object_loss,
    v25_model_contract,
)
from dynlaneseq_eg.tools.audit_v33_primary_aux_gradient_interaction import (
    _load_model,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v25_image_mediated_lane_objects import (
    _assert_finite,
    _configure_runtime,
    _configured,
    _loss_weights,
    _move_images,
    _next_batch,
    _set_learning_rate,
    _to_host,
)
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


FIXED_SEED = 3407
PRIVATE_PREFIX = "detector.proposal_memory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain only the V33 private proposal-memory head while keeping "
            "the causal parent shared/primary state bit-exact."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--expected-parent-iteration", type=int, default=11_110)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--mode", choices=("gate", "smoke"), default="gate")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--private-steps", type=int, default=0)
    parser.add_argument("--data-start-iteration", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-interval", type=int, default=250)
    return parser.parse_args()


def auxiliary_only_weights(weights: V25LossWeights) -> V25LossWeights:
    if float(weights.proposal_coverage) <= 0.0:
        raise ValueError("private-head pretraining requires proposal_coverage > 0")
    return replace(
        weights,
        existence=0.0,
        row_distribution=0.0,
        point=0.0,
        strip_iou=0.0,
        range=0.0,
        quality50=0.0,
        quality75=0.0,
        smoothness=0.0,
        order=0.0,
        duplicate=0.0,
        visibility=0.0,
        proposal_coverage=1.0,
        tail_emphasis=0.0,
    )


def private_parameter_partition(
    model: torch.nn.Module,
    *,
    prefix: str = PRIVATE_PREFIX,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    private: list[tuple[str, torch.nn.Parameter]] = []
    frozen: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        is_private = name.startswith(prefix)
        parameter.requires_grad_(is_private)
        (private if is_private else frozen).append((name, parameter))
    if not private:
        raise ValueError(f"private prefix matched no parameters: {prefix}")
    return private, frozen


def _frozen_state_snapshot(
    model: torch.nn.Module,
    *,
    prefix: str = PRIVATE_PREFIX,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(prefix)
    }


def _frozen_state_mismatches(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
) -> list[str]:
    current = model.state_dict()
    return [
        name
        for name, value in reference.items()
        if name not in current or not torch.equal(current[name].detach().cpu(), value)
    ]


def _private_gradient_contract(
    private: list[tuple[str, torch.nn.Parameter]],
    frozen: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    private_nonzero = [
        name
        for name, parameter in private
        if parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        and float(parameter.grad.detach().abs().sum().item()) > 0.0
    ]
    frozen_with_grad = [
        name for name, parameter in frozen if parameter.grad is not None
    ]
    return {
        "private_parameter_tensors": len(private),
        "private_nonzero_gradient_tensors": len(private_nonzero),
        "frozen_parameter_tensors": len(frozen),
        "frozen_tensors_with_gradient": frozen_with_grad,
        "passed": bool(private_nonzero) and not frozen_with_grad,
    }


def main() -> None:
    args = parse_args()
    seed_everything(FIXED_SEED)
    root = Path(args.dataset_root).expanduser().resolve()
    train_population = official_v23_culane_list_contract(root, split="train")
    val_population = official_v23_culane_list_contract(root, split="val")
    configured_args = argparse.Namespace(
        config=args.config,
        dataset_root=str(root),
        num_workers=int(args.num_workers),
    )
    cfg = _configured(
        configured_args,
        official_train_list=str(train_population["list_path"]),
        official_val_list=str(val_population["list_path"]),
    )
    if not bool(cfg.get("v25", {}).get("enable_dual_energy_multi_path", False)):
        raise ValueError("V33 private pretraining requires the auxiliary graph")
    if bool(cfg.get("v25", {}).get("enable_proposal_fusion", True)):
        raise ValueError("V33 private pretraining requires training-only proposals")
    if int(cfg["training"].get("seed", -1)) != FIXED_SEED:
        raise ValueError("V33 private pretraining uses fixed seed 3407")

    physical_batch = int(cfg["training"]["batch_size"])
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 1))
    effective_batch = physical_batch * accumulation
    quarter_steps = math.ceil(
        math.ceil(int(train_population["expected_nonempty_rows"]) / effective_batch)
        / 4.0
    )
    if args.mode == "smoke":
        steps = int(args.smoke_steps)
        if not 1 <= steps <= 10:
            raise ValueError("private-head smoke is limited to 1..10 steps")
    else:
        steps = int(args.private_steps) if int(args.private_steps) > 0 else quarter_steps
    if steps < 1:
        raise ValueError("private pretraining requires at least one step")

    device = torch.device(args.device)
    _configure_runtime(cfg, device)
    channels_last = bool(cfg["training"].get("channels_last", False))
    parent = Path(args.parent_checkpoint).expanduser().resolve()
    model, parent_iteration, initialization = _load_model(
        cfg,
        parent,
        advanced_partial_init=True,
        expected_iteration=int(args.expected_parent_iteration),
        device=device,
        channels_last=channels_last,
    )
    private, frozen = private_parameter_partition(model)
    frozen_reference = _frozen_state_snapshot(model)
    raw_optimizer = cfg["v25"]["optimizer"]
    private_parameters = [parameter for _name, parameter in private]
    optimizer = torch.optim.AdamW(
        (
            {
                "params": private_parameters,
                "lr": float(raw_optimizer["detector_lr"]),
                "initial_lr": float(raw_optimizer["detector_lr"]),
                "name": "private_proposal_memory",
            },
        ),
        betas=(0.9, 0.999),
        weight_decay=float(raw_optimizer["weight_decay"]),
    )

    runtime_seed = FIXED_SEED * 1_000_003 + parent_iteration + 33
    seed_everything(runtime_seed)
    local_start = 0
    resumed_from = ""
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        local_start = int(
            load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                strict=True,
                restore_rng_state=True,
            )
        )
        if not 0 < local_start < steps:
            raise ValueError("private-head resume step is outside the gate")
        resumed_from = str(resume_path)
        resume_mismatches = _frozen_state_mismatches(model, frozen_reference)
        if resume_mismatches:
            raise ValueError(
                "private-head resume changed frozen parent state: "
                + json.dumps(resume_mismatches[:10])
            )

    loader = build_dataloader(
        cfg,
        split="train",
        training=True,
        start_iteration=int(args.data_start_iteration) + local_start,
    )
    if len(loader.dataset) != int(train_population["expected_nonempty_rows"]):
        raise ValueError("private-head loader altered official train.txt")
    iterator = iter(loader)
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    amp_name = str(cfg["training"].get("amp_dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    weights = auxiliary_only_weights(_loss_weights(cfg))

    first_batch, iterator = _next_batch(loader, iterator)
    images, targets, _metas = first_batch
    images = _move_images(images, device=device, channels_last=channels_last)
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        output = model(images)
        gate_loss, _gate_diagnostics = v25_lane_object_loss(
            output,
            targets,
            input_w=int(cfg["model"]["input_w"]),
            weights=weights,
        )
    _assert_finite(gate_loss, "non-finite private-head Gate 0 loss")
    gate_loss.backward()
    gradient_contract = _private_gradient_contract(private, frozen)
    if not gradient_contract["passed"]:
        raise RuntimeError(
            "private-head gradient ownership failed: "
            + json.dumps(gradient_contract)
        )
    model.zero_grad(set_to_none=True)
    del output, images, gate_loss
    iterator = iter(loader)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "private_train_metrics.jsonl"
    metrics_file = metrics_path.open("a" if local_start else "w", encoding="utf-8")
    resume_path = output_dir / "private_resume_latest.pt"
    clip_norm = float(cfg["training"].get("clip_grad_norm", 50.0))
    started = time.perf_counter()
    run_images = 0
    final_diagnostics: dict[str, float] = {}
    model.train()
    for local_step in range(local_start + 1, steps + 1):
        ratio = _set_learning_rate(
            optimizer,
            step=local_step,
            total_steps=steps,
            warmup_steps=min(
                int(raw_optimizer["warmup_steps"]), max(steps // 10, 1)
            ),
            minimum_ratio=float(raw_optimizer["minimum_lr_ratio"]),
        )
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, torch.Tensor] = {}
        for _micro_index in range(accumulation):
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
            _assert_finite(scaled, f"non-finite private loss at {local_step}")
            scaled.backward()
            for name, value in diagnostics.items():
                accumulated[name] = (
                    accumulated.get(name, 0.0)
                    + value.detach() / float(accumulation)
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            private_parameters, max_norm=clip_norm
        )
        _assert_finite(gradient_norm, "non-finite private-head gradient")
        optimizer.step()
        if (
            local_step == 1
            or local_step % int(args.log_interval) == 0
            or local_step == steps
        ):
            host = _to_host(accumulated, extra={"gradient_norm": gradient_norm})
            gradient_value = host.pop("gradient_norm")
            final_diagnostics = dict(host)
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            row = {
                "phase": "v33_private_head_pretraining",
                "local_step": local_step,
                "shared_updates": 0,
                "run_images_seen": run_images,
                "images_per_second": run_images / elapsed,
                "gradient_norm_pre_clip": gradient_value,
                "gradient_was_clipped": gradient_value > clip_norm,
                "learning_rate_ratio": ratio,
                **host,
            }
            text = json.dumps(row, sort_keys=True)
            print(text, flush=True)
            metrics_file.write(text + "\n")
            metrics_file.flush()
        if (
            args.mode == "gate"
            and args.resume_interval > 0
            and local_step < steps
            and local_step % int(args.resume_interval) == 0
        ):
            save_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                iteration=local_step,
                cfg=cfg,
                include_rng_state=True,
            )
    metrics_file.close()

    frozen_mismatches = _frozen_state_mismatches(model, frozen_reference)
    if frozen_mismatches:
        raise RuntimeError(
            "private-head training changed frozen parent tensors: "
            + json.dumps(frozen_mismatches[:10])
        )
    endpoint = output_dir / (
        "private_head_endpoint.pt"
        if args.mode == "gate"
        else "private_head_smoke.pt"
    )
    # Head-only updates are not shared-model iterations. Retaining the parent
    # iteration lets the exact paired main continuation start at 11,110.
    save_checkpoint(endpoint, model, iteration=parent_iteration, cfg=cfg)
    elapsed = time.perf_counter() - started
    report = {
        "experiment": "V33-PH private auxiliary cold-start pretraining",
        "scientific_gate": args.mode == "gate",
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": sha256_file(parent),
        "parent_iteration": parent_iteration,
        "initialization": initialization,
        "private_prefix": PRIVATE_PREFIX,
        "private_parameter_tensors": len(private),
        "private_parameter_count": sum(
            int(parameter.numel()) for _name, parameter in private
        ),
        "private_steps": steps,
        "shared_updates": 0,
        "data_start_iteration": int(args.data_start_iteration),
        "runtime_seed": runtime_seed,
        "gradient_ownership_gate": gradient_contract,
        "frozen_parent_state_exact": True,
        "endpoint": str(endpoint),
        "endpoint_iteration_retains_parent": parent_iteration,
        "endpoint_sha256": sha256_file(endpoint),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "model_contract": v25_model_contract(model.detector),
        "official_train_population_contract": train_population,
        "official_val_population_contract": val_population,
        "final_training_diagnostics": final_diagnostics,
        "images_per_second": run_images / max(elapsed, 1.0e-9),
        "maximum_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "resumed_from": resumed_from,
        "checkpoint_selection_performed": False,
        "threshold_selection_performed": False,
        "test_set_used": False,
    }
    (output_dir / "private_training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
