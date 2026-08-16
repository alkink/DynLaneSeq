from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch

from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v23 import DynLaneSeqV23
from dynlaneseq_eg.modeling.v23_ordered_slot_cost_volume import (
    build_v23_owned_targets,
    soft_viterbi_marginals,
    v23_ordered_cost_volume_loss,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v23_ordered_slot_cost_volume import (
    FIXED_SEED,
    _configure_cuda_runtime,
    _configured,
    _loss_weights,
    _move_images,
    _next_batch,
    _optimizer,
)
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one real V23 training step")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--profiler-rows", type=int, default=30)
    parser.add_argument("--loop-steps", type=int, default=20)
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function: Callable[[], Any]) -> tuple[Any, float]:
    _synchronize(device)
    start = time.perf_counter()
    value = function()
    _synchronize(device)
    return value, 1_000.0 * (time.perf_counter() - start)


def _teacher_forward(
    model: DynLaneSeqV23, images: torch.Tensor
) -> dict[str, torch.Tensor]:
    with torch.no_grad(), torch.autocast(
        device_type=images.device.type,
        enabled=False,
    ):
        return model.teacher(images.float(), inference_only=True)


def main() -> None:
    args = parse_args()
    seed_everything(FIXED_SEED)
    device = torch.device(args.device)
    train_population = official_v23_culane_list_contract(
        args.dataset_root, split="train"
    )
    val_population = official_v23_culane_list_contract(
        args.dataset_root, split="val"
    )
    cfg: dict[str, Any] = _configured(
        args,
        official_train_list=str(train_population["list_path"]),
        official_val_list=str(val_population["list_path"]),
    )
    _configure_cuda_runtime(cfg, device)
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV23):
        raise TypeError("profile tool requires DynLaneSeqV23")
    model.to(device)
    channels_last = bool(cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = _optimizer(model, cfg)
    try:
        # Current V23 resume files intentionally exclude the immutable V7
        # teacher.  Keep a legacy fallback so this diagnostic can still
        # profile the pre-optimization step-1000 checkpoint.
        iteration = int(
            load_checkpoint(
                args.resume,
                model.student,
                optimizer=optimizer,
                strict=True,
            )
        )
    except RuntimeError:
        optimizer = _optimizer(model, cfg)
        iteration = int(
            load_checkpoint(args.resume, model, optimizer=optimizer, strict=True)
        )
    model.train()

    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=iteration
    )
    load_start = time.perf_counter()
    images, targets, _metas = next(iter(loader))
    loader_ms = 1_000.0 * (time.perf_counter() - load_start)
    transfer_start = time.perf_counter()
    images = _move_images(
        images, device=device, channels_last=channels_last
    )
    _synchronize(device)
    transfer_ms = 1_000.0 * (time.perf_counter() - transfer_start)
    weights = _loss_weights(cfg)
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"

    # Warm up convolution selection and allocator state before reporting.
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            warm_output = model(images)
            warm_loss, _ = v23_ordered_cost_volume_loss(
                warm_output,
                targets,
                input_h=int(cfg["model"]["input_h"]),
                input_w=int(cfg["model"]["input_w"]),
                weights=weights,
            )
        warm_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.student.parameters(),
            max_norm=float(cfg["training"]["clip_grad_norm"]),
        )
        optimizer.step()
        del warm_output, warm_loss
    optimizer.zero_grad(set_to_none=True)
    _synchronize(device)

    teacher_output, teacher_ms = _timed(
        device, lambda: _teacher_forward(model, images)
    )

    def student_forward():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            return model.student(images, teacher_output)

    output, student_ms = _timed(device, student_forward)

    def target_build():
        return build_v23_owned_targets(
            targets,
            source_x=output["source_x_rows"],
            source_range=output["source_range_norm"],
            source_active=output["source_active"],
            input_h=int(cfg["model"]["input_h"]),
            input_w=int(cfg["model"]["input_w"]),
            minimum_valid_rows=weights.minimum_valid_rows,
        )

    _owned, target_build_ms = _timed(device, target_build)

    def loss_forward():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            return v23_ordered_cost_volume_loss(
                output,
                targets,
                input_h=int(cfg["model"]["input_h"]),
                input_w=int(cfg["model"]["input_w"]),
                weights=weights,
            )

    (loss, _diagnostics), loss_ms = _timed(device, loss_forward)
    _unused, backward_ms = _timed(device, lambda: loss.backward())
    _gradient_norm, gradient_clip_ms = _timed(
        device,
        lambda: torch.nn.utils.clip_grad_norm_(
            model.student.parameters(),
            max_norm=float(cfg["training"]["clip_grad_norm"]),
        ),
    )
    _unused, optimizer_step_ms = _timed(device, optimizer.step)
    optimizer.zero_grad(set_to_none=True)

    detached_unary = output["unary_logits"].detach()

    def path_only():
        with torch.no_grad():
            return soft_viterbi_marginals(
                detached_unary,
                transition_radius_bins=model.student.transition_radius_bins,
                transition_penalty=model.student.transition_penalty,
            )

    _path, path_forward_only_ms = _timed(device, path_only)

    # Measure the real steady training loop as well as isolated components.
    # This includes online augmentation, worker IPC, H2D/layout, forward,
    # target construction, backward, clipping and AdamW.
    loop_iterator = iter(loader)
    loop_images = 0
    _synchronize(device)
    loop_start = time.perf_counter()
    for _ in range(max(int(args.loop_steps), 0)):
        (loop_batch, loop_iterator) = _next_batch(loader, loop_iterator)
        loop_input, loop_targets, _loop_metas = loop_batch
        loop_input = _move_images(
            loop_input, device=device, channels_last=channels_last
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            loop_output = model(loop_input)
            loop_loss, _ = v23_ordered_cost_volume_loss(
                loop_output,
                loop_targets,
                input_h=int(cfg["model"]["input_h"]),
                input_w=int(cfg["model"]["input_w"]),
                weights=weights,
            )
        loop_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.student.parameters(),
            max_norm=float(cfg["training"]["clip_grad_norm"]),
        )
        optimizer.step()
        loop_images += int(loop_input.shape[0])
    _synchronize(device)
    loop_seconds = time.perf_counter() - loop_start

    # A userspace torch profile identifies both the many-small-kernel path and
    # host synchronisation without needing unavailable kernel perf/eBPF access.
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            profiled_output = model(images)
            profiled_loss, _ = v23_ordered_cost_volume_loss(
                profiled_output,
                targets,
                input_h=int(cfg["model"]["input_h"]),
                input_w=int(cfg["model"]["input_w"]),
                weights=weights,
            )
        profiled_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.student.parameters(),
            max_norm=float(cfg["training"]["clip_grad_norm"]),
        )
        optimizer.step()
    _synchronize(device)

    report = {
        "iteration": iteration,
        "batch_size": int(images.shape[0]),
        "loader_first_batch_ms": loader_ms,
        "host_to_device_and_layout_ms": transfer_ms,
        "teacher_forward_ms": teacher_ms,
        "student_forward_ms": student_ms,
        "owned_target_build_ms": target_build_ms,
        "complete_loss_forward_ms_including_target_build": loss_ms,
        "backward_ms": backward_ms,
        "gradient_clip_ms": gradient_clip_ms,
        "optimizer_step_ms": optimizer_step_ms,
        "soft_viterbi_forward_only_ms": path_forward_only_ms,
        "measured_compute_step_ms": (
            teacher_ms
            + student_ms
            + loss_ms
            + backward_ms
            + gradient_clip_ms
            + optimizer_step_ms
        ),
        "measured_compute_images_per_second": (
            float(images.shape[0])
            / (
                (
                    teacher_ms
                    + student_ms
                    + loss_ms
                    + backward_ms
                    + gradient_clip_ms
                    + optimizer_step_ms
                )
                / 1_000.0
            )
        ),
        "steady_loop_steps": int(args.loop_steps),
        "steady_loop_seconds": loop_seconds,
        "steady_loop_images_per_second": (
            float(loop_images) / loop_seconds if loop_seconds > 0.0 else 0.0
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(
        profile.key_averages().table(
            sort_by=sort_key, row_limit=int(args.profiler_rows)
        ),
        flush=True,
    )
    if device.type == "cuda":
        print(
            profile.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total",
                row_limit=int(args.profiler_rows),
            ),
            flush=True,
        )
        print(
            profile.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=int(args.profiler_rows)
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
