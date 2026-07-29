from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
    build_optimizer,
)
from dynlaneseq_eg.modeling.common import nested_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one real LaneRowNet optimizer step without saving or changing a run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--breakdown-steps", type=int, default=3)
    parser.add_argument("--profiler-steps", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=25)
    parser.add_argument("--trace", default="")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_stage(
    totals: dict[str, float],
    name: str,
    start: float,
    device: torch.device,
) -> None:
    sync(device)
    totals[name] += time.perf_counter() - start


def amp_settings(cfg: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype | None, bool]:
    training = cfg.get("training", {})
    enabled = bool(training.get("amp", False) and device.type == "cuda")
    name = str(training.get("amp_dtype", "")).strip().lower()
    dtype = None
    if name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif name in {"fp16", "float16", "half"}:
        dtype = torch.float16
    use_scaler = bool(enabled and dtype != torch.bfloat16)
    return enabled, dtype, use_scaler


def autocast_kwargs(
    device: torch.device,
    enabled: bool,
    dtype: torch.dtype | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"device_type": device.type, "enabled": enabled}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def move_batch(
    batch: tuple[torch.Tensor, list[dict[str, torch.Tensor]], object],
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, targets, _ = batch
    if channels_last:
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    else:
        images = images.to(device, non_blocking=True)
    return images, nested_to_device(targets, device)


def train_micro_batch(
    *,
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matcher: object,
    criterion: torch.nn.Module,
    cfg: dict[str, Any],
    iteration: int,
    accumulation_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
    totals: dict[str, float] | None,
    device: torch.device,
) -> None:
    start = time.perf_counter()
    with torch.autocast(**autocast_kwargs(device, amp_enabled, amp_dtype)):
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            iteration,
        )
    if totals is not None:
        elapsed_stage(totals, "forward_and_final_match", start, device)

    start = time.perf_counter()
    with torch.autocast(**autocast_kwargs(device, amp_enabled, amp_dtype)):
        if hasattr(criterion, "set_iteration"):
            criterion.set_iteration(iteration)
        losses = criterion(outputs, targets, matches)
        loss = losses["loss_total"] / float(accumulation_steps)
    if totals is not None:
        elapsed_stage(totals, "loss_and_aux_matches", start, device)

    start = time.perf_counter()
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    if totals is not None:
        elapsed_stage(totals, "backward", start, device)


def optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    clip_norm: float,
) -> None:
    if scaler is not None:
        scaler.unscale_(optimizer)
    if clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def run_optimizer_steps(
    *,
    count: int,
    iterator: Any,
    model: torch.nn.Module,
    matcher: object,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    device: torch.device,
    channels_last: bool,
    accumulation_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
    clip_norm: float,
    totals: dict[str, float] | None = None,
) -> None:
    for iteration in range(count):
        for _ in range(accumulation_steps):
            start = time.perf_counter()
            batch = next(iterator)
            if totals is not None:
                totals["data_wait"] += time.perf_counter() - start

            start = time.perf_counter()
            images, targets = move_batch(batch, device, channels_last)
            if totals is not None:
                elapsed_stage(totals, "host_to_device", start, device)

            train_micro_batch(
                model=model,
                images=images,
                targets=targets,
                matcher=matcher,
                criterion=criterion,
                cfg=cfg,
                iteration=iteration,
                accumulation_steps=accumulation_steps,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
                totals=totals,
                device=device,
            )

        start = time.perf_counter()
        optimizer_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            clip_norm=clip_norm,
        )
        if totals is not None:
            elapsed_stage(totals, "clip_and_optimizer", start, device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    if args.batch_size > 0:
        cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    if args.grad_accum > 0:
        cfg.setdefault("training", {})["gradient_accumulation_steps"] = int(args.grad_accum)

    training = cfg.get("training", {})
    seed_everything(int(training.get("seed", 3407)))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU.")
    torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", False))
    if bool(training.get("tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    channels_last = bool(training.get("channels_last", False))
    model = build_model(cfg).to(device).train()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    amp_enabled, amp_dtype, use_scaler = amp_settings(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_scaler else None
    accumulation_steps = max(
        int(training.get("gradient_accumulation_steps", 1)),
        1,
    )
    clip_norm = float(training.get("clip_grad_norm", 1.0))
    loader = build_dataloader(cfg, split="train", training=True)

    def batches() -> Any:
        while True:
            yield from loader

    iterator = iter(batches())
    batch_size = int(training.get("batch_size", 1))
    row_reference = (
        cfg.get("model", {})
        .get("structured_query", {})
        .get("row_reference", {})
    )
    print(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "amp_dtype": str(amp_dtype),
            "channels_last": channels_last,
            "sampling_backend": row_reference.get("sampling_backend", "grid_sample"),
            "warmup_steps": args.warmup_steps,
            "breakdown_steps": args.breakdown_steps,
            "profiler_steps": args.profiler_steps,
        }
    )

    optimizer.zero_grad(set_to_none=True)
    run_optimizer_steps(
        count=max(args.warmup_steps, 0),
        iterator=iterator,
        model=model,
        matcher=matcher,
        criterion=criterion,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        channels_last=channels_last,
        accumulation_steps=accumulation_steps,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
        clip_norm=clip_norm,
    )

    totals: dict[str, float] = defaultdict(float)
    torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    wall_start = time.perf_counter()
    run_optimizer_steps(
        count=max(args.breakdown_steps, 1),
        iterator=iterator,
        model=model,
        matcher=matcher,
        criterion=criterion,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        channels_last=channels_last,
        accumulation_steps=accumulation_steps,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
        clip_norm=clip_norm,
        totals=totals,
    )
    sync(device)
    wall_elapsed = time.perf_counter() - wall_start
    accounted = sum(totals.values())
    print("\nSYNCHRONIZED STAGE BREAKDOWN")
    for name, value in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        print(
            f"{name:26s} {value / max(args.breakdown_steps, 1):8.4f} s/optimizer-step "
            f"({100.0 * value / max(accounted, 1e-9):5.1f}%)"
        )
    effective_images = (
        max(args.breakdown_steps, 1)
        * batch_size
        * accumulation_steps
    )
    print(
        f"{'synchronized_total':26s} "
        f"{wall_elapsed / max(args.breakdown_steps, 1):8.4f} s/optimizer-step "
        f"| {effective_images / max(wall_elapsed, 1e-9):.2f} img/s "
        f"| peak={torch.cuda.max_memory_allocated(device) / (1024.0 ** 3):.2f} GiB"
    )

    if args.profiler_steps <= 0:
        return
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        run_optimizer_steps(
            count=args.profiler_steps,
            iterator=iterator,
            model=model,
            matcher=matcher,
            criterion=criterion,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            channels_last=channels_last,
            accumulation_steps=accumulation_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            clip_norm=clip_norm,
        )
        sync(device)
    print("\nTOP CUDA OPERATORS")
    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=max(args.row_limit, 1),
        )
    )
    print("\nTOP CPU OPERATORS")
    print(
        profiler.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=max(args.row_limit, 1),
        )
    )
    if args.trace:
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
        print(f"trace: {trace_path}")


if __name__ == "__main__":
    main()
