from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import build_criterion, build_dataloader, build_matcher, build_model, build_optimizer
from dynlaneseq_eg.modeling.common import nested_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break down DynLaneSeq training throughput bottlenecks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "synthetic-model", "synthetic-train", "dataloader", "loader-h2d", "real-train"],
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=0)
    parser.add_argument("--persistent-workers", type=int, choices=[0, 1], default=-1)
    parser.add_argument("--synthetic-lanes", type=int, default=4)
    parser.add_argument("--synthetic-seg", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--clip-grad-norm", type=float, default=-1.0)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_runtime(cfg: dict[str, Any], device: torch.device, no_amp: bool) -> bool:
    train_cfg = cfg.get("training", {})
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", False))
        if bool(train_cfg.get("tf32", False)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    return bool(train_cfg.get("amp", False) and device.type == "cuda" and not no_amp)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("training", {})
    cfg.setdefault("dataloader", {})
    if args.batch_size > 0:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.num_workers >= 0:
        cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if args.prefetch_factor > 0:
        cfg["dataloader"]["prefetch_factor"] = int(args.prefetch_factor)
    if args.persistent_workers >= 0:
        cfg["dataloader"]["persistent_workers"] = bool(args.persistent_workers)
    return cfg


def make_synthetic_targets(
    batch_size: int,
    lanes_per_image: int,
    num_rows: int,
    input_h: int,
    input_w: int,
    x_bins: int,
    device: torch.device,
    dtype: torch.dtype,
    include_seg: bool,
) -> list[dict[str, torch.Tensor]]:
    y = torch.linspace(0.0, 1.0, num_rows, device=device, dtype=dtype).view(1, num_rows)
    base_positions = torch.linspace(0.22, 0.78, lanes_per_image, device=device, dtype=dtype).view(lanes_per_image, 1)
    slope = torch.linspace(-0.08, 0.08, lanes_per_image, device=device, dtype=dtype).view(lanes_per_image, 1)
    curve = torch.linspace(0.04, -0.04, lanes_per_image, device=device, dtype=dtype).view(lanes_per_image, 1)
    row_shape = base_positions + slope * (y - 0.5) + curve * (y - 0.5).pow(2)
    row_shape = row_shape.clamp(0.02, 0.98) * float(input_w - 1)
    valid_mask = torch.ones((lanes_per_image, num_rows), device=device, dtype=torch.bool)
    range_y = torch.tensor([[0.0, float(input_h - 1)]], device=device, dtype=dtype).expand(lanes_per_image, 2).clone()
    x_bin_tensor = (row_shape / (float(input_w) / float(x_bins))).long().clamp(0, x_bins - 1)
    targets: list[dict[str, torch.Tensor]] = []
    for _ in range(batch_size):
        target = {
            "x_rows": row_shape.clone(),
            "valid_mask": valid_mask.clone(),
            "range_y": range_y.clone(),
            "x_bins": x_bin_tensor.clone(),
        }
        if include_seg:
            target["seg_mask"] = make_seg_mask(row_shape, valid_mask, input_h, input_w, device, dtype)
            target["seg_valid"] = torch.tensor(True, device=device)
        targets.append(target)
    return targets


def make_seg_mask(
    x_rows: torch.Tensor,
    valid_mask: torch.Tensor,
    input_h: int,
    input_w: int,
    device: torch.device,
    dtype: torch.dtype,
    half_width: int = 4,
) -> torch.Tensor:
    mask = torch.zeros((1, input_h, input_w), device=device, dtype=dtype)
    ys = torch.linspace(0, input_h - 1, x_rows.shape[1], device=device).round().long()
    for lane_idx in range(x_rows.shape[0]):
        xs = x_rows[lane_idx].round().long().clamp(0, input_w - 1)
        valid = valid_mask[lane_idx]
        for dx in range(-half_width, half_width + 1):
            xx = (xs + dx).clamp(0, input_w - 1)
            mask[0, ys[valid], xx[valid]] = 1.0
    return mask


def clone_targets_to_cpu(targets: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in targets]


def model_forward_no_match(model: torch.nn.Module, images: torch.Tensor, cfg: dict[str, Any]) -> dict[str, Any]:
    name = cfg.get("model", {}).get("name", "DynLaneSeqS0")
    if name in {"DynLaneSeqS2", "DynLaneSeqS3"}:
        return model(images, sampler_alpha=0.0)
    if name == "DynLaneSeqS4":
        return model(images, sampler_alpha=0.0, sampler_beta=0.0)
    return model(images)


def output_anchor_loss(outputs: Any) -> torch.Tensor:
    terms: list[torch.Tensor] = []

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            terms.append(value.float().mean())
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(outputs)
    if not terms:
        raise RuntimeError("No floating point tensor found in model outputs.")
    return sum(terms) / float(len(terms))


def optimizer_step(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    clip_norm: float,
) -> None:
    if amp_enabled:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_norm > 0:
            clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        return
    loss.backward()
    if clip_norm > 0:
        clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()


def infinite_loader(loader: Iterable[Any]) -> Iterable[Any]:
    while True:
        for batch in loader:
            yield batch


def report(name: str, steps: int, batch_size: int, elapsed: float, device: torch.device) -> None:
    img_per_sec = float(steps * batch_size) / max(elapsed, 1e-9)
    sec_per_iter = elapsed / max(steps, 1)
    mem = ""
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024.0**3)
        mem = f" | peak_mem={peak_gb:.2f} GB"
    print(f"{name}: {steps} steps | batch={batch_size} | {sec_per_iter:.4f}s/it | {img_per_sec:.1f} img/s{mem}")


def run_synthetic_model(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    args: argparse.Namespace,
) -> None:
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    model_cfg = cfg.get("model", {})
    images = torch.randn(
        batch_size,
        3,
        int(model_cfg.get("input_h", 288)),
        int(model_cfg.get("input_w", 800)),
        device=device,
    )
    clip_norm = args.clip_grad_norm if args.clip_grad_norm >= 0 else float(cfg.get("training", {}).get("clip_grad_norm", 1.0))
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    sync(device)
    start = None
    for step in range(args.warmup + args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model_forward_no_match(model, images, cfg)
            loss = output_anchor_loss(outputs)
        optimizer_step(loss, model, optimizer, scaler, amp_enabled, clip_norm)
        if step == args.warmup - 1:
            sync(device)
            start = time.perf_counter()
    sync(device)
    report("synthetic-model", args.steps, batch_size, time.perf_counter() - (start or time.perf_counter()), device)


def run_synthetic_train(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    matcher,
    criterion,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    args: argparse.Namespace,
    h2d: bool = False,
) -> None:
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    model_cfg = cfg.get("model", {})
    input_h = int(model_cfg.get("input_h", 288))
    input_w = int(model_cfg.get("input_w", 800))
    num_rows = int(model_cfg.get("num_rows", 72))
    x_bins = int(model_cfg.get("x_bins", 200))
    tensor_device = torch.device("cpu") if h2d else device
    images = torch.randn(batch_size, 3, input_h, input_w, device=tensor_device)
    targets = make_synthetic_targets(
        batch_size=batch_size,
        lanes_per_image=int(args.synthetic_lanes),
        num_rows=num_rows,
        input_h=input_h,
        input_w=input_w,
        x_bins=x_bins,
        device=tensor_device,
        dtype=torch.float32,
        include_seg=bool(args.synthetic_seg),
    )
    if h2d:
        targets = clone_targets_to_cpu(targets)
    clip_norm = args.clip_grad_norm if args.clip_grad_norm >= 0 else float(cfg.get("training", {}).get("clip_grad_norm", 1.0))
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    sync(device)
    start = None
    for step in range(args.warmup + args.steps):
        step_images = images.to(device, non_blocking=True) if h2d else images
        step_targets = nested_to_device(targets, device) if h2d else targets
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs, matches = forward_with_matches(model, step_images, step_targets, matcher, cfg, step)
            loss_dict = criterion(outputs, step_targets, matches)
            loss = loss_dict["loss_total"]
        optimizer_step(loss, model, optimizer, scaler, amp_enabled, clip_norm)
        if step == args.warmup - 1:
            sync(device)
            start = time.perf_counter()
    sync(device)
    name = "synthetic-train-h2d" if h2d else "synthetic-train-gpu"
    report(name, args.steps, batch_size, time.perf_counter() - (start or time.perf_counter()), device)


def run_dataloader(cfg: dict[str, Any], device: torch.device, args: argparse.Namespace, h2d: bool) -> None:
    loader = build_dataloader(cfg, split="train", training=True)
    iterator = infinite_loader(loader)
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    sync(device)
    start = None
    for step in range(args.warmup + args.steps):
        images, targets, _ = next(iterator)
        if h2d:
            images = images.to(device, non_blocking=True)
            targets = nested_to_device(targets, device)
            sync(device)
        if step == args.warmup - 1:
            start = time.perf_counter()
    report("loader-h2d" if h2d else "dataloader", args.steps, batch_size, time.perf_counter() - (start or time.perf_counter()), device)


def run_real_train(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    matcher,
    criterion,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    args: argparse.Namespace,
) -> None:
    loader = build_dataloader(cfg, split="train", training=True)
    iterator = infinite_loader(loader)
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    clip_norm = args.clip_grad_norm if args.clip_grad_norm >= 0 else float(cfg.get("training", {}).get("clip_grad_norm", 1.0))
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    sync(device)
    start = None
    for step in range(args.warmup + args.steps):
        images, targets, _ = next(iterator)
        images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs, matches = forward_with_matches(model, images, targets, matcher, cfg, step)
            loss_dict = criterion(outputs, targets, matches)
            loss = loss_dict["loss_total"]
        optimizer_step(loss, model, optimizer, scaler, amp_enabled, clip_norm)
        if step == args.warmup - 1:
            sync(device)
            start = time.perf_counter()
    sync(device)
    report("real-train", args.steps, batch_size, time.perf_counter() - (start or time.perf_counter()), device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    cfg = apply_overrides(load_config(args.config), args)
    device = torch.device(args.device)
    amp_enabled = configure_runtime(cfg, device, args.no_amp)
    model = build_model(cfg).to(device).train()
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    print(
        {
            "mode": args.mode,
            "model": cfg.get("model", {}).get("name"),
            "device": str(device),
            "batch_size": int(cfg.get("training", {}).get("batch_size", 1)),
            "amp": amp_enabled,
            "num_workers": int(cfg.get("dataloader", {}).get("num_workers", 0)),
            "prefetch_factor": cfg.get("dataloader", {}).get("prefetch_factor"),
            "persistent_workers": cfg.get("dataloader", {}).get("persistent_workers"),
            "steps": args.steps,
            "warmup": args.warmup,
        }
    )
    modes = [args.mode] if args.mode != "all" else ["synthetic-model", "synthetic-train", "loader-h2d", "real-train"]
    for mode in modes:
        if mode == "synthetic-model":
            run_synthetic_model(cfg, model, optimizer, scaler, device, amp_enabled, args)
        elif mode == "synthetic-train":
            run_synthetic_train(cfg, model, matcher, criterion, optimizer, scaler, device, amp_enabled, args, h2d=False)
        elif mode == "dataloader":
            run_dataloader(cfg, device, args, h2d=False)
        elif mode == "loader-h2d":
            run_dataloader(cfg, device, args, h2d=True)
        elif mode == "real-train":
            run_real_train(cfg, model, matcher, criterion, optimizer, scaler, device, amp_enabled, args)
        else:
            raise ValueError(f"Unhandled mode: {mode}")


if __name__ == "__main__":
    main()
