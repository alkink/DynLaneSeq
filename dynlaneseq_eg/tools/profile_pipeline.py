from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import build_criterion, build_dataloader, build_matcher, build_model, build_optimizer
from dynlaneseq_eg.modeling.common import nested_to_device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": statistics.fmean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "max_ms": max(values) * 1000.0,
    }


def _next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--loss-breakdown", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    train_cfg = cfg.get("training", {})
    if args.channels_last:
        train_cfg["channels_last"] = True
    if args.compile_model:
        train_cfg["compile_model"] = True
        train_cfg["compile_mode"] = args.compile_mode
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", False))
        if bool(train_cfg.get("tf32", False)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    model = build_model(cfg).to(device)
    channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    if bool(train_cfg.get("compile_model", False)):
        compile_kwargs = {}
        if train_cfg.get("compile_backend") is not None:
            compile_kwargs["backend"] = str(train_cfg.get("compile_backend"))
        if train_cfg.get("compile_mode") is not None:
            compile_kwargs["mode"] = str(train_cfg.get("compile_mode"))
        model = torch.compile(model, **compile_kwargs)
    amp = bool(train_cfg.get("amp", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    clip_norm = float(train_cfg.get("clip_grad_norm", 1.0))
    loader = build_dataloader(cfg, split="train", training=True)
    iterator = iter(loader)

    stages: dict[str, list[float]] = {
        "data_next": [],
        "h2d": [],
        "model_forward": [],
        "matcher": [],
        "loss": [],
        "backward_step": [],
        "total_iter": [],
    }
    loss_parts: dict[str, list[float]] = {}
    measured_images = 0
    wall_start = 0.0
    total_iters = max(0, args.warmup) + max(1, args.iters)

    model.train()
    for idx in range(total_iters):
        iter_start = time.perf_counter()

        stage_start = time.perf_counter()
        batch, iterator = _next_batch(loader, iterator)
        data_dt = time.perf_counter() - stage_start
        images, targets, _metas = batch

        stage_start = time.perf_counter()
        if channels_last:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        _sync(device)
        h2d_dt = time.perf_counter() - stage_start

        optimizer.zero_grad(set_to_none=True)
        model_name = cfg.get("model", {}).get("name", "")
        if model_name in {"DynLaneSeqS2", "DynLaneSeqS3", "DynLaneSeqS4"}:
            stage_start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs, matches = forward_with_matches(model, images, targets, matcher, cfg, idx)
            _sync(device)
            forward_dt = time.perf_counter() - stage_start
            matcher_dt = 0.0
        else:
            stage_start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(images)
            _sync(device)
            forward_dt = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            matches = matcher(outputs, targets)
            _sync(device)
            matcher_dt = time.perf_counter() - stage_start

        if args.loss_breakdown:
            timed_parts = [
                ("exist", lambda: criterion.compute_exist_loss(outputs, matches)),
                ("point", lambda: criterion.compute_point_loss(outputs, targets, matches)),
                ("range", lambda: criterion.compute_range_loss(outputs, targets, matches)),
                ("smooth", lambda: criterion.compute_smoothness_loss(outputs, targets, matches)),
                ("line_iou", lambda: criterion.compute_line_iou_loss(outputs, targets, matches)),
                ("seg", lambda: criterion.compute_seg_loss(outputs, targets)),
                ("quality", lambda: criterion.compute_quality_loss(outputs, targets, matches)),
                ("centerline", lambda: criterion.compute_centerline_loss(outputs, targets)),
                ("dynamic_proposal", lambda: sum(criterion.compute_dynamic_proposal_losses(outputs, targets).values())),
            ]
            for name, fn in timed_parts:
                part_start = time.perf_counter()
                with torch.autocast(device_type=device.type, enabled=amp):
                    part_loss = fn()
                    if isinstance(part_loss, torch.Tensor):
                        part_loss.detach()
                _sync(device)
                if idx >= args.warmup:
                    loss_parts.setdefault(name, []).append(time.perf_counter() - part_start)

        stage_start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = criterion(outputs, targets, matches)["loss_total"]
        _sync(device)
        loss_dt = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        _sync(device)
        backward_dt = time.perf_counter() - stage_start

        total_dt = time.perf_counter() - iter_start
        if idx == args.warmup:
            wall_start = time.perf_counter()
        if idx >= args.warmup:
            measured_images += int(images.shape[0])
            stages["data_next"].append(data_dt)
            stages["h2d"].append(h2d_dt)
            stages["model_forward"].append(forward_dt)
            stages["matcher"].append(matcher_dt)
            stages["loss"].append(loss_dt)
            stages["backward_step"].append(backward_dt)
            stages["total_iter"].append(total_dt)

    wall_dt = max(time.perf_counter() - wall_start, 1e-9)
    report: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "amp": amp,
        "tf32": bool(train_cfg.get("tf32", False)),
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "num_workers": int(cfg.get("dataloader", {}).get("num_workers", 0)),
        "pin_memory": bool(cfg.get("dataloader", {}).get("pin_memory", False)),
        "persistent_workers": bool(cfg.get("dataloader", {}).get("persistent_workers", False)),
        "prefetch_factor": int(cfg.get("dataloader", {}).get("prefetch_factor", 2)),
        "channels_last": channels_last,
        "compile_model": bool(train_cfg.get("compile_model", False)),
        "measured_iters": len(stages["total_iter"]),
        "img_s_wall": measured_images / wall_dt,
        "stages": {name: _summary(values) for name, values in stages.items()},
    }
    if args.loss_breakdown:
        report["loss_breakdown"] = {name: _summary(values) for name, values in loss_parts.items()}
    print(report)


if __name__ == "__main__":
    main()
