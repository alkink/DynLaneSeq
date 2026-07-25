from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
    build_optimizer,
)
from dynlaneseq_eg.modeling.common import nested_to_device


def _gib(value: int) -> float:
    return float(value) / float(1024**3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure peak CUDA memory for real training microbatches.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--seg-aux-amp-dtype", choices=("inherit", "float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("measure_training_memory requires CUDA")
    cfg = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg["training"]["amp"] = True
    cfg["training"]["amp_dtype"] = str(args.amp_dtype)
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg.setdefault("dataloader", {})["num_workers"] = 0
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {}).setdefault("seg_aux", {})["amp_dtype"] = str(args.seg_aux_amp_dtype)

    seed = int(cfg.get("training", {}).get("seed", 3407))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = bool(cfg["training"].get("cudnn_benchmark", False))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["training"].get("tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(cfg["training"].get("tf32", False))
    torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    channels_last = bool(cfg["training"].get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    loader = iter(build_dataloader(cfg, split="train", training=True))
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

    global_peak_allocated = 0
    global_peak_reserved = 0
    for step in range(max(1, int(args.steps))):
        images, targets, _ = next(loader)
        if channels_last:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(images)
            matches = matcher(outputs, targets)
            if hasattr(criterion, "set_iteration"):
                criterion.set_iteration(step)
            loss = criterion(outputs, targets, matches)["loss_total"]
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        global_peak_allocated = max(global_peak_allocated, peak_allocated)
        global_peak_reserved = max(global_peak_reserved, peak_reserved)
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
        print(
            {
                "step": step + 1,
                "batch_size": int(images.shape[0]),
                "loss": float(loss.detach().float().cpu()),
                "peak_allocated_gib": round(_gib(peak_allocated), 3),
                "peak_reserved_gib": round(_gib(peak_reserved), 3),
                "device_free_gib_after_step": round(_gib(int(free_bytes)), 3),
                "device_total_gib": round(_gib(int(total_bytes)), 3),
            }
        )
        del images, targets, outputs, matches, loss

    print(
        {
            "config": args.config,
            "amp_dtype": args.amp_dtype,
            "seg_aux_amp_dtype": args.seg_aux_amp_dtype,
            "max_peak_allocated_gib": round(_gib(global_peak_allocated), 3),
            "max_peak_reserved_gib": round(_gib(global_peak_reserved), 3),
        }
    )


if __name__ == "__main__":
    main()
