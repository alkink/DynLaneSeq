from __future__ import annotations

import time

import torch


@torch.no_grad()
def measure_forward_fps(model, images: torch.Tensor, warmup: int = 100, measure: int = 500) -> dict[str, float]:
    model.eval()
    device = images.device
    for _ in range(warmup):
        _ = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(measure):
        _ = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {"elapsed_sec": elapsed, "fps": measure / max(elapsed, 1e-9)}

