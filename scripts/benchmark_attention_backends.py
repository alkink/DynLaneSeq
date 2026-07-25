from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the row-local attention shape used by LaneRowNet.")
    parser.add_argument("--batch-rows", type=int, default=640)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--keys", type=int, default=400)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def backend_context(name: str):
    if name == "default":
        return nullcontext()
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        backend = {
            "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            "mem_efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            "math": torch.nn.attention.SDPBackend.MATH,
        }[name]
        return torch.nn.attention.sdpa_kernel(backend)
    return torch.backends.cuda.sdp_kernel(
        enable_flash=name == "flash",
        enable_mem_efficient=name == "mem_efficient",
        enable_math=name == "math",
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.float16
    q = torch.randn(args.batch_rows, args.queries, args.dim, device=device, dtype=dtype)
    k = torch.randn(args.batch_rows, args.keys, args.dim, device=device, dtype=dtype)
    v = torch.randn(args.batch_rows, args.keys, args.dim, device=device, dtype=dtype)
    for name in ("default", "flash", "mem_efficient", "math"):
        module = nn.MultiheadAttention(
            args.dim,
            args.heads,
            dropout=args.dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        ).train()
        try:
            with backend_context(name):
                for _ in range(args.warmup):
                    module.zero_grad(set_to_none=True)
                    out = module(q, k, v, need_weights=False)[0]
                    out.mean().backward()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.steps):
                    module.zero_grad(set_to_none=True)
                    out = module(q, k, v, need_weights=False)[0]
                    out.mean().backward()
                end.record()
                end.synchronize()
            elapsed_ms = float(start.elapsed_time(end)) / float(args.steps)
            print(f"{name:>14s}: {elapsed_ms:8.3f} ms forward+backward")
        except RuntimeError as exc:
            print(f"{name:>14s}: unavailable ({exc})")


if __name__ == "__main__":
    main()
