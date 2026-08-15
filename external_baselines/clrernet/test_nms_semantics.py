#!/usr/bin/env python3
"""Check the Torch-2.11 compatibility build against the original NMS rule."""

from __future__ import annotations

import torch

from nms import nms


N_OFFSETS = 72
N_STRIPS = N_OFFSETS - 1


def overlaps(a: torch.Tensor, b: torch.Tensor, threshold: float) -> bool:
    start_a = int(float(a[2]) * N_STRIPS + 0.5)
    start_b = int(float(b[2]) * N_STRIPS + 0.5)
    start = max(start_a, start_b)
    length_a = float(a[4])
    length_b = float(b[4])
    end_a = int(start_a + length_a - 1 + 0.5 - ((length_a - 1) < 0))
    end_b = int(start_b + length_b - 1 + 0.5 - ((length_b - 1) < 0))
    end = min(end_a, end_b, N_OFFSETS - 1)
    if end < start:
        return False
    distance = (a[5 + start : 5 + end + 1] - b[5 + start : 5 + end + 1]).abs().sum()
    return float(distance) < threshold * (end - start + 1)


def reference_nms(
    boxes: torch.Tensor, scores: torch.Tensor, threshold: float, top_k: int
) -> list[int]:
    order = torch.argsort(scores, descending=True).tolist()
    removed = [False] * len(order)
    keep: list[int] = []
    boxes_cpu = boxes.cpu()
    for position, original_idx in enumerate(order):
        if removed[position]:
            continue
        keep.append(original_idx)
        if len(keep) == top_k:
            break
        for later in range(position + 1, len(order)):
            if not removed[later] and overlaps(
                boxes_cpu[original_idx], boxes_cpu[order[later]], threshold
            ):
                removed[later] = True
    return keep


def make_case(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    count = 192
    boxes = torch.zeros(count, 5 + N_OFFSETS, dtype=torch.float32)
    boxes[:, 2] = torch.randint(0, 12, (count,), generator=generator).float() / N_STRIPS
    boxes[:, 4] = torch.randint(45, 73, (count,), generator=generator).float()
    rows = torch.arange(N_OFFSETS, dtype=torch.float32)
    clusters = torch.randint(0, 12, (count,), generator=generator).float() * 62.0 + 40.0
    slopes = torch.randn(count, generator=generator) * 0.18
    noise = torch.randn(count, N_OFFSETS, generator=generator) * 1.2
    boxes[:, 5:] = clusters[:, None] + slopes[:, None] * rows + noise
    # Strictly unique scores avoid implementation-dependent tie ordering.
    scores = torch.rand(count, generator=generator) + torch.arange(count) * 1e-7
    return boxes, scores


def main() -> None:
    assert torch.cuda.is_available()
    for seed in range(12):
        boxes, scores = make_case(seed)
        expected = reference_nms(boxes, scores, threshold=50.0, top_k=4)
        keep, count, _ = nms(
            boxes.cuda(), scores.cuda(), overlap=50.0, top_k=4
        )
        actual = keep[: int(count)].cpu().tolist()
        assert actual == expected, (seed, actual, expected)
    print("NMS semantic parity PASS: 12/12 deterministic cases")


if __name__ == "__main__":
    main()
