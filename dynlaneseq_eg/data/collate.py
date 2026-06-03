from __future__ import annotations

from typing import Any

import torch


def lane_collate(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["targets"] for item in batch]
    metas = [item["meta"] for item in batch]
    return images, targets, metas

