from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
import numpy as np
import torch

from .lane_target_builder import decode_targets_to_points
from .transforms import IMAGENET_MEAN, IMAGENET_STD


COLORS = [
    (255, 64, 64),
    (64, 255, 64),
    (64, 160, 255),
    (255, 220, 64),
    (255, 64, 220),
    (64, 255, 220),
]


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    img = image.detach().cpu() * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def draw_lanes(
    image: Image.Image,
    lanes: Iterable[Iterable[tuple[float, float]]],
    width: int = 4,
    draw_points: bool = True,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for idx, lane in enumerate(lanes):
        pts = [(float(x), float(y)) for x, y in lane]
        color = COLORS[idx % len(COLORS)]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width)
        if draw_points:
            for x, y in pts:
                r = 2
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return out


def save_target_visualization(sample: dict, output_path: str | Path) -> None:
    image = tensor_to_pil(sample["image"])
    target = sample["targets"]
    lanes = decode_targets_to_points(
        target["x_rows"].cpu().numpy(),
        target["valid_mask"].cpu().numpy(),
        input_h=int(sample["meta"]["input_h"]),
    )
    out = draw_lanes(image, lanes)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)

