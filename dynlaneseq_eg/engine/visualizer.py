from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.data.visualization import draw_lanes, tensor_to_pil
from dynlaneseq_eg.evaluation.postprocess import predictions_to_lanes


@torch.no_grad()
def save_prediction_visuals(
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    metas: list[dict[str, Any]],
    outputs: dict,
    out_dir: str | Path,
    step: int,
    score_thresh: float = 0.5,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = outputs.get("stage2") or outputs.get("final") or outputs
    for b in range(images.shape[0]):
        image = tensor_to_pil(images[b].cpu())
        pred = {k: v[b : b + 1] for k, v in flat.items() if isinstance(v, torch.Tensor) and v.shape[0] == images.shape[0]}
        lanes = predictions_to_lanes(pred, score_thresh=score_thresh)[0]
        out = draw_lanes(image, lanes, width=3)
        out.save(out_dir / f"step_{step:07d}_sample_{b}_range_filtered.jpg")

