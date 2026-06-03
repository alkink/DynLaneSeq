from __future__ import annotations

from pathlib import Path

import torch

from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions


@torch.no_grad()
def write_validation_predictions(model, dataloader, device: torch.device, out_dir: str | Path, score_thresh: float = 0.5) -> None:
    model.eval()
    for images, targets, metas in dataloader:
        images = images.to(device)
        outputs = model(images)
        write_culane_predictions(outputs, metas, out_dir, score_thresh=score_thresh)

