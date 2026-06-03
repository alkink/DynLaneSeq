from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.engine.visualizer import save_prediction_visuals
from dynlaneseq_eg.factory import build_dataloader, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/pred_vis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()
    loader = build_dataloader(cfg, split=args.split, training=False)
    with torch.no_grad():
        for step, (images, targets, metas) in enumerate(loader):
            if step >= args.num_batches:
                break
            images = images.to(device)
            outputs = model(images)
            save_prediction_visuals(images, targets, metas, outputs, Path(args.output_dir), step)


if __name__ == "__main__":
    main()

