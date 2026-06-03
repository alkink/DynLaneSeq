from __future__ import annotations

import argparse

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.engine.validate_s0 import validate_simple
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, strict=False)
    loader = build_dataloader(cfg, split=args.split, training=False)
    matcher = build_matcher(cfg)
    stats = validate_simple(
        model,
        loader,
        matcher,
        device,
        max_batches=args.max_batches or None,
        score_thresh=args.score_thresh,
    )
    print(stats)


if __name__ == "__main__":
    main()
