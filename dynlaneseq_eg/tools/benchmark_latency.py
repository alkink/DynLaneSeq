from __future__ import annotations

import argparse

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.evaluation.profiler import measure_forward_fps
from dynlaneseq_eg.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measure", type=int, default=500)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    images = torch.randn(1, 3, cfg["model"].get("input_h", 288), cfg["model"].get("input_w", 800), device=device)
    print(measure_forward_fps(model, images, warmup=args.warmup, measure=args.measure))


if __name__ == "__main__":
    main()

