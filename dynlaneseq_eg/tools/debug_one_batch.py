from __future__ import annotations

import argparse

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_criterion, build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    loader = build_dataloader(cfg, split="train", training=True)
    images, targets, metas = next(iter(loader))
    images = images.to(device)
    targets = nested_to_device(targets, device)
    outputs = model(images)
    matches = matcher(outputs, targets)
    losses = criterion(outputs, targets, matches)
    losses["loss_total"].backward()
    print({k: float(v.detach().cpu()) for k, v in losses.items()})
    print({"batch": tuple(images.shape), "num_gt": [int(t["x_rows"].shape[0]) for t in targets]})


if __name__ == "__main__":
    main()

