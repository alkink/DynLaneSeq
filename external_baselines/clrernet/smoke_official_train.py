#!/usr/bin/env python3
"""One full CLRerNet training forward/backward without an optimizer update."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS, MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = Config.fromfile(str(args.config.resolve()))
    init_default_scope(cfg.default_scope)

    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
    assert len(train_dataset) == 88_880
    assert len(val_dataset) == 9_675

    samples = [train_dataset[index] for index in range(args.batch_size)]
    batch = pseudo_collate(samples)

    model = MODELS.build(cfg.model).cuda().train()
    processed = model.data_preprocessor(batch, training=True)
    losses = model(**processed, mode="loss")
    total_loss, log_vars = model.parse_losses(losses)
    assert torch.isfinite(total_loss).item()
    total_loss.backward()

    finite_grad_tensors = 0
    nonzero_grad_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all().item()
        finite_grad_tensors += 1
        nonzero_grad_tensors += int(parameter.grad.abs().max().item() > 0)
    assert finite_grad_tensors > 0
    assert nonzero_grad_tensors > 0

    report = {
        "passed": True,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "batch_size": args.batch_size,
        "input_shape": list(processed["inputs"].shape),
        "total_loss": float(total_loss.detach().cpu()),
        "losses": {
            key: float(value) if math.isfinite(float(value)) else None
            for key, value in log_vars.items()
        },
        "finite_grad_tensors": finite_grad_tensors,
        "nonzero_grad_tensors": nonzero_grad_tensors,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
