from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_compatible_model_weights
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import build_criterion, build_dataloader, build_matcher, build_model
from dynlaneseq_eg.losses.loss_s0 import S0Criterion
from dynlaneseq_eg.modeling.common import nested_to_device


def group_name(name: str) -> str:
    if name.startswith("encoder.backbone"):
        return "encoder.backbone"
    if name.startswith("encoder."):
        return "encoder.non_backbone"
    if name.startswith("structured_query_head."):
        return "structured_query_head"
    if name.startswith("active_corridor"):
        return "active_corridor"
    if name.startswith("active_corridor_sampler"):
        return "active_corridor_sampler"
    if name.startswith("quality_calibrator"):
        return "quality_calibrator"
    if name.startswith("bridge."):
        return "bridge"
    if name.startswith("adapter."):
        return "adapter"
    if name.startswith("row_decoder."):
        return "row_decoder"
    if name.startswith("row_embedding."):
        return "row_embedding"
    if name.startswith("heads."):
        return "coarse_heads"
    return "other"


def grad_summary(model: torch.nn.Module) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"params": 0, "grad_params": 0, "grad_l2_sq": 0.0, "grad_abs_max": 0.0}
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = stats[group_name(name)]
        group["params"] += int(param.numel())
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        group["grad_params"] += int(param.numel())
        group["grad_l2_sq"] += float(grad.pow(2).sum().item())
        group["grad_abs_max"] = max(float(group["grad_abs_max"]), float(grad.abs().max().item()))
    out = {}
    for key, value in stats.items():
        grad_l2_sq = float(value.pop("grad_l2_sq"))
        out[key] = {
            **value,
            "grad_l2": grad_l2_sq**0.5,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init-from", default="")
    parser.add_argument("--loss-mode", choices=["final", "total"], default="final")
    parser.add_argument("--iteration", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    if args.init_from:
        stats = load_compatible_model_weights(args.init_from, model)
        print(f"initialized compatible weights from {args.init_from}: {stats}")
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    loader = build_dataloader(cfg, split="train", training=True)
    images, targets, _metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)

    model.train()
    model.zero_grad(set_to_none=True)
    outputs, matches = forward_with_matches(model, images, targets, matcher, cfg, args.iteration)
    if args.loss_mode == "final" and isinstance(outputs, dict) and isinstance(outputs.get("final"), dict):
        final_criterion = S0Criterion(getattr(criterion, "cfg", None))
        loss_dict = final_criterion(outputs["final"], targets, matches)
    else:
        loss_dict = criterion(outputs, targets, matches)
    loss = loss_dict["loss_total"]
    loss.backward()
    result = {
        "config": args.config,
        "init_from": args.init_from,
        "loss_mode": args.loss_mode,
        "loss_total": float(loss.detach().float().item()),
        "grad_groups": grad_summary(model),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
