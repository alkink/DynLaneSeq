from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.train import seed_everything


SELECTOR = "structured_query_head.set_selection_head."
ACTIVE = SELECTOR + "active."
REFINER = SELECTOR + "slot_refinement."
ROUTES = (SELECTOR + "slot_query.", SELECTOR + "candidate_key.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paired V7 hard/direct reference gradients."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16", "float16"),
        default="none",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _group(name: str) -> str:
    if name.startswith(ACTIVE):
        return "active"
    if name.startswith(ROUTES):
        return "route"
    if name.startswith(REFINER + "range_delta_head."):
        return "range"
    if name.startswith(REFINER):
        return "refiner"
    if name.startswith(SELECTOR):
        return "trunk"
    return "proposal"


def _snapshot(model: torch.nn.Module) -> tuple[dict[str, float], torch.Tensor]:
    sums = {name: 0.0 for name in ("active", "route", "range", "refiner", "trunk", "proposal")}
    route_parts: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        group = _group(name)
        if gradient is not None:
            sums[group] += float(gradient.detach().float().square().sum())
        if group == "route":
            if gradient is None:
                route_parts.append(torch.zeros(parameter.numel(), device="cpu"))
            else:
                route_parts.append(gradient.detach().float().reshape(-1).cpu())
    norms = {name: math.sqrt(value) for name, value in sums.items()}
    return norms, torch.cat(route_parts) if route_parts else torch.empty(0)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    if denominator <= 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.train()
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(int(iteration))
    loader = build_dataloader(cfg, split="train", training=True)
    images, targets, _metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        args.amp_dtype
    )
    with torch.autocast(
        device_type=device.type,
        enabled=device.type == "cuda" and dtype is not None,
        dtype=dtype if dtype is not None else torch.float32,
    ):
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            int(iteration),
        )
        losses = criterion(outputs, targets, matches)

    model.zero_grad(set_to_none=True)
    losses["loss_four_slot_geometry"].backward(retain_graph=True)
    geometry, geometry_route = _snapshot(model)
    model.zero_grad(set_to_none=True)
    losses["loss_four_slot_selection"].backward()
    selection, selection_route = _snapshot(model)

    selection_cfg = cfg["model"]["structured_query"]["set_selection"]
    reference_mode = str(
        selection_cfg.get("four_slot_refinement_reference_mode", "hard_st")
    )
    route_ratio = geometry["route"] / max(selection["route"], 1.0e-12)
    checks = {
        "reference_mode_valid": reference_mode in {"hard_st", "soft"},
        "geometry_active_zero": geometry["active"] == 0.0,
        "geometry_trunk_zero": geometry["trunk"] == 0.0,
        "geometry_proposal_zero": geometry["proposal"] == 0.0,
        "geometry_route_positive": geometry["route"] > 0.0,
        "geometry_refiner_positive": geometry["refiner"] > 0.0,
        "geometry_range_positive": geometry["range"] > 0.0,
        "selection_active_positive": selection["active"] > 0.0,
        "selection_route_positive": selection["route"] > 0.0,
        "selection_trunk_positive": selection["trunk"] > 0.0,
        "selection_proposal_zero": selection["proposal"] == 0.0,
    }
    payload = {
        "experiment": "V7 paired slot-reference gradient contract",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "iteration": int(iteration),
        "reference_mode": reference_mode,
        "geometry_gradients": geometry,
        "selection_gradients": selection,
        "route_geometry_to_selection_norm_ratio": route_ratio,
        "route_geometry_selection_cosine": _cosine(
            geometry_route,
            selection_route,
        ),
        "losses": {
            "selection": float(losses["loss_four_slot_selection"].detach().cpu()),
            "geometry": float(losses["loss_four_slot_geometry"].detach().cpu()),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")
    if not payload["passed"]:
        raise SystemExit("V7 reference gradient contract failed")


if __name__ == "__main__":
    main()
