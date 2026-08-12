from __future__ import annotations

import argparse
import gc
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
REFINER = SELECTOR + "slot_refinement."
NEIGHBORHOOD = REFINER + "neighborhood_"
ROUTES = (SELECTOR + "slot_query.", SELECTOR + "candidate_key.")
ACTIVE = SELECTOR + "active."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact V7 initialization parity and gradient isolation for "
            "the V8 route-anchored proposal-neighborhood refiner."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _prepare_config(path: str, args: argparse.Namespace) -> dict:
    cfg = load_config(path)
    cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _group(name: str) -> str:
    if name.startswith(NEIGHBORHOOD):
        return "neighborhood"
    if name.startswith(REFINER):
        return "existing_refiner"
    if name.startswith(ACTIVE):
        return "active"
    if name.startswith(ROUTES):
        return "route"
    if name.startswith(SELECTOR):
        return "selector_trunk"
    return "proposal_detector"


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    squares = {
        name: 0.0
        for name in (
            "neighborhood",
            "existing_refiner",
            "active",
            "route",
            "selector_trunk",
            "proposal_detector",
        )
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            squares[_group(name)] += float(
                parameter.grad.detach().float().square().sum()
            )
    return {name: math.sqrt(value) for name, value in squares.items()}


def _cpu_snapshot(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    names = (
        "selection_slot_indices",
        "selection_slot_scores",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
        "selection_slot_active",
    )
    return {
        name: outputs[name].detach().float().cpu()
        if outputs[name].is_floating_point()
        else outputs[name].detach().cpu()
        for name in names
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.dtype == torch.bool or not left.is_floating_point():
        return float((left != right).sum())
    return float((left.float() - right.float()).abs().max())


def main() -> None:
    args = parse_args()
    cfg = _prepare_config(args.config, args)
    baseline_cfg = _prepare_config(args.baseline_config, args)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    loader = build_dataloader(cfg, split="train", training=True)
    images, targets, _metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)

    baseline = build_model(baseline_cfg).to(device)
    iteration = load_checkpoint(args.checkpoint, baseline, strict=False)
    baseline.eval()
    with torch.no_grad():
        baseline_snapshot = _cpu_snapshot(baseline(images))
    del baseline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    loaded_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    if int(loaded_iteration) != int(iteration):
        raise ValueError("baseline and V8 checkpoint iterations differ")
    model.eval()
    with torch.no_grad():
        initial_outputs = model(images)
        candidate_snapshot = _cpu_snapshot(initial_outputs)
    parity = {
        name: _max_abs(baseline_snapshot[name], candidate_snapshot[name])
        for name in baseline_snapshot
    }

    # Keep the immutable detector deterministic while enabling the V8
    # gradient-only straight-through neighborhood path.
    model.eval()
    refiner = model.structured_query_head.set_selection_head.slot_refinement
    if refiner is None:
        raise ValueError("V8 config has no slot refiner")
    refiner.train(True)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(int(iteration))
    outputs, matches = forward_with_matches(
        model,
        images,
        targets,
        matcher,
        cfg,
        int(iteration),
    )
    losses = criterion(outputs, targets, matches)
    geometry_loss = losses["loss_four_slot_geometry"]
    model.zero_grad(set_to_none=True)
    geometry_loss.backward()
    gradients = _gradient_norms(model)
    mix = refiner.neighborhood_mix
    if mix is None:
        raise ValueError("V8 neighborhood mix parameter is absent")
    mix_gradient = (
        0.0
        if mix.grad is None
        else float(mix.grad.detach().float().abs())
    )
    support = outputs["selection_slot_neighborhood_support"].detach().float()
    valid = outputs["selection_slot_geometry_valid"].detach().bool()
    valid_support = support[valid]
    mean_support = float(valid_support.mean()) if valid_support.numel() else 0.0
    alternative_fraction = (
        float((valid_support > 1).float().mean())
        if valid_support.numel()
        else 0.0
    )
    checks = {
        "checkpoint_iteration_preserved": int(iteration) == int(loaded_iteration),
        "selected_indices_exact": parity["selection_slot_indices"] == 0.0,
        "selected_scores_exact": parity["selection_slot_scores"] == 0.0,
        "initial_x_exact": parity["selection_slot_pred_x_rows"] <= 1.0e-6,
        "initial_range_exact": parity["selection_slot_range_norm"] <= 1.0e-7,
        "initial_active_exact": parity["selection_slot_active"] == 0.0,
        "neighborhood_gradient_positive": gradients["neighborhood"] > 0.0,
        "mix_gradient_positive": mix_gradient > 0.0,
        "refiner_gradient_positive": gradients["existing_refiner"] > 0.0,
        "geometry_to_active_zero": gradients["active"] == 0.0,
        "geometry_to_global_route_zero": gradients["route"] == 0.0,
        "geometry_to_selector_trunk_zero": gradients["selector_trunk"] == 0.0,
        "geometry_to_proposal_detector_zero": gradients["proposal_detector"] == 0.0,
        "neighborhood_has_alternatives": alternative_fraction > 0.0,
    }
    payload = {
        "experiment": "V8 anchor-neighborhood initialization and gradient contract",
        "diagnostic_only": True,
        "test_set_used": False,
        "config": str(Path(args.config).resolve()),
        "baseline_config": str(Path(args.baseline_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "iteration": int(iteration),
        "initial_parity": parity,
        "geometry_gradients": gradients,
        "mix_gradient_abs": mix_gradient,
        "geometry_loss": float(geometry_loss.detach().cpu()),
        "neighborhood": {
            "mean_support": mean_support,
            "alternative_fraction": alternative_fraction,
            "mean_entropy": float(
                outputs["selection_slot_neighborhood_entropy"].detach().mean()
            ),
            "mean_top1_mass": float(
                outputs["selection_slot_neighborhood_top1_mass"].detach().mean()
            ),
            "initial_mix": float(
                outputs["selection_slot_neighborhood_mix"].detach()
            ),
            "initial_reference_shift_px": float(
                outputs[
                    "selection_slot_neighborhood_reference_shift_px"
                ].detach().mean()
            ),
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
        raise SystemExit("V8 anchor-neighborhood contract failed")


if __name__ == "__main__":
    main()
