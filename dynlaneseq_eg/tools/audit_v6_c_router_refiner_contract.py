from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import (
    _materialize_model_state,
    load_compatible_model_weights,
)
from dynlaneseq_eg.engine.frozen_training import (
    freeze_except_parameter_prefixes,
    set_frozen_detector_eval,
)
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.train import seed_everything


SELECTOR_PREFIX = "structured_query_head.set_selection_head."
REFINER_PREFIX = SELECTOR_PREFIX + "slot_refinement."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gradient_groups(model: torch.nn.Module) -> dict[str, dict[str, float | int]]:
    accumulators = {
        "slot_refinement": [0.0, 0],
        "four_slot_router": [0.0, 0],
        "proposal_detector": [0.0, 0],
        "route_projection": [0.0, 0],
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        squared = float(parameter.grad.detach().float().square().sum())
        if name.startswith(REFINER_PREFIX):
            group = "slot_refinement"
        elif name.startswith(SELECTOR_PREFIX):
            group = "four_slot_router"
        else:
            group = "proposal_detector"
        accumulators[group][0] += squared
        accumulators[group][1] += 1
        if name.startswith(
            (
                SELECTOR_PREFIX + "slot_query.",
                SELECTOR_PREFIX + "candidate_key.",
            )
        ):
            accumulators["route_projection"][0] += squared
            accumulators["route_projection"][1] += 1
    return {
        name: {
            "norm": float(values[0]) ** 0.5,
            "tensor_count": int(values[1]),
        }
        for name, values in accumulators.items()
    }


def _captured_gradients(
    model: torch.nn.Module,
    *,
    prefix: str,
    exclude_prefix: str | None = None,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and name.startswith(prefix)
        and (exclude_prefix is None or not name.startswith(exclude_prefix))
    }


def _gradient_alignment(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    names = sorted(set(left) & set(right))
    if not names:
        return {"cosine": 0.0, "left_norm": 0.0, "right_norm": 0.0, "tensors": 0}
    dot = sum(float((left[name] * right[name]).sum()) for name in names)
    left_sq = sum(float(left[name].square().sum()) for name in names)
    right_sq = sum(float(right[name].square().sum()) for name in names)
    denominator = max((left_sq * right_sq) ** 0.5, 1.0e-12)
    return {
        "cosine": dot / denominator,
        "left_norm": left_sq**0.5,
        "right_norm": right_sq**0.5,
        "tensors": len(names),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V6-C straight-through geometry-to-router gradients while "
            "the V5 proposal detector remains frozen."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-source-iteration", type=int, default=28000)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _forward(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matcher: torch.nn.Module,
    criterion: torch.nn.Module,
    cfg: dict[str, Any],
    iteration: int,
    autocast_kwargs: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    with torch.autocast(**autocast_kwargs):
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            iteration,
        )
        losses = criterion(outputs, targets, matches)
    return outputs, losses


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    selector = model.structured_query_head.set_selection_head
    refiner = selector.slot_refinement
    if refiner is None:
        raise ValueError("V6-C config did not construct a slot refiner")
    load_stats = load_compatible_model_weights(checkpoint_path, model)
    source_state, source_payload = _materialize_model_state(checkpoint_path)
    missing_source = [
        name for name in model.state_dict() if name not in source_state
    ]

    prefixes = tuple(cfg["training"]["trainable_parameter_prefixes"])
    freeze_stats = freeze_except_parameter_prefixes(model, prefixes)
    set_frozen_detector_eval(
        model,
        tuple(cfg["training"]["trainable_module_prefixes"]),
    )
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    loader = build_dataloader(cfg, split="train", training=True)
    images, targets, _metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(args.amp_dtype)
    autocast_kwargs: dict[str, Any] = {
        "device_type": device.type,
        "enabled": device.type == "cuda" and amp_dtype is not None,
    }
    if amp_dtype is not None:
        autocast_kwargs["dtype"] = amp_dtype

    # Straight-through routing must change backward only.  In eval mode its
    # deployed hard curves must remain bitwise-equivalent to the V6-B gather.
    model.eval()
    refiner.straight_through_routing = True
    with torch.no_grad():
        st_outputs, _ = _forward(
            model,
            images,
            targets,
            matcher,
            criterion,
            cfg,
            int(args.expected_source_iteration),
            autocast_kwargs,
        )
    refiner.straight_through_routing = False
    with torch.no_grad():
        hard_outputs, _ = _forward(
            model,
            images,
            targets,
            matcher,
            criterion,
            cfg,
            int(args.expected_source_iteration),
            autocast_kwargs,
        )
    refiner.straight_through_routing = True
    forward_identity = float(
        (
            st_outputs["selection_slot_pred_x_rows"].float()
            - hard_outputs["selection_slot_pred_x_rows"].float()
        )
        .abs()
        .max()
        .cpu()
    )
    route_identity = torch.equal(
        st_outputs["selection_slot_indices"],
        hard_outputs["selection_slot_indices"],
    )

    set_frozen_detector_eval(
        model,
        tuple(cfg["training"]["trainable_module_prefixes"]),
    )
    model.zero_grad(set_to_none=True)
    outputs, losses = _forward(
        model,
        images,
        targets,
        matcher,
        criterion,
        cfg,
        int(args.expected_source_iteration),
        autocast_kwargs,
    )
    geometry = losses["loss_four_slot_geometry"]
    geometry.backward(retain_graph=True)
    gradient = _gradient_groups(model)
    geometry_router_grads = _captured_gradients(
        model,
        prefix=SELECTOR_PREFIX,
        exclude_prefix=REFINER_PREFIX,
    )
    model.zero_grad(set_to_none=True)
    losses["loss_four_slot_selection"].backward()
    selection_gradient = _gradient_groups(model)
    selection_router_grads = _captured_gradients(
        model,
        prefix=SELECTOR_PREFIX,
        exclude_prefix=REFINER_PREFIX,
    )
    router_alignment = _gradient_alignment(
        geometry_router_grads,
        selection_router_grads,
    )
    unique_routes = True
    for row in outputs["selection_slot_indices"]:
        selected = row[row >= 0]
        unique_routes &= int(selected.numel()) == int(selected.unique().numel())

    selector_parameter_count = sum(
        parameter.numel() for parameter in selector.parameters()
    )
    checks = {
        "source_iteration": int(source_payload.get("iteration", -1))
        == int(args.expected_source_iteration),
        "all_source_tensors_loaded": not missing_source,
        "only_selector_trainable": int(
            freeze_stats["trainable_parameter_count"]
        )
        == int(selector_parameter_count),
        "straight_through_enabled": bool(refiner.straight_through_routing),
        "slot_state_attached": not bool(refiner.detach_slot_states),
        "hard_forward_identity": forward_identity <= 1.0e-5,
        "hard_route_identity": bool(route_identity),
        "routes_are_globally_unique": bool(unique_routes),
        "finite_geometry_loss": bool(torch.isfinite(geometry.detach()).cpu()),
        "refiner_geometry_gradient_positive": float(
            gradient["slot_refinement"]["norm"]
        )
        > 0.0,
        "router_geometry_gradient_positive": float(
            gradient["four_slot_router"]["norm"]
        )
        > 0.0,
        "route_projection_geometry_gradient_positive": float(
            gradient["route_projection"]["norm"]
        )
        > 0.0,
        "proposal_detector_gradient_zero": float(
            gradient["proposal_detector"]["norm"]
        )
        == 0.0,
        "selection_proposal_detector_gradient_zero": float(
            selection_gradient["proposal_detector"]["norm"]
        )
        == 0.0,
    }
    payload = {
        "experiment": "V6-C straight-through router/refiner co-training",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_iteration": int(source_payload.get("iteration", -1)),
        "load_stats": load_stats,
        "missing_source_tensors": missing_source,
        "freeze_stats": freeze_stats,
        "selector_parameter_count": int(selector_parameter_count),
        "hard_forward_max_abs_px": forward_identity,
        "hard_route_identity": bool(route_identity),
        "geometry_gradient_contract": gradient,
        "selection_gradient_contract": selection_gradient,
        "router_loss_gradient_alignment": router_alignment,
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in losses.items()
            if name.startswith(("loss_four_slot", "four_slot_geometry"))
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")
    if not payload["passed"]:
        raise SystemExit("V6-C router/refiner gradient contract failed")


if __name__ == "__main__":
    main()
