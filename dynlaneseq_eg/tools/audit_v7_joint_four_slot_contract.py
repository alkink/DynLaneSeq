from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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
from dynlaneseq_eg.modeling.four_slot_selection import (
    structured_unique_route_marginals,
)
from dynlaneseq_eg.tools.train import seed_everything


SELECTOR = "structured_query_head.set_selection_head."
REFINER = SELECTOR + "slot_refinement."
ACTIVE = SELECTOR + "active."
ROUTE_PREFIXES = (
    SELECTOR + "slot_query.",
    SELECTOR + "candidate_key.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_group(name: str) -> str:
    if name.startswith(ACTIVE):
        return "active_head"
    if name.startswith(ROUTE_PREFIXES):
        return "route_projection"
    if name.startswith(REFINER + "range_delta_head."):
        return "range_refinement"
    if name.startswith(REFINER):
        return "slot_refinement"
    if name.startswith(SELECTOR):
        return "slot_router_trunk"
    return "proposal_detector"


def _gradient_groups(model: torch.nn.Module) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float | int]] = {
        "active_head": [0.0, 0],
        "route_projection": [0.0, 0],
        "range_refinement": [0.0, 0],
        "slot_refinement": [0.0, 0],
        "slot_router_trunk": [0.0, 0],
        "proposal_detector": [0.0, 0],
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = _parameter_group(name)
        values[group][0] = float(values[group][0]) + float(
            parameter.grad.detach().float().square().sum()
        )
        values[group][1] = int(values[group][1]) + 1
    return {
        name: {
            "norm": float(value[0]) ** 0.5,
            "tensor_count": int(value[1]),
        }
        for name, value in values.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the from-scratch V7 four-slot cardinality, structured "
            "routing and gradient-isolation contract."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optionally audit a trained joint checkpoint instead of step zero.",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16", "float16"),
        default="none",
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
    autocast_kwargs: dict[str, Any],
    iteration: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    with torch.autocast(**autocast_kwargs):
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            int(iteration),
        )
        losses = criterion(outputs, targets, matches)
    return outputs, losses


def _gradient_vector(
    model: torch.nn.Module,
    prefixes: tuple[str, ...],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes):
            continue
        if parameter.grad is None:
            parts.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            parts.append(parameter.grad.detach().float().reshape(-1).cpu())
    return torch.cat(parts) if parts else torch.empty(0)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    if denominator <= 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
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
    iteration = 0
    if args.checkpoint:
        iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.train()
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(int(iteration))
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

    outputs, losses = _forward(
        model,
        images,
        targets,
        matcher,
        criterion,
        cfg,
        autocast_kwargs,
        int(iteration),
    )
    real_logits = outputs["selection_slot_real_route_logits"]
    candidate_valid = outputs["selection_slot_candidate_valid"]
    marginal = structured_unique_route_marginals(
        real_logits,
        candidate_valid,
        temperature=float(
            cfg["model"]["structured_query"]["set_selection"][
                "four_slot_refinement_route_temperature"
            ]
        ),
    )
    row_error = float((marginal.sum(dim=-1) - 1.0).abs().max().detach().cpu())
    max_column_mass = float(marginal.sum(dim=1).max().detach().cpu())
    invalid_mass = float(
        marginal.masked_select(~candidate_valid[:, None, :]).sum().detach().cpu()
    )

    expected_gt_count = torch.stack(
        [
            target["valid_mask"].bool().any(dim=-1).sum().clamp(max=4)
            for target in targets
        ]
    ).float().mean()
    reported_gt_count = losses["four_slot_target_mean_representable_count"]

    model.zero_grad(set_to_none=True)
    losses["loss_four_slot_geometry"].backward(retain_graph=True)
    geometry_gradients = _gradient_groups(model)
    geometry_route_vector = _gradient_vector(model, ROUTE_PREFIXES)
    model.zero_grad(set_to_none=True)
    losses["loss_four_slot_selection"].backward(retain_graph=True)
    selection_gradients = _gradient_groups(model)
    selection_route_vector = _gradient_vector(model, ROUTE_PREFIXES)
    model.zero_grad(set_to_none=True)
    losses["loss_total"].backward()
    total_gradients = _gradient_groups(model)

    geometry_routes = outputs["selection_slot_geometry_route_indices"]
    unique_routes = True
    for row in geometry_routes:
        valid = row[row >= 0]
        unique_routes &= int(valid.numel()) == int(valid.unique().numel())

    selection = cfg["model"]["structured_query"]["set_selection"]
    loss_cfg = cfg["loss"]
    checks = {
        "factorized_cardinality": bool(
            selection.get("four_slot_factorized_routing")
        ),
        "hard_slot_assignment": (
            loss_cfg.get("four_slot_assignment_mode") == "hard_min"
        ),
        "structured_unique_backward": bool(
            selection.get("four_slot_refinement_structured_unique_routing")
        ),
        "range_refinement_enabled": bool(
            selection.get("four_slot_range_refinement_enabled")
        ),
        "all_gt_target_mode": loss_cfg.get("four_slot_target_mode") == "all_gt",
        "all_slot_geometry_matching": bool(
            loss_cfg.get("four_slot_geometry_match_all_slots")
        ),
        "target_count_equals_gt_count": bool(
            torch.allclose(
                reported_gt_count.detach().float(),
                expected_gt_count.to(reported_gt_count.device),
                atol=1.0e-5,
            )
        ),
        "hard_geometry_routes_unique": bool(unique_routes),
        "structured_rows_sum_to_one": row_error <= 1.0e-4,
        "structured_columns_at_most_one": max_column_mass <= 1.0 + 1.0e-4,
        "structured_invalid_mass_zero": abs(invalid_mass) <= 1.0e-7,
        "geometry_active_gradient_zero": float(
            geometry_gradients["active_head"]["norm"]
        )
        == 0.0,
        "geometry_router_trunk_gradient_zero": float(
            geometry_gradients["slot_router_trunk"]["norm"]
        )
        == 0.0,
        "geometry_route_projection_gradient_positive": float(
            geometry_gradients["route_projection"]["norm"]
        )
        > 0.0,
        "geometry_refiner_gradient_positive": float(
            geometry_gradients["slot_refinement"]["norm"]
        )
        > 0.0,
        "geometry_range_gradient_positive": float(
            geometry_gradients["range_refinement"]["norm"]
        )
        > 0.0,
        "geometry_proposal_gradient_zero": float(
            geometry_gradients["proposal_detector"]["norm"]
        )
        == 0.0,
        "selection_active_gradient_positive": float(
            selection_gradients["active_head"]["norm"]
        )
        > 0.0,
        "selection_route_gradient_positive": float(
            selection_gradients["route_projection"]["norm"]
        )
        > 0.0,
        "selection_proposal_gradient_zero": float(
            selection_gradients["proposal_detector"]["norm"]
        )
        == 0.0,
        "joint_proposal_training_positive": float(
            total_gradients["proposal_detector"]["norm"]
        )
        > 0.0,
    }
    payload = {
        "experiment": "V7 joint from-scratch four-slot contract",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": (
            str(Path(args.checkpoint).resolve()) if args.checkpoint else ""
        ),
        "iteration": int(iteration),
        "batch_size": int(images.shape[0]),
        "expected_mean_gt_count": float(expected_gt_count.cpu()),
        "reported_mean_slot_target_count": float(
            reported_gt_count.detach().float().cpu()
        ),
        "structured_route": {
            "max_row_sum_error": row_error,
            "max_column_mass": max_column_mass,
            "invalid_mass": invalid_mass,
        },
        "geometry_gradients": geometry_gradients,
        "selection_gradients": selection_gradients,
        "total_gradients": total_gradients,
        "route_geometry_selection_cosine": _cosine(
            geometry_route_vector,
            selection_route_vector,
        ),
        "route_geometry_to_selection_norm_ratio": float(
            geometry_route_vector.norm()
            / selection_route_vector.norm().clamp_min(1.0e-12)
        ),
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in losses.items()
            if name.startswith(("loss_four_slot", "four_slot_"))
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
        raise SystemExit("V7 joint four-slot contract failed")


if __name__ == "__main__":
    main()
