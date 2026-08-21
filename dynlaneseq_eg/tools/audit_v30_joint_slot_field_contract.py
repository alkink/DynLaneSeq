from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_compatible_model_weights
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.train import seed_everything


PARITY_TENSORS = (
    "pred_x_rows",
    "range_norm",
    "selection_slot_real_route_logits",
    "selection_slot_geometry_route_indices",
    "selection_slot_active_logits",
    "selection_slot_indices",
    "selection_slot_scores",
    "selection_slot_input_reference_x_rows",
    "selection_slot_input_range_norm",
    "selection_slot_pred_x_rows",
    "selection_slot_range_norm",
    "selection_slot_active",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V30 zero-step parity and route-to-image gradients."
    )
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--treatment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-iteration", type=int, default=30000)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def prepare(path: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg.setdefault("dataloader", {})["num_workers"] = 0
    cfg["dataloader"]["persistent_workers"] = False
    cfg["dataloader"]["eval_batch_size"] = 1
    return cfg


def snapshot(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: outputs[name].detach().cpu().clone() for name in PARITY_TENSORS
    }


def difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.dtype == torch.bool or not left.is_floating_point():
        return float((left != right).sum())
    return float((left.float() - right.float()).abs().max())


def gradient_norms(
    names: tuple[str, ...],
    gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, float]:
    prefixes = {
        "backbone": ("encoder.backbone.",),
        "fpn": ("encoder.fpn.",),
        "row_feature_projection": ("structured_query_head.feature_proj.",),
        "joint_field": (
            "structured_query_head.set_selection_head.joint_slot_field.",
        ),
        "slot_decoder": (
            "structured_query_head.set_selection_head.slot_decoder.",
            "structured_query_head.set_selection_head.slot_tokens.",
            "structured_query_head.set_selection_head.slot_norm.",
        ),
        "legacy_coordinate_heads": (
            "structured_query_head.row_delta_heads.",
            "structured_query_head.reference_",
        ),
    }
    result: dict[str, float] = {}
    for group, group_prefixes in prefixes.items():
        squared = 0.0
        for name, gradient in zip(names, gradients):
            if gradient is None or not name.startswith(group_prefixes):
                continue
            squared += float(gradient.detach().float().square().sum())
        result[group] = math.sqrt(squared)
    return result


def gradient_cosines(
    names: tuple[str, ...],
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> dict[str, float]:
    prefixes = {
        "backbone": ("encoder.backbone.",),
        "fpn": ("encoder.fpn.",),
        "row_feature_projection": ("structured_query_head.feature_proj.",),
    }
    result: dict[str, float] = {}
    for group, group_prefixes in prefixes.items():
        dot = left_sq = right_sq = 0.0
        for name, left_grad, right_grad in zip(names, left, right):
            if not name.startswith(group_prefixes):
                continue
            if left_grad is not None:
                left_value = left_grad.detach().float()
                left_sq += float(left_value.square().sum())
            if right_grad is not None:
                right_value = right_grad.detach().float()
                right_sq += float(right_value.square().sum())
            if left_grad is not None and right_grad is not None:
                dot += float((left_value * right_value).sum())
        denominator = math.sqrt(left_sq * right_sq)
        result[group] = 0.0 if denominator == 0.0 else dot / denominator
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    source_cfg = prepare(args.source_config, args)
    treatment_cfg = prepare(args.treatment_config, args)
    seed = int(treatment_cfg.get("training", {}).get("seed", 3407))
    seed_everything(seed)

    loader = build_dataloader(treatment_cfg, split="val", training=False)
    images, targets, _metas = next(iter(loader))
    images = images.to(device)
    targets = nested_to_device(targets, device)

    source = build_model(source_cfg).to(device).eval()
    source_load = load_compatible_model_weights(args.checkpoint, source)
    source_matcher = build_matcher(source_cfg)
    with torch.no_grad():
        source_outputs, _source_matches = forward_with_matches(
            source,
            images,
            targets,
            source_matcher,
            source_cfg,
            int(args.start_iteration),
        )
    source_snapshot = snapshot(source_outputs)
    del source, source_outputs, source_matcher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seed_everything(seed)
    treatment = build_model(treatment_cfg).to(device).eval()
    treatment_load = load_compatible_model_weights(args.checkpoint, treatment)
    matcher = build_matcher(treatment_cfg)
    criterion = build_criterion(treatment_cfg).to(device)
    criterion.set_iteration(int(args.start_iteration))
    treatment_outputs, matches = forward_with_matches(
        treatment,
        images,
        targets,
        matcher,
        treatment_cfg,
        int(args.start_iteration),
    )
    losses = criterion(treatment_outputs, targets, matches)
    treatment_snapshot = snapshot(treatment_outputs)
    parity = {
        name: difference(source_snapshot[name], treatment_snapshot[name])
        for name in PARITY_TENSORS
    }

    named_parameters = tuple(
        (name, parameter)
        for name, parameter in treatment.named_parameters()
        if parameter.requires_grad
    )
    names = tuple(name for name, _parameter in named_parameters)
    parameters = tuple(parameter for _name, parameter in named_parameters)
    field_gradients = torch.autograd.grad(
        losses["loss_four_slot_joint_field"],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    field_weight = float(
        treatment_cfg.get("loss", {}).get("w_four_slot_joint_field", 0.0)
    )
    legacy_total = losses["loss_total"] - field_weight * losses[
        "loss_four_slot_joint_field"
    ]
    legacy_gradients = torch.autograd.grad(
        legacy_total,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    route_gradients = torch.autograd.grad(
        losses["loss_four_slot_selection"],
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    field_norms = gradient_norms(names, field_gradients)
    weighted_field_norms = {
        name: field_weight * value for name, value in field_norms.items()
    }
    legacy_norms = gradient_norms(names, legacy_gradients)
    field_to_legacy_ratio = {
        name: weighted_field_norms[name] / max(legacy_norms[name], 1.0e-12)
        for name in field_norms
    }
    field_legacy_cosine = gradient_cosines(
        names, field_gradients, legacy_gradients
    )
    route_norms = gradient_norms(names, route_gradients)

    selector = treatment.structured_query_head.set_selection_head
    route_gate = float(selector.joint_slot_field.route_gate.detach().cpu())
    route_residual = treatment_outputs[
        "selection_slot_joint_field_route_residual"
    ]
    discrete_names = {
        "selection_slot_geometry_route_indices",
        "selection_slot_indices",
        "selection_slot_active",
    }
    numeric_tolerances = {
        "pred_x_rows": 0.0,
        "range_norm": 0.0,
        "selection_slot_real_route_logits": 2.0e-3,
        "selection_slot_active_logits": 2.0e-3,
        "selection_slot_scores": 2.0e-4,
        "selection_slot_input_reference_x_rows": 0.0,
        "selection_slot_input_range_norm": 0.0,
        "selection_slot_pred_x_rows": 2.5e-3,
        "selection_slot_range_norm": 2.5e-6,
    }
    parity_within_tolerance = all(
        value == 0.0
        if name in discrete_names
        else value <= numeric_tolerances[name]
        for name, value in parity.items()
    )
    checks = {
        "source_checkpoint_loaded": source_load["loaded"] > 0,
        "treatment_loaded_all_v7_tensors": treatment_load["loaded"]
        == source_load["loaded"],
        "zero_step_public_parity_within_gpu_tolerance": (
            parity_within_tolerance
        ),
        "zero_step_hard_decisions_exact": all(
            parity[name] == 0.0 for name in discrete_names
        ),
        "zero_step_route_residual_exact_zero": bool(
            torch.count_nonzero(route_residual.detach()).item() == 0
        ),
        "zero_step_route_gate_exact_zero": route_gate == 0.0,
        "field_loss_finite": bool(
            torch.isfinite(losses["loss_four_slot_joint_field"]).item()
        ),
        "field_reaches_backbone": field_norms["backbone"] > 0.0,
        "field_reaches_fpn": field_norms["fpn"] > 0.0,
        "field_reaches_row_projection": field_norms[
            "row_feature_projection"
        ]
        > 0.0,
        "field_reaches_private_head": field_norms["joint_field"] > 0.0,
        "field_reaches_slot_decoder": field_norms["slot_decoder"] > 0.0,
        "field_does_not_directly_move_proposal_coordinates": field_norms[
            "legacy_coordinate_heads"
        ]
        == 0.0,
        "route_loss_can_open_field_gate": route_norms["joint_field"] > 0.0,
        "weighted_field_gradient_is_material": field_to_legacy_ratio[
            "backbone"
        ]
        > 0.05,
    }
    report = {
        "experiment": "V30 joint slot-field zero-step contract",
        "checkpoint": str(args.checkpoint),
        "source_load": source_load,
        "treatment_load": treatment_load,
        "zero_step_parity": parity,
        "route_gate": route_gate,
        "loss_joint_field": float(
            losses["loss_four_slot_joint_field"].detach().cpu()
        ),
        "field_gradient_norms": field_norms,
        "weighted_field_gradient_norms": weighted_field_norms,
        "legacy_gradient_norms": legacy_norms,
        "weighted_field_to_legacy_gradient_ratio": field_to_legacy_ratio,
        "field_legacy_gradient_cosine": field_legacy_cosine,
        "route_gradient_norms": route_norms,
        "checks": checks,
        "passed": all(checks.values()),
        "test_split_used": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("V30 joint slot-field contract failed")


if __name__ == "__main__":
    main()
