from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_matcher, build_model
from dynlaneseq_eg.modeling.unified_lane_set import ProtectedOwnershipLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the V5 protected ownership architecture, matcher schedule, "
            "and differentiable gradient boundary before expensive training."
        )
    )
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--assignment-config", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _grad_norm(parameters) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().double().square().sum()
    return float(total.sqrt().item())


def _source_grad_norm(tensor: torch.Tensor) -> float:
    if tensor.grad is None:
        return 0.0
    return float(tensor.grad.detach().double().norm().item())


def gradient_boundary_audit() -> dict[str, Any]:
    torch.manual_seed(17)
    layer = ProtectedOwnershipLayer(
        dim=16,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        semantic_context={
            "enabled": True,
            "scales": ["p4", "p5"],
            "pool_size": [2, 3],
        },
    )
    layer.train()
    state = torch.randn(2, 4, 16, requires_grad=True)
    identity = torch.randn(4, 16, requires_grad=True)
    geometry_lane = torch.randn(2, 4, 16, requires_grad=True)
    geometry_rows = torch.randn(2, 4, 6, 16, requires_grad=True)
    p4 = torch.randn(2, 16, 4, 6, requires_grad=True)
    p5 = torch.randn(2, 16, 2, 3, requires_grad=True)
    output = layer(
        state,
        identity,
        geometry_lane,
        geometry_rows,
        multi_scale_features={"p4": p4, "p5": p5},
    )
    output.float().square().mean().backward()
    result = {
        "ownership_parameter_grad_norm": _grad_norm(layer.parameters()),
        "ownership_state_grad_norm": _source_grad_norm(state),
        "ownership_identity_grad_norm": _source_grad_norm(identity),
        "geometry_lane_grad_norm": _source_grad_norm(geometry_lane),
        "geometry_rows_grad_norm": _source_grad_norm(geometry_rows),
        "p4_grad_norm": _source_grad_norm(p4),
        "p5_grad_norm": _source_grad_norm(p5),
    }
    result["passed"] = bool(
        result["ownership_parameter_grad_norm"] > 0.0
        and result["ownership_state_grad_norm"] > 0.0
        and result["ownership_identity_grad_norm"] > 0.0
        and result["geometry_lane_grad_norm"] == 0.0
        and result["geometry_rows_grad_norm"] == 0.0
        and result["p4_grad_norm"] == 0.0
        and result["p5_grad_norm"] == 0.0
    )
    return result


def matcher_intervention_audit(
    control_cfg: dict[str, Any],
    assignment_cfg: dict[str, Any],
) -> dict[str, Any]:
    control = build_matcher(control_cfg)
    assignment = build_matcher(assignment_cfg)
    rows = 12
    target_x = torch.full((1, rows), 800.0)
    target = {
        "x_rows": target_x,
        "valid_mask": torch.ones_like(target_x, dtype=torch.bool),
        "range_y": torch.tensor([[0.0, 639.0]]),
    }
    # Candidate 0 is geometrically exact but has low ownership confidence.
    # Candidate 1 is only 0.5 px worse and has high ownership confidence.
    pred_x = torch.stack((target_x[0], target_x[0] + 0.5), dim=0)
    ranges = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    exist_logits = torch.tensor([[-4.0, 0.0], [4.0, 0.0]])

    def selected(matcher, iteration: int) -> tuple[int, float]:
        matcher.set_iteration(iteration)
        cost, _ = matcher.compute_cost_for_image(
            exist_logits,
            pred_x,
            ranges,
            target,
        )
        return int(cost[:, 0].argmin().item()), matcher.effective_lambda_obj()

    control_choice, control_weight = selected(control, 25000)
    warmup_choice, warmup_weight = selected(assignment, 10000)
    final_choice, final_weight = selected(assignment, 25000)
    result = {
        "control_choice": control_choice,
        "control_weight": control_weight,
        "assignment_warmup_choice": warmup_choice,
        "assignment_warmup_weight": warmup_weight,
        "assignment_final_choice": final_choice,
        "assignment_final_weight": final_weight,
    }
    result["passed"] = bool(
        control_choice == 0
        and warmup_choice == 0
        and final_choice == 1
        and abs(control_weight) < 1e-12
        and abs(warmup_weight) < 1e-12
        and abs(final_weight - 0.25) < 1e-12
    )
    return result


def config_checks(
    control: dict[str, Any],
    assignment: dict[str, Any],
) -> dict[str, bool]:
    structured = control["model"]["structured_query"]
    ownership = structured["ownership"]
    loss = control["loss"]
    control_matcher = control["matcher"]
    assignment_matcher = assignment["matcher"]
    coupling_fields = {
        "lambda_obj",
        "lambda_obj_start",
        "lambda_obj_end",
        "lambda_obj_ramp_start_iter",
        "lambda_obj_ramp_end_iter",
    }
    control_matcher_common = {
        key: value
        for key, value in control_matcher.items()
        if key not in coupling_fields
    }
    assignment_matcher_common = {
        key: value
        for key, value in assignment_matcher.items()
        if key not in coupling_fields
    }
    groups = {
        entry["name"]: entry
        for entry in control["optimizer"]["parameter_groups"]
    }
    return {
        "arms_share_model": control["model"] == assignment["model"],
        "arms_share_loss": control["loss"] == assignment["loss"],
        "arms_share_optimizer": control["optimizer"] == assignment["optimizer"],
        "arms_share_scheduler": control["scheduler"] == assignment["scheduler"],
        "arms_share_training": control["training"] == assignment["training"],
        "arms_differ_only_in_matcher_coupling": (
            control_matcher_common == assignment_matcher_common
        ),
        "one_primary_query_set": int(structured["num_instances"]) == 32
        and int(structured["num_groups"]) == 1
        and not structured.get("training_auxiliary_group_sizes"),
        "bounded_delta_geometry": structured["row_reference"]["prediction_mode"]
        == "bounded_delta",
        "geometry_reference_detached": bool(
            structured["row_reference"]["detach_between_layers"]
        ),
        "ownership_enabled": bool(ownership["enabled"]),
        "ownership_geometry_protected": bool(
            ownership["detach_geometry_inputs"]
        )
        and bool(structured["lane_state"]["detach_score_geometry"]),
        "ownership_semantics_each_layer": bool(
            ownership["semantic_context"]["enabled"]
        )
        and tuple(ownership["semantic_context"]["scales"]) == ("p4", "p5"),
        "posthoc_selector_disabled": not bool(
            structured.get("set_selection", {}).get("enabled", False)
        ),
        "fresh_layer_local_hungarian": not bool(
            control_matcher["reuse_final_assignment_for_intermediate"]
        ),
        "direct_binary_ownership": loss["exist_target_mode"] == "binary"
        and loss["exist_loss_type"] == "ce"
        and float(loss["w_exist"]) > 0.0
        and float(loss["w_intermediate_exist"]) > 0.0,
        "quality_pointer_shortcuts_off": float(loss["w_quality"]) == 0.0
        and float(loss["w_set_selection"]) == 0.0
        and float(loss["w_pointer_selection"]) == 0.0,
        "ownership_optimizer_group": "ownership" in groups
        and abs(float(groups["ownership"]["lr"]) - 1e-4) < 1e-12,
        "control_has_no_assignment_edge": float(
            control_matcher["lambda_obj_end"]
        )
        == 0.0,
        "assignment_has_warmup_ramp": float(
            assignment_matcher["lambda_obj_start"]
        )
        == 0.0
        and float(assignment_matcher["lambda_obj_end"]) == 0.25
        and int(assignment_matcher["lambda_obj_ramp_start_iter"]) == 10000
        and int(assignment_matcher["lambda_obj_ramp_end_iter"]) == 25000,
        "from_scratch_25k_gate": int(control["training"]["max_iters"]) == 25000
        and int(control["scheduler"]["total_iters"]) == 278000,
    }


def main() -> None:
    args = parse_args()
    control = load_config(args.control_config)
    assignment = load_config(args.assignment_config)
    checks = config_checks(control, assignment)

    model = build_model(control)
    head = model.structured_query_head
    model_contract = {
        "ownership_layer_count": len(head.ownership_layers),
        "geometry_layer_count": len(head.layers),
        "ownership_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(
                (
                    "structured_query_head.ownership_",
                    "structured_query_head.exist.",
                    "structured_query_head.decision_norm.",
                )
            )
        ),
        "set_selection_head_is_none": head.set_selection_head is None,
    }
    model_contract["passed"] = bool(
        model_contract["ownership_layer_count"]
        == model_contract["geometry_layer_count"]
        and model_contract["ownership_layer_count"] > 0
        and model_contract["ownership_parameter_count"] > 0
        and model_contract["set_selection_head_is_none"]
    )
    del model

    control_matcher = build_matcher(control)
    assignment_matcher = build_matcher(assignment)
    schedule_iterations = (0, 9999, 10000, 17500, 25000)
    schedules = {
        "iterations": list(schedule_iterations),
        "control": [
            control_matcher.effective_lambda_obj(value)
            for value in schedule_iterations
        ],
        "assignment": [
            assignment_matcher.effective_lambda_obj(value)
            for value in schedule_iterations
        ],
    }
    schedules["passed"] = bool(
        schedules["control"] == [0.0] * len(schedule_iterations)
        and schedules["assignment"] == [0.0, 0.0, 0.0, 0.125, 0.25]
    )

    gradient = gradient_boundary_audit()
    intervention = matcher_intervention_audit(control, assignment)
    passed = bool(
        all(checks.values())
        and model_contract["passed"]
        and schedules["passed"]
        and gradient["passed"]
        and intervention["passed"]
    )
    payload = {
        "experiment": "V5 protected dual-state ownership gate",
        "control_config": str(Path(args.control_config)),
        "assignment_config": str(Path(args.assignment_config)),
        "checks": checks,
        "model_contract": model_contract,
        "matcher_schedule": schedules,
        "gradient_boundary": gradient,
        "assignment_intervention": intervention,
        "passed": passed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("V5 protected ownership contract audit failed")


if __name__ == "__main__":
    main()
