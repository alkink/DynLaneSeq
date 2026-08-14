from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.engine.frozen_training import (
    freeze_except_parameter_prefixes,
    set_frozen_detector_eval,
)
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.train import seed_everything


MODULE_PREFIX = "structured_query_head.set_selection_head.joint_exact_set_energy"
FORBIDDEN_FORWARD_ARGUMENTS = {
    "targets",
    "target",
    "gt",
    "gt_x",
    "gt_valid",
    "matches",
    "assignment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V18 exact-set parity, finite masking and the two-phase "
            "treatment/control gradient contract."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start-iteration", type=int, default=225000)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _configure(path: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.dtype != right.dtype:
        return float("inf")
    if left.dtype == torch.bool or not left.dtype.is_floating_point:
        return float((left != right).sum().detach().cpu())
    if left.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().detach().cpu())


def _writer_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    post = cfg.get("postprocess", {})
    return {
        "score_thresh": float(post.get("score_thresh", 0.0)),
        "min_pred_points": int(post.get("min_pred_points", 5)),
        "nms_distance_thresh_px": float(
            post.get("lane_nms_distance_thresh_px", 0.0)
        ),
        "nms_min_overlap_points": int(
            post.get("lane_nms_min_overlap_points", 5)
        ),
        "top_k": int(post.get("top_k", 4)),
        "row_visibility_thresh": float(
            post.get("row_visibility_thresh", 0.0)
        ),
        "quality_score_power": float(post.get("quality_score_power", 0.0)),
        "score_mode": str(post.get("score_mode", "four_slot")),
    }


def _parameter_norm(
    gradients: Iterable[torch.Tensor | None],
) -> float:
    value = 0.0
    for gradient in gradients:
        if gradient is not None:
            value += float(gradient.detach().float().square().sum())
    return math.sqrt(value)


def _group_gradients(
    loss: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool,
) -> tuple[dict[str, float], dict[str, bool]]:
    parameters = [parameter for _name, parameter in named_parameters]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    groups: dict[str, list[torch.Tensor | None]] = {
        "set_output_heads": [],
        "association": [],
        "refiner": [],
        "shared_image": [],
        "proposal_coordinate_heads": [],
        "other": [],
    }
    for (name, _parameter), gradient in zip(named_parameters, gradients):
        if name.startswith(MODULE_PREFIX + ".unary_interaction.2") or name.startswith(
            MODULE_PREFIX + ".pair_output"
        ):
            group = "set_output_heads"
        elif name.startswith(MODULE_PREFIX + ".association"):
            group = "association"
        elif name.startswith(MODULE_PREFIX + ".refiner"):
            group = "refiner"
        elif name.startswith(
            ("encoder.backbone.", "encoder.fpn.", "encoder.proj.", "encoder.ms_proj.")
        ):
            group = "shared_image"
        elif (
            not name.startswith(MODULE_PREFIX)
            and any(token in name for token in ("row_x", "range_head", "pred_x"))
        ):
            group = "proposal_coordinate_heads"
        else:
            group = "other"
        groups[group].append(gradient)
    norms = {name: _parameter_norm(values) for name, values in groups.items()}
    finite = {
        name: all(
            gradient is None or bool(torch.isfinite(gradient).all())
            for gradient in values
        )
        for name, values in groups.items()
    }
    return norms, finite


def _zero_head_contract(module: torch.nn.Module) -> dict[str, float]:
    names = {
        "unary": module.unary_interaction[-1].weight,
        "pair": module.pair_output.weight,
        "delta": module.refiner.delta_head.weight,
        "range": module.refiner.range_head.weight,
        "policy_weight": module.refiner.policy_head.weight,
        "policy_bias": module.refiner.policy_head.bias,
        "activity_weight": module.refiner.activity_head.weight,
        "activity_bias": module.refiner.activity_head.bias,
    }
    return {
        name: float(value.detach().float().abs().max().cpu())
        for name, value in names.items()
    }


def main() -> None:
    args = parse_args()
    cfg = _configure(args.config, args)
    control_cfg = _configure(args.control_config, args)
    source_cfg = _configure(args.source_config, args)
    seed = int(cfg.get("training", {}).get("seed", 3407))
    seed_everything(seed)
    device = torch.device(args.device)

    treatment_selection = cfg["model"]["structured_query"]["set_selection"]
    control_selection = control_cfg["model"]["structured_query"]["set_selection"]
    treatment_enabled = bool(
        treatment_selection.get("four_slot_joint_exact_set_energy_enabled", False)
    )
    only_causal_difference = all(
        (
            key == "four_slot_joint_exact_set_energy_detach_association_for_set_loss"
            or treatment_selection.get(key) == control_selection.get(key)
        )
        for key in set(treatment_selection) | set(control_selection)
        if key.startswith("four_slot_joint_exact_set_energy")
    ) and (
        treatment_selection.get(
            "four_slot_joint_exact_set_energy_detach_association_for_set_loss"
        )
        is False
        and control_selection.get(
            "four_slot_joint_exact_set_energy_detach_association_for_set_loss"
        )
        is True
    )
    loss_cfg = cfg.get("loss", {})
    auxiliary_zero = all(
        float(loss_cfg.get(name, 0.0)) == 0.0
        for name in (
            "lambda_coarse",
            "lambda_geometry_draft",
            "lambda_intermediate",
            "lambda_training_auxiliary",
        )
    )
    old_v_losses_zero = all(
        float(loss_cfg.get(f"w_four_slot_v{version}", 0.0)) == 0.0
        for version in range(11, 18)
    ) and all(
        float(loss_cfg.get(name, 0.0)) == 0.0
        for name in (
            "w_four_slot_selection",
            "w_four_slot_geometry",
            "w_four_slot_unified",
            "w_four_slot_visual_first",
            "w_four_slot_visual_precision",
        )
    )

    # Exact state comparison catches a wrong base checkpoint even when one
    # sampled forward happens to look plausible.
    source_model = build_model(source_cfg)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source_model, strict=False)
    )
    source_state = source_model.state_dict()

    model = build_model(cfg)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model_state = model.state_dict()
    shared_names = [
        name
        for name, value in source_state.items()
        if name in model_state and model_state[name].shape == value.shape
    ]
    source_state_exact = len(shared_names) == len(source_state) and all(
        torch.equal(source_state[name], model_state[name]) for name in shared_names
    )
    del source_model, source_state, model_state

    model = model.to(device)
    trainable_prefixes = tuple(
        str(value)
        for value in cfg.get("training", {}).get(
            "trainable_parameter_prefixes", ()
        )
    )
    freeze_stats = freeze_except_parameter_prefixes(model, trainable_prefixes)
    set_frozen_detector_eval(model, trainable_prefixes)
    selector = model.structured_query_head.set_selection_head
    module = selector.joint_exact_set_energy
    if module is None:
        raise ValueError("V18 exact-set module is absent")
    zero_heads = _zero_head_contract(module)
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    loader = build_dataloader(
        cfg,
        split="train",
        training=True,
        start_iteration=int(args.start_iteration),
    )
    images, targets, metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(iteration)

    # Disable only V18 on the already-loaded model.  This is the strongest
    # parity comparison because all mature V7 parameters and input bytes are
    # literally shared by the two forwards.
    selector.joint_exact_set_energy = None
    with torch.no_grad():
        source_outputs, _ = forward_with_matches(
            model, images, targets, matcher, cfg, iteration
        )
    selector.joint_exact_set_energy = module
    outputs, matches = forward_with_matches(
        model, images, targets, matcher, cfg, iteration
    )
    losses = criterion(outputs, targets, matches)

    parity_names = (
        "selection_slot_real_route_logits",
        "selection_slot_geometry_route_indices",
        "selection_slot_indices",
        "selection_slot_scores",
        "selection_slot_active_logits",
        "selection_slot_active",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    )
    parity = {
        name: _tensor_max_difference(source_outputs[name], outputs[name])
        for name in parity_names
    }
    internal_parity_pairs = {
        "real_route_logits": (
            "selection_slot_v18_v7_real_route_logits",
            "selection_slot_real_route_logits",
        ),
        "v18_zero_unary": (
            "selection_slot_v18_v7_real_route_logits",
            "selection_slot_v18_unary",
        ),
        "geometry_route_indices": (
            "selection_slot_v18_v7_geometry_route_indices",
            "selection_slot_geometry_route_indices",
        ),
        "deployment_indices": (
            "selection_slot_v18_v7_indices",
            "selection_slot_indices",
        ),
        "deployment_scores": (
            "selection_slot_v18_v7_scores",
            "selection_slot_scores",
        ),
        "activity_logits": (
            "selection_slot_v18_v7_active_logits",
            "selection_slot_active_logits",
        ),
        "activity_mask": (
            "selection_slot_v18_v7_active",
            "selection_slot_active",
        ),
        "final_x": (
            "selection_slot_v18_anchor_x_rows",
            "selection_slot_pred_x_rows",
        ),
        "final_range": (
            "selection_slot_v18_anchor_range_norm",
            "selection_slot_range_norm",
        ),
    }
    same_forward_parity = {
        name: _tensor_max_difference(outputs[left], outputs[right])
        for name, (left, right) in internal_parity_pairs.items()
    }
    with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
        left_paths = write_culane_predictions(
            source_outputs, metas, left_dir, **_writer_kwargs(cfg)
        )
        right_paths = write_culane_predictions(
            outputs, metas, right_dir, **_writer_kwargs(cfg)
        )
        left_files = {
            path.relative_to(left_dir): path.read_bytes() for path in left_paths
        }
        right_files = {
            path.relative_to(right_dir): path.read_bytes() for path in right_paths
        }
        writer_exact = left_files == right_files

    v18_tensors_finite = all(
        bool(torch.isfinite(value).all())
        for name, value in outputs.items()
        if name.startswith("selection_slot_v18_")
        and isinstance(value, torch.Tensor)
        and value.dtype.is_floating_point
    )
    route_indices = outputs["selection_slot_geometry_route_indices"]
    no_repeated_ids = all(
        len(set(int(value) for value in row.tolist() if int(value) >= 0))
        == int((row >= 0).sum())
        for row in route_indices.detach().cpu()
    )
    exact_table = {
        "physical_sets": int(module.combination_table.shape[0]),
        "permutations": int(module.permutation_table.shape[0]),
        "ordered_assignments": int(module.ordered_assignments.numel() // 4),
    }
    target_free_forward = not bool(
        FORBIDDEN_FORWARD_ARGUMENTS.intersection(
            inspect.signature(module.route).parameters
        )
        or FORBIDDEN_FORWARD_ARGUMENTS.intersection(
            inspect.signature(module.refine).parameters
        )
    )

    zero_norms, zero_finite = _group_gradients(
        losses["loss_four_slot_v18_set"],
        named_parameters,
        retain_graph=False,
    )
    del outputs, losses

    # Exact parity needs zero output heads, which necessarily makes upstream
    # set gradients zero.  The second phase uses an unsaved 1e-4 perturbation
    # to prove the intended edge exists once those heads take their first
    # optimizer update.  The checkpoint is restored before exit.
    unary_backup = module.unary_interaction[-1].weight.detach().clone()
    pair_backup = module.pair_output.weight.detach().clone()
    with torch.no_grad():
        module.unary_interaction[-1].weight.normal_(std=1.0e-4)
        module.pair_output.weight.normal_(std=1.0e-4)
    module.detach_association_for_set_loss = False
    treatment_outputs, treatment_matches = forward_with_matches(
        model, images, targets, matcher, cfg, iteration
    )
    treatment_losses = criterion(treatment_outputs, targets, treatment_matches)
    treatment_norms, treatment_finite = _group_gradients(
        treatment_losses["loss_four_slot_v18_set"],
        named_parameters,
        retain_graph=False,
    )

    module.detach_association_for_set_loss = True
    control_outputs, control_matches = forward_with_matches(
        model, images, targets, matcher, control_cfg, iteration
    )
    control_losses = criterion(control_outputs, targets, control_matches)
    control_norms, control_finite = _group_gradients(
        control_losses["loss_four_slot_v18_set"],
        named_parameters,
        retain_graph=False,
    )
    control_forward_names = (
        "selection_slot_v18_unordered_set_scores",
        "selection_slot_geometry_route_indices",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
        "selection_slot_active_logits",
        "selection_slot_scores",
    )
    treatment_control_parity = {
        name: _tensor_max_difference(
            treatment_outputs[name], control_outputs[name]
        )
        for name in control_forward_names
    }
    with torch.no_grad():
        module.unary_interaction[-1].weight.copy_(unary_backup)
        module.pair_output.weight.copy_(pair_backup)
    module.detach_association_for_set_loss = False

    total_finite = bool(torch.isfinite(treatment_losses["loss_total"])) and bool(
        torch.isfinite(control_losses["loss_total"])
    )
    zero_phase_pass = (
        zero_norms["set_output_heads"] > 0.0
        and zero_norms["association"] == 0.0
        and zero_norms["shared_image"] == 0.0
        and all(zero_finite.values())
    )
    perturbed_phase_pass = (
        treatment_norms["association"] > 0.0
        and treatment_norms["shared_image"] > 0.0
        and control_norms["association"] == 0.0
        and control_norms["shared_image"] == 0.0
        and all(treatment_finite.values())
        and all(control_finite.values())
    )
    # A second full CPU/GPU replay may differ by sub-ulp reduction noise even
    # before V18 executes.  The causal contract is therefore checked against
    # V7 tensors captured earlier in the *same* forward, while independent
    # writer-byte replay remains exact.  Cross-forward tensor differences are
    # still reported and must never change the serialized predictions.
    parity_pass = (
        all(value == 0.0 for value in same_forward_parity.values())
        and writer_exact
    )
    control_parity_pass = all(
        value == 0.0 for value in treatment_control_parity.values()
    )
    config_pass = (
        treatment_enabled
        and only_causal_difference
        and auxiliary_zero
        and old_v_losses_zero
        and float(loss_cfg.get("w_four_slot_v18", 0.0)) == 1.0
        and bool(
            cfg.get("training", {})
            .get("v18_gradient_conflict_projection", {})
            .get("enabled", False)
        )
    )
    exact_search_pass = exact_table == {
        "physical_sets": 35960,
        "permutations": 24,
        "ordered_assignments": 863040,
    }
    passed = all(
        (
            source_iteration == int(args.start_iteration),
            iteration == int(args.start_iteration),
            source_state_exact,
            config_pass,
            parity_pass,
            control_parity_pass,
            exact_search_pass,
            no_repeated_ids,
            target_free_forward,
            v18_tensors_finite,
            total_finite,
            zero_phase_pass,
            perturbed_phase_pass,
            all(value == 0.0 for value in zero_heads.values()),
        )
    )

    report = {
        "version": "v18_joint_exact_set_energy_gate0",
        "passed": bool(passed),
        "long_training_authorized": False,
        "source_iteration": source_iteration,
        "checkpoint_iteration": iteration,
        "source_checkpoint_sha256": _file_sha256(args.source_checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "config_sha256": _file_sha256(args.config),
        "control_config_sha256": _file_sha256(args.control_config),
        "source_state_exact": source_state_exact,
        "freeze_stats": freeze_stats,
        "trainable_parameter_count": sum(
            parameter.numel() for _name, parameter in named_parameters
        ),
        "config_contract": {
            "passed": config_pass,
            "treatment_enabled": treatment_enabled,
            "only_treatment_control_difference": only_causal_difference,
            "auxiliary_zero": auxiliary_zero,
            "old_v_losses_zero": old_v_losses_zero,
            "conflict_projection_enabled": bool(
                cfg.get("training", {})
                .get("v18_gradient_conflict_projection", {})
                .get("enabled", False)
            ),
        },
        "initialization": {
            "zero_heads_max_abs": zero_heads,
            "same_forward_v7_parity": same_forward_parity,
            "public_tensor_max_abs_or_mismatch": parity,
            "writer_bytes_exact": writer_exact,
            "passed": parity_pass,
        },
        "exact_search": {
            **exact_table,
            "passed": exact_search_pass,
            "no_repeated_ids": no_repeated_ids,
        },
        "graph": {
            "target_free_inference_forward": target_free_forward,
            "v18_tensors_finite": v18_tensors_finite,
            "losses_finite": total_finite,
            "zero_phase": {
                "gradient_norms": zero_norms,
                "finite": zero_finite,
                "expected": (
                    "output heads nonzero; association/shared image exactly zero"
                ),
                "passed": zero_phase_pass,
            },
            "unsaved_head_perturbation_phase": {
                "std": 1.0e-4,
                "treatment_gradient_norms": treatment_norms,
                "control_gradient_norms": control_norms,
                "treatment_finite": treatment_finite,
                "control_finite": control_finite,
                "treatment_control_forward_difference": treatment_control_parity,
                "forward_parity_passed": control_parity_pass,
                "passed": perturbed_phase_pass,
                "checkpoint_restored_before_exit": True,
            },
        },
        "diagnostics": {
            "zero_step_set_loss": float(
                control_losses["loss_four_slot_v18_set"].detach().cpu()
            ),
            "target_entropy": float(
                control_losses["four_slot_v18_target_entropy"].detach().cpu()
            ),
            "target_support_size": float(
                control_losses["four_slot_v18_target_support_size"].detach().cpu()
            ),
            "chosen_set_regret": float(
                control_losses["four_slot_v18_chosen_set_regret"].detach().cpu()
            ),
        },
    }
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
