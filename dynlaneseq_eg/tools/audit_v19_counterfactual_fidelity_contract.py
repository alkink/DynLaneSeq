from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
from typing import Any

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


MODULE_PREFIX = (
    "structured_query_head.set_selection_head.counterfactual_fidelity"
)
FORBIDDEN_FORWARD_ARGUMENTS = {
    "targets",
    "target",
    "gt",
    "matches",
    "assignment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen-V7 V19 parity and gradient isolation."
    )
    parser.add_argument("--config", required=True)
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


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
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


def _norm(values: list[torch.Tensor | None]) -> float:
    return math.sqrt(
        sum(
            float(value.detach().float().square().sum())
            for value in values
            if value is not None
        )
    )


def _gradient_groups(
    loss: torch.Tensor,
    named: list[tuple[str, torch.nn.Parameter]],
) -> tuple[dict[str, float], dict[str, bool]]:
    gradients = torch.autograd.grad(
        loss,
        [parameter for _name, parameter in named],
        allow_unused=True,
    )
    groups: dict[str, list[torch.Tensor | None]] = {
        "quality_output": [],
        "intra": [],
        "visual": [],
        "proposal_slot_geometry": [],
        "other_v19": [],
    }
    for (name, _parameter), gradient in zip(named, gradients):
        if ".quality_output." in name:
            group = "quality_output"
        elif ".intra." in name:
            group = "intra"
        elif any(
            token in name
            for token in (
                ".scale_",
                ".visual_",
                ".offset_",
            )
        ):
            group = "visual"
        elif any(
            token in name
            for token in (
                ".proposal_",
                ".slot_",
                ".geometry_",
                ".row_position_",
                ".fusion",
            )
        ):
            group = "proposal_slot_geometry"
        else:
            group = "other_v19"
        groups[group].append(gradient)
    norms = {name: _norm(values) for name, values in groups.items()}
    finite = {
        name: all(
            value is None or bool(torch.isfinite(value).all())
            for value in values
        )
        for name, values in groups.items()
    }
    return norms, finite


def main() -> None:
    args = parse_args()
    cfg = _configure(args.config, args)
    source_cfg = _configure(args.source_config, args)
    seed = int(cfg.get("training", {}).get("seed", 3407))
    seed_everything(seed)
    device = torch.device(args.device)

    # The initialized V19 model must contain every mature V7 state exactly.
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
    source_state_exact = (
        len(shared_names) == len(source_state)
        and all(
            torch.equal(source_state[name], model_state[name])
            for name in shared_names
        )
    )
    del source_model, source_state, model_state

    model = model.to(device)
    train_cfg = cfg.get("training", {})
    parameter_prefixes = tuple(
        str(value)
        for value in train_cfg.get("trainable_parameter_prefixes", ())
    )
    module_prefixes = tuple(
        str(value)
        for value in train_cfg.get("trainable_module_prefixes", ())
    )
    freeze_stats = freeze_except_parameter_prefixes(model, parameter_prefixes)
    set_frozen_detector_eval(model, module_prefixes)
    selector = model.structured_query_head.set_selection_head
    module = selector.counterfactual_fidelity
    if module is None:
        raise ValueError("V19 fidelity module is absent")
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    only_v19_trainable = bool(named) and all(
        name == MODULE_PREFIX or name.startswith(MODULE_PREFIX + ".")
        for name, _parameter in named
    )
    optimizer_parameter_ids = {id(parameter) for _name, parameter in named}
    optimizer_isolated = optimizer_parameter_ids == {
        id(parameter)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    frozen_buffers_before = {
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers()
        if not (name == MODULE_PREFIX or name.startswith(MODULE_PREFIX + "."))
    }

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

    # Same-loaded-model comparison isolates V19 from checkpoint or kernel
    # provenance differences.
    selector.counterfactual_fidelity = None
    with torch.no_grad():
        v7_outputs, _ = forward_with_matches(
            model, images, targets, matcher, cfg, iteration
        )
    selector.counterfactual_fidelity = module
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
    public_parity = {
        name: _difference(v7_outputs[name], outputs[name])
        for name in parity_names
    }
    internal_pairs = {
        "route_logits": (
            "selection_slot_v19_v7_real_route_logits",
            "selection_slot_real_route_logits",
        ),
        "geometry_indices": (
            "selection_slot_v19_v7_geometry_route_indices",
            "selection_slot_geometry_route_indices",
        ),
        "deployment_indices": (
            "selection_slot_v19_v7_indices",
            "selection_slot_indices",
        ),
        "deployment_scores": (
            "selection_slot_v19_v7_scores",
            "selection_slot_scores",
        ),
        "activity_logits": (
            "selection_slot_v19_v7_active_logits",
            "selection_slot_active_logits",
        ),
        "activity": (
            "selection_slot_v19_v7_active",
            "selection_slot_active",
        ),
    }
    same_forward_parity = {
        name: _difference(outputs[left], outputs[right])
        for name, (left, right) in internal_pairs.items()
    }
    with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
        left_paths = write_culane_predictions(
            v7_outputs, metas, left, **_writer_kwargs(cfg)
        )
        right_paths = write_culane_predictions(
            outputs, metas, right, **_writer_kwargs(cfg)
        )
        left_bytes = {
            path.relative_to(left): path.read_bytes() for path in left_paths
        }
        right_bytes = {
            path.relative_to(right): path.read_bytes() for path in right_paths
        }
        writer_exact = left_bytes == right_bytes

    v19_finite = all(
        bool(torch.isfinite(value).all())
        for name, value in outputs.items()
        if name.startswith("selection_slot_v19_")
        and isinstance(value, torch.Tensor)
        and value.dtype.is_floating_point
    )
    neutral = {
        "quality_weight": float(module.quality_output.weight.abs().max()),
        "quality_bias": float(module.quality_output.bias.abs().max()),
        "fidelity_delta": float(
            outputs["selection_slot_v19_fidelity_delta"].abs().max()
        ),
        "p50_from_half": float(
            (outputs["selection_slot_v19_p50"] - 0.5).abs().max()
        ),
        "p75_from_half": float(
            (outputs["selection_slot_v19_p75"] - 0.5).abs().max()
        ),
        "iou_from_half": float(
            (outputs["selection_slot_v19_expected_iou"] - 0.5).abs().max()
        ),
    }
    shape_contract = tuple(
        outputs["selection_slot_v19_quality_logits"].shape[1:]
    ) == (4, 32, 3)

    zero_norms, zero_finite = _gradient_groups(
        losses["loss_four_slot_v19"], named
    )
    del outputs, losses

    # Exact parity zero-initialization necessarily delays trunk gradients by
    # one update. An unsaved tiny output perturbation proves every intended
    # intra/visual/proposal-state edge is reachable thereafter.
    module_backup = {
        name: value.detach().clone() for name, value in module.state_dict().items()
    }
    with torch.no_grad():
        module.quality_output.weight.normal_(std=1.0e-4)
    perturbed_outputs, perturbed_matches = forward_with_matches(
        model, images, targets, matcher, cfg, iteration
    )
    perturbed_losses = criterion(perturbed_outputs, targets, perturbed_matches)
    perturbed_norms, perturbed_finite = _gradient_groups(
        perturbed_losses["loss_four_slot_v19"], named
    )
    module.load_state_dict(module_backup, strict=True)

    frozen_buffers_after = {
        name: value.detach().cpu()
        for name, value in model.named_buffers()
        if name in frozen_buffers_before
    }
    frozen_buffers_exact = all(
        torch.equal(value, frozen_buffers_after[name])
        for name, value in frozen_buffers_before.items()
    )
    target_free_forward = not bool(
        FORBIDDEN_FORWARD_ARGUMENTS.intersection(
            inspect.signature(module.forward).parameters
        )
    )
    config_loss = cfg.get("loss", {})
    old_losses_zero = all(
        float(config_loss.get(name, 0.0)) == 0.0
        for name in (
            "w_exist",
            "w_point",
            "w_range",
            "w_line_iou",
            "w_seg",
            "w_centerline",
            "w_four_slot_selection",
            "w_four_slot_geometry",
            "w_four_slot_unified",
            "w_four_slot_visual_first",
            "w_four_slot_visual_precision",
            "w_four_slot_v14_stage_a",
            "w_four_slot_v14_stage_b",
            "w_four_slot_v15",
            "w_four_slot_v16",
            "w_four_slot_v17",
            "w_four_slot_v18",
        )
    )
    config_pass = all(
        (
            bool(
                cfg["model"]["structured_query"]["set_selection"].get(
                    "four_slot_counterfactual_fidelity_enabled", False
                )
            ),
            float(config_loss.get("w_four_slot_v19", 0.0)) == 1.0,
            old_losses_zero,
            bool(train_cfg.get("frozen_detector_eval", False)),
            tuple(module.scale_names) == ("p2", "p4", "p5"),
            not any("inter" in name.lower() for name, _ in module.named_modules()),
        )
    )
    parity_pass = (
        all(value == 0.0 for value in same_forward_parity.values())
        and writer_exact
    )
    zero_gradient_pass = (
        zero_norms["quality_output"] > 0.0
        and all(
            zero_norms[name] == 0.0
            for name in (
                "intra",
                "visual",
                "proposal_slot_geometry",
            )
        )
        and all(zero_finite.values())
    )
    perturbed_gradient_pass = (
        all(
            perturbed_norms[name] > 0.0
            for name in (
                "quality_output",
                "intra",
                "visual",
                "proposal_slot_geometry",
            )
        )
        and all(perturbed_finite.values())
    )
    passed = all(
        (
            source_iteration == int(args.start_iteration),
            iteration == int(args.start_iteration),
            source_state_exact,
            only_v19_trainable,
            optimizer_isolated,
            frozen_buffers_exact,
            config_pass,
            parity_pass,
            shape_contract,
            v19_finite,
            target_free_forward,
            all(value == 0.0 for value in neutral.values()),
            bool(torch.isfinite(perturbed_losses["loss_total"])),
            zero_gradient_pass,
            perturbed_gradient_pass,
        )
    )

    report = {
        "version": "v19_frozen_counterfactual_fidelity_gate0",
        "passed": bool(passed),
        "training_authorized": bool(passed),
        "long_training_authorized": False,
        "source_iteration": source_iteration,
        "checkpoint_iteration": iteration,
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "config_sha256": _sha256(args.config),
        "source_state_exact": source_state_exact,
        "frozen_buffers_exact_after_forwards": frozen_buffers_exact,
        "freeze_stats": freeze_stats,
        "only_v19_trainable": only_v19_trainable,
        "optimizer_parameter_set_isolated": optimizer_isolated,
        "config_contract": {
            "passed": config_pass,
            "old_losses_zero": old_losses_zero,
            "candidate_interaction": "none",
            "image_scales": list(module.scale_names),
            "target_semantics": (
                "online detached range-aware row-strip IoU surrogate; "
                "fixed deployment gate uses exact official raster IoU"
            ),
            "exact_official_raster_target_cache": False,
        },
        "initialization": {
            "neutral_max_abs": neutral,
            "same_forward_v7_parity": same_forward_parity,
            "public_cross_forward_difference": public_parity,
            "writer_bytes_exact": writer_exact,
            "counterfactual_shape": list(
                perturbed_outputs[
                    "selection_slot_v19_quality_logits"
                ].shape
            ),
            "shape_contract_passed": shape_contract,
            "passed": parity_pass,
        },
        "graph": {
            "target_free_inference_forward": target_free_forward,
            "v19_tensors_finite": v19_finite,
            "zero_head_phase": {
                "gradient_norms": zero_norms,
                "finite": zero_finite,
                "passed": zero_gradient_pass,
            },
            "unsaved_output_perturbation_phase": {
                "std": 1.0e-4,
                "gradient_norms": perturbed_norms,
                "finite": perturbed_finite,
                "passed": perturbed_gradient_pass,
                "checkpoint_restored_before_exit": True,
            },
        },
        "diagnostics": {
            "loss_total": float(
                perturbed_losses["loss_total"].detach().cpu()
            ),
            "loss_p50": float(
                perturbed_losses["loss_four_slot_v19_p50"].detach().cpu()
            ),
            "loss_p75": float(
                perturbed_losses["loss_four_slot_v19_p75"].detach().cpu()
            ),
            "loss_iou": float(
                perturbed_losses["loss_four_slot_v19_iou"].detach().cpu()
            ),
            "loss_rank": float(
                perturbed_losses["loss_four_slot_v19_rank"].detach().cpu()
            ),
        },
        "test_set_used": False,
    }
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
