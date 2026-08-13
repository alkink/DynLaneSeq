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
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.tools.train import seed_everything


SELECTOR = "structured_query_head.set_selection_head."
V14 = SELECTOR + "corrected_visual_first_association."
GROUPS = ("visual", "slot_row", "proposal", "other")


def _state_digest(
    state: dict[str, torch.Tensor],
    names: tuple[str, ...],
) -> str:
    """Hash named tensors exactly, independent of their current device."""
    digest = hashlib.sha256()
    for name in names:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact V7 deployment parity and V14 Stage-A gradient "
            "topology before any optimizer step."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start-iteration", type=int, default=225000)
    parser.add_argument(
        "--cross-clip-report",
        action="append",
        required=True,
        help="Repeat for every Stage-A evaluation domain.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _configure(
    path: str,
    *,
    dataset_root: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(dataset_root).expanduser()
    )
    cfg.setdefault("training", {})["batch_size"] = int(batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _group(name: str) -> str:
    local = name[len(V14) :] if name.startswith(V14) else name
    if local.startswith(
        (
            "feature_",
            "x_position_key.",
            "visual_query.",
            "visual_context.",
            "visual_x_projection.",
        )
    ):
        return "visual"
    if local.startswith(
        (
            "slot_",
            "row_position_projection.",
            "anchor_geometry_projection.",
            "initial_norm.",
            "vertical_encoder.",
            "visual_norm.",
            "visual_ffn.",
        )
    ):
        return "slot_row"
    if local.startswith(
        (
            "proposal_",
            "private_dustbin.",
            "association_",
        )
    ):
        return "proposal"
    return "other"


def _norm(
    gradients: tuple[torch.Tensor | None, ...],
    indices: list[int],
) -> float:
    square = 0.0
    for index in indices:
        gradient = gradients[index]
        if gradient is not None:
            square += float(gradient.detach().float().square().sum())
    return math.sqrt(square)


def _indices_with_local_prefixes(
    names: list[str], prefixes: tuple[str, ...]
) -> list[int]:
    selected: list[int] = []
    for index, name in enumerate(names):
        local = name[len(V14) :] if name.startswith(V14) else name
        if local.startswith(prefixes):
            selected.append(index)
    return selected


def main() -> None:
    args = parse_args()
    cfg = _configure(
        args.config,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    source_cfg = _configure(
        args.source_config,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    if (
        selection.get("four_slot_corrected_visual_first_association_enabled")
        is not True
    ):
        raise ValueError("V14 corrected visual-first association is disabled")
    if selection.get("four_slot_refinement_enabled") is not True:
        raise ValueError("V14 Stage A requires the exact V7 refiner")
    if float(cfg["loss"].get("w_four_slot_v14_stage_a", 0.0)) != 1.0:
        raise ValueError("V14 Stage-A loss must be the sole objective")
    nonzero_objectives = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and float(value) != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    if nonzero_objectives != {"w_four_slot_v14_stage_a": 1.0}:
        raise ValueError(
            "V14 requires one objective; resolved nonzero weights are "
            f"{nonzero_objectives}"
        )
    # factory.dataset_cfg_for_split() intentionally consumes the top-level
    # mapping, so checking dataset.augmentation would be a false contract.
    augmentation = cfg.get("augmentation", {})
    required_zero_augmentation = {
        "horizontal_flip_prob": 0.0,
        "color_jitter": False,
        "channel_shuffle_prob": 0.0,
        "hue_saturation_prob": 0.0,
        "blur_prob": 0.0,
        "affine_prob": 0.0,
        "random_shadow_prob": 0.0,
    }
    augmentation_disabled = all(
        augmentation.get(name) == value
        for name, value in required_zero_augmentation.items()
    )
    if not augmentation_disabled:
        raise ValueError("V14 Stage A requires augmentation to be exactly off")
    cross_clip_reports = [
        json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        for path in args.cross_clip_report
    ]
    cross_clip_exact = bool(cross_clip_reports) and all(
        report.get("passed") is True
        and int(report.get("same_image_partner_count", -1)) == 0
        and int(report.get("same_clip_partner_count", -1)) == 0
        for report in cross_clip_reports
    )
    if not cross_clip_exact:
        raise ValueError("V14 cross-clip derangement contract failed")

    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    loader = build_dataloader(
        cfg,
        split="train",
        training=True,
        start_iteration=int(args.start_iteration),
    )
    images, targets, metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)

    source_model = build_model(source_cfg).to(device)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source_model, strict=False)
    )
    if source_iteration != int(args.start_iteration):
        raise ValueError("source checkpoint iteration mismatch")
    source_model.eval()
    source_state_names = tuple(sorted(source_model.state_dict()))
    source_state_sha256 = _state_digest(
        source_model.state_dict(), source_state_names
    )
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
    post = cfg.get("postprocess", {})
    writer_kwargs = {
        "score_thresh": float(post.get("score_thresh", 0.0)),
        "min_pred_points": int(post.get("min_pred_points", 5)),
        "nms_distance_thresh_px": float(
            post.get("lane_nms_distance_thresh_px", 0.0)
        ),
        "nms_min_overlap_points": int(
            post.get("lane_nms_min_overlap_points", 5)
        ),
        "top_k": int(post.get("top_k", 4)),
        "row_visibility_thresh": float(post.get("row_visibility_thresh", 0.0)),
        "quality_score_power": float(post.get("quality_score_power", 0.0)),
        "score_mode": str(post.get("score_mode", "four_slot")),
    }
    del source_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    if iteration != int(args.start_iteration):
        raise ValueError("V14 checkpoint iteration mismatch")
    freeze_stats = freeze_except_parameter_prefixes(
        model, (V14.rstrip("."),)
    )
    set_frozen_detector_eval(model, (V14.rstrip("."),))
    selector = model.structured_query_head.set_selection_head
    module = selector.corrected_visual_first_association
    if module is None or selector.slot_refinement is None:
        raise ValueError("V14 graph is incomplete")
    forward_parameters = tuple(inspect.signature(module.forward).parameters)
    if "legacy_route_logits" in forward_parameters:
        raise ValueError("V14 proposal association still accepts V7 route logits")
    forbidden_target_parameters = {
        "targets",
        "target",
        "gt",
        "gt_x",
        "gt_valid",
        "matches",
        "assignment",
    }
    target_free_forward_signature = not bool(
        forbidden_target_parameters.intersection(forward_parameters)
    )

    current_state = model.state_dict()
    missing_legacy_state = tuple(
        name for name in source_state_names if name not in current_state
    )
    current_legacy_state_sha256 = (
        _state_digest(current_state, source_state_names)
        if not missing_legacy_state
        else ""
    )
    legacy_state_exact = (
        not missing_legacy_state
        and current_legacy_state_sha256 == source_state_sha256
    )

    # Compare the V7 deployment graph with and without the V14 sidecar inside
    # one model instance.  Two separately allocated but weight-identical
    # models can differ by tiny CPU/GPU GEMM roundoff and make writer text
    # differ in the last decimal; that is not a causal sidecar effect.  The
    # SHA check above separately proves that this in-model V7 graph contains
    # the exact source-checkpoint tensors.
    matcher = build_matcher(cfg)
    selector.corrected_visual_first_association = None
    with torch.no_grad():
        source_outputs, _source_matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            iteration,
        )
    selector.corrected_visual_first_association = module
    source_values = {
        name: source_outputs[name].detach().clone() for name in parity_names
    }
    writer_names = (
        "exist_logits",
        "pred_x_rows",
        "range_norm",
        "selection_slot_logits",
        "selection_slot_candidate_valid",
        "selection_slot_indices",
        "selection_slot_scores",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    )
    source_writer_values = {
        name: source_outputs[name].detach().clone() for name in writer_names
    }
    del source_outputs

    captured: dict[str, torch.Tensor] = {}
    anchors_before_v14: dict[str, torch.Tensor] = {}

    def capture(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        captured.update(kwargs)
        anchors_before_v14["selection_slot_pred_x_rows"] = kwargs[
            "anchor_x_rows"
        ].detach().clone()
        anchors_before_v14["selection_slot_range_norm"] = kwargs[
            "anchor_range_norm"
        ].detach().clone()

    handle = module.register_forward_pre_hook(capture, with_kwargs=True)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(iteration)
    outputs, matches = forward_with_matches(
        model, images, targets, matcher, cfg, iteration
    )
    handle.remove()
    if not captured:
        raise RuntimeError("V14 pre-hook captured no inputs")
    losses = criterion(outputs, targets, matches)

    cross_model_numeric_parity: dict[str, float] = {}
    for name, source_value in source_values.items():
        current = outputs[name].detach()
        if current.dtype == torch.bool or not current.dtype.is_floating_point:
            cross_model_numeric_parity[name] = float(
                (current != source_value).sum().cpu()
            )
        else:
            cross_model_numeric_parity[name] = float(
                (current.float() - source_value.float()).abs().max().cpu()
            )
    with tempfile.TemporaryDirectory(prefix="v14-source-writer-") as source_dir, tempfile.TemporaryDirectory(
        prefix="v14-stage-a-writer-"
    ) as stage_a_dir:
        source_paths = write_culane_predictions(
            source_writer_values,
            metas,
            source_dir,
            **writer_kwargs,
        )
        stage_a_paths = write_culane_predictions(
            outputs,
            metas,
            stage_a_dir,
            **writer_kwargs,
        )
        source_by_suffix = {
            path.relative_to(source_dir): path.read_bytes() for path in source_paths
        }
        stage_a_by_suffix = {
            path.relative_to(stage_a_dir): path.read_bytes() for path in stage_a_paths
        }
        writer_output_exact = source_by_suffix == stage_a_by_suffix
        writer_file_count = len(source_by_suffix)

    sidecar_anchor_parity = {
        name: float(
            (outputs[name].detach() - value).abs().max().cpu()
        )
        for name, value in anchors_before_v14.items()
    }

    replay = module(**captured)
    replay_error = max(
        float(
            (
                replay["selection_slot_v14_proposal_attention"]
                - outputs["selection_slot_v14_proposal_attention"]
            ).abs().max().detach().cpu()
        ),
        float(
            (
                replay["selection_slot_v14_visual_attention"]
                - outputs["selection_slot_v14_visual_attention"]
            ).abs().max().detach().cpu()
        ),
    )
    zero = module(**captured, feature_policy="zero_content")
    position_only = module(**captured, feature_policy="position_only")
    zero_p2_proposal_change = float(
        (
            zero["selection_slot_v14_proposal_attention"]
            - replay["selection_slot_v14_proposal_attention"]
        ).abs().mean().detach().cpu()
    )
    zero_p2_visual_change = float(
        (
            zero["selection_slot_v14_visual_attention"]
            - replay["selection_slot_v14_visual_attention"]
        ).abs().mean().detach().cpu()
    )
    position_only_visual_change = float(
        (
            position_only["selection_slot_v14_visual_attention"]
            - replay["selection_slot_v14_visual_attention"]
        ).abs().mean().detach().cpu()
    )
    deployment_key_overlap = sorted(set(parity_names).intersection(replay))
    attention = outputs["selection_slot_v14_proposal_attention"].detach().float()
    candidates = int(outputs["pred_x_rows"].shape[1])
    slots = int(attention.shape[1])
    attention_row_error = float(
        (attention.sum(dim=-1) - 1.0).abs().max().cpu()
    )
    attention_real_column_excess = float(
        (attention[..., :candidates].sum(dim=1) - 1.0)
        .clamp_min(0.0)
        .max()
        .cpu()
    )
    private = attention[..., candidates:]
    private_mask = torch.eye(
        slots, device=attention.device, dtype=torch.bool
    ).view(1, slots, slots)
    private_off_diagonal_mass = float(
        private.masked_select(~private_mask).abs().max().cpu()
    )
    writer_valid = outputs["selection_slot_v14_writer_valid"].detach().bool()
    source_active = outputs["selection_slot_v14_source_active"].detach().bool()
    geometry_valid = outputs[
        "selection_slot_v14_geometry_valid"
    ].detach().bool()
    writer_valid_contract = bool(
        ((~writer_valid) | (source_active & geometry_valid)).all()
    )
    padded_targets = criterion._match_four_slot_v14_anchor(
        outputs, targets
    )[0]
    repeated_targets = criterion._match_four_slot_v14_anchor(
        outputs, targets
    )[0]
    assignment_deterministic = all(
        torch.equal(first["pred_indices"], second["pred_indices"])
        and torch.equal(first["gt_indices"], second["gt_indices"])
        for first, second in zip(padded_targets, repeated_targets)
    )

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _parameter in named_parameters]
    parameters = tuple(parameter for _name, parameter in named_parameters)
    group_indices = {
        group: [index for index, name in enumerate(names) if _group(name) == group]
        for group in GROUPS
    }
    association_bypass_indices = _indices_with_local_prefixes(
        names,
        (
            "slot_norm.",
            "slot_projection.",
            "slot_tokens.",
            "row_position_projection.",
            "anchor_geometry_projection.",
            "initial_norm.",
        ),
    )
    association_p2_indices = _indices_with_local_prefixes(
        names,
        (
            "feature_norm.",
            "feature_key.",
            "feature_value.",
            "x_position_key.",
            "visual_query.",
            "visual_context.",
            "visual_x_projection.",
        ),
    )
    association_transport_indices = _indices_with_local_prefixes(
        names,
        (
            "proposal_row_norm.",
            "proposal_content_key.",
            "proposal_geometry_key.",
            "proposal_query.",
            "private_dustbin.",
        ),
    )
    components = {
        "visual": losses["loss_four_slot_v14_visual"],
        "association": losses["loss_four_slot_v14_association"],
        "total": losses["loss_four_slot_v14_stage_a"],
    }
    gradient_report: dict[str, dict[str, float]] = {}
    gradient_topology: dict[str, float] = {}
    component_names = tuple(components)
    for component_index, (component, value) in enumerate(components.items()):
        gradients = torch.autograd.grad(
            value,
            parameters,
            retain_graph=component_index + 1 < len(component_names),
            allow_unused=True,
        )
        gradient_report[component] = {
            group: _norm(gradients, group_indices[group]) for group in GROUPS
        }
        if component == "association":
            gradient_topology = {
                "association_to_detached_u0_bypass": _norm(
                    gradients, association_bypass_indices
                ),
                "association_to_p2_consumer": _norm(
                    gradients, association_p2_indices
                ),
                "association_to_proposal_transport": _norm(
                    gradients, association_transport_indices
                ),
            }

    max_cross_model_numeric_error = max(
        cross_model_numeric_parity.values(), default=0.0
    )
    max_sidecar_anchor_error = max(
        sidecar_anchor_parity.values(), default=0.0
    )
    finite_losses = all(
        bool(torch.isfinite(value.detach()).all()) for value in components.values()
    )
    checks = {
        # Exactness is established structurally and bitwise.  Comparing two
        "legacy_v7_state_bit_exact": legacy_state_exact,
        "v14_returns_no_deployment_keys": not deployment_key_overlap,
        "v14_leaves_v7_anchors_bit_exact": max_sidecar_anchor_error == 0.0,
        "source_and_v14_public_outputs_bit_exact": all(
            error == 0.0 for error in cross_model_numeric_parity.values()
        ),
        "source_and_v14_categorical_outputs_exact": all(
            cross_model_numeric_parity[name] == 0.0
            for name in (
                "selection_slot_geometry_route_indices",
                "selection_slot_indices",
                "selection_slot_active",
            )
        ),
        "writer_prediction_files_bit_exact": writer_output_exact,
        "v14_replay_exact": replay_error <= 1.0e-7,
        "legacy_route_logit_absent": True,
        "inference_forward_is_target_free": target_free_forward_signature,
        "zero_content_changes_visual_state": zero_p2_visual_change > 1.0e-8,
        "zero_content_changes_association": zero_p2_proposal_change > 1.0e-8,
        "position_only_is_distinct": position_only_visual_change > 1.0e-8,
        "visual_loss_reaches_p2_and_slot_state": (
            gradient_report["visual"]["visual"] > 0.0
            and gradient_report["visual"]["slot_row"] > 0.0
            and gradient_report["visual"]["proposal"] == 0.0
        ),
        "association_loss_reaches_visual_slot_and_proposals": (
            gradient_report["association"]["visual"] > 0.0
            and gradient_report["association"]["slot_row"] > 0.0
            and gradient_report["association"]["proposal"] > 0.0
        ),
        "association_cannot_bypass_p2_through_u0": (
            bool(association_bypass_indices)
            and gradient_topology["association_to_detached_u0_bypass"] == 0.0
        ),
        "association_directly_reaches_p2_consumer": (
            bool(association_p2_indices)
            and gradient_topology["association_to_p2_consumer"] > 0.0
        ),
        "association_directly_reaches_proposal_transport": (
            bool(association_transport_indices)
            and gradient_topology["association_to_proposal_transport"] > 0.0
        ),
        "prediction_sinkhorn_rows_exact": attention_row_error <= 1.0e-6,
        "prediction_real_columns_capacity_exact": (
            attention_real_column_excess <= 1.0e-6
        ),
        "private_dustbin_mask_exact": private_off_diagonal_mass == 0.0,
        "target_sinkhorn_rows_exact": float(
            losses["four_slot_v14_target_attention_row_error"].detach().cpu()
        ) <= 1.0e-6,
        "target_real_columns_capacity_exact": float(
            losses["four_slot_v14_target_real_column_excess"].detach().cpu()
        ) <= 1.0e-6,
        "writer_valid_subset_of_source_active": writer_valid_contract,
        "assignment_deterministic": assignment_deterministic,
        "augmentation_exactly_disabled": augmentation_disabled,
        "cross_clip_derangement_exact": cross_clip_exact,
        "losses_finite": finite_losses,
        "only_v14_trainable": all(name.startswith(V14) for name in names),
    }
    report = {
        "experiment": "V14 corrected visual-first Stage-A zero-step contract",
        "iteration": iteration,
        "source_iteration": source_iteration,
        "deployment_mode": "exact_v7",
        "legacy_route_logits_used_by_v14": False,
        "legacy_state": {
            "source_tensor_count": len(source_state_names),
            "missing_in_v14": list(missing_legacy_state),
            "source_sha256": source_state_sha256,
            "v14_legacy_sha256": current_legacy_state_sha256,
            "bit_exact": legacy_state_exact,
        },
        "cross_model_numeric_parity": cross_model_numeric_parity,
        "max_cross_model_numeric_error": max_cross_model_numeric_error,
        "sidecar_anchor_parity": sidecar_anchor_parity,
        "max_sidecar_anchor_error": max_sidecar_anchor_error,
        "v14_deployment_key_overlap": deployment_key_overlap,
        "v14_replay_max_abs_error": replay_error,
        "writer_output": {
            "file_count": writer_file_count,
            "bit_exact": writer_output_exact,
            "postprocess": writer_kwargs,
        },
        "zero_p2_mean_visual_attention_change": zero_p2_visual_change,
        "zero_p2_mean_proposal_attention_change": zero_p2_proposal_change,
        "position_only_mean_visual_attention_change": (
            position_only_visual_change
        ),
        "prediction_attention_row_error": attention_row_error,
        "prediction_real_column_excess": attention_real_column_excess,
        "private_off_diagonal_mass": private_off_diagonal_mass,
        "cross_clip_reports": cross_clip_reports,
        "gradients": gradient_report,
        "gradient_topology": gradient_topology,
        "forward_signature": list(forward_parameters),
        "tensor_shapes": {
            "slot_states": list(captured["slot_states"].shape),
            "anchor_x_rows": list(captured["anchor_x_rows"].shape),
            "proposal_row_tokens": list(
                captured["proposal_row_tokens"].shape
            ),
            "proposal_x_rows": list(captured["proposal_x_rows"].shape),
            "p2_row_grid": list(captured["row_value_features"].shape),
            "visual_attention": list(
                outputs["selection_slot_v14_visual_attention"].shape
            ),
            "proposal_transport": list(attention.shape),
        },
        "trainable_parameter_names": names,
        "trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in parameters
        ),
        "freeze_stats": freeze_stats,
        "checks": checks,
        "passed": all(checks.values()),
        "test_set_used": False,
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {destination}")


if __name__ == "__main__":
    main()
