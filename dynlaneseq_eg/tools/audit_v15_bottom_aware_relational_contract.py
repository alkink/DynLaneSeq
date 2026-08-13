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


PREFIX = (
    "structured_query_head.set_selection_head."
    "bottom_aware_relational_geometry."
)
GROUPS = ("visual", "graph", "fusion", "output", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V15 exact parity, graph contract and gradients."
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
    parser.add_argument("--cross-clip-report", action="append", default=[])
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


def _digest(state: dict[str, torch.Tensor], names: tuple[str, ...]) -> str:
    value = hashlib.sha256()
    for name in names:
        tensor = state[name].detach().cpu().contiguous()
        value.update(name.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(str(tensor.dtype).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def _file_digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _group(name: str) -> str:
    local = name[len(PREFIX) :] if name.startswith(PREFIX) else name
    if local.startswith(
        (
            "slot_",
            "row_position_projection.",
            "anchor_geometry_projection.",
            "initial_norm.",
            "feature_",
            "x_position_key.",
            "visual_",
        )
    ):
        return "visual"
    if local.startswith(
        (
            "proposal_row_norm.",
            "proposal_content.",
            "proposal_geometry.",
            "graph_",
            "slot_memory_query.",
            "proposal_memory_",
        )
    ):
        return "graph"
    if local.startswith(
        (
            "proposal_context.",
            "context_geometry_projection.",
            "fusion_",
            "slot_interaction.",
        )
    ):
        return "fusion"
    if local.startswith(("output_norm.", "delta_head.", "range_head.")):
        return "output"
    return "other"


def _norm(
    gradients: tuple[torch.Tensor | None, ...], indices: list[int]
) -> float:
    return math.sqrt(
        sum(
            float(gradients[index].detach().float().square().sum())
            for index in indices
            if gradients[index] is not None
        )
    )


def main() -> None:
    args = parse_args()
    cfg = _configure(args.config, args)
    source_cfg = _configure(args.source_config, args)
    selection = cfg["model"]["structured_query"]["set_selection"]
    if selection.get("four_slot_bottom_aware_relational_geometry_enabled") is not True:
        raise ValueError("V15 relational geometry is disabled")
    nonzero = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and float(value) != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    if nonzero != {"w_four_slot_v15": 1.0}:
        raise ValueError(f"V15 requires one objective, found {nonzero}")
    augmentation = cfg.get("augmentation", {})
    augmentation_disabled = all(
        augmentation.get(name) == value
        for name, value in {
            "horizontal_flip_prob": 0.0,
            "color_jitter": False,
            "channel_shuffle_prob": 0.0,
            "hue_saturation_prob": 0.0,
            "blur_prob": 0.0,
            "affine_prob": 0.0,
            "random_shadow_prob": 0.0,
        }.items()
    )
    if not augmentation_disabled:
        raise ValueError("V15 requires augmentation to be exactly off")

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

    source = build_model(source_cfg).to(device)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source, strict=False)
    )
    source.eval()
    source_names = tuple(sorted(source.state_dict()))
    source_sha = _digest(source.state_dict(), source_names)
    del source
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    freeze_stats = freeze_except_parameter_prefixes(model, (PREFIX.rstrip("."),))
    set_frozen_detector_eval(model, (PREFIX.rstrip("."),))
    selector = model.structured_query_head.set_selection_head
    module = selector.bottom_aware_relational_geometry
    if module is None or selector.slot_refinement is None:
        raise ValueError("V15 graph is incomplete")
    current_state = model.state_dict()
    source_missing = [name for name in source_names if name not in current_state]
    frozen_source_exact = (
        not source_missing and _digest(current_state, source_names) == source_sha
    )
    forward_parameters = tuple(inspect.signature(module.forward).parameters)
    forbidden = {
        "targets",
        "target",
        "gt",
        "gt_x",
        "gt_valid",
        "matches",
        "assignment",
    }
    target_free_forward = not bool(forbidden.intersection(forward_parameters))

    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(iteration)
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
    module_ref = module
    selector.bottom_aware_relational_geometry = None
    with torch.no_grad():
        source_outputs, _ = forward_with_matches(
            model, images, targets, matcher, cfg, iteration
        )
    selector.bottom_aware_relational_geometry = module_ref
    outputs, matches = forward_with_matches(
        model, images, targets, matcher, cfg, iteration
    )
    losses = criterion(outputs, targets, matches)
    parity: dict[str, float] = {}
    for name in parity_names:
        left = source_outputs[name].detach()
        right = outputs[name].detach()
        if left.dtype == torch.bool or not left.dtype.is_floating_point:
            parity[name] = float((left != right).sum().cpu())
        else:
            parity[name] = float((left.float() - right.float()).abs().max().cpu())

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
    with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
        left_paths = write_culane_predictions(
            source_outputs, metas, left_dir, **writer_kwargs
        )
        right_paths = write_culane_predictions(
            outputs, metas, right_dir, **writer_kwargs
        )
        left = {path.relative_to(left_dir): path.read_bytes() for path in left_paths}
        right = {
            path.relative_to(right_dir): path.read_bytes() for path in right_paths
        }
        writer_exact = left == right

    attention = outputs["selection_slot_v15_graph_attention"].float()
    pair_valid = outputs["selection_slot_v15_graph_pair_valid"].bool()
    graph_row_error = float(
        (
            attention.sum(dim=-1)
            - outputs["selection_slot_candidate_valid"].float()
        )
        .abs()
        .max()
        .detach()
        .cpu()
    )
    invalid_values = attention.masked_select(~pair_valid)
    graph_invalid_mass = (
        float(invalid_values.abs().max().detach().cpu())
        if invalid_values.numel()
        else 0.0
    )
    graph_finite = bool(
        torch.isfinite(outputs["selection_slot_v15_graph_edge_features"]).all()
        and torch.isfinite(attention).all()
    )
    no_collapse_outputs = not any(
        "cluster" in name.lower() or "prototype" in name.lower()
        for name in outputs
        if name.startswith("selection_slot_v15_")
    )

    anchor_view = dict(outputs)
    anchor_view["selection_slot_v14_anchor_x_rows"] = outputs[
        "selection_slot_v15_anchor_x_rows"
    ]
    anchor_view["selection_slot_v14_anchor_range_norm"] = outputs[
        "selection_slot_v15_anchor_range_norm"
    ]
    anchor_view["selection_slot_v14_writer_valid"] = outputs[
        "selection_slot_v15_writer_valid"
    ]
    assignment_a = criterion._match_four_slot_v14_anchor(
        anchor_view, targets
    )[0]
    assignment_b = criterion._match_four_slot_v14_anchor(
        anchor_view, targets
    )[0]
    score_perturbed = dict(anchor_view)
    score_perturbed["selection_slot_scores"] = torch.randn_like(
        outputs["selection_slot_scores"].float()
    )
    assignment_score_perturbed = criterion._match_four_slot_v14_anchor(
        score_perturbed, targets
    )[0]
    assignment_deterministic = all(
        torch.equal(first["pred_indices"], second["pred_indices"])
        and torch.equal(first["gt_indices"], second["gt_indices"])
        for first, second in zip(assignment_a, assignment_b)
    )
    assignment_score_independent = all(
        torch.equal(first["pred_indices"], second["pred_indices"])
        and torch.equal(first["gt_indices"], second["gt_indices"])
        for first, second in zip(assignment_a, assignment_score_perturbed)
    )

    cross_clip_reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.cross_clip_report
    ]
    cross_clip_exact = len(cross_clip_reports) >= 2 and all(
        report.get("passed") is True
        and int(report.get("same_image_partner_count", -1)) == 0
        and int(report.get("same_clip_partner_count", -1)) == 0
        for report in cross_clip_reports
    )

    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in named]
    parameters = tuple(parameter for _, parameter in named)
    groups = {
        group: [index for index, name in enumerate(names) if _group(name) == group]
        for group in GROUPS
    }
    v15_losses = criterion.compute_four_slot_v15_loss(outputs, targets)
    visual_gradients = torch.autograd.grad(
        v15_losses["visual"],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    geometry_loss = (
        float(criterion.cfg.four_slot_v15_point_weight) * v15_losses["point"]
        + float(criterion.cfg.four_slot_v15_range_weight) * v15_losses["range"]
        + float(criterion.cfg.four_slot_v15_line_iou_weight)
        * v15_losses["line_iou"]
        + float(criterion.cfg.four_slot_v15_dfl_weight) * v15_losses["dfl"]
    )
    geometry_zero_gradients = torch.autograd.grad(
        geometry_loss,
        parameters,
        allow_unused=True,
    )
    zero_gradient_norms = {
        "visual_loss": {
            group: _norm(visual_gradients, indices)
            for group, indices in groups.items()
        },
        "geometry_loss": {
            group: _norm(geometry_zero_gradients, indices)
            for group, indices in groups.items()
        },
    }

    # One deterministic diagnostic update to only the zero-initialized output
    # heads proves the graph becomes live without introducing a gradient-only
    # straight-through path.  Nothing is saved and this is not an optimizer
    # step or checkpoint-selection arm.
    with torch.no_grad():
        for name, parameter, gradient in zip(names, parameters, geometry_zero_gradients):
            if gradient is not None and (
                name.endswith("delta_head.weight")
                or name.endswith("range_head.weight")
            ):
                parameter.add_(-1.0e-3 * gradient)
    second_outputs, second_matches = forward_with_matches(
        model, images, targets, matcher, cfg, iteration
    )
    second_v15 = criterion.compute_four_slot_v15_loss(second_outputs, targets)
    second_geometry = (
        float(criterion.cfg.four_slot_v15_point_weight) * second_v15["point"]
        + float(criterion.cfg.four_slot_v15_range_weight) * second_v15["range"]
        + float(criterion.cfg.four_slot_v15_line_iou_weight)
        * second_v15["line_iou"]
        + float(criterion.cfg.four_slot_v15_dfl_weight) * second_v15["dfl"]
    )
    post_update_gradients = torch.autograd.grad(
        second_geometry,
        parameters,
        allow_unused=True,
    )
    post_update_norms = {
        group: _norm(post_update_gradients, indices)
        for group, indices in groups.items()
    }

    checks = {
        "source_iteration_exact": source_iteration == int(args.start_iteration),
        "initial_iteration_exact": iteration == int(args.start_iteration),
        "frozen_v7_state_bit_exact": frozen_source_exact,
        "public_tensor_parity_exact": all(value == 0.0 for value in parity.values()),
        "writer_files_bit_exact": writer_exact,
        "target_free_inference_signature": target_free_forward,
        "no_hard_cluster_or_prototype_outputs": no_collapse_outputs,
        "graph_finite": graph_finite,
        "graph_row_sum_error_le_1e_6": graph_row_error <= 1.0e-6,
        "invalid_graph_edge_mass_zero": graph_invalid_mass == 0.0,
        "assignment_deterministic": assignment_deterministic,
        "assignment_score_independent": assignment_score_independent,
        "cross_clip_negative_maps_exact": cross_clip_exact,
        "only_v15_trainable": bool(names)
        and all(name.startswith(PREFIX) for name in names),
        "visual_loss_reaches_visual_only": (
            zero_gradient_norms["visual_loss"]["visual"] > 0.0
            and zero_gradient_norms["visual_loss"]["graph"] == 0.0
            and zero_gradient_norms["visual_loss"]["fusion"] == 0.0
            and zero_gradient_norms["visual_loss"]["output"] == 0.0
        ),
        "zero_step_geometry_reaches_output": (
            zero_gradient_norms["geometry_loss"]["output"] > 0.0
        ),
        "post_head_update_geometry_reaches_graph": post_update_norms["graph"] > 0.0,
        "post_head_update_geometry_reaches_fusion": post_update_norms["fusion"] > 0.0,
        "post_head_update_geometry_reaches_visual": post_update_norms["visual"] > 0.0,
        "loss_finite": bool(torch.isfinite(losses["loss_four_slot_v15"])),
        "test_closed": True,
    }
    report = {
        "experiment": "V15 zero-step and post-head-update contract",
        "iteration": iteration,
        "source_iteration": source_iteration,
        "parity": parity,
        "writer_files_bit_exact": writer_exact,
        "graph_row_sum_error": graph_row_error,
        "graph_invalid_mass": graph_invalid_mass,
        "gradient_norms": zero_gradient_norms,
        "post_diagnostic_head_update_geometry_gradient_norms": post_update_norms,
        "trainable_parameter_names": names,
        "trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in parameters
        ),
        "freeze_stats": freeze_stats,
        "source_missing_state_names": source_missing,
        "artifact_manifest": {
            "config": {
                "path": str(Path(args.config).resolve()),
                "sha256": _file_digest(args.config),
            },
            "source_config": {
                "path": str(Path(args.source_config).resolve()),
                "sha256": _file_digest(args.source_config),
            },
            "initial_checkpoint": {
                "path": str(Path(args.checkpoint).resolve()),
                "sha256": _file_digest(args.checkpoint),
            },
            "source_checkpoint": {
                "path": str(Path(args.source_checkpoint).resolve()),
                "sha256": _file_digest(args.source_checkpoint),
            },
            "model_code": {
                "path": str(
                    Path(inspect.getsourcefile(type(module)) or "").resolve()
                ),
                "sha256": _file_digest(
                    inspect.getsourcefile(type(module)) or ""
                ),
            },
        },
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "cross_clip_reports": [
            {
                "path": str(Path(path).resolve()),
                "sha256": _file_digest(path),
            }
            for path in args.cross_clip_report
        ],
        "checks": checks,
        "passed": all(checks.values()),
        "diagnostic_parameter_update_saved": False,
        "optimizer_steps_during_audit": 0,
        "test_set_used": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
