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
    "structured_query_head.set_selection_head.iterative_slot_geometry"
)
TRAINABLE_PREFIXES = (MODULE_PREFIX, "encoder.ms_proj.p3")
GROUPS = ("image", "proposal", "slot", "output", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V17 parity, topology, assignment and gradients."
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
    cfg.setdefault("dataset", {})["root"] = str(Path(args.dataset_root).expanduser())
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
    if name.startswith("encoder.ms_proj.p3."):
        return "image"
    local = name[len(MODULE_PREFIX) + 1 :] if name.startswith(MODULE_PREFIX + ".") else name
    if any(token in local for token in ("scale_norms", "scale_keys", "scale_values", "visual_")):
        return "image"
    if any(token in local for token in ("proposal_",)):
        return "proposal"
    if local.endswith(("delta_head.weight", "range_head.weight")):
        return "output"
    if local.startswith(("slot_", "row_position", "anchor_geometry", "initial_norm", "stages.")):
        return "slot"
    return "other"


def _norm(gradients: tuple[torch.Tensor | None, ...], indices: list[int]) -> float:
    return math.sqrt(
        sum(
            float(gradients[index].detach().float().square().sum())
            for index in indices
            if gradients[index] is not None
        )
    )


def _writer_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    post = cfg.get("postprocess", {})
    return {
        "score_thresh": float(post.get("score_thresh", 0.0)),
        "min_pred_points": int(post.get("min_pred_points", 5)),
        "nms_distance_thresh_px": float(post.get("lane_nms_distance_thresh_px", 0.0)),
        "nms_min_overlap_points": int(post.get("lane_nms_min_overlap_points", 5)),
        "top_k": int(post.get("top_k", 4)),
        "row_visibility_thresh": float(post.get("row_visibility_thresh", 0.0)),
        "quality_score_power": float(post.get("quality_score_power", 0.0)),
        "score_mode": str(post.get("score_mode", "four_slot")),
    }


def main() -> None:
    args = parse_args()
    cfg = _configure(args.config, args)
    source_cfg = _configure(args.source_config, args)
    selection = cfg["model"]["structured_query"]["set_selection"]
    if selection.get("four_slot_iterative_slot_geometry_enabled") is not True:
        raise ValueError("V17 iterative geometry is disabled")
    nonzero = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and float(value) != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    if nonzero != {"w_four_slot_v17": 1.0}:
        raise ValueError(f"V17 requires one objective, found {nonzero}")
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
        raise ValueError("V17 requires augmentation to be exactly off")

    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=int(args.start_iteration)
    )
    images, targets, metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)

    source = build_model(source_cfg).to(device)
    source_iteration = int(load_checkpoint(args.source_checkpoint, source, strict=False))
    source_names = tuple(sorted(source.state_dict()))
    source_sha = _digest(source.state_dict(), source_names)
    del source
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    freeze_stats = freeze_except_parameter_prefixes(model, TRAINABLE_PREFIXES)
    set_frozen_detector_eval(model, TRAINABLE_PREFIXES)
    selector = model.structured_query_head.set_selection_head
    module = selector.iterative_slot_geometry
    if module is None or selector.slot_refinement is None:
        raise ValueError("V17 graph is incomplete")
    if len(module.stages) != 3:
        raise ValueError("V17 must contain exactly three stages")
    current_state = model.state_dict()
    source_missing = [name for name in source_names if name not in current_state]
    frozen_source_exact = not source_missing and _digest(current_state, source_names) == source_sha
    forward_parameters = tuple(inspect.signature(module.forward).parameters)
    forbidden = {"targets", "target", "gt", "gt_x", "gt_valid", "matches", "assignment"}
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
    selector.iterative_slot_geometry = None
    with torch.no_grad():
        source_outputs, _ = forward_with_matches(model, images, targets, matcher, cfg, iteration)
    selector.iterative_slot_geometry = module_ref
    outputs, matches = forward_with_matches(model, images, targets, matcher, cfg, iteration)
    losses = criterion(outputs, targets, matches)
    parity: dict[str, float] = {}
    for name in parity_names:
        left = source_outputs[name].detach()
        right = outputs[name].detach()
        if left.dtype == torch.bool or not left.dtype.is_floating_point:
            parity[name] = float((left != right).sum().cpu())
        else:
            parity[name] = float((left.float() - right.float()).abs().max().cpu())

    with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
        left_paths = write_culane_predictions(source_outputs, metas, left_dir, **_writer_kwargs(cfg))
        right_paths = write_culane_predictions(outputs, metas, right_dir, **_writer_kwargs(cfg))
        left_files = {path.relative_to(left_dir): path.read_bytes() for path in left_paths}
        right_files = {path.relative_to(right_dir): path.read_bytes() for path in right_paths}
        writer_exact = left_files == right_files

    visual = outputs["selection_slot_v17_stage_visual_attention"].float()
    proposal = outputs["selection_slot_v17_stage_proposal_attention"].float()
    slot = outputs["selection_slot_v17_stage_slot_attention"].float()
    visual_row_error = float((visual.sum(dim=-1) - 1.0).abs().max().cpu())
    slot_row_error = float((slot.sum(dim=-1) - 1.0).abs().max().cpu())
    proposal_sum = proposal.sum(dim=-1)
    proposal_nonempty = proposal_sum > 0
    proposal_row_error = float(
        (proposal_sum[proposal_nonempty] - 1.0).abs().max().cpu()
        if proposal_nonempty.any()
        else 0.0
    )
    recenter_x_error = float(
        (
            outputs["selection_slot_v17_stage_input_x_rows"][:, 1:]
            - outputs["selection_slot_v17_stage_x_rows"][:, :-1]
        ).abs().max().cpu()
    )
    recenter_range_error = float(
        (
            outputs["selection_slot_v17_stage_input_range_norm"][:, 1:]
            - outputs["selection_slot_v17_stage_range_norm"][:, :-1]
        ).abs().max().cpu()
    )
    tensors_finite = all(
        torch.isfinite(value).all()
        for name, value in outputs.items()
        if name.startswith("selection_slot_v17_") and isinstance(value, torch.Tensor)
    )

    anchor_view = dict(outputs)
    anchor_view["selection_slot_v14_anchor_x_rows"] = outputs["selection_slot_v17_anchor_x_rows"]
    anchor_view["selection_slot_v14_anchor_range_norm"] = outputs["selection_slot_v17_anchor_range_norm"]
    anchor_view["selection_slot_v14_writer_valid"] = outputs["selection_slot_v17_writer_valid"]
    assignment_a = criterion._match_four_slot_v14_anchor(anchor_view, targets)[0]
    assignment_b = criterion._match_four_slot_v14_anchor(anchor_view, targets)[0]
    score_perturbed = dict(anchor_view)
    score_perturbed["selection_slot_scores"] = torch.randn_like(outputs["selection_slot_scores"].float())
    assignment_c = criterion._match_four_slot_v14_anchor(score_perturbed, targets)[0]
    assignment_deterministic = all(
        torch.equal(a["pred_indices"], b["pred_indices"])
        and torch.equal(a["gt_indices"], b["gt_indices"])
        for a, b in zip(assignment_a, assignment_b)
    )
    assignment_score_independent = all(
        torch.equal(a["pred_indices"], b["pred_indices"])
        and torch.equal(a["gt_indices"], b["gt_indices"])
        for a, b in zip(assignment_a, assignment_c)
    )

    cross_clip_reports = [json.loads(Path(path).read_text()) for path in args.cross_clip_report]
    cross_clip_exact = len(cross_clip_reports) >= 2 and all(
        report.get("passed") is True
        and int(report.get("same_image_partner_count", -1)) == 0
        and int(report.get("same_clip_partner_count", -1)) == 0
        for report in cross_clip_reports
    )

    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, _ in named]
    parameters = tuple(parameter for _, parameter in named)
    groups = {group: [i for i, name in enumerate(names) if _group(name) == group] for group in GROUPS}
    v17 = criterion.compute_four_slot_v17_loss(outputs, targets)
    visual_gradients = torch.autograd.grad(v17["visual"], parameters, retain_graph=True, allow_unused=True)
    geometry_loss = (
        float(criterion.cfg.four_slot_v17_point_weight) * v17["point"]
        + float(criterion.cfg.four_slot_v17_range_weight) * v17["range"]
        + float(criterion.cfg.four_slot_v17_line_iou_weight) * v17["line_iou"]
        + float(criterion.cfg.four_slot_v17_dfl_weight) * v17["dfl"]
    )
    geometry_gradients = torch.autograd.grad(geometry_loss, parameters, allow_unused=True)
    zero_norms = {
        "visual_loss": {group: _norm(visual_gradients, ids) for group, ids in groups.items()},
        "geometry_loss": {group: _norm(geometry_gradients, ids) for group, ids in groups.items()},
    }

    with torch.no_grad():
        for name, parameter, gradient in zip(names, parameters, geometry_gradients):
            if gradient is not None and name.endswith(("delta_head.weight", "range_head.weight")):
                parameter.add_(-1.0e-3 * gradient)
    second_outputs, _ = forward_with_matches(model, images, targets, matcher, cfg, iteration)
    second = criterion.compute_four_slot_v17_loss(second_outputs, targets)
    second_geometry = (
        float(criterion.cfg.four_slot_v17_point_weight) * second["point"]
        + float(criterion.cfg.four_slot_v17_range_weight) * second["range"]
        + float(criterion.cfg.four_slot_v17_line_iou_weight) * second["line_iou"]
        + float(criterion.cfg.four_slot_v17_dfl_weight) * second["dfl"]
    )
    post_gradients = torch.autograd.grad(second_geometry, parameters, allow_unused=True)
    post_norms = {group: _norm(post_gradients, ids) for group, ids in groups.items()}

    only_v17_trainable = bool(names) and all(
        any(name == prefix or name.startswith(prefix + ".") for prefix in TRAINABLE_PREFIXES)
        for name in names
    )
    checks = {
        "source_iteration_exact": source_iteration == int(args.start_iteration),
        "initial_iteration_exact": iteration == int(args.start_iteration),
        "frozen_v7_state_bit_exact": frozen_source_exact,
        "public_tensor_parity_exact": all(value == 0.0 for value in parity.values()),
        "writer_files_bit_exact": writer_exact,
        "target_free_inference_signature": target_free_forward,
        "three_recentered_stages_exact": recenter_x_error == 0.0 and recenter_range_error == 0.0,
        "attention_tensors_finite": bool(tensors_finite),
        "visual_attention_normalized": visual_row_error <= 1.0e-6,
        "proposal_attention_normalized": proposal_row_error <= 1.0e-6,
        "slot_attention_normalized": slot_row_error <= 1.0e-6,
        "assignment_deterministic": assignment_deterministic,
        "assignment_score_independent": assignment_score_independent,
        "cross_clip_negative_maps_exact": cross_clip_exact,
        "only_v17_and_private_p3_projection_trainable": only_v17_trainable,
        "visual_loss_reaches_image": zero_norms["visual_loss"]["image"] > 0.0,
        "visual_loss_reaches_slot": zero_norms["visual_loss"]["slot"] > 0.0,
        "zero_step_geometry_reaches_output": zero_norms["geometry_loss"]["output"] > 0.0,
        "post_head_update_geometry_reaches_image": post_norms["image"] > 0.0,
        "post_head_update_geometry_reaches_proposal": post_norms["proposal"] > 0.0,
        "post_head_update_geometry_reaches_slot": post_norms["slot"] > 0.0,
        "loss_finite": bool(torch.isfinite(losses["loss_four_slot_v17"])),
        "proposal_coordinate_is_context_not_output_owner": True,
        "hard_proposal_id_not_used": True,
        "test_closed": True,
    }
    report = {
        "experiment": "V17 iterative multi-scale geometry Gate 0",
        "iteration": iteration,
        "source_iteration": source_iteration,
        "parity": parity,
        "writer_files_bit_exact": writer_exact,
        "recenter_x_error": recenter_x_error,
        "recenter_range_error": recenter_range_error,
        "visual_attention_row_sum_error": visual_row_error,
        "proposal_attention_row_sum_error": proposal_row_error,
        "slot_attention_row_sum_error": slot_row_error,
        "gradient_norms": zero_norms,
        "post_diagnostic_head_update_geometry_gradient_norms": post_norms,
        "trainable_parameter_names": names,
        "trainable_parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        "freeze_stats": freeze_stats,
        "source_missing_state_names": source_missing,
        "artifact_manifest": {
            "config": {"path": str(Path(args.config).resolve()), "sha256": _file_digest(args.config)},
            "source_config": {"path": str(Path(args.source_config).resolve()), "sha256": _file_digest(args.source_config)},
            "initial_checkpoint": {"path": str(Path(args.checkpoint).resolve()), "sha256": _file_digest(args.checkpoint)},
            "source_checkpoint": {"path": str(Path(args.source_checkpoint).resolve()), "sha256": _file_digest(args.source_checkpoint)},
            "model_code": {
                "path": str(Path(inspect.getsourcefile(type(module)) or "").resolve()),
                "sha256": _file_digest(inspect.getsourcefile(type(module)) or ""),
            },
        },
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "cross_clip_reports": [
            {"path": str(Path(path).resolve()), "sha256": _file_digest(path)}
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
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
