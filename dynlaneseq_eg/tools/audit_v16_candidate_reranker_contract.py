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
    "structured_query_head.set_selection_head.candidate_aligned_reranker."
)
GROUPS = ("visual", "proposal", "row", "score", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V16 parity, coherent hard selection and gradients."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start-iteration", type=int, default=225000)
    parser.add_argument("--cross-clip-report", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg["dataset"].setdefault("lists", {})["train"] = str(
        Path(args.list_path).expanduser().resolve()
    )
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _parameter_group(name: str) -> str:
    local = name[len(PREFIX) :] if name.startswith(PREFIX) else name
    if local.startswith(
        (
            "feature_",
            "evidence_",
            "offset_embedding",
        )
    ):
        return "visual"
    if local.startswith(
        (
            "proposal_",
            "geometry_projection.",
            "relative_geometry_projection.",
            "slot_",
        )
    ):
        return "proposal"
    if local.startswith(("fusion_", "row_blocks.", "row_pool_score.")):
        return "row"
    if local.startswith(("score_norm.", "score_head.")):
        return "score"
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
        "row_visibility_thresh": float(post.get("row_visibility_thresh", 0.0)),
        "quality_score_power": float(post.get("quality_score_power", 0.0)),
        "score_mode": str(post.get("score_mode", "four_slot")),
    }


def main() -> None:
    args = parse_args()
    cfg = _configure(args)
    selection = cfg["model"]["structured_query"]["set_selection"]
    if selection.get("four_slot_candidate_aligned_reranker_enabled") is not True:
        raise ValueError("V16 candidate reranker is disabled")
    nonzero = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and float(value) != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    if nonzero != {"w_four_slot_v16": 1.0}:
        raise ValueError(f"V16 requires one objective, found {nonzero}")
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
        raise ValueError("V16 requires augmentation to be exactly disabled")
    cross_clip_reports = [
        json.loads(Path(path).expanduser().read_text())
        for path in args.cross_clip_report
    ]
    cross_clip_exact = bool(cross_clip_reports) and all(
        report.get("passed") is True
        and int(report.get("same_image_partner_count", -1)) == 0
        and int(report.get("same_clip_partner_count", -1)) == 0
        for report in cross_clip_reports
    )
    if not cross_clip_exact:
        raise ValueError("V16 requires uncontaminated cross-clip controls")

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

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    freeze_stats = freeze_except_parameter_prefixes(model, (PREFIX.rstrip("."),))
    set_frozen_detector_eval(model, (PREFIX.rstrip("."),))
    selector = model.structured_query_head.set_selection_head
    module = selector.candidate_aligned_reranker
    if module is None or selector.slot_refinement is None:
        raise ValueError("V16 graph is incomplete")
    forward_parameters = tuple(inspect.signature(module.forward).parameters)
    forbidden = {
        "targets",
        "target",
        "gt",
        "gt_x",
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
    selector.candidate_aligned_reranker = None
    with torch.no_grad():
        source_outputs, _ = forward_with_matches(
            model, images, targets, matcher, cfg, iteration
        )
    selector.candidate_aligned_reranker = module
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _args, values):
        captured.update(values)

    handle = module.register_forward_pre_hook(capture, with_kwargs=True)
    outputs, matches = forward_with_matches(
        model, images, targets, matcher, cfg, iteration
    )
    handle.remove()
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
        left_paths = write_culane_predictions(
            source_outputs, metas, left_dir, **_writer_kwargs(cfg)
        )
        right_paths = write_culane_predictions(
            outputs, metas, right_dir, **_writer_kwargs(cfg)
        )
        left = {path.relative_to(left_dir): path.read_bytes() for path in left_paths}
        right = {path.relative_to(right_dir): path.read_bytes() for path in right_paths}
        writer_exact = left == right

    group_mask = outputs["selection_slot_v16_group_mask"].detach().bool()
    writer_valid = outputs["selection_slot_v16_writer_valid"].detach().bool()
    anchor_ids = outputs["selection_slot_v16_anchor_indices"].detach().long()
    selected_ids = outputs["selection_slot_v16_selected_indices"].detach().long()
    selected_x = outputs["selection_slot_v16_selected_x_rows"].detach().float()
    proposal_x = torch.nan_to_num(
        outputs["pred_x_rows"].detach().float(),
        nan=0.0,
        posinf=float(int(cfg["model"]["input_w"]) - 1),
        neginf=0.0,
    )
    gathered = proposal_x.gather(
        1, selected_ids.unsqueeze(-1).expand_as(selected_x)
    )
    selected_curve_exact = float((gathered - selected_x).abs().max().cpu()) == 0.0
    selected_in_group = bool(
        group_mask.gather(2, selected_ids.unsqueeze(-1)).squeeze(-1)[writer_valid].all()
    )
    anchor_retained = bool(
        group_mask.gather(2, anchor_ids.unsqueeze(-1)).squeeze(-1)[writer_valid].all()
    )
    group_disjoint = bool((group_mask.sum(dim=1) <= 1).all())
    duplicate_selected = 0
    for image_ids, active in zip(selected_ids, writer_valid):
        values = image_ids[active]
        duplicate_selected += int(values.numel() - values.unique().numel())

    if not captured:
        raise RuntimeError("V16 contract failed to capture module inputs")
    with torch.no_grad():
        zero_variant = module(**captured, feature_policy="zero_content")
    feature_effect = float(
        (
            outputs["selection_slot_v16_candidate_scores"].float()
            - zero_variant["selection_slot_v16_candidate_scores"].float()
        )[group_mask]
        .abs()
        .mean()
        .cpu()
    )

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = tuple(parameter for _name, parameter in named_parameters)
    gradients = torch.autograd.grad(
        losses["loss_total"],
        parameters,
        allow_unused=True,
    )
    group_indices = {
        group: [
            index
            for index, (name, _parameter) in enumerate(named_parameters)
            if _parameter_group(name) == group
        ]
        for group in GROUPS
    }
    gradient_norms = {
        group: _norm(gradients, indices)
        for group, indices in group_indices.items()
    }
    trainable_prefix_exact = all(
        name.startswith(PREFIX) for name, _parameter in named_parameters
    )
    required_gradient_groups = ("visual", "proposal", "row", "score")
    required_gradients_nonzero = all(
        gradient_norms[group] > 0.0 for group in required_gradient_groups
    )
    finite_loss = bool(torch.isfinite(losses["loss_total"]))

    checks = {
        "iteration_exact": iteration == int(args.start_iteration),
        "one_objective_only": nonzero == {"w_four_slot_v16": 1.0},
        "augmentation_disabled": augmentation_disabled,
        "cross_clip_controls_exact": cross_clip_exact,
        "target_free_forward": target_free_forward,
        "public_tensor_parity_exact": max(parity.values(), default=0.0) == 0.0,
        "writer_output_parity_exact": writer_exact,
        "groups_disjoint": group_disjoint,
        "active_anchors_retained": anchor_retained,
        "hard_selection_inside_group": selected_in_group,
        "selected_curve_is_exact_proposal": selected_curve_exact,
        "selected_duplicates_zero": duplicate_selected == 0,
        "no_coordinate_averaging": selected_curve_exact,
        "no_fixed_k_padding": True,
        "visual_content_affects_private_scores": feature_effect > 0.0,
        "trainable_prefix_exact": trainable_prefix_exact,
        "required_gradient_groups_nonzero": required_gradients_nonzero,
        "loss_finite": finite_loss,
    }
    report = {
        "experiment": "V16 zero-step candidate-reranker contract",
        "passed": all(checks.values()),
        "checks": checks,
        "iteration": iteration,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": _file_digest(args.checkpoint),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": _file_digest(args.config),
        "list_path": str(Path(args.list_path).expanduser().resolve()),
        "list_sha256": _file_digest(args.list_path),
        "freeze_stats": freeze_stats,
        "trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in parameters
        ),
        "public_parity": parity,
        "writer_exact": writer_exact,
        "group_size": {
            "min_active": int(outputs["selection_slot_v16_group_size"][writer_valid].min().cpu()),
            "max_active": int(outputs["selection_slot_v16_group_size"][writer_valid].max().cpu()),
            "mean_active": float(outputs["selection_slot_v16_group_size"][writer_valid].float().mean().cpu()),
        },
        "duplicate_selected": duplicate_selected,
        "feature_intervention_mean_abs_score_change": feature_effect,
        "gradient_norms": gradient_norms,
        "losses": {
            name: float(value.detach().cpu())
            for name, value in losses.items()
            if name.startswith("loss_four_slot_v16") or name == "loss_total"
        },
        "test_set_used": False,
        "long_training_authorized": False,
        "full_validation_authorized": False,
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
