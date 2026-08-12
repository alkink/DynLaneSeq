from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

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
from dynlaneseq_eg.tools.train import seed_everything


SELECTOR = "structured_query_head.set_selection_head."
UNIFIED = SELECTOR + "unified_slot_decoder."
GROUPS = (
    "visual_evidence",
    "proposal_memory",
    "slot_row_trunk",
    "geometry_output",
    "post_geometry_activity",
    "other_unified",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V11 discrete warm-start parity, one-loss topology and "
            "population gradients before any optimizer step."
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
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--start-iteration", type=int, default=225000)
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
    local = name[len(UNIFIED) :] if name.startswith(UNIFIED) else name
    if local.startswith(
        (
            "feature_",
            "x_position_projection.",
            "first_visual_",
            "second_visual_",
        )
    ):
        return "visual_evidence"
    if local.startswith(
        (
            "proposal_row_norm.",
            "proposal_key.",
            "proposal_value.",
            "global_proposal_",
            "coarse_geometry_projection.",
            "proposal_fusion_",
        )
    ):
        return "proposal_memory"
    if local.startswith(
        (
            "slot_norm.",
            "slot_projection.",
            "slot_tokens.",
            "row_position_projection.",
            "initial_norm.",
            "vertical_encoder.",
            "final_norm.",
            "final_ffn.",
        )
    ):
        return "slot_row_trunk"
    if local.startswith(
        (
            "delta_head.",
            "range_delta_head.",
        )
    ):
        return "geometry_output"
    if local.startswith("post_geometry_activity."):
        return "post_geometry_activity"
    return "other_unified"


def _norm(
    gradients: tuple[torch.Tensor | None, ...],
    indices: list[int],
) -> float:
    square = 0.0
    for index in indices:
        value = gradients[index]
        if value is not None:
            square += float(value.detach().float().square().sum())
    return math.sqrt(square)


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _component_gradients(
    losses: dict[str, torch.Tensor],
    parameters: tuple[torch.nn.Parameter, ...],
) -> dict[str, tuple[torch.Tensor | None, ...]]:
    components = {
        "geometry": (
            losses["loss_four_slot_unified_point"]
            + losses["loss_four_slot_unified_range"]
            + losses["loss_four_slot_unified_line_iou"]
            + losses["loss_four_slot_unified_dfl"]
            + losses["loss_four_slot_unified_aux_point"]
            + losses["loss_four_slot_unified_aux_range"]
            + losses["loss_four_slot_unified_aux_line_iou"]
        ),
        "attention": losses["loss_four_slot_unified_attention"],
        "activity": losses["loss_four_slot_unified_active"],
        "total": losses["loss_four_slot_unified"],
    }
    result: dict[str, tuple[torch.Tensor | None, ...]] = {}
    names = tuple(components)
    for index, name in enumerate(names):
        result[name] = torch.autograd.grad(
            components[name],
            parameters,
            retain_graph=index + 1 < len(names),
            allow_unused=True,
        )
    return result


def main() -> None:
    args = parse_args()
    if int(args.batches) < 1:
        raise ValueError("--batches must be positive")
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
    if selection.get("four_slot_unified_slot_decoder_enabled") is not True:
        raise ValueError("V11 unified slot decoder is disabled")
    if selection.get("four_slot_refinement_enabled") is not False:
        raise ValueError("V11 must not reuse the legacy V7 refiner")
    if float(cfg["loss"].get("w_four_slot_unified", 0.0)) != 1.0:
        raise ValueError("V11 unified loss must be the sole final-slot loss")
    nonzero_objectives = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and float(value) != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    if nonzero_objectives != {"w_four_slot_unified": 1.0}:
        raise ValueError(
            "V11 requires one objective; resolved nonzero weights are "
            f"{nonzero_objectives}"
        )

    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    loader = build_dataloader(
        cfg,
        split="train",
        training=True,
        start_iteration=int(args.start_iteration),
    )
    iterator = iter(loader)
    parity_images, parity_targets, _metas = next(iterator)
    parity_images = parity_images.to(device, non_blocking=True)
    parity_targets = nested_to_device(parity_targets, device)

    # Compute the authoritative V7 source values, then release that model
    # before constructing the gradient-audited V11 model.
    source_model = build_model(source_cfg).to(device)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source_model, strict=False)
    )
    if source_iteration != int(args.start_iteration):
        raise ValueError("source checkpoint iteration mismatch")
    source_model.eval()
    source_matcher = build_matcher(source_cfg)
    with torch.no_grad():
        source_outputs, _source_matches = forward_with_matches(
            source_model,
            parity_images,
            parity_targets,
            source_matcher,
            source_cfg,
            source_iteration,
        )
    parity_names = (
        "selection_slot_real_route_logits",
        "selection_slot_geometry_route_indices",
        "selection_slot_indices",
        "selection_slot_active_logits",
        "selection_slot_active",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    )
    source_values = {
        name: source_outputs[name].detach().clone() for name in parity_names
    }
    del source_outputs, source_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    if iteration != int(args.start_iteration):
        raise ValueError("V11 checkpoint iteration mismatch")
    freeze_stats = freeze_except_parameter_prefixes(
        model,
        (UNIFIED.rstrip("."),),
    )
    set_frozen_detector_eval(model, (UNIFIED.rstrip("."),))
    selector = model.structured_query_head.set_selection_head
    if (
        selector.unified_slot_decoder is None
        or selector.slot_refinement is not None
    ):
        raise ValueError("V11 model graph is incomplete")

    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(iteration)
    v11_outputs, v11_matches = forward_with_matches(
        model,
        parity_images,
        parity_targets,
        matcher,
        cfg,
        iteration,
    )
    parity: dict[str, float] = {}
    for name, source_value in source_values.items():
        current = v11_outputs[name].detach()
        if current.dtype == torch.bool or not current.dtype.is_floating_point:
            parity[name] = float((current != source_value).sum().cpu())
        else:
            parity[name] = float(
                (current.float() - source_value.float()).abs().max().cpu()
            )
    source_to_soft_x_shift = float(
        (
            v11_outputs["selection_slot_unified_base_x_rows"].detach().float()
            - source_values["selection_slot_pred_x_rows"].float()
        )
        .abs()
        .max()
        .cpu()
    )
    source_to_soft_range_shift = float(
        (
            v11_outputs["selection_slot_unified_base_range_norm"]
            .detach()
            .float()
            - source_values["selection_slot_range_norm"].float()
        )
        .abs()
        .max()
        .cpu()
    )
    score_error = float(
        (
            v11_outputs["selection_slot_scores"].detach().float()
            - torch.sigmoid(
                v11_outputs["selection_slot_active_logits"].detach().float()
            )
        )
        .abs()
        .max()
        .cpu()
    )
    zero_delta_x_error = float(
        (
            v11_outputs["selection_slot_pred_x_rows"].detach().float()
            - v11_outputs["selection_slot_unified_aux_x_rows"].detach().float()
        )
        .abs()
        .max()
        .cpu()
    )
    zero_delta_range_error = float(
        (
            v11_outputs["selection_slot_range_norm"].detach().float()
            - v11_outputs["selection_slot_unified_aux_range_norm"]
            .detach()
            .float()
        )
        .abs()
        .max()
        .cpu()
    )
    del v11_outputs, v11_matches

    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    names = tuple(name for name, _parameter in named_parameters)
    parameters = tuple(parameter for _name, parameter in named_parameters)
    group_ids = {
        group: [
            index
            for index, name in enumerate(names)
            if _group(name) == group
        ]
        for group in GROUPS
    }
    rows: list[dict[str, Any]] = []
    # Restart the deterministic loader so the population audit includes the
    # exact first batch rather than silently skipping it after parity.
    iterator = iter(loader)
    for batch_index in tqdm(range(int(args.batches)), desc="V11 contract"):
        try:
            images, targets, _metas = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets, _metas = next(iterator)
        images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            iteration,
        )
        losses = criterion(outputs, targets, matches)
        gradients = _component_gradients(losses, parameters)
        grouped = {
            group: {
                component: _norm(values, group_ids[group])
                for component, values in gradients.items()
            }
            for group in GROUPS
        }
        proposal_attention = outputs[
            "selection_slot_unified_proposal_attention"
        ].detach().float()
        visual_attention = outputs[
            "selection_slot_unified_visual_attention"
        ].detach().float()
        rows.append(
            {
                "batch": batch_index,
                "loss_total": float(
                    losses["loss_four_slot_unified"].detach().cpu()
                ),
                "groups": grouped,
                "proposal_attention_sum_error": float(
                    (proposal_attention.sum(dim=-1) - 1.0).abs().max().cpu()
                ),
                "proposal_attention_column_excess": float(
                    (proposal_attention.sum(dim=1) - 1.0)
                    .clamp_min(0.0)
                    .max()
                    .cpu()
                ),
                "visual_attention_sum_error": float(
                    (visual_attention.sum(dim=-1) - 1.0).abs().max().cpu()
                ),
                "proposal_attention_finite": bool(
                    torch.isfinite(proposal_attention).all()
                ),
                "visual_attention_finite": bool(
                    torch.isfinite(visual_attention).all()
                ),
                "mean_matched": float(
                    losses["four_slot_unified_mean_matched"].detach().cpu()
                ),
                "mean_target_support_mass": float(
                    losses[
                        "four_slot_unified_mean_target_support_mass"
                    ].detach().cpu()
                ),
            }
        )
        del outputs, matches, losses, gradients

    group_summary = {
        group: {
            component: _summary(
                [row["groups"][group][component] for row in rows]
            )
            for component in ("geometry", "attention", "activity", "total")
        }
        for group in GROUPS
    }
    checks = {
        "only_unified_module_trainable": bool(names)
        and all(name.startswith(UNIFIED) for name in names),
        "source_route_logits_within_1e4": parity[
            "selection_slot_real_route_logits"
        ] <= 1.0e-4,
        "source_route_indices_exact": parity[
            "selection_slot_geometry_route_indices"
        ] == 0.0,
        "source_public_indices_exact": parity["selection_slot_indices"] == 0.0,
        "source_activity_logits_within_1e3": parity[
            "selection_slot_active_logits"
        ] <= 1.0e-3,
        "source_activity_exact": parity["selection_slot_active"] == 0.0,
        "legacy_refiner_absent": selector.slot_refinement is None,
        "initial_x_residual_below_0p1px": zero_delta_x_error <= 0.1,
        "initial_range_residual_below_1e3": (
            zero_delta_range_error <= 1.0e-3
        ),
        "deployment_score_is_final_activity": score_error <= 1.0e-7,
        "geometry_reaches_visual_evidence": min(
            row["groups"]["visual_evidence"]["geometry"] for row in rows
        ) > 0.0,
        "geometry_reaches_proposal_memory": min(
            row["groups"]["proposal_memory"]["geometry"] for row in rows
        ) > 0.0,
        "geometry_reaches_slot_row_trunk": min(
            row["groups"]["slot_row_trunk"]["geometry"] for row in rows
        ) > 0.0,
        "geometry_reaches_geometry_output": min(
            row["groups"]["geometry_output"]["geometry"] for row in rows
        ) > 0.0,
        "geometry_does_not_reach_activity_head": max(
            row["groups"]["post_geometry_activity"]["geometry"]
            for row in rows
        ) == 0.0,
        "activity_does_not_reach_geometry_heads": max(
            row["groups"]["geometry_output"]["activity"] for row in rows
        ) == 0.0,
        "activity_reads_post_geometry_slot_state": min(
            min(
                row["groups"]["visual_evidence"]["activity"],
                row["groups"]["proposal_memory"]["activity"],
                row["groups"]["slot_row_trunk"]["activity"],
            )
            for row in rows
        ) > 0.0,
        "attention_reaches_proposal_and_slot_state": min(
            min(
                row["groups"]["proposal_memory"]["attention"],
                row["groups"]["slot_row_trunk"]["attention"],
            )
            for row in rows
        ) > 0.0,
        "proposal_attention_normalized": max(
            row["proposal_attention_sum_error"] for row in rows
        ) <= 1.0e-5,
        "proposal_attention_column_capacity": max(
            row["proposal_attention_column_excess"] for row in rows
        ) <= 1.0e-5,
        "visual_attention_normalized": max(
            row["visual_attention_sum_error"] for row in rows
        ) <= 1.0e-5,
        "attention_is_finite": all(
            row["proposal_attention_finite"]
            and row["visual_attention_finite"]
            for row in rows
        ),
        "unified_losses_are_finite": all(
            math.isfinite(row["loss_total"]) for row in rows
        ),
    }
    report = {
        "experiment": "V11 unified soft-memory slot-row contract",
        "config": str(Path(args.config).expanduser().resolve()),
        "source_config": str(Path(args.source_config).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "source_checkpoint": str(
            Path(args.source_checkpoint).expanduser().resolve()
        ),
        "iteration": iteration,
        "batches": len(rows),
        "freeze_stats": freeze_stats,
        "trainable_parameter_tensors": len(names),
        "trainable_parameter_names": list(names),
        "source_discrete_warm_start_max_abs_or_count": parity,
        "source_to_soft_memory_x_max_abs_shift_px": source_to_soft_x_shift,
        "source_to_soft_memory_range_max_abs_shift": (
            source_to_soft_range_shift
        ),
        "initial_soft_x_residual_max_abs_px": zero_delta_x_error,
        "initial_soft_range_residual_max_abs": zero_delta_range_error,
        "deployment_score_semantic_max_abs_error": score_error,
        "checks": checks,
        "passed": all(checks.values()),
        "group_summary": group_summary,
        "mean_target_support_mass": _summary(
            [row["mean_target_support_mass"] for row in rows]
        ),
        "per_batch": rows,
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"output_json: {output_json}")
    if report["passed"] is not True:
        raise SystemExit("V11 initialization contract failed")


if __name__ == "__main__":
    main()
