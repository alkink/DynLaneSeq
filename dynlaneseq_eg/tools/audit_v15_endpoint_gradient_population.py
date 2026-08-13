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
from dynlaneseq_eg.engine.checkpoint import _torch_load, load_checkpoint
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
    build_optimizer,
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
        description=(
            "Measure V15 endpoint loss gradients, clipping and the next "
            "AdamW update implied by the saved optimizer moments."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


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


def _cosine(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    indices: list[int],
) -> float:
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    for index in indices:
        first = left[index]
        second = right[index]
        if first is None or second is None:
            continue
        first_float = first.detach().float()
        second_float = second.detach().float()
        dot += float((first_float * second_float).sum())
        left_square += float(first_float.square().sum())
        right_square += float(second_float.square().sum())
    return dot / max(math.sqrt(left_square * right_square), 1.0e-12)


def _adam_update_ratio(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    gradients: tuple[torch.Tensor | None, ...],
    group_indices: dict[str, list[int]],
    clip_scale: float,
) -> dict[str, dict[str, float | int]]:
    hyper: dict[torch.nn.Parameter, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            hyper[parameter] = group
    result: dict[str, dict[str, float | int]] = {}
    for group_name, indices in group_indices.items():
        update_square = 0.0
        parameter_square = 0.0
        states = 0
        for index in indices:
            gradient = gradients[index]
            parameter = parameters[index]
            if gradient is None:
                continue
            specification = hyper[parameter]
            state = optimizer.state.get(parameter, {})
            beta1, beta2 = specification.get("betas", (0.9, 0.999))
            epsilon = float(specification.get("eps", 1.0e-8))
            learning_rate = float(specification["lr"])
            weight_decay = float(specification.get("weight_decay", 0.0))
            raw_step = state.get("step", 0)
            step = int(raw_step.item()) if torch.is_tensor(raw_step) else int(raw_step)
            average = state.get("exp_avg", torch.zeros_like(parameter))
            average_square = state.get(
                "exp_avg_sq", torch.zeros_like(parameter)
            )
            used = gradient.detach().to(parameter.dtype) * float(clip_scale)
            next_average = average.to(parameter.device) * beta1
            next_average = next_average + used * (1.0 - beta1)
            next_square = average_square.to(parameter.device) * beta2
            next_square = next_square + used.square() * (1.0 - beta2)
            next_step = step + 1
            corrected_average = next_average / max(
                1.0 - beta1**next_step, 1.0e-12
            )
            corrected_square = next_square / max(
                1.0 - beta2**next_step, 1.0e-12
            )
            update = (
                learning_rate
                * corrected_average
                / (corrected_square.sqrt() + epsilon)
            )
            if weight_decay:
                update = update + (
                    learning_rate * weight_decay * parameter.detach()
                )
            update_square += float(update.float().square().sum())
            parameter_square += float(parameter.detach().float().square().sum())
            states += int(bool(state))
        update_l2 = math.sqrt(update_square)
        parameter_l2 = math.sqrt(parameter_square)
        result[group_name] = {
            "predicted_update_l2": update_l2,
            "parameter_l2": parameter_l2,
            "predicted_update_to_parameter": update_l2
            / max(parameter_l2, 1.0e-12),
            "optimizer_state_entries": states,
        }
    return result


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg["dataset"].setdefault("lists", {})["train"] = str(
        Path(args.list_path).expanduser().resolve()
    )
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    freeze_except_parameter_prefixes(model, (PREFIX.rstrip("."),))
    set_frozen_detector_eval(model, (PREFIX.rstrip("."),))
    optimizer = build_optimizer(cfg, model)
    payload = _torch_load(args.checkpoint)
    optimizer_payload = payload.get("optimizer")
    if not isinstance(optimizer_payload, dict):
        raise ValueError("V15 endpoint checkpoint has no optimizer state")
    optimizer.load_state_dict(optimizer_payload)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(iteration)
    loader = build_dataloader(
        cfg, split="train", training=False, start_iteration=iteration
    )

    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _parameter in named]
    parameters = tuple(parameter for _name, parameter in named)
    if not names or not all(name.startswith(PREFIX) for name in names):
        raise ValueError("V15 gradient audit found an invalid trainable set")
    group_indices = {
        group: [index for index, name in enumerate(names) if _group(name) == group]
        for group in GROUPS
    }
    parameter_counts = {
        group: sum(int(parameters[index].numel()) for index in indices)
        for group, indices in group_indices.items()
    }
    clip_norm = float(cfg.get("training", {}).get("clip_grad_norm", 0.0))
    rows: list[dict[str, Any]] = []
    for batch_index, (images, targets, _metas) in enumerate(
        tqdm(loader, desc="V15 endpoint gradients", ncols=90)
    ):
        if batch_index >= int(args.batches):
            break
        images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        outputs, matches = forward_with_matches(
            model, images, targets, matcher, cfg, iteration
        )
        losses = criterion(outputs, targets, matches)
        visual = losses["loss_four_slot_v15_visual"]
        geometry = (
            float(criterion.cfg.four_slot_v15_point_weight)
            * losses["loss_four_slot_v15_point"]
            + float(criterion.cfg.four_slot_v15_range_weight)
            * losses["loss_four_slot_v15_range"]
            + float(criterion.cfg.four_slot_v15_line_iou_weight)
            * losses["loss_four_slot_v15_line_iou"]
            + float(criterion.cfg.four_slot_v15_dfl_weight)
            * losses["loss_four_slot_v15_dfl"]
        )
        components = {
            "visual": visual,
            "geometry": geometry,
            "total": losses["loss_four_slot_v15"],
        }
        gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        for component_index, (name, value) in enumerate(components.items()):
            gradients[name] = torch.autograd.grad(
                value,
                parameters,
                retain_graph=component_index + 1 < len(components),
                allow_unused=True,
            )
        total_norm = _norm(gradients["total"], list(range(len(parameters))))
        clip_scale = (
            min(1.0, clip_norm / max(total_norm, 1.0e-12))
            if clip_norm > 0.0
            else 1.0
        )
        predicted = _adam_update_ratio(
            optimizer,
            parameters,
            gradients["total"],
            group_indices,
            clip_scale,
        )
        group_rows: dict[str, Any] = {}
        for group, indices in group_indices.items():
            group_rows[group] = {
                "parameter_count": parameter_counts[group],
                "visual_grad_l2": _norm(gradients["visual"], indices),
                "geometry_grad_l2": _norm(gradients["geometry"], indices),
                "total_grad_l2": _norm(gradients["total"], indices),
                "total_grad_rms": _norm(gradients["total"], indices)
                / math.sqrt(max(parameter_counts[group], 1)),
                "visual_geometry_cosine": _cosine(
                    gradients["visual"], gradients["geometry"], indices
                ),
                **predicted[group],
            }
        rows.append(
            {
                "batch": batch_index,
                "loss_visual": float(visual.detach().cpu()),
                "loss_geometry": float(geometry.detach().cpu()),
                "loss_total": float(components["total"].detach().cpu()),
                "total_grad_l2_pre_clip": total_norm,
                "clip_scale": clip_scale,
                "total_grad_l2_post_clip": total_norm * clip_scale,
                "groups": group_rows,
            }
        )
    if not rows:
        raise ValueError("V15 endpoint gradient audit processed no batches")

    summaries: dict[str, Any] = {
        "total_grad_l2_pre_clip": _summary(
            [float(row["total_grad_l2_pre_clip"]) for row in rows]
        ),
        "clip_scale": _summary([float(row["clip_scale"]) for row in rows]),
        "total_grad_l2_post_clip": _summary(
            [float(row["total_grad_l2_post_clip"]) for row in rows]
        ),
        "groups": {},
    }
    for group in GROUPS:
        summaries["groups"][group] = {
            "parameter_count": parameter_counts[group],
            **{
                metric: _summary(
                    [float(row["groups"][group][metric]) for row in rows]
                )
                for metric in (
                    "visual_grad_l2",
                    "geometry_grad_l2",
                    "total_grad_l2",
                    "total_grad_rms",
                    "visual_geometry_cosine",
                    "predicted_update_l2",
                    "predicted_update_to_parameter",
                )
            },
            "optimizer_state_entries": max(
                int(row["groups"][group]["optimizer_state_entries"])
                for row in rows
            ),
        }
    checks = {
        "requested_batches_processed": len(rows) == int(args.batches),
        "only_v15_trainable": all(name.startswith(PREFIX) for name in names),
        "all_parameters_classified": parameter_counts["other"] == 0,
        "visual_loss_reaches_visual": summaries["groups"]["visual"][
            "visual_grad_l2"
        ]["mean"]
        > 0.0,
        "visual_loss_does_not_reach_graph": summaries["groups"]["graph"][
            "visual_grad_l2"
        ]["mean"]
        == 0.0,
        "geometry_reaches_visual": summaries["groups"]["visual"][
            "geometry_grad_l2"
        ]["mean"]
        > 0.0,
        "geometry_reaches_graph": summaries["groups"]["graph"][
            "geometry_grad_l2"
        ]["mean"]
        > 0.0,
        "geometry_reaches_fusion": summaries["groups"]["fusion"][
            "geometry_grad_l2"
        ]["mean"]
        > 0.0,
        "geometry_reaches_output": summaries["groups"]["output"][
            "geometry_grad_l2"
        ]["mean"]
        > 0.0,
        "adam_updates_all_live_groups": all(
            summaries["groups"][group]["predicted_update_l2"]["mean"] > 0.0
            for group in ("visual", "graph", "fusion", "output")
        ),
        "optimizer_state_present": all(
            summaries["groups"][group]["optimizer_state_entries"] > 0
            for group in ("visual", "graph", "fusion", "output")
        ),
        "all_values_finite": all(
            math.isfinite(float(value))
            for row in rows
            for value in (
                row["loss_visual"],
                row["loss_geometry"],
                row["loss_total"],
                row["total_grad_l2_pre_clip"],
                row["clip_scale"],
            )
        ),
        "test_closed": True,
    }
    report = {
        "experiment": "V15 endpoint gradient population",
        "iteration": iteration,
        "batches": len(rows),
        "group_parameter_counts": parameter_counts,
        "summaries": summaries,
        "rows": rows,
        "checks": checks,
        "passed": all(checks.values()),
        "optimizer_steps_during_audit": 0,
        "test_set_used": False,
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(destination), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
