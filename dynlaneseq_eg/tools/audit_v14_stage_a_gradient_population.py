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
    "corrected_visual_first_association."
)
GROUPS = ("visual", "slot_row", "proposal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure V14 Stage-A component gradients, clipping and the next "
            "AdamW update implied by the endpoint optimizer moments."
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
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
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
    return "proposal"


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
        first = first.detach().float()
        second = second.detach().float()
        dot += float((first * second).sum())
        left_square += float(first.square().sum())
        right_square += float(second.square().sum())
    return dot / max(math.sqrt(left_square * right_square), 1.0e-12)


def _adam_update_ratio(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    gradients: tuple[torch.Tensor | None, ...],
    group_indices: dict[str, list[int]],
    clip_scale: float,
) -> dict[str, dict[str, float]]:
    hyper: dict[torch.nn.Parameter, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            hyper[parameter] = group
    update_squares = {group: 0.0 for group in GROUPS}
    parameter_squares = {group: 0.0 for group in GROUPS}
    state_entries = {group: 0 for group in GROUPS}
    for group, indices in group_indices.items():
        for index in indices:
            gradient = gradients[index]
            parameter = parameters[index]
            if gradient is None:
                continue
            spec = hyper[parameter]
            state = optimizer.state.get(parameter, {})
            beta1, beta2 = spec.get("betas", (0.9, 0.999))
            epsilon = float(spec.get("eps", 1.0e-8))
            lr = float(spec["lr"])
            weight_decay = float(spec.get("weight_decay", 0.0))
            raw_step = state.get("step", 0)
            step = int(raw_step.item()) if torch.is_tensor(raw_step) else int(raw_step)
            exp_avg = state.get("exp_avg", torch.zeros_like(parameter))
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter))
            used_gradient = gradient.detach().to(parameter.dtype) * float(clip_scale)
            next_avg = exp_avg.to(parameter.device) * beta1
            next_avg = next_avg + used_gradient * (1.0 - beta1)
            next_sq = exp_avg_sq.to(parameter.device) * beta2
            next_sq = next_sq + used_gradient.square() * (1.0 - beta2)
            next_step = step + 1
            corrected_avg = next_avg / max(1.0 - beta1**next_step, 1.0e-12)
            corrected_sq = next_sq / max(1.0 - beta2**next_step, 1.0e-12)
            update = lr * corrected_avg / (corrected_sq.sqrt() + epsilon)
            if weight_decay:
                update = update + lr * weight_decay * parameter.detach()
            update_squares[group] += float(update.float().square().sum())
            parameter_squares[group] += float(
                parameter.detach().float().square().sum()
            )
            state_entries[group] += int(bool(state))
    return {
        group: {
            "predicted_update_l2": math.sqrt(update_squares[group]),
            "parameter_l2": math.sqrt(parameter_squares[group]),
            "predicted_update_to_parameter": math.sqrt(update_squares[group])
            / max(math.sqrt(parameter_squares[group]), 1.0e-12),
            "optimizer_state_entries": state_entries[group],
        }
        for group in GROUPS
    }


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(Path(args.dataset_root).expanduser())
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
    if "optimizer" not in payload:
        raise ValueError("endpoint checkpoint has no optimizer state")
    optimizer.load_state_dict(payload["optimizer"])
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
    names = [name for name, _ in named]
    parameters = tuple(parameter for _, parameter in named)
    if not names or not all(name.startswith(PREFIX) for name in names):
        raise ValueError("V14 gradient audit found an invalid trainable set")
    group_indices = {
        group: [index for index, name in enumerate(names) if _group(name) == group]
        for group in GROUPS
    }
    group_parameter_count = {
        group: sum(int(parameters[index].numel()) for index in indices)
        for group, indices in group_indices.items()
    }
    rows: list[dict[str, Any]] = []
    clip_norm = float(cfg.get("training", {}).get("clip_grad_norm", 0.0))
    for batch_index, (images, targets, _metas) in enumerate(
        tqdm(loader, desc="V14 gradient population", ncols=90)
    ):
        if batch_index >= int(args.batches):
            break
        images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        outputs, matches = forward_with_matches(
            model, images, targets, matcher, cfg, iteration
        )
        losses = criterion(outputs, targets, matches)
        components = {
            "visual": losses["loss_four_slot_v14_visual"],
            "association": losses["loss_four_slot_v14_association"],
            "total": losses["loss_four_slot_v14_stage_a"],
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
        groups: dict[str, Any] = {}
        predicted = _adam_update_ratio(
            optimizer,
            parameters,
            gradients["total"],
            group_indices,
            clip_scale,
        )
        for group, indices in group_indices.items():
            groups[group] = {
                "parameter_count": group_parameter_count[group],
                "visual_grad_l2": _norm(gradients["visual"], indices),
                "association_grad_l2": _norm(
                    gradients["association"], indices
                ),
                "total_grad_l2": _norm(gradients["total"], indices),
                "total_grad_rms": _norm(gradients["total"], indices)
                / math.sqrt(max(group_parameter_count[group], 1)),
                "visual_association_cosine": _cosine(
                    gradients["visual"], gradients["association"], indices
                ),
                **predicted[group],
            }
        rows.append(
            {
                "batch": batch_index,
                "loss_visual": float(components["visual"].detach().cpu()),
                "loss_association": float(
                    components["association"].detach().cpu()
                ),
                "total_grad_l2_pre_clip": total_norm,
                "clip_scale": clip_scale,
                "total_grad_l2_post_clip": total_norm * clip_scale,
                "groups": groups,
            }
        )

    population = {
        group: {
            key: _summary([float(row["groups"][group][key]) for row in rows])
            for key in (
                "visual_grad_l2",
                "association_grad_l2",
                "total_grad_l2",
                "total_grad_rms",
                "visual_association_cosine",
                "predicted_update_l2",
                "parameter_l2",
                "predicted_update_to_parameter",
            )
        }
        for group in GROUPS
    }
    report = {
        "experiment": "V14 Stage-A endpoint gradient and AdamW population",
        "iteration": iteration,
        "batches": len(rows),
        "batch_size": int(args.batch_size),
        "parameter_groups": population,
        "clipping": {
            "clip_grad_norm": clip_norm,
            "pre_clip_norm": _summary(
                [float(row["total_grad_l2_pre_clip"]) for row in rows]
            ),
            "post_clip_norm": _summary(
                [float(row["total_grad_l2_post_clip"]) for row in rows]
            ),
            "clip_scale": _summary([float(row["clip_scale"]) for row in rows]),
            "clip_active_fraction": float(
                np.mean([float(row["clip_scale"]) < 1.0 for row in rows])
            )
            if rows
            else 0.0,
        },
        "per_batch": rows,
        "optimizer_steps_during_audit": 0,
        "test_set_used": False,
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_batch"}, indent=2))
    print(f"output_json: {destination}")


if __name__ == "__main__":
    main()
