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
VISUAL = SELECTOR + "global_visual_geometry."
ACTIVE = SELECTOR + "active."
ROUTE = (SELECTOR + "slot_query.", SELECTOR + "candidate_key.")
CANDIDATE = (
    SELECTOR + "input_norm.",
    SELECTOR + "input_projection.",
    SELECTOR + "proposal_encoder.",
    SELECTOR + "candidate_norm.",
)
SLOT = (
    SELECTOR + "slot_decoder.",
    SELECTOR + "slot_tokens.",
    SELECTOR + "slot_norm.",
)
GROUPS = (
    "global_visual_geometry",
    "candidate_trunk",
    "slot_trunk",
    "route_projection",
    "active_head",
    "other_selector",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Population gradient/topology audit for V10 full-width visual "
            "slot geometry before any optimizer step."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--start-iteration", type=int, default=225000)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _group(name: str) -> str:
    if name.startswith(VISUAL):
        return "global_visual_geometry"
    if name.startswith(ACTIVE):
        return "active_head"
    if name.startswith(ROUTE):
        return "route_projection"
    if name.startswith(CANDIDATE):
        return "candidate_trunk"
    if name.startswith(SLOT):
        return "slot_trunk"
    if name.startswith(SELECTOR):
        return "other_selector"
    raise ValueError(f"unexpected trainable parameter outside selector: {name}")


def _norm(
    gradients: tuple[torch.Tensor | None, ...],
    indices: list[int],
) -> float:
    total = 0.0
    for index in indices:
        gradient = gradients[index]
        if gradient is not None:
            total += float(gradient.detach().float().square().sum())
    return math.sqrt(total)


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
        if first is not None:
            left_square += float(first.detach().float().square().sum())
        if second is not None:
            right_square += float(second.detach().float().square().sum())
        if first is not None and second is not None:
            dot += float((first.detach().float() * second.detach().float()).sum())
    denominator = math.sqrt(left_square * right_square)
    return 0.0 if denominator <= 0.0 else dot / denominator


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def main() -> None:
    args = parse_args()
    if int(args.batches) < 1:
        raise ValueError("--batches must be positive")
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    selection_cfg = cfg["model"]["structured_query"]["set_selection"]
    if selection_cfg.get("four_slot_global_visual_geometry_enabled") is not True:
        raise ValueError("V10 global visual geometry is not enabled")
    if selection_cfg.get("four_slot_refinement_enabled") is not False:
        raise ValueError("legacy refinement must be disabled")
    if selection_cfg.get("four_slot_slot_owned_geometry_enabled") is not False:
        raise ValueError("V9 proposal-memory geometry must be disabled")

    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    if iteration != int(args.start_iteration):
        raise ValueError(
            f"expected iteration {args.start_iteration}, found {iteration}"
        )
    freeze_stats = freeze_except_parameter_prefixes(
        model,
        (SELECTOR.rstrip("."),),
    )
    set_frozen_detector_eval(model, (SELECTOR.rstrip("."),))
    selector = model.structured_query_head.set_selection_head
    if selector.global_visual_geometry is None:
        raise ValueError("built model has no V10 visual module")
    if selector.slot_owned_geometry is not None or selector.slot_refinement is not None:
        raise ValueError("V10 unexpectedly contains proposal-based geometry")

    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(iteration)
    loader = build_dataloader(
        cfg,
        split="train",
        training=True,
        start_iteration=iteration,
    )
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    names = tuple(name for name, _parameter in named_parameters)
    parameters = tuple(parameter for _name, parameter in named_parameters)
    group_ids = {
        group: [index for index, name in enumerate(names) if _group(name) == group]
        for group in GROUPS
    }
    shared_ids = group_ids["candidate_trunk"] + group_ids["slot_trunk"]
    rows: list[dict[str, Any]] = []
    iterator = iter(loader)
    for batch_index in tqdm(range(int(args.batches)), desc="V10 contract"):
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
        geometry = torch.autograd.grad(
            losses["loss_four_slot_geometry"],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        selection = torch.autograd.grad(
            losses["loss_four_slot_selection"],
            parameters,
            retain_graph=False,
            allow_unused=True,
        )
        attention = outputs["selection_slot_visual_attention"].detach().float()
        attention_error = float(
            (attention.sum(dim=-1) - 1.0).abs().max().cpu()
        )
        hard_indices = outputs[
            "selection_slot_geometry_route_indices"
        ].detach().clamp(min=0)
        proposal_x = outputs["pred_x_rows"].detach().float()
        hard_x = proposal_x.gather(
            1,
            hard_indices.unsqueeze(-1).expand(
                -1,
                -1,
                int(proposal_x.shape[-1]),
            ),
        )
        direct_reference = outputs[
            "selection_slot_input_reference_x_rows"
        ].detach().float()
        groups: dict[str, Any] = {}
        for group in GROUPS:
            indices = group_ids[group]
            groups[group] = {
                "geometry_norm": _norm(geometry, indices),
                "selection_norm": _norm(selection, indices),
                "cosine": _cosine(geometry, selection, indices),
            }
        rows.append(
            {
                "batch": batch_index,
                "loss_geometry": float(
                    losses["loss_four_slot_geometry"].detach().cpu()
                ),
                "loss_selection": float(
                    losses["loss_four_slot_selection"].detach().cpu()
                ),
                "groups": groups,
                "shared_cosine": _cosine(geometry, selection, shared_ids),
                "attention_probability_sum_max_abs_error": attention_error,
                "attention_finite": bool(torch.isfinite(attention).all()),
                "geometry_valid_fraction": float(
                    outputs["selection_slot_geometry_valid"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "mean_reference_distance_from_hard_proposal_px": float(
                    (direct_reference - hard_x).abs().mean().cpu()
                ),
            }
        )
        del outputs, matches, losses, geometry, selection

    group_summary: dict[str, Any] = {}
    for group in GROUPS:
        group_summary[group] = {
            "geometry_norm": _summary(
                [row["groups"][group]["geometry_norm"] for row in rows]
            ),
            "selection_norm": _summary(
                [row["groups"][group]["selection_norm"] for row in rows]
            ),
            "cosine": _summary(
                [row["groups"][group]["cosine"] for row in rows]
            ),
        }
    checks = {
        "only_selector_trainable": all(name.startswith(SELECTOR) for name in names),
        "visual_geometry_receives_geometry": min(
            row["groups"]["global_visual_geometry"]["geometry_norm"]
            for row in rows
        ) > 0.0,
        "candidate_trunk_receives_geometry": min(
            row["groups"]["candidate_trunk"]["geometry_norm"] for row in rows
        ) > 0.0,
        "slot_trunk_receives_geometry": min(
            row["groups"]["slot_trunk"]["geometry_norm"] for row in rows
        ) > 0.0,
        "active_head_is_geometry_isolated": max(
            row["groups"]["active_head"]["geometry_norm"] for row in rows
        ) == 0.0,
        "route_projection_is_geometry_isolated": max(
            row["groups"]["route_projection"]["geometry_norm"] for row in rows
        ) == 0.0,
        "visual_geometry_is_selection_isolated": max(
            row["groups"]["global_visual_geometry"]["selection_norm"]
            for row in rows
        ) == 0.0,
        "full_width_attention_is_normalized": max(
            row["attention_probability_sum_max_abs_error"] for row in rows
        ) <= 1.0e-5,
        "full_width_attention_is_finite": all(
            row["attention_finite"] for row in rows
        ),
        "all_four_geometry_hypotheses_valid": min(
            row["geometry_valid_fraction"] for row in rows
        ) == 1.0,
        "geometry_is_not_hard_proposal_gather": min(
            row["mean_reference_distance_from_hard_proposal_px"] for row in rows
        ) >= 1.0,
        "losses_are_finite": all(
            math.isfinite(row["loss_geometry"])
            and math.isfinite(row["loss_selection"])
            for row in rows
        ),
    }
    report = {
        "experiment": "V10 zero-step global visual geometry contract",
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "iteration": iteration,
        "batches": len(rows),
        "freeze_stats": freeze_stats,
        "trainable_parameter_tensors": len(names),
        "checks": checks,
        "passed": all(checks.values()),
        "group_summary": group_summary,
        "shared_geometry_selection_cosine": _summary(
            [row["shared_cosine"] for row in rows]
        ),
        "reference_distance_from_hard_proposal_px": _summary(
            [row["mean_reference_distance_from_hard_proposal_px"] for row in rows]
        ),
        "per_batch": rows,
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {output_json}")
    if report["passed"] is not True:
        raise SystemExit("V10 zero-step contract failed")


if __name__ == "__main__":
    main()
