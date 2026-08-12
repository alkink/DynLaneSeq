from __future__ import annotations

import argparse
import copy
import gc
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
ROUTE_PREFIXES = (SELECTOR + "slot_query.", SELECTOR + "candidate_key.")
ACTIVE_PREFIX = SELECTOR + "active."
REFINER_PREFIX = SELECTOR + "slot_refinement."
CANDIDATE_PREFIXES = (
    SELECTOR + "input_norm.",
    SELECTOR + "input_projection.",
    SELECTOR + "proposal_encoder.",
    SELECTOR + "candidate_norm.",
)
SLOT_PREFIXES = (
    SELECTOR + "slot_decoder.",
    SELECTOR + "slot_tokens.",
    SELECTOR + "slot_norm.",
)
PARITY_TENSORS = (
    "selection_slot_real_route_logits",
    "selection_slot_geometry_route_indices",
    "selection_slot_active_logits",
    "selection_slot_indices",
    "selection_slot_scores",
    "selection_slot_input_reference_x_rows",
    "selection_slot_input_range_norm",
    "selection_slot_pred_x_rows",
    "selection_slot_range_norm",
    "selection_slot_active",
)
GROUPS = (
    "candidate_trunk",
    "slot_trunk",
    "route_projection",
    "active_head",
    "refiner",
    "other_selector",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Population forward-parity, gradient-alignment and clipping audit "
            "for the V8.1 geometry-to-global-router-state causal edge."
        )
    )
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--treatment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--start-iteration", type=int, default=225000)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-router-ratio", type=float, default=0.05)
    parser.add_argument("--max-router-ratio", type=float, default=0.50)
    parser.add_argument("--min-router-cosine", type=float, default=0.05)
    parser.add_argument("--max-negative-cosine-fraction", type=float, default=0.35)
    parser.add_argument("--max-median-clip-scale-relative-delta", type=float, default=0.10)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _prepare_config(path: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(Path(args.dataset_root).expanduser())
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _without_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    # The comparison below temporarily normalizes the treatment edge to the
    # control value.  Keep that mutation isolated from the live config used to
    # build the treatment model.
    value = copy.deepcopy(cfg)
    value.pop("_config_path", None)
    value.pop("output_dir", None)
    return value


def _group(name: str) -> str:
    if name.startswith(ACTIVE_PREFIX):
        return "active_head"
    if name.startswith(ROUTE_PREFIXES):
        return "route_projection"
    if name.startswith(REFINER_PREFIX):
        return "refiner"
    if name.startswith(CANDIDATE_PREFIXES):
        return "candidate_trunk"
    if name.startswith(SLOT_PREFIXES):
        return "slot_trunk"
    if name.startswith(SELECTOR):
        return "other_selector"
    raise ValueError(f"unexpected trainable parameter outside selector: {name}")


def _rng_snapshot(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def _snapshot(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: outputs[name].detach().cpu().clone() for name in PARITY_TENSORS}


def _parity_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.dtype == torch.bool or not left.is_floating_point():
        return float((left != right).sum())
    return float((left.float() - right.float()).abs().max())


def _gradients(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _norm(grads: tuple[torch.Tensor | None, ...], ids: list[int]) -> float:
    total = 0.0
    for index in ids:
        gradient = grads[index]
        if gradient is not None:
            total += float(gradient.detach().float().square().sum())
    return math.sqrt(total)


def _cosine(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    ids: list[int],
) -> float:
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    for index in ids:
        left_gradient = left[index]
        right_gradient = right[index]
        if left_gradient is not None:
            left_square += float(left_gradient.detach().float().square().sum())
        if right_gradient is not None:
            right_square += float(right_gradient.detach().float().square().sum())
        if left_gradient is not None and right_gradient is not None:
            dot += float(
                (
                    left_gradient.detach().float()
                    * right_gradient.detach().float()
                ).sum()
            )
    denominator = math.sqrt(left_square * right_square)
    return 0.0 if denominator <= 0.0 else dot / denominator


def _combined_norm(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
) -> float:
    total = 0.0
    for left, right in zip(first, second):
        if left is None and right is None:
            continue
        if left is None:
            value = right.detach().float()
        elif right is None:
            value = left.detach().float()
        else:
            value = left.detach().float() + right.detach().float()
        total += float(value.square().sum())
    return math.sqrt(total)


def _max_gradient_difference(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    ids: list[int],
) -> float:
    maximum = 0.0
    for index in ids:
        left_gradient = left[index]
        right_gradient = right[index]
        if left_gradient is None and right_gradient is None:
            continue
        if left_gradient is None or right_gradient is None:
            return math.inf
        maximum = max(
            maximum,
            float(
                (
                    left_gradient.detach().float()
                    - right_gradient.detach().float()
                ).abs().max()
            ),
        )
    return maximum


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def main() -> None:
    args = parse_args()
    if int(args.batches) < 1:
        raise ValueError("--batches must be positive")
    control_cfg = _prepare_config(args.control_config, args)
    treatment_cfg = _prepare_config(args.treatment_config, args)
    control_selection = control_cfg["model"]["structured_query"]["set_selection"]
    treatment_selection = treatment_cfg["model"]["structured_query"]["set_selection"]
    if control_selection.get("four_slot_geometry_detach_router_states") is not True:
        raise ValueError("control must detach geometry router states")
    if treatment_selection.get("four_slot_geometry_detach_router_states") is not False:
        raise ValueError("treatment must attach geometry router states")
    normalized_treatment = _without_metadata(treatment_cfg)
    normalized_treatment["model"]["structured_query"]["set_selection"][
        "four_slot_geometry_detach_router_states"
    ] = True
    if _without_metadata(control_cfg) != normalized_treatment:
        raise ValueError("control/treatment configs differ beyond the causal edge")

    seed_everything(int(treatment_cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(treatment_cfg).to(device)
    iteration = load_checkpoint(args.checkpoint, model, strict=False)
    if int(iteration) != int(args.start_iteration):
        raise ValueError(
            f"expected checkpoint iteration {args.start_iteration}, found {iteration}"
        )
    freeze_stats = freeze_except_parameter_prefixes(model, (SELECTOR.rstrip("."),))
    set_frozen_detector_eval(model, (SELECTOR.rstrip("."),))
    selector = model.structured_query_head.set_selection_head
    matcher = build_matcher(treatment_cfg)
    criterion = build_criterion(treatment_cfg).to(device)
    criterion.set_iteration(int(iteration))
    loader = build_dataloader(
        treatment_cfg,
        split="train",
        training=True,
        start_iteration=int(iteration),
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
    router_ids = (
        group_ids["candidate_trunk"]
        + group_ids["slot_trunk"]
        + group_ids["route_projection"]
        + group_ids["other_selector"]
    )
    parity = {name: 0.0 for name in PARITY_TENSORS}
    rows: list[dict[str, Any]] = []

    iterator = iter(loader)
    for batch_index in tqdm(range(int(args.batches)), desc="V8.1 gradient population"):
        try:
            images, targets, _metas = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets, _metas = next(iterator)
        images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        rng = _rng_snapshot(device)

        selector.geometry_detach_router_states = True
        _restore_rng(rng, device)
        control_outputs, control_matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            control_cfg,
            int(iteration),
        )
        criterion.set_iteration(int(iteration))
        control_losses = criterion(control_outputs, targets, control_matches)
        control_snapshot = _snapshot(control_outputs)
        control_geometry = _gradients(
            control_losses["loss_four_slot_geometry"],
            parameters,
            retain_graph=False,
        )
        del control_outputs, control_matches, control_losses

        selector.geometry_detach_router_states = False
        _restore_rng(rng, device)
        treatment_outputs, treatment_matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            treatment_cfg,
            int(iteration),
        )
        criterion.set_iteration(int(iteration))
        treatment_losses = criterion(
            treatment_outputs,
            targets,
            treatment_matches,
        )
        treatment_snapshot = _snapshot(treatment_outputs)
        for name in PARITY_TENSORS:
            parity[name] = max(
                parity[name],
                _parity_delta(control_snapshot[name], treatment_snapshot[name]),
            )
        treatment_geometry = _gradients(
            treatment_losses["loss_four_slot_geometry"],
            parameters,
            retain_graph=True,
        )
        selection = _gradients(
            treatment_losses["loss_four_slot_selection"],
            parameters,
            retain_graph=False,
        )

        group_row: dict[str, Any] = {}
        for group in GROUPS:
            ids = group_ids[group]
            geometry_norm = _norm(treatment_geometry, ids)
            selection_norm = _norm(selection, ids)
            group_row[group] = {
                "geometry_norm": geometry_norm,
                "selection_norm": selection_norm,
                "ratio": geometry_norm / max(selection_norm, 1.0e-12),
                "cosine": _cosine(treatment_geometry, selection, ids),
            }
        router_geometry_norm = _norm(treatment_geometry, router_ids)
        router_selection_norm = _norm(selection, router_ids)
        router_cosine = _cosine(treatment_geometry, selection, router_ids)
        control_norm = _combined_norm(selection, control_geometry)
        treatment_norm = _combined_norm(selection, treatment_geometry)
        clip = float(args.clip_grad_norm)
        control_scale = min(1.0, clip / max(control_norm, 1.0e-12))
        treatment_scale = min(1.0, clip / max(treatment_norm, 1.0e-12))
        rows.append(
            {
                "batch": batch_index,
                "loss_selection": float(
                    treatment_losses["loss_four_slot_selection"].detach()
                ),
                "loss_geometry": float(
                    treatment_losses["loss_four_slot_geometry"].detach()
                ),
                "groups": group_row,
                "router_state": {
                    "geometry_norm": router_geometry_norm,
                    "selection_norm": router_selection_norm,
                    "ratio": router_geometry_norm
                    / max(router_selection_norm, 1.0e-12),
                    "cosine": router_cosine,
                },
                "control_total_grad_norm": control_norm,
                "treatment_total_grad_norm": treatment_norm,
                "control_clip_scale": control_scale,
                "treatment_clip_scale": treatment_scale,
                "clip_scale_relative_delta": abs(treatment_scale - control_scale)
                / max(control_scale, 1.0e-12),
                "route_projection_control_treatment_gradient_max_abs_delta": (
                    _max_gradient_difference(
                        control_geometry,
                        treatment_geometry,
                        group_ids["route_projection"],
                    )
                ),
            }
        )
        del treatment_outputs, treatment_matches, treatment_losses
        del control_geometry, treatment_geometry, selection
        gc.collect()

    def values(path: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            result.append(float(value))
        return result

    router_ratio = values(("router_state", "ratio"))
    router_cosine = values(("router_state", "cosine"))
    clip_delta = values(("clip_scale_relative_delta",))
    group_population = {
        group: {
            metric: _summary(values(("groups", group, metric)))
            for metric in ("geometry_norm", "selection_norm", "ratio", "cosine")
        }
        for group in GROUPS
    }
    median_ratio = float(np.median(router_ratio))
    median_cosine = float(np.median(router_cosine))
    negative_fraction = float(np.mean(np.asarray(router_cosine) < 0.0))
    median_clip_delta = float(np.median(clip_delta))
    checks = {
        "checkpoint_iteration": int(iteration) == int(args.start_iteration),
        "configs_differ_only_by_backward_edge": True,
        "all_forward_tensors_exact": all(value == 0.0 for value in parity.values()),
        "geometry_candidate_trunk_positive": (
            group_population["candidate_trunk"]["geometry_norm"]["median"] > 0.0
        ),
        "geometry_slot_trunk_positive": (
            group_population["slot_trunk"]["geometry_norm"]["median"] > 0.0
        ),
        "geometry_route_projection_positive": (
            group_population["route_projection"]["geometry_norm"]["median"] > 0.0
        ),
        "geometry_active_head_exact_zero": (
            group_population["active_head"]["geometry_norm"]["p90"] == 0.0
        ),
        "proposal_detector_frozen": int(
            freeze_stats["trainable_tensor_count"]
        ) == len(named_parameters),
        "router_ratio_in_band": float(args.min_router_ratio)
        <= median_ratio
        <= float(args.max_router_ratio),
        "router_cosine_sufficient": median_cosine >= float(args.min_router_cosine),
        "negative_cosine_fraction_controlled": negative_fraction
        <= float(args.max_negative_cosine_fraction),
        "clip_scale_coupling_controlled": median_clip_delta
        <= float(args.max_median_clip_scale_relative_delta),
        "route_projection_gradient_preserved": max(
            values(("route_projection_control_treatment_gradient_max_abs_delta",))
        ) <= 1.0e-6,
    }
    payload = {
        "experiment": "V8.1 geometry-to-global-router-state population contract",
        "diagnostic_only": True,
        "test_set_used": False,
        "training_steps": 0,
        "control_config": str(Path(args.control_config).resolve()),
        "treatment_config": str(Path(args.treatment_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "iteration": int(iteration),
        "batches": len(rows),
        "batch_size": int(args.batch_size),
        "forward_parity": parity,
        "gradient_population": {
            "groups": group_population,
            "router_state_ratio": _summary(router_ratio),
            "router_state_cosine": _summary(router_cosine),
            "negative_router_cosine_fraction": negative_fraction,
        },
        "clipping": {
            "clip_grad_norm": float(args.clip_grad_norm),
            "control_total_grad_norm": _summary(
                values(("control_total_grad_norm",))
            ),
            "treatment_total_grad_norm": _summary(
                values(("treatment_total_grad_norm",))
            ),
            "control_clip_scale": _summary(values(("control_clip_scale",))),
            "treatment_clip_scale": _summary(
                values(("treatment_clip_scale",))
            ),
            "relative_scale_delta": _summary(clip_delta),
        },
        "gate": {
            "min_router_ratio": float(args.min_router_ratio),
            "max_router_ratio": float(args.max_router_ratio),
            "min_router_cosine": float(args.min_router_cosine),
            "max_negative_cosine_fraction": float(
                args.max_negative_cosine_fraction
            ),
            "max_median_clip_scale_relative_delta": float(
                args.max_median_clip_scale_relative_delta
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "per_batch": rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment": payload["experiment"],
                "forward_parity": parity,
                "gradient_population": payload["gradient_population"],
                "clipping": payload["clipping"],
                "checks": checks,
                "passed": payload["passed"],
            },
            indent=2,
        )
    )
    print(f"output_json: {output}")
    if not payload["passed"]:
        raise SystemExit("V8.1 population gradient contract failed")


if __name__ == "__main__":
    main()
