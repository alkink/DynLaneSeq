from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
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
from dynlaneseq_eg.tools.train import seed_everything


SELECTOR = "structured_query_head.set_selection_head."
V12 = SELECTOR + "visual_first_association."
GROUPS = ("visual", "slot_set", "proposal", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact V7 deployment parity and V12 Stage-A gradient "
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
    local = name[len(V12) :] if name.startswith(V12) else name
    if local.startswith(
        (
            "feature_",
            "first_visual_",
            "second_visual_",
        )
    ):
        return "visual"
    if local.startswith(
        (
            "slot_",
            "row_position_projection.",
            "anchor_geometry_projection.",
            "initial_norm.",
            "cross_slot_",
            "vertical_encoder.",
            "visual_norm.",
            "visual_ffn.",
        )
    ):
        return "slot_set"
    if local.startswith(
        (
            "proposal_",
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
    if selection.get("four_slot_visual_first_association_enabled") is not True:
        raise ValueError("V12 visual-first association is disabled")
    if selection.get("four_slot_refinement_enabled") is not True:
        raise ValueError("V12 Stage A requires the exact V7 refiner")
    if float(cfg["loss"].get("w_four_slot_visual_first", 0.0)) != 1.0:
        raise ValueError("V12 visual-first loss must be the sole objective")
    nonzero_objectives = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and float(value) != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    if nonzero_objectives != {"w_four_slot_visual_first": 1.0}:
        raise ValueError(
            "V12 requires one objective; resolved nonzero weights are "
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
    images, targets, _metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)

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
            images,
            targets,
            source_matcher,
            source_cfg,
            source_iteration,
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
    source_values = {
        name: source_outputs[name].detach().clone() for name in parity_names
    }
    del source_outputs, source_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    if iteration != int(args.start_iteration):
        raise ValueError("V12 checkpoint iteration mismatch")
    freeze_stats = freeze_except_parameter_prefixes(
        model, (V12.rstrip("."),)
    )
    set_frozen_detector_eval(model, (V12.rstrip("."),))
    selector = model.structured_query_head.set_selection_head
    module = selector.visual_first_association
    if module is None or selector.slot_refinement is None:
        raise ValueError("V12 graph is incomplete")
    if "legacy_route_logits" in inspect.signature(module.forward).parameters:
        raise ValueError("V12 proposal association still accepts V7 route logits")

    captured: dict[str, torch.Tensor] = {}

    def capture(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        captured.update(kwargs)

    handle = module.register_forward_pre_hook(capture, with_kwargs=True)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(iteration)
    outputs, matches = forward_with_matches(
        model, images, targets, matcher, cfg, iteration
    )
    handle.remove()
    if not captured:
        raise RuntimeError("V12 pre-hook captured no inputs")
    losses = criterion(outputs, targets, matches)

    parity: dict[str, float] = {}
    for name, source_value in source_values.items():
        current = outputs[name].detach()
        if current.dtype == torch.bool or not current.dtype.is_floating_point:
            parity[name] = float((current != source_value).sum().cpu())
        else:
            parity[name] = float(
                (current.float() - source_value.float()).abs().max().cpu()
            )

    replay = module(**captured)
    replay_error = max(
        float(
            (
                replay["selection_slot_v12_proposal_attention"]
                - outputs["selection_slot_v12_proposal_attention"]
            ).abs().max().detach().cpu()
        ),
        float(
            (
                replay["selection_slot_v12_visual_attention"]
                - outputs["selection_slot_v12_visual_attention"]
            ).abs().max().detach().cpu()
        ),
    )
    wrong_kwargs = dict(captured)
    wrong_kwargs["row_value_features"] = torch.roll(
        captured["row_value_features"], shifts=1, dims=0
    )
    wrong = module(**wrong_kwargs)
    wrong_p2_proposal_change = float(
        (
            wrong["selection_slot_v12_proposal_attention"]
            - replay["selection_slot_v12_proposal_attention"]
        ).abs().mean().detach().cpu()
    )
    wrong_p2_visual_change = float(
        (
            wrong["selection_slot_v12_visual_attention"]
            - replay["selection_slot_v12_visual_attention"]
        ).abs().mean().detach().cpu()
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
    components = {
        "first_visual": losses["loss_four_slot_visual_first_first"],
        "final_visual": losses["loss_four_slot_visual_first_final"],
        "proposal": losses["loss_four_slot_visual_first_proposal"],
        "total": losses["loss_four_slot_visual_first"],
    }
    gradient_report: dict[str, dict[str, float]] = {}
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

    max_parity_error = max(parity.values(), default=0.0)
    finite_losses = all(
        bool(torch.isfinite(value.detach()).all()) for value in components.values()
    )
    checks = {
        "exact_v7_deployment_parity": max_parity_error <= 1.0e-6,
        "v12_replay_exact": replay_error <= 1.0e-7,
        "legacy_route_logit_absent": True,
        "wrong_p2_changes_visual_state": wrong_p2_visual_change > 1.0e-8,
        "wrong_p2_changes_proposal_association": (
            wrong_p2_proposal_change > 1.0e-8
        ),
        "first_loss_reaches_visual": (
            gradient_report["first_visual"]["visual"] > 0.0
        ),
        "final_loss_reaches_visual_and_set": (
            gradient_report["final_visual"]["visual"] > 0.0
            and gradient_report["final_visual"]["slot_set"] > 0.0
        ),
        "proposal_loss_reaches_visual_set_and_proposals": (
            gradient_report["proposal"]["visual"] > 0.0
            and gradient_report["proposal"]["slot_set"] > 0.0
            and gradient_report["proposal"]["proposal"] > 0.0
        ),
        "losses_finite": finite_losses,
        "only_v12_trainable": all(name.startswith(V12) for name in names),
    }
    report = {
        "experiment": "V12 visual-first Stage-A zero-step contract",
        "iteration": iteration,
        "source_iteration": source_iteration,
        "deployment_mode": "exact_v7",
        "legacy_route_logits_used_by_v12": False,
        "parity": parity,
        "max_parity_error": max_parity_error,
        "v12_replay_max_abs_error": replay_error,
        "wrong_p2_mean_visual_attention_change": wrong_p2_visual_change,
        "wrong_p2_mean_proposal_attention_change": wrong_p2_proposal_change,
        "gradients": gradient_report,
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
