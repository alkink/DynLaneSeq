from __future__ import annotations

import argparse
import hashlib
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
V13 = SELECTOR + "visual_precision_geometry."
GROUPS = ("proposal", "local_p2", "slot_row", "heads", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V13 parity, topology and full-width geometry support."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start-iteration", type=int, default=227000)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _configure(
    path: str, *, dataset_root: str, batch_size: int, num_workers: int
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


def _state_digest(
    state: dict[str, torch.Tensor], names: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    for name in names:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _group(name: str) -> str:
    local = name[len(V13) :] if name.startswith(V13) else name
    if local.startswith("proposal_"):
        return "proposal"
    if local.startswith(("local_",)):
        return "local_p2"
    if local.startswith(
        (
            "visual_",
            "geometry_projection.",
            "fusion_",
            "cross_slot_",
            "vertical_encoder.",
            "output_norm.",
        )
    ):
        return "slot_row"
    if local.startswith(
        ("candidate_", "delta_head.", "range_delta_head.")
    ):
        return "heads"
    return "other"


def _norm(
    gradients: tuple[torch.Tensor | None, ...], indices: list[int]
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
    if selection.get("four_slot_visual_precision_geometry_enabled") is not True:
        raise ValueError("V13 precision geometry is disabled")
    nonzero_objectives = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and float(value) != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    if nonzero_objectives != {"w_four_slot_visual_precision": 1.0}:
        raise ValueError(
            "V13 requires one objective; resolved weights are "
            f"{nonzero_objectives}"
        )

    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=args.start_iteration
    )
    images, targets, _metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)

    source_model = build_model(source_cfg).to(device)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source_model, strict=False)
    )
    if source_iteration != int(args.start_iteration):
        raise ValueError("V12 source iteration mismatch")
    source_names = tuple(sorted(source_model.state_dict()))
    source_sha = _state_digest(source_model.state_dict(), source_names)
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
    categorical_names = (
        "selection_slot_geometry_route_indices",
        "selection_slot_indices",
        "selection_slot_active",
    )
    source_categorical = {
        name: source_outputs[name].detach().clone()
        for name in categorical_names
    }
    del source_outputs, source_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    if iteration != int(args.start_iteration):
        raise ValueError("V13 checkpoint iteration mismatch")
    freeze_stats = freeze_except_parameter_prefixes(
        model, (V13.rstrip("."),)
    )
    set_frozen_detector_eval(model, (V13.rstrip("."),))
    selector = model.structured_query_head.set_selection_head
    module = selector.visual_precision_geometry
    if module is None or selector.visual_first_association is None:
        raise ValueError("V13 graph is incomplete")
    forbidden = {"route_indices", "route_logits", "legacy_route_logits"}
    if forbidden.intersection(inspect.signature(module.forward).parameters):
        raise ValueError("V13 geometry still accepts a proposal-ID route")

    current_state = model.state_dict()
    missing_source = tuple(name for name in source_names if name not in current_state)
    current_sha = (
        _state_digest(current_state, source_names) if not missing_source else ""
    )
    legacy_state_exact = not missing_source and current_sha == source_sha

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
        raise RuntimeError("V13 pre-hook captured no inputs")
    losses = criterion(outputs, targets, matches)

    anchor_x_error = float(
        (
            outputs["selection_slot_pred_x_rows"].detach()
            - outputs["selection_slot_v13_anchor_x_rows"].detach()
        ).abs().max().cpu()
    )
    anchor_range_error = float(
        (
            outputs["selection_slot_range_norm"].detach()
            - outputs["selection_slot_v13_anchor_range_norm"].detach()
        ).abs().max().cpu()
    )
    categorical_mismatches = {
        name: int((outputs[name].detach() != value).sum().cpu())
        for name, value in source_categorical.items()
    }
    replay = module(**captured)
    activity_score_overlap = sorted(
        {
            "selection_slot_active",
            "selection_slot_active_logits",
            "selection_slot_scores",
            "selection_slot_indices",
        }.intersection(replay)
    )
    wrong_kwargs = dict(captured)
    wrong_kwargs["row_value_features"] = torch.roll(
        captured["row_value_features"], shifts=1, dims=0
    )
    wrong = module(**wrong_kwargs)
    wrong_hidden_change = float(
        (
            wrong["selection_slot_v13_hidden"]
            - replay["selection_slot_v13_hidden"]
        ).abs().mean().detach().cpu()
    )
    wrong_local_change = float(
        (
            wrong["selection_slot_v13_local_attention"]
            - replay["selection_slot_v13_local_attention"]
        ).abs().mean().detach().cpu()
    )

    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _parameter in named]
    parameters = tuple(parameter for _name, parameter in named)
    group_indices = {
        group: [index for index, name in enumerate(names) if _group(name) == group]
        for group in GROUPS
    }
    gradients = torch.autograd.grad(
        losses["loss_four_slot_visual_precision"],
        parameters,
        allow_unused=True,
    )
    gradient_report = {
        group: _norm(gradients, group_indices[group]) for group in GROUPS
    }
    offsets = module.delta_offsets_px.detach().float()
    checks = {
        "legacy_v12_state_bit_exact": legacy_state_exact,
        "v13_starts_at_exact_v7_x": anchor_x_error == 0.0,
        "v13_starts_at_exact_v7_range": anchor_range_error == 0.0,
        "source_and_v13_categorical_outputs_exact": all(
            value == 0 for value in categorical_mismatches.values()
        ),
        "v13_does_not_return_activity_score_or_indices": (
            not activity_score_overlap
        ),
        "hard_proposal_id_absent_from_v13": True,
        "full_width_movement_support": (
            float(offsets.min()) <= -float(cfg["model"]["input_w"])
            and float(offsets.max()) >= float(cfg["model"]["input_w"])
        ),
        "wrong_p2_changes_local_state": (
            wrong_hidden_change > 1.0e-8 and wrong_local_change > 1.0e-8
        ),
        "geometry_reaches_proposal_context": gradient_report["proposal"] > 0.0,
        "geometry_reaches_local_p2": gradient_report["local_p2"] > 0.0,
        "geometry_reaches_slot_row_trunk": gradient_report["slot_row"] > 0.0,
        "geometry_reaches_output_heads": gradient_report["heads"] > 0.0,
        "no_unclassified_trainable_parameters": gradient_report["other"] == 0.0,
        "loss_finite": bool(
            torch.isfinite(
                losses["loss_four_slot_visual_precision"].detach()
            ).all()
        ),
        "only_v13_trainable": all(name.startswith(V13) for name in names),
    }
    report = {
        "experiment": "V13 visual-precision zero-step contract",
        "iteration": iteration,
        "source_iteration": source_iteration,
        "hard_proposal_id_produces_final_geometry": False,
        "activity_and_score_source": "exact_v7",
        "legacy_state": {
            "source_tensor_count": len(source_names),
            "missing_in_v13": list(missing_source),
            "source_sha256": source_sha,
            "v13_source_state_sha256": current_sha,
            "bit_exact": legacy_state_exact,
        },
        "anchor_x_max_abs_error": anchor_x_error,
        "anchor_range_max_abs_error": anchor_range_error,
        "categorical_mismatches": categorical_mismatches,
        "activity_score_key_overlap": activity_score_overlap,
        "wrong_p2_mean_hidden_change": wrong_hidden_change,
        "wrong_p2_mean_local_attention_change": wrong_local_change,
        "delta_offset_min": float(offsets.min()),
        "delta_offset_max": float(offsets.max()),
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
    destination.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"output_json: {destination}")


if __name__ == "__main__":
    main()
