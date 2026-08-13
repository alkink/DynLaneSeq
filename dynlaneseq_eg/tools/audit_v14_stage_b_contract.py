from __future__ import annotations

import argparse
import hashlib
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
    "structured_query_head.set_selection_head.corrected_visual_first_geometry."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V14 Stage-B exact parity and gradient isolation."
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


def main() -> None:
    args = parse_args()
    cfg = _configure(args.config, args)
    source_cfg = _configure(args.source_config, args)
    selection = cfg["model"]["structured_query"]["set_selection"]
    if selection.get("four_slot_corrected_visual_first_geometry_enabled") is not True:
        raise ValueError("V14 Stage-B geometry is disabled")
    nonzero = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and float(value) != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    if nonzero != {"w_four_slot_v14_stage_b": 1.0}:
        raise ValueError(f"V14 Stage B requires one objective, found {nonzero}")
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    loader = build_dataloader(
        cfg, split="train", training=True, start_iteration=args.start_iteration
    )
    images, targets, metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)

    source = build_model(source_cfg).to(device)
    source_iteration = int(load_checkpoint(args.source_checkpoint, source, strict=False))
    source.eval()
    source_matcher = build_matcher(source_cfg)
    with torch.no_grad():
        source_outputs, _ = forward_with_matches(
            source, images, targets, source_matcher, source_cfg, source_iteration
        )
    source_names = tuple(sorted(source.state_dict()))
    source_sha = _digest(source.state_dict(), source_names)

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    freeze_stats = freeze_except_parameter_prefixes(model, (PREFIX.rstrip("."),))
    set_frozen_detector_eval(model, (PREFIX.rstrip("."),))
    selector = model.structured_query_head.set_selection_head
    module = selector.corrected_visual_first_geometry
    if module is None or selector.corrected_visual_first_association is None:
        raise ValueError("V14 Stage-B graph is incomplete")
    current_state = model.state_dict()
    source_missing = [name for name in source_names if name not in current_state]
    frozen_source_exact = not source_missing and _digest(current_state, source_names) == source_sha
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(iteration)
    outputs, matches = forward_with_matches(
        model, images, targets, matcher, cfg, iteration
    )
    losses = criterion(outputs, targets, matches)
    parity_names = (
        "selection_slot_geometry_route_indices",
        "selection_slot_indices",
        "selection_slot_scores",
        "selection_slot_active_logits",
        "selection_slot_active",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    )
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
        "nms_distance_thresh_px": float(post.get("lane_nms_distance_thresh_px", 0.0)),
        "nms_min_overlap_points": int(post.get("lane_nms_min_overlap_points", 5)),
        "top_k": int(post.get("top_k", 4)),
        "row_visibility_thresh": float(post.get("row_visibility_thresh", 0.0)),
        "quality_score_power": float(post.get("quality_score_power", 0.0)),
        "score_mode": str(post.get("score_mode", "four_slot")),
    }
    with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
        left_paths = write_culane_predictions(source_outputs, metas, left_dir, **writer_kwargs)
        right_paths = write_culane_predictions(outputs, metas, right_dir, **writer_kwargs)
        left = {path.relative_to(left_dir): path.read_bytes() for path in left_paths}
        right = {path.relative_to(right_dir): path.read_bytes() for path in right_paths}
        writer_exact = left == right

    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, _ in named]
    parameters = tuple(parameter for _, parameter in named)
    gradients = torch.autograd.grad(
        losses["loss_four_slot_v14_stage_b"], parameters, allow_unused=True
    )
    gradient_norm = math.sqrt(
        sum(
            float(gradient.detach().float().square().sum())
            for gradient in gradients
            if gradient is not None
        )
    )
    checks = {
        "source_iteration_exact": source_iteration == int(args.start_iteration),
        "initial_iteration_exact": iteration == int(args.start_iteration),
        "frozen_stage_a_and_v7_state_bit_exact": frozen_source_exact,
        "x_residual_zero": parity["selection_slot_pred_x_rows"] <= 0.1,
        "range_residual_zero": parity["selection_slot_range_norm"] <= 1.0e-3,
        "activity_score_indices_exact": all(
            parity[name] == 0.0
            for name in (
                "selection_slot_geometry_route_indices",
                "selection_slot_indices",
                "selection_slot_scores",
                "selection_slot_active_logits",
                "selection_slot_active",
            )
        ),
        "writer_files_bit_exact": writer_exact,
        "geometry_gradient_positive": gradient_norm > 0.0,
        "only_stage_b_trainable": bool(names) and all(name.startswith(PREFIX) for name in names),
        "loss_finite": bool(torch.isfinite(losses["loss_four_slot_v14_stage_b"])),
    }
    report = {
        "experiment": "V14 Stage-B zero-step contract",
        "iteration": iteration,
        "source_iteration": source_iteration,
        "parity": parity,
        "writer_files_bit_exact": writer_exact,
        "gradient_norm": gradient_norm,
        "trainable_parameter_names": names,
        "trainable_parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        "freeze_stats": freeze_stats,
        "source_missing_state_names": source_missing,
        "checks": checks,
        "passed": all(checks.values()),
        "test_set_used": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
