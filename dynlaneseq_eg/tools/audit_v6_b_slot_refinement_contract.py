from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import (
    _materialize_model_state,
    load_compatible_model_weights,
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gradient_norms(
    model: torch.nn.Module,
    *,
    refinement_prefix: str,
    selector_prefix: str,
) -> dict[str, dict[str, float | int]]:
    accumulators = {
        "slot_refinement": [0.0, 0],
        "frozen_router": [0.0, 0],
        "frozen_proposal_detector": [0.0, 0],
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith(refinement_prefix):
            group = "slot_refinement"
        elif name.startswith(selector_prefix):
            group = "frozen_router"
        else:
            group = "frozen_proposal_detector"
        accumulators[group][0] += float(
            parameter.grad.detach().float().square().sum()
        )
        accumulators[group][1] += 1
    return {
        name: {
            "norm": float(values[0]) ** 0.5,
            "tensor_count": int(values[1]),
        }
        for name, values in accumulators.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the V6-B frozen-router bounded slot refiner."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-source-iteration", type=int, default=25000)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    structured = model.structured_query_head
    selector = structured.set_selection_head
    refiner = selector.slot_refinement
    if refiner is None:
        raise ValueError("V6-B config did not construct a slot refiner")
    load_stats = load_compatible_model_weights(checkpoint_path, model)
    source_state, source_payload = _materialize_model_state(checkpoint_path)
    refinement_prefix = (
        "structured_query_head.set_selection_head.slot_refinement."
    )
    selector_prefix = "structured_query_head.set_selection_head."
    missing_source = []
    mismatched_source = []
    for name, value in model.state_dict().items():
        if name.startswith(refinement_prefix):
            continue
        source_value = source_state.get(name)
        if source_value is None:
            missing_source.append(name)
        elif tuple(source_value.shape) != tuple(value.shape):
            mismatched_source.append(name)

    prefixes = tuple(cfg["training"]["trainable_parameter_prefixes"])
    freeze_stats = freeze_except_parameter_prefixes(model, prefixes)
    set_frozen_detector_eval(
        model,
        tuple(cfg["training"]["trainable_module_prefixes"]),
    )
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    loader = build_dataloader(cfg, split="train", training=True)
    images, targets, _metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(args.amp_dtype)
    autocast_kwargs: dict[str, Any] = {
        "device_type": device.type,
        "enabled": device.type == "cuda" and amp_dtype is not None,
    }
    if amp_dtype is not None:
        autocast_kwargs["dtype"] = amp_dtype
    with torch.autocast(**autocast_kwargs):
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            int(args.expected_source_iteration),
        )
        losses = criterion(outputs, targets, matches)
        total = losses["loss_total"]
    total.backward()

    routes = outputs["selection_slot_indices"]
    active = outputs["selection_slot_active"].bool()
    reference = outputs["selection_slot_input_reference_x_rows"].float()
    refined = outputs["selection_slot_pred_x_rows"].float()
    identity_error = (refined - reference).abs().masked_fill(
        ~active.unsqueeze(-1),
        0.0,
    ).amax()
    unique_routes = True
    for row in routes:
        selected = row[row >= 0]
        unique_routes &= int(selected.numel()) == int(selected.unique().numel())
    gradient = _gradient_norms(
        model,
        refinement_prefix=refinement_prefix,
        selector_prefix=selector_prefix,
    )
    refiner_parameter_count = sum(
        parameter.numel() for parameter in refiner.parameters()
    )
    max_delta = float(
        outputs["selection_slot_delta_max_abs"].detach().float().amax().cpu()
    )
    offset_bound = float(refiner.delta_offsets_px.detach().abs().amax().cpu())
    checks = {
        "source_iteration": int(source_payload.get("iteration", -1))
        == int(args.expected_source_iteration),
        "all_non_refiner_source_tensors_loaded": not missing_source
        and not mismatched_source,
        "only_refiner_trainable": int(
            freeze_stats["trainable_parameter_count"]
        )
        == int(refiner_parameter_count),
        "source_forward_is_identity": float(identity_error.detach().cpu())
        <= 1.0e-4,
        "routes_are_globally_unique": bool(unique_routes),
        "finite_loss": bool(torch.isfinite(total.detach()).cpu()),
        "refiner_gradient_positive": float(
            gradient["slot_refinement"]["norm"]
        )
        > 0.0,
        "router_gradient_zero": float(gradient["frozen_router"]["norm"])
        == 0.0,
        "proposal_detector_gradient_zero": float(
            gradient["frozen_proposal_detector"]["norm"]
        )
        == 0.0,
        "bounded_delta": max_delta <= offset_bound + 1.0e-5,
        "matched_slot_geometry": float(
            losses["four_slot_geometry_mean_matched"].detach().cpu()
        )
        > 0.0,
    }
    payload = {
        "experiment": "V6-B frozen router + slot-owned bounded refinement",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_iteration": int(source_payload.get("iteration", -1)),
        "load_stats": load_stats,
        "missing_non_refiner_source_tensors": missing_source,
        "mismatched_non_refiner_source_tensors": mismatched_source,
        "refiner_parameter_count": int(refiner_parameter_count),
        "freeze_stats": freeze_stats,
        "identity_max_abs_px": float(identity_error.detach().cpu()),
        "max_abs_delta_px": max_delta,
        "offset_bound_px": offset_bound,
        "gradient_contract": gradient,
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in losses.items()
            if name.startswith(("loss_four_slot_geometry", "four_slot_geometry"))
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")
    if not payload["passed"]:
        raise SystemExit("V6-B slot refinement contract failed")


if __name__ == "__main__":
    main()
