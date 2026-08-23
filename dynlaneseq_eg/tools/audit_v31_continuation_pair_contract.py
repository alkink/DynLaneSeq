from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import _torch_load
from dynlaneseq_eg.factory import build_dataloader
from dynlaneseq_eg.tools.audit_v30_exact_pair_contract import (
    _config_differences,
    _stream_digest,
)


ALLOWED_DIFFERENCES = {
    "_config_path",
    "output_dir",
    (
        "model.structured_query.set_selection."
        "four_slot_selection_row_token_gradient_scale"
    ),
}


def _nested_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _tensor_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(str(tuple(item.shape)).encode("ascii"))
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _prepare(path: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(Path(args.dataset_root))
    cfg.setdefault("training", {})["seed"] = int(args.seed)
    cfg["training"]["batch_size"] = int(args.batch_size)
    cfg["training"]["gradient_accumulation_steps"] = int(args.grad_accum)
    cfg.setdefault("dataloader", {})["resume_safe"] = True
    cfg["dataloader"]["num_workers"] = 0
    cfg["dataloader"]["persistent_workers"] = False
    return cfg


def _optimizer_group_names(payload: dict[str, Any]) -> list[str]:
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict):
        return []
    return [str(group.get("name", "")) for group in optimizer.get("param_groups", [])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the exact-paired V30/V31 continuation from global 35K to 50K."
        )
    )
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--treatment-config", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--treatment-checkpoint", required=True)
    parser.add_argument("--rng-reference-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--start-iteration", type=int, default=35000)
    parser.add_argument("--optimizer-steps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_control = load_config(args.control_config)
    raw_treatment = load_config(args.treatment_config)
    differences = _config_differences(raw_control, raw_treatment)
    unexpected = sorted(set(differences) - ALLOWED_DIFFERENCES)

    control_cfg = _prepare(args.control_config, args)
    treatment_cfg = _prepare(args.treatment_config, args)
    control_loader = build_dataloader(
        control_cfg, split="train", training=True,
        start_iteration=int(args.start_iteration),
    )
    treatment_loader = build_dataloader(
        treatment_cfg, split="train", training=True,
        start_iteration=int(args.start_iteration),
    )
    microbatches = int(args.optimizer_steps) * int(args.grad_accum)
    control_stream = _stream_digest(control_loader, microbatches)
    treatment_stream = _stream_digest(treatment_loader, microbatches)

    control_payload = _torch_load(args.control_checkpoint)
    treatment_payload = _torch_load(args.treatment_checkpoint)
    reference_payload = _torch_load(args.rng_reference_checkpoint)
    control_rng = control_payload.get("rng_state")
    treatment_rng = treatment_payload.get("rng_state")
    reference_rng = reference_payload.get("rng_state")
    control_groups = _optimizer_group_names(control_payload)
    treatment_groups = _optimizer_group_names(treatment_payload)

    control_selection = control_cfg["model"]["structured_query"]["set_selection"]
    treatment_selection = treatment_cfg["model"]["structured_query"]["set_selection"]
    checks = {
        "control_iteration_exact": int(control_payload.get("iteration", -1))
        == int(args.start_iteration),
        "treatment_iteration_exact": int(treatment_payload.get("iteration", -1))
        == int(args.start_iteration),
        "rng_reference_iteration_exact": int(reference_payload.get("iteration", -1))
        == int(args.start_iteration),
        "rng_reference_is_control_checkpoint": Path(args.rng_reference_checkpoint).resolve()
        == Path(args.control_checkpoint).resolve(),
        "optimizer_state_present_in_both": isinstance(control_payload.get("optimizer"), dict)
        and isinstance(treatment_payload.get("optimizer"), dict),
        "scheduler_state_present_in_both": isinstance(control_payload.get("scheduler"), dict)
        and isinstance(treatment_payload.get("scheduler"), dict),
        "scheduler_state_exact": _nested_equal(
            control_payload.get("scheduler"), treatment_payload.get("scheduler")
        ),
        "optimizer_group_topology_exact": control_groups == treatment_groups
        and bool(control_groups),
        "model_parameter_keys_exact": control_payload.get("model", {}).keys()
        == treatment_payload.get("model", {}).keys(),
        "common_rng_reference_present": isinstance(reference_rng, dict)
        and bool(reference_rng),
        "dataset_size_exact": len(control_loader.dataset)
        == len(treatment_loader.dataset),
        "dataset_config_exact": control_cfg.get("dataset")
        == treatment_cfg.get("dataset"),
        "augmentation_config_exact": control_cfg.get("augmentation")
        == treatment_cfg.get("augmentation"),
        "sampler_contract_exact": control_loader.batch_sampler.contract()
        == treatment_loader.batch_sampler.contract(),
        "full_15k_stream_digest_exact": control_stream["sha256"]
        == treatment_stream["sha256"],
        "effective_batch_exact_16": int(args.batch_size) * int(args.grad_accum)
        == 16,
        "physical_batch_preserved_4": int(args.batch_size) == 4,
        "gradient_accumulation_preserved_4": int(args.grad_accum) == 4,
        "no_unexpected_config_differences": not unexpected,
        "field_enabled_in_both": bool(
            control_selection.get("four_slot_joint_field_enabled", False)
        ) and bool(treatment_selection.get("four_slot_joint_field_enabled", False)),
        "route_residual_disabled_in_both": float(
            control_selection.get("four_slot_joint_field_route_residual_scale", -1.0)
        ) == float(
            treatment_selection.get("four_slot_joint_field_route_residual_scale", -2.0)
        ) == 0.0,
        "control_bridge_closed": float(
            control_selection.get("four_slot_selection_row_token_gradient_scale", 0.0)
        ) == 0.0,
        "treatment_bridge_open_0p10": float(
            treatment_selection.get("four_slot_selection_row_token_gradient_scale", 0.0)
        ) == 0.10,
    }
    report = {
        "experiment": "V31 exact-paired 35K-to-50K continuation contract",
        "control_checkpoint": str(args.control_checkpoint),
        "treatment_checkpoint": str(args.treatment_checkpoint),
        "rng_reference_checkpoint": str(args.rng_reference_checkpoint),
        "start_iteration": int(args.start_iteration),
        "optimizer_steps": int(args.optimizer_steps),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.grad_accum),
        "effective_batch_size": int(args.batch_size) * int(args.grad_accum),
        "allowed_config_differences": sorted(ALLOWED_DIFFERENCES),
        "observed_config_differences": differences,
        "unexpected_config_differences": unexpected,
        "control_stream": control_stream,
        "treatment_stream": treatment_stream,
        "checkpoint_rng_equal_before_override": _nested_equal(
            control_rng, treatment_rng
        ),
        "control_rng_sha256": _tensor_digest(control_rng),
        "treatment_rng_sha256": _tensor_digest(treatment_rng),
        "reference_rng_sha256": _tensor_digest(reference_rng),
        "common_rng_override_required_for_both_arms": True,
        "checks": checks,
        "passed": all(checks.values()),
        "test_split_used": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("V31 35K-to-50K exact-pair contract failed")


if __name__ == "__main__":
    main()
