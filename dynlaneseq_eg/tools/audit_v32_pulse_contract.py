from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import _torch_load
from dynlaneseq_eg.factory import build_dataloader
from dynlaneseq_eg.tools.audit_v30_exact_pair_contract import (
    _config_differences,
    _stream_digest,
)
from dynlaneseq_eg.tools.audit_v31_continuation_pair_contract import (
    _nested_equal,
    _optimizer_group_names,
    _tensor_digest,
)


ALLOWED_V7_PULSE_DIFFERENCES = {
    "_config_path",
    "output_dir",
    "loss.w_four_slot_joint_field",
    "model.structured_query.set_selection.four_slot_joint_field_enabled",
    "model.structured_query.set_selection.four_slot_joint_field_forward_enabled",
    "model.structured_query.set_selection.four_slot_joint_field_hidden_dim",
    (
        "model.structured_query.set_selection."
        "four_slot_joint_field_route_residual_scale"
    ),
    (
        "model.structured_query.set_selection."
        "four_slot_selection_row_token_gradient_scale"
    ),
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove the exact V32 35K-to-50K auxiliary-pulse "
            "consolidation contract before either GPU run starts."
        )
    )
    parser.add_argument("--v7-config", required=True)
    parser.add_argument("--pulse-config", required=True)
    parser.add_argument("--v7-source", required=True)
    parser.add_argument("--v7-endpoint", required=True)
    parser.add_argument("--field-source", required=True)
    parser.add_argument("--bridge-source", required=True)
    parser.add_argument("--rng-reference-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--start-iteration", type=int, default=35000)
    parser.add_argument("--end-iteration", type=int, default=50000)
    parser.add_argument("--optimizer-steps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_v7 = load_config(args.v7_config)
    raw_pulse = load_config(args.pulse_config)
    differences = _config_differences(raw_v7, raw_pulse)
    unexpected = sorted(
        set(differences) - ALLOWED_V7_PULSE_DIFFERENCES
    )

    v7_cfg = _prepare(args.v7_config, args)
    pulse_cfg = _prepare(args.pulse_config, args)
    v7_loader = build_dataloader(
        v7_cfg,
        split="train",
        training=True,
        start_iteration=int(args.start_iteration),
    )
    pulse_loader = build_dataloader(
        pulse_cfg,
        split="train",
        training=True,
        start_iteration=int(args.start_iteration),
    )
    microbatches = int(args.optimizer_steps) * int(args.grad_accum)
    v7_stream = _stream_digest(v7_loader, microbatches)
    pulse_stream = _stream_digest(pulse_loader, microbatches)

    v7_source = _torch_load(args.v7_source)
    v7_endpoint = _torch_load(args.v7_endpoint)
    field_source = _torch_load(args.field_source)
    bridge_source = _torch_load(args.bridge_source)
    reference = _torch_load(args.rng_reference_checkpoint)
    field_groups = _optimizer_group_names(field_source)
    bridge_groups = _optimizer_group_names(bridge_source)

    selection = pulse_cfg["model"]["structured_query"]["set_selection"]
    pulse_rng = reference.get("rng_state")
    checks = {
        "v7_source_is_35k": int(v7_source.get("iteration", -1))
        == int(args.start_iteration),
        "field_source_is_35k": int(field_source.get("iteration", -1))
        == int(args.start_iteration),
        "bridge_source_is_35k": int(bridge_source.get("iteration", -1))
        == int(args.start_iteration),
        "v7_control_endpoint_is_50k": int(v7_endpoint.get("iteration", -1))
        == int(args.end_iteration),
        "rng_reference_is_field_source": Path(
            args.rng_reference_checkpoint
        ).resolve()
        == Path(args.field_source).resolve(),
        "common_rng_reference_present": isinstance(pulse_rng, dict)
        and bool(pulse_rng),
        "source_optimizer_state_present": isinstance(
            field_source.get("optimizer"), dict
        )
        and isinstance(bridge_source.get("optimizer"), dict),
        "source_scheduler_state_present": isinstance(
            field_source.get("scheduler"), dict
        )
        and isinstance(bridge_source.get("scheduler"), dict),
        "source_scheduler_state_exact": _nested_equal(
            field_source.get("scheduler"), bridge_source.get("scheduler")
        ),
        "source_optimizer_group_topology_exact": field_groups
        == bridge_groups
        and bool(field_groups),
        "source_model_parameter_keys_exact": field_source.get(
            "model", {}
        ).keys()
        == bridge_source.get("model", {}).keys(),
        "dataset_size_exact": len(v7_loader.dataset)
        == len(pulse_loader.dataset),
        "dataset_config_exact": v7_cfg.get("dataset")
        == pulse_cfg.get("dataset"),
        "augmentation_config_exact": v7_cfg.get("augmentation")
        == pulse_cfg.get("augmentation"),
        "sampler_contract_exact": v7_loader.batch_sampler.contract()
        == pulse_loader.batch_sampler.contract(),
        "full_15k_stream_digest_exact": v7_stream["sha256"]
        == pulse_stream["sha256"],
        "legacy_loss_contract_exact": {
            key: value
            for key, value in v7_cfg.get("loss", {}).items()
            if key != "w_four_slot_joint_field"
        }
        == {
            key: value
            for key, value in pulse_cfg.get("loss", {}).items()
            if key != "w_four_slot_joint_field"
        },
        "optimizer_config_exact": v7_cfg.get("optimizer")
        == pulse_cfg.get("optimizer"),
        "scheduler_config_exact": v7_cfg.get("scheduler")
        == pulse_cfg.get("scheduler"),
        "postprocess_config_exact": v7_cfg.get("postprocess")
        == pulse_cfg.get("postprocess"),
        "no_unexpected_config_differences": not unexpected,
        "field_module_retained_for_resume": bool(
            selection.get("four_slot_joint_field_enabled", False)
        ),
        "field_forward_disabled": not bool(
            selection.get("four_slot_joint_field_forward_enabled", True)
        ),
        "field_loss_disabled": float(
            pulse_cfg.get("loss", {}).get("w_four_slot_joint_field", -1.0)
        )
        == 0.0,
        "field_route_residual_disabled": float(
            selection.get("four_slot_joint_field_route_residual_scale", -1.0)
        )
        == 0.0,
        "selection_bridge_disabled": float(
            selection.get("four_slot_selection_row_token_gradient_scale", -1.0)
        )
        == 0.0,
        "effective_batch_exact_16": int(args.batch_size)
        * int(args.grad_accum)
        == 16,
    }
    report = {
        "experiment": (
            "V32 exact-paired 35K-to-50K auxiliary-pulse consolidation "
            "contract"
        ),
        "sources": {
            "v7_35k": args.v7_source,
            "v7_50k_control": args.v7_endpoint,
            "field_pulse_35k": args.field_source,
            "field_bridge_pulse_35k": args.bridge_source,
            "common_rng_reference": args.rng_reference_checkpoint,
        },
        "start_iteration": int(args.start_iteration),
        "end_iteration": int(args.end_iteration),
        "optimizer_steps": int(args.optimizer_steps),
        "seed": int(args.seed),
        "effective_batch_size": int(args.batch_size)
        * int(args.grad_accum),
        "allowed_v7_pulse_config_differences": sorted(
            ALLOWED_V7_PULSE_DIFFERENCES
        ),
        "observed_v7_pulse_config_differences": differences,
        "unexpected_v7_pulse_config_differences": unexpected,
        "v7_stream": v7_stream,
        "pulse_stream": pulse_stream,
        "field_source_rng_sha256": _tensor_digest(
            field_source.get("rng_state")
        ),
        "bridge_source_rng_sha256": _tensor_digest(
            bridge_source.get("rng_state")
        ),
        "common_rng_override_required_for_both_pulse_arms": True,
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
        raise SystemExit("V32 pulse consolidation contract failed")


if __name__ == "__main__":
    main()
