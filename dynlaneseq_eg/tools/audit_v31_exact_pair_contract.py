from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove that V30 field-only control and the V31 backward-only "
            "selection bridge consume the same 30K-to-35K data stream."
        )
    )
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--treatment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--start-iteration", type=int, default=30000)
    parser.add_argument("--optimizer-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def prepare(path: str, args: argparse.Namespace) -> dict:
    cfg = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg.setdefault("training", {})["seed"] = int(args.seed)
    cfg["training"]["batch_size"] = int(args.batch_size)
    cfg["training"]["gradient_accumulation_steps"] = int(args.grad_accum)
    cfg.setdefault("dataloader", {})["resume_safe"] = True
    cfg["dataloader"]["num_workers"] = 0
    cfg["dataloader"]["persistent_workers"] = False
    return cfg


def main() -> None:
    args = parse_args()
    raw_control = load_config(args.control_config)
    raw_treatment = load_config(args.treatment_config)
    differences = _config_differences(raw_control, raw_treatment)
    unexpected = sorted(set(differences) - ALLOWED_DIFFERENCES)

    control_cfg = prepare(args.control_config, args)
    treatment_cfg = prepare(args.treatment_config, args)
    control_loader = build_dataloader(
        control_cfg,
        split="train",
        training=True,
        start_iteration=int(args.start_iteration),
    )
    treatment_loader = build_dataloader(
        treatment_cfg,
        split="train",
        training=True,
        start_iteration=int(args.start_iteration),
    )
    microbatch_count = int(args.optimizer_steps) * int(args.grad_accum)
    control_stream = _stream_digest(control_loader, microbatch_count)
    treatment_stream = _stream_digest(treatment_loader, microbatch_count)

    payload = _torch_load(args.checkpoint)
    rng_state = payload.get("rng_state")
    rng_keys = sorted(rng_state) if isinstance(rng_state, dict) else []
    control_selection = control_cfg["model"]["structured_query"][
        "set_selection"
    ]
    treatment_selection = treatment_cfg["model"]["structured_query"][
        "set_selection"
    ]
    checks = {
        "source_iteration_matches": int(payload.get("iteration", -1))
        == int(args.start_iteration),
        "source_optimizer_present": isinstance(payload.get("optimizer"), dict),
        "source_scheduler_present": isinstance(payload.get("scheduler"), dict),
        "source_rng_state_present": bool(rng_keys),
        "dataset_size_matches": len(control_loader.dataset)
        == len(treatment_loader.dataset),
        "augmentation_config_exact": control_cfg.get("augmentation")
        == treatment_cfg.get("augmentation"),
        "dataset_config_exact": control_cfg.get("dataset")
        == treatment_cfg.get("dataset"),
        "sampler_contract_exact": control_loader.batch_sampler.contract()
        == treatment_loader.batch_sampler.contract(),
        "full_5k_stream_digest_exact": control_stream["sha256"]
        == treatment_stream["sha256"],
        "no_unexpected_config_differences": not unexpected,
        "field_enabled_in_both": bool(
            control_selection.get("four_slot_joint_field_enabled", False)
        )
        and bool(
            treatment_selection.get("four_slot_joint_field_enabled", False)
        ),
        "field_loss_exact": float(
            control_cfg.get("loss", {}).get("w_four_slot_joint_field", 0.0)
        )
        == float(
            treatment_cfg.get("loss", {}).get("w_four_slot_joint_field", 0.0)
        )
        > 0.0,
        "route_residual_disabled_in_both": float(
            control_selection.get(
                "four_slot_joint_field_route_residual_scale", -1.0
            )
        )
        == float(
            treatment_selection.get(
                "four_slot_joint_field_route_residual_scale", -2.0
            )
        )
        == 0.0,
        "control_bridge_closed": float(
            control_selection.get(
                "four_slot_selection_row_token_gradient_scale", 0.0
            )
        )
        == 0.0,
        "treatment_bridge_open": float(
            treatment_selection.get(
                "four_slot_selection_row_token_gradient_scale", 0.0
            )
        )
        == 0.10,
    }
    report = {
        "experiment": "V31 selection-gradient bridge exact-pair contract",
        "source_checkpoint": str(args.checkpoint),
        "start_iteration": int(args.start_iteration),
        "optimizer_steps": int(args.optimizer_steps),
        "seed": int(args.seed),
        "effective_batch_size": int(args.batch_size) * int(args.grad_accum),
        "source_rng_keys": rng_keys,
        "allowed_config_differences": sorted(ALLOWED_DIFFERENCES),
        "observed_config_differences": differences,
        "unexpected_config_differences": unexpected,
        "control_stream": control_stream,
        "treatment_stream": treatment_stream,
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
        raise SystemExit("V31 exact-pair contract failed")


if __name__ == "__main__":
    main()
