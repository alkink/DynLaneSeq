from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.data.resume_safe import SeededSampleIndex
from dynlaneseq_eg.engine.checkpoint import _torch_load
from dynlaneseq_eg.factory import build_dataloader


ALLOWED_CONFIG_DIFFERENCES = {
    "_config_path",
    "output_dir",
    "training.max_iters",
    "loss.w_four_slot_joint_field",
    (
        "model.structured_query.set_selection."
        "four_slot_joint_field_enabled"
    ),
    (
        "model.structured_query.set_selection."
        "four_slot_joint_field_hidden_dim"
    ),
    (
        "model.structured_query.set_selection."
        "four_slot_joint_field_route_residual_scale"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove that V7 control and V30 field-only treatment consume the "
            "same complete 30K-to-35K sample and augmentation stream."
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


def _config_differences(
    left: Any,
    right: Any,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        output: dict[str, dict[str, Any]] = {}
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                output[path] = {"control": None, "treatment": right[key]}
            elif key not in right:
                output[path] = {"control": left[key], "treatment": None}
            else:
                output.update(_config_differences(left[key], right[key], path))
        return output
    if left != right:
        return {prefix: {"control": left, "treatment": right}}
    return {}


def _prepare_config(path: str, args: argparse.Namespace) -> dict[str, Any]:
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


def _microbatches(loader, count: int) -> Iterator[list[SeededSampleIndex]]:
    sampler = loader.batch_sampler
    emitted = 0
    while emitted < count:
        for batch in sampler:
            yield batch
            emitted += 1
            if emitted >= count:
                return


def _stream_digest(loader, count: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    first: list[list[list[int]]] = []
    last: list[list[list[int]]] = []
    for microbatch_index, batch in enumerate(_microbatches(loader, count)):
        encoded_batch: list[list[int]] = []
        for item in batch:
            if not isinstance(item, SeededSampleIndex):
                raise TypeError("exact-pair sampler did not emit SeededSampleIndex")
            record = [
                int(item.index),
                int(item.augmentation_seed),
                int(item.epoch),
                int(item.position),
            ]
            encoded_batch.append(record)
            digest.update(
                (
                    f"{microbatch_index}:"
                    + ":".join(str(value) for value in record)
                    + "\n"
                ).encode("ascii")
            )
        if len(first) < 3:
            first.append(encoded_batch)
        last.append(encoded_batch)
        if len(last) > 3:
            last.pop(0)
    return {
        "microbatches": int(count),
        "samples": int(count) * int(loader.batch_sampler.batch_size),
        "sha256": digest.hexdigest(),
        "first_three_microbatches": first,
        "last_three_microbatches": last,
    }


def main() -> None:
    args = parse_args()
    raw_control = load_config(args.control_config)
    raw_treatment = load_config(args.treatment_config)
    differences = _config_differences(raw_control, raw_treatment)
    unexpected = sorted(set(differences) - ALLOWED_CONFIG_DIFFERENCES)

    control_cfg = _prepare_config(args.control_config, args)
    treatment_cfg = _prepare_config(args.treatment_config, args)
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
    set_selection = treatment_cfg["model"]["structured_query"][
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
        "field_enabled_only_in_treatment": bool(
            set_selection.get("four_slot_joint_field_enabled", False)
        )
        and not bool(
            control_cfg["model"]["structured_query"]["set_selection"].get(
                "four_slot_joint_field_enabled", False
            )
        ),
        "field_loss_positive_only_in_treatment": float(
            treatment_cfg.get("loss", {}).get("w_four_slot_joint_field", 0.0)
        )
        > 0.0
        and float(
            control_cfg.get("loss", {}).get("w_four_slot_joint_field", 0.0)
        )
        == 0.0,
        "route_residual_exactly_disabled": float(
            set_selection.get(
                "four_slot_joint_field_route_residual_scale", -1.0
            )
        )
        == 0.0,
    }
    report = {
        "experiment": "V30 field-only exact-paired 30K-to-35K contract",
        "source_checkpoint": str(args.checkpoint),
        "start_iteration": int(args.start_iteration),
        "optimizer_steps": int(args.optimizer_steps),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.grad_accum),
        "effective_batch_size": int(args.batch_size) * int(args.grad_accum),
        "source_rng_keys": rng_keys,
        "allowed_config_differences": sorted(ALLOWED_CONFIG_DIFFERENCES),
        "observed_config_differences": differences,
        "unexpected_config_differences": unexpected,
        "control_sampler_contract": control_loader.batch_sampler.contract(),
        "treatment_sampler_contract": treatment_loader.batch_sampler.contract(),
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
        raise SystemExit("V30 exact-pair contract failed")


if __name__ == "__main__":
    main()
