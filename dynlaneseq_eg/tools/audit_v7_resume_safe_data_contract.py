from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.data.resume_safe import GlobalIterationBatchSampler
from dynlaneseq_eg.engine.checkpoint import _torch_load
from dynlaneseq_eg.factory import build_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the V7 legacy/resume-safe paired-gate contract."
    )
    parser.add_argument("--legacy-config", required=True)
    parser.add_argument("--fixed-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--expected-iteration", type=int, default=125000)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _without_intervention_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(cfg)
    normalized.pop("output_dir", None)
    normalized.pop("_config_path", None)
    dataloader = normalized.setdefault("dataloader", {})
    dataloader.pop("resume_safe", None)
    return normalized


def _optimizer_steps(payload: dict[str, Any]) -> list[int]:
    optimizer = payload.get("optimizer", {})
    state = optimizer.get("state", {}) if isinstance(optimizer, dict) else {}
    steps: list[int] = []
    for parameter_state in state.values():
        if not isinstance(parameter_state, dict) or "step" not in parameter_state:
            continue
        step = parameter_state["step"]
        if isinstance(step, torch.Tensor):
            step = step.detach().cpu().item()
        steps.append(int(step))
    return steps


def audit(args: argparse.Namespace) -> dict[str, Any]:
    legacy = load_config(args.legacy_config)
    fixed = load_config(args.fixed_config)
    legacy.setdefault("dataset", {})["root"] = args.dataset_root
    fixed.setdefault("dataset", {})["root"] = args.dataset_root
    fixed_enabled = bool(fixed.get("dataloader", {}).get("resume_safe", False))
    legacy_enabled = bool(
        legacy.get("dataloader", {}).get("resume_safe", False)
    )
    only_expected_config_difference = (
        _without_intervention_fields(legacy)
        == _without_intervention_fields(fixed)
    )

    payload = _torch_load(args.checkpoint)
    checkpoint_iteration = int(payload.get("iteration", -1))
    optimizer_steps = _optimizer_steps(payload)
    scheduler = payload.get("scheduler", {})
    scheduler_last_epoch = (
        int(scheduler.get("last_epoch", -1))
        if isinstance(scheduler, dict)
        else -1
    )
    rng_state = payload.get("rng_state", {})

    dataset = build_dataset(fixed, split="train", training=True)
    sampler = GlobalIterationBatchSampler(
        dataset,
        batch_size=args.batch_size,
        base_seed=int(fixed.get("training", {}).get("seed", 3407)),
        start_iteration=checkpoint_iteration,
        gradient_accumulation_steps=args.grad_accum,
    )
    checks = {
        "legacy_mode_disabled": not legacy_enabled,
        "fixed_mode_enabled": fixed_enabled,
        "only_data_contract_differs": only_expected_config_difference,
        "checkpoint_iteration": (
            checkpoint_iteration == int(args.expected_iteration)
        ),
        "optimizer_present": "optimizer" in payload,
        "scheduler_present": "scheduler" in payload,
        "rng_state_present": isinstance(rng_state, dict) and bool(rng_state),
        "optimizer_step_matches_iteration": bool(optimizer_steps)
        and min(optimizer_steps) == checkpoint_iteration
        and max(optimizer_steps) == checkpoint_iteration,
        "scheduler_matches_iteration": scheduler_last_epoch
        == checkpoint_iteration,
        "effective_batch_16": args.batch_size * args.grad_accum == 16,
    }
    return {
        "experiment": "V7 resume-safe data-stream preflight",
        "legacy_config": str(args.legacy_config),
        "fixed_config": str(args.fixed_config),
        "source_checkpoint": str(args.checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "optimizer_step_min": min(optimizer_steps) if optimizer_steps else None,
        "optimizer_step_max": max(optimizer_steps) if optimizer_steps else None,
        "scheduler_last_epoch": scheduler_last_epoch,
        "checkpoint_rng_keys": sorted(rng_state) if isinstance(rng_state, dict) else [],
        "dataset_size": len(dataset),
        "resume_safe_sampler": sampler.contract(),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = _parse_args()
    report = audit(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"output_json: {output}")
    if not report["passed"]:
        raise SystemExit("V7 resume-safe data preflight failed")


if __name__ == "__main__":
    main()
