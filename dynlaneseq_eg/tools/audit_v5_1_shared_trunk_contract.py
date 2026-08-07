from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_matcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the exact V5.1 shared-trunk causal fork contract."
    )
    parser.add_argument("--trunk-config", required=True)
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--assignment-config", required=True)
    parser.add_argument("--trunk-checkpoint", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("shared trunk checkpoint payload must be a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    trunk = load_config(args.trunk_config)
    control = load_config(args.control_config)
    assignment = load_config(args.assignment_config)

    trunk_matcher = build_matcher(trunk)
    control_matcher = build_matcher(control)
    assignment_matcher = build_matcher(assignment)
    checks = {
        "same_model": trunk["model"] == control["model"] == assignment["model"],
        "same_loss": trunk["loss"] == control["loss"] == assignment["loss"],
        "same_optimizer": (
            trunk["optimizer"] == control["optimizer"] == assignment["optimizer"]
        ),
        "same_scheduler": (
            trunk["scheduler"] == control["scheduler"] == assignment["scheduler"]
        ),
        "trunk_geometry_only_assignment": (
            trunk_matcher.effective_lambda_obj(0) == 0.0
            and trunk_matcher.effective_lambda_obj(10000) == 0.0
        ),
        "control_remains_geometry_only": (
            control_matcher.effective_lambda_obj(10000) == 0.0
            and control_matcher.effective_lambda_obj(25000) == 0.0
        ),
        "assignment_starts_at_zero": (
            assignment_matcher.effective_lambda_obj(10000) == 0.0
        ),
        "assignment_ramp_midpoint": (
            assignment_matcher.effective_lambda_obj(17500) == 0.125
        ),
        "assignment_finishes_at_quarter": (
            assignment_matcher.effective_lambda_obj(25000) == 0.25
        ),
        "trunk_is_10k": int(trunk["training"]["max_iters"]) == 10000,
        "trunk_keeps_optimizer": bool(
            trunk["training"]["checkpoint_include_optimizer"]
        ),
        "forks_are_15k": (
            int(control["training"]["max_iters"]) == 15000
            and int(assignment["training"]["max_iters"]) == 15000
        ),
        "forks_share_training": control["training"] == assignment["training"],
    }
    checkpoint_audit: dict[str, Any] | None = None
    if args.trunk_checkpoint:
        checkpoint = Path(args.trunk_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing shared trunk checkpoint: {checkpoint}")
        payload = _load_checkpoint(checkpoint)
        rng = payload.get("rng_state", {})
        checkpoint_checks = {
            "iteration_is_10k": int(payload.get("iteration", -1)) == 10000,
            "full_model_state": payload.get("model_state_mode", "full") == "full",
            "has_optimizer": isinstance(payload.get("optimizer"), dict),
            "has_scheduler": isinstance(payload.get("scheduler"), dict),
            "has_python_rng": isinstance(rng, dict) and "python" in rng,
            "has_numpy_rng": isinstance(rng, dict) and "numpy" in rng,
            "has_torch_cpu_rng": isinstance(rng, dict) and "torch_cpu" in rng,
            "has_torch_cuda_rng": isinstance(rng, dict) and "torch_cuda" in rng,
        }
        checkpoint_audit = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "checks": checkpoint_checks,
            "passed": all(checkpoint_checks.values()),
        }

    passed = all(checks.values()) and (
        checkpoint_audit is None or bool(checkpoint_audit["passed"])
    )
    result = {
        "experiment": "V5.1 exact shared-trunk ownership fork",
        "checks": checks,
        "checkpoint": checkpoint_audit,
        "passed": passed,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {output}")
    if not passed:
        raise SystemExit("V5.1 shared-trunk contract failed")


if __name__ == "__main__":
    main()
