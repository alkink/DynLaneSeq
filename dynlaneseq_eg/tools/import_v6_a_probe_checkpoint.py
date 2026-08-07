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
    save_checkpoint,
)
from dynlaneseq_eg.factory import build_model


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import the successful cached V5 four-slot probe into the exact "
            "V6-A production head and save a compact inference checkpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--reference-report", default="")
    parser.add_argument("--expected-source-iteration", type=int, default=25000)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    source_path = Path(args.source_checkpoint).resolve()
    probe_path = Path(args.probe_checkpoint).resolve()
    output_checkpoint = Path(args.output_checkpoint)
    cfg = load_config(config_path)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    source_sha = _sha256(source_path)

    _source_state, source_payload = _materialize_model_state(source_path)
    source_iteration = int(source_payload.get("iteration", -1))
    if source_iteration != int(args.expected_source_iteration):
        raise ValueError(
            "unexpected V5.1 source iteration: "
            f"{source_iteration} != {int(args.expected_source_iteration)}"
        )

    model = build_model(cfg)
    load_stats = load_compatible_model_weights(source_path, model)
    selector = model.structured_query_head.set_selection_head
    if selector is None:
        raise ValueError("V6-A config did not construct a selection head")
    expected_state = selector.state_dict()

    payload = _torch_load(probe_path)
    if not isinstance(payload, dict):
        raise TypeError("probe checkpoint must contain a mapping")
    probe_state = payload.get("four_slot_router")
    if not isinstance(probe_state, dict):
        raise ValueError("probe checkpoint has no four_slot_router state")
    missing = sorted(set(expected_state) - set(probe_state))
    unexpected = sorted(set(probe_state) - set(expected_state))
    mismatched = sorted(
        name
        for name in set(expected_state) & set(probe_state)
        if tuple(expected_state[name].shape) != tuple(probe_state[name].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "probe/production slot state mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}"
        )
    selector.load_state_dict(probe_state, strict=True)

    reference: dict[str, Any] | None = None
    reference_checkpoint_sha = None
    if args.reference_report:
        reference_path = Path(args.reference_report).resolve()
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_checkpoint_sha = reference.get("checkpoint_sha256")
        if reference_checkpoint_sha != source_sha:
            raise ValueError(
                "reference probe and source checkpoint differ: "
                f"{reference_checkpoint_sha} != {source_sha}"
            )

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_checkpoint,
        model,
        iteration=int(args.expected_source_iteration),
        cfg=cfg,
        model_state_prefixes=("structured_query_head.set_selection_head",),
        base_checkpoint=str(args.source_checkpoint),
    )
    saved = _torch_load(output_checkpoint)
    saved_state = saved.get("model") if isinstance(saved, dict) else None
    prefixed_expected = {
        f"structured_query_head.set_selection_head.{name}": value
        for name, value in probe_state.items()
    }
    exact_tensor_copy = isinstance(saved_state, dict) and set(saved_state) == set(
        prefixed_expected
    ) and all(
        torch.equal(saved_state[name], prefixed_expected[name])
        for name in prefixed_expected
    )
    checks = {
        "source_weights_loaded": int(load_stats.get("loaded", 0)) > 0,
        "source_iteration": source_iteration
        == int(args.expected_source_iteration),
        "exact_probe_state_contract": not missing
        and not unexpected
        and not mismatched,
        "exact_tensor_copy": bool(exact_tensor_copy),
        "parameter_count": sum(parameter.numel() for parameter in selector.parameters())
        == 2_980_711,
        "reference_source_sha": (
            True if reference is None else reference_checkpoint_sha == source_sha
        ),
    }
    report = {
        "experiment": "V6-A successful-probe -> production-head import",
        "diagnostic_only": True,
        "config": str(config_path),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "source_iteration": source_iteration,
        "probe_checkpoint": str(probe_path),
        "probe_checkpoint_sha256": _sha256(probe_path),
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": _sha256(output_checkpoint),
        "load_stats": load_stats,
        "probe_args": payload.get("args"),
        "reference_report": str(Path(args.reference_report))
        if args.reference_report
        else None,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"output_checkpoint: {output_checkpoint}")
    print(f"output_json: {output_json}")
    if not report["passed"]:
        raise SystemExit("V6-A probe import contract failed")


if __name__ == "__main__":
    main()
