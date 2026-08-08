from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import (
    load_compatible_model_weights,
    save_checkpoint,
)
from dynlaneseq_eg.factory import build_model
from dynlaneseq_eg.tools.train import seed_everything


HEAD_PREFIX = "structured_query_head.set_selection_head."


def _tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one deterministic V7 slot-head initialization over "
            "a frozen V5 proposal detector for paired reference arms."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-iteration", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    source_path = Path(args.source_checkpoint).resolve()
    output_path = Path(args.output_checkpoint)
    cfg = load_config(config_path)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg.setdefault("training", {})["seed"] = int(args.seed)

    seed_everything(int(args.seed))
    model = build_model(cfg)
    head = model.structured_query_head.set_selection_head
    if head is None:
        raise RuntimeError("reference gate requires a four-slot head")
    # Preserve exactly one newly initialized slot head.  Loading the mature
    # detector may contain historical selector keys; they must never leak into
    # this paired initialization.
    fresh_head = {
        name: value.detach().clone()
        for name, value in head.state_dict().items()
    }
    load_stats = load_compatible_model_weights(source_path, model)
    head.load_state_dict(fresh_head, strict=True)
    head_state = {
        HEAD_PREFIX + name: value.detach().clone()
        for name, value in head.state_dict().items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_path,
        model,
        iteration=int(args.source_iteration),
        cfg=cfg,
        model_state_prefixes=(HEAD_PREFIX,),
        base_checkpoint=source_path,
    )
    payload = {
        "experiment": "V7 paired hard/direct reference initialization",
        "config": str(config_path),
        "source_checkpoint": str(source_path),
        "source_checkpoint_size": int(source_path.stat().st_size),
        "source_checkpoint_mtime_ns": int(source_path.stat().st_mtime_ns),
        "source_iteration": int(args.source_iteration),
        "seed": int(args.seed),
        "compatible_detector_load": load_stats,
        "slot_head_parameter_tensors": len(head_state),
        "slot_head_sha256": _tensor_digest(head_state),
        "output_checkpoint": str(output_path.resolve()),
    }
    report_path = Path(args.output_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_checkpoint: {output_path}")
    print(f"output_json: {report_path}")


if __name__ == "__main__":
    main()
