from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import (
    _torch_load,
    load_compatible_model_weights,
    save_checkpoint,
)
from dynlaneseq_eg.factory import build_model
from dynlaneseq_eg.tools.train import seed_everything


MODULE_PREFIX = (
    "structured_query_head.set_selection_head.corrected_visual_first_geometry"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create zero-residual V14 Stage-B geometry initialization on top "
            "of the fixed Stage-A endpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--iteration", type=int, default=227000)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = Path(args.source_checkpoint).expanduser().resolve()
    payload = _torch_load(source)
    source_iteration = int(payload.get("iteration", -1))
    if source_iteration != int(args.iteration):
        raise ValueError(
            f"expected Stage-A iteration {args.iteration}, found "
            f"{source_iteration}"
        )
    cfg = load_config(args.config)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(args.seed))
    model = build_model(cfg)
    stats = load_compatible_model_weights(source, model)
    selector = model.structured_query_head.set_selection_head
    if selector.corrected_visual_first_association is None:
        raise ValueError("V14 Stage B requires the learned Stage-A consumer")
    if selector.corrected_visual_first_geometry is None:
        raise ValueError("V14 Stage-B geometry consumer is unavailable")
    if selector.slot_refinement is None:
        raise ValueError("V14 Stage B requires exact V7 anchors")
    new_state = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if name == MODULE_PREFIX or name.startswith(MODULE_PREFIX + ".")
    }
    if not new_state:
        raise ValueError("V14 Stage-B initialization found no new parameters")
    destination = Path(args.output_checkpoint).expanduser()
    save_checkpoint(
        destination,
        model,
        iteration=int(args.iteration),
        cfg=cfg,
        model_state_prefixes=(MODULE_PREFIX,),
        base_checkpoint=source,
        include_rng_state=True,
    )
    report = {
        "experiment": "V14 deterministic parity-anchored Stage-B initialization",
        "source_checkpoint": str(source),
        "source_iteration": source_iteration,
        "seed": int(args.seed),
        "compatible_load": stats,
        "stage_a_consumer_present": True,
        "stage_b_geometry_present": True,
        "activity_score_route_source": "exact_v7",
        "geometry_anchor": "exact_v7",
        "new_parameter_tensors": len(new_state),
        "new_parameter_count": sum(
            int(parameter.numel()) for parameter in new_state.values()
        ),
        "new_parameter_sha256": _state_digest(new_state),
        "checkpoint_model_prefix": MODULE_PREFIX,
        "output_checkpoint": str(destination.resolve()),
        "test_set_used": False,
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
