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
    "structured_query_head.set_selection_head.unified_slot_decoder"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the deterministic V11 unified slot-row initialization "
            "on top of a V7 checkpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--iteration", type=int, default=225000)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_checkpoint).expanduser().resolve()
    source_payload = _torch_load(source_path)
    source_iteration = int(source_payload.get("iteration", -1))
    if source_iteration != int(args.iteration):
        raise ValueError(
            f"expected source iteration {args.iteration}, found "
            f"{source_iteration}"
        )

    cfg = load_config(args.config)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(args.seed))
    model = build_model(cfg)
    stats = load_compatible_model_weights(source_path, model)
    selector = model.structured_query_head.set_selection_head
    if selector.slot_refinement is not None:
        raise ValueError("V11 must not reuse the V7 hard-route refiner")
    if selector.slot_owned_geometry is not None:
        raise ValueError("V11 must not instantiate the V9 soft-route refiner")
    if selector.global_visual_geometry is not None:
        raise ValueError("V11 must not instantiate the V10 P2-only decoder")
    if selector.unified_slot_decoder is None:
        raise ValueError("V11 unified slot-row decoder is unavailable")

    new_state = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if name == MODULE_PREFIX or name.startswith(MODULE_PREFIX + ".")
    }
    if not new_state:
        raise ValueError("V11 initialization found no new parameter tensors")
    output_checkpoint = Path(args.output_checkpoint).expanduser()
    save_checkpoint(
        output_checkpoint,
        model,
        iteration=int(args.iteration),
        cfg=cfg,
        model_state_prefixes=(MODULE_PREFIX,),
        base_checkpoint=source_path,
        include_rng_state=True,
    )
    report = {
        "experiment": "V11 deterministic unified slot-row initialization",
        "source_checkpoint": str(source_path),
        "source_iteration": source_iteration,
        "seed": int(args.seed),
        "compatible_load": stats,
        "v7_refiner_present": False,
        "v9_slot_owned_geometry_present": False,
        "v10_global_visual_geometry_present": False,
        "v11_unified_slot_decoder_present": True,
        "new_parameter_tensors": len(new_state),
        "new_parameter_count": sum(
            int(parameter.numel()) for parameter in new_state.values()
        ),
        "new_parameter_sha256": _state_digest(new_state),
        "checkpoint_model_prefix": MODULE_PREFIX,
        "output_checkpoint": str(output_checkpoint.resolve()),
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
