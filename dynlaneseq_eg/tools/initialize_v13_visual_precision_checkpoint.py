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
    "structured_query_head.set_selection_head.visual_precision_geometry"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic V13 visual-precision initialization on top "
            "of the fixed V12 visual locator endpoint."
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
            f"expected V12 iteration {args.iteration}, found "
            f"{source_iteration}"
        )

    cfg = load_config(args.config)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(args.seed))
    model = build_model(cfg)
    stats = load_compatible_model_weights(source, model)
    selector = model.structured_query_head.set_selection_head
    if selector.visual_first_association is None:
        raise ValueError("V13 requires the learned V12 visual locator")
    if selector.visual_precision_geometry is None:
        raise ValueError("V13 precision geometry module is unavailable")
    if selector.slot_refinement is None:
        raise ValueError("V13 requires the exact V7 parity anchor")

    new_state = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if name == MODULE_PREFIX or name.startswith(MODULE_PREFIX + ".")
    }
    if not new_state:
        raise ValueError("V13 initialization found no new parameter tensors")
    output = Path(args.output_checkpoint).expanduser()
    save_checkpoint(
        output,
        model,
        iteration=int(args.iteration),
        cfg=cfg,
        model_state_prefixes=(MODULE_PREFIX,),
        base_checkpoint=source,
        include_rng_state=True,
    )
    report = {
        "experiment": "V13 deterministic visual-precision initialization",
        "source_checkpoint": str(source),
        "source_iteration": source_iteration,
        "seed": int(args.seed),
        "compatible_load": stats,
        "v7_refiner_present": True,
        "v12_visual_locator_present": True,
        "v13_visual_precision_present": True,
        "hard_proposal_id_produces_final_geometry": False,
        "activity_and_score_source": "exact_v7",
        "new_parameter_tensors": len(new_state),
        "new_parameter_count": sum(
            int(parameter.numel()) for parameter in new_state.values()
        ),
        "new_parameter_sha256": _state_digest(new_state),
        "checkpoint_model_prefix": MODULE_PREFIX,
        "output_checkpoint": str(output.resolve()),
        "test_set_used": False,
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"output_json: {destination}")


if __name__ == "__main__":
    main()
