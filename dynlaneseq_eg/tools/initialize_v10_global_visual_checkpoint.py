from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import (
    _torch_load,
    load_compatible_model_weights,
    save_checkpoint,
)
from dynlaneseq_eg.factory import build_model
from dynlaneseq_eg.tools.train import seed_everything


SELECTOR_PREFIX = "structured_query_head.set_selection_head"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic compact V10 initialization from the "
            "V7 225k selector while leaving the global visual geometry fresh."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--iteration", type=int, default=225000)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


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
        raise ValueError("V10 must not instantiate the legacy hard refiner")
    if selector.slot_owned_geometry is not None:
        raise ValueError("V10 must not instantiate V9 proposal-memory geometry")
    if selector.global_visual_geometry is None:
        raise ValueError("V10 global visual geometry module is unavailable")

    output_checkpoint = Path(args.output_checkpoint).expanduser()
    save_checkpoint(
        output_checkpoint,
        model,
        iteration=int(args.iteration),
        cfg=cfg,
        model_state_prefixes=(SELECTOR_PREFIX,),
        base_checkpoint=source_path,
        include_rng_state=True,
    )
    new_names = sorted(
        name
        for name, _parameter in model.named_parameters()
        if name.startswith(SELECTOR_PREFIX + ".global_visual_geometry.")
    )
    report = {
        "experiment": "V10 deterministic global visual slot initialization",
        "source_checkpoint": str(source_path),
        "source_iteration": source_iteration,
        "seed": int(args.seed),
        "compatible_load": stats,
        "new_parameter_tensors": len(new_names),
        "new_parameter_names": new_names,
        "legacy_refiner_present": False,
        "v9_slot_owned_geometry_present": False,
        "global_visual_geometry_present": True,
        "proposal_identity_in_geometry_forward": False,
        "output_checkpoint": str(output_checkpoint.resolve()),
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
