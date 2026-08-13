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
    "structured_query_head.set_selection_head."
    "bottom_aware_relational_geometry"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic V15 bottom-aware relational initialization "
            "on top of the exact V7 deployment checkpoint."
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
            f"expected source iteration {args.iteration}, found {source_iteration}"
        )

    cfg = load_config(args.config)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(args.seed))
    model = build_model(cfg)
    stats = load_compatible_model_weights(source_path, model)
    selector = model.structured_query_head.set_selection_head
    if selector.slot_refinement is None:
        raise ValueError("V15 requires the exact V7 refined deployment anchor")
    if selector.bottom_aware_relational_geometry is None:
        raise ValueError("V15 relational module is unavailable")
    for label, module in (
        ("V11", selector.unified_slot_decoder),
        ("V12", selector.visual_first_association),
        ("V13", selector.visual_precision_geometry),
        ("V14A", selector.corrected_visual_first_association),
        ("V14B", selector.corrected_visual_first_geometry),
    ):
        if module is not None:
            raise ValueError(f"V15 must not instantiate legacy experimental {label}")

    new_state = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if name == MODULE_PREFIX or name.startswith(MODULE_PREFIX + ".")
    }
    if not new_state:
        raise ValueError("V15 initialization found no new parameter tensors")
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
        "experiment": "V15 deterministic bottom-aware relational initialization",
        "source_checkpoint": str(source_path),
        "source_iteration": source_iteration,
        "seed": int(args.seed),
        "compatible_load": stats,
        "v7_refiner_present": True,
        "v15_bottom_aware_relational_geometry_present": True,
        "hard_clustering_present": False,
        "proposal_prototype_present": False,
        "proposal_id_supervision_present": False,
        "deployment_mode": "exact_v7_at_initialization",
        "new_parameter_tensors": len(new_state),
        "new_parameter_count": sum(
            int(parameter.numel()) for parameter in new_state.values()
        ),
        "new_parameter_sha256": _state_digest(new_state),
        "checkpoint_model_prefix": MODULE_PREFIX,
        "output_checkpoint": str(output_checkpoint.resolve()),
        "test_set_used": False,
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
