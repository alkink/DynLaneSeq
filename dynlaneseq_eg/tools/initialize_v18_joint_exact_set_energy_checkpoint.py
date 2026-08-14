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
    "structured_query_head.set_selection_head.joint_exact_set_energy"
)
PREFIXES = (MODULE_PREFIX, "encoder.ms_proj.p3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize the corrected V18 exact set-energy gate."
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
        digest.update(str(tensor.dtype).encode("ascii"))
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
    module = selector.joint_exact_set_energy
    if selector.slot_refinement is None or module is None:
        raise ValueError("V18 requires the V7 anchor and exact set module")
    forbidden = {
        "unified_slot_decoder": selector.unified_slot_decoder,
        "visual_first_association": selector.visual_first_association,
        "visual_precision_geometry": selector.visual_precision_geometry,
        "corrected_visual_first_association": (
            selector.corrected_visual_first_association
        ),
        "corrected_visual_first_geometry": (
            selector.corrected_visual_first_geometry
        ),
        "bottom_aware_relational_geometry": (
            selector.bottom_aware_relational_geometry
        ),
        "candidate_aligned_reranker": selector.candidate_aligned_reranker,
        "iterative_slot_geometry": selector.iterative_slot_geometry,
    }
    present = [name for name, value in forbidden.items() if value is not None]
    if present:
        raise ValueError(f"V18 must not instantiate V11-V17 modules: {present}")

    exact_zero = {
        "unary_output": float(module.unary_interaction[-1].weight.abs().max()),
        "pair_output": float(module.pair_output.weight.abs().max()),
        "delta_output": float(module.refiner.delta_head.weight.abs().max()),
        "range_output": float(module.refiner.range_head.weight.abs().max()),
        "policy_output": float(module.refiner.policy_head.weight.abs().max()),
        "activity_output": float(module.refiner.activity_head.weight.abs().max()),
    }
    if any(value != 0.0 for value in exact_zero.values()):
        raise ValueError(f"V18 residual initialization is not exact zero: {exact_zero}")
    new_state = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in PREFIXES)
    }
    output_checkpoint = Path(args.output_checkpoint).expanduser()
    save_checkpoint(
        output_checkpoint,
        model,
        iteration=int(args.iteration),
        cfg=cfg,
        model_state_prefixes=PREFIXES,
        base_checkpoint=source_path,
        include_rng_state=True,
    )
    report = {
        "experiment": "V18 corrected joint exact set-energy initialization",
        "source_checkpoint": str(source_path),
        "source_iteration": source_iteration,
        "seed": int(args.seed),
        "compatible_load": stats,
        "exact_zero_residual_heads": exact_zero,
        "physical_set_count": int(module.combination_table.shape[0]),
        "permutations_per_set": int(module.permutation_table.shape[0]),
        "ordered_assignment_count": int(module.ordered_assignments.shape[0])
        * int(module.ordered_assignments.shape[1]),
        "target_semantics": "unordered_physical_set",
        "deployment_semantics": "exact_ordered_argmax",
        "new_parameter_tensors": len(new_state),
        "new_parameter_count": sum(int(value.numel()) for value in new_state.values()),
        "new_parameter_sha256": _state_digest(new_state),
        "checkpoint_model_prefixes": list(PREFIXES),
        "output_checkpoint": str(output_checkpoint.resolve()),
        "test_set_used": False,
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
