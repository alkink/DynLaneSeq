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
    "structured_query_head.set_selection_head.slot_owned_safe_replacement"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize zero-edit V20 from the fixed V19 endpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--iteration", type=int, default=233000)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _digest(state: dict[str, torch.Tensor]) -> str:
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
            f"expected V19 iteration {args.iteration}, found {source_iteration}"
        )
    cfg = load_config(args.config)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(args.seed))
    model = build_model(cfg)
    stats = load_compatible_model_weights(source, model)
    selector = model.structured_query_head.set_selection_head
    module = selector.slot_owned_safe_replacement
    if selector.counterfactual_fidelity is None or module is None:
        raise ValueError("V20 requires frozen V19 and the replacement head")
    outputs = (
        module.policy_output,
        module.delta50_output,
        module.delta75_output,
        module.duplicate_output,
        module.abandon_output,
        module.delta_iou_output,
    )
    zero = {
        f"output_{index}_weight": float(output.weight.abs().max())
        for index, output in enumerate(outputs)
    }
    zero.update(
        {
            f"output_{index}_bias": float(output.bias.abs().max())
            for index, output in enumerate(outputs)
        }
    )
    if any(value != 0.0 for value in zero.values()):
        raise ValueError(f"V20 output heads are not exactly neutral: {zero}")
    state = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if name == MODULE_PREFIX or name.startswith(MODULE_PREFIX + ".")
    }
    if not state:
        raise ValueError("V20 module prefix matched no parameters")
    output_checkpoint = Path(args.output_checkpoint).expanduser()
    save_checkpoint(
        output_checkpoint,
        model,
        iteration=int(args.iteration),
        cfg=cfg,
        model_state_prefixes=(MODULE_PREFIX,),
        base_checkpoint=source,
        include_rng_state=True,
    )
    report = {
        "experiment": "V20 slot-owned safe replacement initialization",
        "source_checkpoint": str(source),
        "source_iteration": source_iteration,
        "seed": int(args.seed),
        "compatible_load": stats,
        "zero_output_heads": zero,
        "context_mode": module.context_mode,
        "max_active_edits": 1,
        "new_parameter_tensors": len(state),
        "new_parameter_count": sum(int(value.numel()) for value in state.values()),
        "new_parameter_sha256": _digest(state),
        "output_checkpoint": str(output_checkpoint.resolve()),
        "test_set_used": False,
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

