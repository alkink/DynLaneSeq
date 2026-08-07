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
)
from dynlaneseq_eg.engine.frozen_training import (
    freeze_except_parameter_prefixes,
    set_frozen_detector_eval,
)
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.train import seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V6-A four-slot shape, provenance, and gradients."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-source-iteration", type=int, default=25000)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    selector_prefix = "structured_query_head.set_selection_head."
    selector = model.structured_query_head.set_selection_head
    selector_parameters = sum(
        parameter.numel() for parameter in selector.parameters()
    )
    load_stats = load_compatible_model_weights(checkpoint_path, model)
    source_state, source_payload = _materialize_model_state(checkpoint_path)
    missing_detector = []
    mismatched_detector = []
    for name, target_tensor in model.state_dict().items():
        if name.startswith(selector_prefix):
            continue
        source_tensor = source_state.get(name)
        if source_tensor is None:
            missing_detector.append(name)
        elif tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mismatched_detector.append(name)

    prefixes = tuple(cfg["training"]["trainable_parameter_prefixes"])
    freeze_stats = freeze_except_parameter_prefixes(model, prefixes)
    set_frozen_detector_eval(
        model,
        tuple(cfg["training"]["trainable_module_prefixes"]),
    )
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    loader = build_dataloader(cfg, split="train", training=True)
    images, targets, _metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(args.amp_dtype)
    autocast_kwargs: dict[str, Any] = {
        "device_type": device.type,
        "enabled": device.type == "cuda" and amp_dtype is not None,
    }
    if amp_dtype is not None:
        autocast_kwargs["dtype"] = amp_dtype
    with torch.autocast(**autocast_kwargs):
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            int(args.expected_source_iteration),
        )
        losses = criterion(outputs, targets, matches)
        total = losses["loss_total"]
    total.backward()

    selector_squared = 0.0
    selector_grad_tensors = 0
    frozen_squared = 0.0
    frozen_grad_tensors = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        squared = float(parameter.grad.detach().float().square().sum())
        if name.startswith(selector_prefix):
            selector_squared += squared
            selector_grad_tensors += 1
        else:
            frozen_squared += squared
            frozen_grad_tensors += 1

    logits = outputs.get("selection_slot_logits")
    valid = outputs.get("selection_slot_candidate_valid")
    checks = {
        "source_iteration": int(source_payload.get("iteration", -1))
        == int(args.expected_source_iteration),
        "all_frozen_detector_tensors_loaded": not missing_detector
        and not mismatched_detector,
        "exact_probe_parameter_count": int(selector_parameters) == 2_980_711,
        "only_selector_trainable": int(
            freeze_stats["trainable_parameter_count"]
        )
        == int(selector_parameters),
        "slot_logit_shape": isinstance(logits, torch.Tensor)
        and tuple(logits.shape[1:]) == (4, 33),
        "candidate_valid_shape": isinstance(valid, torch.Tensor)
        and tuple(valid.shape[1:]) == (32,),
        "finite_loss": bool(torch.isfinite(total.detach()).cpu()),
        "selector_gradient_positive": selector_squared > 0.0,
        "frozen_gradient_zero": frozen_squared == 0.0,
    }
    payload = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_iteration": int(source_payload.get("iteration", -1)),
        "load_stats": load_stats,
        "missing_frozen_detector_tensors": missing_detector,
        "mismatched_frozen_detector_tensors": mismatched_detector,
        "selector_parameter_count": int(selector_parameters),
        "freeze_stats": freeze_stats,
        "slot_shape": list(logits.shape) if isinstance(logits, torch.Tensor) else None,
        "gradient_contract": {
            "selector_norm": selector_squared ** 0.5,
            "selector_tensor_count": int(selector_grad_tensors),
            "frozen_norm": frozen_squared ** 0.5,
            "frozen_tensor_count": int(frozen_grad_tensors),
        },
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in losses.items()
            if name.startswith(("loss_four_slot", "four_slot_"))
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")
    if not payload["passed"]:
        raise SystemExit("V6-A four-slot contract failed")


if __name__ == "__main__":
    main()

