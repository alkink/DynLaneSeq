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


SEMANTIC_MARKERS = (
    ".semantic_attention.",
    ".semantic_router.",
    ".semantic_scale_embedding",
    ".norm_semantic_query.",
    ".norm_semantic_ffn.",
    ".semantic_ffn.",
    "structured_query_head.decision_norm.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gradient_summary(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[tuple[str, torch.Tensor]]] = {
        "selector": [],
        "semantic_score": [],
        "geometry_or_encoder": [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith("structured_query_head.set_selection_head."):
            bucket = "selector"
        elif any(marker in name for marker in SEMANTIC_MARKERS):
            bucket = "semantic_score"
        else:
            bucket = "geometry_or_encoder"
        if parameter.grad is not None:
            buckets[bucket].append((name, parameter.grad.detach().float()))

    result: dict[str, dict[str, Any]] = {}
    for name, rows in buckets.items():
        squared = sum(float(grad.square().sum()) for _param, grad in rows)
        result[name] = {
            "gradient_norm": squared ** 0.5,
            "tensor_count": len(rows),
            "parameter_names": [param for param, _grad in rows],
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V4.2 score gradient isolation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-source-iteration",
        type=int,
        default=50000,
        help=(
            "Logical iteration stored in the source checkpoint. The default "
            "preserves the original 50k score-gate contract."
        ),
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="bfloat16")
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
    seed = int(cfg.get("training", {}).get("seed", 3407))
    seed_everything(seed)
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    load_stats = load_compatible_model_weights(checkpoint_path, model)
    source_state, source_payload = _materialize_model_state(checkpoint_path)
    target_state = model.state_dict()
    missing_detector_tensors = []
    shape_mismatched_detector_tensors = []
    for name, target_tensor in target_state.items():
        if name.startswith("structured_query_head.set_selection_head."):
            continue
        source_tensor = source_state.get(name)
        if source_tensor is None:
            missing_detector_tensors.append(name)
        elif tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatched_detector_tensors.append(
                {
                    "name": name,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
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
    amp_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(args.amp_dtype)
    autocast_kwargs: dict[str, Any] = {
        "device_type": device.type,
        "enabled": amp_enabled,
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
            args.expected_source_iteration,
        )
        loss_dict = criterion(outputs, targets, matches)
        loss = loss_dict["loss_total"]
    loss.backward()
    gradients = _gradient_summary(model)
    semantic_expected = any("semantic_" in prefix for prefix in prefixes)
    checks = {
        "source_checkpoint_iteration_matches_expected": (
            int(source_payload.get("iteration", -1))
            == args.expected_source_iteration
        ),
        "all_non_selector_detector_tensors_loaded": (
            not missing_detector_tensors
            and not shape_mismatched_detector_tensors
        ),
        "finite_loss": bool(torch.isfinite(loss.detach()).cpu()),
        "selector_gradient_positive": gradients["selector"]["gradient_norm"] > 0.0,
        "semantic_gradient_contract": (
            gradients["semantic_score"]["gradient_norm"] > 0.0
            if semantic_expected
            else gradients["semantic_score"]["gradient_norm"] == 0.0
        ),
        "geometry_encoder_gradient_zero": (
            gradients["geometry_or_encoder"]["gradient_norm"] == 0.0
        ),
    }
    payload = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "expected_source_iteration": args.expected_source_iteration,
        "load_stats": load_stats,
        "checkpoint_contract": {
            "internal_iteration": int(source_payload.get("iteration", -1)),
            "missing_non_selector_tensors": missing_detector_tensors,
            "shape_mismatched_non_selector_tensors": (
                shape_mismatched_detector_tensors
            ),
        },
        "freeze_stats": freeze_stats,
        "losses": {
            key: float(value.detach().float().cpu())
            for key, value in loss_dict.items()
            if key.startswith(("loss_set_selection", "loss_pointer"))
        },
        "gradients": gradients,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")
    if not payload["passed"]:
        raise SystemExit("V4.2 gradient contract failed")


if __name__ == "__main__":
    main()
