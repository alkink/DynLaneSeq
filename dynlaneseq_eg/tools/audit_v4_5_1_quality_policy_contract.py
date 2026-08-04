from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
NEW_SPLIT_PREFIXES = (
    "structured_query_head.set_selection_head.pointer_quality_adapter.",
    "structured_query_head.set_selection_head.pointer_policy_output_norm.",
    "structured_query_head.set_selection_head.pointer_policy_output.",
    "structured_query_head.set_selection_head.pointer_quality_scale_raw",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_group(name: str) -> str:
    selector = "structured_query_head.set_selection_head."
    if name.startswith(selector + "pointer_quality_adapter."):
        return "quality_head"
    if name.startswith(selector + "output_norm."):
        return "quality_head"
    if name.startswith(selector + "output."):
        return "quality_head"
    if name.startswith(selector + "pointer_policy_output_norm."):
        return "policy_head"
    if name.startswith(selector + "pointer_policy_output."):
        return "policy_head"
    if name == selector + "pointer_quality_scale_raw":
        return "policy_head"
    if name.startswith(selector + "pointer_"):
        return "pointer_core"
    if name.startswith(selector):
        return "candidate_encoder"
    if any(marker in name for marker in SEMANTIC_MARKERS):
        return "semantic_score"
    return "geometry_or_encoder"


def _gradient_vector_stats(
    names: list[str],
    gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, dict[str, float | int]]:
    accumulators: dict[str, dict[str, float | int]] = {}
    for name, gradient in zip(names, gradients, strict=True):
        group = _parameter_group(name)
        row = accumulators.setdefault(
            group,
            {"squared_norm": 0.0, "tensor_count": 0, "nonzero_tensor_count": 0},
        )
        row["tensor_count"] = int(row["tensor_count"]) + 1
        if gradient is not None:
            value = gradient.detach().float()
            squared = float(value.square().sum().cpu())
            row["squared_norm"] = float(row["squared_norm"]) + squared
            if squared > 0.0:
                row["nonzero_tensor_count"] = int(row["nonzero_tensor_count"]) + 1
    result: dict[str, dict[str, float | int]] = {}
    for group in (
        "quality_head",
        "policy_head",
        "pointer_core",
        "candidate_encoder",
        "semantic_score",
        "geometry_or_encoder",
    ):
        row = accumulators.get(
            group,
            {"squared_norm": 0.0, "tensor_count": 0, "nonzero_tensor_count": 0},
        )
        result[group] = {
            "gradient_norm": float(row["squared_norm"]) ** 0.5,
            "tensor_count": int(row["tensor_count"]),
            "nonzero_tensor_count": int(row["nonzero_tensor_count"]),
        }
    return result


def _cosine(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
) -> float | None:
    dot = 0.0
    first_square = 0.0
    second_square = 0.0
    for left, right in zip(first, second, strict=True):
        if left is not None:
            first_square += float(left.detach().float().square().sum().cpu())
        if right is not None:
            second_square += float(right.detach().float().square().sum().cpu())
        if left is not None and right is not None:
            dot += float((left.detach().float() * right.detach().float()).sum().cpu())
    if first_square <= 0.0 or second_square <= 0.0:
        return None
    return dot / (first_square**0.5 * second_square**0.5)


def _rms(value: object) -> float | None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return float(value.detach().float().square().mean().sqrt().cpu())


def _autocast_kwargs(device: torch.device, amp_dtype: str) -> dict[str, Any]:
    enabled = device.type == "cuda" and amp_dtype != "none"
    kwargs: dict[str, Any] = {"device_type": device.type, "enabled": enabled}
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        amp_dtype
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the V4.5.1 quality/policy gradient contract."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-iteration", type=int, default=60000)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
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
    baseline_config_path = Path(args.baseline_config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    seed = int(cfg.get("training", {}).get("seed", 3407))
    seed_everything(seed)
    device = torch.device(args.device)
    autocast_kwargs = _autocast_kwargs(device, args.amp_dtype)

    loader = build_dataloader(cfg, split="train", training=True)
    images, targets, _metas = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = nested_to_device(targets, device)

    model = build_model(cfg).to(device)
    load_stats = load_compatible_model_weights(checkpoint_path, model)
    source_state, source_payload = _materialize_model_state(checkpoint_path)
    missing_common = []
    shape_mismatched_common = []
    for name, target_tensor in model.state_dict().items():
        if name.startswith(NEW_SPLIT_PREFIXES):
            continue
        source_tensor = source_state.get(name)
        if source_tensor is None:
            missing_common.append(name)
        elif source_tensor.shape != target_tensor.shape:
            shape_mismatched_common.append(name)

    # The new branches are initialized to reproduce V4.5 exactly: residual
    # quality adapter = identity, policy bias = zero, bounded quality scale = 1.
    baseline_cfg = load_config(baseline_config_path)
    baseline = build_model(baseline_cfg).to(device).eval()
    load_compatible_model_weights(checkpoint_path, baseline)
    model.eval()
    with torch.no_grad(), torch.autocast(**autocast_kwargs):
        baseline_outputs = baseline(images)
        initial_outputs = model(images)
    baseline_pointer = baseline_outputs["selection_pointer_logits"].float()
    initial_pointer = initial_outputs["selection_pointer_logits"].float()
    initial_pointer_max_abs = float(
        (baseline_pointer - initial_pointer).abs().max().cpu()
    )
    initial_indices_equal = bool(
        torch.equal(
            baseline_outputs["selection_pointer_indices"],
            initial_outputs["selection_pointer_indices"],
        )
    )
    del baseline, baseline_outputs, initial_outputs

    prefixes = tuple(cfg["training"]["trainable_parameter_prefixes"])
    freeze_stats = freeze_except_parameter_prefixes(model, prefixes)
    set_frozen_detector_eval(
        model,
        tuple(cfg["training"]["trainable_module_prefixes"]),
    )
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    with torch.autocast(**autocast_kwargs):
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            int(args.source_iteration),
        )
        loss_dict = criterion(outputs, targets, matches)

    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _parameter in named_trainable]
    parameters = [parameter for _name, parameter in named_trainable]
    loss_cfg = cfg["loss"]
    objectives = {
        "sequence": loss_dict["loss_pointer_sequence"],
        "quality_weighted": float(loss_cfg["pointer_quality_weight"])
        * loss_dict["loss_pointer_quality"],
        "listwise_weighted": float(
            loss_cfg.get("pointer_cluster_listwise_weight", 0.0)
        )
        * loss_dict["loss_pointer_listwise"],
    }
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    gradient_groups: dict[str, dict[str, dict[str, float | int]]] = {}
    for objective_name, objective in objectives.items():
        gradient = torch.autograd.grad(
            objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradients[objective_name] = gradient
        gradient_groups[objective_name] = _gradient_vector_stats(names, gradient)

    model.zero_grad(set_to_none=True)
    loss_dict["loss_total"].backward()
    total_gradients = tuple(parameter.grad for parameter in parameters)
    gradient_groups["total"] = _gradient_vector_stats(names, total_gradients)

    pointer_logits = outputs["selection_pointer_logits"][..., :-1].float()
    reconstructed = (
        outputs["selection_pointer_unary_component"].float()
        + outputs["selection_pointer_policy_component"].float()
        + outputs["selection_pointer_content_component"].float()
        + outputs["selection_pointer_relation_bias"].float()
    )
    available = pointer_logits > -1000.0
    component_max_abs = float(
        (pointer_logits[available] - reconstructed[available]).abs().max().cpu()
    )
    scale = outputs["selection_pointer_quality_scale"].detach().float()
    listwise_enabled = float(
        loss_cfg.get("pointer_cluster_listwise_weight", 0.0)
    ) > 0.0
    quality_policy_mode = str(
        cfg["model"]["structured_query"]["set_selection"].get(
            "pointer_quality_policy_mode",
            "shared",
        )
    ).strip().lower()
    sequence_groups = gradient_groups["sequence"]
    quality_groups = gradient_groups["quality_weighted"]
    listwise_groups = gradient_groups["listwise_weighted"]
    checks = {
        "source_iteration_matches": int(source_payload.get("iteration", -1))
        == int(args.source_iteration),
        "all_preexisting_tensors_loaded": not missing_common
        and not shape_mismatched_common,
        "initial_pointer_logits_preserved": initial_pointer_max_abs <= 1e-5,
        "initial_greedy_sequence_preserved": initial_indices_equal,
        "finite_total_loss": bool(
            torch.isfinite(loss_dict["loss_total"].detach()).cpu()
        ),
        "component_sum_exact": component_max_abs <= 1e-5,
        "quality_scale_strictly_bounded": bool(
            ((scale > 0.0) & (scale < 2.0)).all().cpu()
        ),
        "quality_policy_routing_contract": (
            sequence_groups["policy_head"]["gradient_norm"] > 0.0
            and sequence_groups["quality_head"]["gradient_norm"] == 0.0
            and quality_groups["quality_head"]["gradient_norm"] > 0.0
            and quality_groups["policy_head"]["gradient_norm"] == 0.0
            and quality_groups["pointer_core"]["gradient_norm"] == 0.0
            and quality_groups["candidate_encoder"]["gradient_norm"] == 0.0
            and quality_groups["semantic_score"]["gradient_norm"] == 0.0
            if quality_policy_mode == "decoupled"
            else (
                sequence_groups["quality_head"]["gradient_norm"] > 0.0
                and quality_groups["quality_head"]["gradient_norm"] > 0.0
            )
        ),
        "listwise_gradient_contract": (
            listwise_groups["quality_head"]["gradient_norm"] > 0.0
            and listwise_groups["policy_head"]["gradient_norm"] == 0.0
            and listwise_groups["pointer_core"]["gradient_norm"] == 0.0
            and listwise_groups["candidate_encoder"]["gradient_norm"] == 0.0
            and listwise_groups["semantic_score"]["gradient_norm"] == 0.0
            if listwise_enabled
            else all(
                row["gradient_norm"] == 0.0 for row in listwise_groups.values()
            )
        ),
        "geometry_encoder_gradient_zero": gradient_groups["total"][
            "geometry_or_encoder"
        ]["gradient_norm"]
        == 0.0,
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    payload = {
        "experiment": "V4.5.1 quality-policy gradient contract",
        "quality_policy_mode": quality_policy_mode,
        "git_commit": commit,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "baseline_config": str(baseline_config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_iteration": int(source_payload.get("iteration", -1)),
        "load_stats": load_stats,
        "missing_preexisting_tensors": missing_common,
        "shape_mismatched_preexisting_tensors": shape_mismatched_common,
        "freeze_stats": freeze_stats,
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in loss_dict.items()
            if name.startswith("loss_pointer")
        },
        "initialization_equivalence": {
            "pointer_logits_max_abs": initial_pointer_max_abs,
            "pointer_indices_equal": initial_indices_equal,
        },
        "pointer_components": {
            "unary_rms": _rms(outputs.get("selection_pointer_unary_component")),
            "policy_rms": _rms(outputs.get("selection_pointer_policy_component")),
            "content_rms": _rms(
                outputs.get("selection_pointer_content_component")
            ),
            "relation_rms": _rms(outputs.get("selection_pointer_relation_bias")),
            "stop_rms": _rms(outputs.get("selection_pointer_stop_component")),
            "quality_scale": scale.cpu().tolist(),
            "candidate_component_sum_max_abs": component_max_abs,
        },
        "gradient_groups": gradient_groups,
        "gradient_cosines": {
            "sequence_vs_quality_weighted": _cosine(
                gradients["sequence"], gradients["quality_weighted"]
            ),
            "sequence_vs_listwise_weighted": _cosine(
                gradients["sequence"], gradients["listwise_weighted"]
            ),
            "quality_vs_listwise_weighted": _cosine(
                gradients["quality_weighted"],
                gradients["listwise_weighted"],
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")
    if not payload["passed"]:
        raise SystemExit("V4.5.1 quality-policy contract failed")


if __name__ == "__main__":
    main()
