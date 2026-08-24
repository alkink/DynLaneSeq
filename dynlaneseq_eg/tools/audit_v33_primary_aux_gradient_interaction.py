from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import (
    V25LossWeights,
    v25_lane_object_loss,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v25_image_mediated_lane_objects import (
    _configure_runtime,
    _configured,
    _loss_weights,
    _move_images,
)
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


FIXED_SEED = 3407
DEFAULT_RUNTIME_SEED = FIXED_SEED * 1_000_003 + 11_110
SHARED_PREFIXES = (
    "detector.backbone",
    "detector.fpn",
    "detector.fine_stem",
    "detector.p2_projection",
    "detector.fine_projection",
    "detector.image_fusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure V33 primary/proposal gradient interaction on the exact "
            "shared image feature extractor."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-pairs", type=int, default=32)
    parser.add_argument("--data-start-iteration", type=int, default=0)
    parser.add_argument("--expected-parent-iteration", type=int, default=11_110)
    parser.add_argument("--expected-trained-iteration", type=int, default=13_888)
    parser.add_argument("--runtime-seed", type=int, default=DEFAULT_RUNTIME_SEED)
    parser.add_argument(
        "--relative-step-scales",
        type=float,
        nargs="+",
        default=(1.0e-5, 3.0e-5, 1.0e-4),
        help=(
            "Equal global parameter-relative displacements used for the "
            "cross-batch finite-step control."
        ),
    )
    return parser.parse_args()


def split_primary_auxiliary_weights(
    weights: V25LossWeights,
) -> tuple[V25LossWeights, V25LossWeights]:
    """Return the two objectives whose shared gradients V33 combines."""

    if float(weights.proposal_coverage) <= 0.0:
        raise ValueError("V33 gradient audit requires proposal_coverage > 0")
    primary = replace(weights, proposal_coverage=0.0)
    auxiliary = replace(
        weights,
        existence=0.0,
        row_distribution=0.0,
        point=0.0,
        strip_iou=0.0,
        range=0.0,
        quality50=0.0,
        quality75=0.0,
        smoothness=0.0,
        order=0.0,
        duplicate=0.0,
        visibility=0.0,
        proposal_coverage=1.0,
        tail_emphasis=0.0,
    )
    return primary, auxiliary


def _parameter_groups(name: str) -> tuple[str, ...]:
    if not name.startswith("detector."):
        raise ValueError(f"unexpected shared parameter name: {name}")
    relative = name[len("detector.") :]
    top = relative.split(".", 1)[0]
    groups = ["all_shared", top]
    if top == "backbone":
        parts = relative.split(".")
        if len(parts) > 1:
            groups.append(".".join(parts[:2]))
    elif top == "fpn":
        parts = relative.split(".")
        if len(parts) > 2:
            groups.append(".".join(parts[:3]))
    return tuple(dict.fromkeys(groups))


def gradient_interaction_metrics(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    primary_gradients: tuple[torch.Tensor | None, ...],
    auxiliary_gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, dict[str, float | int]]:
    if not (
        len(named_parameters)
        == len(primary_gradients)
        == len(auxiliary_gradients)
    ):
        raise ValueError("parameter and gradient tuple lengths differ")
    accumulators: dict[str, dict[str, float]] = {}
    for (name, _parameter), primary, auxiliary in zip(
        named_parameters, primary_gradients, auxiliary_gradients
    ):
        if primary is None or auxiliary is None:
            continue
        p = primary.detach().float().reshape(-1)
        a = auxiliary.detach().float().reshape(-1)
        if not bool(torch.isfinite(p).all() and torch.isfinite(a).all()):
            raise FloatingPointError(f"non-finite shared gradient: {name}")
        p_sq = float(torch.dot(p, p).cpu())
        a_sq = float(torch.dot(a, a).cpu())
        if p_sq <= 0.0 or a_sq <= 0.0:
            continue
        dot = float(torch.dot(p, a).cpu())
        for group in _parameter_groups(name):
            values = accumulators.setdefault(
                group,
                {
                    "dot": 0.0,
                    "primary_norm_sq": 0.0,
                    "auxiliary_norm_sq": 0.0,
                    "negative_dot_abs": 0.0,
                    "dot_abs": 0.0,
                    "joint_parameter_tensors": 0.0,
                    "negative_parameter_tensors": 0.0,
                },
            )
            values["dot"] += dot
            values["primary_norm_sq"] += p_sq
            values["auxiliary_norm_sq"] += a_sq
            values["dot_abs"] += abs(dot)
            values["negative_dot_abs"] += abs(min(dot, 0.0))
            values["joint_parameter_tensors"] += 1.0
            values["negative_parameter_tensors"] += float(dot < 0.0)
    result: dict[str, dict[str, float | int]] = {}
    for group, values in accumulators.items():
        primary_norm = values["primary_norm_sq"] ** 0.5
        auxiliary_norm = values["auxiliary_norm_sq"] ** 0.5
        tensor_count = int(values["joint_parameter_tensors"])
        result[group] = {
            "cosine": values["dot"]
            / max(primary_norm * auxiliary_norm, 1.0e-30),
            "primary_norm": primary_norm,
            "auxiliary_norm": auxiliary_norm,
            "auxiliary_over_primary_norm": auxiliary_norm
            / max(primary_norm, 1.0e-30),
            "negative_tensor_fraction": values["negative_parameter_tensors"]
            / max(float(tensor_count), 1.0),
            "negative_dot_energy_fraction": values["negative_dot_abs"]
            / max(values["dot_abs"], 1.0e-30),
            "joint_parameter_tensors": tensor_count,
        }
    if "all_shared" not in result:
        raise RuntimeError("primary and auxiliary losses share no nonzero gradients")
    return result


def _global_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
    total = 0.0
    for gradient in gradients:
        if gradient is not None:
            value = gradient.detach().float()
            total += float(torch.sum(value * value).cpu())
    return total**0.5


def _parameter_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        value = parameter.detach().float()
        total += float(torch.sum(value * value).cpu())
    return total**0.5


def _apply_normalized_direction(
    parameters: list[torch.nn.Parameter],
    originals: list[torch.Tensor],
    gradients: tuple[torch.Tensor | None, ...],
    *,
    relative_step: float,
    parameter_norm: float,
) -> float:
    gradient_norm = _global_norm(gradients)
    if gradient_norm <= 0.0:
        raise RuntimeError("cannot apply a zero gradient direction")
    scale = float(relative_step) * float(parameter_norm) / gradient_norm
    with torch.no_grad():
        for parameter, original, gradient in zip(parameters, originals, gradients):
            if gradient is None:
                parameter.copy_(original)
            else:
                parameter.copy_(original - scale * gradient.detach())
    return scale


def _restore_parameters(
    parameters: list[torch.nn.Parameter], originals: list[torch.Tensor]
) -> None:
    with torch.no_grad():
        for parameter, original in zip(parameters, originals):
            parameter.copy_(original)


def _tensor_summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "minimum": float(tensor.amin()),
        "maximum": float(tensor.amax()),
    }


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("cannot summarize an empty audit")
    group_names = sorted(
        {group for pair in pairs for group in pair["gradient_groups"]}
    )
    groups: dict[str, Any] = {}
    metric_names = (
        "cosine",
        "primary_norm",
        "auxiliary_norm",
        "auxiliary_over_primary_norm",
        "negative_tensor_fraction",
        "negative_dot_energy_fraction",
    )
    for group in group_names:
        rows = [pair["gradient_groups"][group] for pair in pairs if group in pair["gradient_groups"]]
        summary = {
            name: _tensor_summary([float(row[name]) for row in rows])
            for name in metric_names
        }
        summary["negative_pair_fraction"] = sum(
            float(row["cosine"]) < 0.0 for row in rows
        ) / float(len(rows))
        summary["pairs"] = len(rows)
        summary["joint_parameter_tensors"] = int(rows[0]["joint_parameter_tensors"])
        groups[group] = summary

    scales = sorted(
        {
            scale
            for pair in pairs
            for scale in pair["cross_batch_virtual_steps"]
        },
        key=float,
    )
    virtual: dict[str, Any] = {}
    for scale in scales:
        rows = [pair["cross_batch_virtual_steps"][scale] for pair in pairs]
        primary_delta = [float(row["primary_direction_relative_delta"]) for row in rows]
        auxiliary_delta = [float(row["auxiliary_direction_relative_delta"]) for row in rows]
        virtual[scale] = {
            "primary_direction_relative_delta": _tensor_summary(primary_delta),
            "auxiliary_direction_relative_delta": _tensor_summary(auxiliary_delta),
            "primary_direction_harm_fraction": sum(value > 0.0 for value in primary_delta)
            / float(len(rows)),
            "auxiliary_direction_harm_fraction": sum(value > 0.0 for value in auxiliary_delta)
            / float(len(rows)),
            "auxiliary_worse_than_primary_fraction": sum(
                auxiliary > primary
                for auxiliary, primary in zip(auxiliary_delta, primary_delta)
            )
            / float(len(rows)),
        }
    return {
        "pairs": len(pairs),
        "primary_loss": _tensor_summary([float(pair["primary_loss"]) for pair in pairs]),
        "auxiliary_loss": _tensor_summary([float(pair["auxiliary_loss"]) for pair in pairs]),
        "gradient_groups": groups,
        "cross_batch_virtual_steps": virtual,
    }


def classify_interaction(summary: dict[str, Any]) -> dict[str, Any]:
    shared = summary["gradient_groups"]["all_shared"]
    cosine_median = float(shared["cosine"]["median"])
    negative_pair_fraction = float(shared["negative_pair_fraction"])
    largest_scale = sorted(
        summary["cross_batch_virtual_steps"], key=float
    )[-1]
    virtual = summary["cross_batch_virtual_steps"][largest_scale]
    auxiliary_harm = float(virtual["auxiliary_direction_harm_fraction"])
    primary_harm = float(virtual["primary_direction_harm_fraction"])
    conflict = (
        cosine_median < -0.10
        or negative_pair_fraction > 0.60
        or (auxiliary_harm > 0.70 and primary_harm < 0.50)
    )
    aligned = (
        cosine_median > 0.10
        and negative_pair_fraction < 0.40
        and auxiliary_harm < 0.60
    )
    if conflict:
        label = "gradient_conflict"
    elif aligned:
        label = "aligned_or_redundant"
    else:
        label = "mixed_or_orthogonal"
    return {
        "label": label,
        "checks": {
            "median_cosine_below_minus_0p10": cosine_median < -0.10,
            "negative_pair_fraction_above_0p60": negative_pair_fraction > 0.60,
            "auxiliary_virtual_harm_above_0p70": auxiliary_harm > 0.70,
            "primary_virtual_harm_below_0p50": primary_harm < 0.50,
            "median_cosine_above_0p10": cosine_median > 0.10,
            "negative_pair_fraction_below_0p40": negative_pair_fraction < 0.40,
            "auxiliary_virtual_harm_below_0p60": auxiliary_harm < 0.60,
        },
        "measured": {
            "all_shared_cosine_median": cosine_median,
            "all_shared_negative_pair_fraction": negative_pair_fraction,
            "largest_relative_step": float(largest_scale),
            "auxiliary_direction_harm_fraction": auxiliary_harm,
            "primary_direction_harm_fraction": primary_harm,
        },
        "warning": (
            "This is a local gradient/directional diagnostic. It explains the "
            "V33 mechanism but is not itself an F1 result."
        ),
    }


def _metadata_digest(metas: Any) -> str:
    encoded = json.dumps(metas, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_model(
    cfg: dict[str, Any],
    checkpoint: Path,
    *,
    advanced_partial_init: bool,
    expected_iteration: int,
    device: torch.device,
    channels_last: bool,
) -> tuple[DynLaneSeqV25, int, dict[str, Any]]:
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("V33 gradient audit requires DynLaneSeqV25")
    initialization: dict[str, Any] = {
        "advanced_partial_init": bool(advanced_partial_init),
    }
    if advanced_partial_init:
        base_cfg = copy.deepcopy(cfg)
        base_cfg.setdefault("v25", {})["enable_dual_energy_multi_path"] = False
        base = build_model(base_cfg)
        base_iteration = int(load_checkpoint(checkpoint, base, strict=True))
        iteration = int(load_checkpoint(checkpoint, model, strict=False))
        advanced_state = model.state_dict()
        mismatches = [
            name
            for name, value in base.state_dict().items()
            if name not in advanced_state
            or tuple(advanced_state[name].shape) != tuple(value.shape)
            or not torch.equal(advanced_state[name].cpu(), value.cpu())
        ]
        if base_iteration != iteration or mismatches:
            raise ValueError(
                "advanced initialization changed shared tensors: "
                + json.dumps(
                    {
                        "base_iteration": base_iteration,
                        "advanced_iteration": iteration,
                        "mismatch_count": len(mismatches),
                        "first_mismatches": mismatches[:10],
                    }
                )
            )
        initialization["shared_tensor_parity"] = True
        del base
    else:
        iteration = int(load_checkpoint(checkpoint, model, strict=True))
        initialization["strict_checkpoint_load"] = True
    if iteration != int(expected_iteration):
        raise ValueError(
            f"checkpoint iteration={iteration}, expected={expected_iteration}"
        )
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    model.train()
    return model, iteration, initialization


def _primary_loss_value(
    model: DynLaneSeqV25,
    batch: tuple[Any, Any, Any],
    *,
    device: torch.device,
    channels_last: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    input_w: int,
    primary_weights: V25LossWeights,
) -> float:
    images, targets, _metas = batch
    images = _move_images(images, device=device, channels_last=channels_last)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        output = model(images)
        loss, _diagnostics = v25_lane_object_loss(
            output,
            targets,
            input_w=input_w,
            weights=primary_weights,
        )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("non-finite virtual-step primary loss")
    return float(loss.cpu())


def audit_checkpoint(
    *,
    label: str,
    cfg: dict[str, Any],
    checkpoint: Path,
    advanced_partial_init: bool,
    expected_iteration: int,
    device: torch.device,
    runtime_seed: int,
    num_pairs: int,
    data_start_iteration: int,
    relative_step_scales: list[float],
) -> dict[str, Any]:
    channels_last = bool(cfg["training"].get("channels_last", False))
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    amp_name = str(cfg["training"].get("amp_dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    model, iteration, initialization = _load_model(
        cfg,
        checkpoint,
        advanced_partial_init=advanced_partial_init,
        expected_iteration=expected_iteration,
        device=device,
        channels_last=channels_last,
    )
    weights = _loss_weights(cfg)
    primary_weights, auxiliary_weights = split_primary_auxiliary_weights(weights)
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(SHARED_PREFIXES)
    ]
    if not named_parameters:
        raise RuntimeError("no shared image parameters matched the V33 contract")
    parameters = [parameter for _name, parameter in named_parameters]
    shared_parameter_norm = _parameter_norm(parameters)

    seed_everything(int(runtime_seed))
    loader = build_dataloader(
        cfg,
        split="train",
        training=True,
        start_iteration=int(data_start_iteration),
    )
    iterator = iter(loader)
    pairs: list[dict[str, Any]] = []
    manifest: list[dict[str, str]] = []
    for pair_index in range(int(num_pairs)):
        train_batch = next(iterator)
        probe_batch = next(iterator)
        train_images, train_targets, train_metas = train_batch
        train_images = _move_images(
            train_images, device=device, channels_last=channels_last
        )
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(train_images)
            primary_loss, primary_diagnostics = v25_lane_object_loss(
                output,
                train_targets,
                input_w=int(cfg["model"]["input_w"]),
                weights=primary_weights,
            )
            auxiliary_loss, auxiliary_diagnostics = v25_lane_object_loss(
                output,
                train_targets,
                input_w=int(cfg["model"]["input_w"]),
                weights=auxiliary_weights,
            )
        if not bool(
            torch.isfinite(primary_loss).all() and torch.isfinite(auxiliary_loss).all()
        ):
            raise FloatingPointError("non-finite V33 gradient audit loss")
        primary_gradients = torch.autograd.grad(
            primary_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        auxiliary_gradients = torch.autograd.grad(
            auxiliary_loss,
            parameters,
            retain_graph=False,
            allow_unused=True,
        )
        gradient_groups = gradient_interaction_metrics(
            named_parameters, primary_gradients, auxiliary_gradients
        )
        del output, train_images

        originals = [parameter.detach().clone() for parameter in parameters]
        probe_before = _primary_loss_value(
            model,
            probe_batch,
            device=device,
            channels_last=channels_last,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            input_w=int(cfg["model"]["input_w"]),
            primary_weights=primary_weights,
        )
        virtual_steps: dict[str, Any] = {}
        for relative_step in relative_step_scales:
            primary_scale = _apply_normalized_direction(
                parameters,
                originals,
                primary_gradients,
                relative_step=float(relative_step),
                parameter_norm=shared_parameter_norm,
            )
            probe_after_primary = _primary_loss_value(
                model,
                probe_batch,
                device=device,
                channels_last=channels_last,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                input_w=int(cfg["model"]["input_w"]),
                primary_weights=primary_weights,
            )
            _restore_parameters(parameters, originals)
            auxiliary_scale = _apply_normalized_direction(
                parameters,
                originals,
                auxiliary_gradients,
                relative_step=float(relative_step),
                parameter_norm=shared_parameter_norm,
            )
            probe_after_auxiliary = _primary_loss_value(
                model,
                probe_batch,
                device=device,
                channels_last=channels_last,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                input_w=int(cfg["model"]["input_w"]),
                primary_weights=primary_weights,
            )
            _restore_parameters(parameters, originals)
            denominator = max(abs(probe_before), 1.0e-12)
            virtual_steps[f"{float(relative_step):.1e}"] = {
                "probe_primary_before": probe_before,
                "probe_primary_after_primary_direction": probe_after_primary,
                "probe_primary_after_auxiliary_direction": probe_after_auxiliary,
                "primary_direction_relative_delta": (
                    probe_after_primary - probe_before
                )
                / denominator,
                "auxiliary_direction_relative_delta": (
                    probe_after_auxiliary - probe_before
                )
                / denominator,
                "primary_gradient_to_parameter_scale": primary_scale,
                "auxiliary_gradient_to_parameter_scale": auxiliary_scale,
            }
        _restore_parameters(parameters, originals)
        del originals
        pair = {
            "pair_index": pair_index,
            "primary_loss": float(primary_loss.detach().cpu()),
            "auxiliary_loss": float(auxiliary_loss.detach().cpu()),
            "primary_row_loss": float(primary_diagnostics["loss_row_distribution"].cpu()),
            "auxiliary_proposal_coverage_loss": float(
                auxiliary_diagnostics["loss_proposal_coverage"].cpu()
            ),
            "gradient_groups": gradient_groups,
            "cross_batch_virtual_steps": virtual_steps,
            "train_metadata_sha256": _metadata_digest(train_metas),
            "probe_metadata_sha256": _metadata_digest(probe_batch[2]),
        }
        pairs.append(pair)
        manifest.append(
            {
                "train": pair["train_metadata_sha256"],
                "probe": pair["probe_metadata_sha256"],
            }
        )
        del primary_gradients, auxiliary_gradients, primary_loss, auxiliary_loss
        model.zero_grad(set_to_none=True)
    summary = summarize_pairs(pairs)
    verdict = classify_interaction(summary)
    payload = {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_iteration": iteration,
        "initialization": initialization,
        "runtime_seed": int(runtime_seed),
        "data_start_iteration": int(data_start_iteration),
        "batch_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "shared_parameter_prefixes": list(SHARED_PREFIXES),
        "shared_parameter_tensors": len(named_parameters),
        "shared_parameter_norm": shared_parameter_norm,
        "relative_step_scales": [float(value) for value in relative_step_scales],
        "summary": summary,
        "verdict": verdict,
        "per_pair": pairs,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def main() -> None:
    args = parse_args()
    if args.num_pairs < 4:
        raise ValueError("V33-GI requires at least four train/probe batch pairs")
    if any(value <= 0.0 for value in args.relative_step_scales):
        raise ValueError("relative step scales must be positive")
    seed_everything(FIXED_SEED)
    root = Path(args.dataset_root).expanduser().resolve()
    train_population = official_v23_culane_list_contract(root, split="train")
    val_population = official_v23_culane_list_contract(root, split="val")
    configured_args = argparse.Namespace(
        config=args.config,
        dataset_root=str(root),
        num_workers=int(args.num_workers),
    )
    cfg = _configured(
        configured_args,
        official_train_list=str(train_population["list_path"]),
        official_val_list=str(val_population["list_path"]),
    )
    cfg["training"]["batch_size"] = int(args.batch_size)
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = int(args.num_workers) > 0
    if not bool(cfg.get("v25", {}).get("enable_dual_energy_multi_path", False)):
        raise ValueError("V33-GI requires the auxiliary V25 graph")
    if bool(cfg.get("v25", {}).get("enable_proposal_fusion", True)):
        raise ValueError("V33-GI must use the training-only auxiliary arm")
    device = torch.device(args.device)
    _configure_runtime(cfg, device)
    parent = audit_checkpoint(
        label="parent_with_new_auxiliary_graph",
        cfg=cfg,
        checkpoint=Path(args.parent_checkpoint).expanduser().resolve(),
        advanced_partial_init=True,
        expected_iteration=int(args.expected_parent_iteration),
        device=device,
        runtime_seed=int(args.runtime_seed),
        num_pairs=int(args.num_pairs),
        data_start_iteration=int(args.data_start_iteration),
        relative_step_scales=list(args.relative_step_scales),
    )
    trained = audit_checkpoint(
        label="trained_auxiliary_endpoint",
        cfg=cfg,
        checkpoint=Path(args.trained_checkpoint).expanduser().resolve(),
        advanced_partial_init=False,
        expected_iteration=int(args.expected_trained_iteration),
        device=device,
        runtime_seed=int(args.runtime_seed),
        num_pairs=int(args.num_pairs),
        data_start_iteration=int(args.data_start_iteration),
        relative_step_scales=list(args.relative_step_scales),
    )
    manifests_match = (
        parent["batch_manifest_sha256"] == trained["batch_manifest_sha256"]
    )
    if not manifests_match:
        raise RuntimeError("parent/trained audits did not see exact paired batches")
    labels = {parent["verdict"]["label"], trained["verdict"]["label"]}
    if "gradient_conflict" in labels:
        combined = "gradient_conflict_detected"
    elif labels == {"aligned_or_redundant"}:
        combined = "aligned_but_no_f1_gain_supports_redundancy"
    else:
        combined = "mixed_interaction_requires_layer_localization"
    payload = {
        "experiment": "V33-GI shared primary/auxiliary gradient interaction audit",
        "question": (
            "Did V33-B fail because proposal-coverage gradients conflict with "
            "the direct-primary objective, or because they are aligned but redundant?"
        ),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(Path(args.config).expanduser().resolve()),
        "official_train_population_contract": train_population,
        "official_val_population_contract": val_population,
        "num_pairs": int(args.num_pairs),
        "images_per_checkpoint": int(args.num_pairs) * int(args.batch_size) * 2,
        "exact_batch_manifest_match": manifests_match,
        "parent": parent,
        "trained": trained,
        "combined_verdict": combined,
        "decision_contract": {
            "gradient_conflict_detected": (
                "Close shared-encoder auxiliary training; inspect layer-local "
                "conflict before authorizing at most one surgical detach gate."
            ),
            "aligned_but_no_f1_gain_supports_redundancy": (
                "Close the direct-primary auxiliary family; more gradient or "
                "loss-weight sweeps are not justified."
            ),
            "mixed_interaction_requires_layer_localization": (
                "Use the reported layer groups to decide whether one localized "
                "gradient boundary is testable; do not start a long run."
            ),
            "test_set_used": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
        },
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "combined_verdict": combined,
        "parent_verdict": parent["verdict"],
        "trained_verdict": trained["verdict"],
        "exact_batch_manifest_match": manifests_match,
    }, indent=2))


if __name__ == "__main__":
    main()
