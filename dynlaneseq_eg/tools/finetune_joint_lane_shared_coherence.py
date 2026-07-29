from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint, save_checkpoint
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.probe_query_conditioned_dense_curve import (
    QueryConditionedDenseCurveProbe,
    matched_dense_curve_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short causal fine-tune for lane-shared visual coherence. The "
            "backbone stays frozen; the FPN/P2 projection, structured decoder, "
            "and a pretrained lane-shared dynamic mask probe are optimized "
            "jointly. This is diagnostic and does not define a final model."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--model-learning-rate", type=float, default=2e-5)
    parser.add_argument("--fpn-learning-rate", type=float, default=1e-5)
    parser.add_argument("--probe-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coherence-weight", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--evidence-width", type=int, default=400)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-probe", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def select_joint_trainable_modules(model: nn.Module) -> dict[str, int]:
    """Unfreeze only the path needed to test the causal hypothesis."""

    model.requires_grad_(False)
    modules = {
        "fpn": model.encoder.fpn,
        "p2_projection": model.encoder.proj,
        "structured_decoder": model.structured_query_head,
    }
    if model.encoder.seg_aux_head is not None:
        modules["seg_aux"] = model.encoder.seg_aux_head
    if model.encoder.centerline_aux_head is not None:
        modules["centerline_aux"] = model.encoder.centerline_aux_head
    for module in modules.values():
        if module is not None:
            module.requires_grad_(True)
    return {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in modules.items()
        if module is not None
    }


def _module_parameters(module: nn.Module | None) -> list[nn.Parameter]:
    if module is None:
        return []
    return [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def build_joint_optimizer(
    model: nn.Module,
    probe: nn.Module,
    *,
    model_lr: float,
    fpn_lr: float,
    probe_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    fpn_parameters = (
        _module_parameters(model.encoder.fpn)
        + _module_parameters(model.encoder.proj)
    )
    model_parameters = (
        _module_parameters(model.structured_query_head)
        + _module_parameters(model.encoder.seg_aux_head)
        + _module_parameters(model.encoder.centerline_aux_head)
    )
    groups = [
        {
            "params": fpn_parameters,
            "lr": float(fpn_lr),
            "weight_decay": float(weight_decay),
            "name": "fpn",
        },
        {
            "params": model_parameters,
            "lr": float(model_lr),
            "weight_decay": float(weight_decay),
            "name": "structured_model",
        },
        {
            "params": list(probe.parameters()),
            "lr": float(probe_lr),
            "weight_decay": float(weight_decay),
            "name": "lane_shared_probe",
        },
    ]
    return torch.optim.AdamW(
        [group for group in groups if group["params"]],
        betas=(0.9, 0.999),
    )


def _load_probe_state(path: str | Path, probe: nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("probe", payload) if isinstance(payload, dict) else payload
    probe.load_state_dict(state, strict=True)
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        return dict(payload["metadata"])
    return {}


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(int(args.num_workers) > 0)
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg["training"]["seed"] = int(args.seed)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    if (
        amp_dtype == torch.bfloat16
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.float16
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(
        args.checkpoint,
        model,
        strict=False,
    )
    if model.structured_query_head is None:
        raise ValueError("joint lane coherence requires structured_query")
    trainable_modules = select_joint_trainable_modules(model)
    model = model.to(device)

    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(
        structured_cfg.get("num_instances", model_cfg.get("num_slots", 0))
    )
    num_groups = int(structured_cfg.get("num_groups", 1))
    if num_instances % num_groups != 0:
        raise ValueError("num_instances must be divisible by num_groups")
    group_size = num_instances // num_groups
    probe = QueryConditionedDenseCurveProbe(
        in_dim=int(model_cfg.get("dim", 256)),
        state_dim=int(model_cfg.get("dim", 256)),
        hidden_dim=int(args.hidden_dim),
        num_rows=int(model_cfg.get("num_rows", 72)),
        evidence_width=int(args.evidence_width),
        input_w=int(model_cfg.get("input_w", 800)),
        conditioning_mode="lane_shared",
        explicit_coordinates=True,
    )
    probe_metadata = _load_probe_state(args.probe_checkpoint, probe)
    probe = probe.to(device)

    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    optimizer = build_joint_optimizer(
        model,
        probe,
        model_lr=float(args.model_learning_rate),
        fpn_lr=float(args.fpn_learning_rate),
        probe_lr=float(args.probe_learning_rate),
        weight_decay=float(args.weight_decay),
    )
    loader = build_dataloader(cfg, split="train", training=True)
    iterator = iter(loader)
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )

    model.train()
    probe.train()
    optimizer.zero_grad(set_to_none=True)
    initial_summary: dict[str, float] | None = None
    final_summary: dict[str, float] | None = None
    wall_start = time.perf_counter()
    progress = tqdm(
        range(1, int(args.steps) + 1),
        desc="joint lane-shared coherence fine-tune",
        ncols=112,
    )
    for step in progress:
        try:
            images, targets, _metas = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets, _metas = next(iterator)
        if channels_last:
            images = images.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)

        with _autocast_context(device, amp_dtype):
            outputs = model(images, return_features=True)
            p2 = outputs.pop("features")
            matches = matcher(outputs, targets)
            criterion.set_iteration(int(checkpoint_iteration) + step - 1)
            main_losses = criterion(outputs, targets, matches)
            coherence_logits = probe(
                p2,
                outputs["structured_row_tokens"],
            )
            coherence_loss, lane_count, row_count = matched_dense_curve_loss(
                coherence_logits,
                targets,
                matches,
                group_size=group_size,
                input_w=int(model_cfg.get("input_w", 800)),
            )
            total_loss = (
                main_losses["loss_total"]
                + float(args.coherence_weight) * coherence_loss
            )
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(
                f"non-finite joint loss at step {step}: {float(total_loss)}"
            )
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ],
            max_norm=1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        summary = {
            "main_loss": float(main_losses["loss_total"].detach()),
            "coherence_loss": float(coherence_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "grad_norm": float(grad_norm.detach()),
            "matched_lanes": int(lane_count),
            "valid_rows": int(row_count),
        }
        if initial_summary is None:
            initial_summary = dict(summary)
        final_summary = dict(summary)
        if step % int(args.log_interval) == 0 or step == int(args.steps):
            progress.set_postfix(
                main=f"{summary['main_loss']:.3f}",
                coherent=f"{summary['coherence_loss']:.3f}",
                grad=f"{summary['grad_norm']:.2f}",
            )

    output_checkpoint = Path(args.output_checkpoint)
    save_checkpoint(
        output_checkpoint,
        model,
        optimizer=None,
        scaler=None,
        iteration=int(checkpoint_iteration) + int(args.steps),
        cfg=cfg,
    )
    output_probe = Path(args.output_probe)
    output_probe.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "probe": probe.state_dict(),
            "metadata": {
                "conditioning_mode": "lane_shared",
                "explicit_coordinates": True,
                "base_checkpoint": args.checkpoint,
                "candidate_checkpoint": str(output_checkpoint),
                "steps": int(args.steps),
            },
        },
        output_probe,
    )
    payload = {
        "diagnostic_only": True,
        "hypothesis": (
            "A lane-shared dynamic visual filter can make P2 evidence coherent "
            "only when P2 and the structured decoder receive its gradient."
        ),
        "config": args.config,
        "base_checkpoint": args.checkpoint,
        "base_iteration": int(checkpoint_iteration),
        "probe_checkpoint": args.probe_checkpoint,
        "probe_source_metadata": probe_metadata,
        "candidate_checkpoint": str(output_checkpoint),
        "candidate_probe": str(output_probe),
        "candidate_iteration": int(checkpoint_iteration) + int(args.steps),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "coherence_weight": float(args.coherence_weight),
        "learning_rates": {
            "fpn": float(args.fpn_learning_rate),
            "structured_model": float(args.model_learning_rate),
            "probe": float(args.probe_learning_rate),
        },
        "trainable_parameters": dict(
            trainable_modules,
            lane_shared_probe=sum(p.numel() for p in probe.parameters()),
        ),
        "initial_step": initial_summary,
        "final_step": final_summary,
        "wall_seconds": time.perf_counter() - wall_start,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
