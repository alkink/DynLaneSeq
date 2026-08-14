from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn import functional as F

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint, save_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_model
from dynlaneseq_eg.tools.cache_v20_slot_owned_replacement import (
    FEATURE_FIELDS,
    TARGET_FIELDS,
)
from dynlaneseq_eg.tools.train import seed_everything


MODULE_PREFIX = (
    "structured_query_head.set_selection_head.slot_owned_safe_replacement"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only the V20 replacement head from the exact frozen cache."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-mode", choices=("treatment", "masked"), required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume")
    return parser.parse_args()


def _resolve_shard(path: str, manifest_path: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    local = manifest_path.parent / candidate.name
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError(f"V20 cache shard is unavailable: {path}")


def _load_cache(manifest_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract", {}).get("passed") is not True:
        raise ValueError("V20 cache contract did not pass")
    fields = FEATURE_FIELDS + TARGET_FIELDS
    pieces: dict[str, list[torch.Tensor]] = {name: [] for name in fields}
    image_ids: list[str] = []
    for item in manifest.get("shards", []):
        path = _resolve_shard(str(item["path"]), manifest_path)
        if sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"V20 cache shard digest mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        image_ids.extend(str(value) for value in payload["image_ids"])
        for name in fields:
            pieces[name].append(payload[name])
    cache = {name: torch.cat(values, dim=0) for name, values in pieces.items()}
    if int(cache["source_route"].shape[0]) != int(manifest["images"]):
        raise ValueError("V20 cache image count mismatch")
    manifest["image_ids"] = image_ids
    return cache, manifest


def _batch_indices(
    images: int,
    batch_size: int,
    step: int,
    seed: int,
) -> torch.Tensor:
    if images % batch_size:
        raise ValueError("V20 paired cache requires an exact full-batch partition")
    batches_per_epoch = images // batch_size
    epoch = step // batches_per_epoch
    position = step % batches_per_epoch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1_000_003 * int(epoch))
    permutation = torch.randperm(images, generator=generator)
    start = position * batch_size
    return permutation[start : start + batch_size]


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = cfg.get("v20", {}).get("loss", {})
    action_valid = batch["action_valid"].bool()
    if not torch.equal(action_valid, output["action_valid"]):
        raise RuntimeError("cached and forward V20 action masks disagree")
    batch_size = int(action_valid.shape[0])
    action_count = int(action_valid[0].numel())
    policy_candidate = output["policy_logits"].reshape(batch_size, action_count)
    policy_logits = torch.cat(
        (policy_candidate.new_zeros((batch_size, 1)), policy_candidate), dim=-1
    )
    full_valid = torch.cat(
        (
            torch.ones((batch_size, 1), dtype=torch.bool, device=policy_logits.device),
            action_valid.reshape(batch_size, action_count),
        ),
        dim=-1,
    )
    policy_logits = policy_logits.masked_fill(~full_valid, -1.0e4)
    policy_target = batch["policy_target"].float()
    policy_per_image = -(
        policy_target * F.log_softmax(policy_logits.float(), dim=-1)
    ).sum(dim=-1)
    replacement_image = policy_target[:, 0] < 0.5
    policy_image_weight = 1.0 + replacement_image.float() * (
        float(loss_cfg.get("replacement_image_weight", 3.0)) - 1.0
    )
    policy = (policy_per_image * policy_image_weight).sum() / policy_image_weight.sum()

    flat_valid = action_valid.reshape(-1)
    delta50_logits = output["delta50_logits"].reshape(-1, 3)
    delta75_logits = output["delta75_logits"].reshape(-1, 3)
    delta50_target = batch["delta50_class"][:, 1:].reshape(-1).long()
    delta75_target = batch["delta75_class"][:, 1:].reshape(-1).long()
    class50 = delta50_logits.new_tensor(
        loss_cfg.get("delta50_class_weights", (2.0, 0.25, 4.0))
    )
    class75 = delta75_logits.new_tensor(
        loss_cfg.get("delta75_class_weights", (2.0, 0.25, 2.0))
    )
    delta50 = _masked_mean(
        F.cross_entropy(
            delta50_logits.float(), delta50_target, weight=class50, reduction="none"
        ),
        flat_valid,
    )
    delta75 = _masked_mean(
        F.cross_entropy(
            delta75_logits.float(), delta75_target, weight=class75, reduction="none"
        ),
        flat_valid,
    )

    duplicate_logits = output["duplicate_logits"].reshape(-1)
    abandon_logits = output["abandon_logits"].reshape(-1)
    duplicate_target = batch["duplicate"][:, 1:].reshape(-1).float()
    abandon_target = batch["abandon"][:, 1:].reshape(-1).float()
    duplicate = _masked_mean(
        F.binary_cross_entropy_with_logits(
            duplicate_logits.float(),
            duplicate_target,
            reduction="none",
            pos_weight=duplicate_logits.new_tensor(
                float(loss_cfg.get("duplicate_positive_weight", 2.0))
            ),
        ),
        flat_valid,
    )
    abandon = _masked_mean(
        F.binary_cross_entropy_with_logits(
            abandon_logits.float(),
            abandon_target,
            reduction="none",
            pos_weight=abandon_logits.new_tensor(
                float(loss_cfg.get("abandon_positive_weight", 4.0))
            ),
        ),
        flat_valid,
    )
    delta_iou_target = batch["delta_iou"][:, 1:].reshape(-1).float()
    delta_iou = _masked_mean(
        F.smooth_l1_loss(
            output["delta_iou"].reshape(-1).float(),
            delta_iou_target,
            reduction="none",
            beta=0.05,
        ),
        flat_valid,
    )
    total = (
        float(loss_cfg.get("policy_weight", 1.0)) * policy
        + float(loss_cfg.get("delta50_weight", 1.0)) * delta50
        + float(loss_cfg.get("delta75_weight", 0.5)) * delta75
        + float(loss_cfg.get("duplicate_weight", 0.5)) * duplicate
        + float(loss_cfg.get("abandon_weight", 2.0)) * abandon
        + float(loss_cfg.get("delta_iou_weight", 0.1)) * delta_iou
    )
    with torch.no_grad():
        predicted_action = policy_logits.argmax(dim=-1)
        target_is_keep = policy_target[:, 0] > 0.5
        predicted_keep = predicted_action == 0
        selected_target_mass = policy_target.gather(
            1, predicted_action.unsqueeze(-1)
        ).squeeze(-1)
        diagnostics = {
            "total": total.detach(),
            "policy": policy.detach(),
            "delta50": delta50.detach(),
            "delta75": delta75.detach(),
            "duplicate": duplicate.detach(),
            "abandon": abandon.detach(),
            "delta_iou": delta_iou.detach(),
            "policy_hit": (selected_target_mass > 0.0).float().mean(),
            "opportunity_fraction": (~target_is_keep).float().mean(),
            "replacement_recall": (
                (~predicted_keep & ~target_is_keep).float().sum()
                / (~target_is_keep).float().sum().clamp_min(1.0)
            ),
            "keep_accuracy": (
                (predicted_keep & target_is_keep).float().sum()
                / target_is_keep.float().sum().clamp_min(1.0)
            ),
        }
    return total, diagnostics


def _to_device_batch(
    cache: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value.index_select(0, indices).to(device, non_blocking=True)
        for name, value in cache.items()
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    expected_mode = str(cfg.get("v20", {}).get("context_mode", ""))
    if expected_mode != args.context_mode:
        raise ValueError(
            f"config context mode {expected_mode!r} != CLI {args.context_mode!r}"
        )
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    manifest_path = Path(args.cache_manifest).expanduser().resolve()
    cache, manifest = _load_cache(manifest_path)
    model = build_model(cfg)
    base_iteration = int(
        load_checkpoint(args.base_checkpoint, model, strict=False)
    )
    head = model.structured_query_head.set_selection_head.slot_owned_safe_replacement
    if head is None:
        raise ValueError("V20 replacement module is missing")
    head.to(device)
    head.train()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg.get("optimizer", {}).get("base_lr", 1.0e-4)),
        betas=tuple(float(v) for v in cfg.get("optimizer", {}).get("betas", (0.9, 0.999))),
        weight_decay=float(cfg.get("optimizer", {}).get("weight_decay", 1.0e-4)),
    )
    start_step = 0
    if args.resume:
        resume_iteration = int(
            load_checkpoint(
                args.resume,
                model,
                optimizer=optimizer,
                strict=False,
                restore_rng_state=True,
            )
        )
        start_step = resume_iteration - base_iteration
        if start_step < 0:
            raise ValueError("V20 resume iteration predates its base")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_metrics.jsonl"
    images = int(cache["source_route"].shape[0])
    if images != int(manifest["images"]):
        raise ValueError("V20 manifest/cache population mismatch")
    if int(args.batch_size) < 1:
        raise ValueError("V20 cached batch size must be positive")
    use_amp = bool(cfg.get("training", {}).get("amp", False) and device.type == "cuda")
    amp_dtype = torch.bfloat16
    start_time = time.monotonic()
    window_start = start_time
    window_images = 0
    latest: dict[str, float] = {}

    def save(step: int) -> None:
        iteration = base_iteration + step
        save_checkpoint(
            output_dir / f"iter_{iteration:07d}.pt",
            model,
            optimizer=optimizer,
            iteration=iteration,
            cfg=cfg,
            model_state_prefixes=(MODULE_PREFIX,),
            base_checkpoint=Path(args.base_checkpoint).resolve(),
            include_rng_state=True,
        )

    for step in range(start_step, int(args.steps)):
        indices = _batch_indices(images, int(args.batch_size), step, int(args.seed))
        batch = _to_device_batch(cache, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            output = head(
                candidate_state=batch["candidate_state"],
                p50=batch["p50"],
                p75=batch["p75"],
                expected_iou=batch["expected_iou"],
                legacy_route_logits=batch["legacy_route_logits"],
                counterfactual_valid=batch["counterfactual_valid"],
                source_route=batch["source_route"],
                source_active=batch["source_active"],
                precomputed_relations=batch["curve_relations"],
                force_context_mode=args.context_mode,
            )
            loss, diagnostics = _loss(output, batch, cfg)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite V20 loss at step {step + 1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(
                f"non-finite V20 gradient at step {step + 1}"
            )
        optimizer.step()
        window_images += int(args.batch_size)
        completed = step + 1
        latest = {name: float(value.cpu()) for name, value in diagnostics.items()}
        latest["gradient_norm"] = float(gradient_norm.detach().cpu())
        if completed % 50 == 0 or completed == 1:
            now = time.monotonic()
            row = {
                "step": completed,
                "iteration": base_iteration + completed,
                "context_mode": args.context_mode,
                "images_per_second": window_images / max(now - window_start, 1.0e-9),
                **latest,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            window_start = now
            window_images = 0
        if args.checkpoint_interval > 0 and completed % int(args.checkpoint_interval) == 0:
            save(completed)
    if int(args.steps) % max(int(args.checkpoint_interval), 1):
        save(int(args.steps))
    elapsed = time.monotonic() - start_time
    report = {
        "experiment": "V20 cached slot-owned safe replacement training",
        "context_mode": args.context_mode,
        "config": str(Path(args.config).resolve()),
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_iteration": base_iteration,
        "endpoint_iteration": base_iteration + int(args.steps),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "effective_images": int(args.steps) * int(args.batch_size),
        "elapsed_seconds": elapsed,
        "average_images_per_second": (
            int(args.steps) * int(args.batch_size) / max(elapsed, 1.0e-9)
        ),
        "latest_metrics": latest,
        "test_set_used": False,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
