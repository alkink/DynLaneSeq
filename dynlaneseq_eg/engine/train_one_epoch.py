from __future__ import annotations

from typing import Any
import time

import torch
from torch.nn.utils import clip_grad_norm_

from dynlaneseq_eg.modeling.common import nested_to_device
from .logger import match_stats


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _sampler_alpha(cfg: dict[str, Any], iteration: int) -> float:
    sched = cfg.get("sampler_curriculum", {})
    warmup = int(sched.get("warmup_iters", 1000))
    decay = int(sched.get("decay_iters", 1000))
    if iteration < warmup:
        return float(sched.get("alpha_start", 1.0))
    if iteration >= warmup + decay:
        return float(sched.get("alpha_end", 0.0))
    t = (iteration - warmup) / max(1, decay)
    return (1 - t) * float(sched.get("alpha_start", 1.0)) + t * float(sched.get("alpha_end", 0.0))


def _sampler_beta(cfg: dict[str, Any], iteration: int) -> float:
    sched = cfg.get("sampler_curriculum", {})
    beta_start = float(sched.get("beta_start", 1.0))
    beta_mid = float(sched.get("beta_mid", 0.5))
    beta_end = float(sched.get("beta_end", 0.0))
    warmup = int(sched.get("warmup_iters", 1000))
    decay = int(sched.get("decay_iters", 1000))
    if iteration < warmup:
        return beta_start
    if iteration < warmup + decay:
        t = (iteration - warmup) / max(1, decay)
        return (1 - t) * beta_start + t * beta_mid
    if iteration < warmup + 2 * decay:
        t = (iteration - warmup - decay) / max(1, decay)
        return (1 - t) * beta_mid + t * beta_end
    return beta_end


def forward_with_matches(model, images, targets, matcher, cfg, iteration):
    name = cfg.get("model", {}).get("name", "DynLaneSeqS0")
    if name in {"DynLaneSeqS2", "DynLaneSeqS3"}:
        probe = model(images, sampler_alpha=0.0)
        matches = matcher(probe["coarse"], targets)
        outputs = model(images, targets=targets, matches=matches, sampler_alpha=_sampler_alpha(cfg, iteration))
        return outputs, matches
    if name == "DynLaneSeqS4":
        probe = model(images, sampler_alpha=0.0, sampler_beta=0.0)
        matches = matcher(probe["coarse"], targets)
        outputs = model(
            images,
            targets=targets,
            matches=matches,
            sampler_alpha=_sampler_alpha(cfg, iteration),
            sampler_beta=_sampler_beta(cfg, iteration),
        )
        return outputs, matches
    outputs = model(images)
    matches = matcher(outputs, targets)
    return outputs, matches


def output_debug_stats(outputs) -> dict[str, torch.Tensor]:
    stats = {}
    evidence = outputs.get("evidence") if isinstance(outputs, dict) else None
    for evidence_dict in [evidence, outputs.get("geometry_evidence") if isinstance(outputs, dict) else None]:
        if not isinstance(evidence_dict, dict):
            continue
        for key, value in evidence_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                stats[key] = value
        if "evidence_scale" in evidence_dict:
            stats["evidence_scale"] = evidence_dict["evidence_scale"]
        if "sample_x_rows" in evidence_dict:
            stats["sample_x_mean"] = evidence_dict["sample_x_rows"].detach().mean()
        if "E_seq" in evidence_dict:
            stats["evidence_abs_mean"] = evidence_dict["E_seq"].detach().abs().mean()
    return stats


def train_one_epoch(
    model,
    dataloader,
    matcher,
    criterion,
    optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    start_iter: int = 0,
    max_iters: int | None = None,
    scaler=None,
    scheduler=None,
    logger=None,
    visualizer=None,
    checkpoint_saver=None,
) -> int:
    model.train()
    amp = bool(cfg.get("training", {}).get("amp", False))
    clip_norm = float(cfg.get("training", {}).get("clip_grad_norm", 1.0))
    log_interval = int(cfg.get("training", {}).get("log_interval", 10))
    iteration = start_iter
    max_iters = max_iters or int(cfg.get("training", {}).get("max_iters", len(dataloader)))
    end_iter = start_iter + max_iters
    wall_start = time.perf_counter()
    loader_len = max(len(dataloader), 1) if hasattr(dataloader, "__len__") else 1
    processed_images = 0
    while iteration < end_iter:
        for images, targets, metas in dataloader:
            if iteration >= end_iter:
                break
            processed_images += int(images.shape[0])
            images = images.to(device, non_blocking=True)
            targets = nested_to_device(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs, matches = forward_with_matches(model, images, targets, matcher, cfg, iteration)
                loss_dict = criterion(outputs, targets, matches)
                loss = loss_dict["loss_total"]
            if not torch.isfinite(loss):
                print(f"iter {iteration + 1:07d} | non-finite loss; skipping optimizer step")
                iteration += 1
                continue
            if scaler is not None and amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), clip_norm)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                else:
                    print(f"iter {iteration + 1:07d} | non-finite grad norm; skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    iteration += 1
                    continue
            else:
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), clip_norm)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                else:
                    print(f"iter {iteration + 1:07d} | non-finite grad norm; skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    iteration += 1
                    continue
            if logger is not None:
                stats = {k: v for k, v in loss_dict.items()}
                stats.update(match_stats(outputs, matches))
                stats.update(output_debug_stats(outputs))
                stats["grad_norm"] = grad_norm
                stats["lr_model"] = optimizer.param_groups[-1]["lr"]
                logger.update(**stats)
                if (iteration + 1) % log_interval == 0:
                    done = max(iteration + 1 - start_iter, 1)
                    total = max(max_iters, 1)
                    elapsed = time.perf_counter() - wall_start
                    sec_per_iter = elapsed / done
                    img_per_sec = processed_images / max(elapsed, 1e-6)
                    eta = sec_per_iter * max(total - done, 0)
                    pct = 100.0 * done / total
                    epoch = float(iteration + 1) / float(loader_len)
                    prefix = (
                        f"iter {iteration + 1:07d}/{end_iter:07d} "
                        f"({pct:5.1f}%) | epoch {epoch:.2f} | "
                        f"{sec_per_iter:.3f}s/it | {img_per_sec:.1f} img/s | "
                        f"elapsed {_format_duration(elapsed)} | "
                        f"eta {_format_duration(eta)} | "
                    )
                    print(logger.format_and_reset(prefix=prefix))
            if visualizer is not None:
                visualizer(images, targets, metas, outputs, iteration + 1)
            iteration += 1
            if checkpoint_saver is not None:
                checkpoint_saver(iteration)
    return iteration
