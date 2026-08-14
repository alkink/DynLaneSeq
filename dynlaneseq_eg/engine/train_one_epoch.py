from __future__ import annotations

import math
from typing import Any
import time

import torch
from torch.nn.utils import clip_grad_norm_

from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.losses.loss_s0 import (
    build_pointer_cluster_soft_targets,
    build_pointer_sequence_targets,
)
from .frozen_training import set_frozen_detector_eval
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


def clip_optimizer_gradients(model, optimizer, max_norm: float, mode: str = "global") -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "global":
        return clip_grad_norm_(model.parameters(), max_norm)
    if mode not in {"optimizer_group", "optimizer_groups", "per_group"}:
        raise ValueError(f"Unsupported clip_grad_norm_mode: {mode}")

    group_norms = []
    for group in optimizer.param_groups:
        params = [param for param in group["params"] if param.grad is not None]
        if params:
            group_norms.append(clip_grad_norm_(params, max_norm).detach().float())
    if not group_norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(group_norms))


def _v18_conflict_safe_backward(
    model,
    loss_dict: dict[str, torch.Tensor],
    backward_loss: torch.Tensor,
    cfg: dict[str, Any],
    *,
    accumulation_steps: int,
    scaler,
    amp: bool,
) -> dict[str, torch.Tensor] | None:
    """Project the conflicting complete-V18 gradient on shared image tensors.

    The normal backward is retained for every private V18 parameter and every
    proposal objective. On configured shared prefixes, the raw V18 component is
    subtracted and its projection orthogonal to a conflicting proposal-
    protection gradient is inserted. BF16 needs no GradScaler; FP16 scaling is
    rejected here rather than silently mixing scaled and unscaled gradients.
    """

    contract = cfg.get("training", {}).get(
        "v18_gradient_conflict_projection", {}
    )
    if not bool(contract.get("enabled", False)):
        return None
    if scaler is not None and amp:
        raise ValueError(
            "V18 conflict projection requires BF16 or FP32 without GradScaler"
        )
    set_loss = loss_dict.get("loss_v18_set_backward")
    protection_loss = loss_dict.get("loss_v18_proposal_protection")
    if not isinstance(set_loss, torch.Tensor) or not isinstance(
        protection_loss, torch.Tensor
    ):
        raise ValueError("V18 projection requires explicit criterion components")
    prefixes = tuple(str(value) for value in contract.get("shared_prefixes", ()))
    excludes = tuple(str(value) for value in contract.get("exclude_prefixes", ()))
    if not prefixes:
        raise ValueError("V18 projection requires shared parameter prefixes")
    root_model = getattr(model, "_orig_mod", model)
    named_parameters = [
        (name, parameter)
        for name, parameter in root_model.named_parameters()
        if parameter.requires_grad
        and any(name.startswith(prefix) for prefix in prefixes)
        and not any(name.startswith(prefix) for prefix in excludes)
    ]
    if not named_parameters:
        raise ValueError("V18 projection matched no trainable shared parameters")
    parameters = [parameter for _name, parameter in named_parameters]
    scale = 1.0 / float(accumulation_steps)
    backward_mode = str(
        contract.get("backward_mode", "reference_three_pass")
    ).strip().lower()
    if backward_mode not in {
        "reference_three_pass",
        "partitioned_two_pass",
    }:
        raise ValueError(
            "V18 conflict projection backward_mode must be "
            "reference_three_pass or partitioned_two_pass"
        )

    if backward_mode == "reference_three_pass":
        set_gradients = torch.autograd.grad(
            set_loss * scale,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        protection_gradients = torch.autograd.grad(
            protection_loss * scale,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        backward_loss.backward()
    else:
        # The V18 criterion exposes a complete two-way partition of the
        # training objective: the mature proposal objective and the complete
        # V18 objective. Back-propagating those two components once each is
        # algebraically identical to the historical implementation above,
        # which evaluated two VJPs and then traversed both graphs a second
        # time through ``backward_loss.backward()``. Leaf hooks capture only
        # the current micro-batch contribution, so gradient accumulation is
        # preserved without cloning pre-existing ``parameter.grad`` values.
        if not bool(contract.get("complete_loss_partition", False)):
            raise ValueError(
                "partitioned_two_pass requires complete_loss_partition=true"
            )

        protection_by_parameter: dict[int, torch.Tensor] = {}
        protection_handles = []

        def capture_protection(parameter_id: int):
            def hook(
                gradient: torch.Tensor | None,
            ) -> torch.Tensor | None:
                # AOTAutograd may invoke a leaf hook with ``None`` for a
                # parameter that is unused by this partition. Eager autograd
                # normally skips the hook instead. Treat both forms as the
                # same absent gradient rather than forcing a materialized
                # zero tensor.
                if gradient is None:
                    return None
                protection_by_parameter[parameter_id] = gradient.detach().clone()
                return gradient

            return hook

        for parameter in parameters:
            protection_handles.append(
                parameter.register_hook(capture_protection(id(parameter)))
            )
        try:
            (protection_loss * scale).backward(retain_graph=True)
        finally:
            for handle in protection_handles:
                handle.remove()

        set_by_parameter: dict[int, torch.Tensor] = {}
        set_handles = []

        def capture_set(parameter_id: int):
            def hook(
                gradient: torch.Tensor | None,
            ) -> torch.Tensor | None:
                if gradient is None:
                    return None
                set_by_parameter[parameter_id] = gradient.detach().clone()
                return gradient

            return hook

        for parameter in parameters:
            set_handles.append(parameter.register_hook(capture_set(id(parameter))))
        try:
            (set_loss * scale).backward()
        finally:
            for handle in set_handles:
                handle.remove()

        set_gradients = tuple(
            set_by_parameter.get(id(parameter)) for parameter in parameters
        )
        protection_gradients = tuple(
            protection_by_parameter.get(id(parameter)) for parameter in parameters
        )

    device = backward_loss.device
    dot = torch.zeros((), device=device, dtype=torch.float32)
    set_norm_sq = torch.zeros_like(dot)
    protection_norm_sq = torch.zeros_like(dot)
    for set_gradient, protection_gradient in zip(
        set_gradients, protection_gradients
    ):
        if set_gradient is not None:
            set_norm_sq = set_norm_sq + set_gradient.detach().float().square().sum()
        if protection_gradient is not None:
            protection_norm_sq = protection_norm_sq + protection_gradient.detach().float().square().sum()
        if set_gradient is not None and protection_gradient is not None:
            dot = dot + (
                set_gradient.detach().float()
                * protection_gradient.detach().float()
            ).sum()
    coefficient = torch.where(
        (dot < 0.0) & (protection_norm_sq > 0.0),
        dot / protection_norm_sq.clamp_min(1.0e-20),
        torch.zeros_like(dot),
    )
    for parameter, set_gradient, protection_gradient in zip(
        parameters, set_gradients, protection_gradients
    ):
        if set_gradient is None or protection_gradient is None:
            continue
        correction = -coefficient.to(protection_gradient) * protection_gradient
        if parameter.grad is None:
            parameter.grad = correction.to(parameter)
        else:
            parameter.grad.add_(correction.to(parameter.grad))
    cosine = dot / (
        set_norm_sq.sqrt() * protection_norm_sq.sqrt()
    ).clamp_min(1.0e-20)
    return {
        "v18_shared_set_grad_norm": set_norm_sq.sqrt().detach(),
        "v18_shared_protection_grad_norm": protection_norm_sq.sqrt().detach(),
        "v18_shared_gradient_cosine": cosine.detach(),
        "v18_shared_projection_active": (coefficient != 0.0).float().detach(),
    }


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


def _apply_pointer_teacher_forcing(
    model,
    outputs,
    targets,
    cfg,
    *,
    iteration: int = 0,
    teacher_visit: int = 0,
) -> None:
    if not isinstance(outputs, dict):
        return
    root_model = getattr(model, "_orig_mod", model)
    structured = getattr(root_model, "structured_query_head", None)
    selector = getattr(structured, "set_selection_head", None)
    if selector is None or getattr(
        selector,
        "candidate_interaction",
        "",
    ) != "sequential_pointer":
        return
    loss_cfg = cfg.get("loss", {})
    selection_cfg = (
        cfg.get("model", {})
        .get("structured_query", {})
        .get("set_selection", {})
    )
    pointer_teacher_mode = getattr(selector, "pointer_teacher_mode", "")
    if pointer_teacher_mode in {
        "cluster_soft_randomized",
        "cluster_soft_remaining_mixture",
    }:
        teacher = build_pointer_cluster_soft_targets(
            outputs,
            targets,
            max_selections=int(selector.pointer_max_selections),
            input_h=int(
                loss_cfg.get("input_h", cfg.get("model", {}).get("input_h", 288))
            ),
            line_width=float(loss_cfg.get("set_selection_line_width", 30.0)),
            min_valid_rows=int(
                loss_cfg.get("set_selection_min_valid_rows", 5)
            ),
            representable_min=float(
                selection_cfg.get("pointer_cluster_representable_min", 0.20)
            ),
            support_quality_delta=float(
                selection_cfg.get("pointer_cluster_quality_delta", 0.10)
            ),
            temperature=float(
                selection_cfg.get("pointer_cluster_temperature", 0.03)
            ),
            base_seed=int(cfg.get("training", {}).get("seed", 0)),
            iteration=int(iteration),
            visit=int(teacher_visit),
            target_mode=(
                "remaining_cluster_mixture"
                if pointer_teacher_mode == "cluster_soft_remaining_mixture"
                else "sampled_cluster"
            ),
        )
        selector.reroll_pointer_with_cluster_teacher(outputs, teacher)
        return
    teacher = build_pointer_sequence_targets(
        outputs,
        targets,
        max_selections=int(selector.pointer_max_selections),
        input_h=int(loss_cfg.get("input_h", cfg.get("model", {}).get("input_h", 288))),
        line_width=float(loss_cfg.get("set_selection_line_width", 30.0)),
        min_valid_rows=int(loss_cfg.get("set_selection_min_valid_rows", 5)),
    )
    selector.reroll_pointer_with_teacher(outputs, teacher)


def forward_with_matches(
    model,
    images,
    targets,
    matcher,
    cfg,
    iteration,
    pointer_teacher_visit: int = 0,
):
    if hasattr(matcher, "set_iteration"):
        matcher.set_iteration(iteration)
    name = cfg.get("model", {}).get("name", "DynLaneSeqS0")
    if name in {"DynLaneSeqS2", "DynLaneSeqS3"}:
        probe = model(images, sampler_alpha=0.0)
        matches = matcher(probe["coarse"], targets)
        outputs = model(images, targets=targets, matches=matches, sampler_alpha=_sampler_alpha(cfg, iteration))
        _apply_pointer_teacher_forcing(
            model,
            outputs,
            targets,
            cfg,
            iteration=iteration,
            teacher_visit=pointer_teacher_visit,
        )
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
        _apply_pointer_teacher_forcing(
            model,
            outputs,
            targets,
            cfg,
            iteration=iteration,
            teacher_visit=pointer_teacher_visit,
        )
        return outputs, matches
    outputs = model(images)
    aux_outputs = outputs.get("aux_outputs") if isinstance(outputs, dict) else None
    training_auxiliary = (
        outputs.get("_training_auxiliary_outputs")
        if isinstance(outputs, dict)
        else None
    )
    training_auxiliary_layers = (
        outputs.get("_training_auxiliary_aux_outputs")
        if isinstance(outputs, dict)
        else None
    )
    training_auxiliary_group_sizes = (
        outputs.get("_training_auxiliary_group_sizes")
        if isinstance(outputs, dict)
        else None
    )
    reuse_final_assignment = bool(
        cfg.get("matcher", {}).get(
            "reuse_final_assignment_for_intermediate",
            False,
        )
    )
    if isinstance(training_auxiliary, dict) and hasattr(matcher, "match_many"):
        main_layers = list(aux_outputs) if isinstance(aux_outputs, (list, tuple)) else []
        auxiliary_layers = (
            list(training_auxiliary_layers)
            if isinstance(training_auxiliary_layers, (list, tuple))
            else []
        )
        if not isinstance(training_auxiliary_group_sizes, (list, tuple)):
            raise ValueError(
                "training auxiliary outputs require explicit group sizes"
            )
        group_sizes = tuple(int(size) for size in training_auxiliary_group_sizes)
        if reuse_final_assignment:
            # The final geometry defines one stable ownership identity.  Deep
            # supervision then improves the same candidates at earlier blocks
            # instead of asking every block to solve a different Hungarian
            # permutation.  Auxiliary groups keep their own grouped identity,
            # but reuse it across their intermediate outputs as well.
            sequence = (outputs, training_auxiliary)
            assignment_specs = [
                (str(matcher.cfg.assignment), None),
                ("grouped_one_to_many", group_sizes),
            ]
        else:
            sequence = (
                outputs,
                *main_layers,
                training_auxiliary,
                *auxiliary_layers,
            )
            assignment_specs = [
                (str(matcher.cfg.assignment), None)
                for _ in range(1 + len(main_layers))
            ] + [
                ("grouped_one_to_many", group_sizes)
                for _ in range(1 + len(auxiliary_layers))
            ]
        all_matches = matcher.match_many(
            sequence,
            targets,
            assignment_specs=assignment_specs,
        )
        matches = all_matches[0]
        if reuse_final_assignment:
            auxiliary_matches = all_matches[1]
            outputs["_aux_matches"] = [matches for _ in main_layers]
            outputs["_training_auxiliary_matches"] = auxiliary_matches
            outputs["_training_auxiliary_aux_matches"] = [
                auxiliary_matches for _ in auxiliary_layers
            ]
        else:
            main_count = 1 + len(main_layers)
            outputs["_aux_matches"] = all_matches[1:main_count]
            outputs["_training_auxiliary_matches"] = all_matches[main_count]
            outputs["_training_auxiliary_aux_matches"] = all_matches[
                main_count + 1 :
            ]
        _apply_pointer_teacher_forcing(
            model,
            outputs,
            targets,
            cfg,
            iteration=iteration,
            teacher_visit=pointer_teacher_visit,
        )
        return outputs, matches
    if (
        isinstance(aux_outputs, (list, tuple))
        and aux_outputs
        and hasattr(matcher, "match_many")
    ):
        if reuse_final_assignment:
            matches = matcher(outputs, targets)
            outputs["_aux_matches"] = [matches for _ in aux_outputs]
        else:
            all_matches = matcher.match_many((outputs, *aux_outputs), targets)
            matches = all_matches[0]
            # Private training-only transport: the criterion consumes these exact
            # per-layer assignments instead of repeating GPU/CPU matcher traffic.
            outputs["_aux_matches"] = all_matches[1:]
    else:
        matches = matcher(outputs, targets)
    _apply_pointer_teacher_forcing(
        model,
        outputs,
        targets,
        cfg,
        iteration=iteration,
        teacher_visit=pointer_teacher_visit,
    )
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


def format_loss_diagnostics(loss_dict: dict[str, Any], metas: list[dict[str, Any]] | None = None) -> str:
    """Format scalar losses only when a non-finite total is encountered.

    Copying scalar values to CPU is deliberately confined to the failure path,
    so normal training has no synchronization or throughput penalty.
    """
    terms = []
    for name, value in loss_dict.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            continue
        scalar = float(value.detach().float().cpu().item())
        marker = "*" if not math.isfinite(scalar) else ""
        terms.append(f"{name}={scalar:.7g}{marker}")
    sample_paths = []
    for meta in metas or []:
        if not isinstance(meta, dict):
            continue
        path = meta.get("image_path")
        if path:
            sample_paths.append(str(path))
    parts = ["losses[" + ", ".join(terms) + "]"]
    if sample_paths:
        parts.append("samples[" + ", ".join(sample_paths) + "]")
    return " | ".join(parts)


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
    train_cfg = cfg.get("training", {})
    trainable_modules = tuple(train_cfg.get("trainable_module_prefixes", ()))
    if bool(train_cfg.get("frozen_detector_eval", False)):
        if not trainable_modules:
            raise ValueError(
                "training.frozen_detector_eval requires "
                "training.trainable_module_prefixes"
            )
        set_frozen_detector_eval(model, trainable_modules)
    amp = bool(train_cfg.get("amp", False))
    amp_dtype_name = str(train_cfg.get("amp_dtype", "")).lower()
    amp_dtype = None
    if device.type == "cuda":
        if amp_dtype_name in {"bf16", "bfloat16"}:
            amp_dtype = torch.bfloat16
        elif amp_dtype_name in {"fp16", "float16", "half"}:
            amp_dtype = torch.float16
    channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")
    clip_norm = float(train_cfg.get("clip_grad_norm", 1.0))
    clip_mode = str(train_cfg.get("clip_grad_norm_mode", "global"))
    check_finite_grad = bool(train_cfg.get("check_finite_grad", True))
    resume_safe_data = bool(
        cfg.get("dataloader", {}).get("resume_safe", False)
    )
    accumulation_steps = max(int(train_cfg.get("gradient_accumulation_steps", 1)), 1)
    log_interval = int(train_cfg.get("log_interval", 10))
    iteration = start_iter
    max_iters = max_iters or int(cfg.get("training", {}).get("max_iters", len(dataloader)))
    end_iter = start_iter + max_iters
    wall_start = time.perf_counter()
    last_log_time = wall_start
    last_log_iteration = start_iter
    last_log_processed_images = 0
    loader_len = max(len(dataloader), 1) if hasattr(dataloader, "__len__") else 1
    processed_images = 0
    micro_in_step = 0
    optimizer.zero_grad(set_to_none=True)
    while iteration < end_iter:
        for images, targets, metas in dataloader:
            if iteration >= end_iter:
                break
            processed_images += int(images.shape[0])
            if channels_last:
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                images = images.to(device, non_blocking=True)
            targets = nested_to_device(targets, device)
            autocast_kwargs = {"device_type": device.type, "enabled": amp}
            if amp_dtype is not None:
                autocast_kwargs["dtype"] = amp_dtype
            with torch.autocast(**autocast_kwargs):
                outputs, matches = forward_with_matches(
                    model,
                    images,
                    targets,
                    matcher,
                    cfg,
                    iteration,
                    pointer_teacher_visit=micro_in_step,
                )
                if hasattr(criterion, "set_iteration"):
                    criterion.set_iteration(iteration)
                loss_dict = criterion(outputs, targets, matches)
                loss = loss_dict["loss_total"]
            if not torch.isfinite(loss):
                if resume_safe_data:
                    raise FloatingPointError(
                        "resume-safe data addressing requires exactly "
                        f"{accumulation_steps} micro-batches per logical "
                        "iteration; refusing to skip a non-finite loss"
                    )
                print(f"iter {iteration + 1:07d} | non-finite loss; skipping optimizer step")
                print(f"iter {iteration + 1:07d} | {format_loss_diagnostics(loss_dict, metas)}")
                optimizer.zero_grad(set_to_none=True)
                micro_in_step = 0
                iteration += 1
                continue
            backward_loss = loss / float(accumulation_steps)
            v18_projection_stats = _v18_conflict_safe_backward(
                model,
                loss_dict,
                backward_loss,
                cfg,
                accumulation_steps=accumulation_steps,
                scaler=scaler,
                amp=amp,
            )
            if v18_projection_stats is None:
                if scaler is not None and amp:
                    scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
            else:
                loss_dict.update(v18_projection_stats)
            micro_in_step += 1
            if micro_in_step < accumulation_steps:
                continue

            if scaler is not None and amp:
                scaler.unscale_(optimizer)
                grad_norm = clip_optimizer_gradients(model, optimizer, clip_norm, clip_mode)
                if check_finite_grad and not bool(torch.isfinite(grad_norm).detach().cpu()):
                    if resume_safe_data:
                        raise FloatingPointError(
                            "resume-safe data addressing refuses to skip a "
                            "non-finite gradient step"
                        )
                    print(f"iter {iteration + 1:07d} | non-finite grad norm; skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    micro_in_step = 0
                    scaler.update()
                    iteration += 1
                    continue
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            else:
                grad_norm = clip_optimizer_gradients(model, optimizer, clip_norm, clip_mode)
                if check_finite_grad and not bool(torch.isfinite(grad_norm).detach().cpu()):
                    if resume_safe_data:
                        raise FloatingPointError(
                            "resume-safe data addressing refuses to skip a "
                            "non-finite gradient step"
                        )
                    print(f"iter {iteration + 1:07d} | non-finite grad norm; skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    micro_in_step = 0
                    iteration += 1
                    continue
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro_in_step = 0
            if logger is not None:
                stats = {k: v for k, v in loss_dict.items()}
                stats.update(match_stats(outputs, matches))
                stats.update(output_debug_stats(outputs))
                if hasattr(matcher, "effective_lambda_obj"):
                    stats["matcher_lambda_obj"] = (
                        matcher.effective_lambda_obj()
                    )
                stats["grad_norm"] = grad_norm
                stats["lr_model"] = optimizer.param_groups[-1]["lr"]
                logger.update(**stats)
                if (iteration + 1) % log_interval == 0:
                    done = max(iteration + 1 - start_iter, 1)
                    total = max(max_iters, 1)
                    now = time.perf_counter()
                    elapsed = now - wall_start
                    # Report the most recent log window rather than averaging
                    # from process start.  A cold torch.compile step can take
                    # minutes; including it forever made a healthy steady-state
                    # run look artificially slow for thousands of iterations.
                    window_elapsed = max(now - last_log_time, 1e-6)
                    window_iterations = max(iteration + 1 - last_log_iteration, 1)
                    window_images = max(processed_images - last_log_processed_images, 0)
                    sec_per_iter = window_elapsed / float(window_iterations)
                    img_per_sec = float(window_images) / window_elapsed
                    eta = sec_per_iter * max(total - done, 0)
                    pct = 100.0 * done / total
                    epoch = float((iteration + 1) * accumulation_steps) / float(loader_len)
                    prefix = (
                        f"iter {iteration + 1:07d}/{end_iter:07d} "
                        f"({pct:5.1f}%) | epoch {epoch:.2f} | "
                        f"{sec_per_iter:.3f}s/it | {img_per_sec:.1f} img/s | "
                        f"elapsed {_format_duration(elapsed)} | "
                        f"eta {_format_duration(eta)} | "
                    )
                    print(logger.format_and_reset(prefix=prefix))
                    last_log_time = now
                    last_log_iteration = iteration + 1
                    last_log_processed_images = processed_images
            if visualizer is not None:
                visualizer(images, targets, metas, outputs, iteration + 1)
            iteration += 1
            if checkpoint_saver is not None:
                checkpoint_saver(iteration)
    return iteration
