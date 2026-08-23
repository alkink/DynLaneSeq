from __future__ import annotations

import argparse
import math
import random
import warnings
from pathlib import Path

import numpy as np
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import (
    _torch_load,
    load_checkpoint,
    load_compatible_model_weights,
    remap_optimizer_state_by_parameter,
    restore_checkpoint_rng_state,
    save_checkpoint,
)
from dynlaneseq_eg.engine.frozen_training import freeze_except_parameter_prefixes
from dynlaneseq_eg.engine.logger import SmoothedLogger
from dynlaneseq_eg.engine.train_one_epoch import train_one_epoch
from dynlaneseq_eg.engine.visualizer import save_prediction_visuals
from dynlaneseq_eg.factory import build_criterion, build_dataloader, build_matcher, build_model, build_optimizer, build_scheduler


def seed_everything(seed: int) -> None:
    """Seed model initialization and the RNGs used by data augmentation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_init_start_iteration(init_from: str, init_iteration: int) -> int:
    """Resolve a logical iteration without restoring optimizer state."""
    if init_iteration >= 0 and not init_from:
        raise ValueError("--init-iteration requires --init-from")
    return int(init_iteration) if init_iteration >= 0 else 0


def parse_optimizer_group_lr_overrides(specs: list[str]) -> dict[str, float]:
    """Parse repeatable ``GROUP=LR`` optimizer overrides."""
    overrides: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"invalid optimizer group LR override {spec!r}; expected GROUP=LR"
            )
        name, raw_lr = (part.strip() for part in spec.split("=", 1))
        if not name:
            raise ValueError("optimizer group LR override has an empty group name")
        if name in overrides:
            raise ValueError(f"duplicate optimizer group LR override: {name}")
        try:
            lr = float(raw_lr)
        except ValueError as exc:
            raise ValueError(
                f"invalid LR {raw_lr!r} for optimizer group {name!r}"
            ) from exc
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError(
                f"optimizer group LR must be finite and non-negative: {name}={lr}"
            )
        overrides[name] = lr
    return overrides


def apply_optimizer_group_lr_overrides(
    optimizer: torch.optim.Optimizer,
    overrides: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Change selected group LRs after restoring optimizer state.

    This deliberately leaves parameters and AdamW moments untouched.  Group
    names must be unique so an experimental intervention cannot silently
    affect an unintended parameter set.
    """
    groups_by_name: dict[str, dict] = {}
    duplicate_names: set[str] = set()
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name in groups_by_name:
            duplicate_names.add(name)
        groups_by_name[name] = group
    if duplicate_names:
        raise ValueError(
            "optimizer contains duplicate group names: "
            + ", ".join(sorted(repr(name) for name in duplicate_names))
        )
    missing = sorted(set(overrides) - set(groups_by_name))
    if missing:
        raise ValueError(
            "optimizer LR override refers to missing groups: "
            + ", ".join(missing)
            + "; available groups: "
            + ", ".join(sorted(name for name in groups_by_name if name))
        )

    changes: dict[str, dict[str, float]] = {}
    for name, lr in overrides.items():
        group = groups_by_name[name]
        previous = float(group["lr"])
        group["lr"] = float(lr)
        # A resumed constant-schedule experiment never reads ``initial_lr``,
        # but keeping it consistent makes the checkpoint audit unambiguous.
        if "initial_lr" in group:
            group["initial_lr"] = float(lr)
        changes[name] = {"before": previous, "after": float(lr)}
    return changes


def align_scheduler_to_iteration(
    scheduler,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> dict[str, object]:
    """Put a newly built scheduler at a resumed optimizer-step phase.

    Optimizer-group remapping intentionally keeps the *new* group topology, so
    the old scheduler state cannot be loaded when the number of groups changes.
    Stepping the newly built scheduler to the checkpoint iteration preserves
    its new base learning rates while matching the original cosine/multistep
    phase.  Without this alignment, a 25k intervention silently restarts at
    warmup step zero and is not a controlled continuation.
    """

    iteration = int(iteration)
    if iteration < 0:
        raise ValueError("scheduler alignment iteration must be non-negative")
    if scheduler is None:
        return {
            "enabled": False,
            "iteration": iteration,
            "group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The epoch parameter in `scheduler.step\(\)` was not necessary",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`",
        )
        scheduler.step(iteration)
    if int(scheduler.last_epoch) != iteration:
        raise RuntimeError(
            "scheduler phase alignment failed: "
            f"last_epoch={scheduler.last_epoch}, expected={iteration}"
        )
    return {
        "enabled": True,
        "iteration": iteration,
        "last_epoch": int(scheduler.last_epoch),
        "step_count": int(getattr(scheduler, "_step_count", 0)),
        "base_lrs": [float(value) for value in scheduler.base_lrs],
        "group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-from", default="", help="Initialize compatible model weights only; optimizer and iteration stay fresh.")
    parser.add_argument(
        "--checkpoint-base",
        default="",
        help=(
            "Base checkpoint referenced by compact model-delta checkpoints. "
            "Requires training.checkpoint_model_prefixes."
        ),
    )
    parser.add_argument(
        "--init-iteration",
        type=int,
        default=-1,
        help=(
            "Logical starting iteration used with --init-from. Model weights are "
            "loaded without optimizer/scheduler state, while curricula, logs, and "
            "checkpoint names continue from this iteration."
        ),
    )
    parser.add_argument("--max-iters", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Override training.seed for paired or multi-seed experiments.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=-1,
        help=(
            "Override training.checkpoint_interval. Zero disables periodic "
            "checkpoints; the final checkpoint is still written."
        ),
    )
    parser.add_argument("--output-dir", default="", help="Override cfg.output_dir.")
    parser.add_argument("--dataset-root", default="", help="Override cfg.dataset.root.")
    parser.add_argument(
        "--train-list",
        default="",
        help="Override dataset.lists.train (used by fixed-set memorization gates).",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Override training.batch_size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="Override dataloader.num_workers; negative keeps the config value.",
    )
    parser.add_argument(
        "--seg-aux-amp-dtype",
        choices=("inherit", "float16", "bfloat16"),
        default="",
        help="Override model.seg_aux.amp_dtype without changing the main model AMP dtype.",
    )
    parser.add_argument(
        "--grad-accum",
        "--gradient-accumulation-steps",
        dest="grad_accum",
        type=int,
        default=0,
        help="Override training.gradient_accumulation_steps.",
    )
    parser.add_argument(
        "--compile-model",
        choices=("config", "true", "false"),
        default="config",
        help=(
            "Override training.compile_model. 'config' preserves the YAML; "
            "'false' is useful on local CUDA/Inductor combinations that cannot "
            "compile the model while keeping the mathematical training contract unchanged."
        ),
    )
    parser.add_argument(
        "--resume-safe-data",
        choices=("config", "true", "false"),
        default="config",
        help=(
            "Override dataloader.resume_safe. The resume-safe path addresses "
            "shuffle and augmentation by global optimizer iteration so a "
            "process restart cannot replay the data stream."
        ),
    )
    parser.add_argument(
        "--resume-group-lr",
        action="append",
        default=[],
        metavar="GROUP=LR",
        help=(
            "After restoring --resume, change only the named optimizer group's "
            "current LR while preserving parameters and optimizer moments. "
            "Repeat for multiple groups; supported only with no/constant scheduler."
        ),
    )
    parser.add_argument(
        "--resume-remap-optimizer-groups",
        action="store_true",
        help=(
            "Restore model weights and per-parameter optimizer moments while "
            "using the parameter-group topology and scheduler from the new "
            "config. The checkpoint must contain its original expanded config."
        ),
    )
    args = parser.parse_args()
    resume_group_lr_overrides = parse_optimizer_group_lr_overrides(
        args.resume_group_lr
    )
    if resume_group_lr_overrides and not args.resume:
        raise ValueError("--resume-group-lr requires --resume")
    if args.resume_remap_optimizer_groups and not args.resume:
        raise ValueError("--resume-remap-optimizer-groups requires --resume")
    if args.resume_remap_optimizer_groups and resume_group_lr_overrides:
        raise ValueError(
            "--resume-remap-optimizer-groups cannot be combined with "
            "--resume-group-lr"
        )
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    if args.train_list:
        cfg.setdefault("dataset", {}).setdefault("lists", {})["train"] = str(
            Path(args.train_list).expanduser().resolve()
        )
    if args.batch_size > 0:
        cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    if args.num_workers >= 0:
        cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if args.seg_aux_amp_dtype:
        cfg.setdefault("model", {}).setdefault("seg_aux", {})["amp_dtype"] = args.seg_aux_amp_dtype
    if args.grad_accum > 0:
        cfg.setdefault("training", {})["gradient_accumulation_steps"] = int(args.grad_accum)
    if args.compile_model != "config":
        cfg.setdefault("training", {})["compile_model"] = (
            args.compile_model == "true"
        )
    if args.resume_safe_data != "config":
        cfg.setdefault("dataloader", {})["resume_safe"] = (
            args.resume_safe_data == "true"
        )
    if args.seed >= 0:
        cfg.setdefault("training", {})["seed"] = int(args.seed)
    if args.checkpoint_interval >= 0:
        cfg.setdefault("training", {})["checkpoint_interval"] = int(
            args.checkpoint_interval
        )
    train_cfg = cfg.get("training", {})
    checkpoint_model_prefixes = tuple(
        train_cfg.get("checkpoint_model_prefixes", ())
    )
    checkpoint_base = str(
        args.checkpoint_base or train_cfg.get("checkpoint_base", "")
    ).strip()
    checkpoint_include_optimizer = bool(
        train_cfg.get("checkpoint_include_optimizer", True)
    )
    save_last_alias = bool(train_cfg.get("save_last_alias", True))
    if checkpoint_model_prefixes and not checkpoint_base:
        raise ValueError(
            "training.checkpoint_model_prefixes requires --checkpoint-base "
            "or training.checkpoint_base"
        )
    if checkpoint_base and not checkpoint_model_prefixes:
        raise ValueError(
            "--checkpoint-base requires training.checkpoint_model_prefixes"
        )
    if checkpoint_base and args.init_from:
        if Path(checkpoint_base).expanduser().resolve() != Path(
            args.init_from
        ).expanduser().resolve():
            raise ValueError(
                "compact checkpoint base must be the exact --init-from model"
            )
    amp_dtype_name = str(train_cfg.get("amp_dtype", "")).lower()
    amp_dtype_is_bf16 = amp_dtype_name in {"bf16", "bfloat16"}
    seed_value = train_cfg.get("seed")
    seed = int(seed_value) if seed_value is not None else None
    if seed is not None:
        seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", False))
        if bool(train_cfg.get("tf32", False)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    model = build_model(cfg).to(device)
    trainable_prefixes = tuple(
        train_cfg.get("trainable_parameter_prefixes", ())
    )
    frozen_training_stats = None
    if trainable_prefixes:
        frozen_training_stats = freeze_except_parameter_prefixes(
            model,
            trainable_prefixes,
        )
    train_model = model
    channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        train_model = model
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    use_grad_scaler = bool(
        cfg.get("training", {}).get("amp", False)
        and device.type == "cuda"
        and not amp_dtype_is_bf16
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler) if use_grad_scaler else None
    resume_safe_data = bool(cfg.get("dataloader", {}).get("resume_safe", False))
    # The legacy DataLoader must retain its exact construction timing for the
    # paired control. The resume-safe loader is deliberately deferred until
    # after checkpoint restoration because its first batch is a pure function
    # of the restored logical iteration.
    loader = (
        None
        if resume_safe_data
        else build_dataloader(cfg, split="train", training=True)
    )
    out_dir = Path(cfg.get("output_dir", "outputs/train"))
    vis_interval = int(cfg.get("training", {}).get("vis_interval", 100))
    configured_max_iters = int(cfg.get("training", {}).get("max_iters", 0))
    if args.max_iters:
        planned_iters = int(args.max_iters)
    elif configured_max_iters > 0:
        planned_iters = configured_max_iters
    elif loader is not None:
        planned_iters = len(loader)
    else:
        raise ValueError(
            "resume-safe training requires --max-iters or training.max_iters"
        )
    scheduler = build_scheduler(cfg, optimizer, total_iters=planned_iters)
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    start_iter = model_init_start_iteration(args.init_from, args.init_iteration)
    if args.init_from:
        stats = load_compatible_model_weights(args.init_from, model)
        print(f"initialized compatible weights from {args.init_from}: {stats}")
    if args.resume:
        if args.resume_remap_optimizer_groups:
            # PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by
            # default.  Full training checkpoints intentionally contain RNG
            # and optimizer metadata, so use the project's compatibility
            # loader just like every other checkpoint path.
            payload = _torch_load(args.resume)
            source_cfg = payload.get("cfg")
            if not isinstance(source_cfg, dict) or not source_cfg:
                raise ValueError(
                    "optimizer-group remap requires the checkpoint's original cfg"
                )
            if "optimizer" not in payload:
                raise ValueError(
                    "optimizer-group remap requires optimizer state in checkpoint"
                )
            model.load_state_dict(payload["model"], strict=False)
            source_model_state = payload["model"]
            newly_added_parameters = []
            for name, parameter in model.named_parameters():
                source_tensor = source_model_state.get(name)
                if (
                    source_tensor is None
                    or tuple(source_tensor.shape) != tuple(parameter.shape)
                ):
                    newly_added_parameters.append((name, parameter))
            original_requires_grad = {
                parameter: bool(parameter.requires_grad)
                for _, parameter in newly_added_parameters
            }
            try:
                for _, parameter in newly_added_parameters:
                    parameter.requires_grad_(False)
                source_optimizer = build_optimizer(source_cfg, model)
            finally:
                for _, parameter in newly_added_parameters:
                    parameter.requires_grad_(
                        original_requires_grad[parameter]
                    )
            source_optimizer.load_state_dict(payload["optimizer"])
            remap_stats = remap_optimizer_state_by_parameter(
                source_optimizer,
                optimizer,
                allow_target_superset=bool(newly_added_parameters),
            )
            if scaler is not None and "scaler" in payload:
                scaler.load_state_dict(payload["scaler"])
            start_iter = int(payload.get("iteration", 0))
            scheduler_alignment = align_scheduler_to_iteration(
                scheduler,
                optimizer,
                start_iter,
            )
            # Optimizer-group remapping is still a true resume.  Restore the
            # checkpoint RNG *after* constructing the larger treatment model
            # and its new optimizer groups so added-module initialization
            # cannot shift dropout or any other model-side stochastic stream.
            # Resume-safe data augmentation is independently addressed by
            # global iteration, but exact paired experiments require both
            # sources of randomness to be aligned.
            rng_state_restored = restore_checkpoint_rng_state(payload)
            del source_optimizer
            del payload
            print(
                {
                    "resume_optimizer_group_remap": remap_stats,
                    "new_optimizer_parameters": [
                        name for name, _ in newly_added_parameters
                    ],
                    "scheduler_state_restored": False,
                    "scheduler_phase_alignment": scheduler_alignment,
                    "rng_state_restored_after_optimizer_remap": (
                        rng_state_restored
                    ),
                }
            )
        else:
            start_iter = load_checkpoint(
                args.resume,
                model,
                optimizer,
                scaler,
                strict=False,
                scheduler=scheduler,
                restore_rng_state=True,
            )
        if resume_group_lr_overrides:
            if scheduler is not None:
                raise ValueError(
                    "--resume-group-lr is supported only with a no/constant "
                    "scheduler so a scheduler cannot silently overwrite it"
                )
            changes = apply_optimizer_group_lr_overrides(
                optimizer,
                resume_group_lr_overrides,
            )
            print({"resume_optimizer_lr_overrides": changes})
    if loader is None:
        loader = build_dataloader(
            cfg,
            split="train",
            training=True,
            start_iteration=start_iter,
        )
    if bool(train_cfg.get("compile_model", False)):
        compile_kwargs = {}
        if train_cfg.get("compile_backend") is not None:
            compile_kwargs["backend"] = str(train_cfg.get("compile_backend"))
        if train_cfg.get("compile_mode") is not None:
            compile_kwargs["mode"] = str(train_cfg.get("compile_mode"))
        train_model = torch.compile(model, **compile_kwargs)
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    accumulation_steps = max(int(train_cfg.get("gradient_accumulation_steps", 1)), 1)
    approx_epochs = planned_iters * accumulation_steps / max(len(loader), 1)
    batch_sampler = getattr(loader, "batch_sampler", None)
    data_stream_contract = (
        batch_sampler.contract()
        if hasattr(batch_sampler, "contract")
        else {
            "enabled": False,
            "legacy_generator_reseeded_on_process_start": True,
        }
    )
    row_reference_cfg = (
        cfg.get("model", {})
        .get("structured_query", {})
        .get("row_reference", {})
    )
    print(
        {
            "model": cfg.get("model", {}).get("name", "DynLaneSeq"),
            "backbone": getattr(getattr(model, "encoder", None), "backbone_name", None),
            "output_dir": str(out_dir),
            "device": str(device),
            "train_images": len(loader.dataset),
            "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "seed": seed,
            "data_stream_contract": data_stream_contract,
            "iters": planned_iters,
            "start_iter": start_iter,
            "approx_epochs_this_run": round(approx_epochs, 2),
            "amp": bool(cfg.get("training", {}).get("amp", False) and device.type == "cuda"),
            "amp_dtype": amp_dtype_name or "default",
            "seg_aux_amp_dtype": str(cfg.get("model", {}).get("seg_aux", {}).get("amp_dtype", "inherit")),
            "channels_last": channels_last,
            "compile_model": bool(train_cfg.get("compile_model", False)),
            "row_reference_sampling_backend": (
                str(row_reference_cfg.get("sampling_backend", "grid_sample"))
                if bool(row_reference_cfg.get("enabled", False))
                else None
            ),
            "row_reference_projection_backend": (
                str(row_reference_cfg.get("projection_backend", "separate"))
                if bool(row_reference_cfg.get("enabled", False))
                else None
            ),
            "row_reference_attention_backend": (
                str(row_reference_cfg.get("attention_backend", "materialized"))
                if bool(row_reference_cfg.get("enabled", False))
                else None
            ),
            "clip_grad_norm": float(train_cfg.get("clip_grad_norm", 1.0)),
            "clip_grad_norm_mode": str(train_cfg.get("clip_grad_norm_mode", "global")),
            "log_interval": int(cfg.get("training", {}).get("log_interval", 10)),
            "scheduler": cfg.get("scheduler", {"name": "none"}),
            "optimizer_group_lrs": {
                str(group.get("name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            },
            "frozen_training": frozen_training_stats,
            "checkpoint_policy": {
                "model_state_mode": (
                    "delta" if checkpoint_model_prefixes else "full"
                ),
                "model_state_prefixes": list(checkpoint_model_prefixes),
                "base_checkpoint": checkpoint_base or None,
                "include_optimizer": checkpoint_include_optimizer,
                "include_rng_state": checkpoint_include_optimizer,
                "save_last_alias": save_last_alias,
            },
        }
    )

    def vis(images, targets, metas, outputs, iteration):
        if iteration % vis_interval == 0:
            save_prediction_visuals(images, targets, metas, outputs, out_dir / "vis", iteration)

    checkpoint_interval = int(cfg.get("training", {}).get("checkpoint_interval", 0))

    def save_training_checkpoint(path: Path, iteration: int) -> None:
        include_state = checkpoint_include_optimizer
        save_checkpoint(
            path,
            model,
            optimizer if include_state else None,
            scaler if include_state else None,
            iteration,
            cfg,
            scheduler=scheduler if include_state else None,
            model_state_prefixes=checkpoint_model_prefixes,
            base_checkpoint=checkpoint_base or None,
            include_rng_state=include_state,
        )

    def save_periodic(iteration: int):
        if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            save_training_checkpoint(
                out_dir / f"iter_{iteration:07d}.pt",
                iteration,
            )

    end_iter = train_one_epoch(
        train_model,
        loader,
        matcher,
        criterion,
        optimizer,
        device,
        cfg,
        start_iter=start_iter,
        max_iters=planned_iters,
        scaler=scaler,
        scheduler=scheduler,
        logger=SmoothedLogger(),
        visualizer=vis,
        checkpoint_saver=save_periodic,
    )
    if save_last_alias:
        save_training_checkpoint(out_dir / "last.pt", end_iter)
    save_training_checkpoint(out_dir / f"iter_{end_iter:07d}.pt", end_iter)


if __name__ == "__main__":
    main()
