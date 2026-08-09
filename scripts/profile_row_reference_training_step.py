from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any
from types import MethodType

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
    build_optimizer,
)
from dynlaneseq_eg.modeling.common import nested_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one real LaneRowNet optimizer step without saving or changing a run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--breakdown-steps", type=int, default=3)
    parser.add_argument("--profiler-steps", type=int, default=1)
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Include allocation events in the operator profiler (much slower).",
    )
    parser.add_argument("--row-limit", type=int, default=25)
    parser.add_argument("--trace", default="")
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Collect Python source stacks for synchronization/root-cause analysis.",
    )
    parser.add_argument(
        "--dispatch-stack-audit",
        action="store_true",
        help=(
            "Diagnostic only: attribute scalar extraction and nonzero ops to "
            "their exact Python source lines with TorchDispatchMode."
        ),
    )
    parser.add_argument(
        "--module-breakdown",
        action="store_true",
        help="Time major V7 model classes and criterion methods with CUDA events.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Benchmark torch.compile on the model only; the criterion and matcher stay eager.",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument(
        "--disable-intermediate-supervision",
        action="store_true",
        help=(
            "Diagnostic only: retain the row-reference decoder but suppress "
            "auxiliary decoder-layer outputs and losses to isolate their cost."
        ),
    )
    return parser.parse_args()


class SynchronizationStackAudit(TorchDispatchMode):
    """Attribute eager synchronization-prone ops to repository source lines."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[tuple[str, str, str, int, str], int] = defaultdict(int)

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        name = str(func)
        if name in {
            "aten._local_scalar_dense.default",
            "aten.nonzero.default",
        }:
            tensor = next(
                (value for value in args if isinstance(value, torch.Tensor)),
                None,
            )
            device = str(tensor.device) if tensor is not None else "no-tensor"
            frames = traceback.extract_stack(limit=32)
            selected = None
            for frame in reversed(frames[:-1]):
                path = Path(frame.filename).resolve()
                if ROOT in path.parents and path.name != Path(__file__).name:
                    selected = frame
                    break
            if selected is not None:
                relative = str(Path(selected.filename).resolve().relative_to(ROOT))
                key = (
                    name,
                    device,
                    relative,
                    int(selected.lineno),
                    str(selected.line or ""),
                )
            else:
                key = (name, device, "<outside repository>", 0, "")
            self.counts[key] += 1
        return func(*args, **(kwargs or {}))

    def report(self) -> None:
        print("\nDISPATCH SYNCHRONIZATION CALL SITES")
        rows = sorted(
            self.counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for (name, device, path, line, source), count in rows:
            print(
                f"{count:5d} {name:34s} {device:8s} "
                f"{path}:{line} {source}"
            )


class CudaRegionTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.records: dict[
            str,
            list[tuple[torch.cuda.Event, torch.cuda.Event, float]],
        ] = defaultdict(list)

    def begin(self) -> tuple[torch.cuda.Event, float]:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, time.perf_counter()

    def end(self, name: str, token: tuple[torch.cuda.Event, float]) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        start, cpu_start = token
        self.records[name].append((start, end, time.perf_counter() - cpu_start))

    def report(self, optimizer_steps: int) -> None:
        sync(self.device)
        rows = []
        for name, records in self.records.items():
            cuda_ms = sum(start.elapsed_time(end) for start, end, _cpu in records)
            cpu_ms = 1000.0 * sum(cpu for _start, _end, cpu in records)
            rows.append((cuda_ms, cpu_ms, name, len(records)))
        rows.sort(reverse=True)
        print("\nCLASS / METHOD CUDA BREAKDOWN (inclusive)")
        for cuda_ms, cpu_ms, name, calls in rows:
            print(
                f"{name:78s} cuda={cuda_ms / max(optimizer_steps, 1):9.3f} "
                f"ms/step cpu={cpu_ms / max(optimizer_steps, 1):9.3f} "
                f"ms/step calls={calls / max(optimizer_steps, 1):5.1f}/step"
            )


class V7ModuleBreakdown:
    MODULE_NAMES = {
        "encoder.backbone",
        "encoder.fpn",
        "encoder.proj",
        "encoder.ms_proj.p4",
        "encoder.ms_proj.p5",
        "encoder.seg_aux_head",
        "encoder.centerline_aux_head",
        "structured_query_head.feature_proj",
        "structured_query_head.layers.0",
        "structured_query_head.layers.1",
        "structured_query_head.layers.2",
        "structured_query_head.layers.3",
        "structured_query_head.lane_state_layers.0",
        "structured_query_head.lane_state_layers.1",
        "structured_query_head.lane_state_layers.2",
        "structured_query_head.lane_state_layers.3",
        "structured_query_head.ownership_layers.0",
        "structured_query_head.ownership_layers.1",
        "structured_query_head.ownership_layers.2",
        "structured_query_head.ownership_layers.3",
        "structured_query_head.set_selection_head.input_projection",
        "structured_query_head.set_selection_head.proposal_encoder",
        "structured_query_head.set_selection_head.slot_decoder",
        "structured_query_head.set_selection_head.slot_refinement",
    }
    STRUCTURED_HEAD_METHODS = (
        "_row_features",
        "_initialize_image_reference",
        "_predict_from_row_tokens",
        "_sample_final_curve_evidence",
    )
    LANE_STATE_METHODS = (
        "prepare",
        "inject_rows",
        "collect",
        "decision",
        "_semantic_context",
    )
    ROW_REFERENCE_METHODS = (
        "_sample_local_profiles",
        "_grouped_inter_attention",
    )
    OWNERSHIP_METHODS = (
        "_set_attention_by_group",
        "_semantic_context",
    )
    FOUR_SLOT_METHODS = (
        "_proposal_features",
        "_candidate_valid",
    )
    CRITERION_METHODS = (
        "compute_exist_loss",
        "compute_point_loss",
        "compute_range_loss",
        "compute_line_iou_loss",
        "compute_seg_loss",
        "compute_centerline_loss",
        "compute_row_dfl_loss",
        "compute_four_slot_selection_loss",
        "compute_four_slot_geometry_loss",
        "add_intermediate_losses",
    )

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        matcher: object,
        device: torch.device,
    ) -> None:
        self.timer = CudaRegionTimer(device)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.original_methods: list[tuple[object, str, object]] = []
        self.active: dict[str, list[tuple[torch.cuda.Event, float]]] = defaultdict(list)
        for name, module in model.named_modules():
            if name not in self.MODULE_NAMES:
                continue

            def pre_hook(_module, _args, *, region=name):
                self.active[region].append(self.timer.begin())

            def post_hook(_module, _args, _output, *, region=name):
                self.timer.end(region, self.active[region].pop())

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))
        named_modules = dict(model.named_modules())
        structured_head = named_modules.get("structured_query_head")
        if structured_head is not None:
            self._wrap_existing_methods(
                structured_head,
                self.STRUCTURED_HEAD_METHODS,
                prefix="structured_query_head",
            )
        for name, module in named_modules.items():
            if name.startswith("structured_query_head.lane_state_layers.") and name.count(".") == 2:
                self._wrap_existing_methods(
                    module,
                    self.LANE_STATE_METHODS,
                    prefix=name,
                )
            elif name.startswith("structured_query_head.layers.") and name.count(".") == 2:
                self._wrap_existing_methods(
                    module,
                    self.ROW_REFERENCE_METHODS,
                    prefix=name,
                )
            elif name.startswith("structured_query_head.ownership_layers.") and name.count(".") == 2:
                self._wrap_existing_methods(
                    module,
                    self.OWNERSHIP_METHODS,
                    prefix=name,
                )
        four_slot_head = named_modules.get("structured_query_head.set_selection_head")
        if four_slot_head is not None:
            self._wrap_existing_methods(
                four_slot_head,
                self.FOUR_SLOT_METHODS,
                prefix="structured_query_head.set_selection_head",
            )
        for method_name in self.CRITERION_METHODS:
            self._wrap_method(criterion, method_name, prefix="criterion")
        if hasattr(matcher, "match_many"):
            self._wrap_method(matcher, "match_many", prefix="matcher")

    def _wrap_existing_methods(
        self,
        owner: object,
        names: tuple[str, ...],
        *,
        prefix: str,
    ) -> None:
        for name in names:
            if hasattr(owner, name):
                self._wrap_method(owner, name, prefix=prefix)

    def _wrap_method(self, owner: object, name: str, *, prefix: str) -> None:
        original = getattr(owner, name)
        self.original_methods.append((owner, name, original))

        def wrapped(_owner, *args, **kwargs):
            token = self.timer.begin()
            try:
                return original(*args, **kwargs)
            finally:
                self.timer.end(f"{prefix}.{name}", token)

        setattr(owner, name, MethodType(wrapped, owner))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        for owner, name, original in self.original_methods:
            setattr(owner, name, original)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_stage(
    totals: dict[str, float],
    name: str,
    start: float,
    device: torch.device,
) -> None:
    sync(device)
    totals[name] += time.perf_counter() - start


def amp_settings(cfg: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype | None, bool]:
    training = cfg.get("training", {})
    enabled = bool(training.get("amp", False) and device.type == "cuda")
    name = str(training.get("amp_dtype", "")).strip().lower()
    dtype = None
    if name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif name in {"fp16", "float16", "half"}:
        dtype = torch.float16
    use_scaler = bool(enabled and dtype != torch.bfloat16)
    return enabled, dtype, use_scaler


def autocast_kwargs(
    device: torch.device,
    enabled: bool,
    dtype: torch.dtype | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"device_type": device.type, "enabled": enabled}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def move_batch(
    batch: tuple[torch.Tensor, list[dict[str, torch.Tensor]], object],
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, targets, _ = batch
    if channels_last:
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    else:
        images = images.to(device, non_blocking=True)
    return images, nested_to_device(targets, device)


def train_micro_batch(
    *,
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matcher: object,
    criterion: torch.nn.Module,
    cfg: dict[str, Any],
    iteration: int,
    accumulation_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
    totals: dict[str, float] | None,
    device: torch.device,
) -> None:
    start = time.perf_counter()
    with torch.autocast(**autocast_kwargs(device, amp_enabled, amp_dtype)):
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            iteration,
        )
    if totals is not None:
        elapsed_stage(totals, "forward_and_final_match", start, device)

    start = time.perf_counter()
    with torch.autocast(**autocast_kwargs(device, amp_enabled, amp_dtype)):
        if hasattr(criterion, "set_iteration"):
            criterion.set_iteration(iteration)
        losses = criterion(outputs, targets, matches)
        loss = losses["loss_total"] / float(accumulation_steps)
    if totals is not None:
        elapsed_stage(totals, "loss_and_aux_matches", start, device)

    start = time.perf_counter()
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    if totals is not None:
        elapsed_stage(totals, "backward", start, device)


def optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    clip_norm: float,
) -> None:
    if scaler is not None:
        scaler.unscale_(optimizer)
    if clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def run_optimizer_steps(
    *,
    count: int,
    iterator: Any,
    model: torch.nn.Module,
    matcher: object,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    device: torch.device,
    channels_last: bool,
    accumulation_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
    clip_norm: float,
    totals: dict[str, float] | None = None,
) -> None:
    for iteration in range(count):
        for _ in range(accumulation_steps):
            start = time.perf_counter()
            batch = next(iterator)
            if totals is not None:
                totals["data_wait"] += time.perf_counter() - start

            start = time.perf_counter()
            images, targets = move_batch(batch, device, channels_last)
            if totals is not None:
                elapsed_stage(totals, "host_to_device", start, device)

            train_micro_batch(
                model=model,
                images=images,
                targets=targets,
                matcher=matcher,
                criterion=criterion,
                cfg=cfg,
                iteration=iteration,
                accumulation_steps=accumulation_steps,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
                totals=totals,
                device=device,
            )

        start = time.perf_counter()
        optimizer_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            clip_norm=clip_norm,
        )
        if totals is not None:
            elapsed_stage(totals, "clip_and_optimizer", start, device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    if args.batch_size > 0:
        cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    if args.grad_accum > 0:
        cfg.setdefault("training", {})["gradient_accumulation_steps"] = int(args.grad_accum)
    if args.num_workers > 0:
        cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if args.disable_intermediate_supervision:
        cfg.setdefault("model", {}).setdefault("structured_query", {})[
            "intermediate_supervision"
        ] = False
        cfg.setdefault("loss", {})["lambda_intermediate"] = 0.0

    training = cfg.get("training", {})
    seed_everything(int(training.get("seed", 3407)))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU.")
    torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", False))
    if bool(training.get("tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    channels_last = bool(training.get("channels_last", False))
    model = build_model(cfg).to(device).train()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    if args.compile_model:
        model = torch.compile(model, mode=str(args.compile_mode))
    amp_enabled, amp_dtype, use_scaler = amp_settings(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_scaler else None
    accumulation_steps = max(
        int(training.get("gradient_accumulation_steps", 1)),
        1,
    )
    clip_norm = float(training.get("clip_grad_norm", 1.0))
    loader = build_dataloader(cfg, split="train", training=True)

    def batches() -> Any:
        while True:
            yield from loader

    iterator = iter(batches())
    batch_size = int(training.get("batch_size", 1))
    row_reference = (
        cfg.get("model", {})
        .get("structured_query", {})
        .get("row_reference", {})
    )
    print(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "num_workers": int(cfg.get("dataloader", {}).get("num_workers", 2)),
            "amp_dtype": str(amp_dtype),
            "channels_last": channels_last,
            "sampling_backend": row_reference.get("sampling_backend", "grid_sample"),
            "projection_backend": row_reference.get("projection_backend", "separate"),
            "attention_backend": row_reference.get("attention_backend", "materialized"),
            "compile_model": bool(args.compile_model),
            "compile_mode": str(args.compile_mode) if args.compile_model else None,
            "intermediate_supervision": bool(
                cfg.get("model", {})
                .get("structured_query", {})
                .get("intermediate_supervision", False)
            ),
            "warmup_steps": args.warmup_steps,
            "breakdown_steps": args.breakdown_steps,
            "profiler_steps": args.profiler_steps,
        }
    )

    optimizer.zero_grad(set_to_none=True)
    run_optimizer_steps(
        count=max(args.warmup_steps, 0),
        iterator=iterator,
        model=model,
        matcher=matcher,
        criterion=criterion,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        channels_last=channels_last,
        accumulation_steps=accumulation_steps,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
        clip_norm=clip_norm,
    )

    module_breakdown = (
        V7ModuleBreakdown(model, criterion, matcher, device)
        if bool(args.module_breakdown)
        else None
    )
    totals: dict[str, float] = defaultdict(float)
    torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    wall_start = time.perf_counter()
    dispatch_audit = (
        SynchronizationStackAudit()
        if bool(args.dispatch_stack_audit)
        else None
    )
    dispatch_context = dispatch_audit if dispatch_audit is not None else nullcontext()
    with dispatch_context:
        run_optimizer_steps(
            count=max(args.breakdown_steps, 1),
            iterator=iterator,
            model=model,
            matcher=matcher,
            criterion=criterion,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            channels_last=channels_last,
            accumulation_steps=accumulation_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            clip_norm=clip_norm,
            totals=totals,
        )
    sync(device)
    wall_elapsed = time.perf_counter() - wall_start
    accounted = sum(totals.values())
    print("\nSYNCHRONIZED STAGE BREAKDOWN")
    for name, value in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        print(
            f"{name:26s} {value / max(args.breakdown_steps, 1):8.4f} s/optimizer-step "
            f"({100.0 * value / max(accounted, 1e-9):5.1f}%)"
        )
    effective_images = (
        max(args.breakdown_steps, 1)
        * batch_size
        * accumulation_steps
    )
    print(
        f"{'synchronized_total':26s} "
        f"{wall_elapsed / max(args.breakdown_steps, 1):8.4f} s/optimizer-step "
        f"| {effective_images / max(wall_elapsed, 1e-9):.2f} img/s "
        f"| allocated={torch.cuda.max_memory_allocated(device) / (1024.0 ** 3):.2f} GiB "
        f"| reserved={torch.cuda.max_memory_reserved(device) / (1024.0 ** 3):.2f} GiB "
        f"| device={torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 3):.2f} GiB"
    )
    if dispatch_audit is not None:
        dispatch_audit.report()
    if module_breakdown is not None:
        module_breakdown.timer.report(max(args.breakdown_steps, 1))
        module_breakdown.close()

    if args.profiler_steps <= 0:
        return
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=bool(args.with_stack),
        profile_memory=bool(args.profile_memory),
        with_stack=bool(args.with_stack),
    ) as profiler:
        run_optimizer_steps(
            count=args.profiler_steps,
            iterator=iterator,
            model=model,
            matcher=matcher,
            criterion=criterion,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            channels_last=channels_last,
            accumulation_steps=accumulation_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            clip_norm=clip_norm,
        )
        sync(device)
    print("\nTOP CUDA OPERATORS")
    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=max(args.row_limit, 1),
        )
    )
    print("\nTOP CPU OPERATORS")
    print(
        profiler.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=max(args.row_limit, 1),
        )
    )
    if args.with_stack:
        print("\nSYNCHRONIZATION SOURCE STACKS")
        grouped = profiler.key_averages(group_by_stack_n=8)
        interesting = {
            "aten::_local_scalar_dense",
            "aten::item",
            "aten::nonzero",
            "aten::to",
            "aten::_to_copy",
        }
        rows = [event for event in grouped if event.key in interesting]
        rows.sort(
            key=lambda event: (
                int(event.count),
                float(event.self_cpu_time_total),
            ),
            reverse=True,
        )
        for event in rows[:40]:
            stack = tuple(getattr(event, "stack", ()) or ())
            source = " <- ".join(stack[-5:]) if stack else "<stack unavailable>"
            print(
                f"{event.key:28s} count={int(event.count):5d} "
                f"self_cpu={float(event.self_cpu_time_total) / 1000.0:9.3f} ms "
                f"{source}"
            )
    if args.trace:
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
        print(f"trace: {trace_path}")


if __name__ == "__main__":
    main()
