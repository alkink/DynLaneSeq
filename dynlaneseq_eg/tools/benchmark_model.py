from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import platform
from typing import Any

import torch
from torch import nn

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_model


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list.")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class _InferenceOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module, inference_only: bool) -> None:
        super().__init__()
        self.model = model
        self.inference_only = bool(inference_only)

    def forward(self, images: torch.Tensor):
        if self.inference_only:
            return self.model(images, inference_only=True)
        return self.model(images)


def _sdpa_flop_handle(inputs, _outputs) -> Counter:
    # fvcore counts one fused multiply-add as one FLOP. Match that convention
    # for QK^T and attention-value multiplication.
    from fvcore.nn.jit_handles import get_shape

    q_shape = get_shape(inputs[0])
    k_shape = get_shape(inputs[1])
    if q_shape is None or k_shape is None:
        return Counter()
    batch_heads = 1
    for size in q_shape[:-2]:
        batch_heads *= int(size)
    flops = 2 * batch_heads * int(q_shape[-2]) * int(k_shape[-2]) * int(q_shape[-1])
    return Counter({"scaled_dot_product_attention": flops})


def _measure_complexity(model: nn.Module, images: torch.Tensor, inference_only: bool) -> dict[str, Any]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("--compute-flops requires fvcore.") from exc

    wrapper = _InferenceOnlyWrapper(model, inference_only=inference_only).eval()
    analysis = FlopCountAnalysis(wrapper, images)
    analysis.set_op_handle("aten::scaled_dot_product_attention", _sdpa_flop_handle)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    total = int(analysis.total())
    unsupported = {str(name): int(count) for name, count in analysis.unsupported_ops().items()}
    return {
        "fvcore_flops": total,
        "fvcore_gflops": total / 1.0e9,
        "fvcore_convention": "one fused multiply-add counts as one FLOP",
        "custom_counted_ops": ["aten::scaled_dot_product_attention (QK^T and AV matmuls)"],
        "unsupported_ops": unsupported,
    }


def _amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def _measure_cuda_latency(
    model: nn.Module,
    images: torch.Tensor,
    *,
    inference_only: bool,
    amp_dtype: str,
    warmup_iters: int,
    measure_iters: int,
) -> dict[str, Any]:
    for _ in range(warmup_iters):
        with _amp_context(images.device, amp_dtype):
            if inference_only:
                outputs = model(images, inference_only=True)
            else:
                outputs = model(images)
        del outputs
    torch.cuda.synchronize(images.device)

    baseline_allocated = int(torch.cuda.memory_allocated(images.device))
    baseline_reserved = int(torch.cuda.memory_reserved(images.device))
    torch.cuda.reset_peak_memory_stats(images.device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    for index in range(measure_iters):
        starts[index].record()
        with _amp_context(images.device, amp_dtype):
            if inference_only:
                outputs = model(images, inference_only=True)
            else:
                outputs = model(images)
        ends[index].record()
        del outputs
    torch.cuda.synchronize(images.device)

    latencies_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    return {
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": _percentile(latencies_ms, 0.50),
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "throughput_images_per_second": images.shape[0] * 1000.0 / mean_ms,
        "baseline_memory_allocated_mb": baseline_allocated / (1024.0**2),
        "baseline_memory_reserved_mb": baseline_reserved / (1024.0**2),
        "peak_memory_allocated_mb": torch.cuda.max_memory_allocated(images.device) / (1024.0**2),
        "peak_memory_reserved_mb": torch.cuda.max_memory_reserved(images.device) / (1024.0**2),
        "latency_samples_ms": latencies_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DynLaneSeq model-forward complexity and CUDA latency.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--measure-iters", type=int, default=500)
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="none")
    parser.add_argument("--legacy-forward", action="store_true")
    parser.add_argument("--compute-flops", action="store_true")
    parser.add_argument("--no-pretrained-init", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.batch_size < 1 or args.warmup_iters < 0 or args.measure_iters < 1:
        raise ValueError("batch-size and measure-iters must be positive; warmup-iters cannot be negative.")

    cfg = load_config(args.config)
    if args.no_pretrained_init or args.checkpoint:
        cfg.setdefault("model", {})["pretrained_backbone"] = False
    model = build_model(cfg).eval()
    checkpoint_iteration = 0
    if args.checkpoint:
        checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)

    # Keep the architectural/checkpoint count as the primary parameter number.
    # This includes training-only auxiliary modules and prevents deployment
    # pruning from making the paper-facing count look artificially small.
    params_total = sum(parameter.numel() for parameter in model.parameters())
    params_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model_cfg = cfg.get("model", {})
    input_h = int(model_cfg["input_h"])
    input_w = int(model_cfg["input_w"])
    inference_only = bool(getattr(model, "supports_inference_only", False) and not args.legacy_forward)

    if inference_only and hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    resident_params = sum(parameter.numel() for parameter in model.parameters())

    report: dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "model_name": model_cfg.get("name", model.__class__.__name__),
        "parameters_total": params_total,
        "parameters_trainable": params_trainable,
        "parameters_millions": params_total / 1.0e6,
        "inference_resident_parameters": resident_params,
        "inference_resident_parameters_millions": resident_params / 1.0e6,
        "parameter_count_note": (
            "parameters_total is the unpruned checkpoint architecture; "
            "inference_resident_parameters is reported separately after safe deployment pruning"
        ),
        "input_height": input_h,
        "input_width": input_w,
        "batch_size": args.batch_size,
        "forward_path": "inference_only" if inference_only else "legacy_full_forward",
        "amp_dtype": args.amp_dtype,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    if args.compute_flops:
        complexity_input = torch.randn(args.batch_size, 3, input_h, input_w)
        report["complexity"] = _measure_complexity(model, complexity_input, inference_only=inference_only)
        del complexity_input

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmarking requested, but CUDA is not available.")
        training_cfg = cfg.get("training", {})
        tf32 = bool(training_cfg.get("tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = bool(training_cfg.get("cudnn_benchmark", False))
        channels_last = bool(training_cfg.get("channels_last", False))
        model = model.to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
            images = torch.randn(args.batch_size, 3, input_h, input_w, device=device).to(
                memory_format=torch.channels_last
            )
        else:
            images = torch.randn(args.batch_size, 3, input_h, input_w, device=device)
        report.update(
            {
                "device": torch.cuda.get_device_name(device),
                "device_compute_capability": list(torch.cuda.get_device_capability(device)),
                "cuda_runtime": torch.version.cuda,
                "tf32": tf32,
                "channels_last": channels_last,
                "warmup_iterations": args.warmup_iters,
                "measurement_iterations": args.measure_iters,
            }
        )
        report["latency"] = _measure_cuda_latency(
            model,
            images,
            inference_only=inference_only,
            amp_dtype=args.amp_dtype,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )
    else:
        report["device"] = str(device)
        report["latency"] = None

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
