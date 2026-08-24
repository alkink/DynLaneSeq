from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # Lightweight unit-test environments may omit tqdm.
    def tqdm(iterable, **_kwargs):
        return iterable

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.losses.range_aware_iou import (
    batched_pairwise_range_aware_row_strip_iou,
)
from dynlaneseq_eg.modeling.common import nested_to_device


AUDIT_VERSION = 1


def _canonical_image_id(value: str, dataset_root: Path) -> str:
    path = Path(str(value))
    if path.is_absolute():
        try:
            return path.resolve().relative_to(dataset_root.resolve()).as_posix()
        except ValueError:
            pass
    parts = path.parts
    if "CULane" in parts:
        index = parts.index("CULane")
        return Path(*parts[index + 1 :]).as_posix()
    return path.as_posix().lstrip("/")


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(np.mean(finite))


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "q10": float(np.quantile(finite, 0.10)),
        "q90": float(np.quantile(finite, 0.90)),
    }


def _clip_bootstrap_ci(
    rows: list[dict[str, Any]],
    key: str,
    *,
    repetitions: int,
    seed: int,
) -> list[float] | None:
    by_clip: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        by_clip[str(row["clip"])].append(float(value))
    clips = sorted(by_clip)
    if not clips:
        return None
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(repetitions), dtype=np.float64)
    for index in range(int(repetitions)):
        selected = rng.choice(clips, size=len(clips), replace=True)
        values = [value for clip in selected for value in by_clip[str(clip)]]
        samples[index] = float(np.mean(values))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _candidate_quality(
    pred_x: torch.Tensor,
    pred_range: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    input_h: int,
    line_width: float,
    min_valid_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quality, candidate_valid, gt_valid = (
        batched_pairwise_range_aware_row_strip_iou(
            pred_x.float().unsqueeze(0),
            pred_range.float().unsqueeze(0),
            target["x_rows"].float().unsqueeze(0),
            target["valid_mask"].bool().unsqueeze(0),
            input_h=int(input_h),
            line_width=float(line_width),
            min_valid_rows=int(min_valid_rows),
        )
    )
    return quality[0].cpu(), candidate_valid[0].cpu(), gt_valid[0].cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace V36 good/current-wrong queries through every V7 decoder "
            "assignment and measure final-vs-intermediate existence gradients."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--replay-list", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--v36-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--gradient-images-per-fold", type=int, default=64)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bf16", "fp16"),
        default="bf16",
    )
    return parser.parse_args()


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"expected dictionary torch payload: {path}")
    return payload


def _uniform_take(values: list[str], count: int) -> list[str]:
    if count <= 0 or len(values) <= count:
        return list(values)
    indices = np.linspace(0, len(values) - 1, num=int(count), dtype=np.int64)
    return [values[int(index)] for index in indices.tolist()]


def layer_loss_coefficients(
    intermediate_weights: list[float] | tuple[float, ...],
    intermediate_strength: float,
) -> list[float]:
    weights = [float(value) for value in intermediate_weights]
    if not weights:
        raise ValueError("V37 requires explicit intermediate layer weights")
    denominator = float(sum(weights))
    if denominator <= 0.0:
        raise ValueError("intermediate layer weights must sum positive")
    return [
        float(intermediate_strength) * weight / denominator
        for weight in weights
    ] + [1.0]


def _assignment_map(match: dict[str, torch.Tensor]) -> dict[int, int]:
    return {
        int(candidate): int(gt)
        for candidate, gt in zip(
            match["pred_indices"].detach().cpu().tolist(),
            match["gt_indices"].detach().cpu().tolist(),
        )
    }


def _amp_context(device: torch.device, name: str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(name))
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _exist_labels(
    logits: torch.Tensor,
    matches: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    labels = torch.zeros(
        logits.shape[:2],
        dtype=torch.float32,
        device=logits.device,
    )
    for batch_index, match in enumerate(matches):
        indices = match["pred_indices"].to(logits.device)
        labels[batch_index, indices] = 1.0
    return labels


def exact_exist_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    no_lane_weight: float,
) -> torch.Tensor:
    targets = (labels <= 0.0).long()
    weights = logits.new_tensor([1.0, float(no_lane_weight)])
    return F.cross_entropy(
        logits.float().reshape(-1, 2),
        targets.reshape(-1),
        weight=weights.float(),
    )


def exact_candidate_descent_update(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    no_lane_weight: float,
    coefficient: float,
    exist_loss_weight: float,
) -> torch.Tensor:
    """Exact CE descent direction for each foreground logit in one batch."""

    probability = torch.softmax(logits.float(), dim=-1)[..., 0]
    sample_weight = torch.where(
        labels > 0.0,
        torch.ones_like(labels),
        torch.full_like(labels, float(no_lane_weight)),
    )
    denominator = sample_weight.sum().clamp_min(1.0)
    return (
        float(coefficient)
        * float(exist_loss_weight)
        * sample_weight
        * (labels - probability)
        / denominator
    )


def _flatten_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: tuple[torch.nn.Parameter, ...],
) -> torch.Tensor:
    pieces = []
    for gradient, parameter in zip(gradients, parameters):
        pieces.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
        )
    return torch.cat(pieces).float()


def _gradient_pair(left: torch.Tensor, right: torch.Tensor) -> dict[str, float] | None:
    left_norm = float(left.double().norm())
    right_norm = float(right.double().norm())
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    cosine = float(
        torch.dot(left.double(), right.double()) / (left_norm * right_norm)
    )
    return {
        "cosine": cosine,
        "right_to_left_norm_ratio": right_norm / left_norm,
    }


def _gradient_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {
            "batches": 0,
            "cosine": _quantiles([]),
            "right_to_left_norm_ratio": _quantiles([]),
            "negative_fraction": None,
        }
    return {
        "batches": len(rows),
        "cosine": _quantiles(row["cosine"] for row in rows),
        "right_to_left_norm_ratio": _quantiles(
            row["right_to_left_norm_ratio"] for row in rows
        ),
        "negative_fraction": float(
            np.mean([row["cosine"] < 0.0 for row in rows])
        ),
    }


def _cohort_summary(
    rows: list[dict[str, Any]],
    *,
    layers: int,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    binary_keys = (
        "good_final_any_positive",
        "good_all_layers_intended",
        "good_all_layers_any_positive",
        "good_first_acquired_at_final",
        "good_intermediate_has_negative",
        "intended_query_stable_all_layers",
        "good_total_update_up",
        "good_total_update_down",
        "wrong_total_update_down",
        "wrong_total_update_up",
    )
    scalar_keys = (
        "good_total_exist_update",
        "wrong_total_exist_update",
        "good_final_exist_update",
        "wrong_final_exist_update",
        "good_intermediate_exist_update",
        "wrong_intermediate_exist_update",
        "intended_query_switches",
    )
    metrics: dict[str, Any] = {}
    for key in binary_keys:
        metrics[key] = {
            "mean": _mean(row[key] for row in rows),
            "clip_bootstrap_95": _clip_bootstrap_ci(
                rows,
                key,
                repetitions=bootstrap_reps,
                seed=seed,
            ),
        }
    for key in scalar_keys:
        metrics[key] = _quantiles(row[key] for row in rows)

    layer_summary = []
    for layer in range(int(layers)):
        item = {}
        for key in (
            "good_intended_trace",
            "good_any_trace",
            "wrong_intended_trace",
            "wrong_any_trace",
        ):
            item[key.replace("_trace", "_rate")] = float(
                np.mean([row[key][layer] for row in rows])
            )
        for key in (
            "good_exist_trace",
            "wrong_exist_trace",
            "good_quality_trace",
            "wrong_quality_trace",
        ):
            item[f"mean_{key.replace('_trace', '')}"] = float(
                np.mean([row[key][layer] for row in rows])
            )
        layer_summary.append(item)
    return {
        "pairs": len(rows),
        "clips": len({str(row["clip"]) for row in rows}),
        "metrics": metrics,
        "layers": layer_summary,
    }


def classify_verdict(
    overall: dict[str, Any],
    final_positive: dict[str, Any],
    gradients: dict[str, Any],
) -> dict[str, Any]:
    good_up = float(overall["metrics"]["good_total_update_up"]["mean"])
    wrong_down = float(overall["metrics"]["wrong_total_update_down"]["mean"])
    final_good_down = float(
        final_positive["metrics"]["good_total_update_down"]["mean"]
    )
    all_layer_intended = float(
        final_positive["metrics"]["good_all_layers_intended"]["mean"]
    )
    negative = gradients.get("negative_fraction")
    ratio = gradients.get("right_to_left_norm_ratio", {}).get("median")
    negative_value = 0.0 if negative is None else float(negative)
    ratio_value = 0.0 if ratio is None else float(ratio)

    if final_good_down >= 0.25 or (
        negative_value >= 0.50 and ratio_value >= 0.25
    ):
        decision = "DEEP_SUPERVISION_SCORE_CONFLICT"
        interpretation = (
            "Intermediate assignment/existence supervision final-good query "
            "için net ters score baskısı oluşturuyor."
        )
    elif (
        all_layer_intended < 0.65
        and final_good_down < 0.10
        and negative_value < 0.25
    ):
        decision = "ASSIGNMENT_CHURN_WITH_ALIGNED_SCORE_GRADIENT"
        interpretation = (
            "Query ownership layerlar arasında değişiyor, fakat exact weighted "
            "score gradienti good'u net olarak aşağı itmiyor."
        )
    elif good_up >= 0.90 and wrong_down >= 0.90 and negative_value < 0.25:
        decision = "DEEP_SUPERVISION_NOT_PRIMARY"
        interpretation = (
            "Intermediate ve final existence supervision net olarak doğru "
            "yönde; deep supervision ters belief'in ana nedeni değil."
        )
    else:
        decision = "MIXED_DEEP_SUPERVISION_EFFECT"
        interpretation = (
            "Deep supervision etkisi tek bir temiz conflict veya consistency "
            "sınıfına girmiyor; uzun run açmadan alt kırılım gerekir."
        )
    return {
        "decision": decision,
        "interpretation": interpretation,
        "good_total_update_up_rate": good_up,
        "wrong_total_update_down_rate": wrong_down,
        "final_good_positive_total_update_down_rate": final_good_down,
        "final_good_positive_all_layer_intended_rate": all_layer_intended,
        "exist_head_gradient_negative_fraction": negative,
        "exist_head_intermediate_to_final_norm_ratio_median": ratio,
        "test_split_used": False,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    def show(value: Any) -> str:
        return "—" if value is None else f"{float(value):.4f}"

    overall = payload["summary"]["overall"]
    lines = [
        "# V37 Deep-Supervision Assignment Trajectory Results",
        "",
        f"- Decision: **{payload['verdict']['decision']}**",
        f"- Pairs: `{overall['pairs']}`",
        f"- Decoder layers: `{payload['settings']['layer_coefficients']}`",
        f"- Final parity max abs px: `{payload['parity']['pred_x_max_abs']:.6f}`",
        "- Test split used: `False`",
        "",
        "| Layer | Good intended + | Good any + | Wrong intended + | Good exist | Wrong exist | Good quality | Wrong quality |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(overall["layers"], start=1):
        lines.append(
            f"| {index} | {show(item['good_intended_rate'])} | "
            f"{show(item['good_any_rate'])} | {show(item['wrong_intended_rate'])} | "
            f"{show(item['mean_good_exist'])} | {show(item['mean_wrong_exist'])} | "
            f"{show(item['mean_good_quality'])} | {show(item['mean_wrong_quality'])} |"
        )
    metrics = overall["metrics"]
    gradient = payload["gradient_audit"]["aggregate_intermediate_vs_final"]
    lines.extend(
        (
            "",
            "## Ana ölçümler",
            "",
            f"- Good total update up: `{show(metrics['good_total_update_up']['mean'])}`",
            f"- Wrong total update down: `{show(metrics['wrong_total_update_down']['mean'])}`",
            f"- Good all-layer intended: `{show(metrics['good_all_layers_intended']['mean'])}`",
            f"- Good first acquired at final: `{show(metrics['good_first_acquired_at_final']['mean'])}`",
            f"- Exist-head gradient negative fraction: `{show(gradient['negative_fraction'])}`",
            f"- Intermediate/final gradient norm ratio median: `{show(gradient['right_to_left_norm_ratio']['median'])}`",
            "",
            "## Yorum",
            "",
            payload["verdict"]["interpretation"],
            "",
            "Bu bir training-contract diagnostic'idir; official validation F1 sonucu değildir.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    replay_list = Path(args.replay_list).expanduser().resolve()
    cache_path = Path(args.target_cache).expanduser().resolve()
    v36_path = Path(args.v36_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    v36 = json.loads(v36_path.read_text(encoding="utf-8"))
    pairs = v36.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("V36 JSON does not contain evaluated pairs")
    pairs_by_image: dict[str, list[dict[str, Any]]] = {}
    fold_by_image: dict[str, str] = {}
    for pair in pairs:
        image_id = _canonical_image_id(str(pair["image_id"]), dataset_root)
        pairs_by_image.setdefault(image_id, []).append(pair)
        fold_by_image.setdefault(image_id, str(pair["fold"]))

    gradient_images: list[str] = []
    for fold in ("a", "b"):
        candidates = sorted(
            image for image, value in fold_by_image.items() if value == fold
        )
        gradient_images.extend(
            _uniform_take(candidates, int(args.gradient_images_per_fold))
        )
    gradient_set = set(gradient_images)
    if not replay_list.exists():
        raise FileNotFoundError(f"V34 replay list does not exist: {replay_list}")
    replay_images = {
        Path(line.strip().split()[0]).as_posix().lstrip("/")
        for line in replay_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    missing_replay = sorted(set(pairs_by_image) - replay_images)
    if missing_replay:
        raise ValueError(
            f"V34 replay list is missing {len(missing_replay)} V36 pair images"
        )
    (output_dir / "v37_gradient_images.txt").write_text(
        "".join(f"/{image.lstrip('/')}\n" for image in gradient_images),
        encoding="utf-8",
    )

    target_cache = _load_torch(cache_path)
    cache_records = {
        _canonical_image_id(
            str(record.get("meta", {}).get("image_path", record.get("image_id", ""))),
            dataset_root,
        ): record
        for record in target_cache.get("records", [])
    }
    missing = sorted(set(pairs_by_image) - set(cache_records))
    if missing:
        raise ValueError(f"target cache is missing {len(missing)} V36 images")

    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = str(dataset_root)
    cfg["dataset"].setdefault("lists", {})["val"] = str(replay_list)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(int(args.num_workers) > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg["model"].setdefault("structured_query", {})["intermediate_supervision"] = True

    device = torch.device(args.device)
    model = build_model(cfg)
    checkpoint_iteration = int(load_checkpoint(checkpoint_path, model, strict=False))
    model = model.to(device).eval()
    channels_last = bool(cfg.get("training", {}).get("channels_last", False))
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    head = model.structured_query_head
    if head is None:
        raise ValueError("V37 requires the structured query head")
    head.intermediate_supervision = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    exist_parameters = tuple(head.exist.parameters())
    for parameter in exist_parameters:
        parameter.requires_grad_(True)

    matcher = build_matcher(cfg)
    matcher.set_iteration(checkpoint_iteration)
    loss_cfg = cfg.get("loss", {})
    configured_weights = [
        float(value) for value in loss_cfg.get("intermediate_layer_weights", [])
    ]
    coefficients = layer_loss_coefficients(
        configured_weights,
        float(loss_cfg.get("lambda_intermediate", 0.0)),
    )
    no_lane_weight = float(loss_cfg.get("no_lane_weight", 0.10))
    final_exist_loss_weight = float(loss_cfg.get("w_exist", 0.0))
    intermediate_exist_loss_weight = float(
        loss_cfg.get("w_intermediate_exist", final_exist_loss_weight)
    )
    exist_loss_weights = [
        intermediate_exist_loss_weight for _ in configured_weights
    ] + [final_exist_loss_weight]
    input_h = int(cfg.get("model", {}).get("input_h", 640))
    loader = build_dataloader(cfg, split="val", training=False)

    trajectory_rows: list[dict[str, Any]] = []
    gradient_aggregate: list[dict[str, float]] = []
    gradient_by_layer: list[list[dict[str, float]]] = [
        [] for _ in configured_weights
    ]
    pred_abs_sum = 0.0
    pred_count = 0
    pred_abs_max = 0.0
    range_abs_max = 0.0
    pair_parity_rows: list[dict[str, Any]] = []
    processed_gradient_images = 0
    seen_images: set[str] = set()
    layer_count: int | None = None

    for images, targets_cpu, metas in tqdm(loader, desc="V37 trajectory", ncols=100):
        batch_ids = [
            _canonical_image_id(str(meta["image_path"]), dataset_root)
            for meta in metas
        ]
        gradient_batch_indices = [
            index
            for index, image_id in enumerate(batch_ids)
            if image_id in gradient_set
        ]
        do_gradient = bool(gradient_batch_indices)
        if do_gradient:
            processed_gradient_images += len(gradient_batch_indices)
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last
                if channels_last and device.type == "cuda"
                else torch.contiguous_format
            ),
        )
        targets = nested_to_device(targets_cpu, device)
        with torch.set_grad_enabled(do_gradient), _amp_context(device, args.amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
            outputs = head(
                encoded["features"],
                multi_scale_features=encoded.get("multi_scale_features"),
                inference_only=False,
            )
            auxiliaries = outputs.get("aux_outputs")
            if not isinstance(auxiliaries, (list, tuple)) or not auxiliaries:
                raise ValueError("V7 did not expose intermediate outputs")
            layers = [*auxiliaries, outputs]
            if layer_count is None:
                layer_count = len(layers)
                if layer_count != len(coefficients):
                    raise ValueError(
                        f"loss coefficients ({len(coefficients)}) do not match decoder layers ({layer_count})"
                    )
            elif layer_count != len(layers):
                raise RuntimeError("decoder layer count changed between batches")
            matches_by_layer = matcher.match_many(tuple(layers), targets)
            labels_by_layer = [
                _exist_labels(layer["exist_logits"], matches)
                for layer, matches in zip(layers, matches_by_layer)
            ]
            updates_by_layer = [
                exact_candidate_descent_update(
                    layer["exist_logits"],
                    labels,
                    no_lane_weight=no_lane_weight,
                    coefficient=coefficient,
                    exist_loss_weight=exist_weight,
                )
                for layer, labels, coefficient, exist_weight in zip(
                    layers,
                    labels_by_layer,
                    coefficients,
                    exist_loss_weights,
                )
            ]

            if do_gradient:
                gradient_indices = torch.tensor(
                    gradient_batch_indices,
                    dtype=torch.long,
                    device=device,
                )
                final_loss = (
                    final_exist_loss_weight
                    * exact_exist_loss(
                        layers[-1]["exist_logits"].index_select(
                            0, gradient_indices
                        ),
                        labels_by_layer[-1].index_select(0, gradient_indices),
                        no_lane_weight=no_lane_weight,
                    )
                )
                intermediate_losses = [
                    intermediate_exist_loss_weight
                    * coefficient
                    * exact_exist_loss(
                        layer["exist_logits"].index_select(0, gradient_indices),
                        labels.index_select(0, gradient_indices),
                        no_lane_weight=no_lane_weight,
                    )
                    for layer, labels, coefficient in zip(
                        layers[:-1], labels_by_layer[:-1], coefficients[:-1]
                    )
                ]
                final_grad = _flatten_gradients(
                    torch.autograd.grad(
                        final_loss,
                        exist_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    ),
                    exist_parameters,
                )
                intermediate_vectors = []
                for layer_index, layer_loss in enumerate(intermediate_losses):
                    vector = _flatten_gradients(
                        torch.autograd.grad(
                            layer_loss,
                            exist_parameters,
                            retain_graph=layer_index < len(intermediate_losses) - 1,
                            allow_unused=True,
                        ),
                        exist_parameters,
                    )
                    intermediate_vectors.append(vector)
                    item = _gradient_pair(final_grad, vector)
                    if item is not None:
                        gradient_by_layer[layer_index].append(item)
                aggregate_vector = torch.stack(intermediate_vectors).sum(dim=0)
                aggregate_item = _gradient_pair(final_grad, aggregate_vector)
                if aggregate_item is not None:
                    gradient_aggregate.append(aggregate_item)

        layer_cpu = [
            {
                "pred_x_rows": layer["pred_x_rows"].detach().float().cpu(),
                "range_norm": layer["range_norm"].detach().float().cpu(),
                "exist_logits": layer["exist_logits"].detach().float().cpu(),
            }
            for layer in layers
        ]
        update_cpu = [value.detach().float().cpu() for value in updates_by_layer]
        for batch_index, (image_id, target_cpu) in enumerate(
            zip(batch_ids, targets_cpu)
        ):
            if image_id not in pairs_by_image:
                continue
            seen_images.add(image_id)
            cached = cache_records[image_id]["stages"]["main"]
            final_x = layer_cpu[-1]["pred_x_rows"][batch_index]
            difference = (final_x - cached["pred_x_rows"].float()).abs()
            pred_abs_sum += float(difference.sum())
            pred_count += int(difference.numel())
            pred_abs_max = max(pred_abs_max, float(difference.max()))
            range_abs_max = max(
                range_abs_max,
                float(
                    (
                        layer_cpu[-1]["range_norm"][batch_index]
                        - cached["range_norm"].float()
                    ).abs().max()
                ),
            )
            assignments = [
                _assignment_map(matches[batch_index])
                for matches in matches_by_layer
            ]
            inverse_assignments = [
                {gt: candidate for candidate, gt in assignment.items()}
                for assignment in assignments
            ]
            qualities = []
            for layer in layer_cpu:
                quality, _candidate_valid, _gt_valid = _candidate_quality(
                    layer["pred_x_rows"][batch_index],
                    layer["range_norm"][batch_index],
                    target_cpu,
                    input_h=input_h,
                    line_width=float(args.line_width),
                    min_valid_rows=int(args.min_valid_rows),
                )
                qualities.append(quality)
            probabilities = [
                torch.softmax(layer["exist_logits"][batch_index], dim=-1)[:, 0]
                for layer in layer_cpu
            ]

            for pair in pairs_by_image[image_id]:
                good = int(pair["good_candidate"])
                wrong = int(pair["wrong_candidate"])
                intended = int(pair["training_gt"])
                good_assignment = [assignment.get(good, -1) for assignment in assignments]
                wrong_assignment = [assignment.get(wrong, -1) for assignment in assignments]
                intended_candidates = [
                    inverse.get(intended, -1) for inverse in inverse_assignments
                ]
                good_intended = [float(value == intended) for value in good_assignment]
                wrong_intended = [float(value == intended) for value in wrong_assignment]
                good_any = [float(value >= 0) for value in good_assignment]
                wrong_any = [float(value >= 0) for value in wrong_assignment]
                fresh_final_state = (
                    good_intended[-1],
                    wrong_intended[-1],
                    good_any[-1],
                    wrong_any[-1],
                )
                cached_final_state = (
                    float(pair["configured_good_intended"]),
                    float(pair["configured_wrong_intended"]),
                    float(pair["configured_good_any_positive"]),
                    float(pair["configured_wrong_any_positive"]),
                )
                pair_parity_rows.append(
                    {
                        "clip": str(pair["clip"]),
                        "curve_mean_abs": float(
                            torch.stack((difference[good], difference[wrong])).mean()
                        ),
                        "quality_good_abs": abs(
                            float(qualities[-1][good, intended])
                            - float(pair["target_quality_good"])
                        ),
                        "quality_wrong_abs": abs(
                            float(qualities[-1][wrong, intended])
                            - float(pair["target_quality_wrong"])
                        ),
                        "exist_good_abs": abs(
                            float(probabilities[-1][good])
                            - float(pair["exist_probability_good"])
                        ),
                        "exist_wrong_abs": abs(
                            float(probabilities[-1][wrong])
                            - float(pair["exist_probability_wrong"])
                        ),
                        "assignment_pair_state_same": float(
                            fresh_final_state == cached_final_state
                        ),
                    }
                )
                good_updates = [
                    float(update_cpu[layer][batch_index, good])
                    for layer in range(len(layers))
                ]
                wrong_updates = [
                    float(update_cpu[layer][batch_index, wrong])
                    for layer in range(len(layers))
                ]
                good_total = float(sum(good_updates))
                wrong_total = float(sum(wrong_updates))
                switches = sum(
                    int(before != after)
                    for before, after in zip(
                        intended_candidates[:-1], intended_candidates[1:]
                    )
                )
                trajectory_rows.append(
                    {
                        "fold": str(pair["fold"]),
                        "clip": str(pair["clip"]),
                        "image_id": image_id,
                        "thresholds": [float(value) for value in pair["thresholds"]],
                        "training_gt": intended,
                        "good_candidate": good,
                        "wrong_candidate": wrong,
                        "good_assignment_trace": good_assignment,
                        "wrong_assignment_trace": wrong_assignment,
                        "intended_candidate_trace": intended_candidates,
                        "good_intended_trace": good_intended,
                        "wrong_intended_trace": wrong_intended,
                        "good_any_trace": good_any,
                        "wrong_any_trace": wrong_any,
                        "good_exist_trace": [float(value[good]) for value in probabilities],
                        "wrong_exist_trace": [float(value[wrong]) for value in probabilities],
                        "good_quality_trace": [float(value[good, intended]) for value in qualities],
                        "wrong_quality_trace": [float(value[wrong, intended]) for value in qualities],
                        "good_exist_update_trace": good_updates,
                        "wrong_exist_update_trace": wrong_updates,
                        "good_final_any_positive": good_any[-1],
                        "good_all_layers_intended": float(all(value > 0.0 for value in good_intended)),
                        "good_all_layers_any_positive": float(all(value > 0.0 for value in good_any)),
                        "good_first_acquired_at_final": float(good_any[-1] > 0.0 and not any(good_any[:-1])),
                        "good_intermediate_has_negative": float(any(value == 0.0 for value in good_any[:-1])),
                        "intended_query_stable_all_layers": float(len(set(int(value) for value in intended_candidates)) == 1),
                        "intended_query_switches": float(switches),
                        "good_total_exist_update": good_total,
                        "wrong_total_exist_update": wrong_total,
                        "good_final_exist_update": good_updates[-1],
                        "wrong_final_exist_update": wrong_updates[-1],
                        "good_intermediate_exist_update": float(sum(good_updates[:-1])),
                        "wrong_intermediate_exist_update": float(sum(wrong_updates[:-1])),
                        "good_total_update_up": float(good_total > 0.0),
                        "good_total_update_down": float(good_total < 0.0),
                        "wrong_total_update_down": float(wrong_total < 0.0),
                        "wrong_total_update_up": float(wrong_total > 0.0),
                    }
                )

    if seen_images != set(pairs_by_image):
        raise ValueError(
            f"V37 dataloader saw {len(seen_images)} of {len(pairs_by_image)} pair images"
        )
    pred_mean_abs = pred_abs_sum / float(max(pred_count, 1))
    assignment_pair_state_same = _mean(
        row["assignment_pair_state_same"] for row in pair_parity_rows
    )
    quality_abs_mean = _mean(
        value
        for row in pair_parity_rows
        for value in (row["quality_good_abs"], row["quality_wrong_abs"])
    )
    # The V34 cache may have been produced on the preceding GPU.  BF16 replay
    # is therefore validated by aggregate geometry and decision stability,
    # not by one unstable row's maximum absolute coordinate difference.
    if (
        pred_mean_abs > 1.0
        or assignment_pair_state_same is None
        or assignment_pair_state_same < 0.95
        or quality_abs_mean is None
        or quality_abs_mean > 0.03
    ):
        raise ValueError(
            "fresh final proposal output failed cross-device replay stability: "
            f"mean_px={pred_mean_abs:.6f}, "
            f"assignment_state={assignment_pair_state_same}, "
            f"quality_abs={quality_abs_mean}"
        )
    if layer_count is None or not trajectory_rows:
        raise RuntimeError("V37 produced no trajectory rows")

    cohorts: dict[str, list[dict[str, Any]]] = {
        "overall": trajectory_rows,
        "fold_a": [row for row in trajectory_rows if row["fold"] == "a"],
        "fold_b": [row for row in trajectory_rows if row["fold"] == "b"],
        "iou_050": [row for row in trajectory_rows if 0.5 in row["thresholds"]],
        "iou_075": [row for row in trajectory_rows if 0.75 in row["thresholds"]],
    }
    final_positive_rows = [
        row for row in trajectory_rows if row["good_final_any_positive"] > 0.0
    ]
    summaries = {
        name: _cohort_summary(
            cohort,
            layers=layer_count,
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + index * 101,
        )
        for index, (name, cohort) in enumerate(cohorts.items())
    }
    final_positive_summary = _cohort_summary(
        final_positive_rows,
        layers=layer_count,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 997,
    )
    aggregate_gradient_summary = _gradient_summary(gradient_aggregate)
    gradient_payload = {
        "requested_images": len(gradient_set),
        "processed_images": processed_gradient_images,
        "aggregate_intermediate_vs_final": aggregate_gradient_summary,
        "individual_intermediate_vs_final": [
            _gradient_summary(values) for values in gradient_by_layer
        ],
    }
    verdict = classify_verdict(
        summaries["overall"],
        final_positive_summary,
        aggregate_gradient_summary,
    )
    payload = {
        "audit_version": AUDIT_VERSION,
        "experiment": "V37 Deep-Supervision Assignment Trajectory Audit",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": checkpoint_iteration,
        "target_cache": str(cache_path),
        "v36_json": str(v36_path),
        "settings": {
            "input_h": input_h,
            "eval_batch_size": int(args.eval_batch_size),
            "num_workers": int(args.num_workers),
            "amp_dtype": str(args.amp_dtype),
            "intermediate_layer_weights": configured_weights,
            "intermediate_strength": float(loss_cfg.get("lambda_intermediate", 0.0)),
            "layer_coefficients": coefficients,
            "no_lane_weight": no_lane_weight,
            "final_exist_loss_weight": final_exist_loss_weight,
            "intermediate_exist_loss_weight": intermediate_exist_loss_weight,
            "bootstrap_reps": int(args.bootstrap_reps),
            "seed": int(args.seed),
        },
        "counts": {
            "pairs": len(trajectory_rows),
            "images": len(seen_images),
            "clips": len({row["clip"] for row in trajectory_rows}),
            "decoder_layers": layer_count,
            "final_good_positive_pairs": len(final_positive_rows),
        },
        "parity": {
            "contract": "cross_device_bf16_stability",
            "replay_list": str(replay_list),
            "pred_x_mean_abs": pred_mean_abs,
            "pred_x_max_abs": pred_abs_max,
            "range_max_abs": range_abs_max,
            "pair_curve_mean_abs": _quantiles(
                row["curve_mean_abs"] for row in pair_parity_rows
            ),
            "quality_good_abs": _quantiles(
                row["quality_good_abs"] for row in pair_parity_rows
            ),
            "quality_wrong_abs": _quantiles(
                row["quality_wrong_abs"] for row in pair_parity_rows
            ),
            "exist_good_abs": _quantiles(
                row["exist_good_abs"] for row in pair_parity_rows
            ),
            "exist_wrong_abs": _quantiles(
                row["exist_wrong_abs"] for row in pair_parity_rows
            ),
            "assignment_pair_state_same": assignment_pair_state_same,
        },
        "summary": summaries,
        "final_good_positive_summary": final_positive_summary,
        "gradient_audit": gradient_payload,
        "verdict": verdict,
        "pairs": trajectory_rows,
    }
    json_path = output_dir / "v37_deep_supervision_trajectory.json"
    markdown_path = output_dir / "v37_deep_supervision_trajectory.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(payload, markdown_path)
    print(json.dumps({"verdict": verdict, "counts": payload["counts"], "parity": payload["parity"]}, indent=2))


if __name__ == "__main__":
    main()
