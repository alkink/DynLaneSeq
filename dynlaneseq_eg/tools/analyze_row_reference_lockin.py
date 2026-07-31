from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    proposal_gt_iou_matrix,
    stage_scores,
    trace_postprocess,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import soft_expected_x
from dynlaneseq_eg.modeling.structured_queries import (
    ReferenceGuidedRowLayer,
    StructuredLaneQueryHead,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


CONDITIONS = (
    "normal",
    "fixed_initial",
    "blend_initial_0p5",
    "wide_offsets_2x",
    "global_rescue_mid",
    "global_rescue_mid_wrong_image",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a trained row-reference decoder is locked to an "
            "incorrect initial/local curve. All counterfactuals are inference-"
            "only and leave the checkpoint untouched."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument(
        "--sample-strategy", choices=("uniform", "sequential"), default="uniform"
    )
    parser.add_argument(
        "--amp-dtype", choices=("none", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quality-power", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--rescue-layer", type=int, default=3)
    parser.add_argument("--rescue-prior-sigma-px", type=float, default=240.0)
    parser.add_argument("--rescue-prior-strength", type=float, default=1.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, name: str):
    if device.type != "cuda" or name == "none":
        return nullcontext()
    dtype = torch.float16 if name == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _global_reacquire_reference(
    head: StructuredLaneQueryHead,
    row_tokens: torch.Tensor,
    row_value_features: torch.Tensor,
    current_reference: torch.Tensor,
    *,
    prior_sigma_px: float,
    prior_strength: float,
) -> torch.Tensor:
    """Re-run global row evidence around the current curve.

    This is deliberately a diagnostic counterfactual rather than a new model
    path. It reuses only modules already trained by the checkpoint and changes
    no parameters.
    """

    if (
        head.reference_query_norm is None
        or head.reference_query is None
        or head.reference_key is None
        or head.reference_logit_scale is None
    ):
        raise RuntimeError("global rescue requires an enabled row-reference head")
    row_key_features = row_value_features
    if head.x_tokens is not None:
        x_pos = head.x_tokens.weight.to(
            device=row_value_features.device,
            dtype=row_value_features.dtype,
        ).view(1, 1, int(row_value_features.shape[2]), int(row_value_features.shape[3]))
        row_key_features = row_key_features + x_pos
    query = F.normalize(
        head.reference_query(head.reference_query_norm(row_tokens)),
        dim=-1,
        eps=1e-6,
    )
    key = F.normalize(head.reference_key(row_key_features), dim=-1, eps=1e-6)
    visual_logits = torch.einsum("bnrc,brxc->bnrx", query, key)
    visual_logits = visual_logits * head.reference_logit_scale.exp().clamp(
        min=1.0,
        max=100.0,
    )
    prior = head._reference_prior_logits(
        current_reference.to(dtype=visual_logits.dtype),
        x_bins=int(row_value_features.shape[2]),
        sigma_px=float(prior_sigma_px),
        strength=float(prior_strength),
    )
    full_logits = head._resize_reference_logits(visual_logits + prior)
    return soft_expected_x(
        full_logits,
        input_w=head.input_w,
        x_bins=head.x_bins,
    ).to(dtype=current_reference.dtype)


@contextmanager
def _reference_counterfactual(
    head: StructuredLaneQueryHead,
    condition: str,
    *,
    rescue_layer: int,
    rescue_prior_sigma_px: float,
    rescue_prior_strength: float,
) -> Iterator[None]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown row-reference condition: {condition}")
    layers = list(head.layers)
    if not layers or not all(isinstance(layer, ReferenceGuidedRowLayer) for layer in layers):
        raise TypeError("lock-in audit requires ReferenceGuidedRowLayer blocks")
    if not 1 <= int(rescue_layer) <= len(layers):
        raise ValueError("rescue_layer must address an existing one-indexed decoder layer")

    offset_backups: list[torch.Tensor] = []
    handles: list[Any] = []
    state: dict[str, torch.Tensor] = {}
    try:
        if condition == "wide_offsets_2x":
            for layer in layers:
                offset_backups.append(layer.offsets_px.detach().clone())
                layer.offsets_px.mul_(2.0)
        elif condition != "normal":
            for layer_index, layer in enumerate(layers):
                def hook(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    index: int = layer_index,
                ):
                    if len(args) < 3:
                        raise RuntimeError("unexpected ReferenceGuidedRowLayer call contract")
                    row_tokens, row_value, reference = args[:3]
                    if index == 0:
                        state["initial"] = reference.detach().clone()
                    replacement = reference
                    if index > 0 and condition == "fixed_initial":
                        replacement = state["initial"]
                    elif index > 0 and condition == "blend_initial_0p5":
                        replacement = 0.5 * reference + 0.5 * state["initial"]
                    elif index == int(rescue_layer) - 1 and condition.startswith("global_rescue_mid"):
                        rescue_value = row_value
                        if condition.endswith("wrong_image"):
                            if int(row_value.shape[0]) < 2:
                                raise ValueError("wrong-image rescue requires eval_batch_size >= 2")
                            rescue_value = torch.roll(row_value, shifts=1, dims=0)
                        replacement = _global_reacquire_reference(
                            head,
                            row_tokens,
                            rescue_value,
                            reference,
                            prior_sigma_px=rescue_prior_sigma_px,
                            prior_strength=rescue_prior_strength,
                        )
                    return (row_tokens, row_value, replacement, *args[3:]), kwargs

                handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        if offset_backups:
            for layer, backup in zip(layers, offset_backups):
                layer.offsets_px.copy_(backup)


def _new_counter() -> dict[str, float | int]:
    return {
        "gt": 0,
        "all_hits": 0,
        "oracle_topk_hits": 0,
        "model_topk_hits": 0,
        "deployed_hits": 0,
        "deployed_predictions": 0,
        "best_iou_sum": 0.0,
    }


def _hit_set(assignment) -> set[int]:
    return {int(gt_index) for gt_index, _proposal_index in assignment.pairs}


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()
    head = model.structured_query_head
    if not isinstance(head, StructuredLaneQueryHead) or not head.row_reference_enabled:
        raise TypeError("checkpoint must use StructuredLaneQueryHead row-reference mode")

    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.max_batches,
        num_workers=int(args.num_workers),
    )
    input_h = int(cfg["model"].get("input_h", 640))
    input_w = int(cfg["model"].get("input_w", 1600))
    thresholds = tuple(float(value) for value in args.iou_thresholds)
    counters = {
        condition: {threshold: _new_counter() for threshold in thresholds}
        for condition in CONDITIONS
    }
    paired = {
        condition: {
            threshold: {
                "normal_only": 0,
                "condition_only": 0,
                "common_hit": 0,
                "common_miss": 0,
                "union_hits": 0,
                "gt": 0,
            }
            for threshold in thresholds
        }
        for condition in CONDITIONS
        if condition != "normal"
    }
    images_seen = 0

    for images, targets, _metas in tqdm(loader, ncols=100, desc="row-reference lock-in audit"):
        images = images.to(device, non_blocking=True)
        with _amp_context(device, args.amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
        condition_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for condition in CONDITIONS:
            with _reference_counterfactual(
                head,
                condition,
                rescue_layer=args.rescue_layer,
                rescue_prior_sigma_px=args.rescue_prior_sigma_px,
                rescue_prior_strength=args.rescue_prior_strength,
            ):
                with _amp_context(device, args.amp_dtype):
                    output = head(encoded["features"], inference_only=True)
            condition_outputs[condition] = {
                key: value.detach().float().cpu()
                for key, value in output.items()
                if isinstance(value, torch.Tensor)
            }

        for batch_index, target in enumerate(targets):
            matrices: dict[str, torch.Tensor] = {}
            valid_masks: dict[str, torch.Tensor] = {}
            all_assignments: dict[tuple[str, float], Any] = {}
            for condition, output in condition_outputs.items():
                stage = {key: value[batch_index] for key, value in output.items()}
                iou, _valid_gt, candidate_valid = proposal_gt_iou_matrix(
                    stage,
                    target,
                    input_h=input_h,
                    input_w=input_w,
                    line_width=args.line_width,
                    min_valid_rows=args.min_valid_rows,
                )
                matrices[condition] = iou
                valid_masks[condition] = candidate_valid
                scores = stage_scores(stage, quality_power=args.quality_power)
                valid_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten().tolist()
                ranked_ids = sorted(valid_ids, key=lambda index: float(scores[index]), reverse=True)
                model_ids = ranked_ids[: int(args.top_k)]
                trace = trace_postprocess(
                    stage,
                    input_h=input_h,
                    input_w=input_w,
                    score_thresh=args.score_threshold,
                    quality_power=args.quality_power,
                    min_valid_rows=args.min_valid_rows,
                    nms_distance_thresh_px=args.nms_distance,
                    nms_min_overlap_points=args.nms_min_overlap_points,
                    top_k=args.top_k,
                )
                selected_ids = list(trace["selected_ids"])
                gt_count = int(iou.shape[0])
                for threshold in thresholds:
                    all_assignment = cardinality_oracle_assignment(
                        iou,
                        threshold,
                        top_k=max(int(iou.shape[1]), 1),
                        candidate_valid=candidate_valid,
                    )
                    all_assignments[(condition, threshold)] = all_assignment
                    oracle_topk = cardinality_oracle_assignment(
                        iou,
                        threshold,
                        top_k=args.top_k,
                        candidate_valid=candidate_valid,
                    )
                    model_assignment = evaluator_hungarian_assignment(iou, model_ids, threshold)
                    deployed = evaluator_hungarian_assignment(iou, selected_ids, threshold)
                    counter = counters[condition][threshold]
                    counter["gt"] += gt_count
                    counter["all_hits"] += int(all_assignment.hit_count)
                    counter["oracle_topk_hits"] += int(oracle_topk.hit_count)
                    counter["model_topk_hits"] += int(model_assignment.hit_count)
                    counter["deployed_hits"] += int(deployed.hit_count)
                    counter["deployed_predictions"] += len(selected_ids)
                    if gt_count:
                        counter["best_iou_sum"] += float(iou.max(dim=1).values.sum())

            for condition in paired:
                normal_iou = matrices["normal"]
                condition_iou = matrices[condition]
                union_iou = torch.cat((normal_iou, condition_iou), dim=1)
                union_valid = torch.cat((valid_masks["normal"], valid_masks[condition]))
                for threshold in thresholds:
                    normal_hits = _hit_set(all_assignments[("normal", threshold)])
                    condition_hits = _hit_set(all_assignments[(condition, threshold)])
                    union = cardinality_oracle_assignment(
                        union_iou,
                        threshold,
                        top_k=max(int(union_iou.shape[1]), 1),
                        candidate_valid=union_valid,
                    )
                    gt_count = int(union_iou.shape[0])
                    row = paired[condition][threshold]
                    row["normal_only"] += len(normal_hits - condition_hits)
                    row["condition_only"] += len(condition_hits - normal_hits)
                    row["common_hit"] += len(normal_hits & condition_hits)
                    row["common_miss"] += gt_count - len(normal_hits | condition_hits)
                    row["union_hits"] += int(union.hit_count)
                    row["gt"] += gt_count
            images_seen += 1

    summaries: dict[str, Any] = {}
    for condition, by_threshold in counters.items():
        summaries[condition] = {}
        for threshold, counter in by_threshold.items():
            gt = max(int(counter["gt"]), 1)
            predictions = max(int(counter["deployed_predictions"]), 1)
            deployed_hits = int(counter["deployed_hits"])
            summaries[condition][f"{threshold:.2f}"] = {
                **counter,
                "all_candidate_recall": float(counter["all_hits"]) / gt,
                "oracle_topk_recall": float(counter["oracle_topk_hits"]) / gt,
                "model_topk_recall": float(counter["model_topk_hits"]) / gt,
                "deployed_recall": deployed_hits / gt,
                "deployed_precision": deployed_hits / predictions,
                "mean_best_iou": float(counter["best_iou_sum"]) / gt,
            }

    paired_summaries: dict[str, Any] = {}
    for condition, by_threshold in paired.items():
        paired_summaries[condition] = {}
        for threshold, row in by_threshold.items():
            gt = max(int(row["gt"]), 1)
            normal_hits = int(counters["normal"][threshold]["all_hits"])
            paired_summaries[condition][f"{threshold:.2f}"] = {
                **row,
                "union_recall": float(row["union_hits"]) / gt,
                "union_gain_over_normal_points": 100.0
                * (float(row["union_hits"] - normal_hits) / gt),
                "condition_minus_normal_hits": int(row["condition_only"])
                - int(row["normal_only"]),
            }

    rescue_key = f"{thresholds[0]:.2f}"
    rescue_gain = paired_summaries["global_rescue_mid"][rescue_key][
        "union_gain_over_normal_points"
    ]
    wrong_gain = paired_summaries["global_rescue_mid_wrong_image"][rescue_key][
        "union_gain_over_normal_points"
    ]
    wide_gain = paired_summaries["wide_offsets_2x"][rescue_key][
        "union_gain_over_normal_points"
    ]
    payload = {
        "diagnostic_only": True,
        "warning": (
            "Counterfactual references are inference-only diagnostics, not deployable "
            "benchmark predictions or a trained architecture."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "images": images_seen,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "settings": {
            "conditions": list(CONDITIONS),
            "rescue_layer": args.rescue_layer,
            "rescue_prior_sigma_px": args.rescue_prior_sigma_px,
            "rescue_prior_strength": args.rescue_prior_strength,
            "quality_power": args.quality_power,
            "score_threshold": args.score_threshold,
            "top_k": args.top_k,
            "iou_thresholds": list(thresholds),
            "iou_space": "input_row_strip",
        },
        "conditions": summaries,
        "paired_vs_normal": paired_summaries,
        "gate": {
            "reference_lockin_signal": bool(
                max(float(rescue_gain - wrong_gain), float(wide_gain)) >= 1.0
            ),
            "correct_global_rescue_union_gain_points": rescue_gain,
            "wrong_image_global_rescue_union_gain_points": wrong_gain,
            "correct_minus_wrong_rescue_gain_points": rescue_gain - wrong_gain,
            "wide_offset_union_gain_points": wide_gain,
            "definition": (
                "Positive when a correct-image mid-decoder global rescue exceeds its "
                "wrong-image control by at least one recall point, or a 2x corridor "
                "adds at least one oracle-union recall point at the first IoU threshold."
            ),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
