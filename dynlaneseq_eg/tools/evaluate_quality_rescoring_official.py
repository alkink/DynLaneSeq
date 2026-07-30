from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    official_proposal_gt_iou_matrix,
    trace_postprocess,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    QualityProbe,
    _amp_context,
    _average_precision,
    _frozen_outputs,
    _prepare_config,
    all_proposal_quality_targets,
    geometry_aware_features,
    query_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen quality-rescoring probes with row-space and "
            "official CULane raster IoU, before and after lane NMS."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-batches", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-strategy", choices=("uniform", "sequential"), default="uniform")
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quality-power", type=float, default=0.5)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _new_counts() -> dict[str, Any]:
    return {
        "gt": 0,
        "selected": 0,
        "hits": {0.5: 0, 0.7: 0},
        "scores": [],
        "labels": {0.5: [], 0.7: []},
    }


def _update_counts(
    counts: dict[str, Any],
    iou: torch.Tensor,
    selected_ids: Iterable[int],
    *,
    scores: torch.Tensor | None = None,
    candidate_valid: torch.Tensor | None = None,
) -> None:
    selected = [int(index) for index in selected_ids]
    counts["gt"] += int(iou.shape[0])
    counts["selected"] += len(selected)
    for threshold in (0.5, 0.7):
        assignment = evaluator_hungarian_assignment(
            iou,
            selected,
            threshold=threshold,
        )
        counts["hits"][threshold] += int(assignment.hit_count)

    if scores is None:
        return
    if candidate_valid is None:
        candidate_valid = torch.ones(
            int(iou.shape[1]),
            dtype=torch.bool,
            device=iou.device,
        )
    valid_ids = [
        index
        for index in range(int(iou.shape[1]))
        if bool(candidate_valid[index])
    ]
    best = (
        iou.max(dim=0).values
        if int(iou.shape[0]) > 0
        else iou.new_zeros(int(iou.shape[1]))
    )
    counts["scores"].extend(float(scores[index]) for index in valid_ids)
    for threshold in (0.5, 0.7):
        counts["labels"][threshold].extend(
            int(float(best[index]) > threshold)
            for index in valid_ids
        )


def _finish_counts(counts: dict[str, Any]) -> dict[str, Any]:
    gt = int(counts["gt"])
    selected = int(counts["selected"])
    output: dict[str, Any] = {
        "gt_lanes": gt,
        "selected_predictions": selected,
    }
    for threshold in (0.5, 0.7):
        hits = int(counts["hits"][threshold])
        precision = float(hits) / float(max(selected, 1))
        recall = float(hits) / float(max(gt, 1))
        f1 = (
            2.0 * precision * recall / max(precision + recall, 1e-12)
        )
        suffix = f"{int(round(threshold * 100)):03d}"
        output[f"tp_{suffix}"] = hits
        output[f"precision_{suffix}"] = precision
        output[f"recall_{suffix}"] = recall
        output[f"f1_{suffix}"] = f1
        output[f"candidate_ap_{suffix}"] = _average_precision(
            counts["scores"],
            counts["labels"][threshold],
        )
    return output


def _topk_ids(
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    top_k: int,
) -> list[int]:
    valid = [
        index
        for index in range(int(scores.shape[0]))
        if bool(candidate_valid[index])
    ]
    valid.sort(key=lambda index: float(scores[index]), reverse=True)
    return valid[: int(top_k)]


def _stage_for_image(
    outputs: dict[str, torch.Tensor],
    batch_index: int,
) -> dict[str, torch.Tensor]:
    fields = (
        "pred_x_rows",
        "range_norm",
        "exist_logits",
        "quality_logits",
        "row_visibility_logits",
    )
    return {
        name: outputs[name][batch_index].detach().cpu()
        for name in fields
        if isinstance(outputs.get(name), torch.Tensor)
    }


def _load_probe(
    path: str,
    query_dim: int,
    geometry_dim: int,
    device: torch.device,
) -> tuple[QualityProbe, QualityProbe, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if int(payload["query_feature_dim"]) != int(query_dim):
        raise ValueError(
            f"query feature mismatch: {payload['query_feature_dim']} vs {query_dim}"
        )
    if int(payload["geometry_feature_dim"]) != int(geometry_dim):
        raise ValueError(
            "geometry feature mismatch: "
            f"{payload['geometry_feature_dim']} vs {geometry_dim}"
        )
    query_probe = QualityProbe(query_dim).to(device)
    geometry_probe = QualityProbe(geometry_dim).to(device)
    query_probe.load_state_dict(payload["query_probe"], strict=True)
    geometry_probe.load_state_dict(payload["geometry_probe"], strict=True)
    return query_probe.eval(), geometry_probe.eval(), payload


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    cfg = _prepare_config(
        args.config,
        dataset_root=args.dataset_root,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    model_cfg = cfg.get("model", {})
    input_h = int(model_cfg.get("input_h", 288))
    input_w = int(model_cfg.get("input_w", 800))
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if model.structured_query_head is None:
        raise ValueError("quality rescoring requires a structured query head")
    model.structured_query_head.intermediate_supervision = False
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    loader = build_dataloader(cfg, split="val", training=False)
    loader, eval_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.eval_max_batches,
        num_workers=args.num_workers,
    )

    first_images, _first_targets, _first_metas = next(iter(loader))
    if channels_last:
        first_images = first_images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    else:
        first_images = first_images.to(device, non_blocking=True)
    with _amp_context(device, amp_dtype):
        first_outputs = _frozen_outputs(model, first_images)
    query_dim = int(query_features(first_outputs).shape[-1])
    geometry_dim = int(
        geometry_aware_features(first_outputs, input_w=input_w).shape[-1]
    )
    query_probe, geometry_probe, probe_payload = _load_probe(
        args.probe_checkpoint,
        query_dim,
        geometry_dim,
        device,
    )
    del first_images, first_outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    strategy_names = (
        "current_exist_quality",
        "exist_only",
        "query_probe",
        "geometry_probe",
        "query_quality_only",
        "geometry_quality_only",
        "query_quality_dominant",
        "geometry_quality_dominant",
        "query_equal_power",
        "geometry_equal_power",
    )
    spaces = ("row_space", "official_raster")
    selection_modes = ("raw_top4", "nms_top4")
    stats = {
        space: {
            mode: {name: _new_counts() for name in strategy_names}
            for mode in selection_modes
        }
        for space in spaces
    }
    oracle_stats = {
        space: {threshold: _new_counts() for threshold in (0.5, 0.7)}
        for space in spaces
    }

    for images, targets, metas in tqdm(
        loader,
        desc="official quality-rescoring eval",
        ncols=80,
    ):
        if channels_last:
            images = images.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images.to(device, non_blocking=True)
        with _amp_context(device, amp_dtype):
            outputs = _frozen_outputs(model, images)
        _quality_targets, row_matrices = all_proposal_quality_targets(
            outputs,
            targets,
            input_h=input_h,
            line_width=args.line_width,
        )
        query_logits = query_probe(query_features(outputs))
        geometry_logits = geometry_probe(
            geometry_aware_features(outputs, input_w=input_w)
        )
        existence = torch.softmax(
            outputs["exist_logits"].float(),
            dim=-1,
        )[..., 0]
        old_quality = torch.sigmoid(
            outputs["quality_logits"].float()
        ).clamp_min(1e-6)
        query_quality = torch.sigmoid(query_logits).clamp_min(1e-6)
        geometry_quality = torch.sigmoid(geometry_logits).clamp_min(1e-6)
        scores_by_strategy = {
            "current_exist_quality": (
                existence * old_quality.pow(float(args.quality_power))
            ),
            "exist_only": existence,
            "query_probe": (
                existence
                * query_quality.pow(float(args.quality_power))
            ),
            "geometry_probe": (
                existence
                * geometry_quality.pow(float(args.quality_power))
            ),
            # Exploratory fusion controls. These do not tune a benchmark
            # result; they distinguish missing quality information from the
            # fixed existence-dominant fusion used by the detector.
            "query_quality_only": query_quality,
            "geometry_quality_only": geometry_quality,
            "query_quality_dominant": existence.sqrt() * query_quality,
            "geometry_quality_dominant": existence.sqrt() * geometry_quality,
            "query_equal_power": existence * query_quality,
            "geometry_equal_power": existence * geometry_quality,
        }

        for batch_index, (target, meta) in enumerate(zip(targets, metas)):
            stage = _stage_for_image(outputs, batch_index)
            record = {"stages": {"main": stage}, "meta": meta}
            official_matrix, candidate_valid = official_proposal_gt_iou_matrix(
                record,
                "main",
                line_width=args.line_width,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=0.0,
            )
            matrices = {
                "row_space": row_matrices[batch_index].detach().cpu(),
                "official_raster": official_matrix,
            }
            valid = candidate_valid.bool()
            for strategy_name, batch_scores in scores_by_strategy.items():
                scores = batch_scores[batch_index].detach().cpu()
                raw_ids = _topk_ids(scores, valid, args.top_k)
                score_override = {
                    index: float(scores[index])
                    for index in range(int(scores.shape[0]))
                }
                trace = trace_postprocess(
                    stage,
                    input_h=input_h,
                    input_w=input_w,
                    score_thresh=-1.0,
                    quality_power=0.0,
                    min_valid_rows=args.min_valid_rows,
                    nms_distance_thresh_px=args.nms_distance,
                    nms_min_overlap_points=args.nms_min_overlap_points,
                    top_k=args.top_k,
                    row_visibility_thresh=0.0,
                    score_override=score_override,
                )
                nms_ids = trace["selected_ids"]
                for space, matrix in matrices.items():
                    _update_counts(
                        stats[space]["raw_top4"][strategy_name],
                        matrix,
                        raw_ids,
                        scores=scores,
                        candidate_valid=valid,
                    )
                    _update_counts(
                        stats[space]["nms_top4"][strategy_name],
                        matrix,
                        nms_ids,
                        scores=scores,
                        candidate_valid=valid,
                    )

            for space, matrix in matrices.items():
                for threshold in (0.5, 0.7):
                    assignment = cardinality_oracle_assignment(
                        matrix,
                        threshold=threshold,
                        top_k=args.top_k,
                        candidate_valid=valid,
                    )
                    _update_counts(
                        oracle_stats[space][threshold],
                        matrix,
                        assignment.proposal_ids,
                    )

    finished = {
        space: {
            mode: {
                name: _finish_counts(row)
                for name, row in strategies.items()
            }
            for mode, strategies in modes.items()
        }
        for space, modes in stats.items()
    }
    finished_oracle = {
        space: {
            f"{threshold:.2f}": _finish_counts(row)
            for threshold, row in thresholds.items()
        }
        for space, thresholds in oracle_stats.items()
    }
    official_nms = finished["official_raster"]["nms_top4"]
    baseline = official_nms["current_exist_quality"]
    gains = {}
    for name in ("query_probe", "geometry_probe"):
        row = official_nms[name]
        gains[name] = {
            "recall_050_points": 100.0
            * (float(row["recall_050"]) - float(baseline["recall_050"])),
            "recall_070_points": 100.0
            * (float(row["recall_070"]) - float(baseline["recall_070"])),
            "f1_050_points": 100.0
            * (float(row["f1_050"]) - float(baseline["f1_050"])),
            "f1_070_points": 100.0
            * (float(row["f1_070"]) - float(baseline["f1_070"])),
        }

    payload = {
        "diagnostic_only": True,
        "warning": (
            "Frozen ranking diagnostic with no score threshold. Official "
            "raster results include the configured lane NMS and Top-K but "
            "are not a full validation-selected benchmark protocol. "
            "Quality-only and alternative-power rows are explicitly "
            "exploratory fusion controls."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "probe_checkpoint": args.probe_checkpoint,
        "probe_source_checkpoint": probe_payload.get("source_checkpoint"),
        "probe_source_iteration": int(probe_payload.get("source_iteration", -1)),
        "probe_train_steps": int(probe_payload.get("train_steps", -1)),
        "probe_target_mode": probe_payload.get("target_mode", "legacy_unspecified"),
        "eval_images": len(eval_indices),
        "eval_dataset_indices": eval_indices,
        "settings": {
            "line_width": float(args.line_width),
            "top_k": int(args.top_k),
            "quality_power": float(args.quality_power),
            "nms_distance": float(args.nms_distance),
            "nms_min_overlap_points": int(args.nms_min_overlap_points),
            "min_valid_rows": int(args.min_valid_rows),
            "score_threshold": None,
        },
        "results": finished,
        "oracle_top4": finished_oracle,
        "official_nms_gains": gains,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
