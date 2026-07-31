from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    diagnostic_iou_matrix,
    ensure_official_iou_cache,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    stage_scores,
    trace_postprocess,
    write_json,
)
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace official-IoU good and false-positive lane candidates through "
            "the independent Hungarian assignments of every decoder layer."
        )
    )
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--cache-dir", default="outputs/diagnostic_cache/rowref_precision")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--sample-strategy", choices=("uniform", "sequential"), default="uniform"
    )
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="none")
    parser.add_argument("--control-score-threshold", type=float, default=0.30)
    parser.add_argument("--control-quality-power", type=float, default=0.50)
    parser.add_argument("--candidate-score-threshold", type=float, default=0.20)
    parser.add_argument("--candidate-quality-power", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--near-min-iou", type=float, default=0.30)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


@dataclass
class CandidateTraceAccumulator:
    num_layers: int
    count: int = 0
    score_sum: float = 0.0
    exist_sum: float = 0.0
    quality_sum: float = 0.0
    best_iou_sum: float = 0.0
    matched_by_layer: list[int] = field(default_factory=list)
    ever_matched: int = 0
    never_matched: int = 0
    final_matched: int = 0
    earlier_only: int = 0
    acquired_at_final: int = 0
    matched_all_layers: int = 0
    matched_all_layers_same_gt: int = 0
    presence_flips: int = 0
    assignment_transitions: int = 0
    same_gt_transitions: int = 0
    changed_gt_transitions: int = 0
    statuses: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if not self.matched_by_layer:
            self.matched_by_layer = [0 for _ in range(int(self.num_layers))]

    def update(
        self,
        *,
        score: float,
        exist: float,
        quality: float,
        best_iou: float,
        status: str,
        assignments: list[int | None],
    ) -> None:
        if len(assignments) != int(self.num_layers):
            raise ValueError("assignment trace length does not match decoder layers")
        self.count += 1
        self.score_sum += float(score)
        self.exist_sum += float(exist)
        self.quality_sum += float(quality)
        self.best_iou_sum += float(best_iou)
        self.statuses[str(status)] += 1
        flags = [value is not None for value in assignments]
        for layer_index, matched in enumerate(flags):
            self.matched_by_layer[layer_index] += int(matched)
        self.ever_matched += int(any(flags))
        self.never_matched += int(not any(flags))
        self.final_matched += int(flags[-1])
        self.earlier_only += int(any(flags[:-1]) and not flags[-1])
        self.acquired_at_final += int(flags[-1] and not any(flags[:-1]))
        self.matched_all_layers += int(all(flags))
        self.matched_all_layers_same_gt += int(
            all(flags) and len(set(int(value) for value in assignments if value is not None)) == 1
        )
        for before, after in zip(assignments[:-1], assignments[1:]):
            self.presence_flips += int((before is None) != (after is None))
            if before is None or after is None:
                continue
            self.assignment_transitions += 1
            self.same_gt_transitions += int(int(before) == int(after))
            self.changed_gt_transitions += int(int(before) != int(after))

    def summary(self) -> dict[str, Any]:
        count = max(int(self.count), 1)
        return {
            "count": int(self.count),
            "mean_score": self.score_sum / count,
            "mean_exist_probability": self.exist_sum / count,
            "mean_quality_probability": self.quality_sum / count,
            "mean_best_official_iou": self.best_iou_sum / count,
            "postprocess_status": dict(self.statuses),
            "matched_fraction_by_decoder_layer": [
                value / count for value in self.matched_by_layer
            ],
            "ever_matched_fraction": self.ever_matched / count,
            "never_matched_fraction": self.never_matched / count,
            "final_layer_matched_fraction": self.final_matched / count,
            "matched_earlier_but_not_final_fraction": self.earlier_only / count,
            "first_acquired_at_final_fraction": self.acquired_at_final / count,
            "matched_all_layers_fraction": self.matched_all_layers / count,
            "matched_all_layers_same_gt_fraction": (
                self.matched_all_layers_same_gt / count
            ),
            "assignment_presence_flip_rate": self.presence_flips
            / max(count * max(self.num_layers - 1, 1), 1),
            "gt_identity_change_rate_when_consecutively_matched": (
                self.changed_gt_transitions / max(self.assignment_transitions, 1)
            ),
            "gt_identity_retention_rate_when_consecutively_matched": (
                self.same_gt_transitions / max(self.assignment_transitions, 1)
            ),
        }


def _prepare_config(
    path: str,
    *,
    dataset_root: str,
    eval_batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg = load_config(path)
    if dataset_root:
        cfg.setdefault("dataset", {})["root"] = dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(eval_batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(num_workers)
    if int(num_workers) == 0:
        cfg.setdefault("dataloader", {})["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg.setdefault("model", {})["require_pretrained_backbone"] = False
    cfg.setdefault("model", {}).setdefault("structured_query", {})[
        "intermediate_supervision"
    ] = True
    return cfg


def _resolve_stage(record: dict[str, Any]) -> str:
    for name in ("main", "final", "stage2", "coarse"):
        if name in record["stages"]:
            return name
    if len(record["stages"]) == 1:
        return next(iter(record["stages"]))
    raise KeyError(f"No final prediction stage for {record['image_id']}")


def _path_tail(value: Any, parts: int = 4) -> tuple[str, ...]:
    return tuple(Path(str(value)).parts[-int(parts) :])


def _assignment_map(match: dict[str, torch.Tensor]) -> dict[int, int]:
    return {
        int(pred): int(gt)
        for pred, gt in zip(
            match["pred_indices"].detach().cpu().tolist(),
            match["gt_indices"].detach().cpu().tolist(),
        )
    }


def _amp_context(device: torch.device, name: str):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(name)
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _update_scope(
    scopes: dict[str, CandidateTraceAccumulator],
    name: str,
    *,
    num_layers: int,
    score: float,
    exist: float,
    quality: float,
    best_iou: float,
    status: str,
    assignments: list[int | None],
) -> None:
    scopes.setdefault(name, CandidateTraceAccumulator(num_layers=num_layers)).update(
        score=score,
        exist=exist,
        quality=quality,
        best_iou=best_iou,
        status=status,
        assignments=assignments,
    )


def _fp_label(best_iou: float, gt_count: int, threshold: float, near_min: float) -> str:
    if gt_count == 0:
        return "empty_scene_fp"
    if float(best_iou) > float(threshold):
        return "duplicate_fp"
    if float(best_iou) >= float(near_min):
        return "near_miss_fp"
    return "background_fp"


def _provisional_diagnosis(threshold_summary: dict[str, Any]) -> dict[str, Any]:
    hidden = threshold_summary.get("scopes", {}).get(
        "oracle_missed_gt_rescue_candidate", {}
    )
    count = int(hidden.get("count", 0))
    if count == 0:
        return {
            "primary_signal": "no_hidden_oracle_candidate",
            "explanation": (
                "This sample contains no unselected oracle proposal associated with a GT "
                "missed by deployed predictions."
            ),
        }
    never = float(hidden.get("never_matched_fraction", 0.0))
    final = float(hidden.get("final_layer_matched_fraction", 0.0))
    dropped = float(hidden.get("matched_earlier_but_not_final_fraction", 0.0))
    if never >= 0.50:
        signal = "assignment_coverage_failure"
        explanation = (
            "Most hidden oracle-useful proposals never receive a positive Hungarian assignment; "
            "existence/quality supervision therefore suppresses usable geometry."
        )
    elif dropped >= 0.25 and final < 0.60:
        signal = "decoder_assignment_churn"
        explanation = (
            "Many hidden useful proposals are positive in an earlier decoder layer but lose their "
            "assignment before the final scoring heads."
        )
    elif final >= 0.60:
        signal = "score_learning_failure_after_positive_assignment"
        explanation = (
            "Most hidden useful proposals are positively matched at the final layer yet remain "
            "unselected, pointing to score-target, feature, or optimization misalignment."
        )
    else:
        signal = "mixed_assignment_and_score_failure"
        explanation = (
            "Hidden useful proposals show neither one dominant assignment failure nor one dominant "
            "post-assignment scoring failure."
        )
    return {
        "primary_signal": signal,
        "explanation": explanation,
        "oracle_rescue_candidates": count,
        "never_matched_fraction": never,
        "final_layer_matched_fraction": final,
        "matched_earlier_but_not_final_fraction": dropped,
    }


@torch.no_grad()
def _analyze_run(
    *,
    name: str,
    config_path: str,
    checkpoint_path: str,
    score_threshold: float,
    quality_power: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cache = load_or_collect_cache(
        config_path,
        checkpoint_path,
        split=args.split,
        dataset_root=args.dataset_root or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=bool(args.reuse_cache),
        max_batches=args.max_batches,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        sample_strategy=args.sample_strategy,
        desc=f"{name} final-output cache",
    )
    cache = ensure_official_iou_cache(
        cache,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
    )
    cfg = _prepare_config(
        config_path,
        dataset_root=args.dataset_root,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    device = torch.device(args.device)
    model = build_model(cfg)
    iteration = int(load_checkpoint(checkpoint_path, model, strict=False))
    model = model.to(device).eval()
    channels_last = bool(cfg.get("training", {}).get("channels_last", False))
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    head = model.structured_query_head
    if head is None:
        raise ValueError("assignment trace requires a structured query head")
    head.intermediate_supervision = True
    matcher = build_matcher(cfg)

    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
    if sampled_indices != cache["metadata"].get("sampled_dataset_indices"):
        raise ValueError("fresh loader and official-IoU cache use different sample indices")

    thresholds = tuple(float(value) for value in args.iou_thresholds)
    scopes_by_threshold: dict[float, dict[str, CandidateTraceAccumulator]] = {
        threshold: {} for threshold in thresholds
    }
    record_offset = 0
    layer_count: int | None = None
    parity_abs_sum = 0.0
    parity_count = 0
    parity_abs_max = 0.0

    total_batches = min(len(loader), int(args.max_batches)) if args.max_batches > 0 else len(loader)
    for batch_index, (images, targets_cpu, metas) in enumerate(
        tqdm(loader, total=total_batches, ncols=100, desc=f"{name} assignment trace")
    ):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
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
        with _amp_context(device, args.amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
            outputs = head(encoded["features"], inference_only=False)
        auxiliaries = outputs.get("aux_outputs")
        if not isinstance(auxiliaries, (list, tuple)) or not auxiliaries:
            raise ValueError("checkpoint/config did not expose intermediate decoder outputs")
        layers = [*auxiliaries, outputs]
        if layer_count is None:
            layer_count = len(layers)
        elif layer_count != len(layers):
            raise RuntimeError("decoder layer count changed between batches")
        matches_by_layer = matcher.match_many(tuple(layers), targets)

        for image_index, meta in enumerate(metas):
            if record_offset >= len(cache["records"]):
                raise RuntimeError("fresh loader contains more images than the candidate cache")
            record = cache["records"][record_offset]
            record_offset += 1
            if _path_tail(meta.get("image_path")) != _path_tail(record["meta"].get("image_path")):
                raise ValueError("fresh loader and candidate cache image order differ")
            stage_name = _resolve_stage(record)
            stage = record["stages"][stage_name]
            cached_x = stage["pred_x_rows"].float()
            fresh_x = outputs["pred_x_rows"][image_index].detach().float().cpu()
            if cached_x.shape != fresh_x.shape:
                raise ValueError("fresh final output and cached proposal count differ")
            difference = (cached_x - fresh_x).abs()
            parity_abs_sum += float(difference.sum())
            parity_count += int(difference.numel())
            parity_abs_max = max(parity_abs_max, float(difference.max()))

            assignments_by_layer = [
                _assignment_map(layer_matches[image_index])
                for layer_matches in matches_by_layer
            ]
            iou, _valid_gt, candidate_valid = diagnostic_iou_matrix(
                record,
                stage_name,
                use_official=True,
                input_h=int(cache["metadata"]["input_h"]),
                input_w=int(cache["metadata"]["input_w"]),
                line_width=args.line_width,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            scores = stage_scores(stage, quality_power=quality_power)
            exist = torch.softmax(stage["exist_logits"].float(), dim=-1)[..., 0]
            quality_logits = stage.get("quality_logits")
            quality = (
                torch.sigmoid(quality_logits.float())
                if isinstance(quality_logits, torch.Tensor)
                else torch.ones_like(exist)
            )
            trace = trace_postprocess(
                stage,
                input_h=int(cache["metadata"]["input_h"]),
                input_w=int(cache["metadata"]["input_w"]),
                score_thresh=score_threshold,
                quality_power=quality_power,
                min_valid_rows=args.min_valid_rows,
                nms_distance_thresh_px=args.nms_distance,
                nms_min_overlap_points=args.nms_min_overlap_points,
                top_k=args.top_k,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            selected = [int(value) for value in trace["selected_ids"]]
            selected_set = set(selected)
            gt_count = int(iou.shape[0])
            best_iou = (
                iou.max(dim=0).values
                if gt_count
                else torch.zeros(int(iou.shape[1]), dtype=torch.float32)
            )
            valid_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten().tolist()

            for threshold in thresholds:
                evaluator = evaluator_hungarian_assignment(iou, selected, threshold)
                deployed_tp = {int(value) for value in evaluator.proposal_ids}
                deployed_gt = {int(gt) for gt, _proposal in evaluator.pairs}
                oracle = cardinality_oracle_assignment(
                    iou,
                    threshold,
                    top_k=args.top_k,
                    candidate_valid=candidate_valid,
                )
                oracle_set = {int(value) for value in oracle.proposal_ids}
                oracle_gt_by_proposal = {
                    int(proposal): int(gt) for gt, proposal in oracle.pairs
                }
                scopes = scopes_by_threshold[threshold]
                for proposal_index in valid_ids:
                    value = float(best_iou[proposal_index]) if gt_count else 0.0
                    status = str(trace["status"][proposal_index])
                    assignment_trace = [
                        mapping.get(int(proposal_index)) for mapping in assignments_by_layer
                    ]
                    fields = {
                        "num_layers": int(layer_count),
                        "score": float(scores[proposal_index]),
                        "exist": float(exist[proposal_index]),
                        "quality": float(quality[proposal_index]),
                        "best_iou": value,
                        "status": status,
                        "assignments": assignment_trace,
                    }
                    _update_scope(scopes, "all_valid", **fields)
                    if proposal_index in deployed_tp:
                        _update_scope(scopes, "deployed_tp", **fields)
                    elif proposal_index in selected_set:
                        label = _fp_label(
                            value,
                            gt_count,
                            threshold,
                            args.near_min_iou,
                        )
                        _update_scope(scopes, f"deployed_{label}", **fields)
                    if proposal_index not in selected_set and value > threshold:
                        _update_scope(scopes, "matchable_unselected_any", **fields)
                        _update_scope(scopes, f"matchable_{status}", **fields)
                    if proposal_index in oracle_set and proposal_index not in selected_set:
                        _update_scope(scopes, "oracle_rescue_candidate", **fields)
                        if oracle_gt_by_proposal[proposal_index] not in deployed_gt:
                            _update_scope(
                                scopes,
                                "oracle_missed_gt_rescue_candidate",
                                **fields,
                            )

    if record_offset != len(cache["records"]):
        raise RuntimeError("fresh loader contains fewer images than the candidate cache")
    summaries: dict[str, Any] = {}
    for threshold, scopes in scopes_by_threshold.items():
        scope_summaries = {name: value.summary() for name, value in sorted(scopes.items())}
        row = {"scopes": scope_summaries}
        row["provisional_diagnosis"] = _provisional_diagnosis(row)
        summaries[f"{threshold:.2f}"] = row
    structured = cfg.get("model", {}).get("structured_query", {})
    return {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "checkpoint_iteration": iteration,
        "images": len(cache["records"]),
        "sampled_dataset_indices": sampled_indices,
        "score_threshold": float(score_threshold),
        "quality_power": float(quality_power),
        "decoder_layers": int(layer_count or 0),
        "num_instances": int(structured.get("num_instances", 0)),
        "matcher": cfg.get("matcher", {}),
        "fresh_vs_cached_final_output": {
            "mean_abs_x_difference_px": parity_abs_sum / max(parity_count, 1),
            "max_abs_x_difference_px": parity_abs_max,
        },
        "thresholds": summaries,
    }


def _paired_readout(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for threshold in control["thresholds"]:
        control_row = control["thresholds"][threshold]
        candidate_row = candidate["thresholds"][threshold]
        control_hidden = control_row["scopes"].get(
            "oracle_missed_gt_rescue_candidate", {}
        )
        candidate_hidden = candidate_row["scopes"].get(
            "oracle_missed_gt_rescue_candidate", {}
        )
        output[threshold] = {
            "control_primary_signal": control_row["provisional_diagnosis"]["primary_signal"],
            "candidate_primary_signal": candidate_row["provisional_diagnosis"]["primary_signal"],
            "control_hidden_oracle_candidates": int(control_hidden.get("count", 0)),
            "candidate_hidden_oracle_candidates": int(candidate_hidden.get("count", 0)),
            "control_hidden_final_match_fraction": float(
                control_hidden.get("final_layer_matched_fraction", 0.0)
            ),
            "candidate_hidden_final_match_fraction": float(
                candidate_hidden.get("final_layer_matched_fraction", 0.0)
            ),
            "control_hidden_never_match_fraction": float(
                control_hidden.get("never_matched_fraction", 0.0)
            ),
            "candidate_hidden_never_match_fraction": float(
                candidate_hidden.get("never_matched_fraction", 0.0)
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.near_min_iou) < min(args.iou_thresholds):
        raise ValueError("near_min_iou must be below every evaluated IoU threshold")
    control = _analyze_run(
        name="control",
        config_path=args.control_config,
        checkpoint_path=args.control_checkpoint,
        score_threshold=args.control_score_threshold,
        quality_power=args.control_quality_power,
        args=args,
    )
    candidate = _analyze_run(
        name="lambda_obj_0p5",
        config_path=args.candidate_config,
        checkpoint_path=args.candidate_checkpoint,
        score_threshold=args.candidate_score_threshold,
        quality_power=args.candidate_quality_power,
        args=args,
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "Oracle-rescue candidates are diagnostic upper-bound choices, not deployable "
            "predictions. Scopes intentionally overlap."
        ),
        "purpose": (
            "Separate missing Hungarian supervision, decoder-layer assignment churn, and "
            "post-assignment score-learning failure for official-IoU useful proposals."
        ),
        "control": control,
        "candidate": candidate,
        "paired_readout": _paired_readout(control, candidate),
    }
    write_json(args.output_json, payload)
    compact = {
        "paired_readout": payload["paired_readout"],
        "control": {
            key: value["provisional_diagnosis"]
            for key, value in control["thresholds"].items()
        },
        "candidate": {
            key: value["provisional_diagnosis"]
            for key, value in candidate["thresholds"].items()
        },
    }
    print(json.dumps(compact, indent=2))
    print(f"output_json: {Path(args.output_json)}")


if __name__ == "__main__":
    main()
