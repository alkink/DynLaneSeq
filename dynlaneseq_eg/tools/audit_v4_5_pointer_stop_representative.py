from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.losses.loss_s0 import build_pointer_cluster_soft_targets
from dynlaneseq_eg.losses.range_aware_iou import (
    pairwise_range_aware_row_strip_iou,
)
from dynlaneseq_eg.tools.audit_v4_3_pointer_full_validation import (
    _official_for_record,
    _selected_pointer_ids,
    _slice_stage,
    _stage_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V4.5 STOP decisions and representative selection without "
            "training, thresholds, or NMS."
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
    parser.add_argument("--metric-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
    )
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.50, 0.75]
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}

    def at(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return ordered[min(max(int(index), 0), len(ordered) - 1)]

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": at(0.50),
        "p90": at(0.90),
        "max": ordered[-1],
    }


def _metric(tp: int, predictions: int, gt: int) -> dict[str, float | int]:
    precision = _ratio(tp, predictions)
    recall = _ratio(tp, gt)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": int(tp),
        "fp": int(predictions - tp),
        "fn": int(gt - tp),
        "predictions": int(predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def prefix_extension_oracle_hit_count(
    iou_matrix: torch.Tensor,
    fixed_ids: Iterable[int],
    candidate_valid: torch.Tensor,
    *,
    threshold: float,
    top_k: int,
) -> int:
    """Exact maximum matching after preserving an already emitted prefix.

    Fixed candidates consume deployment slots even when they are false
    positives.  Additional candidates cost one slot.  A small bit-mask DP is
    exact because CULane has at most a handful of GT lanes per image.
    """

    iou = iou_matrix.float().cpu()
    valid = candidate_valid.bool().cpu()
    fixed = tuple(dict.fromkeys(int(value) for value in fixed_ids))
    remaining_slots = max(int(top_k) - len(fixed), 0)
    gt_count, candidates = iou.shape
    if gt_count == 0 or candidates == 0:
        return 0

    fixed_set = set(fixed)
    proposal_rows: list[tuple[int, int]] = []
    for candidate in range(candidates):
        if candidate not in fixed_set and not bool(valid[candidate]):
            continue
        edge_mask = 0
        for gt in range(gt_count):
            if float(iou[gt, candidate]) > float(threshold):
                edge_mask |= 1 << gt
        if edge_mask:
            proposal_rows.append((edge_mask, 0 if candidate in fixed_set else 1))

    reachable: set[tuple[int, int]] = {(0, 0)}
    for edge_mask, cost in proposal_rows:
        updated = set(reachable)
        for covered, used in reachable:
            next_used = used + cost
            if next_used > remaining_slots:
                continue
            available_gt = edge_mask & ~covered
            while available_gt:
                bit = available_gt & -available_gt
                updated.add((covered | bit, next_used))
                available_gt -= bit
        reachable = updated
    return max(bin(mask).count("1") for mask, _used in reachable)


def _first_stop_margin(
    indices: torch.Tensor,
    logits: torch.Tensor,
    candidates: int,
) -> tuple[int, float] | None:
    values = indices.flatten().tolist()
    stop_step = next((step for step, value in enumerate(values) if int(value) < 0), None)
    if stop_step is None:
        return None
    row = logits[stop_step].float()
    return int(stop_step), float(row[candidates] - row[:candidates].max())


def _margin_bin(margin: float) -> str:
    if margin < 0.25:
        return "[0,0.25)"
    if margin < 0.50:
        return "[0.25,0.50)"
    if margin < 1.0:
        return "[0.50,1.0)"
    if margin < 2.0:
        return "[1.0,2.0)"
    return "[2.0,inf)"


def _replacement_sequence(
    official_iou: torch.Tensor,
    row_quality: torch.Tensor,
    candidate_valid: torch.Tensor,
    selected_ids: list[int],
    unary_logits: torch.Tensor,
    pointer_logits: torch.Tensor,
    *,
    mode: str,
    representable_min: float,
    assignment_threshold: float = 0.50,
) -> list[int]:
    """Replace candidates inside the GT clusters reached by the pointer."""

    if not selected_ids or official_iou.shape[0] == 0:
        return list(selected_ids)
    base = evaluator_hungarian_assignment(
        official_iou,
        selected_ids,
        float(assignment_threshold),
    )
    gt_by_candidate = {candidate: gt for gt, candidate in base.pairs}
    rows = [
        (slot, candidate, gt_by_candidate[candidate])
        for slot, candidate in enumerate(selected_ids)
        if candidate in gt_by_candidate
    ]
    if not rows:
        return list(selected_ids)

    candidates = int(official_iou.shape[1])
    valid = candidate_valid.bool().cpu()
    q = row_quality.float().cpu()
    best_gt = q.argmax(dim=1) if q.shape[1] else torch.full((candidates,), -1)
    cost = torch.full((len(rows), candidates), 1e9, dtype=torch.float64)
    unmatched_reserved = {
        candidate for candidate in selected_ids if candidate not in gt_by_candidate
    }
    for row_index, (slot, current, gt) in enumerate(rows):
        eligible = (
            valid
            & (best_gt == int(gt))
            & (q[:, gt] > float(representable_min))
        )
        # A slot-local replacement may not steal a candidate already emitted
        # by another slot.  This keeps both count and the untouched portion of
        # the pointer sequence invariant.
        for selected_candidate in selected_ids:
            if selected_candidate != current:
                eligible[selected_candidate] = False
        for reserved in unmatched_reserved:
            eligible[reserved] = False
        eligible[current] = True
        if mode == "unary":
            score = unary_logits.float().cpu()
        elif mode == "step_logit":
            score = pointer_logits[slot, :candidates].float().cpu()
        elif mode == "row_strip":
            score = q[:, gt]
        elif mode == "official":
            score = official_iou[gt].float().cpu()
        else:
            raise ValueError(f"unknown replacement mode: {mode}")
        cost[row_index, eligible] = -score[eligible].double()

    row_ids, candidate_ids = linear_sum_assignment(cost.numpy())
    result = list(selected_ids)
    for row_index, candidate in zip(row_ids.tolist(), candidate_ids.tolist()):
        if float(cost[row_index, candidate]) >= 1e8:
            continue
        slot = rows[row_index][0]
        result[slot] = int(candidate)
    if len(result) != len(set(result)):
        raise RuntimeError("representative replacement produced a duplicate ID")
    return result


def _classify_extra_candidate(
    official_iou: torch.Tensor,
    before_ids: list[int],
    candidate: int,
    *,
    threshold: float,
) -> str:
    if official_iou.shape[0] == 0:
        return "empty_scene_fp"
    before = evaluator_hungarian_assignment(official_iou, before_ids, threshold)
    after = evaluator_hungarian_assignment(
        official_iou, [*before_ids, candidate], threshold
    )
    if after.hit_count > before.hit_count:
        return "new_tp"
    values = official_iou[:, int(candidate)]
    best_iou = float(values.max()) if values.numel() else 0.0
    if best_iou > float(threshold):
        return "duplicate_or_assignment_competition"
    if float(threshold) >= 0.75 and best_iou > 0.50:
        return "near_miss_0.50_to_0.75"
    if best_iou >= 0.30:
        return "near_miss_0.30_to_threshold"
    return "background"


class AuditAccumulator:
    def __init__(self, thresholds: tuple[float, ...], top_k: int) -> None:
        self.thresholds = thresholds
        self.top_k = int(top_k)
        self.images = 0
        self.gt = 0
        self.mode_predictions: Counter[str] = Counter()
        self.mode_tp: dict[str, Counter[str]] = {
            f"{value:.2f}": Counter() for value in thresholds
        }
        self.prefix_extension_tp: dict[str, int] = Counter()
        self.same_count_oracle_tp: dict[str, int] = Counter()
        self.top4_oracle_tp: dict[str, int] = Counter()
        self.forced_classes: dict[str, Counter[str]] = {
            f"{value:.2f}": Counter() for value in thresholds
        }
        self.stop_images = 0
        self.stop_margins: list[float] = []
        self.margin_groups: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "images": 0,
                "extra_predictions": 0,
                "tp_gain": {f"{value:.2f}": 0 for value in thresholds},
            }
        )
        self.count_groups: dict[int, Counter[str]] = defaultdict(Counter)
        self.support_sizes: list[float] = []
        self.target_entropies: list[float] = []
        self.target_qualities: list[float] = []
        self.validity_disagreements = 0
        self.official_only_valid = 0
        self.row_strip_only_valid = 0
        self.replacement_changed: Counter[str] = Counter()
        self.reroll_changed: Counter[str] = Counter()

    def update(
        self,
        *,
        official_iou: torch.Tensor,
        candidate_valid: torch.Tensor,
        normal_indices: torch.Tensor,
        normal_logits: torch.Tensor,
        forced_indices: torch.Tensor,
        replacements: dict[str, list[int]],
        rerolled: dict[str, torch.Tensor],
        teacher_indices: torch.Tensor,
        teacher_count: int,
        support_sizes: list[float],
        target_entropies: list[float],
        target_qualities: list[float],
        official_candidate_valid: torch.Tensor,
        row_candidate_valid: torch.Tensor,
    ) -> None:
        valid = candidate_valid.bool().cpu()
        official_valid = official_candidate_valid.bool().cpu()
        row_valid = row_candidate_valid.bool().cpu()
        normal, _ = _selected_pointer_ids(normal_indices, valid, top_k=self.top_k)
        forced, _ = _selected_pointer_ids(forced_indices, valid, top_k=self.top_k)
        if forced[: len(normal)] != normal:
            raise RuntimeError("forced continuation changed the pre-STOP prefix")
        teacher, _ = _selected_pointer_ids(
            teacher_indices, valid, top_k=self.top_k
        )
        modes: dict[str, list[int]] = {
            "greedy": normal,
            "teacher_sample": teacher,
            "forced_continuation": forced,
            **replacements,
        }
        for name, indices in rerolled.items():
            selected, _ = _selected_pointer_ids(indices, valid, top_k=self.top_k)
            modes[f"reroll_{name}"] = selected

        gt_count = int(official_iou.shape[0])
        self.images += 1
        self.gt += gt_count
        self.validity_disagreements += int(
            (official_valid != row_valid).sum()
        )
        self.official_only_valid += int((official_valid & ~row_valid).sum())
        self.row_strip_only_valid += int((row_valid & ~official_valid).sum())
        self.support_sizes.extend(support_sizes)
        self.target_entropies.extend(target_entropies)
        self.target_qualities.extend(target_qualities)
        for name, ids in modes.items():
            self.mode_predictions[name] += len(ids)
            for threshold in self.thresholds:
                key = f"{threshold:.2f}"
                self.mode_tp[key][name] += evaluator_hungarian_assignment(
                    official_iou, ids, threshold
                ).hit_count

        for name, ids in replacements.items():
            self.replacement_changed[name] += int(ids != normal)
        for name, indices in rerolled.items():
            reroll_ids, _ = _selected_pointer_ids(indices, valid, top_k=self.top_k)
            self.reroll_changed[name] += int(reroll_ids != normal)

        group = self.count_groups[gt_count]
        top4_half = cardinality_oracle_assignment(
            official_iou, self.thresholds[0], self.top_k, valid
        ).hit_count
        group["images"] += 1
        group["annotation_gt"] += gt_count
        group["official_oracle_hits_0.50"] += top4_half
        group["teacher_representable"] += int(teacher_count)
        group["teacher_sample_hits_0.50"] += evaluator_hungarian_assignment(
            official_iou,
            teacher,
            self.thresholds[0],
        ).hit_count
        group["greedy_emitted"] += len(normal)
        group["greedy_hits_0.50"] += evaluator_hungarian_assignment(
            official_iou,
            normal,
            self.thresholds[0],
        ).hit_count
        group["forced_emitted"] += len(forced)
        group["teacher_count_shortfall_vs_official_hit_count"] += max(
            top4_half - int(teacher_count), 0
        )
        group["greedy_count_shortfall_vs_teacher_count"] += max(
            int(teacher_count) - len(normal), 0
        )
        group["greedy_count_excess_vs_teacher_count"] += max(
            len(normal) - int(teacher_count), 0
        )

        stopped = _first_stop_margin(
            normal_indices.cpu(), normal_logits.cpu(), official_iou.shape[1]
        )
        if stopped is not None:
            _step, margin = stopped
            self.stop_images += 1
            self.stop_margins.append(margin)
            margin_key = _margin_bin(margin)
            margin_group = self.margin_groups[margin_key]
            margin_group["images"] += 1
            margin_group["extra_predictions"] += max(len(forced) - len(normal), 0)

        for threshold in self.thresholds:
            key = f"{threshold:.2f}"
            self.same_count_oracle_tp[key] += cardinality_oracle_assignment(
                official_iou,
                threshold,
                len(normal),
                valid,
            ).hit_count
            self.top4_oracle_tp[key] += cardinality_oracle_assignment(
                official_iou,
                threshold,
                self.top_k,
                valid,
            ).hit_count
            self.prefix_extension_tp[key] += prefix_extension_oracle_hit_count(
                official_iou,
                normal,
                valid,
                threshold=threshold,
                top_k=self.top_k,
            )
            normal_tp = evaluator_hungarian_assignment(
                official_iou, normal, threshold
            ).hit_count
            forced_tp = evaluator_hungarian_assignment(
                official_iou, forced, threshold
            ).hit_count
            if stopped is not None:
                self.margin_groups[_margin_bin(stopped[1])]["tp_gain"][key] += (
                    forced_tp - normal_tp
                )
            for step in range(len(normal), len(forced)):
                candidate = forced[step]
                category = _classify_extra_candidate(
                    official_iou,
                    forced[:step],
                    candidate,
                    threshold=threshold,
                )
                self.forced_classes[key][category] += 1

    def finish(self) -> dict[str, Any]:
        modes: dict[str, Any] = {}
        for name, predictions in sorted(self.mode_predictions.items()):
            modes[name] = {
                key: _metric(int(rows[name]), int(predictions), self.gt)
                for key, rows in self.mode_tp.items()
            }
        gap_closure: dict[str, Any] = {}
        for name in sorted(self.mode_predictions):
            if name in {"greedy", "teacher_sample", "forced_continuation"}:
                continue
            gap_closure[name] = {}
            for threshold in self.thresholds:
                key = f"{threshold:.2f}"
                greedy = int(self.mode_tp[key]["greedy"])
                candidate = int(self.mode_tp[key][name])
                same_count = int(self.same_count_oracle_tp[key])
                gap_closure[name][key] = {
                    "tp_gain": candidate - greedy,
                    "same_count_oracle_gap_closed": _ratio(
                        candidate - greedy,
                        max(same_count - greedy, 0),
                    ),
                }
        greedy_half = int(self.mode_tp[f"{self.thresholds[0]:.2f}"]["greedy"])
        margin_groups = {}
        for key, row in sorted(self.margin_groups.items()):
            extra = int(row["extra_predictions"])
            margin_groups[key] = {
                **row,
                "marginal_precision": {
                    threshold: _ratio(gain, extra)
                    for threshold, gain in row["tp_gain"].items()
                },
            }
        count_groups = {}
        for gt_count, row in sorted(self.count_groups.items()):
            images = int(row["images"])
            count_groups[str(gt_count)] = {
                "images": images,
                **{
                    key: int(value)
                    for key, value in row.items()
                    if key != "images"
                },
                "mean_teacher_representable": _ratio(
                    row["teacher_representable"], images
                ),
                "mean_greedy_emitted": _ratio(row["greedy_emitted"], images),
                "mean_forced_emitted": _ratio(row["forced_emitted"], images),
            }
        extension = {}
        for threshold in self.thresholds:
            key = f"{threshold:.2f}"
            greedy = int(self.mode_tp[key]["greedy"])
            oracle = int(self.prefix_extension_tp[key])
            extension[key] = {
                "greedy_tp": greedy,
                "prefix_preserving_extension_oracle_tp": oracle,
                "recoverable_tp": oracle - greedy,
            }
        extra = self.mode_predictions["forced_continuation"] - self.mode_predictions["greedy"]
        actual_gain = int(self.mode_tp[f"{self.thresholds[0]:.2f}"]["forced_continuation"]) - greedy_half
        return {
            "images": self.images,
            "gt_lanes": self.gt,
            "selection_counterfactuals": modes,
            "representative_gap_closure": gap_closure,
            "reference_oracles": {
                f"{threshold:.2f}": {
                    "same_emitted_count_tp": int(
                        self.same_count_oracle_tp[f"{threshold:.2f}"]
                    ),
                    "top4_tp": int(self.top4_oracle_tp[f"{threshold:.2f}"]),
                }
                for threshold in self.thresholds
            },
            "stop": {
                "images_with_greedy_stop": self.stop_images,
                "stop_margin": _summary(self.stop_margins),
                "forced_extra_predictions": int(extra),
                "forced_tp_gain_at_primary_threshold": actual_gain,
                "forced_marginal_precision_at_primary_threshold": _ratio(
                    actual_gain, extra
                ),
                "f1_break_even_marginal_precision": modes["greedy"][
                    f"{self.thresholds[0]:.2f}"
                ]["f1"] / 2.0,
                "extra_action_classes": {
                    key: dict(value) for key, value in self.forced_classes.items()
                },
                "by_stop_margin": margin_groups,
                "prefix_preserving_extension_oracle": extension,
            },
            "target_policy_cardinality": count_groups,
            "teacher_support": {
                "support_size": _summary(self.support_sizes),
                "target_entropy": _summary(self.target_entropies),
                "target_quality": _summary(self.target_qualities),
            },
            "replacement_images_changed": dict(self.replacement_changed),
            "first_divergence_reroll_images_changed": dict(self.reroll_changed),
            "official_vs_row_strip_candidate_valid_disagreements": (
                self.validity_disagreements
            ),
            "candidate_valid_disagreement_direction": {
                "official_only_valid": self.official_only_valid,
                "row_strip_pointer_only_valid": self.row_strip_only_valid,
            },
        }


def _root_model(model):
    return getattr(model, "module", model)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    thresholds = tuple(sorted(set(float(value) for value in args.iou_thresholds)))
    if not thresholds:
        raise ValueError("at least one IoU threshold is required")
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = str(args.dataset_root)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    checkpoint_iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.requires_grad_(False).eval()
    root = _root_model(model)
    structured = getattr(root, "structured_query_head", None)
    selector = getattr(structured, "set_selection_head", None)
    if selector is None or selector.candidate_interaction != "sequential_pointer":
        raise ValueError("audit requires a sequential pointer selector")
    selector.retain_pointer_diagnostic_tensors = True
    supports_inference_only = bool(getattr(model, "supports_inference_only", False))
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    loader = build_dataloader(cfg, split=args.split, training=False)

    selection_cfg = (
        cfg.get("model", {}).get("structured_query", {}).get("set_selection", {})
    )
    loss_cfg = cfg.get("loss", {})
    input_h = int(loss_cfg.get("input_h", cfg.get("model", {}).get("input_h", 288)))
    representable_min = float(
        selection_cfg.get("pointer_cluster_representable_min", 0.20)
    )
    quality_delta = float(selection_cfg.get("pointer_cluster_quality_delta", 0.10))
    temperature = float(selection_cfg.get("pointer_cluster_temperature", 0.03))
    base_seed = int(cfg.get("training", {}).get("seed", 0))
    accumulator = AuditAccumulator(thresholds, args.top_k)

    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    executor = ThreadPoolExecutor(max_workers=max(int(args.metric_workers), 1))
    try:
        iterator = tqdm(loader, ncols=100, desc="V4.5 STOP/representative audit")
        for batch_index, (images, targets, metas) in enumerate(iterator):
            if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
                break
            if channels_last:
                images = images.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                images = images.to(device, non_blocking=True)
            with _amp_context(device, args.amp_dtype):
                outputs = (
                    model(images, inference_only=True)
                    if supports_inference_only
                    else model(images)
                )
            required_private = (
                "_selection_pointer_hidden",
                "_selection_pointer_relations",
                "_selection_pointer_unary_logits",
                "_selection_pointer_candidate_valid",
            )
            missing = [name for name in required_private if name not in outputs]
            if missing:
                raise ValueError("pointer diagnostic tensors missing: " + ", ".join(missing))
            stage_batch = _stage_batch(outputs)
            stages = [_slice_stage(stage_batch, index) for index in range(len(metas))]
            records = [
                {"meta": dict(meta), "stages": {"main": stage}}
                for meta, stage in zip(metas, stages)
            ]
            futures = [
                executor.submit(
                    _official_for_record,
                    record,
                    line_width=args.line_width,
                    min_valid_rows=args.min_valid_rows,
                )
                for record in records
            ]
            official_rows = [future.result() for future in futures]

            teacher = build_pointer_cluster_soft_targets(
                outputs,
                targets,
                max_selections=int(selector.pointer_max_selections),
                input_h=input_h,
                line_width=float(args.line_width),
                min_valid_rows=int(args.min_valid_rows),
                representable_min=representable_min,
                support_quality_delta=quality_delta,
                temperature=temperature,
                base_seed=base_seed,
                iteration=checkpoint_iteration,
                visit=batch_index,
            )
            forced_rollout = selector.decode_pointer(
                outputs["_selection_pointer_hidden"],
                outputs["_selection_pointer_relations"],
                outputs["_selection_pointer_unary_logits"],
                outputs["_selection_pointer_candidate_valid"],
                policy_logits=outputs.get("_selection_pointer_policy_logits"),
                force_candidate_after_stop=True,
            )

            row_qualities: list[torch.Tensor] = []
            row_validities: list[torch.Tensor] = []
            for image_index, target in enumerate(targets):
                quality, row_valid, valid_gt = pairwise_range_aware_row_strip_iou(
                    outputs["pred_x_rows"][image_index].detach().float(),
                    outputs["range_norm"][image_index].detach().float(),
                    target["x_rows"],
                    target["valid_mask"],
                    input_h=input_h,
                    line_width=float(args.line_width),
                    min_valid_rows=int(args.min_valid_rows),
                )
                row_qualities.append(quality[:, valid_gt].detach().cpu())
                row_validities.append(row_valid.detach().cpu())

            replacement_contracts = {
                # Pure representative audit: only candidates already matched
                # to a GT at the primary official threshold receive a cluster
                # replacement.
                "matched_unary": ("unary", thresholds[0]),
                "matched_step_logit": ("step_logit", thresholds[0]),
                "matched_row_strip": ("row_strip", thresholds[0]),
                "matched_official": ("official", thresholds[0]),
                # Optimistic diagnostic: assign every emitted slot a GT label,
                # even when it is currently a false positive.  This includes
                # cluster-coverage repair and must not be called pure
                # representative improvement.
                "oracle_labeled_row_strip": ("row_strip", -1.0),
                "oracle_labeled_official": ("official", -1.0),
            }
            replacements_by_image: list[dict[str, list[int]]] = []
            forced_actions = {
                mode: torch.full(
                    (len(metas), int(selector.pointer_max_selections)),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                for mode in replacement_contracts
            }
            for image_index, ((official_iou, official_valid), stage) in enumerate(
                zip(official_rows, stages)
            ):
                # The official raster helper clamps x before testing finiteness,
                # while the deployed pointer validity mask tests the raw curve.
                # Oracle selection must never use a candidate the pointer cannot
                # emit, so eligibility is the intersection of both contracts.
                deployable_valid = official_valid & row_validities[image_index]
                selected, _ = _selected_pointer_ids(
                    stage["selection_pointer_indices"],
                    deployable_valid,
                    top_k=args.top_k,
                )
                image_replacements = {}
                for mode, (score_mode, assignment_threshold) in (
                    replacement_contracts.items()
                ):
                    replacement = _replacement_sequence(
                        official_iou,
                        row_qualities[image_index],
                        deployable_valid,
                        selected,
                        stage["selection_logits"],
                        stage["selection_pointer_logits"],
                        mode=score_mode,
                        representable_min=representable_min,
                        assignment_threshold=assignment_threshold,
                    )
                    image_replacements[mode] = replacement
                    divergence = next(
                        (
                            step
                            for step, (old, new) in enumerate(zip(selected, replacement))
                            if old != new
                        ),
                        None,
                    )
                    if divergence is not None:
                        forced_actions[mode][image_index, divergence] = replacement[divergence]
                replacements_by_image.append(image_replacements)

            rerolls = {
                mode: selector.decode_pointer(
                    outputs["_selection_pointer_hidden"],
                    outputs["_selection_pointer_relations"],
                    outputs["_selection_pointer_unary_logits"],
                    outputs["_selection_pointer_candidate_valid"],
                    policy_logits=outputs.get("_selection_pointer_policy_logits"),
                    forced_actions=actions,
                )
                for mode, actions in forced_actions.items()
            }

            for image_index, ((official_iou, official_valid), stage) in enumerate(
                zip(official_rows, stages)
            ):
                active_candidates = teacher["candidate_steps"][image_index].bool()
                deployable_valid = official_valid & row_validities[image_index]
                accumulator.update(
                    official_iou=official_iou,
                    candidate_valid=deployable_valid,
                    normal_indices=stage["selection_pointer_indices"],
                    normal_logits=stage["selection_pointer_logits"],
                    forced_indices=forced_rollout["selection_pointer_indices"][image_index].cpu(),
                    replacements=replacements_by_image[image_index],
                    rerolled={
                        mode: rollout["selection_pointer_indices"][image_index].cpu()
                        for mode, rollout in rerolls.items()
                    },
                    teacher_indices=teacher["indices"][image_index].cpu(),
                    teacher_count=int(teacher["representable_count"][image_index]),
                    support_sizes=teacher["support_sizes"][image_index][active_candidates].cpu().tolist(),
                    target_entropies=teacher["target_entropy"][image_index][active_candidates].cpu().tolist(),
                    target_qualities=teacher["target_quality"][image_index][active_candidates].cpu().tolist(),
                    official_candidate_valid=official_valid,
                    row_candidate_valid=row_validities[image_index],
                )
    finally:
        executor.shutdown(wait=True)
        cv2.setNumThreads(previous_cv_threads)

    report = {
        "experiment": "V4.5 pointer STOP and representative counterfactual forensics",
        "diagnostic_only": True,
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_iteration": checkpoint_iteration,
        "split": str(args.split),
        "teacher_contract": {
            "mode": str(getattr(selector, "pointer_teacher_mode", "")),
            "representable_min": representable_min,
            "quality_delta": quality_delta,
            "temperature": temperature,
        },
        "protocol": {
            "official_culane_raster_iou": True,
            "line_width": float(args.line_width),
            "thresholds": list(thresholds),
            "top_k": int(args.top_k),
            "score_threshold": 0.0,
            "nms": False,
            "forced_continuation_preserves_pre_stop_prefix": True,
            "replacement_preserves_emitted_count": True,
            "reroll_forces_only_first_replacement_divergence": True,
            "matched_replacement_anchor_threshold": thresholds[0],
            "oracle_labeled_replacement_anchor_threshold": -1.0,
            "cardinality_shortfalls_are_count_diagnostics_not_tp_attribution": True,
            "oracle_candidate_validity": (
                "official_raster_valid AND raw-row pointer-valid"
            ),
            "max_batches": int(args.max_batches),
        },
        "audit": accumulator.finish(),
        "decision_rule": {
            "stop_training_authorized_only_if": (
                "a low-margin STOP subgroup has >0.45 marginal precision and "
                "material full-validation TP capacity"
            ),
            "representative_training_authorized_only_if": (
                "a non-oracle learned score closes a material fraction of the "
                "same-count representative gap"
            ),
        },
        "warning": (
            "This is a validation-only causal diagnostic. Do not select "
            "hyperparameters on the already-observed CULane test split."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
