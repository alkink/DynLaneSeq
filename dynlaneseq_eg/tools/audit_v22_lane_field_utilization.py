from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    evaluator_hungarian_assignment,
    official_proposal_gt_iou_matrix,
    sha256_file,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.v20_slot_owned_replacement import _gather_candidate
from dynlaneseq_eg.modeling.v22_field_utilization import (
    candidate_field_component_scores,
    correct_curves_from_distance,
    equal_rank_ensemble,
    owned_component_centerline_score,
    source_seeded_field_path,
)
from dynlaneseq_eg.modeling.v22_lane_field import (
    sample_lane_field_rows,
    score_candidates_from_lane_field,
    soft_range_weights,
)
from dynlaneseq_eg.tools.audit_v11_causal_replay import _image_id, _plain_meta
from dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_official import _required
from dynlaneseq_eg.tools.audit_v20_decision_sufficiency import _load_head, score_cache
from dynlaneseq_eg.tools.cache_v21a_pairwise_visual_verification import (
    _gather_topk,
    _lexicographic_masks,
)
from dynlaneseq_eg.tools.evaluate_v22_lane_field_stage_a import (
    _load_field,
    _summarize_scores,
    _zero_augmentation,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v20_cached_replacement import _load_cache
from dynlaneseq_eg.tools.v19_semantic_coverage_core import _official_matched_values
from dynlaneseq_eg.tools.v22_official_protocol import official_culane_list_contract


FIXED_SEED = 3407
FIXED_TOP_K = 5
FIXED_DISTANCE_STEPS = 2
FIXED_PATH_OFFSET_STEP_PX = 4.0
FIXED_PATH_TRANSITION_SCALE_PX = 8.0
THRESHOLDS = (0.50, 0.75)
CONTINUOUS_POLICIES = (
    "source_v7",
    "source_distance_step1",
    "source_distance_step2",
    "source_seeded_path",
)
CANDIDATE_VARIANTS = ("raw", "distance_step1", "distance_step2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free V22 Stage-A field-utilization/ownership autopsy on "
            "every row of the untouched official CULane validation list."
        )
    )
    parser.add_argument("--field-config", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--v20-config", required=True)
    parser.add_argument("--v20-geometry-checkpoint", required=True)
    parser.add_argument("--v20-scoring-checkpoint", required=True)
    parser.add_argument("--v20-cache-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    return parser.parse_args()


def _dataset_config(
    base: dict[str, Any],
    *,
    dataset_root: str,
    val_list: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})["val"] = str(
        Path(val_list).expanduser().resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    _zero_augmentation(cfg)
    return cfg


def _hits(quality: torch.Tensor, active: torch.Tensor, threshold: float) -> int:
    selected = torch.nonzero(active.bool(), as_tuple=False).flatten().tolist()
    return int(
        evaluator_hungarian_assignment(
            quality.float(), selected, threshold=float(threshold)
        ).hit_count
    )


def _lexicographic_outcome(delta50: int, delta75: int) -> str:
    if int(delta50) > 0 or (int(delta50) == 0 and int(delta75) > 0):
        return "beneficial"
    if int(delta50) < 0 or (int(delta50) == 0 and int(delta75) < 0):
        return "harmful"
    return "neutral"


def _one_edit_outcomes(
    source_quality: torch.Tensor,
    candidate_quality: torch.Tensor,
    *,
    active: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact maximum-IoU-Hungarian TP deltas for [S,K] actions."""

    gt_count, slots = source_quality.shape
    if candidate_quality.shape[:2] != (gt_count, slots):
        raise ValueError("candidate/source official quality mismatch")
    choices = int(candidate_quality.shape[2])
    active_slots = torch.nonzero(active.bool(), as_tuple=False).flatten().tolist()
    delta50 = torch.zeros((slots, choices), dtype=torch.long)
    delta75 = torch.zeros_like(delta50)
    if not active_slots:
        return delta50, delta75
    base = source_quality[:, active_slots]
    base_sets = base.unsqueeze(0)
    base_matched, _ = _official_matched_values(base_sets)
    base50 = int((base_matched[0] > 0.50).sum())
    base75 = int((base_matched[0] > 0.75).sum())
    sets: list[torch.Tensor] = []
    actions: list[tuple[int, int]] = []
    slot_to_column = {slot: index for index, slot in enumerate(active_slots)}
    for slot in active_slots:
        for choice in range(choices):
            if not bool(valid[slot, choice]):
                continue
            quality_set = base.clone()
            quality_set[:, slot_to_column[slot]] = candidate_quality[:, slot, choice]
            sets.append(quality_set)
            actions.append((slot, choice))
    if not sets:
        return delta50, delta75
    matched, _ = _official_matched_values(torch.stack(sets, dim=0))
    hit50 = (matched > 0.50).sum(dim=-1).long()
    hit75 = (matched > 0.75).sum(dim=-1).long()
    for index, (slot, choice) in enumerate(actions):
        delta50[slot, choice] = hit50[index] - base50
        delta75[slot, choice] = hit75[index] - base75
    return delta50, delta75


def _metric_worker(payload: dict[str, Any]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    curves = payload["curves"]
    ranges = payload["ranges"]
    flat_curves: list[torch.Tensor] = []
    flat_ranges: list[torch.Tensor] = []
    layouts: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    cursor = 0
    for name in CONTINUOUS_POLICIES + CANDIDATE_VARIANTS:
        x = curves[name].float()
        rho = ranges[name].float()
        shape = tuple(int(value) for value in x.shape[:-1])
        count = int(x.reshape(-1, x.shape[-1]).shape[0])
        flat_curves.append(x.reshape(count, x.shape[-1]))
        flat_ranges.append(rho.reshape(count, 2))
        layouts[name] = (cursor, cursor + count, shape)
        cursor += count
    record = {
        "meta": payload["meta"],
        "stages": {
            "combined": {
                "pred_x_rows": torch.cat(flat_curves, dim=0),
                "range_norm": torch.cat(flat_ranges, dim=0),
            }
        },
    }
    quality, raster_valid = official_proposal_gt_iou_matrix(
        record,
        "combined",
        line_width=30.0,
        min_valid_rows=5,
        row_visibility_thresh=0.0,
    )

    def quality_view(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        start, stop, shape = layouts[name]
        return (
            quality[:, start:stop].reshape(quality.shape[0], *shape),
            raster_valid[start:stop].reshape(*shape),
        )

    active = payload["source_active"].bool()
    source_quality, source_raster_valid = quality_view("source_v7")
    gt_count = int(quality.shape[0])
    predictions = int(active.sum())
    continuous: dict[str, dict[str, int]] = {}
    for name in CONTINUOUS_POLICIES:
        local_quality, _local_valid = quality_view(name)
        continuous[name] = {
            "tp50": _hits(local_quality, active, 0.50),
            "tp75": _hits(local_quality, active, 0.75),
            "predictions": predictions,
            "gt": gt_count,
        }

    top_valid = payload["top_valid"].bool()
    candidate_quality: dict[str, torch.Tensor] = {}
    candidate_raster_valid: dict[str, torch.Tensor] = {}
    action_deltas: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    action_outcomes: dict[str, Counter[str]] = {}
    for name in CANDIDATE_VARIANTS:
        local_quality, local_valid = quality_view(name)
        candidate_quality[name] = local_quality
        candidate_raster_valid[name] = local_valid & top_valid
        delta50, delta75 = _one_edit_outcomes(
            source_quality,
            local_quality,
            active=active,
            valid=candidate_raster_valid[name],
        )
        action_deltas[name] = (delta50, delta75)
        counts: Counter[str] = Counter()
        for slot in range(int(top_valid.shape[0])):
            for choice in range(int(top_valid.shape[1])):
                if bool(candidate_raster_valid[name][slot, choice]):
                    counts[
                        _lexicographic_outcome(
                            int(delta50[slot, choice]), int(delta75[slot, choice])
                        )
                    ] += 1
        action_outcomes[name] = counts

    ownership = payload["ownership"].long()
    positive_slot = payload["positive_slot"].bool()
    semantics: dict[str, Counter[str]] = {}
    raw_quality = candidate_quality["raw"]
    raw_valid = candidate_raster_valid["raw"]
    raw_delta50, raw_delta75 = action_deltas["raw"]
    policy_selection = dict(payload["policy_selection"])
    exact_owned = torch.zeros_like(ownership)
    for slot in range(int(ownership.shape[0])):
        owned = int(ownership[slot])
        if owned < 0 or owned >= gt_count:
            exact_owned[slot] = 0
            continue
        score = raw_quality[owned, slot].masked_fill(~raw_valid[slot], -1.0)
        exact_owned[slot] = score.argmax()
    policy_selection["gt_exact_owned_iou_oracle"] = exact_owned

    for policy, selected in policy_selection.items():
        counter: Counter[str] = Counter()
        for slot in torch.nonzero(positive_slot, as_tuple=False).flatten().tolist():
            choice = int(selected[slot])
            if choice < 0 or choice >= int(raw_valid.shape[1]) or not bool(
                raw_valid[slot, choice]
            ):
                counter["invalid"] += 1
                continue
            outcome = _lexicographic_outcome(
                int(raw_delta50[slot, choice]), int(raw_delta75[slot, choice])
            )
            counter[f"outcome_{outcome}"] += 1
            owned = int(ownership[slot])
            if gt_count == 0:
                counter["background_or_near_miss"] += 1
                continue
            best_quality, best_gt = raw_quality[:, slot, choice].max(dim=0)
            if float(best_quality) <= 0.50:
                counter["background_or_near_miss"] += 1
            elif owned < 0 or int(best_gt) != owned:
                counter["adjacent_gt_switch"] += 1
            else:
                source_owned = float(source_quality[owned, slot])
                selected_owned = float(raw_quality[owned, slot, choice])
                if selected_owned > source_owned:
                    counter["same_owned_gt_better"] += 1
                else:
                    counter["same_owned_gt_not_better"] += 1
            if int(raw_delta50[slot, choice]) == 0 and int(
                raw_delta75[slot, choice]
            ) == 0:
                counter["threshold_neutral"] += 1
        semantics[policy] = counter

    cached_beneficial = payload["beneficial"].bool()
    cached_harmful = payload["harmful"].bool()
    cached_neutral = payload["neutral"].bool()
    computed_beneficial = torch.zeros_like(cached_beneficial)
    computed_harmful = torch.zeros_like(cached_harmful)
    computed_neutral = torch.zeros_like(cached_neutral)
    for slot in range(int(top_valid.shape[0])):
        for choice in range(int(top_valid.shape[1])):
            if not bool(top_valid[slot, choice]):
                continue
            outcome = _lexicographic_outcome(
                int(raw_delta50[slot, choice]), int(raw_delta75[slot, choice])
            )
            {"beneficial": computed_beneficial, "harmful": computed_harmful,
             "neutral": computed_neutral}[outcome][slot, choice] = True
    parity_mask = top_valid & raw_valid
    parity_mismatch = int(
        (
            ((cached_beneficial != computed_beneficial)
             | (cached_harmful != computed_harmful)
             | (cached_neutral != computed_neutral))
            & parity_mask
        ).sum()
    )

    owned_values: dict[str, list[float]] = defaultdict(list)
    best_owned_values: dict[str, list[float]] = defaultdict(list)
    source_owned_values: dict[str, list[float]] = defaultdict(list)
    for slot in range(int(ownership.shape[0])):
        owned = int(ownership[slot])
        if owned < 0 or owned >= gt_count or not bool(active[slot]):
            continue
        for name in CONTINUOUS_POLICIES:
            local_quality, _ = quality_view(name)
            source_owned_values[name].append(float(local_quality[owned, slot]))
        if not bool(positive_slot[slot]):
            continue
        for name in CANDIDATE_VARIANTS:
            valid_choices = candidate_raster_valid[name][slot]
            values = candidate_quality[name][owned, slot][valid_choices]
            owned_values[name].extend(float(value) for value in values)
            if values.numel():
                best_owned_values[name].append(float(values.max()))

    correction_crossings: dict[str, Counter[str]] = {}
    raw = candidate_quality["raw"]
    for name in ("distance_step1", "distance_step2"):
        corrected = candidate_quality[name]
        valid = candidate_raster_valid["raw"] & candidate_raster_valid[name]
        counts: Counter[str] = Counter()
        for slot in range(int(ownership.shape[0])):
            owned = int(ownership[slot])
            if owned < 0 or owned >= gt_count:
                continue
            for choice in range(int(valid.shape[1])):
                if not bool(valid[slot, choice]):
                    continue
                before = float(raw[owned, slot, choice])
                after = float(corrected[owned, slot, choice])
                counts["improved"] += int(after > before)
                counts["worsened"] += int(after < before)
                counts["unchanged"] += int(after == before)
                for threshold in THRESHOLDS:
                    key = str(threshold).replace("0.", "")
                    counts[f"cross_up_{key}"] += int(
                        before <= threshold and after > threshold
                    )
                    counts[f"cross_down_{key}"] += int(
                        before > threshold and after <= threshold
                    )
        correction_crossings[name] = counts

    return {
        "continuous": continuous,
        "action_outcomes": {name: dict(value) for name, value in action_outcomes.items()},
        "semantics": {name: dict(value) for name, value in semantics.items()},
        "raw_cache_outcome_mismatch": parity_mismatch,
        "raw_cache_outcome_compared": int(parity_mask.sum()),
        "owned_values": dict(owned_values),
        "best_owned_values": dict(best_owned_values),
        "source_owned_values": dict(source_owned_values),
        "correction_crossings": {
            name: dict(value) for name, value in correction_crossings.items()
        },
        "gt_count_cache_mismatch": int(gt_count != int(payload["gt_count"])),
        "source_raster_invalid_active": int((active & ~source_raster_valid).sum()),
    }


class _MetricAccumulator:
    def __init__(self) -> None:
        self.images = 0
        self.continuous = {
            name: Counter() for name in CONTINUOUS_POLICIES
        }
        self.continuous_images = {
            name: {threshold: Counter() for threshold in THRESHOLDS}
            for name in CONTINUOUS_POLICIES
            if name != "source_v7"
        }
        self.action_outcomes = {
            name: Counter() for name in CANDIDATE_VARIANTS
        }
        self.semantics: dict[str, Counter[str]] = defaultdict(Counter)
        self.values: dict[str, list[float]] = defaultdict(list)
        self.correction_crossings: dict[str, Counter[str]] = defaultdict(Counter)
        self.contract = Counter()

    def add(self, result: dict[str, Any]) -> None:
        self.images += 1
        source = result["continuous"]["source_v7"]
        for name, row in result["continuous"].items():
            self.continuous[name].update(row)
            if name != "source_v7":
                for threshold, key in ((0.50, "tp50"), (0.75, "tp75")):
                    delta = int(row[key]) - int(source[key])
                    bucket = "improved" if delta > 0 else "worsened" if delta < 0 else "same"
                    self.continuous_images[name][threshold][bucket] += 1
        for name, row in result["action_outcomes"].items():
            self.action_outcomes[name].update(row)
        for name, row in result["semantics"].items():
            self.semantics[name].update(row)
        for group in ("owned_values", "best_owned_values", "source_owned_values"):
            for name, values in result[group].items():
                self.values[f"{group}/{name}"].extend(float(value) for value in values)
        for name, row in result["correction_crossings"].items():
            self.correction_crossings[name].update(row)
        self.contract["raw_cache_outcome_mismatch"] += int(
            result["raw_cache_outcome_mismatch"]
        )
        self.contract["raw_cache_outcome_compared"] += int(
            result["raw_cache_outcome_compared"]
        )
        self.contract["gt_count_cache_mismatch"] += int(
            result["gt_count_cache_mismatch"]
        )
        self.contract["source_raster_invalid_active"] += int(
            result["source_raster_invalid_active"]
        )

    @staticmethod
    def _value_summary(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "mean": float("nan"), "p50": float("nan"),
                    "p75": float("nan"), "p90": float("nan")}
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "count": int(tensor.numel()),
            "mean": float(tensor.mean()),
            "p50": float(torch.quantile(tensor, 0.50)),
            "p75": float(torch.quantile(tensor, 0.75)),
            "p90": float(torch.quantile(tensor, 0.90)),
        }

    def finalize(self) -> dict[str, Any]:
        continuous: dict[str, Any] = {}
        for name, row in self.continuous.items():
            predictions = int(row["predictions"])
            gt = int(row["gt"])
            policy = {}
            for threshold, key in ((0.50, "tp50"), (0.75, "tp75")):
                tp = int(row[key])
                fp = predictions - tp
                fn = gt - tp
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
                policy[f"{threshold:.2f}"] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
                if name != "source_v7":
                    policy[f"{threshold:.2f}"]["image_outcomes_vs_source"] = dict(
                        self.continuous_images[name][threshold]
                    )
            continuous[name] = policy
        return {
            "images": self.images,
            "continuous_official_raster": continuous,
            "top5_one_edit_action_outcomes": {
                name: dict(value) for name, value in self.action_outcomes.items()
            },
            "selected_raw_candidate_semantics": {
                name: dict(value) for name, value in self.semantics.items()
            },
            "official_owned_gt_quality": {
                name: self._value_summary(values)
                for name, values in self.values.items()
            },
            "distance_correction_crossings": {
                name: dict(value) for name, value in self.correction_crossings.items()
            },
            "contract": dict(self.contract),
        }


class _RowAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, list[torch.Tensor]]] = {
            band: defaultdict(list) for band in ("all", "top", "middle", "bottom")
        }

    def add(
        self,
        *,
        targets: list[dict[str, torch.Tensor]],
        ownership: torch.Tensor,
        active: torch.Tensor,
        source_x: torch.Tensor,
        step1_x: torch.Tensor,
        step2_x: torch.Tensor,
        predicted_step1: torch.Tensor,
        source_range: torch.Tensor,
        input_w: int,
    ) -> None:
        rows = int(source_x.shape[-1])
        row_index = torch.arange(rows, device=source_x.device)
        bands = {
            "all": torch.ones(rows, dtype=torch.bool, device=source_x.device),
            "top": row_index < rows // 4,
            "middle": (row_index >= rows // 4) & (row_index < 3 * rows // 4),
            "bottom": row_index >= 3 * rows // 4,
        }
        row_axis = torch.linspace(0.0, 1.0, rows, device=source_x.device)
        for item, target in enumerate(targets):
            gt_x = target["x_rows"].to(source_x.device).float()[..., :rows]
            gt_valid = target["valid_mask"].to(source_x.device).bool()[..., :rows]
            for slot in range(int(source_x.shape[1])):
                owned = int(ownership[item, slot])
                if not bool(active[item, slot]) or owned < 0 or owned >= int(gt_x.shape[0]):
                    continue
                visible = (
                    gt_valid[owned]
                    & torch.isfinite(gt_x[owned])
                    & torch.isfinite(source_x[item, slot])
                    & (gt_x[owned] >= 0.0)
                    & (gt_x[owned] < float(input_w))
                    & (row_axis >= source_range[item, slot, 0])
                    & (row_axis <= source_range[item, slot, 1])
                )
                actual = gt_x[owned] - source_x[item, slot]
                predicted = predicted_step1[item, slot]
                for band, band_mask in bands.items():
                    mask = visible & band_mask
                    if not bool(mask.any()):
                        continue
                    store = self.values[band]
                    store["source_error"].append(actual[mask].abs().cpu())
                    store["step1_error"].append(
                        (gt_x[owned] - step1_x[item, slot])[mask].abs().cpu()
                    )
                    store["step2_error"].append(
                        (gt_x[owned] - step2_x[item, slot])[mask].abs().cpu()
                    )
                    store["actual_signed_offset"].append(actual[mask].cpu())
                    store["predicted_signed_offset"].append(predicted[mask].cpu())
                    directional = mask & (actual.abs() >= 2.0)
                    if bool(directional.any()):
                        correct = (
                            torch.sign(actual[directional])
                            == torch.sign(predicted[directional])
                        ).float()
                        store["direction_correct"].append(correct.cpu())

    @staticmethod
    def _summary(chunks: list[torch.Tensor]) -> dict[str, float | int]:
        if not chunks:
            return {"count": 0, "mean": float("nan"), "p50": float("nan"),
                    "p75": float("nan"), "p90": float("nan")}
        value = torch.cat(chunks).float()
        return {
            "count": int(value.numel()),
            "mean": float(value.mean()),
            "p50": float(torch.quantile(value, 0.50)),
            "p75": float(torch.quantile(value, 0.75)),
            "p90": float(torch.quantile(value, 0.90)),
        }

    def finalize(self) -> dict[str, Any]:
        return {
            band: {
                name: self._summary(chunks) for name, chunks in values.items()
            }
            for band, values in self.values.items()
        }


def _append_rank_rows(
    storage: dict[str, list[torch.Tensor]],
    *,
    policies: dict[str, torch.Tensor],
    valid: torch.Tensor,
    beneficial: torch.Tensor,
    harmful: torch.Tensor,
    neutral: torch.Tensor,
    positive_slot: torch.Tensor,
) -> None:
    batch = int(valid.shape[0])
    for image in range(batch):
        slots = torch.nonzero(positive_slot[image], as_tuple=False).flatten()
        if slots.numel() == 0:
            continue
        for name, score in policies.items():
            storage[f"score/{name}"].append(score[image, slots].detach().cpu())
        storage["valid"].append(valid[image, slots].detach().cpu())
        storage["beneficial"].append(beneficial[image, slots].detach().cpu())
        storage["harmful"].append(harmful[image, slots].detach().cpu())
        storage["neutral"].append(neutral[image, slots].detach().cpu())


def _rank_summary(storage: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    valid = torch.cat(storage["valid"]).bool()
    beneficial = torch.cat(storage["beneficial"]).bool()
    harmful = torch.cat(storage["harmful"]).bool()
    neutral = torch.cat(storage["neutral"]).bool()
    result = {}
    for key, chunks in storage.items():
        if not key.startswith("score/"):
            continue
        result[key.split("/", 1)[1]] = _summarize_scores(
            torch.cat(chunks),
            valid=valid,
            beneficial=beneficial,
            harmful=harmful,
            neutral=neutral,
        )
    return result


def _predicted_source_step(
    outputs: dict[str, torch.Tensor],
    source_x: torch.Tensor,
    *,
    input_w: int,
    distance_limit_px: float,
) -> torch.Tensor:
    distance = sample_lane_field_rows(
        torch.tanh(outputs["distance_raw"].float()) * float(distance_limit_px),
        source_x,
        input_w=input_w,
    )[..., 0]
    support = sample_lane_field_rows(
        outputs["support_logits"].float(), source_x, input_w=input_w
    )[..., 0].sigmoid()
    return support * distance


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if int(args.seed) != FIXED_SEED:
        raise ValueError(f"V22 utilization audit requires seed {FIXED_SEED}")
    if int(args.eval_batch_size) < 1:
        raise ValueError("eval batch size must be positive")
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    val_contract = official_culane_list_contract(
        args.dataset_root, split="val", supplied_path=args.val_list
    )

    field_cfg: dict[str, Any] = load_config(args.field_config)
    v20_cfg: dict[str, Any] = load_config(args.v20_config)
    v20_cfg = _dataset_config(
        v20_cfg,
        dataset_root=args.dataset_root,
        val_list=args.val_list,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    field_model, field_payload = _load_field(
        field_cfg, args.field_checkpoint, device
    )
    v20_model = build_model(v20_cfg).to(device)
    geometry_iteration = int(
        load_checkpoint(args.v20_geometry_checkpoint, v20_model, strict=False)
    )
    v20_model.requires_grad_(False).eval()
    v20_head, scoring_iteration = _load_head(
        args.v20_config, args.v20_scoring_checkpoint, device
    )
    cache, cache_manifest = _load_cache(
        Path(args.v20_cache_manifest).expanduser().resolve()
    )
    val_path = Path(args.val_list).expanduser().resolve()
    if str(cache_manifest.get("list_sha256")) != sha256_file(val_path):
        raise ValueError("V22 utilization audit cache/list digest mismatch")
    images_total = int(cache["source_route"].shape[0])
    if images_total != int(val_contract["expected_nonempty_rows"]):
        raise ValueError("V22 utilization audit requires all official val rows")
    loader = build_dataloader(v20_cfg, split="val", training=False)
    if len(loader.dataset) != images_total:
        raise ValueError("V22 utilization loader/cache population mismatch")

    scored = score_cache(
        v20_head, cache, device=device, batch_size=64, context_mode="treatment"
    )
    replacement_scores = scored["raw_action_scores"][:, 1:].reshape_as(
        cache["action_valid"]
    )
    action_valid_all = cache["action_valid"].bool()
    beneficial_all, harmful_all, neutral_all = _lexicographic_masks(
        cache["delta50_class"][:, 1:].reshape_as(action_valid_all),
        cache["delta75_class"][:, 1:].reshape_as(action_valid_all),
    )
    positive_slot_all = (beneficial_all & action_valid_all).any(dim=2)

    model_cfg = field_cfg["model"]
    stage_cfg = field_cfg["v22_stage_a"]
    input_w = int(model_cfg["input_w"])
    distance_limit_px = float(stage_cfg["distance_limit_px"])
    rank_storage: dict[str, list[torch.Tensor]] = defaultdict(list)
    row_accumulator = _RowAccumulator()
    metric_accumulator = _MetricAccumulator()
    contract = Counter()
    metric_workers = max(int(args.metric_workers), 0)
    executor = (
        ProcessPoolExecutor(
            max_workers=metric_workers,
            mp_context=mp.get_context("spawn"),
        )
        if metric_workers > 1
        else None
    )
    pending: deque[Future[dict[str, Any]]] = deque()

    def submit(payload: dict[str, Any]) -> None:
        if executor is None:
            metric_accumulator.add(_metric_worker(payload))
            return
        pending.append(executor.submit(_metric_worker, payload))
        if len(pending) >= 4 * metric_workers:
            metric_accumulator.add(pending.popleft().result())

    global_index = 0
    try:
        progress = tqdm(loader, desc="V22 field utilization full official val", ncols=100)
        for images, targets, metas in progress:
            batch = int(images.shape[0])
            start, stop = global_index, global_index + batch
            expected_ids = [str(value) for value in cache_manifest["image_ids"][start:stop]]
            actual_ids = [
                _image_id(meta, f"v22_util_{start + item:06d}")
                for item, meta in enumerate(metas)
            ]
            contract["image_id_mismatch"] += sum(
                left != right for left, right in zip(expected_ids, actual_ids)
            )
            images = images.to(device, non_blocking=True)
            v20_outputs = v20_model(images)
            field_outputs = field_model(images)
            cached = {
                key: value[start:stop].to(device, non_blocking=True)
                for key, value in cache.items()
            }
            source_route = cached["source_route"].long()
            source_active = cached["source_active"].bool()
            action_valid = cached["action_valid"].bool()
            contract["live_source_route_mismatch"] += int(
                (
                    _required(
                        v20_outputs,
                        "selection_slot_v20_v7_geometry_route_indices",
                    ).long()
                    != source_route
                ).sum().cpu()
            )
            contract["live_source_active_mismatch"] += int(
                (
                    _required(v20_outputs, "selection_slot_v20_v7_active").bool()
                    != source_active
                ).sum().cpu()
            )
            local_rank = replacement_scores[start:stop].to(device).masked_fill(
                ~action_valid, -1.0e4
            )
            top_ids = torch.topk(local_rank, k=FIXED_TOP_K, dim=2).indices
            top_valid = _gather_topk(
                action_valid.unsqueeze(-1), top_ids
            ).squeeze(-1)
            cf_x = _required(
                v20_outputs, "selection_slot_v19_counterfactual_x_rows"
            ).float()
            cf_range = _required(
                v20_outputs, "selection_slot_v19_counterfactual_range_norm"
            ).float()
            source_x = _gather_candidate(cf_x, source_route)
            source_range = _gather_candidate(cf_range, source_route)
            candidate_x = _gather_topk(cf_x, top_ids)
            candidate_range = _gather_topk(cf_range, top_ids)

            components = candidate_field_component_scores(
                field_outputs,
                candidate_x=candidate_x,
                candidate_range=candidate_range,
                candidate_valid=top_valid,
                input_w=input_w,
            )
            old_interface = score_candidates_from_lane_field(
                field_outputs,
                source_x=source_x,
                source_range=source_range,
                candidate_x=candidate_x,
                candidate_range=candidate_range,
                candidate_valid=top_valid,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
            )
            owned_component = owned_component_centerline_score(
                field_outputs,
                candidate_x=candidate_x,
                candidate_range=candidate_range,
                candidate_valid=top_valid,
                targets=targets,
                ownership=cached["ownership"].long(),
                input_w=input_w,
            )
            policies = {
                "v20_shortlist_score": _gather_topk(
                    replacement_scores[start:stop].to(device), top_ids
                ),
                "legacy_corrected_source": old_interface["field_score"],
                "centerline_direct": components["centerline"],
                "distance_direct": components["distance"],
                "support_direct": components["support"],
                "center_distance_equal_rank": equal_rank_ensemble(
                    (components["centerline"], components["distance"]), top_valid
                ),
                "center_support_equal_rank": equal_rank_ensemble(
                    (components["centerline"], components["support"]), top_valid
                ),
                "all_heads_equal_rank": equal_rank_ensemble(
                    (
                        components["centerline"],
                        components["distance"],
                        components["support"],
                    ),
                    top_valid,
                ),
                "gt_owned_voronoi_centerline_upper_bound": owned_component,
            }
            beneficial = _gather_topk(
                beneficial_all[start:stop].to(device).unsqueeze(-1), top_ids
            ).squeeze(-1)
            harmful = _gather_topk(
                harmful_all[start:stop].to(device).unsqueeze(-1), top_ids
            ).squeeze(-1)
            neutral = _gather_topk(
                neutral_all[start:stop].to(device).unsqueeze(-1), top_ids
            ).squeeze(-1)
            positive_slot = positive_slot_all[start:stop].to(device)
            _append_rank_rows(
                rank_storage,
                policies=policies,
                valid=top_valid,
                beneficial=beneficial,
                harmful=harmful,
                neutral=neutral,
                positive_slot=positive_slot,
            )

            source_step1, source_step2 = correct_curves_from_distance(
                field_outputs,
                x_rows=source_x,
                range_norm=source_range,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
                steps=FIXED_DISTANCE_STEPS,
            )
            candidate_step1, candidate_step2 = correct_curves_from_distance(
                field_outputs,
                x_rows=candidate_x,
                range_norm=candidate_range,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
                steps=FIXED_DISTANCE_STEPS,
            )
            source_path = source_seeded_field_path(
                field_outputs,
                source_x=source_x,
                source_range=source_range,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
                offset_step_px=FIXED_PATH_OFFSET_STEP_PX,
                transition_scale_px=FIXED_PATH_TRANSITION_SCALE_PX,
            )
            predicted_source_step = _predicted_source_step(
                field_outputs,
                source_x,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
            )
            row_accumulator.add(
                targets=targets,
                ownership=cached["ownership"].long(),
                active=source_active,
                source_x=source_x,
                step1_x=source_step1,
                step2_x=source_step2,
                predicted_step1=predicted_source_step,
                source_range=source_range,
                input_w=input_w,
            )

            policy_selection = {
                name: score.masked_fill(~top_valid, -1.0e4).argmax(dim=2)
                for name, score in policies.items()
            }
            for item in range(batch):
                submit(
                    {
                        "meta": _plain_meta(metas[item]),
                        "curves": {
                            "source_v7": source_x[item].cpu(),
                            "source_distance_step1": source_step1[item].cpu(),
                            "source_distance_step2": source_step2[item].cpu(),
                            "source_seeded_path": source_path[item].cpu(),
                            "raw": candidate_x[item].cpu(),
                            "distance_step1": candidate_step1[item].cpu(),
                            "distance_step2": candidate_step2[item].cpu(),
                        },
                        "ranges": {
                            "source_v7": source_range[item].cpu(),
                            "source_distance_step1": source_range[item].cpu(),
                            "source_distance_step2": source_range[item].cpu(),
                            "source_seeded_path": source_range[item].cpu(),
                            "raw": candidate_range[item].cpu(),
                            "distance_step1": candidate_range[item].cpu(),
                            "distance_step2": candidate_range[item].cpu(),
                        },
                        "source_active": source_active[item].cpu(),
                        "top_valid": top_valid[item].cpu(),
                        "ownership": cached["ownership"][item].long().cpu(),
                        "positive_slot": positive_slot[item].cpu(),
                        "beneficial": beneficial[item].cpu(),
                        "harmful": harmful[item].cpu(),
                        "neutral": neutral[item].cpu(),
                        "policy_selection": {
                            name: selected[item].cpu()
                            for name, selected in policy_selection.items()
                        },
                        "gt_count": int(cached["gt_count"][item]),
                    }
                )
            global_index = stop
            progress.set_postfix(metric_done=metric_accumulator.images, pending=len(pending))
        while pending:
            metric_accumulator.add(pending.popleft().result())
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if global_index != images_total:
        raise RuntimeError("V22 utilization audit did not consume full val population")
    metric_report = metric_accumulator.finalize()
    contract.update(metric_report["contract"])
    passed_data_contract = (
        int(contract["image_id_mismatch"]) == 0
        and int(contract["gt_count_cache_mismatch"]) == 0
        and int(metric_report["images"]) == images_total
    )
    report = {
        "experiment": "V22 Stage-A field utilization and ownership autopsy",
        "field_config": str(Path(args.field_config).expanduser().resolve()),
        "field_config_sha256": sha256_file(args.field_config),
        "field_checkpoint": str(Path(args.field_checkpoint).expanduser().resolve()),
        "field_checkpoint_sha256": sha256_file(args.field_checkpoint),
        "field_training_iteration": int(field_payload["iteration"]),
        "v20_config": str(Path(args.v20_config).expanduser().resolve()),
        "v20_geometry_checkpoint_sha256": sha256_file(args.v20_geometry_checkpoint),
        "v20_scoring_checkpoint_sha256": sha256_file(args.v20_scoring_checkpoint),
        "v20_geometry_iteration": geometry_iteration,
        "v20_scoring_iteration": scoring_iteration,
        "v20_cache_manifest": str(Path(args.v20_cache_manifest).expanduser().resolve()),
        "v20_cache_manifest_sha256": sha256_file(args.v20_cache_manifest),
        "official_validation_population_contract": val_contract,
        "fixed_diagnostic_contract": {
            "top_k": FIXED_TOP_K,
            "distance_steps": FIXED_DISTANCE_STEPS,
            "path_offset_limit_px": distance_limit_px,
            "path_offset_step_px": FIXED_PATH_OFFSET_STEP_PX,
            "path_transition_scale_px": FIXED_PATH_TRANSITION_SCALE_PX,
            "head_ensembles": "equal within-shortlist ranks; no fitted scales",
            "gt_conditioned_policies_are_upper_bounds_only": True,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "validation_subset_used": False,
            "validation_rows_removed": 0,
            "validation_deduplication_performed": False,
            "test_set_used": False,
        },
        "candidate_head_factorial": _rank_summary(rank_storage),
        "row_band_signed_correction": row_accumulator.finalize(),
        **{key: value for key, value in metric_report.items() if key != "contract"},
        "contract": {
            **dict(contract),
            "all_official_val_images_evaluated": metric_report["images"] == images_total,
            "passed_data_contract": passed_data_contract,
        },
        "recommendation": "manual_architecture_decision_from_fixed_training_free_autopsy",
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True))
    print(
        "V22 utilization autopsy completed on every official val.txt row. "
        "No training, filtering, deduplication, checkpoint selection, threshold "
        "search, or test evaluation was performed."
    )


if __name__ == "__main__":
    main()
