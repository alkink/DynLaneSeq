"""Exact fixed-count audit of raw and counterfactual-refined V7 proposal banks.

The implementation is intentionally self-contained at the policy layer.  It
keeps V7 activity and proposal provenance fixed, evaluates every immutable
curve with the official raster operator, and searches injective proposal-ID
assignments with the predeclared lexicographic objective:

    TP@.50 -> TP@.75 -> source-on-tie -> matched IoU.

V26 changed-lane JSONL input is optional.  When supplied, the same selected
proposal IDs are replayed through both the raw writer and the frozen V7
counterfactual refiner.  The test set is never accepted as an input split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dynlaneseq_eg.engine.checkpoint import _torch_load, load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    official_proposal_gt_iou_matrix,
    sha256_file,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.v19_counterfactual_fidelity import (
    frozen_v7_counterfactual_anchors,
)
from dynlaneseq_eg.tools.audit_v11_causal_replay import _image_id, _plain_meta
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v19_semantic_coverage_core import (
    _official_matched_values,
    _unique_candidate_tuples,
)


THRESHOLDS = (0.50, 0.75)


@dataclass(frozen=True)
class OracleResult:
    routes: tuple[int, ...]
    variants: tuple[int, ...]
    tp50: int
    tp75: int
    edit_count: int
    matched_iou: float
    assignments_evaluated: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact V7 source, raw proposals, counterfactual-refined "
            "proposals, optional V26-ID replay, and raw+refined union capacity."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle-device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--oracle-chunk-size", type=int, default=262_144)
    parser.add_argument(
        "--v26-changed-lanes-jsonl",
        help=(
            "Optional full changed-lane log. Unlisted image/slot pairs preserve "
            "the exact V7 proposal ID."
        ),
    )
    parser.add_argument(
        "--raw-refined-union-oracle",
        action="store_true",
        help="Run the exact, more expensive raw+refined provenance-aware oracle.",
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-details-jsonl")
    parser.add_argument(
        "--output-quality-cache",
        help=(
            "Optional torch cache containing the exact source/raw/refined "
            "official IoU matrices used by downstream observability probes."
        ),
    )
    return parser.parse_args()


def _cfg(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    payload = _torch_load(args.checkpoint)
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint does not contain its training config")
    # Round-trip through JSON so the checkpoint payload remains immutable.
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})[args.split] = str(
        Path(args.list_path).expanduser().resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg, int(payload.get("iteration", 0))


def _required(outputs: dict[str, Any], name: str) -> torch.Tensor:
    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"missing required V7 output: {name}")
    return value


def _load_v26_edits(path: str | None) -> dict[tuple[str, int], tuple[int, int]]:
    if not path:
        return {}
    source = Path(path).expanduser()
    edits: dict[tuple[str, int], tuple[int, int]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["image_id"]), int(row["slot"]))
            value = (
                int(row["anchor_proposal_id"]),
                int(row["selected_proposal_id"]),
            )
            previous = edits.setdefault(key, value)
            if previous != value:
                raise ValueError(
                    f"conflicting V26 edit at {source}:{line_number}: {key}"
                )
    return edits


def _policy_summary(
    quality: torch.Tensor,
    columns: Iterable[int],
) -> dict[str, float | int]:
    ids = tuple(int(value) for value in columns)
    gt_count = int(quality.shape[0])
    prediction_count = len(ids)
    if gt_count == 0 or prediction_count == 0:
        return {
            "tp50": 0,
            "tp75": 0,
            "matched_iou": 0.0,
            "predictions": prediction_count,
            "gt": gt_count,
        }
    selected = quality[:, list(ids)].detach().float().unsqueeze(0)
    matched, total = _official_matched_values(selected)
    return {
        "tp50": int((matched[0] > THRESHOLDS[0]).sum()),
        "tp75": int((matched[0] > THRESHOLDS[1]).sum()),
        "matched_iou": float(total[0]),
        "predictions": prediction_count,
        "gt": gt_count,
    }


def _candidate_key(
    tp50: int,
    tp75: int,
    edits: int,
    total: float,
) -> tuple[int, int, int, float]:
    return int(tp50), int(tp75), -int(edits), float(total)


def exact_source_tied_oracle(
    quality: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    source_routes: torch.Tensor,
    *,
    device: str | torch.device,
    chunk_size: int,
    variants: torch.Tensor | None = None,
) -> OracleResult:
    """Exact injective oracle for one immutable geometry variant per proposal.

    ``quality`` has shape ``[G,S,N]``.  If ``variants`` is omitted all selected
    candidates are variant zero.  Source-on-tie is applied before matched IoU.
    """

    if quality.ndim != 3:
        raise ValueError("quality must have shape [G,S,N]")
    gt_count, slots, candidates = quality.shape
    if valid.shape != (slots, candidates):
        raise ValueError("valid must have shape [S,N]")
    if active.shape != (slots,) or source_routes.shape != (slots,):
        raise ValueError("active/source routes must have shape [S]")
    active_slots = torch.nonzero(active.bool(), as_tuple=False).flatten().tolist()
    active_count = len(active_slots)
    if active_count == 0:
        return OracleResult(
            routes=tuple(-1 for _ in range(slots)),
            variants=tuple(-1 for _ in range(slots)),
            tp50=0,
            tp75=0,
            edit_count=0,
            matched_iou=0.0,
            assignments_evaluated=1,
        )
    tuples = _unique_candidate_tuples(int(candidates), active_count)
    quality_device = quality.detach().float().to(device)
    valid_cpu = valid.detach().bool().cpu()
    source = source_routes.detach().long().cpu()[active_slots]
    best_key: tuple[int, int, int, float] | None = None
    best_routes: tuple[int, ...] | None = None
    best_summary = (0, 0, 0, 0.0)
    evaluated = 0
    for start in range(0, int(tuples.shape[0]), max(int(chunk_size), 1)):
        candidate_tuple = tuples[start : start + max(int(chunk_size), 1)]
        usable = torch.ones(candidate_tuple.shape[0], dtype=torch.bool)
        for column, slot in enumerate(active_slots):
            usable &= valid_cpu[slot, candidate_tuple[:, column]]
        candidate_tuple = candidate_tuple[usable]
        if candidate_tuple.numel() == 0:
            continue
        evaluated += int(candidate_tuple.shape[0])
        candidate_device = candidate_tuple.to(device)
        quality_sets = torch.stack(
            [
                quality_device[:, slot, candidate_device[:, column]].transpose(0, 1)
                for column, slot in enumerate(active_slots)
            ],
            dim=-1,
        )
        matched, total = _official_matched_values(quality_sets)
        hits50 = (matched > THRESHOLDS[0]).sum(dim=-1).cpu()
        hits75 = (matched > THRESHOLDS[1]).sum(dim=-1).cpu()
        edits = (candidate_tuple != source.view(1, -1)).sum(dim=-1)
        total_cpu = total.detach().float().cpu()
        max50 = int(hits50.max())
        eligible = hits50 == max50
        max75 = int(hits75[eligible].max())
        eligible &= hits75 == max75
        min_edits = int(edits[eligible].min())
        eligible &= edits == min_edits
        max_total = float(total_cpu[eligible].max())
        eligible &= total_cpu >= max_total - 1.0e-12
        local = int(torch.nonzero(eligible, as_tuple=False)[0])
        key = _candidate_key(max50, max75, min_edits, max_total)
        if best_key is None or key > best_key:
            best_key = key
            best_routes = tuple(int(value) for value in candidate_tuple[local])
            best_summary = max50, max75, min_edits, max_total
    if best_routes is None:
        raise RuntimeError("no valid injective proposal assignment")
    full_routes = [-1 for _ in range(slots)]
    full_variants = [-1 for _ in range(slots)]
    for offset, (slot, candidate) in enumerate(zip(active_slots, best_routes)):
        full_routes[slot] = int(candidate)
        full_variants[slot] = 0 if variants is None else int(variants[offset])
    return OracleResult(
        routes=tuple(full_routes),
        variants=tuple(full_variants),
        tp50=int(best_summary[0]),
        tp75=int(best_summary[1]),
        edit_count=int(best_summary[2]),
        matched_iou=float(best_summary[3]),
        assignments_evaluated=evaluated,
    )


def exact_raw_refined_union_oracle(
    raw_quality: torch.Tensor,
    refined_quality: torch.Tensor,
    raw_valid: torch.Tensor,
    refined_valid: torch.Tensor,
    active: torch.Tensor,
    source_routes: torch.Tensor,
    *,
    device: str | torch.device,
    chunk_size: int,
) -> OracleResult:
    """Exact raw+refined oracle with proposal provenance uniqueness.

    Proposal-ID tuples are enumerated once.  Every tuple is expanded into its
    ``2**active_count`` immutable raw/refined geometry choices.  Choosing raw
    geometry for the source proposal is an edit; the exact V7 source is the
    refined source proposal.
    """

    if raw_quality.shape != refined_quality.shape or raw_quality.ndim != 3:
        raise ValueError("raw/refined quality must share [G,S,N]")
    gt_count, slots, candidates = raw_quality.shape
    if raw_valid.shape != (slots, candidates) or refined_valid.shape != (
        slots,
        candidates,
    ):
        raise ValueError("raw/refined validity must share [S,N]")
    active_slots = torch.nonzero(active.bool(), as_tuple=False).flatten().tolist()
    active_count = len(active_slots)
    if active_count == 0:
        return OracleResult(
            routes=tuple(-1 for _ in range(slots)),
            variants=tuple(-1 for _ in range(slots)),
            tp50=0,
            tp75=0,
            edit_count=0,
            matched_iou=0.0,
            assignments_evaluated=1,
        )
    proposal_tuples = _unique_candidate_tuples(candidates, active_count)
    variant_bits = torch.stack(
        torch.meshgrid(
            *([torch.arange(2, dtype=torch.long)] * active_count),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, active_count)
    raw_device = raw_quality.detach().float().to(device)
    refined_device = refined_quality.detach().float().to(device)
    raw_valid_cpu = raw_valid.detach().bool().cpu()
    refined_valid_cpu = refined_valid.detach().bool().cpu()
    source = source_routes.detach().long().cpu()[active_slots]
    best_key: tuple[int, int, int, float] | None = None
    best_tuple: torch.Tensor | None = None
    best_variant: torch.Tensor | None = None
    best_summary = (0, 0, 0, 0.0)
    evaluated = 0
    variant_count = int(variant_bits.shape[0])
    base_chunk = max(int(chunk_size) // variant_count, 1)
    for start in range(0, int(proposal_tuples.shape[0]), base_chunk):
        base = proposal_tuples[start : start + base_chunk]
        expanded_routes = base.repeat_interleave(variant_count, dim=0)
        expanded_variants = variant_bits.repeat(int(base.shape[0]), 1)
        usable = torch.ones(expanded_routes.shape[0], dtype=torch.bool)
        for column, slot in enumerate(active_slots):
            candidate = expanded_routes[:, column]
            is_refined = expanded_variants[:, column].bool()
            usable &= torch.where(
                is_refined,
                refined_valid_cpu[slot, candidate],
                raw_valid_cpu[slot, candidate],
            )
        expanded_routes = expanded_routes[usable]
        expanded_variants = expanded_variants[usable]
        if expanded_routes.numel() == 0:
            continue
        evaluated += int(expanded_routes.shape[0])
        route_device = expanded_routes.to(device)
        variant_device = expanded_variants.to(device).bool()
        columns = []
        for column, slot in enumerate(active_slots):
            raw = raw_device[:, slot, route_device[:, column]].transpose(0, 1)
            refined = refined_device[:, slot, route_device[:, column]].transpose(0, 1)
            columns.append(
                torch.where(variant_device[:, column : column + 1], refined, raw)
            )
        quality_sets = torch.stack(columns, dim=-1)
        matched, total = _official_matched_values(quality_sets)
        hits50 = (matched > THRESHOLDS[0]).sum(dim=-1).cpu()
        hits75 = (matched > THRESHOLDS[1]).sum(dim=-1).cpu()
        edits = (
            (expanded_routes != source.view(1, -1))
            | (~expanded_variants.bool())
        ).sum(dim=-1)
        total_cpu = total.detach().float().cpu()
        max50 = int(hits50.max())
        eligible = hits50 == max50
        max75 = int(hits75[eligible].max())
        eligible &= hits75 == max75
        min_edits = int(edits[eligible].min())
        eligible &= edits == min_edits
        max_total = float(total_cpu[eligible].max())
        eligible &= total_cpu >= max_total - 1.0e-12
        local = int(torch.nonzero(eligible, as_tuple=False)[0])
        key = _candidate_key(max50, max75, min_edits, max_total)
        if best_key is None or key > best_key:
            best_key = key
            best_tuple = expanded_routes[local].clone()
            best_variant = expanded_variants[local].clone()
            best_summary = max50, max75, min_edits, max_total
    if best_tuple is None or best_variant is None:
        raise RuntimeError("no valid raw+refined union assignment")
    full_routes = [-1 for _ in range(slots)]
    full_variants = [-1 for _ in range(slots)]
    for slot, candidate, variant in zip(
        active_slots, best_tuple.tolist(), best_variant.tolist()
    ):
        full_routes[slot] = int(candidate)
        full_variants[slot] = int(variant)
    return OracleResult(
        routes=tuple(full_routes),
        variants=tuple(full_variants),
        tp50=int(best_summary[0]),
        tp75=int(best_summary[1]),
        edit_count=int(best_summary[2]),
        matched_iou=float(best_summary[3]),
        assignments_evaluated=evaluated,
    )


def _add_total(
    totals: dict[str, dict[str, float | int]],
    name: str,
    row: dict[str, float | int],
) -> None:
    target = totals.setdefault(
        name,
        {
            "tp50": 0,
            "tp75": 0,
            "matched_iou": 0.0,
            "predictions": 0,
            "gt": 0,
        },
    )
    for key in ("tp50", "tp75", "predictions", "gt"):
        target[key] = int(target[key]) + int(row[key])
    target["matched_iou"] = float(target["matched_iou"]) + float(
        row["matched_iou"]
    )


def _add_oracle_total(
    totals: dict[str, dict[str, float | int]],
    name: str,
    result: OracleResult,
    *,
    predictions: int,
    gt: int,
) -> None:
    _add_total(
        totals,
        name,
        {
            "tp50": result.tp50,
            "tp75": result.tp75,
            "matched_iou": result.matched_iou,
            "predictions": predictions,
            "gt": gt,
        },
    )


def _finalize(totals: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, row in totals.items():
        predictions = int(row["predictions"])
        gt = int(row["gt"])
        denominator = predictions + gt
        output[name] = {
            **row,
            "f1_50_points": (
                0.0
                if denominator <= 0
                else 200.0 * float(row["tp50"]) / float(denominator)
            ),
            "f1_75_points": (
                0.0
                if denominator <= 0
                else 200.0 * float(row["tp75"]) / float(denominator)
            ),
        }
    return output


def _quantiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _initialize_metric_process() -> None:
    # One OpenCV thread per Python worker prevents N x M oversubscription.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _evaluate_official_record(
    task: tuple[dict[str, Any], float, int],
) -> dict[str, torch.Tensor]:
    item, line_width, min_valid_rows = task
    matrix, valid = official_proposal_gt_iou_matrix(
        item["record"],
        "combined",
        line_width=float(line_width),
        min_valid_rows=int(min_valid_rows),
        row_visibility_thresh=0.0,
    )
    return {"quality": matrix, "valid": valid}


def _evaluate_records_processes(
    records: list[dict[str, Any]],
    *,
    line_width: float,
    min_valid_rows: int,
    workers: int,
) -> list[dict[str, torch.Tensor]]:
    tasks = [
        (item, float(line_width), int(min_valid_rows)) for item in records
    ]
    count = max(int(workers), 1)
    if count == 1:
        _initialize_metric_process()
        return [
            _evaluate_official_record(task)
            for task in tqdm(tasks, desc="official refined-bank raster", ncols=94)
        ]
    # CUDA has already been initialized by detector inference.  Spawn avoids
    # inheriting a live CUDA context into CPU-only raster workers.
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=count,
        mp_context=context,
        initializer=_initialize_metric_process,
    ) as executor:
        return list(
            tqdm(
                executor.map(_evaluate_official_record, tasks, chunksize=1),
                total=len(tasks),
                desc="official refined-bank raster",
                ncols=94,
            )
        )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg, expected_iteration = _cfg(args)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=True))
    if iteration != expected_iteration:
        raise RuntimeError("checkpoint iteration changed while loading")
    model.eval()
    selection_head = model.structured_query_head.set_selection_head
    refiner = selection_head.slot_refinement
    if refiner is None:
        raise RuntimeError("refined bank audit requires the frozen V7 refiner")
    edits = _load_v26_edits(args.v26_changed_lanes_jsonl)
    loader = build_dataloader(cfg, split=args.split, training=False)
    captured: dict[str, torch.Tensor] = {}

    def capture_refiner_inputs(_module, _args, kwargs):
        captured.clear()
        captured.update(
            {
                key: value
                for key, value in kwargs.items()
                if isinstance(value, torch.Tensor)
            }
        )

    handle = refiner.register_forward_pre_hook(
        capture_refiner_inputs,
        with_kwargs=True,
    )
    records: list[dict[str, Any]] = []
    source_refiner_x_parity_max = 0.0
    source_refiner_range_parity_max = 0.0
    try:
        for batch_index, (images, _targets, metas) in enumerate(
            tqdm(loader, desc="counterfactual refined bank inference", ncols=96)
        ):
            if args.max_images is not None and len(records) >= int(args.max_images):
                break
            outputs = model(images.to(device, non_blocking=True))
            required_inputs = {
                "slot_states",
                "proposal_row_tokens",
                "proposal_x_rows",
                "proposal_range_norm",
                "candidate_valid",
                "row_value_features",
            }
            missing = required_inputs - set(captured)
            if missing:
                raise RuntimeError(f"refiner hook missed inputs: {sorted(missing)}")
            counterfactual = frozen_v7_counterfactual_anchors(
                refiner,
                slot_states=captured["slot_states"],
                proposal_row_tokens=captured["proposal_row_tokens"],
                proposal_x_rows=captured["proposal_x_rows"],
                proposal_range_norm=captured["proposal_range_norm"],
                candidate_valid=captured["candidate_valid"],
                row_value_features=captured["row_value_features"],
            )
            source_x = _required(outputs, "selection_slot_pred_x_rows")
            source_range = _required(outputs, "selection_slot_range_norm")
            source_routes = _required(
                outputs, "selection_slot_geometry_route_indices"
            ).long()
            source_active = _required(outputs, "selection_slot_active").bool()
            raw_x = captured["proposal_x_rows"].detach().float()
            raw_range = captured["proposal_range_norm"].detach().float()
            batch, slots, candidates, rows = counterfactual["x_rows"].shape
            raw_bank_x = raw_x.unsqueeze(1).expand(
                batch, slots, candidates, rows
            )
            raw_bank_range = raw_range.unsqueeze(1).expand(
                batch, slots, candidates, 2
            )
            safe_source = source_routes.clamp(min=0)
            batch_ids = torch.arange(batch, device=device).view(batch, 1)
            slot_ids = torch.arange(slots, device=device).view(1, slots)
            replayed_source_x = counterfactual["x_rows"][
                batch_ids, slot_ids, safe_source
            ]
            replayed_source_range = counterfactual["range_norm"][
                batch_ids, slot_ids, safe_source
            ]
            valid_source = source_routes >= 0
            if bool(valid_source.any()):
                source_refiner_x_parity_max = max(
                    source_refiner_x_parity_max,
                    float(
                        (
                            replayed_source_x[valid_source]
                            - source_x[valid_source]
                        )
                        .abs()
                        .max()
                    ),
                )
                source_refiner_range_parity_max = max(
                    source_refiner_range_parity_max,
                    float(
                        (
                            replayed_source_range[valid_source]
                            - source_range[valid_source]
                        )
                        .abs()
                        .max()
                    ),
                )
            for batch_item, meta in enumerate(metas):
                if args.max_images is not None and len(records) >= int(args.max_images):
                    break
                image_id = _image_id(
                    meta,
                    f"refined_bank_{batch_index:06d}_{batch_item}",
                )
                selected = source_routes[batch_item].detach().cpu().clone()
                mismatch = []
                if edits:
                    for slot in range(slots):
                        row = edits.get((image_id, slot))
                        if row is None:
                            continue
                        expected_anchor, selected_id = row
                        observed_anchor = int(source_routes[batch_item, slot])
                        if observed_anchor != expected_anchor:
                            mismatch.append(
                                {
                                    "slot": slot,
                                    "expected": expected_anchor,
                                    "observed": observed_anchor,
                                }
                            )
                        selected[slot] = int(selected_id)
                if mismatch:
                    raise RuntimeError(
                        f"V26/V7 anchor mismatch for {image_id}: {mismatch}"
                    )
                safe_selected = selected.clamp(min=0)
                selected_raw_x = raw_bank_x[
                    batch_item,
                    torch.arange(slots, device=device),
                    safe_selected.to(device),
                ]
                selected_raw_range = raw_bank_range[
                    batch_item,
                    torch.arange(slots, device=device),
                    safe_selected.to(device),
                ]
                selected_refined_x = counterfactual["x_rows"][
                    batch_item,
                    torch.arange(slots, device=device),
                    safe_selected.to(device),
                ]
                selected_refined_range = counterfactual["range_norm"][
                    batch_item,
                    torch.arange(slots, device=device),
                    safe_selected.to(device),
                ]
                geometry = torch.cat(
                    (
                        source_x[batch_item],
                        raw_bank_x[batch_item].reshape(
                            slots * candidates, rows
                        ),
                        counterfactual["x_rows"][batch_item].reshape(
                            slots * candidates, rows
                        ),
                        selected_raw_x,
                        selected_refined_x,
                    ),
                    dim=0,
                ).detach().float().cpu()
                ranges = torch.cat(
                    (
                        source_range[batch_item],
                        raw_bank_range[batch_item].reshape(slots * candidates, 2),
                        counterfactual["range_norm"][batch_item].reshape(
                            slots * candidates, 2
                        ),
                        selected_raw_range,
                        selected_refined_range,
                    ),
                    dim=0,
                ).detach().float().cpu()
                records.append(
                    {
                        "image_id": image_id,
                        "record": {
                            "meta": _plain_meta(meta),
                            "stages": {
                                "combined": {
                                    "pred_x_rows": geometry,
                                    "range_norm": ranges,
                                }
                            },
                        },
                        "slots": slots,
                        "candidates": candidates,
                        "source_routes": source_routes[batch_item].detach().cpu(),
                        "source_active": source_active[batch_item].detach().cpu(),
                        "selected_routes": selected,
                        "has_v26_replay": bool(edits),
                    }
                )
    finally:
        handle.remove()

    # Enlarging the private slot axis from four to S*N changes CUDA kernel
    # accumulation order.  Sub-millipixel drift is numerical, far below both
    # writer quantization and one raster pixel, and is recorded in the report.
    if source_refiner_x_parity_max > 5.0e-4:
        raise RuntimeError(
            "counterfactual refiner does not reproduce exact V7 source x: "
            f"max_abs={source_refiner_x_parity_max}"
        )
    if source_refiner_range_parity_max > 1.0e-6:
        raise RuntimeError(
            "counterfactual refiner does not reproduce exact V7 source range: "
            f"max_abs={source_refiner_range_parity_max}"
        )

    evaluated = _evaluate_records_processes(
        records,
        line_width=30.0,
        min_valid_rows=5,
        workers=int(args.metric_workers),
    )
    totals: dict[str, dict[str, float | int]] = {}
    details: list[dict[str, Any]] = []
    quality_cache_records: list[dict[str, Any]] = []
    refiner_delta: list[float] = []
    refiner_helps = 0
    refiner_hurts = 0
    refiner_ties = 0
    oracle_edit_counts: list[int] = []
    union_edit_counts: list[int] = []
    for item, metric in tqdm(
        zip(records, evaluated),
        total=len(records),
        desc="exact source-tied bank oracle",
        ncols=96,
    ):
        quality = metric["quality"].float()
        valid = metric["valid"].bool()
        slots = int(item["slots"])
        candidates = int(item["candidates"])
        source_start = 0
        raw_start = source_start + slots
        refined_start = raw_start + slots * candidates
        selected_raw_start = refined_start + slots * candidates
        selected_refined_start = selected_raw_start + slots
        source_routes = item["source_routes"].long()
        active = item["source_active"].bool()
        active_slots = torch.nonzero(active, as_tuple=False).flatten().tolist()
        source_columns = [source_start + slot for slot in active_slots]
        source_summary = _policy_summary(quality, source_columns)
        _add_total(totals, "exact_refined_v7", source_summary)
        raw_quality = quality[:, raw_start:refined_start].reshape(
            int(quality.shape[0]), slots, candidates
        )
        refined_quality = quality[
            :, refined_start:selected_raw_start
        ].reshape(int(quality.shape[0]), slots, candidates)
        raw_valid = valid[raw_start:refined_start].reshape(slots, candidates)
        refined_valid = valid[
            refined_start:selected_raw_start
        ].reshape(slots, candidates)
        if args.output_quality_cache:
            quality_cache_records.append(
                {
                    "image_id": item["image_id"],
                    "source_routes": source_routes.clone(),
                    "active": active.clone(),
                    "source_quality": quality[:, source_start:raw_start].clone(),
                    "source_valid": valid[source_start:raw_start].clone(),
                    "raw_quality": raw_quality.clone(),
                    "raw_valid": raw_valid.clone(),
                    "refined_quality": refined_quality.clone(),
                    "refined_valid": refined_valid.clone(),
                }
            )
        source_raw_columns = [
            raw_start + slot * candidates + int(source_routes[slot])
            for slot in active_slots
        ]
        _add_total(
            totals,
            "raw_v7_anchor",
            _policy_summary(quality, source_raw_columns),
        )
        if item["has_v26_replay"]:
            selected_raw_columns = [selected_raw_start + slot for slot in active_slots]
            selected_refined_columns = [
                selected_refined_start + slot for slot in active_slots
            ]
            _add_total(
                totals,
                "v26_selected_raw",
                _policy_summary(quality, selected_raw_columns),
            )
            _add_total(
                totals,
                "v26_selected_counterfactual_refined",
                _policy_summary(quality, selected_refined_columns),
            )
        raw_oracle = exact_source_tied_oracle(
            raw_quality,
            raw_valid,
            active,
            source_routes,
            device=args.oracle_device,
            chunk_size=int(args.oracle_chunk_size),
        )
        refined_oracle = exact_source_tied_oracle(
            refined_quality,
            refined_valid,
            active,
            source_routes,
            device=args.oracle_device,
            chunk_size=int(args.oracle_chunk_size),
        )
        _add_oracle_total(
            totals,
            "raw_bank_oracle",
            raw_oracle,
            predictions=len(active_slots),
            gt=int(quality.shape[0]),
        )
        _add_oracle_total(
            totals,
            "counterfactual_refined_bank_oracle",
            refined_oracle,
            predictions=len(active_slots),
            gt=int(quality.shape[0]),
        )
        oracle_edit_counts.append(refined_oracle.edit_count)
        union_oracle = None
        if args.raw_refined_union_oracle:
            union_oracle = exact_raw_refined_union_oracle(
                raw_quality,
                refined_quality,
                raw_valid,
                refined_valid,
                active,
                source_routes,
                device=args.oracle_device,
                chunk_size=int(args.oracle_chunk_size),
            )
            _add_oracle_total(
                totals,
                "raw_refined_union_oracle",
                union_oracle,
                predictions=len(active_slots),
                gt=int(quality.shape[0]),
            )
            union_edit_counts.append(union_oracle.edit_count)
        # Refiner population behavior is measured only on candidate/GT pairs
        # with a finite official raster value in both variants.
        common = (
            raw_valid.unsqueeze(0) & refined_valid.unsqueeze(0)
        ).expand_as(raw_quality)
        delta = (refined_quality - raw_quality)[common]
        if delta.numel():
            refiner_delta.extend(delta.tolist())
            refiner_helps += int((delta > 1.0e-6).sum())
            refiner_hurts += int((delta < -1.0e-6).sum())
            refiner_ties += int((delta.abs() <= 1.0e-6).sum())
        details.append(
            {
                "image_id": item["image_id"],
                "source_routes": source_routes.tolist(),
                "active": active.tolist(),
                "selected_routes": item["selected_routes"].tolist(),
                "source": source_summary,
                "raw_oracle": asdict(raw_oracle),
                "refined_oracle": asdict(refined_oracle),
                "union_oracle": (
                    None if union_oracle is None else asdict(union_oracle)
                ),
            }
        )

    policies = _finalize(totals)
    source = policies["exact_refined_v7"]
    capacity = policies["counterfactual_refined_bank_oracle"]
    report = {
        "experiment": "V26 refined bank and immutable proposal capacity audit",
        "contract": {
            "activity_count": "exact V7",
            "candidate_geometry": "raw and frozen V7 counterfactual refinement",
            "proposal_ids_injective": True,
            "oracle_objective": "TP50 -> TP75 -> source-on-tie -> matched IoU",
            "coordinate_blending": False,
            "checkpoint_selection": False,
            "threshold_selection": False,
            "test_set_used": False,
        },
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint).expanduser()),
        "iteration": iteration,
        "split": args.split,
        "list_path": str(Path(args.list_path).expanduser().resolve()),
        "list_sha256": sha256_file(Path(args.list_path).expanduser()),
        "images": len(records),
        "source_counterfactual_parity": {
            "max_abs_x_px": source_refiner_x_parity_max,
            "max_abs_range_norm": source_refiner_range_parity_max,
            "passed": True,
        },
        "v26_changed_lanes_jsonl": (
            None
            if not args.v26_changed_lanes_jsonl
            else str(Path(args.v26_changed_lanes_jsonl).expanduser().resolve())
        ),
        "policies": policies,
        "primary_capacity_delta": {
            "tp50": int(capacity["tp50"]) - int(source["tp50"]),
            "tp75": int(capacity["tp75"]) - int(source["tp75"]),
            "f1_50_points": float(capacity["f1_50_points"])
            - float(source["f1_50_points"]),
            "f1_75_points": float(capacity["f1_75_points"])
            - float(source["f1_75_points"]),
        },
        "refiner_population": {
            "all_gt_slot_candidate_delta": _quantiles(refiner_delta),
            "helps": refiner_helps,
            "hurts": refiner_hurts,
            "ties": refiner_ties,
        },
        "oracle_edit_count": _quantiles([float(x) for x in oracle_edit_counts]),
        "union_edit_count": _quantiles([float(x) for x in union_edit_counts]),
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_details_jsonl:
        details_path = Path(args.output_details_jsonl).expanduser()
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle_out:
            for row in details:
                handle_out.write(json.dumps(row, sort_keys=True) + "\n")
    if args.output_quality_cache:
        cache_path = Path(args.output_quality_cache).expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": {
                    "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "list_path": str(Path(args.list_path).expanduser().resolve()),
                    "list_sha256": report["list_sha256"],
                    "images": len(quality_cache_records),
                    "quality_contract": (
                        "exact official raster IoU; source geometry is exact V7; "
                        "raw/refined tensors are [GT,slot,proposal]"
                    ),
                },
                "records": quality_cache_records,
            },
            cache_path,
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "images": len(records),
                "source": policies["exact_refined_v7"],
                "refined_bank_oracle": policies[
                    "counterfactual_refined_bank_oracle"
                ],
                "capacity_delta": report["primary_capacity_delta"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
