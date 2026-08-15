from __future__ import annotations

import os

# Exact-raster workers are process-parallel.  Bound nested numerical-library
# threads before NumPy/SciPy/Torch are imported in spawned children.
for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

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
    build_owned_gt_operator_fields,
    correct_curves_from_distance,
    source_seeded_field_path,
)
from dynlaneseq_eg.tools.audit_v11_causal_replay import _image_id, _plain_meta
from dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_official import _required
from dynlaneseq_eg.tools.audit_v22_lane_field_utilization import (
    _dataset_config,
    _dataset_relative_image_id,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v20_cached_replacement import _load_cache
from dynlaneseq_eg.tools.v22_official_protocol import official_culane_list_contract


FIXED_SEED = 3407
POLICIES = (
    "source_v7",
    "exact_owned_gt_curve",
    "exact_owned_distance_step1",
    "exact_owned_centerline_path",
)
PATH_OFFSET_STEP_PX = 4.0
PATH_TRANSITION_SCALE_PX = 8.0
DISTANCE_STRICT_P99_MAX_PX = 0.05
PATH_STRICT_P90_MAX_PX = 2.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free exact owned-GT displacement/path operator oracle "
            "on every untouched official CULane validation row."
        )
    )
    parser.add_argument("--field-config", required=True)
    parser.add_argument("--v20-config", required=True)
    parser.add_argument("--v20-geometry-checkpoint", required=True)
    parser.add_argument("--v20-cache-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    return parser.parse_args()


def _hits(quality: torch.Tensor, active: torch.Tensor, threshold: float) -> int:
    selected = torch.nonzero(active.bool(), as_tuple=False).flatten().tolist()
    return int(
        evaluator_hungarian_assignment(
            quality.float(), selected, threshold=float(threshold)
        ).hit_count
    )


def _metric_worker(payload: dict[str, Any]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    curves = payload["curves"]
    ranges = payload["ranges"]
    flat_curves = [curves[name].float() for name in POLICIES]
    flat_ranges = [ranges[name].float() for name in POLICIES]
    slots = int(flat_curves[0].shape[0])
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
    active = payload["source_active"].bool()
    result: dict[str, Any] = {"policies": {}}
    for index, name in enumerate(POLICIES):
        local = quality[:, index * slots : (index + 1) * slots]
        result["policies"][name] = {
            "tp50": _hits(local, active, 0.50),
            "tp75": _hits(local, active, 0.75),
            "predictions": int(active.sum()),
            "gt": int(quality.shape[0]),
        }
    source_valid = raster_valid[:slots]
    result["source_raster_invalid_active"] = int((active & ~source_valid).sum())
    result["gt_count_cache_mismatch"] = int(
        int(quality.shape[0]) != int(payload["gt_count"])
    )
    return result


class _MetricAccumulator:
    def __init__(self) -> None:
        self.images = 0
        self.totals = {name: Counter() for name in POLICIES}
        self.image_outcomes = {
            name: {threshold: Counter() for threshold in (0.50, 0.75)}
            for name in POLICIES
            if name != "source_v7"
        }
        self.contract = Counter()

    def add(self, result: dict[str, Any]) -> None:
        self.images += 1
        source = result["policies"]["source_v7"]
        for name, row in result["policies"].items():
            self.totals[name].update(row)
            if name == "source_v7":
                continue
            for threshold, key in ((0.50, "tp50"), (0.75, "tp75")):
                delta = int(row[key]) - int(source[key])
                bucket = "improved" if delta > 0 else "worsened" if delta < 0 else "same"
                self.image_outcomes[name][threshold][bucket] += 1
        self.contract["source_raster_invalid_active"] += int(
            result["source_raster_invalid_active"]
        )
        self.contract["gt_count_cache_mismatch"] += int(
            result["gt_count_cache_mismatch"]
        )

    def finalize(self) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for name, row in self.totals.items():
            predictions = int(row["predictions"])
            gt = int(row["gt"])
            policy: dict[str, Any] = {}
            for threshold, key in ((0.50, "tp50"), (0.75, "tp75")):
                tp = int(row[key])
                fp, fn = predictions - tp, gt - tp
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
                        self.image_outcomes[name][threshold]
                    )
            report[name] = policy
        return report


class _RowAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.counts = Counter()

    def add(
        self,
        *,
        targets: list[dict[str, torch.Tensor]],
        ownership: torch.Tensor,
        active: torch.Tensor,
        source_x: torch.Tensor,
        source_range: torch.Tensor,
        distance_x: torch.Tensor,
        path_x: torch.Tensor,
        input_w: int,
        distance_limit_px: float,
        bin_width_px: float,
    ) -> None:
        rows = int(source_x.shape[-1])
        row_axis = torch.linspace(0.0, 1.0, rows, device=source_x.device)
        for item, target in enumerate(targets):
            gt_x = target["x_rows"].to(source_x.device).float()[..., :rows]
            gt_valid = target["valid_mask"].to(source_x.device).bool()[..., :rows]
            for slot in range(int(source_x.shape[1])):
                owned = int(ownership[item, slot])
                if not bool(active[item, slot]) or owned < 0 or owned >= int(gt_x.shape[0]):
                    continue
                valid = (
                    gt_valid[owned]
                    & torch.isfinite(gt_x[owned])
                    & torch.isfinite(source_x[item, slot])
                    & (gt_x[owned] >= 0.0)
                    & (gt_x[owned] < float(input_w))
                    & (source_x[item, slot] >= 0.0)
                    & (source_x[item, slot] < float(input_w))
                    & (row_axis >= source_range[item, slot, 0])
                    & (row_axis <= source_range[item, slot, 1])
                )
                if not bool(valid.any()):
                    continue
                actual = gt_x[owned] - source_x[item, slot]
                strict = valid & (
                    actual.abs() <= float(distance_limit_px) - float(bin_width_px)
                )
                outside = valid & (actual.abs() > float(distance_limit_px))
                self.counts["owned_visible_rows"] += int(valid.sum())
                self.counts["strict_in_support_rows"] += int(strict.sum())
                self.counts["outside_support_rows"] += int(outside.sum())
                for name, value in (
                    ("source_error", (gt_x[owned] - source_x[item, slot]).abs()),
                    ("distance_error", (gt_x[owned] - distance_x[item, slot]).abs()),
                    ("path_error", (gt_x[owned] - path_x[item, slot]).abs()),
                ):
                    self.values[f"all/{name}"].append(value[valid].cpu())
                    if bool(strict.any()):
                        self.values[f"strict/{name}"].append(value[strict].cpu())

    @staticmethod
    def _summary(chunks: list[torch.Tensor]) -> dict[str, float | int]:
        if not chunks:
            return {
                "count": 0,
                "mean": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "p99": float("nan"),
                "maximum": float("nan"),
            }
        value = torch.cat(chunks).float()
        return {
            "count": int(value.numel()),
            "mean": float(value.mean()),
            "p50": float(torch.quantile(value, 0.50)),
            "p90": float(torch.quantile(value, 0.90)),
            "p99": float(torch.quantile(value, 0.99)),
            "maximum": float(value.max()),
        }

    def finalize(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "errors_px": {
                name: self._summary(values) for name, values in self.values.items()
            },
        }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if int(args.seed) != FIXED_SEED:
        raise ValueError(f"exact operator oracle requires seed {FIXED_SEED}")
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
    model = build_model(v20_cfg).to(device)
    geometry_iteration = int(
        load_checkpoint(args.v20_geometry_checkpoint, model, strict=False)
    )
    model.requires_grad_(False).eval()
    cache, cache_manifest = _load_cache(
        Path(args.v20_cache_manifest).expanduser().resolve()
    )
    if str(cache_manifest.get("list_sha256")) != sha256_file(args.val_list):
        raise ValueError("exact operator oracle cache/list digest mismatch")
    images_total = int(cache["source_route"].shape[0])
    if images_total != int(val_contract["expected_nonempty_rows"]):
        raise ValueError("exact operator oracle requires every official val row")
    loader = build_dataloader(v20_cfg, split="val", training=False)
    if len(loader.dataset) != images_total:
        raise ValueError("exact operator oracle loader/cache population mismatch")

    model_cfg = field_cfg["model"]
    stage_cfg = field_cfg["v22_stage_a"]
    input_w = int(model_cfg["input_w"])
    x_bins = int(model_cfg["x_bins"])
    distance_limit_px = float(stage_cfg["distance_limit_px"])
    centerline_sigma_px = float(stage_cfg["centerline_sigma_px"])
    bin_width_px = float(input_w) / float(x_bins)
    metric_accumulator = _MetricAccumulator()
    row_accumulator = _RowAccumulator()
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
        progress = tqdm(loader, desc="V22 exact GT operator full official val", ncols=100)
        for images, targets, metas in progress:
            batch = int(images.shape[0])
            start, stop = global_index, global_index + batch
            expected_ids = [
                _dataset_relative_image_id(value, cache_manifest["dataset_root"])
                for value in cache_manifest["image_ids"][start:stop]
            ]
            actual_ids = [
                _dataset_relative_image_id(
                    _image_id(meta, f"v22_gt_operator_{start + item:06d}"),
                    args.dataset_root,
                )
                for item, meta in enumerate(metas)
            ]
            contract["image_id_mismatch"] += sum(
                left != right for left, right in zip(expected_ids, actual_ids)
            )
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            cached = {
                key: value[start:stop].to(device, non_blocking=True)
                for key, value in cache.items()
            }
            source_route = cached["source_route"].long()
            source_active = cached["source_active"].bool()
            contract["live_source_route_mismatch"] += int(
                (
                    _required(outputs, "selection_slot_v20_v7_geometry_route_indices").long()
                    != source_route
                ).sum().cpu()
            )
            contract["live_source_active_mismatch"] += int(
                (
                    _required(outputs, "selection_slot_v20_v7_active").bool()
                    != source_active
                ).sum().cpu()
            )
            cf_x = _required(outputs, "selection_slot_v19_counterfactual_x_rows").float()
            cf_range = _required(
                outputs, "selection_slot_v19_counterfactual_range_norm"
            ).float()
            source_x = _gather_candidate(cf_x, source_route)
            source_range = _gather_candidate(cf_range, source_route)
            slots, rows = int(source_x.shape[1]), int(source_x.shape[2])
            built = build_owned_gt_operator_fields(
                targets,
                ownership=cached["ownership"].long(),
                slots=slots,
                rows=rows,
                x_bins=x_bins,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
                centerline_sigma_px=centerline_sigma_px,
                device=device,
            )
            flat_source = source_x.reshape(batch * slots, 1, rows)
            flat_range = source_range.reshape(batch * slots, 1, 2)
            exact_distance = correct_curves_from_distance(
                built["distance_outputs"],
                x_rows=flat_source,
                range_norm=flat_range,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
                steps=1,
            )[0].reshape(batch, slots, rows)
            exact_path = source_seeded_field_path(
                built["path_outputs"],
                source_x=flat_source,
                source_range=flat_range,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
                offset_step_px=PATH_OFFSET_STEP_PX,
                transition_scale_px=PATH_TRANSITION_SCALE_PX,
            ).reshape(batch, slots, rows)
            owned_gt = torch.where(
                built["owned_valid"], built["owned_x"], source_x
            )
            row_accumulator.add(
                targets=targets,
                ownership=cached["ownership"].long(),
                active=source_active,
                source_x=source_x,
                source_range=source_range,
                distance_x=exact_distance,
                path_x=exact_path,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
                bin_width_px=bin_width_px,
            )
            for item in range(batch):
                ranges = {
                    name: source_range[item].cpu() for name in POLICIES
                }
                submit(
                    {
                        "meta": _plain_meta(metas[item]),
                        "curves": {
                            "source_v7": source_x[item].cpu(),
                            "exact_owned_gt_curve": owned_gt[item].cpu(),
                            "exact_owned_distance_step1": exact_distance[item].cpu(),
                            "exact_owned_centerline_path": exact_path[item].cpu(),
                        },
                        "ranges": ranges,
                        "source_active": source_active[item].cpu(),
                        "gt_count": int(cached["gt_count"][item]),
                    }
                )
            global_index = stop
            progress.set_postfix(
                metric_done=metric_accumulator.images, pending=len(pending)
            )
        while pending:
            metric_accumulator.add(pending.popleft().result())
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if global_index != images_total:
        raise RuntimeError("exact operator oracle did not consume full val population")
    metric_report = metric_accumulator.finalize()
    row_report = row_accumulator.finalize()
    contract.update(metric_accumulator.contract)
    distance_p99 = float(
        row_report["errors_px"]["strict/distance_error"]["p99"]
    )
    path_p90 = float(row_report["errors_px"]["strict/path_error"]["p90"])
    source50 = float(metric_report["source_v7"]["0.50"]["f1"])
    source75 = float(metric_report["source_v7"]["0.75"]["f1"])
    distance50 = float(
        metric_report["exact_owned_distance_step1"]["0.50"]["f1"]
    )
    distance75 = float(
        metric_report["exact_owned_distance_step1"]["0.75"]["f1"]
    )
    path50 = float(metric_report["exact_owned_centerline_path"]["0.50"]["f1"])
    path75 = float(metric_report["exact_owned_centerline_path"]["0.75"]["f1"])
    distance_operator_pass = (
        distance_p99 <= DISTANCE_STRICT_P99_MAX_PX
        and distance50 >= source50
        and distance75 >= source75
    )
    path_operator_pass = (
        path_p90 <= PATH_STRICT_P90_MAX_PX
        and path50 >= source50
        and path75 >= source75
    )
    data_contract_pass = (
        int(contract["image_id_mismatch"]) == 0
        and int(contract["gt_count_cache_mismatch"]) == 0
        and metric_accumulator.images == images_total
    )
    report = {
        "experiment": "V22 exact owned-GT operator semantic oracle",
        "images": images_total,
        "field_config": str(Path(args.field_config).expanduser().resolve()),
        "field_config_sha256": sha256_file(args.field_config),
        "v20_config": str(Path(args.v20_config).expanduser().resolve()),
        "v20_geometry_checkpoint_sha256": sha256_file(
            args.v20_geometry_checkpoint
        ),
        "v20_geometry_iteration": geometry_iteration,
        "v20_cache_manifest_sha256": sha256_file(args.v20_cache_manifest),
        "official_validation_population_contract": val_contract,
        "fixed_diagnostic_contract": {
            "training_performed": False,
            "test_set_used": False,
            "validation_subset_used": False,
            "validation_rows_removed": 0,
            "validation_deduplication_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "gt_used_for_diagnostic_oracle_only": True,
            "same_stage_a_sampler_update_and_path_code": True,
            "distance_limit_px": distance_limit_px,
            "x_bins": x_bins,
            "bin_width_px": bin_width_px,
            "path_offset_step_px": PATH_OFFSET_STEP_PX,
            "path_transition_scale_px": PATH_TRANSITION_SCALE_PX,
            "distance_strict_p99_max_px": DISTANCE_STRICT_P99_MAX_PX,
            "path_strict_p90_max_px": PATH_STRICT_P90_MAX_PX,
        },
        "official_raster": metric_report,
        "row_operator_oracle": row_report,
        "operator_contract": {
            "distance_operator_pass": distance_operator_pass,
            "path_operator_pass": path_operator_pass,
            "all_operator_contracts_pass": bool(
                distance_operator_pass and path_operator_pass
            ),
        },
        "contract": {
            **dict(contract),
            "all_official_val_images_evaluated": metric_accumulator.images
            == images_total,
            "passed_data_contract": data_contract_pass,
        },
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True))
    print(
        "V22 exact owned-GT operator oracle completed on all 9,675 official "
        "val.txt rows; no training, filtering, deduplication, checkpoint/"
        "threshold selection, or test evaluation was performed."
    )


if __name__ == "__main__":
    main()
