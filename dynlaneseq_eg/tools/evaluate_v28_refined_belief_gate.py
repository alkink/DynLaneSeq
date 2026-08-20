from __future__ import annotations

import argparse
import copy
import hashlib
from itertools import repeat
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_metric import (
    discrete_cross_iou,
    interp,
    list_image_rel_paths,
    load_culane_img_data,
)
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v28 import DynLaneSeqV28
from dynlaneseq_eg.modeling.v23_ordered_slot_cost_volume import (
    build_v23_owned_targets,
)
from dynlaneseq_eg.modeling.v28_refined_belief_router import (
    build_v28_refined_route_targets,
    decode_v28_unique_routes,
    gather_slot_candidates,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


SEED = 3407
THRESHOLDS = (0.50, 0.75)
POLICIES = ("source_v7", "arm_b", "arm_c", "arm_c_wrong_image")
LINE_WIDTH = 30
MIN_POINTS = 5
TOP_K = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired exact-raster evaluation of V28 source, route-only B, "
            "route-plus-field C, and the cross-clip wrong-image C control."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm-b-checkpoint", required=True)
    parser.add_argument("--arm-b-training-report", required=True)
    parser.add_argument("--arm-c-router-checkpoint", required=True)
    parser.add_argument("--arm-c-training-report", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--wrong-image-list", required=True)
    parser.add_argument("--wrong-image-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--metric-workers", type=int, default=0)
    parser.add_argument("--metric-chunksize", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _configured(
    base: dict[str, Any],
    *,
    dataset_root: Path,
    list_path: Path,
    batch_size: int,
    workers: int,
    load_targets: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("dataset", {})["root"] = str(dataset_root)
    cfg["dataset"].setdefault("lists", {})["val"] = str(list_path)
    cfg["dataset"]["load_targets"] = bool(load_targets)
    cfg["dataset"]["infer_seg_labels"] = False
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(workers)
    cfg["dataloader"]["persistent_workers"] = int(workers) > 0
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _validate_endpoint(
    checkpoint: Path,
    report_path: Path,
    *,
    arm: str,
    router_only: bool,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    key = "router_only_checkpoint_sha256" if router_only else "checkpoint_sha256"
    checks = {
        "arm_exact": str(report.get("arm")) == str(arm),
        "iteration_exact": int(report.get("iteration", -1)) == 6_000,
        "checkpoint_sha256_exact": sha256_file(checkpoint) == str(report.get(key)),
        "gate_zero_passed": report.get("gate_zero", {}).get("passed") is True,
        "teacher_exact": report.get("teacher_state_still_exact") is True,
        "checkpoint_selection_absent": report.get("checkpoint_selection_performed")
        is False,
        "threshold_selection_absent": report.get("threshold_selection_performed")
        is False,
        "test_unused": report.get("test_set_used") is False,
    }
    if not all(checks.values()):
        raise ValueError(
            f"V28 arm {arm} endpoint contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "checks": checks,
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_report_sha256": sha256_file(report_path),
    }


def _validate_wrong_list(
    source_list: Path,
    wrong_list: Path,
    report_path: Path,
    *,
    expected: int,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = {
        "report_passed": report.get("passed") is True,
        "source_sha256_exact": str(report.get("input_sha256"))
        == sha256_file(source_list),
        "wrong_sha256_exact": str(report.get("output_sha256"))
        == sha256_file(wrong_list),
        "count_exact": int(report.get("image_count", -1)) == int(expected),
        "same_image_zero": int(report.get("same_image_partner_count", -1)) == 0,
        "same_clip_zero": int(report.get("same_clip_partner_count", -1)) == 0,
    }
    if not all(checks.values()):
        raise ValueError(
            "V28 wrong-image contract failed: " + json.dumps(checks, sort_keys=True)
        )
    return {"checks": checks, "report_sha256": sha256_file(report_path)}


def _move_images(
    images: torch.Tensor,
    *,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        return images.contiguous(memory_format=torch.channels_last)
    return images


def _belief_public(
    model: DynLaneSeqV28,
    router,
    images: torch.Tensor,
    bank: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    output = router(
        images,
        source_x=bank["source_x"],
        source_range=bank["source_range"],
        source_active=bank["source_active"],
        candidate_x=bank["candidate_x"],
        candidate_range=bank["candidate_range"],
        candidate_valid=bank["candidate_valid"],
    )
    route = decode_v28_unique_routes(
        output["candidate_scores"], output["candidate_valid"]
    )
    x_rows = gather_slot_candidates(bank["candidate_x"], route)
    ranges = gather_slot_candidates(bank["candidate_range"], route)
    public = model._public_output(
        selected_x=x_rows,
        selected_range=ranges,
        source_active=bank["source_active"],
        source_slot_indices=bank["source_slot_indices"],
    )
    return public, {**output, "selected_route": route}


def _empty_route_stats() -> dict[str, float]:
    return {
        "matched": 0.0,
        "top1": 0.0,
        "support_mass": 0.0,
        "selected_quality": 0.0,
        "source_quality": 0.0,
        "best_quality": 0.0,
        "changed": 0.0,
        "representable_gap": 0.0,
        "closed_gap": 0.0,
    }


@torch.no_grad()
def _batch_route_stats(
    output: dict[str, torch.Tensor],
    bank: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    input_h: int,
    input_w: int,
) -> list[dict[str, float]]:
    owned = build_v23_owned_targets(
        targets,
        source_x=bank["source_x"],
        source_range=bank["source_range"],
        source_active=bank["source_active"],
        input_h=input_h,
        input_w=input_w,
        minimum_valid_rows=5,
    )
    target = build_v28_refined_route_targets(
        candidate_x=bank["candidate_x"],
        candidate_range=bank["candidate_range"],
        candidate_valid=bank["candidate_valid"],
        owned_x=owned["x_rows"],
        owned_valid=owned["valid_mask"],
        owned_matched=owned["matched"],
        input_h=input_h,
        line_width=30.0,
        minimum_valid_rows=5,
        temperature=0.05,
        support_delta=0.05,
        support_floor=0.0,
    )
    logits = output["candidate_scores"].float().masked_fill(
        ~target["valid"], -1.0e4
    )
    probability = logits.softmax(dim=-1)
    target_best = target["quality"].masked_fill(
        ~target["valid"], -1.0e4
    ).argmax(dim=-1)
    predicted = output["selected_route"]
    selected_quality = target["quality"].gather(
        -1, predicted.unsqueeze(-1)
    ).squeeze(-1)
    source_quality = target["quality"].gather(
        -1, bank["source_route"].clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    support_mass = (probability * target["support"].float()).sum(dim=-1)
    matched = target["matched"]
    matched_float = matched.float()
    representable = matched & (
        target["best_quality"] > source_quality + 0.05
    )
    representable_float = representable.float()
    names = tuple(_empty_route_stats())
    packed = torch.stack(
        (
            matched_float.sum(dim=-1),
            ((predicted == target_best) & matched).float().sum(dim=-1),
            (support_mass * matched_float).sum(dim=-1),
            (selected_quality * matched_float).sum(dim=-1),
            (source_quality * matched_float).sum(dim=-1),
            (target["best_quality"] * matched_float).sum(dim=-1),
            ((predicted != bank["source_route"]) & matched).float().sum(dim=-1),
            (
                (target["best_quality"] - source_quality)
                * representable_float
            ).sum(dim=-1),
            (
                (selected_quality - source_quality) * representable_float
            ).sum(dim=-1),
        ),
        dim=-1,
    ).cpu()
    return [dict(zip(names, values)) for values in packed.tolist()]


@torch.inference_mode()
def _write_predictions(
    model: DynLaneSeqV28,
    arm_c_router,
    source_loader,
    wrong_loader,
    *,
    device: torch.device,
    output_dirs: dict[str, Path],
    channels_last: bool,
    log_interval: int,
    input_h: int,
    input_w: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.requires_grad_(False).eval()
    arm_c_router.requires_grad_(False).eval()
    model.prepare_for_inference()
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    image_count = 0
    started = time.perf_counter()
    for batch_index, (source_batch, wrong_batch) in enumerate(
        zip(source_loader, wrong_loader), start=1
    ):
        images, targets, metas = source_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        if int(images.shape[0]) != int(wrong_images.shape[0]):
            raise ValueError("V28 correct/wrong batch mismatch")
        for meta, wrong_meta in zip(metas, wrong_metas):
            source_clip = str(Path(str(meta["image_path"])).parent)
            wrong_clip = str(Path(str(wrong_meta["image_path"])).parent)
            if source_clip == wrong_clip:
                raise ValueError("V28 runtime wrong-image pair shares a clip")
        images = _move_images(
            images, device=device, channels_last=channels_last
        )
        wrong_images = _move_images(
            wrong_images, device=device, channels_last=channels_last
        )
        with torch.autocast(device_type=device.type, enabled=False):
            _teacher, bank = model._frozen_bank(images.float())
            source_public = model._public_output(
                selected_x=bank["source_x"],
                selected_range=bank["source_range"],
                source_active=bank["source_active"],
                source_slot_indices=bank["source_slot_indices"],
            )
            arm_b_public, arm_b_internal = _belief_public(
                model, model.router, images.float(), bank
            )
            arm_c_public, arm_c_internal = _belief_public(
                model, arm_c_router, images.float(), bank
            )
            arm_c_wrong_public, _ = _belief_public(
                model, arm_c_router, wrong_images.float(), bank
            )
        outputs = {
            "source_v7": source_public,
            "arm_b": arm_b_public,
            "arm_c": arm_c_public,
            "arm_c_wrong_image": arm_c_wrong_public,
        }
        for policy, output in outputs.items():
            write_culane_predictions(
                output,
                metas,
                output_dirs[policy],
                score_thresh=0.5,
                min_pred_points=MIN_POINTS,
                nms_distance_thresh_px=0.0,
                top_k=TOP_K,
                quality_score_power=0.0,
                score_mode="exist",
            )
        arm_b_stats = _batch_route_stats(
            arm_b_internal,
            bank,
            targets,
            input_h=input_h,
            input_w=input_w,
        )
        arm_c_stats = _batch_route_stats(
            arm_c_internal,
            bank,
            targets,
            input_h=input_h,
            input_w=input_w,
        )
        for meta, stats_b, stats_c in zip(metas, arm_b_stats, arm_c_stats):
            rows.append(
                {
                    "image_path": str(meta["image_path"]),
                    "arm_b": stats_b,
                    "arm_c": stats_c,
                }
            )
        image_count += int(images.shape[0])
        if batch_index == 1 or (
            log_interval > 0 and batch_index % log_interval == 0
        ):
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            print(
                json.dumps(
                    {
                        "phase": "v28_paired_writer",
                        "batches": batch_index,
                        "images": image_count,
                        "images_per_second": image_count / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if image_count != len(source_loader.dataset):
        raise RuntimeError("V28 paired writer did not consume the full split")
    return {
        "images": image_count,
        "elapsed_seconds": time.perf_counter() - started,
    }, rows


def _clip_domain(rel: str) -> str:
    clip = str(Path(rel).parent)
    digest = hashlib.sha256(f"{SEED}:v28-val-half:{clip}".encode()).digest()
    return "validation_half_a" if digest[0] % 2 == 0 else "validation_half_b"


def _policy_counts(
    prediction: list[list[tuple[float, float]]],
    annotation: list[list[tuple[float, float]]],
) -> dict[str, tuple[int, int, int, tuple[int, ...]]]:
    pred = [interp(lane, n=5) for lane in prediction if len(lane) >= 2]
    anno = [interp(lane, n=5) for lane in annotation if len(lane) >= 2]
    if pred and anno:
        ious = discrete_cross_iou(
            pred, anno, width=LINE_WIDTH, img_shape=(590, 1640)
        )
        pred_ids, gt_ids = linear_sum_assignment(1.0 - ious)
        matched_values = ious[pred_ids, gt_ids]
    else:
        gt_ids = np.zeros((0,), dtype=np.int64)
        matched_values = np.zeros((0,), dtype=np.float32)
    result = {}
    for threshold in THRESHOLDS:
        hit = matched_values > threshold
        matched = tuple(int(value) for value in gt_ids[hit])
        tp = len(matched)
        result[f"{threshold:.2f}"] = (
            tp,
            len(pred) - tp,
            len(anno) - tp,
            matched,
        )
    return result


def _metric_image(
    rel: str,
    dataset_root: str,
    prediction_roots: dict[str, str],
) -> dict[str, Any]:
    annotation = load_culane_img_data(
        Path(dataset_root) / rel.replace(".jpg", ".lines.txt")
    )
    result = {"rel": rel, "policies": {}, "prediction_count": {}}
    for policy, root in prediction_roots.items():
        prediction = load_culane_img_data(
            Path(root) / rel.replace(".jpg", ".lines.txt")
        )
        result["prediction_count"][policy] = len(prediction)
        result["policies"][policy] = _policy_counts(prediction, annotation)
    return result


def _empty_count() -> dict[str, int]:
    return {"TP": 0, "FP": 0, "FN": 0}


def _finished(count: dict[str, int]) -> dict[str, float | int]:
    precision = count["TP"] / max(count["TP"] + count["FP"], 1)
    recall = count["TP"] / max(count["TP"] + count["FN"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {**count, "Precision": precision, "Recall": recall, "F1": f1}


def _summarize_domain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        policy: {f"{threshold:.2f}": _empty_count() for threshold in THRESHOLDS}
        for policy in POLICIES
    }
    source_correct = {
        policy: {f"{threshold:.2f}": [0, 0] for threshold in THRESHOLDS}
        for policy in ("arm_b", "arm_c")
    }
    equal_count = {policy: 0 for policy in POLICIES if policy != "source_v7"}
    for row in rows:
        for policy in equal_count:
            equal_count[policy] += int(
                row["prediction_count"][policy]
                == row["prediction_count"]["source_v7"]
            )
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            for policy in POLICIES:
                tp, fp, fn, _matched = row["policies"][policy][key]
                counts[policy][key]["TP"] += tp
                counts[policy][key]["FP"] += fp
                counts[policy][key]["FN"] += fn
            source_set = set(row["policies"]["source_v7"][key][3])
            for policy in ("arm_b", "arm_c"):
                policy_set = set(row["policies"][policy][key][3])
                source_correct[policy][key][0] += len(source_set)
                source_correct[policy][key][1] += len(source_set - policy_set)
    metrics = {
        policy: {key: _finished(value) for key, value in threshold.items()}
        for policy, threshold in counts.items()
    }
    degradation = {
        policy: {
            key: {
                "source_correct": values[0],
                "lost": values[1],
                "fraction": values[1] / max(values[0], 1),
            }
            for key, values in threshold.items()
        }
        for policy, threshold in source_correct.items()
    }
    return {
        "images": len(rows),
        "metrics": metrics,
        "source_correct_degradation": degradation,
        "cardinality_exact_source_fraction": {
            policy: value / max(len(rows), 1)
            for policy, value in equal_count.items()
        },
    }


def _evaluate(
    *,
    list_path: Path,
    dataset_root: Path,
    output_dirs: dict[str, Path],
    workers: int,
    chunksize: int,
) -> dict[str, Any]:
    rels = list_image_rel_paths(list_path)
    roots = {key: str(value) for key, value in output_dirs.items()}
    tasks = zip(rels, repeat(str(dataset_root)), repeat(roots))
    worker_count = int(workers) if int(workers) > 0 else cpu_count()
    if worker_count <= 1:
        rows = [_metric_image(rel, str(dataset_root), roots) for rel in rels]
    else:
        with Pool(worker_count) as pool:
            rows = list(
                tqdm(
                    pool.starmap(
                        _metric_image,
                        tasks,
                        chunksize=max(int(chunksize), 1),
                    ),
                    total=len(rels),
                    desc="V28 paired exact raster",
                    ncols=88,
                )
            )
    half_a = [row for row in rows if _clip_domain(row["rel"]) == "validation_half_a"]
    half_b = [row for row in rows if _clip_domain(row["rel"]) == "validation_half_b"]
    return {
        "full_validation": _summarize_domain(rows),
        "validation_half_a": _summarize_domain(half_a),
        "validation_half_b": _summarize_domain(half_b),
        "clip_disjoint_halves": True,
    }


def _finish_route_stats(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for arm in ("arm_b", "arm_c"):
        total = _empty_route_stats()
        for row in rows:
            for key, value in row[arm].items():
                total[key] += float(value)
        result[arm] = {
            "matched_slots": total["matched"],
            "target_top1": total["top1"] / max(total["matched"], 1.0),
            "target_support_probability_mass": total["support_mass"]
            / max(total["matched"], 1.0),
            "mean_selected_quality": total["selected_quality"]
            / max(total["matched"], 1.0),
            "mean_source_quality": total["source_quality"]
            / max(total["matched"], 1.0),
            "mean_best_quality": total["best_quality"]
            / max(total["matched"], 1.0),
            "changed_fraction": total["changed"] / max(total["matched"], 1.0),
            "oracle_gap_closure": total["closed_gap"]
            / max(total["representable_gap"], 1.0e-12),
        }
    return result


def _gate(domains: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    deltas: dict[str, Any] = {}
    for domain_name in ("validation_half_a", "validation_half_b", "full_validation"):
        domain = domains[domain_name]
        metrics = domain["metrics"]
        source50, source75 = metrics["source_v7"]["0.50"], metrics["source_v7"]["0.75"]
        b50, b75 = metrics["arm_b"]["0.50"], metrics["arm_b"]["0.75"]
        c50, c75 = metrics["arm_c"]["0.50"], metrics["arm_c"]["0.75"]
        wrong50, wrong75 = metrics["arm_c_wrong_image"]["0.50"], metrics["arm_c_wrong_image"]["0.75"]
        c_minus_b_50 = 100.0 * (float(c50["F1"]) - float(b50["F1"]))
        c_minus_b_75 = 100.0 * (float(c75["F1"]) - float(b75["F1"]))
        c_minus_source_50 = 100.0 * (float(c50["F1"]) - float(source50["F1"]))
        c_minus_source_75 = 100.0 * (float(c75["F1"]) - float(source75["F1"]))
        prefix = domain_name
        checks[f"{prefix}_c_beats_b_0p20_f1_50"] = c_minus_b_50 >= 0.20
        checks[f"{prefix}_c_nonregression_b_f1_75"] = c_minus_b_75 >= 0.0
        checks[f"{prefix}_c_nonregression_source_f1_50"] = c_minus_source_50 >= 0.0
        checks[f"{prefix}_c_nonregression_source_f1_75"] = c_minus_source_75 >= 0.0
        checks[f"{prefix}_source_correct_loss_below_1pct_50"] = float(
            domain["source_correct_degradation"]["arm_c"]["0.50"]["fraction"]
        ) < 0.01
        checks[f"{prefix}_source_correct_loss_below_1pct_75"] = float(
            domain["source_correct_degradation"]["arm_c"]["0.75"]["fraction"]
        ) < 0.01
        checks[f"{prefix}_correct_image_beats_wrong_tp_50"] = int(c50["TP"]) > int(wrong50["TP"])
        checks[f"{prefix}_correct_image_beats_wrong_tp_75"] = int(c75["TP"]) > int(wrong75["TP"])
        checks[f"{prefix}_cardinality_exact"] = float(
            domain["cardinality_exact_source_fraction"]["arm_c"]
        ) == 1.0
        deltas[domain_name] = {
            "c_minus_b_f1_50_points": c_minus_b_50,
            "c_minus_b_f1_75_points": c_minus_b_75,
            "c_minus_source_f1_50_points": c_minus_source_50,
            "c_minus_source_f1_75_points": c_minus_source_75,
            "c_minus_wrong_tp_50": int(c50["TP"]) - int(wrong50["TP"]),
            "c_minus_wrong_tp_75": int(c75["TP"]) - int(wrong75["TP"]),
        }
    return {"passed": all(checks.values()), "checks": checks, "deltas": deltas}


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(dataset_root, split="val")
    source_list = Path(population["list_path"])
    expected = int(population["expected_nonempty_rows"])
    wrong_list = Path(args.wrong_image_list).expanduser().resolve()
    wrong_contract = _validate_wrong_list(
        source_list,
        wrong_list,
        Path(args.wrong_image_report).expanduser().resolve(),
        expected=expected,
    )
    arm_b_checkpoint = Path(args.arm_b_checkpoint).expanduser().resolve()
    arm_c_checkpoint = Path(args.arm_c_router_checkpoint).expanduser().resolve()
    endpoint_contracts = {
        "arm_b": _validate_endpoint(
            arm_b_checkpoint,
            Path(args.arm_b_training_report).expanduser().resolve(),
            arm="B",
            router_only=False,
        ),
        "arm_c": _validate_endpoint(
            arm_c_checkpoint,
            Path(args.arm_c_training_report).expanduser().resolve(),
            arm="C",
            router_only=True,
        ),
    }

    base_cfg: dict[str, Any] = load_config(args.config)
    source_cfg = _configured(
        base_cfg,
        dataset_root=dataset_root,
        list_path=source_list,
        batch_size=int(args.eval_batch_size),
        workers=int(args.num_workers),
        load_targets=True,
    )
    wrong_cfg = _configured(
        base_cfg,
        dataset_root=dataset_root,
        list_path=wrong_list,
        batch_size=int(args.eval_batch_size),
        workers=int(args.num_workers),
        load_targets=False,
    )
    source_loader = build_dataloader(source_cfg, split="val", training=False)
    wrong_loader = build_dataloader(wrong_cfg, split="val", training=False)
    if len(source_loader.dataset) != expected or len(wrong_loader.dataset) != expected:
        raise ValueError("V28 evaluator altered official validation population")

    device = torch.device(args.device)
    model = build_model(source_cfg)
    if not isinstance(model, DynLaneSeqV28):
        raise TypeError("V28 evaluator did not build DynLaneSeqV28")
    arm_b_iteration = int(load_checkpoint(arm_b_checkpoint, model, strict=True))
    arm_c_router = copy.deepcopy(model.router)
    arm_c_iteration = int(load_checkpoint(arm_c_checkpoint, arm_c_router, strict=True))
    if arm_b_iteration != 6_000 or arm_c_iteration != 6_000:
        raise ValueError("V28 evaluator requires fixed 6000-step endpoints")
    model.to(device)
    arm_c_router.to(device)
    channels_last = bool(
        source_cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model.to(memory_format=torch.channels_last)
        arm_c_router.to(memory_format=torch.channels_last)

    output_dir = Path(args.output_dir).expanduser().resolve()
    prediction_dirs = {
        policy: output_dir / "predictions" / policy for policy in POLICIES
    }
    writer, route_rows = _write_predictions(
        model,
        arm_c_router,
        source_loader,
        wrong_loader,
        device=device,
        output_dirs=prediction_dirs,
        channels_last=channels_last,
        log_interval=int(args.log_interval),
        input_h=int(source_cfg["model"]["input_h"]),
        input_w=int(source_cfg["model"]["input_w"]),
    )
    del model, arm_c_router, source_loader, wrong_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    domains = _evaluate(
        list_path=source_list,
        dataset_root=dataset_root,
        output_dirs=prediction_dirs,
        workers=int(args.metric_workers),
        chunksize=int(args.metric_chunksize),
    )
    route_diagnostics = _finish_route_stats(route_rows)
    gate = _gate(domains)
    report = {
        "experiment": "V28 immutable refined-bank belief B/C official gate",
        "endpoint_contracts": endpoint_contracts,
        "wrong_image_contract": wrong_contract,
        "official_validation_population_contract": population,
        "writer": writer,
        "route_diagnostics": route_diagnostics,
        "domains": domains,
        "gate": gate,
        "contract": {
            "full_official_validation": True,
            "validation_rows_removed": 0,
            "clip_disjoint_deterministic_halves": True,
            "prediction_count_owner": "exact_v7_activity",
            "candidate_geometry": "immutable_counterfactual_refined_v7",
            "nms_enabled": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "test_set_used": False,
            "inference_dtype": "float32",
            "optimized_single_teacher_paired_writer": True,
            "exact_parallel_raster_metric": True,
        },
        "recommendation": (
            "mechanism_passed_review_before_test"
            if gate["passed"]
            else "stop_v28_mechanism_gate_failed"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "official_val_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    route_path = output_dir / "route_diagnostics.jsonl"
    with route_path.open("w", encoding="utf-8") as handle:
        for row in route_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
