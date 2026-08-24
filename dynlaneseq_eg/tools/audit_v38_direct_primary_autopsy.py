from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    cardinality_oracle_assignment,
    ensure_official_iou_cache,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    sha256_file,
    stage_scores,
)
from dynlaneseq_eg.tools.audit_v30_field_geometry_drift import (
    _add_curve,
    _finish_curve_stats,
    _new_curve_stats,
    _official_target_rows,
)
from dynlaneseq_eg.tools.evaluate_v25_g0_and_path_gate import (
    _finish_geometry,
    _geometry_counts,
    _merge_geometry,
)


THRESHOLDS = (0.50, 0.75)
STAGE_NAME = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free full-validation autopsy of the V38 direct-primary "
            "50K endpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v38-report", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--official-workers", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    return parser.parse_args()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _policy_metrics(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": precision,
        "Recall": recall,
        "F1": _safe_ratio(2.0 * precision * recall, precision + recall),
    }


def _binary_auc(labels: Iterable[bool], scores: Iterable[float]) -> float | None:
    y = np.asarray(list(labels), dtype=np.bool_)
    value = np.asarray(list(scores), dtype=np.float64)
    if y.size == 0 or y.size != value.size:
        return None
    positive = int(y.sum())
    negative = int(y.size - positive)
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(value, method="average")
    rank_sum = float(ranks[y].sum())
    return float(
        (rank_sum - positive * (positive + 1) / 2.0)
        / float(positive * negative)
    )


def _spearman(scores: list[float], qualities: list[float]) -> float | None:
    if len(scores) < 2 or len(scores) != len(qualities):
        return None
    if np.ptp(np.asarray(scores, dtype=np.float64)) <= 1.0e-12:
        return None
    if np.ptp(np.asarray(qualities, dtype=np.float64)) <= 1.0e-12:
        return None
    value = float(spearmanr(scores, qualities).statistic)
    return value if math.isfinite(value) else None


def _summary(values: Iterable[float]) -> dict[str, int | float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _stage(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    stage = record.get("stages", {}).get(STAGE_NAME)
    required = (
        "pred_x_rows",
        "range_norm",
        "exist_logits",
        "official_iou",
        "official_candidate_valid",
    )
    if not isinstance(stage, dict) or not all(
        isinstance(stage.get(name), torch.Tensor) for name in required
    ):
        raise ValueError("V38 cache lacks required direct-primary tensors")
    return stage


def _valid_ids(stage: dict[str, torch.Tensor]) -> list[int]:
    return stage["official_candidate_valid"].bool().nonzero(as_tuple=False).flatten().tolist()


def _deployed_ids(
    stage: dict[str, torch.Tensor], *, score_threshold: float
) -> list[int]:
    valid = stage["official_candidate_valid"].bool()
    scores = stage_scores(stage, quality_power=0.0, score_mode="exist")
    return (
        valid & (scores >= float(score_threshold))
    ).nonzero(as_tuple=False).flatten().tolist()


def _best_curve(
    record: dict[str, Any], gt_index: int, candidate_ids: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    stage = _stage(record)
    ids = [
        int(index)
        for index in candidate_ids
        if 0 <= int(index) < int(stage["pred_x_rows"].shape[0])
        and bool(stage["official_candidate_valid"][int(index)])
    ]
    if not ids:
        return None
    quality = stage["official_iou"][int(gt_index), ids].float()
    candidate = int(ids[int(quality.argmax())])
    curves, masks, _valid = candidate_row_masks(
        stage,
        input_h=int(record["meta"].get("input_h", 640)),
        input_w=int(record["meta"].get("input_w", 1600)),
        min_valid_rows=5,
    )
    return curves[candidate], masks[candidate], float(
        stage["official_iou"][int(gt_index), candidate]
    )


def _new_curve_cohorts() -> dict[str, dict[str, Any]]:
    names = (
        "all_gt_best_of_four",
        "all_gt_best_deployed",
        "deployed_tp_050",
        "deployed_tp_075",
        "recoverable_missed_050",
        "recoverable_missed_075",
        "unsupported_050",
        "unsupported_075",
    )
    return {name: _new_curve_stats() for name in names}


def _quality_bins(
    scores: list[float], qualities: list[float]
) -> dict[str, dict[str, int | float]]:
    bins = {
        "background_or_below_050": [],
        "valid_050_below_075": [],
        "valid_075": [],
    }
    for score, quality in zip(scores, qualities):
        if quality >= 0.75:
            bins["valid_075"].append(score)
        elif quality >= 0.50:
            bins["valid_050_below_075"].append(score)
        else:
            bins["background_or_below_050"].append(score)
    return {name: _summary(values) for name, values in bins.items()}


def classify_threshold_bottleneck(
    *, deployed_tp: int, support_tp: int, v7_tp: int
) -> dict[str, Any]:
    total_deficit = max(int(v7_tp) - int(deployed_tp), 0)
    geometry_deficit = max(int(v7_tp) - int(support_tp), 0)
    conversion_loss = max(int(support_tp) - int(deployed_tp), 0)
    if support_tp >= v7_tp:
        decision = "EXISTENCE_OR_SET_CONVERSION_LIMITED"
    elif total_deficit and geometry_deficit / total_deficit >= 0.60:
        decision = "DIRECT_GEOMETRY_CAPACITY_LIMITED"
    else:
        decision = "MIXED_GEOMETRY_AND_CONVERSION_LIMITED"
    return {
        "decision": decision,
        "v7_minus_deployed_tp": int(v7_tp - deployed_tp),
        "v7_minus_support_tp": int(v7_tp - support_tp),
        "support_minus_deployed_tp": int(support_tp - deployed_tp),
        "positive_geometry_share_of_v7_tp_deficit": _safe_ratio(
            geometry_deficit, total_deficit
        ),
        "support_to_deployed_conversion": _safe_ratio(deployed_tp, support_tp),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    policies = payload["policies"]
    lines = [
        "# V38 Direct-Primary 50K Training-Free Autopsy",
        "",
        f"- Decision: **{payload['verdict']['decision']}**",
        f"- Images: `{payload['population']['images']}`",
        f"- GT lanes: `{payload['population']['gt_lanes']}`",
        "- Test split used: `False`",
        "- Training performed: `False`",
        "",
        "## Set-level official-raster decomposition",
        "",
        "| Policy | F1@.50 | TP@.50 | Pred@.50 | F1@.75 | TP@.75 | Pred@.75 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("deployed", "forced_all_valid", "cardinality_oracle"):
        item = policies[name]
        lines.append(
            f"| {name} | {100.0 * item['0.5']['F1']:.3f} | "
            f"{item['0.5']['TP']} | {item['prediction_count_by_threshold']['0.5']} | "
            f"{100.0 * item['0.75']['F1']:.3f} | {item['0.75']['TP']} | "
            f"{item['prediction_count_by_threshold']['0.75']} |"
        )
    lines.extend(("", "## Bottleneck decision", ""))
    for threshold in ("0.5", "0.75"):
        item = payload["bottleneck_by_threshold"][threshold]
        lines.append(
            f"- `{threshold}`: **{item['decision']}** — V7−support TP "
            f"`{item['v7_minus_support_tp']}`, support−deployed TP "
            f"`{item['support_minus_deployed_tp']}`."
        )
    lines.extend(
        (
            "",
            "## Existence–quality observability",
            "",
            f"- Spearman: `{payload['existence_quality']['spearman']}`",
            f"- AUC@.50: `{payload['existence_quality']['auc_050']}`",
            f"- AUC@.75: `{payload['existence_quality']['auc_075']}`",
            "",
            "## Interpretation",
            "",
            payload["verdict"]["interpretation"],
            "",
            (
                "This is a diagnostic-only validation audit. No checkpoint, "
                "threshold, or test-set selection was performed."
            ),
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    v38_report_path = Path(args.v38_report).expanduser().resolve()
    v38_report = json.loads(v38_report_path.read_text(encoding="utf-8"))
    if int(v38_report.get("checkpoint_iteration", -1)) != 50_000:
        raise ValueError("V38 autopsy requires the fixed 50K endpoint")
    if Path(str(v38_report.get("checkpoint", ""))).name != checkpoint.name:
        raise ValueError("V38 report/checkpoint mismatch")
    contract = v38_report.get("contract", {})
    if contract.get("test_set_used") is not False:
        raise ValueError("V38 autopsy refuses a report that used the test split")
    if contract.get("threshold_selection_performed") is not False:
        raise ValueError("V38 autopsy refuses threshold-selected input")

    cache = load_or_collect_cache(
        args.config,
        checkpoint,
        split="val",
        dataset_root=args.dataset_root,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=bool(args.reuse_cache),
        max_batches=0,
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        sample_strategy="sequential",
        amp_dtype="none",
        channels_last=bool(args.channels_last),
        desc="V38 full-val autopsy cache",
    )
    cache = ensure_official_iou_cache(
        cache,
        line_width=float(args.line_width),
        min_valid_rows=int(args.min_valid_rows),
        row_visibility_thresh=0.0,
        workers=int(args.official_workers),
    )
    records = cache.get("records", [])
    expected_images = int(contract.get("official_val_rows", -1))
    if len(records) != expected_images:
        raise ValueError(
            f"V38 autopsy population mismatch: {len(records)} != {expected_images}"
        )

    policy_counts: dict[str, dict[str, dict[str, int]]] = {
        name: {
            str(threshold): {"TP": 0, "FP": 0, "FN": 0}
            for threshold in THRESHOLDS
        }
        for name in ("deployed", "forced_all_valid", "cardinality_oracle")
    }
    prediction_counts = defaultdict(int)
    transitions = {
        str(threshold): {
            "support_gt": 0,
            "deployed_gt": 0,
            "recoverable_missed_gt": 0,
            "unsupported_gt": 0,
        }
        for threshold in THRESHOLDS
    }
    score_values: list[float] = []
    quality_values: list[float] = []
    inactive_good = {str(threshold): 0 for threshold in THRESHOLDS}
    active_bad = {str(threshold): 0 for threshold in THRESHOLDS}
    curve_cohorts = _new_curve_cohorts()
    deployed_geometry: dict[str, float] = {}
    forced_geometry: dict[str, float] = {}
    gt_total = 0
    official_gt_with_fewer_than_five_rows = 0

    for record in records:
        stage = _stage(record)
        official_iou = stage["official_iou"].float()
        valid_ids = _valid_ids(stage)
        deployed_ids = _deployed_ids(
            stage, score_threshold=float(args.score_threshold)
        )
        gt_count = int(official_iou.shape[0])
        gt_total += gt_count
        prediction_counts["deployed"] += len(deployed_ids)
        prediction_counts["forced_all_valid"] += len(valid_ids)

        scores = stage_scores(stage, quality_power=0.0, score_mode="exist")
        quality = (
            official_iou.amax(dim=0)
            if gt_count
            else torch.zeros_like(scores, dtype=torch.float32)
        )
        for candidate in valid_ids:
            score = float(scores[candidate])
            candidate_quality = float(quality[candidate])
            score_values.append(score)
            quality_values.append(candidate_quality)
            for threshold in THRESHOLDS:
                key = str(threshold)
                if candidate_quality > threshold and score < float(args.score_threshold):
                    inactive_good[key] += 1
                if candidate_quality <= threshold and score >= float(args.score_threshold):
                    active_bad[key] += 1

        x_rows = stage["pred_x_rows"].float().unsqueeze(0)
        ranges = stage["range_norm"].float().unsqueeze(0)
        active_deployed = torch.zeros((1, x_rows.shape[1]), dtype=torch.bool)
        active_forced = torch.zeros_like(active_deployed)
        active_deployed[0, deployed_ids] = True
        active_forced[0, valid_ids] = True
        _merge_geometry(
            deployed_geometry,
            _geometry_counts(x_rows, ranges, active_deployed),
        )
        _merge_geometry(
            forced_geometry,
            _geometry_counts(x_rows, ranges, active_forced),
        )

        target_x, target_mask = _official_target_rows(record)
        if int(target_x.shape[0]) != gt_count:
            raise ValueError("official GT row reconstruction count mismatch")
        official_gt_with_fewer_than_five_rows += int(
            (target_mask.sum(dim=-1) < int(args.min_valid_rows)).sum()
        )
        for gt_index in range(gt_count):
            _add_curve(
                curve_cohorts["all_gt_best_of_four"],
                _best_curve(record, gt_index, valid_ids),
                target_x[gt_index],
                target_mask[gt_index],
            )
            _add_curve(
                curve_cohorts["all_gt_best_deployed"],
                _best_curve(record, gt_index, deployed_ids),
                target_x[gt_index],
                target_mask[gt_index],
            )

        for threshold in THRESHOLDS:
            key = str(threshold)
            deployed = evaluator_hungarian_assignment(
                official_iou, deployed_ids, threshold=float(threshold)
            )
            forced = evaluator_hungarian_assignment(
                official_iou, valid_ids, threshold=float(threshold)
            )
            support = cardinality_oracle_assignment(
                official_iou,
                threshold=float(threshold),
                top_k=4,
                candidate_valid=stage["official_candidate_valid"].bool(),
            )
            deployed_gt = {int(gt) for gt, _candidate in deployed.pairs}
            support_gt = {int(gt) for gt, _candidate in support.pairs}
            recoverable = support_gt - deployed_gt
            unsupported = set(range(gt_count)) - support_gt
            transitions[key]["support_gt"] += len(support_gt)
            transitions[key]["deployed_gt"] += len(deployed_gt)
            transitions[key]["recoverable_missed_gt"] += len(recoverable)
            transitions[key]["unsupported_gt"] += len(unsupported)

            for name, assignment, predictions in (
                ("deployed", deployed, len(deployed_ids)),
                ("forced_all_valid", forced, len(valid_ids)),
                ("cardinality_oracle", support, support.hit_count),
            ):
                row = policy_counts[name][key]
                row["TP"] += int(assignment.hit_count)
                row["FP"] += max(0, int(predictions) - int(assignment.hit_count))
                row["FN"] += max(0, gt_count - int(assignment.hit_count))

            suffix = "050" if threshold == 0.50 else "075"
            for gt_index in sorted(deployed_gt):
                _add_curve(
                    curve_cohorts[f"deployed_tp_{suffix}"],
                    _best_curve(record, gt_index, deployed_ids),
                    target_x[gt_index],
                    target_mask[gt_index],
                )
            for gt_index in sorted(recoverable):
                _add_curve(
                    curve_cohorts[f"recoverable_missed_{suffix}"],
                    _best_curve(record, gt_index, valid_ids),
                    target_x[gt_index],
                    target_mask[gt_index],
                )
            for gt_index in sorted(unsupported):
                _add_curve(
                    curve_cohorts[f"unsupported_{suffix}"],
                    _best_curve(record, gt_index, valid_ids),
                    target_x[gt_index],
                    target_mask[gt_index],
                )

    policies: dict[str, Any] = {}
    for name, threshold_rows in policy_counts.items():
        policies[name] = {
            key: _policy_metrics(**{field.lower(): value for field, value in row.items()})
            for key, row in threshold_rows.items()
        }
        if name == "cardinality_oracle":
            policies[name]["prediction_count_by_threshold"] = {
                key: int(row["TP"])
                for key, row in threshold_rows.items()
            }
        else:
            policies[name]["prediction_count_by_threshold"] = {
                key: int(prediction_counts[name]) for key in threshold_rows
            }

    report_metrics = v38_report["metrics"]
    deployed_exact = all(
        all(
            int(policies["deployed"][str(threshold)][field])
            == int(report_metrics[str(threshold)][field])
            for field in ("TP", "FP", "FN")
        )
        for threshold in THRESHOLDS
    )
    if not deployed_exact:
        mismatch = {
            str(threshold): {
                "cache": policies["deployed"][str(threshold)],
                "official_report": report_metrics[str(threshold)],
            }
            for threshold in THRESHOLDS
        }
        raise RuntimeError(
            "V38 cached deployed counts do not match official report: "
            + json.dumps(mismatch, sort_keys=True)
        )

    v7 = v38_report["v7_reference"]["results"]
    bottleneck = {
        str(threshold): classify_threshold_bottleneck(
            deployed_tp=int(policies["deployed"][str(threshold)]["TP"]),
            support_tp=int(
                policies["cardinality_oracle"][str(threshold)]["TP"]
            ),
            v7_tp=int(v7[str(threshold)]["TP"]),
        )
        for threshold in THRESHOLDS
    }
    decisions = {item["decision"] for item in bottleneck.values()}
    if decisions == {"DIRECT_GEOMETRY_CAPACITY_LIMITED"}:
        decision = "DIRECT_GEOMETRY_CAPACITY_LIMITED"
        interpretation = (
            "Even an official-raster best-of-four oracle cannot recover most "
            "of the V7 TP deficit. The current direct-primary family is mainly "
            "missing precise lane geometry; quality rescoring alone cannot close it."
        )
    elif decisions == {"EXISTENCE_OR_SET_CONVERSION_LIMITED"}:
        decision = "EXISTENCE_OR_SET_CONVERSION_LIMITED"
        interpretation = (
            "The four direct curves contain enough official-raster support to "
            "match V7, but the fixed existence policy fails to emit it. A "
            "quality-aware confidence gate is justified before new geometry."
        )
    else:
        decision = "MIXED_DIRECT_GEOMETRY_AND_CONFIDENCE_FAILURE"
        interpretation = (
            "V38 loses performance in both direct geometry support and its "
            "conversion to emitted lanes. A single score loss or query-init "
            "change is not sufficient without threshold-specific evidence."
        )

    quality_labels_050 = [value > 0.50 for value in quality_values]
    quality_labels_075 = [value > 0.75 for value in quality_values]
    payload = {
        "experiment": "V38 direct-primary 50K training-free full-validation autopsy",
        "diagnostic_only": True,
        "training_performed": False,
        "test_set_used": False,
        "threshold_selection_performed": False,
        "checkpoint_selection_performed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_v38_report": str(v38_report_path),
        "population": {
            "images": len(records),
            "gt_lanes": int(gt_total),
            "official_gt_with_fewer_than_five_sampled_rows": int(
                official_gt_with_fewer_than_five_rows
            ),
            "cache_path": cache["metadata"].get("cache_path"),
        },
        "contract": {
            "deployed_counts_exact_official_report": deployed_exact,
            "score_threshold_fixed": float(args.score_threshold),
            "line_width": float(args.line_width),
            "full_validation": len(records) == expected_images,
            "channels_last": bool(args.channels_last),
        },
        "policies": policies,
        "v7_reference": v7,
        "bottleneck_by_threshold": bottleneck,
        "support_to_deployed_transitions": transitions,
        "existence_quality": {
            "candidates": len(score_values),
            "spearman": _spearman(score_values, quality_values),
            "auc_050": _binary_auc(quality_labels_050, score_values),
            "auc_075": _binary_auc(quality_labels_075, score_values),
            "score_by_quality_tier": _quality_bins(score_values, quality_values),
            "inactive_candidate_with_iou_above_threshold": inactive_good,
            "active_candidate_with_iou_not_above_threshold": active_bad,
        },
        "row_and_range": {
            name: _finish_curve_stats(stats)
            for name, stats in curve_cohorts.items()
        },
        "geometry_health": {
            "deployed": _finish_geometry(deployed_geometry),
            "forced_all_valid": _finish_geometry(forced_geometry),
        },
        "verdict": {
            "decision": decision,
            "interpretation": interpretation,
        },
    }
    json_path = output_dir / "v38_direct_primary_50k_autopsy.json"
    markdown_path = output_dir / "V38_DIRECT_PRIMARY_50K_AUTOPSY.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(payload, markdown_path)
    print(json.dumps(payload, indent=2))
    print(f"json_report: {json_path}")
    print(f"markdown_report: {markdown_path}")


if __name__ == "__main__":
    main()
