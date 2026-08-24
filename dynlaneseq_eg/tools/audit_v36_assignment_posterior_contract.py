from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_matcher
from dynlaneseq_eg.losses.matcher_s0 import HungarianMatcherS0
from dynlaneseq_eg.losses.range_aware_iou import (
    batched_pairwise_range_aware_row_strip_iou,
)
from dynlaneseq_eg.modeling.common import fixed_y_rows
from dynlaneseq_eg.tools.analyze_matcher_cost_counterfactual import (
    assignment_from_cost,
    matcher_cost_components,
    weighted_matcher_cost,
)
from dynlaneseq_eg.tools.analyze_selection_assignment_gradient_conflict import (
    _target_to_official_mapping,
)


AUDIT_VERSION = 1
VARIANTS = (
    "configured",
    "no_object",
    "point_only",
    "target_row_iou",
    "official_raster_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay V7 proposal assignment, route-posterior targets and slot "
            "geometry assignment on V34 oracle-good/current-wrong pairs."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--pairs-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-iteration", type=int, default=225000)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--cluster-delta", type=float, default=0.10)
    parser.add_argument("--cluster-temperature", type=float, default=0.03)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"expected dictionary torch payload: {path}")
    return value


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


def target_with_range(
    target: dict[str, torch.Tensor], *, input_h: int
) -> dict[str, torch.Tensor]:
    x_rows = target["x_rows"].detach().float().cpu()
    valid = target["valid_mask"].detach().bool().cpu()
    y_rows = fixed_y_rows(
        int(x_rows.shape[-1]),
        int(input_h),
        dtype=torch.float32,
    ).cpu()
    ranges = torch.zeros((int(x_rows.shape[0]), 2), dtype=torch.float32)
    for lane in range(int(x_rows.shape[0])):
        indices = torch.nonzero(valid[lane], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        ranges[lane, 0] = y_rows[int(indices[0])]
        ranges[lane, 1] = y_rows[int(indices[-1])]
    return {
        "x_rows": x_rows,
        "valid_mask": valid,
        "range_y": ranges,
    }


def route_target_distribution(
    quality: torch.Tensor,
    candidate_valid: torch.Tensor,
    gt_index: int,
    *,
    cluster_delta: float,
    temperature: float,
) -> torch.Tensor:
    """Exact V7 all-GT near-best proposal distribution for one GT."""

    values = quality[:, int(gt_index)].float()
    valid = candidate_valid.bool() & torch.isfinite(values)
    output = torch.zeros_like(values)
    if not bool(valid.any()):
        return output
    best = values[valid].amax()
    cutoff = torch.maximum(
        best.new_zeros(()),
        best - float(cluster_delta),
    )
    support = valid & (values >= cutoff)
    output[support] = torch.softmax(
        values[support] / float(temperature),
        dim=0,
    )
    return output


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


def _official_cost_matrix(
    official_iou: torch.Tensor,
    target_to_official: dict[int, int],
    *,
    candidates: int,
    targets: int,
) -> torch.Tensor:
    cost = torch.full((int(candidates), int(targets)), 1.0e6)
    for target_gt, official_gt in target_to_official.items():
        if not 0 <= int(target_gt) < int(targets):
            continue
        if not 0 <= int(official_gt) < int(official_iou.shape[0]):
            continue
        cost[:, int(target_gt)] = 1.0 - official_iou[int(official_gt)].float()
    return cost


def _slot_geometry_assignment(
    stage: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    input_h: int,
    line_width: float,
    min_valid_rows: int,
) -> dict[int, int]:
    # Factorized V7 keeps a real proposal route for every slot even when the
    # deployed active decision emits dustbin.  The geometry loss consumes
    # that real route; ``selection_slot_indices`` may contain the synthetic
    # dustbin id for inactive slots, whereas raw indices are always real.
    routes_value = stage.get("selection_slot_raw_indices")
    if not isinstance(routes_value, torch.Tensor):
        routes_value = stage["selection_slot_indices"]
    routes = routes_value.long()
    candidate_count = int(stage["pred_x_rows"].shape[0])
    valid_slot_ids = torch.nonzero(
        (routes >= 0) & (routes < candidate_count),
        as_tuple=False,
    ).flatten()
    if valid_slot_ids.numel() == 0:
        return {}
    valid_routes = routes.index_select(0, valid_slot_ids)
    pred_x = stage["pred_x_rows"].float().index_select(0, valid_routes)
    pred_range = stage["range_norm"].float().index_select(0, valid_routes)
    quality, slot_valid, gt_valid = _candidate_quality(
        pred_x,
        pred_range,
        target,
        input_h=input_h,
        line_width=line_width,
        min_valid_rows=min_valid_rows,
    )
    pair_valid = slot_valid[:, None] & gt_valid[None, :]
    cost = (1.0 - quality).masked_fill(~pair_valid, torch.inf)
    finite_slots = torch.nonzero(cost.isfinite().any(dim=1), as_tuple=False).flatten()
    finite_gt = torch.nonzero(cost.isfinite().any(dim=0), as_tuple=False).flatten()
    if finite_slots.numel() == 0 or finite_gt.numel() == 0:
        return {}
    local = cost.index_select(0, finite_slots).index_select(1, finite_gt)
    local_slot, local_gt = HungarianMatcherS0._linear_sum_assignment(local)
    return {
        int(valid_slot_ids[int(finite_slots[int(slot)])]): int(finite_gt[int(gt)])
        for slot, gt in zip(local_slot.tolist(), local_gt.tolist())
    }


def _assignment_flags(
    assignment: dict[int, int],
    *,
    intended_gt: int,
    good: int,
    wrong: int,
) -> dict[str, float]:
    values = set(int(candidate) for candidate in assignment.values())
    selected = assignment.get(int(intended_gt), -1)
    return {
        "good_intended": float(selected == int(good)),
        "wrong_intended": float(selected == int(wrong)),
        "other_intended": float(selected not in {-1, int(good), int(wrong)}),
        "good_any_positive": float(int(good) in values),
        "wrong_any_positive": float(int(wrong) in values),
        "good_other_gt": float(int(good) in values and selected != int(good)),
        "wrong_other_gt": float(int(wrong) in values and selected != int(wrong)),
    }


def _pair_accuracy(margin: float, *, eps: float = 1.0e-12) -> float:
    if margin > eps:
        return 1.0
    if margin < -eps:
        return 0.0
    return 0.5


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


def _cohort_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    scalar_keys = (
        "route_target_good_preference",
        "route_target_good_top1",
        "learned_route_good_preference",
        "configured_cost_good_preference",
        "configured_good_intended",
        "configured_good_any_positive",
        "configured_wrong_intended",
        "configured_wrong_any_positive",
        "good_route_up_but_proposal_down",
        "wrong_route_down_but_proposal_up",
        "slot_geometry_intended_gt",
        "good_used_by_other_slot",
    )
    margin_keys = (
        "official_iou_margin",
        "target_quality_margin",
        "exist_probability_margin",
        "route_target_mass_margin",
        "learned_route_probability_margin",
        "route_descent_update_margin",
        "proposal_descent_update_margin",
        "configured_cost_margin",
        "object_cost_margin",
        "point_cost_margin",
        "range_cost_margin",
        "line_iou_cost_margin",
        "mapping_quality",
    )
    metrics: dict[str, Any] = {}
    for key in scalar_keys:
        metrics[key] = {
            "mean": _mean(row[key] for row in rows),
            "clip_bootstrap_95": _clip_bootstrap_ci(
                rows,
                key,
                repetitions=bootstrap_reps,
                seed=seed,
            ),
        }
    for key in margin_keys:
        metrics[key] = _quantiles(row[key] for row in rows)
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_metrics = {}
        for suffix in (
            "good_preference",
            "good_intended",
            "wrong_intended",
            "good_any_positive",
            "wrong_any_positive",
        ):
            key = f"{variant}_{suffix}"
            variant_metrics[suffix] = {
                "mean": _mean(row[key] for row in rows),
                "clip_bootstrap_95": _clip_bootstrap_ci(
                    rows,
                    key,
                    repetitions=bootstrap_reps,
                    seed=seed + 17,
                ),
            }
        variants[variant] = variant_metrics
    return {
        "pairs": len(rows),
        "clips": len({str(row["clip"]) for row in rows}),
        "metrics": metrics,
        "assignments": variants,
    }


def classify_verdict(overall: dict[str, Any]) -> dict[str, Any]:
    metrics = overall["metrics"]
    route_preference = float(metrics["route_target_good_preference"]["mean"])
    route_top1 = float(metrics["route_target_good_top1"]["mean"])
    matcher_preference = float(metrics["configured_cost_good_preference"]["mean"])
    good_positive = float(metrics["configured_good_any_positive"]["mean"])
    good_intended = float(metrics["configured_good_intended"]["mean"])
    configured = overall["assignments"]["configured"]["good_intended"]["mean"]
    counterfactuals = {
        name: overall["assignments"][name]["good_intended"]["mean"]
        for name in ("no_object", "point_only", "target_row_iou")
    }
    best_name = max(counterfactuals, key=lambda name: float(counterfactuals[name]))
    best_gain = float(counterfactuals[best_name]) - float(configured)

    if route_preference < 0.70 or route_top1 < 0.70:
        decision = "ROUTE_TARGET_MISALIGNED"
        interpretation = (
            "Mevcut four-slot target oracle-good proposal'ı yeterince açık "
            "tarif etmiyor; önce target contract düzeltilmeli."
        )
    elif good_intended < 0.65 and best_gain >= 0.10:
        decision = "HARD_ASSIGNMENT_STARVATION"
        interpretation = (
            "Route target good proposal'ı biliyor fakat proposal Hungarian onu "
            "sıkça pozitif yapmıyor; controlled soft/one-to-many ownership "
            "bir sonraki nedensel deneydir."
        )
    elif (
        route_preference >= 0.80
        and matcher_preference >= 0.75
        and good_positive >= 0.80
    ):
        decision = "POST_ASSIGNMENT_BELIEF_FAILURE"
        interpretation = (
            "Matcher ve route target good proposal'ı zaten biliyor. Yeni "
            "matcher veya daha fazla positive query ana çözüm değildir; doğru "
            "supervision deployed belief'e dönüşmüyor."
        )
    else:
        decision = "MIXED_CONTRACT_FAILURE"
        interpretation = (
            "Target, hard assignment ve deployed posterior arasında tek bir "
            "baskın hata yok; en büyük ölçülmüş sözleşme kaybı izole edilmeden "
            "uzun training açılmamalı."
        )
    return {
        "decision": decision,
        "interpretation": interpretation,
        "route_target_good_preference": route_preference,
        "route_target_good_top1": route_top1,
        "configured_cost_good_preference": matcher_preference,
        "configured_good_any_positive": good_positive,
        "configured_good_intended": good_intended,
        "best_counterfactual": best_name,
        "best_counterfactual_good_intended_gain": best_gain,
        "test_split_used": False,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    def show(value: Any) -> str:
        return "—" if value is None else f"{float(value):.4f}"

    lines = [
        "# V36 Assignment–Posterior Contract Results",
        "",
        f"- Decision: **{payload['verdict']['decision']}**",
        f"- Evaluated pairs: `{payload['counts']['evaluated_pairs']}`",
        f"- Mapping skips: `{payload['counts']['mapping_skips']}`",
        "- Test split used: `False`",
        "",
        "| Cohort | Pairs | Route target good | Target top-1 good | Matcher cost good | Good intended + | Good any + | Slot GT correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, cohort in payload["summary"].items():
        metrics = cohort["metrics"]
        lines.append(
            f"| {name} | {cohort['pairs']} | "
            f"{show(metrics['route_target_good_preference']['mean'])} | "
            f"{show(metrics['route_target_good_top1']['mean'])} | "
            f"{show(metrics['configured_cost_good_preference']['mean'])} | "
            f"{show(metrics['configured_good_intended']['mean'])} | "
            f"{show(metrics['configured_good_any_positive']['mean'])} | "
            f"{show(metrics['slot_geometry_intended_gt']['mean'])} |"
        )
    lines.extend(
        (
            "",
            "## Assignment counterfactuals — overall",
            "",
            "| Matcher | Good preference | Good intended + | Wrong intended + | Good any + |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    overall = payload["summary"]["overall"]
    for name in VARIANTS:
        item = overall["assignments"][name]
        lines.append(
            f"| {name} | {show(item['good_preference']['mean'])} | "
            f"{show(item['good_intended']['mean'])} | "
            f"{show(item['wrong_intended']['mean'])} | "
            f"{show(item['good_any_positive']['mean'])} |"
        )
    lines.extend(
        (
            "",
            "## Yorum",
            "",
            payload["verdict"]["interpretation"],
            "",
            "Bu training-contract diagnostic'idir; official validation F1 veya deploy sonucu değildir.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cache_path = Path(args.target_cache).expanduser().resolve()
    pairs_path = Path(args.pairs_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    matcher = build_matcher(cfg)
    matcher.set_iteration(int(args.checkpoint_iteration))
    effective_cfg = replace(
        matcher.cfg,
        lambda_obj=float(matcher.effective_lambda_obj()),
    )
    no_object_cfg = replace(effective_cfg, lambda_obj=0.0)
    point_only_cfg = replace(
        effective_cfg,
        lambda_obj=0.0,
        lambda_range=0.0,
        lambda_line_iou=0.0,
    )

    cache = _load_torch(cache_path)
    pair_payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    raw_pairs = pair_payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("pairs JSON is missing the V34 pair list")
    metadata = cache.get("metadata", {})
    dataset_root = Path(str(metadata.get("dataset_root", "/workspace/CULane")))
    input_h = int(metadata.get("input_h", cfg.get("model", {}).get("input_h", 640)))
    input_w = int(metadata.get("input_w", cfg.get("model", {}).get("input_w", 1600)))
    records = {
        _canonical_image_id(
            str(record.get("meta", {}).get("image_path", record.get("image_id", ""))),
            dataset_root,
        ): record
        for record in cache.get("records", [])
    }
    pairs_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in raw_pairs:
        pairs_by_image[_canonical_image_id(str(pair["image_id"]), dataset_root)].append(pair)

    missing_images = sorted(set(pairs_by_image) - set(records))
    if missing_images:
        raise ValueError(f"target cache is missing {len(missing_images)} pair images")

    rows: list[dict[str, Any]] = []
    mapping_skips = 0
    mapping_quality_values: list[float] = []
    for image_id in tqdm(sorted(pairs_by_image), desc="V36 assignment replay", ncols=100):
        record = records[image_id]
        stage = record["stages"]["main"]
        target = target_with_range(record["target"], input_h=input_h)
        target_to_official, mapping_qualities = _target_to_official_mapping(
            target,
            record["meta"],
            line_width=float(args.line_width),
        )
        official_to_target = {
            int(official): int(target_gt)
            for target_gt, official in target_to_official.items()
        }
        mapping_quality = float(np.mean(mapping_qualities)) if mapping_qualities else float("nan")
        mapping_quality_values.extend(float(value) for value in mapping_qualities)

        pred_x = stage["pred_x_rows"].float()
        pred_range = stage["range_norm"].float()
        exist_logits = stage["exist_logits"].float()
        components = matcher_cost_components(
            matcher,
            exist_logits,
            pred_x,
            pred_range,
            target,
        )
        configured_cost = weighted_matcher_cost(components, effective_cfg)
        no_object_cost = weighted_matcher_cost(components, no_object_cfg)
        point_only_cost = weighted_matcher_cost(components, point_only_cfg)
        target_quality, candidate_valid, gt_valid = _candidate_quality(
            pred_x,
            pred_range,
            target,
            input_h=input_h,
            line_width=float(args.line_width),
            min_valid_rows=int(args.min_valid_rows),
        )
        target_iou_cost = (1.0 - target_quality).masked_fill(
            ~(candidate_valid[:, None] & gt_valid[None, :]),
            1.0e6,
        )
        official_iou = stage["official_iou"].float()
        official_cost = _official_cost_matrix(
            official_iou,
            target_to_official,
            candidates=int(pred_x.shape[0]),
            targets=int(target["x_rows"].shape[0]),
        )
        costs = {
            "configured": configured_cost,
            "no_object": no_object_cost,
            "point_only": point_only_cost,
            "target_row_iou": target_iou_cost,
            "official_raster_iou": official_cost,
        }
        assignments = {name: assignment_from_cost(cost) for name, cost in costs.items()}
        slot_assignment = _slot_geometry_assignment(
            stage,
            target,
            input_h=input_h,
            line_width=float(args.line_width),
            min_valid_rows=int(args.min_valid_rows),
        )
        exist_probability = torch.softmax(exist_logits, dim=-1)[:, 0]
        selected_routes = stage["selection_slot_indices"].long()

        for pair in pairs_by_image[image_id]:
            official_gt = int(pair["gt_id"])
            intended_gt = official_to_target.get(official_gt)
            if intended_gt is None or not bool(gt_valid[intended_gt]):
                mapping_skips += 1
                continue
            good = int(pair["good_candidate"])
            wrong = int(pair["wrong_candidate"])
            slot = int(pair["slot_id"])
            target_distribution = route_target_distribution(
                target_quality,
                candidate_valid,
                intended_gt,
                cluster_delta=float(args.cluster_delta),
                temperature=float(args.cluster_temperature),
            )
            real_logits = stage["selection_slot_logits"][slot, : int(pred_x.shape[0])].float()
            route_probability = torch.softmax(real_logits, dim=-1)
            target_good = float(target_distribution[good])
            target_wrong = float(target_distribution[wrong])
            route_good = float(route_probability[good])
            route_wrong = float(route_probability[wrong])
            configured_flags = _assignment_flags(
                assignments["configured"],
                intended_gt=intended_gt,
                good=good,
                wrong=wrong,
            )
            proposal_good_update = configured_flags["good_any_positive"] - float(exist_probability[good])
            proposal_wrong_update = configured_flags["wrong_any_positive"] - float(exist_probability[wrong])
            route_good_update = target_good - route_good
            route_wrong_update = target_wrong - route_wrong
            used_slots = torch.nonzero(selected_routes == good, as_tuple=False).flatten().tolist()

            row: dict[str, Any] = {
                "fold": str(pair["fold"]),
                "clip": str(pair["clip"]),
                "image_id": image_id,
                "official_gt": official_gt,
                "training_gt": int(intended_gt),
                "slot_id": slot,
                "good_candidate": good,
                "wrong_candidate": wrong,
                "thresholds": [float(value) for value in pair["thresholds"]],
                "good_iou": float(pair["good_iou"]),
                "wrong_iou": float(pair["wrong_iou"]),
                "official_iou_margin": float(pair["good_iou"]) - float(pair["wrong_iou"]),
                "mapping_quality": mapping_quality,
                "target_quality_good": float(target_quality[good, intended_gt]),
                "target_quality_wrong": float(target_quality[wrong, intended_gt]),
                "target_quality_margin": float(target_quality[good, intended_gt] - target_quality[wrong, intended_gt]),
                "route_target_mass_good": target_good,
                "route_target_mass_wrong": target_wrong,
                "route_target_mass_margin": target_good - target_wrong,
                "route_target_good_preference": _pair_accuracy(target_good - target_wrong),
                "route_target_good_top1": float(int(target_distribution.argmax()) == good),
                "learned_route_probability_good": route_good,
                "learned_route_probability_wrong": route_wrong,
                "learned_route_probability_margin": route_good - route_wrong,
                "learned_route_good_preference": _pair_accuracy(route_good - route_wrong),
                "exist_probability_good": float(exist_probability[good]),
                "exist_probability_wrong": float(exist_probability[wrong]),
                "exist_probability_margin": float(exist_probability[good] - exist_probability[wrong]),
                "proposal_descent_update_good": proposal_good_update,
                "proposal_descent_update_wrong": proposal_wrong_update,
                "proposal_descent_update_margin": proposal_good_update - proposal_wrong_update,
                "route_descent_update_good": route_good_update,
                "route_descent_update_wrong": route_wrong_update,
                "route_descent_update_margin": route_good_update - route_wrong_update,
                "good_route_up_but_proposal_down": float(route_good_update > 0.0 and proposal_good_update < 0.0),
                "wrong_route_down_but_proposal_up": float(route_wrong_update < 0.0 and proposal_wrong_update > 0.0),
                "slot_geometry_intended_gt": float(slot_assignment.get(slot, -1) == intended_gt),
                "slot_geometry_training_gt": int(slot_assignment.get(slot, -1)),
                "good_used_by_other_slot": float(any(int(value) != slot for value in used_slots)),
            }
            component_names = {
                "object": "object_cost_margin",
                "point": "point_cost_margin",
                "range": "range_cost_margin",
                "line_iou": "line_iou_cost_margin",
            }
            for component, output_name in component_names.items():
                matrix = components[component]
                margin = float(matrix[wrong, intended_gt] - matrix[good, intended_gt])
                row[output_name] = margin
            row["configured_cost_margin"] = float(
                configured_cost[wrong, intended_gt] - configured_cost[good, intended_gt]
            )
            row["configured_cost_good_preference"] = _pair_accuracy(row["configured_cost_margin"])
            for name in VARIANTS:
                flags = _assignment_flags(
                    assignments[name],
                    intended_gt=intended_gt,
                    good=good,
                    wrong=wrong,
                )
                cost_margin = float(costs[name][wrong, intended_gt] - costs[name][good, intended_gt])
                row[f"{name}_good_preference"] = _pair_accuracy(cost_margin)
                for key, value in flags.items():
                    row[f"{name}_{key}"] = value
            row["configured_good_intended"] = row["configured_good_intended"]
            row["configured_good_any_positive"] = row["configured_good_any_positive"]
            row["configured_wrong_intended"] = row["configured_wrong_intended"]
            row["configured_wrong_any_positive"] = row["configured_wrong_any_positive"]
            rows.append(row)

    if not rows:
        raise RuntimeError("V36 did not evaluate any V34 pairs")

    cohorts: dict[str, list[dict[str, Any]]] = {
        "overall": rows,
        "fold_a": [row for row in rows if row["fold"] == "a"],
        "fold_b": [row for row in rows if row["fold"] == "b"],
        "iou_050": [row for row in rows if 0.5 in row["thresholds"]],
        "iou_075": [row for row in rows if 0.75 in row["thresholds"]],
        "fold_a_iou_050": [row for row in rows if row["fold"] == "a" and 0.5 in row["thresholds"]],
        "fold_a_iou_075": [row for row in rows if row["fold"] == "a" and 0.75 in row["thresholds"]],
        "fold_b_iou_050": [row for row in rows if row["fold"] == "b" and 0.5 in row["thresholds"]],
        "fold_b_iou_075": [row for row in rows if row["fold"] == "b" and 0.75 in row["thresholds"]],
    }
    summaries = {
        name: _cohort_summary(
            cohort,
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + index * 101,
        )
        for index, (name, cohort) in enumerate(cohorts.items())
        if cohort
    }
    verdict = classify_verdict(summaries["overall"])
    payload = {
        "audit_version": AUDIT_VERSION,
        "experiment": "V36 Assignment-Posterior Contract Audit",
        "config": str(config_path),
        "target_cache": str(cache_path),
        "pairs_json": str(pairs_path),
        "settings": {
            "checkpoint_iteration": int(args.checkpoint_iteration),
            "input_h": input_h,
            "input_w": input_w,
            "line_width": float(args.line_width),
            "min_valid_rows": int(args.min_valid_rows),
            "cluster_delta": float(args.cluster_delta),
            "cluster_temperature": float(args.cluster_temperature),
            "matcher_effective_lambda_obj": float(effective_cfg.lambda_obj),
            "bootstrap_reps": int(args.bootstrap_reps),
            "seed": int(args.seed),
        },
        "counts": {
            "input_pairs": len(raw_pairs),
            "evaluated_pairs": len(rows),
            "mapping_skips": int(mapping_skips),
            "images": len({row["image_id"] for row in rows}),
            "clips": len({row["clip"] for row in rows}),
        },
        "target_to_official_mapping_quality": _quantiles(mapping_quality_values),
        "summary": summaries,
        "verdict": verdict,
        "pairs": rows,
    }
    json_path = output_dir / "v36_assignment_posterior_contract.json"
    markdown_path = output_dir / "v36_assignment_posterior_contract.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(payload, markdown_path)
    print(json.dumps({"verdict": verdict, "counts": payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
