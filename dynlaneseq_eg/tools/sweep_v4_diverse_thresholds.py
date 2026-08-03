from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    diagnostic_iou_matrix,
    ensure_official_iou_cache,
    load_or_collect_cache,
    metadata_for_json,
    stage_scores,
    write_json,
)
from dynlaneseq_eg.tools.analyze_v4_selection_coverage import (
    _curve_distance_matrix,
    _finish_counter,
    _finish_oracle,
    _float_tag,
    _mmr_ids,
    _new_counter,
    _new_oracle_counter,
    _resolve_stage,
    _update_counter,
    _update_oracle,
)


@dataclass(frozen=True)
class SelectionSpec:
    family: str
    score_threshold: float
    distance_px: float | None = None
    sigma_px: float | None = None
    penalty: float | None = None

    @property
    def key(self) -> str:
        threshold = _float_tag(self.score_threshold)
        if self.family == "score_topk":
            return f"score_topk_thr{threshold}"
        if self.family == "hard_diversity":
            return (
                f"hard_d{_float_tag(float(self.distance_px))}_"
                f"thr{threshold}"
            )
        if self.family == "mmr":
            return (
                f"mmr_s{_float_tag(float(self.sigma_px))}_"
                f"p{_float_tag(float(self.penalty))}_thr{threshold}"
            )
        raise ValueError(f"unsupported selection family: {self.family}")

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "family": self.family,
            "score_threshold": float(self.score_threshold),
            "distance_px": (
                None if self.distance_px is None else float(self.distance_px)
            ),
            "sigma_px": None if self.sigma_px is None else float(self.sigma_px),
            "penalty": None if self.penalty is None else float(self.penalty),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep score threshold and geometry diversity over an existing V4 "
            "candidate cache without rerunning detector inference."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cache-dir",
        default="outputs/diagnostic_cache/unified_lane_set_v4_1_score_gate_50k",
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=8)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--stage", default="main")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.50, 0.75]
    )
    parser.add_argument(
        "--score-thresholds",
        type=float,
        nargs="+",
        default=[
            0.00,
            0.025,
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.175,
            0.20,
            0.225,
            0.25,
            0.275,
            0.30,
            0.35,
            0.40,
        ],
    )
    parser.add_argument(
        "--hard-diversity-distances",
        type=float,
        nargs="+",
        default=[10.0, 15.0, 20.0, 25.0, 30.0],
    )
    parser.add_argument(
        "--mmr-sigmas",
        type=float,
        nargs="+",
        default=[10.0, 15.0, 20.0, 30.0, 40.0],
    )
    parser.add_argument(
        "--mmr-penalties",
        type=float,
        nargs="+",
        default=[0.20, 0.35, 0.50, 0.65, 0.80],
    )
    parser.add_argument("--near-min-iou", type=float, default=0.30)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _unique_sorted(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(sorted({float(value) for value in values}))


def _eligible_mask(
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    score_threshold: float,
) -> torch.Tensor:
    return candidate_valid.bool() & scores.float().ge(float(score_threshold))


def _score_topk_ids(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    top_k: int,
) -> list[int]:
    ids = torch.nonzero(eligible.bool(), as_tuple=False).flatten().tolist()
    ids.sort(key=lambda index: float(scores[index]), reverse=True)
    return [int(index) for index in ids[: int(top_k)]]


def _hard_diverse_ids(
    scores: torch.Tensor,
    distance: torch.Tensor,
    eligible: torch.Tensor,
    *,
    distance_threshold: float,
    top_k: int,
) -> list[int]:
    """Greedy score-ordered diversity using the already computed distance."""

    ordered = _score_topk_ids(
        scores,
        eligible,
        top_k=int(eligible.bool().sum()),
    )
    selected: list[int] = []
    for index in ordered:
        if all(
            float(distance[index, previous]) >= float(distance_threshold)
            for previous in selected
        ):
            selected.append(int(index))
            if len(selected) >= int(top_k):
                break
    return selected


def _selection_specs(
    score_thresholds: Iterable[float],
    hard_distances: Iterable[float],
    mmr_sigmas: Iterable[float],
    mmr_penalties: Iterable[float],
) -> tuple[SelectionSpec, ...]:
    specs: list[SelectionSpec] = []
    for threshold in score_thresholds:
        specs.append(SelectionSpec("score_topk", threshold))
        specs.extend(
            SelectionSpec(
                "hard_diversity",
                threshold,
                distance_px=distance,
            )
            for distance in hard_distances
        )
        specs.extend(
            SelectionSpec(
                "mmr",
                threshold,
                sigma_px=sigma,
                penalty=penalty,
            )
            for sigma in mmr_sigmas
            for penalty in mmr_penalties
        )
    return tuple(specs)


def _select_ids(
    spec: SelectionSpec,
    scores: torch.Tensor,
    distance: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    top_k: int,
) -> list[int]:
    eligible = _eligible_mask(
        scores,
        candidate_valid,
        spec.score_threshold,
    )
    if spec.family == "score_topk":
        return _score_topk_ids(scores, eligible, top_k=top_k)
    if spec.family == "hard_diversity":
        return _hard_diverse_ids(
            scores,
            distance,
            eligible,
            distance_threshold=float(spec.distance_px),
            top_k=top_k,
        )
    if spec.family == "mmr":
        return _mmr_ids(
            scores,
            distance,
            eligible,
            penalty=float(spec.penalty),
            sigma=float(spec.sigma_px),
            top_k=top_k,
        )
    raise ValueError(f"unsupported selection family: {spec.family}")


def _row_payload(
    spec: SelectionSpec,
    counters: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        f"{threshold:.2f}": _finish_counter(counter)
        for threshold, counter in counters.items()
    }
    f1_values = [float(row["f1"]) for row in metrics.values()]
    payload = spec.metadata()
    payload["metrics"] = metrics
    payload["objectives"] = {
        "f1_050": float(metrics.get("0.50", next(iter(metrics.values())))["f1"]),
        "mean_f1": sum(f1_values) / max(len(f1_values), 1),
    }
    return payload


def _best_row(
    rows: Iterable[dict[str, Any]],
    *,
    objective: str,
) -> dict[str, Any]:
    candidates = list(rows)
    if not candidates:
        raise ValueError("cannot rank an empty grid")
    return max(
        candidates,
        key=lambda row: (
            float(row["objectives"][objective]),
            float(row["metrics"].get("0.75", {}).get("f1", -1.0)),
            -float(row["score_threshold"]),
        ),
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if int(args.top_k) < 1:
        raise ValueError("top-k must be positive")
    iou_thresholds = _unique_sorted(args.iou_thresholds)
    score_thresholds = _unique_sorted(args.score_thresholds)
    hard_distances = _unique_sorted(args.hard_diversity_distances)
    mmr_sigmas = _unique_sorted(args.mmr_sigmas)
    mmr_penalties = _unique_sorted(args.mmr_penalties)
    if not iou_thresholds:
        raise ValueError("at least one IoU threshold is required")
    if not score_thresholds:
        raise ValueError("at least one score threshold is required")
    if any(value < 0.0 or value >= 1.0 for value in score_thresholds):
        raise ValueError("score thresholds must be in [0, 1)")
    if any(value <= 0.0 for value in hard_distances):
        raise ValueError("hard diversity distances must be positive")
    if any(value <= 0.0 for value in mmr_sigmas):
        raise ValueError("MMR sigmas must be positive")
    if any(value < 0.0 for value in mmr_penalties):
        raise ValueError("MMR penalties must be non-negative")
    if not 0.0 <= float(args.near_min_iou) < min(iou_thresholds):
        raise ValueError("near-min-iou must be below every IoU threshold")

    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        dataset_root=args.dataset_root or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=bool(args.reuse_cache or args.cache_only),
        require_cache=bool(args.cache_only),
        max_batches=args.max_batches,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        sample_strategy=args.sample_strategy,
        desc="V4 diverse-threshold cache",
    )
    cache = ensure_official_iou_cache(
        cache,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        workers=args.metric_workers,
    )
    metadata = cache["metadata"]
    input_h = int(metadata["input_h"])
    input_w = int(metadata["input_w"])
    score_mode = str(
        metadata.get("postprocess", {}).get("score_mode", "exist")
    )
    specs = _selection_specs(
        score_thresholds,
        hard_distances,
        mmr_sigmas,
        mmr_penalties,
    )
    counters = {
        spec: {threshold: _new_counter() for threshold in iou_thresholds}
        for spec in specs
    }
    all_candidate_oracle = {
        threshold: _new_oracle_counter() for threshold in iou_thresholds
    }
    threshold_eligible_oracle = {
        score_threshold: {
            threshold: _new_oracle_counter() for threshold in iou_thresholds
        }
        for score_threshold in score_thresholds
    }

    for record in tqdm(
        cache["records"],
        ncols=100,
        desc="cached threshold/diversity grid",
    ):
        stage_name = _resolve_stage(record, args.stage)
        stage = record["stages"][stage_name]
        iou, _valid_gt, candidate_valid = diagnostic_iou_matrix(
            record,
            stage_name,
            use_official=True,
            input_h=input_h,
            input_w=input_w,
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        scores = stage_scores(
            stage,
            quality_power=0.0,
            score_mode=score_mode,
        ).cpu()
        distance = _curve_distance_matrix(
            stage,
            input_h=input_h,
            input_w=input_w,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
            min_overlap_points=args.nms_min_overlap_points,
        )
        valid_ids = torch.nonzero(
            candidate_valid.bool(), as_tuple=False
        ).flatten().tolist()
        for threshold in iou_thresholds:
            _update_oracle(
                all_candidate_oracle[threshold],
                iou,
                valid_ids,
                threshold=threshold,
                top_k=args.top_k,
            )
        for score_threshold in score_thresholds:
            eligible = _eligible_mask(
                scores,
                candidate_valid,
                score_threshold,
            )
            eligible_ids = torch.nonzero(
                eligible, as_tuple=False
            ).flatten().tolist()
            for threshold in iou_thresholds:
                _update_oracle(
                    threshold_eligible_oracle[score_threshold][threshold],
                    iou,
                    eligible_ids,
                    threshold=threshold,
                    top_k=args.top_k,
                )
        for spec in specs:
            selected = _select_ids(
                spec,
                scores,
                distance,
                candidate_valid,
                top_k=args.top_k,
            )
            for threshold in iou_thresholds:
                _update_counter(
                    counters[spec][threshold],
                    iou,
                    selected,
                    distance,
                    threshold=threshold,
                    near_min_iou=args.near_min_iou,
                )

    rows = [_row_payload(spec, counters[spec]) for spec in specs]
    family_rows = {
        family: [row for row in rows if row["family"] == family]
        for family in ("score_topk", "hard_diversity", "mmr")
    }
    best_by_family = {
        family: {
            "f1_050": _best_row(values, objective="f1_050"),
            "mean_f1": _best_row(values, objective="mean_f1"),
        }
        for family, values in family_rows.items()
    }
    diversity_rows = family_rows["hard_diversity"] + family_rows["mmr"]
    best_overall = {
        "f1_050": _best_row(diversity_rows, objective="f1_050"),
        "mean_f1": _best_row(diversity_rows, objective="mean_f1"),
    }
    zero_raw = next(
        row
        for row in family_rows["score_topk"]
        if abs(float(row["score_threshold"])) < 1e-12
    )
    best_primary = best_overall["f1_050"]
    diagnosis = {
        "zero_threshold_score_topk_f1_050": float(
            zero_raw["objectives"]["f1_050"]
        ),
        "best_diverse_threshold_f1_050": float(
            best_primary["objectives"]["f1_050"]
        ),
        "absolute_f1_050_gain": float(
            best_primary["objectives"]["f1_050"]
            - zero_raw["objectives"]["f1_050"]
        ),
        "best_key": str(best_primary["key"]),
        "interpretation": (
            "thresholded_diversity_clears_0p80_diagnostic_gate"
            if float(best_primary["objectives"]["f1_050"]) >= 0.80
            else "thresholded_diversity_remains_below_0p80_diagnostic_gate"
        ),
    }
    payload = {
        "diagnostic_only": True,
        "warning": (
            "This grid is selected on one uniform validation subset. It is a "
            "design/teacher diagnosis, not a frozen deployment setting or an "
            "official benchmark result. Validate one frozen setting on full "
            "validation before test evaluation."
        ),
        "metadata": metadata_for_json(
            cache,
            tool="sweep_v4_diverse_thresholds",
            iou_space="official_raster",
            top_k=args.top_k,
            iou_thresholds=list(iou_thresholds),
            score_thresholds=list(score_thresholds),
            hard_diversity_distances=list(hard_distances),
            mmr_sigmas=list(mmr_sigmas),
            mmr_penalties=list(mmr_penalties),
            nms_min_overlap_points=args.nms_min_overlap_points,
            score_mode=score_mode,
        ),
        "all_candidate_oracle": {
            f"{threshold:.2f}": _finish_oracle(counter)
            for threshold, counter in all_candidate_oracle.items()
        },
        "threshold_eligible_oracle": {
            f"{score_threshold:g}": {
                f"{threshold:.2f}": _finish_oracle(counter)
                for threshold, counter in by_iou.items()
            }
            for score_threshold, by_iou in threshold_eligible_oracle.items()
        },
        "best_by_family": best_by_family,
        "best_overall_diversity": best_overall,
        "diagnosis": diagnosis,
        "rows": rows,
    }
    write_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "score_mode": score_mode,
                "all_candidate_oracle": payload["all_candidate_oracle"],
                "best_by_family": best_by_family,
                "best_overall_diversity": best_overall,
                "diagnosis": diagnosis,
            },
            indent=2,
        )
    )
    print(f"output_json: {Path(args.output_json)}")


if __name__ == "__main__":
    main()
