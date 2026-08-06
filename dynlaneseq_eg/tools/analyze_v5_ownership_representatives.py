from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import stage_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether direct V5 ownership scores select the best "
            "official-IoU representative inside each recoverable GT cluster."
        )
    )
    parser.add_argument("--oracle-report", required=True)
    parser.add_argument("--stage", default="main")
    parser.add_argument("--representable-min", type=float, default=0.50)
    parser.add_argument("--cluster-min", type=float, default=0.30)
    parser.add_argument("--pair-margin", type=float, default=0.01)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_cache(report_path: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend((Path.cwd() / raw, report_path.parent / raw))
    elif "outputs" in raw.parts:
        output_index = raw.parts.index("outputs")
        candidates.append(Path.cwd().joinpath(*raw.parts[output_index:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "oracle cache is unavailable; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, float(quantile)).item())


def main() -> None:
    args = parse_args()
    report_path = Path(args.oracle_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    cache_path = _resolve_cache(report_path, str(metadata.get("cache_path", "")))
    cache = _torch_load(cache_path)

    support_sizes: list[float] = []
    selected_ranks: list[float] = []
    regrets: list[float] = []
    selected_quality: list[float] = []
    best_quality: list[float] = []
    best_score_ranks: list[float] = []
    pair_correct = 0
    pair_total = 0
    skipped_joint_cluster = 0

    for record in cache.get("records", []):
        stage = record.get("stages", {}).get(args.stage)
        if not isinstance(stage, dict):
            continue
        official_iou = stage.get("official_iou")
        candidate_valid = stage.get("official_candidate_valid")
        if not isinstance(official_iou, torch.Tensor) or not isinstance(
            candidate_valid,
            torch.Tensor,
        ):
            raise ValueError(
                "oracle cache has no official IoU; run analyze_oracle_topk "
                "with --exact-postprocess first"
            )
        official_iou = official_iou.float()
        candidate_valid = candidate_valid.bool()
        if int(official_iou.shape[0]) == 0 or not bool(candidate_valid.any()):
            continue
        scores = stage_scores(stage, quality_power=0.0, score_mode="exist")
        scores = scores.float()
        valid_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten()
        valid_iou = official_iou[:, valid_ids]
        candidate_best_gt = valid_iou.argmax(dim=0)

        for gt_index in range(int(official_iou.shape[0])):
            quality_all = official_iou[gt_index, valid_ids]
            if float(quality_all.max()) < float(args.representable_min):
                continue
            cluster_local = torch.nonzero(
                (candidate_best_gt == gt_index)
                & (quality_all >= float(args.cluster_min)),
                as_tuple=False,
            ).flatten()
            if int(cluster_local.numel()) == 0:
                skipped_joint_cluster += 1
                continue
            cluster_ids = valid_ids[cluster_local]
            quality = official_iou[gt_index, cluster_ids]
            cluster_scores = scores[cluster_ids]
            selected_local = int(cluster_scores.argmax().item())
            selected_q = float(quality[selected_local])
            best_q = float(quality.max())
            quality_rank = 1 + int((quality > selected_q + 1e-8).sum().item())
            best_local = int(quality.argmax().item())
            score_rank_of_best = 1 + int(
                (cluster_scores > cluster_scores[best_local] + 1e-8).sum().item()
            )

            support_sizes.append(float(cluster_ids.numel()))
            selected_ranks.append(float(quality_rank))
            best_score_ranks.append(float(score_rank_of_best))
            regrets.append(max(0.0, best_q - selected_q))
            selected_quality.append(selected_q)
            best_quality.append(best_q)

            for left in range(int(cluster_ids.numel())):
                for right in range(left + 1, int(cluster_ids.numel())):
                    quality_delta = float(quality[left] - quality[right])
                    if abs(quality_delta) <= float(args.pair_margin):
                        continue
                    score_delta = float(cluster_scores[left] - cluster_scores[right])
                    pair_correct += int(quality_delta * score_delta > 0.0)
                    pair_total += 1

    count = len(selected_ranks)
    result = {
        "diagnostic_only": True,
        "experiment": "V5 direct ownership representative observability",
        "oracle_report": str(report_path),
        "cache_path": str(cache_path),
        "checkpoint": metadata.get("checkpoint"),
        "images": int(metadata.get("num_records", 0)),
        "settings": {
            "stage": args.stage,
            "representable_min": float(args.representable_min),
            "cluster_min": float(args.cluster_min),
            "pair_margin": float(args.pair_margin),
            "score": "softmax(exist_logits)[lane]",
            "cluster": "candidate argmax-GT ownership plus official-IoU floor",
        },
        "representable_clusters": count,
        "skipped_joint_cluster": int(skipped_joint_cluster),
        "support_size": {
            "mean": sum(support_sizes) / float(max(count, 1)),
            "p50": _percentile(support_sizes, 0.50),
            "p90": _percentile(support_sizes, 0.90),
        },
        "representative": {
            "top1_rate": sum(rank <= 1.0 for rank in selected_ranks)
            / float(max(count, 1)),
            "top2_rate": sum(rank <= 2.0 for rank in selected_ranks)
            / float(max(count, 1)),
            "mean_selected_quality_rank": sum(selected_ranks)
            / float(max(count, 1)),
            "mean_best_candidate_score_rank": sum(best_score_ranks)
            / float(max(count, 1)),
            "mean_official_iou_regret": sum(regrets) / float(max(count, 1)),
            "p90_official_iou_regret": _percentile(regrets, 0.90),
            "mean_selected_official_iou": sum(selected_quality)
            / float(max(count, 1)),
            "mean_best_official_iou": sum(best_quality)
            / float(max(count, 1)),
            "pair_order_accuracy": pair_correct / float(max(pair_total, 1)),
            "ordered_pairs": int(pair_total),
        },
        "predeclared_gate": {
            "top1_at_least_0p40": (
                sum(rank <= 1.0 for rank in selected_ranks)
                / float(max(count, 1))
            )
            >= 0.40,
            "top2_at_least_0p70": (
                sum(rank <= 2.0 for rank in selected_ranks)
                / float(max(count, 1))
            )
            >= 0.70,
            "mean_regret_at_most_0p10": (
                sum(regrets) / float(max(count, 1))
            )
            <= 0.10,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
