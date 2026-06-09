from __future__ import annotations

import argparse
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import ProposalRecallStats, collect_prediction_stages, update_stage_recall
from dynlaneseq_eg.factory import build_dataloader, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure score-independent proposal recall for each DynLaneSeq stage. "
            "This does not write CULane predictions, run NMS, or apply score thresholds."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=0, help="0 means use every slot/proposal.")
    parser.add_argument(
        "--rank-by",
        default="none",
        choices=["none", "score", "quality", "score_quality"],
        help="Candidate ranking used only when --top-k is positive.",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--categories", action="store_true", help="Also analyze official CULane test_split categories.")
    parser.add_argument("--skip-overall", action="store_true", help="Skip the main split pass and run only requested extras.")
    return parser.parse_args()


def format_summary(name: str, summary: dict[str, float], thresholds: list[float]) -> str:
    parts = [f"{name:>18}", f"gt={int(summary['gt']):5d}"]
    for threshold in thresholds:
        parts.append(f"R@{threshold:g}={summary[f'recall@{threshold:g}']:.4f}")
    parts.extend(
        [
            f"meanIoU={summary['mean_best_iou']:.4f}",
            f"medIoU={summary['median_best_iou']:.4f}",
            f"p90IoU={summary['p90_best_iou']:.4f}",
        ]
    )
    return "  ".join(parts)


@torch.no_grad()
def collect_stats(
    model: torch.nn.Module,
    cfg: dict,
    split: str,
    device: torch.device,
    thresholds: tuple[float, ...],
    args: argparse.Namespace,
    desc: str,
) -> dict[str, ProposalRecallStats]:
    stats: dict[str, ProposalRecallStats] = defaultdict(lambda: ProposalRecallStats(thresholds=thresholds))
    loader = build_dataloader(cfg, split=split, training=False)
    pass_targets = bool(getattr(model, "oracle_coarse_enabled", False))
    for batch_idx, (images, targets, _metas) in enumerate(tqdm(loader, ncols=80, desc=desc)):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        outputs = model(images, targets=targets) if pass_targets else model(images)
        stages = collect_prediction_stages(outputs)
        for stage_name, stage_outputs in stages.items():
            update_stage_recall(
                stats[stage_name],
                stage_outputs,
                targets,
                top_k=int(args.top_k),
                rank_by=str(args.rank_by),
                line_width=float(args.line_width),
                min_valid_rows=int(args.min_valid_rows),
            )
    return stats


def print_stats_block(title: str, stats: dict[str, ProposalRecallStats], thresholds: tuple[float, ...]) -> None:
    print(f"{title}:")
    if not stats:
        print("  no prediction stages found")
        return
    for stage_name in sorted(stats.keys()):
        print("  " + format_summary(stage_name, stats[stage_name].summary(), list(thresholds)))


def category_lists(root: Path) -> dict[str, Path]:
    split_dir = root / "list" / "test_split"
    return {
        "normal": split_dir / "test0_normal.txt",
        "crowd": split_dir / "test1_crowd.txt",
        "hlight": split_dir / "test2_hlight.txt",
        "shadow": split_dir / "test3_shadow.txt",
        "noline": split_dir / "test4_noline.txt",
        "arrow": split_dir / "test5_arrow.txt",
        "curve": split_dir / "test6_curve.txt",
        "cross": split_dir / "test7_cross.txt",
        "night": split_dir / "test8_night.txt",
    }


def with_category_list(cfg: dict, list_path: Path) -> dict:
    out = deepcopy(cfg)
    dataset = dict(out.get("dataset", {}))
    lists = dict(dataset.get("lists", {}))
    lists["category"] = str(list_path.resolve())
    dataset["lists"] = lists
    out["dataset"] = dataset
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()

    thresholds = tuple(float(x) for x in args.iou_thresholds)
    print(f"config: {args.config}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"top_k: {args.top_k}")
    print(f"rank_by: {args.rank_by}")
    print(f"line_width: {args.line_width}")
    print(f"min_valid_rows: {args.min_valid_rows}")
    if not args.skip_overall:
        stats = collect_stats(model, cfg, args.split, device, thresholds, args, desc="proposal recall")
        print_stats_block("proposal_recall", stats, thresholds)

    if args.categories:
        root = Path(cfg.get("dataset", {}).get("root", "dataset"))
        print("category_proposal_recall:")
        for name, cat_list in category_lists(root).items():
            if not cat_list.exists():
                continue
            cat_cfg = with_category_list(cfg, cat_list)
            cat_stats = collect_stats(model, cat_cfg, "category", device, thresholds, args, desc=f"proposal recall {name}")
            print_stats_block(name, cat_stats, thresholds)


if __name__ == "__main__":
    main()
