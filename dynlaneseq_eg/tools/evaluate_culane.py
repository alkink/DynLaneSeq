from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.culane_metric import eval_predictions, format_category_results, format_results
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model


@torch.no_grad()
def write_predictions(
    model,
    loader,
    device: torch.device,
    pred_dir: Path,
    score_thresh: float,
    min_pred_points: int,
    nms_distance_thresh_px: float,
    nms_min_overlap_points: int,
    top_k: int,
    row_visibility_thresh: float,
    quality_score_power: float,
) -> None:
    model.eval()
    pred_dir.mkdir(parents=True, exist_ok=True)
    for images, targets, metas in tqdm(loader, ncols=80, desc="writing predictions"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        write_culane_predictions(
            outputs,
            metas,
            pred_dir,
            score_thresh=score_thresh,
            min_pred_points=min_pred_points,
            nms_distance_thresh_px=nms_distance_thresh_px,
            nms_min_overlap_points=nms_min_overlap_points,
            top_k=top_k,
            row_visibility_thresh=row_visibility_thresh,
            quality_score_power=quality_score_power,
        )


def resolve_list_path(cfg: dict, split: str) -> Path:
    dataset = cfg.get("dataset", {})
    root = Path(dataset.get("root", "dataset"))
    rel = dataset.get("lists", {}).get(split, {"train": "list/train_gt.txt", "val": "list/val.txt", "test": "list/test.txt"}.get(split))
    if rel is None:
        raise KeyError(f"No dataset list configured for split={split}")
    path = Path(rel)
    return path if path.is_absolute() else root / path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Write DynLaneSeq predictions and evaluate CULane F1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pred-dir", default="")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--min-pred-points", type=int, default=5)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=None)
    parser.add_argument("--nms-min-overlap-points", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--row-visibility-thresh", type=float, default=None)
    parser.add_argument("--quality-score-power", type=float, default=None)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--continuous", action="store_true", help="Use shapely continuous IoU instead of CULane-style raster IoU.")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--skip-write", action="store_true", help="Evaluate existing files in --pred-dir.")
    parser.add_argument("--categories", action="store_true", help="Also evaluate CULane official test_split categories.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    post_cfg = cfg.get("postprocess", {})
    nms_distance_thresh_px = (
        float(args.nms_distance_thresh_px)
        if args.nms_distance_thresh_px is not None
        else float(post_cfg.get("lane_nms_distance_thresh_px", 0.0))
    )
    nms_min_overlap_points = (
        int(args.nms_min_overlap_points)
        if args.nms_min_overlap_points is not None
        else int(post_cfg.get("lane_nms_min_overlap_points", 5))
    )
    top_k = int(args.top_k) if args.top_k is not None else int(post_cfg.get("top_k", 0))
    row_visibility_thresh = (
        float(args.row_visibility_thresh)
        if args.row_visibility_thresh is not None
        else float(post_cfg.get("row_visibility_thresh", 0.0))
    )
    quality_score_power = (
        float(args.quality_score_power)
        if args.quality_score_power is not None
        else float(post_cfg.get("quality_score_power", 0.0))
    )
    pred_dir = Path(args.pred_dir) if args.pred_dir else Path(cfg.get("output_dir", "outputs")) / f"culane_pred_{args.split}_thr{args.score_thresh:g}"

    if not args.skip_write:
        model = build_model(cfg).to(device)
        load_checkpoint(args.checkpoint, model, strict=False)
        loader = build_dataloader(cfg, split=args.split, training=False)
        write_predictions(
            model,
            loader,
            device,
            pred_dir,
            args.score_thresh,
            args.min_pred_points,
            nms_distance_thresh_px,
            nms_min_overlap_points,
            top_k,
            row_visibility_thresh,
            quality_score_power,
        )

    list_path = resolve_list_path(cfg, args.split)
    anno_dir = Path(cfg.get("dataset", {}).get("root", "dataset"))
    results = eval_predictions(
        pred_dir=pred_dir,
        anno_dir=anno_dir,
        list_path=list_path,
        iou_thresholds=args.iou_thresholds,
        width=args.width,
        official=not args.continuous,
        sequential=args.sequential,
    )
    print(f"pred_dir: {pred_dir}")
    print(f"anno_dir: {anno_dir}")
    print(f"list_path: {list_path}")
    print(f"lane_nms_distance_thresh_px: {nms_distance_thresh_px}")
    print(f"top_k: {top_k}")
    print(f"row_visibility_thresh: {row_visibility_thresh}")
    print(f"quality_score_power: {quality_score_power}")
    print(format_results(results))

    if args.categories:
        cat_results = {}
        for name, cat_list in category_lists(anno_dir).items():
            if not cat_list.exists():
                continue
            print(f"category {name}: {cat_list}")
            cat_results[name] = eval_predictions(
                pred_dir=pred_dir,
                anno_dir=anno_dir,
                list_path=cat_list,
                iou_thresholds=args.iou_thresholds,
                width=args.width,
                official=not args.continuous,
                sequential=args.sequential,
            )
        if cat_results:
            print("categories:")
            print(format_category_results(cat_results))


if __name__ == "__main__":
    main()
