from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.culane_metric import (
    eval_predictions,
    eval_predictions_with_categories,
    format_category_results,
    format_results,
)
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model


@torch.inference_mode()
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
    score_mode: str,
    channels_last: bool = False,
    pass_targets: bool = False,
    inference_only: bool = False,
    amp_dtype: str = "none",
) -> None:
    model.eval()
    pred_dir.mkdir(parents=True, exist_ok=True)
    for images, targets, metas in tqdm(loader, ncols=80, desc="writing predictions"):
        if channels_last:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        if amp_dtype == "float16" and device.type == "cuda":
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        elif amp_dtype == "bfloat16" and device.type == "cuda":
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            amp_context = nullcontext()
        with amp_context:
            if pass_targets:
                outputs = model(images, targets=targets)
            elif inference_only:
                outputs = model(images, inference_only=True)
            else:
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
            score_mode=score_mode,
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
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset-root", default="", help="Override cfg.dataset.root.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pred-dir", default="")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--min-pred-points", type=int, default=5)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=None)
    parser.add_argument("--nms-min-overlap-points", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--row-visibility-thresh", type=float, default=None)
    parser.add_argument("--quality-score-power", type=float, default=None)
    parser.add_argument(
        "--score-mode",
        choices=(
            "exist",
            "quality",
            "exist_quality",
            "selection",
            "pointer",
            "four_slot",
        ),
        default=None,
        help="Candidate score source; default comes from postprocess.score_mode.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=-1, help="Override dataloader workers; -1 keeps config.")
    parser.add_argument(
        "--eval-prefetch-factor",
        type=int,
        default=-1,
        help="Override dataloader prefetch factor; -1 keeps config.",
    )
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--continuous", action="store_true", help="Use shapely continuous IoU instead of CULane-style raster IoU.")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--metric-workers", type=int, default=0, help="Raster metric workers; 0 uses all CPU cores.")
    parser.add_argument("--metric-chunksize", type=int, default=64)
    parser.add_argument("--skip-write", action="store_true", help="Evaluate existing files in --pred-dir.")
    parser.add_argument("--categories", action="store_true", help="Also evaluate CULane official test_split categories.")
    parser.add_argument("--output-txt", default="", help="Write a compact evaluation report to this text file.")
    parser.add_argument("--output-json", default="", help="Write metrics and metadata to this JSON file.")
    parser.add_argument(
        "--no-pretrained-init",
        action="store_true",
        help="Do not initialize pretrained backbone weights before loading the complete checkpoint.",
    )
    parser.add_argument(
        "--legacy-inference",
        action="store_true",
        help="Disable the exact-output fast path for parity/debugging.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
        help="Optional CUDA autocast. 'none' preserves the original FP32 numerical path.",
    )
    parser.add_argument("--compile-model", action="store_true", help="Optionally run inference through torch.compile.")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    args = parser.parse_args()

    if not args.skip_write and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --skip-write is set.")
    if args.skip_write and not args.pred_dir:
        raise ValueError("--pred-dir is required when --skip-write is set.")

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    if args.no_pretrained_init and not args.skip_write:
        model_cfg = cfg.setdefault("model", {})
        model_cfg["pretrained_backbone"] = False
        model_cfg["require_pretrained_backbone"] = False
    device = torch.device(args.device)
    if args.amp_dtype != "none" and device.type != "cuda":
        raise ValueError("--amp-dtype currently requires a CUDA device")
    if args.compile_model and not hasattr(torch, "compile"):
        raise RuntimeError("--compile-model requires a PyTorch build with torch.compile")
    train_cfg = cfg.get("training", {})
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", False))
        if bool(train_cfg.get("tf32", False)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    if args.eval_batch_size > 0:
        cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    if args.eval_num_workers >= 0:
        cfg.setdefault("dataloader", {})["num_workers"] = int(args.eval_num_workers)
    if args.eval_prefetch_factor > 0:
        cfg.setdefault("dataloader", {})["prefetch_factor"] = int(args.eval_prefetch_factor)
    channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")
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
    score_mode = str(args.score_mode or post_cfg.get("score_mode", "exist_quality"))
    pred_dir = Path(args.pred_dir) if args.pred_dir else Path(cfg.get("output_dir", "outputs")) / f"culane_pred_{args.split}_thr{args.score_thresh:g}"

    inference_only = False
    pass_targets = False
    if not args.skip_write:
        # Checkpoint evaluation must not download or overwrite a backbone with
        # ImageNet initialization before loading the trained model state.
        cfg.setdefault("model", {})["pretrained_backbone"] = False
        cfg.setdefault("model", {})["require_pretrained_backbone"] = False
        model = build_model(cfg)
        load_checkpoint(args.checkpoint, model, strict=False)
        pass_targets = bool(getattr(model, "oracle_coarse_enabled", False))
        inference_only = bool(getattr(model, "supports_inference_only", False) and not args.legacy_inference)
        if inference_only and not pass_targets:
            cfg.setdefault("dataset", {})["load_targets"] = False
            cfg.setdefault("dataset", {})["infer_seg_labels"] = False
        loader = build_dataloader(cfg, split=args.split, training=False)
        if inference_only and hasattr(model, "prepare_for_inference"):
            model.prepare_for_inference()
        model = model.to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        if args.compile_model:
            model = torch.compile(model, mode=args.compile_mode)
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
            score_mode,
            channels_last=channels_last,
            pass_targets=pass_targets,
            inference_only=inference_only,
            amp_dtype=args.amp_dtype,
        )

    list_path = resolve_list_path(cfg, args.split)
    anno_dir = Path(cfg.get("dataset", {}).get("root", "dataset"))
    if args.categories:
        results, cat_results = eval_predictions_with_categories(
            pred_dir=pred_dir,
            anno_dir=anno_dir,
            list_path=list_path,
            category_lists=category_lists(anno_dir),
            iou_thresholds=args.iou_thresholds,
            width=args.width,
            official=not args.continuous,
            sequential=args.sequential,
            num_workers=args.metric_workers,
            chunksize=args.metric_chunksize,
        )
    else:
        results = eval_predictions(
            pred_dir=pred_dir,
            anno_dir=anno_dir,
            list_path=list_path,
            iou_thresholds=args.iou_thresholds,
            width=args.width,
            official=not args.continuous,
            sequential=args.sequential,
            num_workers=args.metric_workers,
            chunksize=args.metric_chunksize,
        )
        cat_results = {}
    report_lines = [
        f"config: {args.config}",
        f"checkpoint: {args.checkpoint}" if args.checkpoint else "checkpoint: <skip-write>",
        f"split: {args.split}",
        f"pred_dir: {pred_dir}",
        f"anno_dir: {anno_dir}",
        f"list_path: {list_path}",
        f"lane_nms_distance_thresh_px: {nms_distance_thresh_px}",
        f"top_k: {top_k}",
        f"row_visibility_thresh: {row_visibility_thresh}",
        f"quality_score_power: {quality_score_power}",
        f"score_mode: {score_mode}",
        f"score_thresh: {args.score_thresh}",
        f"eval_batch_size: {cfg.get('dataloader', {}).get('eval_batch_size', 1)}",
        f"channels_last: {channels_last}",
        f"inference_only: {inference_only}",
        f"amp_dtype: {args.amp_dtype}",
        f"compile_model: {args.compile_model}",
        f"no_pretrained_init: {args.no_pretrained_init}",
        f"metric_workers: {args.metric_workers}",
        f"metric_chunksize: {args.metric_chunksize}",
        format_results(results),
    ]
    category_report = ""

    if args.categories:
        if cat_results:
            category_report = format_category_results(cat_results)
            report_lines.extend(["categories:", category_report])
    else:
        cat_results = {}

    report = "\n".join(report_lines)
    print(report)

    if args.output_txt:
        output_txt = Path(args.output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text(report + "\n", encoding="utf-8")
        print(f"output_txt: {output_txt}")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "pred_dir": str(pred_dir),
            "anno_dir": str(anno_dir),
            "list_path": str(list_path),
            "score_thresh": args.score_thresh,
            "lane_nms_distance_thresh_px": nms_distance_thresh_px,
            "top_k": top_k,
            "row_visibility_thresh": row_visibility_thresh,
            "quality_score_power": quality_score_power,
            "score_mode": score_mode,
            "eval_batch_size": cfg.get("dataloader", {}).get("eval_batch_size", 1),
            "channels_last": channels_last,
            "inference_only": inference_only,
            "amp_dtype": args.amp_dtype,
            "compile_model": args.compile_model,
            "compile_mode": args.compile_mode,
            "no_pretrained_init": args.no_pretrained_init,
            "metric_workers": args.metric_workers,
            "metric_chunksize": args.metric_chunksize,
            "results": results,
            "categories": cat_results,
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
