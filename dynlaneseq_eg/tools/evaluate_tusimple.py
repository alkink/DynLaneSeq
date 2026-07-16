from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.tusimple_metric import TuSimpleLaneEval, format_tusimple_results
from dynlaneseq_eg.evaluation.tusimple_writer import outputs_to_tusimple_records, write_tusimple_json_lines
from dynlaneseq_eg.factory import build_dataloader, build_model


def resolve_ground_truth_path(cfg: dict, split: str) -> Path:
    dataset_cfg = cfg.get("dataset", {})
    root = Path(dataset_cfg.get("root", "dataset/tusimple")).expanduser()
    split_cfg = dataset_cfg.get("splits", {}).get(split)
    if split_cfg is None:
        defaults = {
            "val": {"annotations": ["train_set/label_data_0531.json"]},
            "test": {"annotations": ["test_label.json"]},
        }
        split_cfg = defaults.get(split)
    if split_cfg is None:
        raise KeyError(f"No TuSimple ground-truth configuration for split={split!r}")
    annotations = split_cfg["annotations"]
    if isinstance(annotations, (str, Path)):
        annotations = [annotations]
    if len(annotations) != 1:
        raise ValueError("TuSimple evaluation split must resolve to exactly one annotation JSON file")
    path = Path(annotations[0]).expanduser()
    return path if path.is_absolute() else root / path


@torch.inference_mode()
def collect_predictions(
    model,
    loader,
    device: torch.device,
    score_thresh: float,
    min_pred_points: int,
    nms_distance_thresh_px: float,
    nms_min_overlap_points: int,
    top_k: int,
    row_visibility_thresh: float,
    quality_score_power: float,
    channels_last: bool,
) -> list[dict]:
    model.eval()
    records: list[dict] = []
    for images, _, metas in tqdm(loader, ncols=80, desc="writing TuSimple predictions"):
        if channels_last:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        outputs = model(images)
        records.extend(
            outputs_to_tusimple_records(
                outputs,
                metas,
                score_thresh=score_thresh,
                min_pred_points=min_pred_points,
                nms_distance_thresh_px=nms_distance_thresh_px,
                nms_min_overlap_points=nms_min_overlap_points,
                top_k=top_k,
                row_visibility_thresh=row_visibility_thresh,
                quality_score_power=quality_score_power,
                run_time_ms=0.0,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Write predictions and run the official TuSimple metric.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prediction-file", default="")
    parser.add_argument("--score-thresh", type=float, default=0.30)
    parser.add_argument("--quality-score-power", type=float, default=None)
    parser.add_argument("--min-pred-points", type=int, default=5)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=None)
    parser.add_argument("--nms-min-overlap-points", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--row-visibility-thresh", type=float, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--output-txt", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not args.skip_write and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --skip-write is set")
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    if args.eval_batch_size > 0:
        cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    if str(cfg.get("dataset", {}).get("name", "")).lower() != "tusimple":
        raise ValueError("evaluate_tusimple requires dataset.name: TuSimple")

    post_cfg = cfg.get("postprocess", {})
    nms_distance_thresh_px = float(
        args.nms_distance_thresh_px
        if args.nms_distance_thresh_px is not None
        else post_cfg.get("lane_nms_distance_thresh_px", 20.0)
    )
    nms_min_overlap_points = int(
        args.nms_min_overlap_points
        if args.nms_min_overlap_points is not None
        else post_cfg.get("lane_nms_min_overlap_points", 5)
    )
    top_k = int(args.top_k if args.top_k is not None else post_cfg.get("top_k", 5))
    row_visibility_thresh = float(
        args.row_visibility_thresh
        if args.row_visibility_thresh is not None
        else post_cfg.get("row_visibility_thresh", 0.0)
    )
    quality_score_power = float(
        args.quality_score_power
        if args.quality_score_power is not None
        else post_cfg.get("quality_score_power", 0.5)
    )
    prediction_file = (
        Path(args.prediction_file)
        if args.prediction_file
        else Path(cfg.get("output_dir", "outputs")) / f"tusimple_{args.split}_predictions.jsonl"
    )

    device = torch.device(args.device)
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
    channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")

    if not args.skip_write:
        cfg.setdefault("model", {})["pretrained_backbone"] = False
        cfg.setdefault("model", {})["require_pretrained_backbone"] = False
        model = build_model(cfg).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        load_checkpoint(args.checkpoint, model, strict=False)
        loader = build_dataloader(cfg, split=args.split, training=False)
        records = collect_predictions(
            model,
            loader,
            device,
            score_thresh=float(args.score_thresh),
            min_pred_points=int(args.min_pred_points),
            nms_distance_thresh_px=nms_distance_thresh_px,
            nms_min_overlap_points=nms_min_overlap_points,
            top_k=top_k,
            row_visibility_thresh=row_visibility_thresh,
            quality_score_power=quality_score_power,
            channels_last=channels_last,
        )
        write_tusimple_json_lines(records, prediction_file)

    ground_truth_file = resolve_ground_truth_path(cfg, args.split)
    results = TuSimpleLaneEval.evaluate(prediction_file, ground_truth_file)
    report_lines = [
        f"config: {args.config}",
        f"checkpoint: {args.checkpoint}" if args.checkpoint else "checkpoint: <skip-write>",
        f"split: {args.split}",
        f"prediction_file: {prediction_file}",
        f"ground_truth_file: {ground_truth_file}",
        f"score_thresh: {float(args.score_thresh)}",
        f"quality_score_power: {quality_score_power}",
        f"lane_nms_distance_thresh_px: {nms_distance_thresh_px}",
        f"top_k: {top_k}",
        f"eval_batch_size: {cfg.get('dataloader', {}).get('eval_batch_size', 1)}",
        f"channels_last: {channels_last}",
        "run_time_ms: 0.0 (accuracy-only protocol, matching common public implementations)",
        format_tusimple_results(results),
    ]
    report = "\n".join(report_lines)
    print(report)

    if args.output_txt:
        output_txt = Path(args.output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text(report + "\n", encoding="utf-8")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "prediction_file": str(prediction_file),
                    "ground_truth_file": str(ground_truth_file),
                    "score_thresh": float(args.score_thresh),
                    "quality_score_power": quality_score_power,
                    "lane_nms_distance_thresh_px": nms_distance_thresh_px,
                    "top_k": top_k,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
