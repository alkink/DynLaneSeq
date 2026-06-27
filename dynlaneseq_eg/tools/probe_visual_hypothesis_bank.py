"""Frozen visual hypothesis-bank probe for gated affine lane correction.

This is the sequence-safe follow-up to ``probe_affine_noharm``.  The previous
probe showed that q_ins/row-token latent features do not generalize as a
no-harm gate across CULane sequences.  This probe tests the missing mechanism:
candidate-conditioned visual evidence.

For every deployed Top-K proposal, the script builds a small bank of affine
hypotheses around the predicted polyline:

    x_h(y) = x_pred(y) + a0 + a1 * y_norm

It samples scalar visual evidence maps along each hypothesis and side band
(segmentation logits, centerline logits, FPN activation energy, and image
luma by default), then trains a frozen-probe gate + affine regressor.  The gate
threshold is selected only on a calibration sequence split under a maximum
existing-TP action rate, and final rescue/kill counts are recomputed in exact
official raster IoU space on the test sequence split.

This is deliberately not a training script for DynLaneSeq.  It is a falsifying
diagnostic for whether visual hypothesis verification is worth implementing as
a real module.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    diagnostic_iou_matrix,
    evaluator_hungarian_assignment,
    stage_scores,
    override_eval_list,
)
from dynlaneseq_eg.evaluation.near_miss_oracles import fit_polynomial_delta
from dynlaneseq_eg.evaluation.proposal_recall import collect_prediction_stages
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, input_to_grid
from dynlaneseq_eg.tools.probe_affine_decodability import (
    _canonical_id,
    _geometry_feature,
    _load_torch,
    _stage_name,
)
from dynlaneseq_eg.tools.probe_affine_noharm import (
    KIND_OTHER_FP,
    KIND_REPAIRABLE,
    KIND_TP,
    SplitSets,
    _gt_rasters,
    _postprocess,
    _split_id,
    aggregate_runs,
    exact_iou_vector,
    split_images,
    train_one_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage", default="main")
    parser.add_argument("--score-thresh", type=float, default=0.40)
    parser.add_argument("--quality-power", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--near-min-iou", type=float, default=0.30)
    parser.add_argument("--near-max-iou", type=float, default=0.50)
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--max-affine-displacement", type=float, default=64.0)
    parser.add_argument("--row-segments", type=int, default=8)
    parser.add_argument("--summary-segments", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--split-group",
        choices=["image", "sequence"],
        default="sequence",
    )
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument(
        "--feature-kinds",
        nargs="+",
        choices=["visual_bank", "visual_bank_geometry"],
        default=["visual_bank", "visual_bank_geometry"],
    )
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument(
        "--feature-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Stored feature cache dtype. Training always converts to float32.",
    )
    parser.add_argument(
        "--translations-px",
        type=float,
        nargs="+",
        default=[-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0],
    )
    parser.add_argument(
        "--tilts-px",
        type=float,
        nargs="+",
        default=[-24.0, -12.0, 0.0, 12.0, 24.0],
        help="Affine slope displacement between y_norm=-1 and +1 is 2*tilt_px.",
    )
    parser.add_argument(
        "--side-offsets-px",
        type=float,
        nargs="+",
        default=[-16.0, -8.0, 0.0, 8.0, 16.0],
    )
    parser.add_argument(
        "--visual-map-kinds",
        nargs="+",
        choices=["seg_prob", "center_prob", "feature_abs", "feature_norm", "image_luma"],
        default=["seg_prob", "center_prob", "feature_abs", "image_luma"],
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--probe-batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--max-tp-action-rate", type=float, default=0.005)
    parser.add_argument("--min-gate-precision", type=float, default=0.50)
    parser.add_argument("--threshold-steps", type=int, default=1001)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _hypothesis_bank(args: argparse.Namespace) -> torch.Tensor:
    values = [
        (float(translation), float(tilt))
        for translation in args.translations_px
        for tilt in args.tilts_px
    ]
    if not values:
        raise ValueError("Empty affine hypothesis bank")
    return torch.tensor(values, dtype=torch.float32)


def _visual_maps(
    images: torch.Tensor,
    outputs: dict[str, Any],
    input_h: int,
    input_w: int,
    kinds: list[str],
) -> torch.Tensor:
    maps: list[torch.Tensor] = []
    for kind in kinds:
        if kind == "seg_prob":
            if "seg_logits" not in outputs:
                raise KeyError("visual map kind 'seg_prob' requires model output seg_logits")
            value = torch.sigmoid(outputs["seg_logits"].float())
            if value.shape[-2:] != (input_h, input_w):
                value = F.interpolate(value, size=(input_h, input_w), mode="bilinear", align_corners=False)
            maps.append(value)
        elif kind == "center_prob":
            if "centerline_logits" not in outputs:
                raise KeyError("visual map kind 'center_prob' requires model output centerline_logits")
            value = torch.sigmoid(outputs["centerline_logits"].float())
            if value.shape[-2:] != (input_h, input_w):
                value = F.interpolate(value, size=(input_h, input_w), mode="bilinear", align_corners=False)
            maps.append(value)
        elif kind == "feature_abs":
            if "features" not in outputs:
                raise KeyError("visual map kind 'feature_abs' requires return_features=True")
            value = outputs["features"].float().abs().mean(dim=1, keepdim=True)
            value = F.interpolate(value, size=(input_h, input_w), mode="bilinear", align_corners=False)
            maps.append(value)
        elif kind == "feature_norm":
            if "features" not in outputs:
                raise KeyError("visual map kind 'feature_norm' requires return_features=True")
            value = outputs["features"].float().square().mean(dim=1, keepdim=True).sqrt()
            value = F.interpolate(value, size=(input_h, input_w), mode="bilinear", align_corners=False)
            maps.append(value)
        elif kind == "image_luma":
            maps.append(images.float().mean(dim=1, keepdim=True))
        else:
            raise ValueError(f"Unsupported visual map kind: {kind}")
    return torch.cat(maps, dim=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum(dim=-2).clamp_min(1.0)
    return (values * weights).sum(dim=-2) / denom


def _masked_std(values: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum(dim=-2).clamp_min(1.0)
    var = ((values - mean.unsqueeze(-2)).square() * weights).sum(dim=-2) / denom
    return var.clamp_min(0.0).sqrt()


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    very_low = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, very_low)
    out = masked.amax(dim=-2)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def _segment_means(values: torch.Tensor, mask: torch.Tensor, segments: int) -> torch.Tensor:
    rows = values.shape[-2]
    chunks: list[torch.Tensor] = []
    for segment_id in range(int(segments)):
        start = int(round(rows * segment_id / float(segments)))
        end = int(round(rows * (segment_id + 1) / float(segments)))
        if end <= start:
            chunks.append(torch.zeros_like(values[..., 0, :]))
            continue
        chunks.append(_masked_mean(values[..., start:end, :], mask[..., start:end, :]))
    return torch.cat(chunks, dim=-1)


def _sample_visual_bank_for_image(
    maps: torch.Tensor,
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    bank: torch.Tensor,
    side_offsets: torch.Tensor,
    input_h: int,
    input_w: int,
    summary_segments: int,
) -> torch.Tensor:
    """Return one flattened visual-bank feature per proposal.

    Args:
        maps: [S, input_h, input_w]
        pred_x: [N, R]
        pred_mask: [N, R]
        bank: [H, 2] containing a0/a1.
        side_offsets: [O].
    """
    if pred_x.numel() == 0:
        return pred_x.new_zeros((0, 1))
    device = maps.device
    dtype = maps.dtype
    n, rows = pred_x.shape
    hyp_count = int(bank.shape[0])
    offset_count = int(side_offsets.numel())
    y_norm = torch.linspace(-1.0, 1.0, rows, device=device, dtype=dtype)
    bank = bank.to(device=device, dtype=dtype)
    side_offsets = side_offsets.to(device=device, dtype=dtype)
    deltas = bank[:, 0:1] + bank[:, 1:2] * y_norm.view(1, rows)
    curves = (pred_x.to(device=device, dtype=dtype).unsqueeze(1) + deltas.unsqueeze(0)).clamp(
        0.0, float(input_w - 1)
    )
    sample_x = (
        curves.unsqueeze(2) + side_offsets.view(1, 1, offset_count, 1)
    ).clamp(0.0, float(input_w - 1))
    flat_x = sample_x.reshape(1, n * hyp_count * offset_count, rows)
    y_rows = fixed_y_rows(rows, input_h, device=device, dtype=dtype).view(1, 1, rows).expand_as(flat_x)
    grid = input_to_grid(flat_x, y_rows, input_w=input_w, input_h=input_h).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        maps.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
    sampled = sampled.view(n, hyp_count, offset_count, rows, maps.shape[0])
    valid = pred_mask.to(device=device).view(n, 1, 1, rows, 1)

    mean = _masked_mean(sampled, valid)
    std = _masked_std(sampled, valid, mean)
    max_value = _masked_max(sampled, valid)
    segment_mean = _segment_means(sampled, valid, summary_segments)

    center_index = int(torch.argmin(side_offsets.abs()).item())
    center_mean = mean[:, :, center_index, :]
    side_ids = [idx for idx in range(offset_count) if idx != center_index]
    if side_ids:
        side_mean = mean[:, :, side_ids, :].mean(dim=2)
        side_max = mean[:, :, side_ids, :].amax(dim=2)
    else:
        side_mean = torch.zeros_like(center_mean)
        side_max = torch.zeros_like(center_mean)
    margin_mean = center_mean - side_mean
    margin_max = center_mean - side_max

    pieces = [
        mean.flatten(start_dim=1),
        std.flatten(start_dim=1),
        max_value.flatten(start_dim=1),
        segment_mean.flatten(start_dim=1),
        margin_mean.flatten(start_dim=1),
        margin_max.flatten(start_dim=1),
    ]
    return torch.cat(pieces, dim=1).detach().cpu()


def _store_dtype(tensor: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    return tensor.half() if args.feature_dtype == "float16" else tensor.float()


def extract_visual_features(
    cache: dict[str, Any],
    splits: SplitSets,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = override_eval_list(load_config(args.config), args.split, args.list_path or None)
    input_h = int(cfg.get("model", {}).get("input_h", 288))
    input_w = int(cfg.get("model", {}).get("input_w", 800))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = build_dataloader(cfg, split=args.split, training=False)

    record_by_key: dict[str, tuple[int, dict[str, Any]]] = {}
    for record_index, record in enumerate(cache["records"]):
        key = _canonical_id(record["image_id"])
        if key in record_by_key:
            raise RuntimeError(f"Canonical image ID collision: {key}")
        record_by_key[key] = (record_index, record)

    bank = _hypothesis_bank(args)
    side_offsets = torch.tensor(args.side_offsets_px, dtype=torch.float32)
    tensor_lists: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "visual_bank",
            "visual_bank_geometry",
            "gate_target",
            "kind",
            "coeff_target",
            "fit_mask",
            "pred_x",
            "pred_mask",
            "record_index",
            "proposal_id",
            "split",
        )
    }
    counts = defaultdict(int)
    seen: set[str] = set()
    all_split_ids = splits.train | splits.calibration | splits.test

    for images, _targets, metas in tqdm(loader, ncols=90, desc="extracting visual bank features"):
        keys = [_canonical_id(meta["image_path"]) for meta in metas]
        selected_batch = [index for index, key in enumerate(keys) if key in all_split_ids]
        if not selected_batch:
            continue
        selected_keys = [keys[index] for index in selected_batch]
        image_batch = images[selected_batch].to(device, non_blocking=True)
        with torch.inference_mode():
            outputs = model(image_batch, return_features=True)
        stages = collect_prediction_stages(outputs)
        if args.stage in stages:
            output_stage = stages[args.stage]
        elif args.stage == "main" and len(stages) == 1:
            output_stage = next(iter(stages.values()))
        else:
            raise KeyError(f"Stage {args.stage!r} missing; available={list(stages)}")
        visual_maps = _visual_maps(
            image_batch,
            outputs,
            input_h=input_h,
            input_w=input_w,
            kinds=list(args.visual_map_kinds),
        )

        for local_id, image_key in enumerate(selected_keys):
            seen.add(image_key)
            record_index, record = record_by_key[image_key]
            stage_name = _stage_name(record, args.stage)
            cached_stage = record["stages"][stage_name]
            official, _valid_gt, _candidate_valid = diagnostic_iou_matrix(
                record,
                stage_name,
                use_official=True,
                input_h=input_h,
                input_w=input_w,
                line_width=args.line_width,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            trace = _postprocess(cached_stage, cache, args)
            selected_ids = list(trace["selected_ids"])
            assignment = evaluator_hungarian_assignment(official, selected_ids, args.iou_thresh)
            tp_ids = set(assignment.proposal_ids)

            target_x = record["target"]["x_rows"].float()
            target_mask = record["target"]["valid_mask"].bool() & torch.isfinite(target_x)
            valid_gt = target_mask.sum(dim=-1) >= args.min_valid_rows
            target_x = target_x[valid_gt]
            target_mask = target_mask[valid_gt]
            if target_x.shape[0] != official.shape[0]:
                counts["skipped_gt_alignment_images"] += 1
                continue

            pred_x_all, pred_mask_all, _ = candidate_row_masks(
                cached_stage,
                input_h=input_h,
                input_w=input_w,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            selected_pred_x = pred_x_all[selected_ids] if selected_ids else pred_x_all.new_zeros((0, pred_x_all.shape[-1]))
            selected_pred_mask = (
                pred_mask_all[selected_ids] if selected_ids else pred_mask_all.new_zeros((0, pred_mask_all.shape[-1]))
            )
            visual_features = _sample_visual_bank_for_image(
                visual_maps[local_id].detach(),
                selected_pred_x,
                selected_pred_mask,
                bank,
                side_offsets,
                input_h=input_h,
                input_w=input_w,
                summary_segments=args.summary_segments,
            )
            gt_masks: list[Any] = []
            gt_counts: list[int] = []

            scores = stage_scores(cached_stage, quality_power=args.quality_power)
            for selected_local_id, proposal_id in enumerate(selected_ids):
                counts["selected"] += 1
                kind = KIND_OTHER_FP
                gate_target = False
                coeff = torch.zeros(2, dtype=torch.float32)
                fit_mask = torch.zeros(pred_x_all.shape[-1], dtype=torch.bool)
                pred_x = pred_x_all[proposal_id]
                pred_mask = pred_mask_all[proposal_id]

                if proposal_id in tp_ids:
                    kind = KIND_TP
                    counts["tp"] += 1
                elif official.shape[0] > 0:
                    best_iou, gt_id_tensor = official[:, proposal_id].max(dim=0)
                    best_iou_value = float(best_iou)
                    gt_id = int(gt_id_tensor)
                    if args.near_min_iou <= best_iou_value <= args.near_max_iou:
                        counts["near_miss"] += 1
                        fitted = fit_polynomial_delta(
                            pred_x,
                            pred_mask,
                            target_x[gt_id],
                            target_mask[gt_id],
                            degree=1,
                        )
                        if fitted is not None:
                            if not gt_masks:
                                gt_masks, gt_counts = _gt_rasters(record, args.line_width)
                            y = torch.linspace(-1.0, 1.0, pred_x.numel())
                            delta = (fitted[0] + fitted[1] * y).clamp(
                                -args.max_affine_displacement,
                                args.max_affine_displacement,
                            )
                            corrected = (pred_x + delta).clamp(0.0, float(input_w - 1))
                            exact_vector = exact_iou_vector(
                                corrected,
                                pred_mask,
                                record,
                                gt_masks,
                                gt_counts,
                                args.line_width,
                            )
                            if gt_id < exact_vector.numel() and float(exact_vector[gt_id]) > args.iou_thresh:
                                kind = KIND_REPAIRABLE
                                gate_target = True
                                coeff = fitted.float()
                                fit_mask = pred_mask & target_mask[gt_id]
                                counts["repairable"] += 1
                if kind == KIND_OTHER_FP:
                    counts["other_fp"] += 1

                base_visual = visual_features[selected_local_id].float()
                geometry = _geometry_feature(
                    pred_x,
                    pred_mask,
                    cached_stage,
                    proposal_id,
                    input_w,
                    args.row_segments,
                ).float()
                score_feature = torch.tensor(
                    [
                        float(scores[proposal_id]),
                        float(pred_mask.float().mean()),
                    ],
                    dtype=torch.float32,
                )
                tensor_lists["visual_bank"].append(_store_dtype(base_visual, args))
                tensor_lists["visual_bank_geometry"].append(
                    _store_dtype(torch.cat([base_visual, geometry, score_feature]), args)
                )
                tensor_lists["gate_target"].append(torch.tensor(float(gate_target)))
                tensor_lists["kind"].append(torch.tensor(kind))
                tensor_lists["coeff_target"].append(coeff)
                tensor_lists["fit_mask"].append(fit_mask)
                tensor_lists["pred_x"].append(pred_x)
                tensor_lists["pred_mask"].append(pred_mask)
                tensor_lists["record_index"].append(torch.tensor(record_index))
                tensor_lists["proposal_id"].append(torch.tensor(proposal_id))
                tensor_lists["split"].append(torch.tensor(_split_id(image_key, splits)))

        if seen >= all_split_ids:
            break

    missing = all_split_ids - seen
    if missing:
        raise RuntimeError(f"Missing {len(missing)} images from dataloader; examples={sorted(missing)[:5]}")
    data: dict[str, Any] = {key: torch.stack(values) for key, values in tensor_lists.items()}
    data["metadata"] = {
        "counts": dict(counts),
        "input_h": input_h,
        "input_w": input_w,
        "split_seed": args.split_seed,
        "split_group": args.split_group,
        "train_images": len(splits.train),
        "calibration_images": len(splits.calibration),
        "test_images": len(splits.test),
        "visual_map_kinds": list(args.visual_map_kinds),
        "translations_px": list(args.translations_px),
        "tilts_px": list(args.tilts_px),
        "side_offsets_px": list(args.side_offsets_px),
        "hypothesis_count": int(bank.shape[0]),
        "summary_segments": int(args.summary_segments),
        "feature_dtype": str(args.feature_dtype),
    }
    return data


def _refresh_split(data: dict[str, Any], cache: dict[str, Any], splits: SplitSets, args: argparse.Namespace) -> None:
    record_split = torch.full((len(cache["records"]),), -1, dtype=torch.long)
    for record_index, record in enumerate(cache["records"]):
        image_id = _canonical_id(record["image_id"])
        record_split[record_index] = _split_id(image_id, splits)
    data["split"] = record_split[data["record_index"].long()]
    data.setdefault("metadata", {})["split_seed"] = args.split_seed
    data["metadata"]["split_group"] = args.split_group
    data["metadata"]["train_images"] = len(splits.train)
    data["metadata"]["calibration_images"] = len(splits.calibration)
    data["metadata"]["test_images"] = len(splits.test)


def main() -> None:
    args = parse_args()
    cache = _load_torch(args.candidate_cache)
    image_ids = [_canonical_id(record["image_id"]) for record in cache["records"]]
    splits = split_images(image_ids, args)
    feature_path = Path(args.feature_cache)
    if args.reuse_feature_cache and feature_path.exists():
        data = _load_torch(feature_path)
    else:
        data = extract_visual_features(cache, splits, args)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, feature_path)

    _refresh_split(data, cache, splits, args)

    runs: list[dict[str, Any]] = []
    for seed in args.probe_seeds:
        for feature_name in args.feature_kinds:
            run = train_one_probe(
                feature_name,
                data[feature_name],
                data,
                cache,
                int(seed),
                args,
            )
            runs.append(run)
            official = run["test_official"]
            gate = run["test_gate"]
            print(
                f"{feature_name:>20} seed={seed}: "
                f"gateP/R={gate['precision']:.3f}/{gate['recall']:.3f} "
                f"tpAction={gate['tp_action_rate']:.4f} "
                f"rescued/killed/net={official['rescued_tp']}/"
                f"{official['killed_tp']}/{official['net_tp']} "
                f"dF1={official['f1_delta']:+.5f}"
            )

    payload = {
        "metadata": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "candidate_cache": args.candidate_cache,
            "feature_cache": str(feature_path),
            "feature_extraction": data.get("metadata", {}),
            "postprocess_order": "correction_after_nms_and_topk",
            "max_tp_action_rate": args.max_tp_action_rate,
            "min_gate_precision": args.min_gate_precision,
            "split_seed": args.split_seed,
            "split_group": args.split_group,
            "probe_seeds": args.probe_seeds,
            "feature_kinds": args.feature_kinds,
        },
        "summary": aggregate_runs(runs),
        "runs": runs,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
