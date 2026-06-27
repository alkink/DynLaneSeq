"""Frozen row-segment visual hypothesis ranker for no-harm affine correction.

This probe is stricter than ``probe_visual_hypothesis_bank``:

* no continuous a0/a1 regression;
* each repairable near-miss is assigned to a discrete affine hypothesis;
* the model predicts no-action vs one of the hypotheses;
* visual evidence is preserved as [hypothesis, row-segment, channel] instead
  of collapsing the whole curve to a single summary vector.

The goal is not to ship this exact module.  The goal is to answer one question:
does candidate-conditioned visual hypothesis ranking generalize across CULane
sequences under a no-harm calibration constraint?
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
from dynlaneseq_eg.evaluation.proposal_recall import collect_prediction_stages, line_iou_against_gt
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
    auroc,
    average_precision,
    exact_iou_vector,
    exact_system_evaluation,
    gate_metrics,
    split_images,
)
from dynlaneseq_eg.tools.probe_visual_hypothesis_bank import (
    _hypothesis_bank,
    _store_dtype,
    _visual_maps,
)


class HypothesisRanker(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        geometry_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.geometry_dim = int(geometry_dim)
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.segment_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.geometry_proj = (
            nn.Sequential(
                nn.LayerNorm(geometry_dim),
                nn.Linear(geometry_dim, hidden_dim),
                nn.GELU(),
            )
            if geometry_dim > 0
            else None
        )
        score_dim = hidden_dim * 2 + (hidden_dim if geometry_dim > 0 else 0)
        self.hypothesis_score = nn.Sequential(
            nn.LayerNorm(score_dim),
            nn.Linear(score_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.no_action_score = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + (hidden_dim if geometry_dim > 0 else 0)),
            nn.Linear(hidden_dim * 4 + (hidden_dim if geometry_dim > 0 else 0), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, visual_seq: torch.Tensor, geometry: torch.Tensor | None = None) -> torch.Tensor:
        # visual_seq: [B, H, S, C]
        batch, hypotheses, segments, _channels = visual_seq.shape
        x = self.input(visual_seq)
        flat = x.view(batch * hypotheses, segments, -1).transpose(1, 2)
        flat = self.segment_conv(flat).transpose(1, 2)
        flat = flat.reshape(batch, hypotheses, segments, -1)
        hyp_mean = flat.mean(dim=2)
        hyp_max = flat.amax(dim=2)
        hyp_repr = torch.cat([hyp_mean, hyp_max], dim=-1)
        geom_repr = None
        if self.geometry_proj is not None:
            if geometry is None:
                raise RuntimeError("geometry input is required for this ranker")
            geom_repr = self.geometry_proj(geometry)
            hyp_repr = torch.cat(
                [hyp_repr, geom_repr[:, None, :].expand(-1, hypotheses, -1)],
                dim=-1,
            )
        action_logits = self.hypothesis_score(hyp_repr).squeeze(-1)
        global_mean = hyp_mean.mean(dim=1)
        global_max = hyp_mean.amax(dim=1)
        global_hyp_max = hyp_max.amax(dim=1)
        global_hyp_mean = hyp_max.mean(dim=1)
        no_action_input = torch.cat(
            [global_mean, global_max, global_hyp_mean, global_hyp_max]
            + ([geom_repr] if geom_repr is not None else []),
            dim=-1,
        )
        no_action = self.no_action_score(no_action_input)
        return torch.cat([no_action, action_logits], dim=1)


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
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--split-group", choices=["image", "sequence"], default="sequence")
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument(
        "--feature-kinds",
        nargs="+",
        choices=["visual_rank", "visual_rank_geometry"],
        default=["visual_rank", "visual_rank_geometry"],
    )
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
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
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-tp-action-rate", type=float, default=0.005)
    parser.add_argument("--min-gate-precision", type=float, default=0.50)
    parser.add_argument("--threshold-steps", type=int, default=1001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _sample_rank_features_for_image(
    maps: torch.Tensor,
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    bank: torch.Tensor,
    side_offsets: torch.Tensor,
    input_h: int,
    input_w: int,
    row_segments: int,
) -> torch.Tensor:
    """Return [N, H, S, D] visual features."""
    if pred_x.numel() == 0:
        return pred_x.new_zeros((0, int(bank.shape[0]), int(row_segments), 1))
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
    center_index = int(torch.argmin(side_offsets.abs()).item())
    left_ids = [idx for idx, value in enumerate(side_offsets.tolist()) if value < 0]
    right_ids = [idx for idx, value in enumerate(side_offsets.tolist()) if value > 0]
    side_ids = [idx for idx in range(offset_count) if idx != center_index]
    if not side_ids:
        side_ids = [center_index]
    pieces: list[torch.Tensor] = []
    for segment_id in range(int(row_segments)):
        start = int(round(rows * segment_id / float(row_segments)))
        end = int(round(rows * (segment_id + 1) / float(row_segments)))
        segment = sampled[..., start:end, :]
        segment_valid = valid[..., start:end, :]
        weights = segment_valid.to(dtype=dtype)
        denom = weights.sum(dim=-2).clamp_min(1.0)
        center = (segment[:, :, center_index] * weights.squeeze(2)).sum(dim=-2) / denom.squeeze(2)
        if left_ids:
            left = (segment[:, :, left_ids].mean(dim=2) * weights.squeeze(2)).sum(dim=-2) / denom.squeeze(2)
        else:
            left = center
        if right_ids:
            right = (segment[:, :, right_ids].mean(dim=2) * weights.squeeze(2)).sum(dim=-2) / denom.squeeze(2)
        else:
            right = center
        side = (segment[:, :, side_ids].mean(dim=2) * weights.squeeze(2)).sum(dim=-2) / denom.squeeze(2)
        side_max = segment[:, :, side_ids].amax(dim=2)
        very_low = torch.finfo(dtype).min
        side_valid = segment_valid.squeeze(2).expand(-1, hyp_count, -1, maps.shape[0])
        side_max = side_max.masked_fill(~side_valid, very_low).amax(dim=-2)
        segment_has_valid = segment_valid.squeeze(2).any(dim=-2).expand(-1, hyp_count, maps.shape[0])
        side_max = torch.where(segment_has_valid, side_max, side)
        pieces.append(
            torch.cat(
                [
                    center,
                    left,
                    right,
                    side,
                    center - side,
                    center - side_max,
                ],
                dim=-1,
            )
        )
    features = torch.stack(pieces, dim=2)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.detach().cpu()


def _best_bank_label(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    target_x: torch.Tensor,
    target_mask: torch.Tensor,
    record: dict[str, Any],
    gt_id: int,
    gt_masks: list[Any],
    gt_counts: list[int],
    bank: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[int, torch.Tensor, bool]:
    rows = pred_x.numel()
    y = torch.linspace(-1.0, 1.0, rows)
    bank_delta = bank[:, 0:1] + bank[:, 1:2] * y.view(1, rows)
    bank_delta = bank_delta.clamp(-args.max_affine_displacement, args.max_affine_displacement)
    candidates = (pred_x.view(1, rows) + bank_delta).clamp(
        0.0, float(record["meta"].get("input_w", 800) - 1)
    )
    row_scores = line_iou_against_gt(
        candidates,
        target_x,
        target_mask & pred_mask,
        line_width=args.line_width,
    )
    best_id = int(row_scores.argmax().item()) if row_scores.numel() else 0
    exact_vector = exact_iou_vector(
        candidates[best_id],
        pred_mask,
        record,
        gt_masks,
        gt_counts,
        args.line_width,
    )
    rescued = gt_id < exact_vector.numel() and float(exact_vector[gt_id]) > args.iou_thresh
    return best_id, bank[best_id].float(), bool(rescued)


def extract_rank_features(
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
            "visual_rank",
            "geometry",
            "label",
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

    for images, _targets, metas in tqdm(loader, ncols=90, desc="extracting visual rank features"):
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
            rank_features = _sample_rank_features_for_image(
                visual_maps[local_id].detach(),
                selected_pred_x,
                selected_pred_mask,
                bank,
                side_offsets,
                input_h=input_h,
                input_w=input_w,
                row_segments=args.row_segments,
            )
            scores = stage_scores(cached_stage, quality_power=args.quality_power)
            gt_masks: list[Any] = []
            gt_counts: list[int] = []
            for selected_local_id, proposal_id in enumerate(selected_ids):
                counts["selected"] += 1
                kind = KIND_OTHER_FP
                gate_target = False
                label = 0
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
                            counts["continuous_affine_fit"] += 1
                            if not gt_masks:
                                gt_masks, gt_counts = _gt_rasters(record, args.line_width)
                            best_hypothesis, best_coeff, rescued = _best_bank_label(
                                pred_x,
                                pred_mask,
                                target_x[gt_id],
                                target_mask[gt_id],
                                record,
                                gt_id,
                                gt_masks,
                                gt_counts,
                                bank,
                                args,
                            )
                            if rescued:
                                kind = KIND_REPAIRABLE
                                gate_target = True
                                label = best_hypothesis + 1
                                coeff = best_coeff
                                fit_mask = pred_mask & target_mask[gt_id]
                                counts["repairable"] += 1
                if kind == KIND_OTHER_FP:
                    counts["other_fp"] += 1

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
                tensor_lists["visual_rank"].append(
                    _store_dtype(rank_features[selected_local_id].float(), args)
                )
                tensor_lists["geometry"].append(torch.cat([geometry, score_feature]))
                tensor_lists["label"].append(torch.tensor(label, dtype=torch.long))
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
        "row_segments": int(args.row_segments),
        "feature_dtype": str(args.feature_dtype),
    }
    return data


def choose_threshold_from_scores(
    scores: torch.Tensor,
    kind: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float | int]]:
    if scores.numel() == 0:
        threshold = 1.000001
        return threshold, gate_metrics(scores, kind, threshold)
    order = scores.argsort(descending=True)
    sorted_scores = scores[order]
    sorted_kind = kind[order]
    positive = (sorted_kind == KIND_REPAIRABLE).long()
    existing_tp = (sorted_kind == KIND_TP).long()
    cumulative_positive = positive.cumsum(0)
    cumulative_tp = existing_tp.cumsum(0)
    action_count = torch.arange(1, scores.numel() + 1, dtype=torch.long)
    boundary = torch.ones(scores.numel(), dtype=torch.bool)
    if scores.numel() > 1:
        boundary[:-1] = sorted_scores[:-1] > sorted_scores[1:]
    total_tp = max(int(existing_tp.sum()), 1)
    precision = cumulative_positive.float() / action_count.float()
    tp_action_rate = cumulative_tp.float() / float(total_tp)
    feasible = boundary
    feasible &= precision >= float(args.min_gate_precision)
    feasible &= tp_action_rate <= float(args.max_tp_action_rate)
    feasible_ids = feasible.nonzero(as_tuple=False).flatten()
    if feasible_ids.numel() == 0:
        threshold = 1.000001
        return threshold, gate_metrics(scores, kind, threshold)
    best_index = int(feasible_ids[0])
    best_rank = (
        int(cumulative_positive[best_index]),
        float(precision[best_index]),
        float(sorted_scores[best_index]),
    )
    for candidate_index in feasible_ids[1:].tolist():
        rank = (
            int(cumulative_positive[candidate_index]),
            float(precision[candidate_index]),
            float(sorted_scores[candidate_index]),
        )
        if rank > best_rank:
            best_index = int(candidate_index)
            best_rank = rank
    threshold = float(sorted_scores[best_index])
    return threshold, gate_metrics(scores, kind, threshold)


def normalize_rank_inputs(
    visual: torch.Tensor,
    geometry: torch.Tensor,
    train_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    visual = torch.nan_to_num(visual.float(), nan=0.0, posinf=0.0, neginf=0.0)
    geometry = torch.nan_to_num(geometry.float(), nan=0.0, posinf=0.0, neginf=0.0)
    train_visual = visual[train_mask].float()
    mean = train_visual.mean(dim=(0, 1, 2), keepdim=True)
    std = train_visual.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-5)
    normalized_visual = (visual.float() - mean) / std
    train_geometry = geometry[train_mask].float()
    geometry_mean = train_geometry.mean(dim=0, keepdim=True)
    geometry_std = train_geometry.std(dim=0, keepdim=True).clamp_min(1e-5)
    normalized_geometry = (geometry.float() - geometry_mean) / geometry_std
    stats = {
        "visual_mean": mean.cpu(),
        "visual_std": std.cpu(),
        "geometry_mean": geometry_mean.cpu(),
        "geometry_std": geometry_std.cpu(),
    }
    return normalized_visual, normalized_geometry, stats


def train_one_ranker(
    feature_name: str,
    data: dict[str, Any],
    cache: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    use_geometry = feature_name == "visual_rank_geometry"
    train_mask = data["split"] == 0
    calibration_mask = data["split"] == 1
    test_mask = data["split"] == 2
    visual, geometry, _stats = normalize_rank_inputs(data["visual_rank"], data["geometry"], train_mask)
    gate = data["gate_target"].float()
    labels = data["label"].long()
    positives = int(gate[train_mask].sum())
    negatives = int(train_mask.sum()) - positives
    if positives == 0:
        raise RuntimeError("No repairable training proposals")
    pos_weight = torch.tensor(negatives / max(positives, 1), dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed)
    if use_geometry:
        dataset = TensorDataset(visual[train_mask], geometry[train_mask], gate[train_mask], labels[train_mask])
    else:
        empty_geometry = torch.zeros((int(train_mask.sum()), 1), dtype=torch.float32)
        dataset = TensorDataset(visual[train_mask], empty_geometry, gate[train_mask], labels[train_mask])
    loader = DataLoader(
        dataset,
        batch_size=args.probe_batch_size,
        shuffle=True,
        generator=generator,
    )
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = HypothesisRanker(
        input_dim=int(visual.shape[-1]),
        hidden_dim=int(args.hidden_dim),
        geometry_dim=int(geometry.shape[-1]) if use_geometry else 0,
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gate_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    action_loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(args.epochs):
        for batch_visual, batch_geometry, batch_gate, batch_label in loader:
            batch_visual = batch_visual.to(device)
            batch_geometry = batch_geometry.to(device) if use_geometry else None
            batch_gate = batch_gate.to(device)
            batch_label = batch_label.to(device)
            logits = model(batch_visual, batch_geometry)
            no_action_logit = logits[:, 0]
            action_logits = logits[:, 1:]
            gate_logit = torch.logsumexp(action_logits, dim=1) - no_action_logit
            loss = gate_loss_fn(gate_logit, batch_gate)
            positive = batch_gate > 0.5
            if bool(positive.any()):
                action_target = batch_label[positive] - 1
                loss = loss + float(args.action_loss_weight) * action_loss_fn(
                    action_logits[positive],
                    action_target,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    score_chunks: list[torch.Tensor] = []
    hyp_chunks: list[torch.Tensor] = []
    eval_loader = DataLoader(
        TensorDataset(visual, geometry),
        batch_size=args.probe_batch_size,
        shuffle=False,
    )
    with torch.inference_mode():
        for batch_visual, batch_geometry in eval_loader:
            batch_visual = batch_visual.to(device)
            batch_geometry = batch_geometry.to(device) if use_geometry else None
            logits = model(batch_visual, batch_geometry)
            no_action_logit = logits[:, 0]
            action_logits = logits[:, 1:]
            gate_logit = torch.logsumexp(action_logits, dim=1) - no_action_logit
            score_chunks.append(
                torch.nan_to_num(torch.sigmoid(gate_logit), nan=0.0, posinf=1.0, neginf=0.0).cpu()
            )
            hyp_chunks.append(action_logits.argmax(dim=1).cpu())
    scores = torch.cat(score_chunks)
    hyp_ids = torch.cat(hyp_chunks)
    bank = _hypothesis_bank(args)
    coefficients = bank[hyp_ids].float()

    threshold, calibration_metrics = choose_threshold_from_scores(
        scores[calibration_mask],
        data["kind"][calibration_mask],
        args,
    )
    calibration_metrics["ap"] = average_precision(scores[calibration_mask], data["kind"][calibration_mask] == KIND_REPAIRABLE)
    calibration_metrics["auroc"] = auroc(scores[calibration_mask], data["kind"][calibration_mask] == KIND_REPAIRABLE)
    test_gate = gate_metrics(scores[test_mask], data["kind"][test_mask], threshold)
    exact = exact_system_evaluation(
        scores[test_mask],
        coefficients[test_mask],
        data,
        cache,
        test_mask,
        threshold,
        args,
    )
    label_accuracy = 0.0
    positive_test = test_mask & (data["kind"] == KIND_REPAIRABLE)
    if bool(positive_test.any()):
        label_accuracy = float(((hyp_ids[positive_test] + 1) == data["label"][positive_test]).float().mean())
    return {
        "feature": feature_name,
        "seed": seed,
        "input_shape": list(data["visual_rank"].shape[1:]),
        "geometry_dim": int(data["geometry"].shape[-1]) if use_geometry else 0,
        "train_positive": positives,
        "train_negative": negatives,
        "calibration": calibration_metrics,
        "test_gate": test_gate,
        "test_official": exact,
        "test_repairable_hypothesis_accuracy": label_accuracy,
    }


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
        data = extract_rank_features(cache, splits, args)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, feature_path)

    _refresh_split(data, cache, splits, args)

    runs: list[dict[str, Any]] = []
    for seed in args.probe_seeds:
        for feature_name in args.feature_kinds:
            run = train_one_ranker(feature_name, data, cache, int(seed), args)
            runs.append(run)
            official = run["test_official"]
            gate = run["test_gate"]
            print(
                f"{feature_name:>20} seed={seed}: "
                f"gateP/R={gate['precision']:.3f}/{gate['recall']:.3f} "
                f"tpAction={gate['tp_action_rate']:.4f} "
                f"hypAcc={run['test_repairable_hypothesis_accuracy']:.3f} "
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
