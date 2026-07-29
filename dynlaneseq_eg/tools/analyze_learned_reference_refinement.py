from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, nested_to_device
from dynlaneseq_eg.tools.analyze_noisy_reference_counterfactual import (
    _counterfactual_p2,
    _load_p2_probe,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_gt_curve_aligned_features import (
    CurveAlignedSequenceProbe,
    _selected_row_indices,
)
from dynlaneseq_eg.tools.probe_query_conditioned_dense_curve import (
    ProbeComparison,
    QueryConditionedDenseCurveProbe,
    _best_iou_per_gt,
    _group_zero_matches,
    _paired_iou,
    probe_row_states,
    seed_everything,
)
from dynlaneseq_eg.tools.probe_reference_guided_p2_update import (
    _sample_p2_profiles,
)


REFINEMENT_CONDITIONS = (
    "dense_reference",
    "refined_correct_p2",
    "refined_wrong_image_p2",
    "refined_zero_p2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect a frozen learned dense reference probe to the independently "
            "validated curve-aligned P2 refiner. This tests whether learned "
            "references enter the refiner's visual capture range."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dense-probe-checkpoint", required=True)
    parser.add_argument("--p2-probe-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=16)
    parser.add_argument(
        "--eval-sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--dense-hidden-dim", type=int, default=64)
    parser.add_argument("--dense-evidence-width", type=int, default=400)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _predicted_row_mask(
    range_norm: torch.Tensor,
    *,
    num_rows: int,
    input_h: int,
) -> torch.Tensor:
    if range_norm.ndim != 3 or int(range_norm.shape[-1]) != 2:
        raise ValueError("range_norm must have shape [B,N,2]")
    y = fixed_y_rows(
        int(num_rows),
        int(input_h),
        device=range_norm.device,
        dtype=torch.float32,
    )
    y_norm = y / max(float(input_h), 1.0)
    low = range_norm[..., 0].float().unsqueeze(-1)
    high = range_norm[..., 1].float().unsqueeze(-1)
    mask = (y_norm.view(1, 1, num_rows) >= low) & (
        y_norm.view(1, 1, num_rows) <= high
    )
    # The sequence probe needs several rows to form context. Fall back to all
    # rows only for degenerate range predictions.
    degenerate = mask.sum(dim=-1) < 5
    if bool(degenerate.any()):
        mask = torch.where(degenerate.unsqueeze(-1), torch.ones_like(mask), mask)
    return mask


def _expand_selected_residual(
    residual: torch.Tensor,
    *,
    output_rows: int,
) -> torch.Tensor:
    if residual.ndim != 3:
        raise ValueError("residual must have shape [B,N,R]")
    batch, instances, rows = residual.shape
    expanded = F.interpolate(
        residual.reshape(batch * instances, 1, rows).float(),
        size=int(output_rows),
        mode="linear",
        align_corners=True,
    )
    return expanded.reshape(batch, instances, int(output_rows))


def _refine_references(
    *,
    probe: CurveAlignedSequenceProbe,
    p2: torch.Tensor,
    references: torch.Tensor,
    valid_rows: torch.Tensor,
    row_indices: torch.Tensor,
    offsets: torch.Tensor,
    input_w: int,
    input_h: int,
) -> torch.Tensor:
    selected_reference = references[:, :, row_indices]
    profiles = _sample_p2_profiles(
        p2,
        selected_reference,
        offsets,
        input_w=int(input_w),
        input_h=int(input_h),
    )
    batch, instances, rows, offset_count, channels = profiles.shape
    profiles = F.normalize(profiles.float(), p=2.0, dim=-1, eps=1e-6)
    stacked = torch.zeros(
        batch * instances,
        rows,
        offset_count,
        3,
        channels,
        device=profiles.device,
        dtype=profiles.dtype,
    )
    stacked[:, :, :, 0] = profiles.reshape(
        batch * instances,
        rows,
        offset_count,
        channels,
    )
    scale_mask = torch.zeros(
        batch * instances,
        3,
        device=profiles.device,
        dtype=torch.bool,
    )
    scale_mask[:, 0] = True
    selected_valid = valid_rows[:, :, row_indices].reshape(
        batch * instances,
        rows,
    )
    residual = probe(
        stacked,
        scale_mask,
        selected_valid,
    )["residual"].reshape(batch, instances, rows)
    expanded = _expand_selected_residual(
        residual,
        output_rows=int(references.shape[-1]),
    )
    return (references.float() + expanded).clamp(0.0, float(input_w - 1))


def _load_dense_probe(
    *,
    path: str,
    in_dim: int,
    state_dim: int,
    hidden_dim: int,
    num_rows: int,
    evidence_width: int,
    input_w: int,
    device: torch.device,
) -> tuple[QueryConditionedDenseCurveProbe, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    probe = QueryConditionedDenseCurveProbe(
        in_dim=int(in_dim),
        state_dim=int(state_dim),
        hidden_dim=int(hidden_dim),
        num_rows=int(num_rows),
        evidence_width=int(evidence_width),
        input_w=int(input_w),
    )
    probe.load_state_dict(payload.get("probe", payload), strict=True)
    return probe.to(device).eval(), metadata


@torch.inference_mode()
def _evaluate(
    *,
    model: nn.Module,
    dense_probe: QueryConditionedDenseCurveProbe,
    p2_probe: CurveAlignedSequenceProbe,
    matcher,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    group_size: int,
    line_width: float,
    row_indices: torch.Tensor,
    offsets: torch.Tensor,
    input_w: int,
    input_h: int,
    state_source: str,
) -> tuple[dict[str, Any], int]:
    comparisons = {
        condition: ProbeComparison() for condition in REFINEMENT_CONDITIONS
    }
    images_seen = 0
    for images, targets, _metas in tqdm(
        loader,
        desc="learned-reference -> P2 refiner",
        ncols=100,
    ):
        images = images.to(device)
        targets = nested_to_device(targets, device)
        with _amp_context(device, amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
            outputs = model.structured_query_head(
                encoded["features"],
                inference_only=False,
            )
        matches = matcher(outputs, targets)
        p2 = encoded["features"].float()
        row_states = probe_row_states(
            model.structured_query_head,
            outputs,
            source=state_source,
            group_size=group_size,
            batch_size=int(images.shape[0]),
        )
        dense_logits = dense_probe(p2, row_states)
        dense_reference = dense_probe.decode(dense_logits)
        valid_rows = _predicted_row_mask(
            outputs["range_norm"][:, :group_size],
            num_rows=int(dense_reference.shape[-1]),
            input_h=int(input_h),
        )
        p2_controls = _counterfactual_p2(p2)
        candidates = {
            "dense_reference": dense_reference,
            "refined_correct_p2": _refine_references(
                probe=p2_probe,
                p2=p2_controls["correct_p2"],
                references=dense_reference,
                valid_rows=valid_rows,
                row_indices=row_indices,
                offsets=offsets,
                input_w=input_w,
                input_h=input_h,
            ),
            "refined_wrong_image_p2": _refine_references(
                probe=p2_probe,
                p2=p2_controls["wrong_image_p2"],
                references=dense_reference,
                valid_rows=valid_rows,
                row_indices=row_indices,
                offsets=offsets,
                input_w=input_w,
                input_h=input_h,
            ),
            "refined_zero_p2": _refine_references(
                probe=p2_probe,
                p2=p2_controls["zero_p2"],
                references=dense_reference,
                valid_rows=valid_rows,
                row_indices=row_indices,
                offsets=offsets,
                input_w=input_w,
                input_h=input_h,
            ),
        }
        base = outputs["pred_x_rows"][:, :group_size].float()
        for batch_index, (target, match) in enumerate(zip(targets, matches)):
            pairs = _group_zero_matches(match, group_size=group_size)
            base_best = _best_iou_per_gt(
                base[batch_index],
                target,
                line_width=float(line_width),
            )
            base_paired = _paired_iou(
                base[batch_index],
                target,
                pairs,
                line_width=float(line_width),
            )
            for condition, curves in candidates.items():
                candidate_best = _best_iou_per_gt(
                    curves[batch_index],
                    target,
                    line_width=float(line_width),
                )
                candidate_paired = _paired_iou(
                    curves[batch_index],
                    target,
                    pairs,
                    line_width=float(line_width),
                )
                comparisons[condition].update(
                    base_best,
                    candidate_best,
                    base_paired,
                    candidate_paired,
                )
        images_seen += int(images.shape[0])

    summaries = {
        condition: comparison.summary()
        for condition, comparison in comparisons.items()
    }
    correct = summaries["refined_correct_p2"]
    dense = summaries["dense_reference"]
    wrong = summaries["refined_wrong_image_p2"]
    zero = summaries["refined_zero_p2"]
    union_increment = float(correct["union_gain_050_points"])
    control_union = max(
        float(wrong["union_gain_050_points"]),
        float(zero["union_gain_050_points"]),
    )
    return {
        "conditions": summaries,
        "decision": {
            "positive_gate": bool(
                union_increment >= 2.0
                and union_increment >= control_union + 1.0
                and int(correct["recovered_base_misses_050"])
                > int(dense["recovered_base_misses_050"])
            ),
            "correct_refined_union_gain_050_points": union_increment,
            "largest_refiner_control_union_gain_050_points": control_union,
            "dense_reference_recovered_base_misses_050": int(
                dense["recovered_base_misses_050"]
            ),
            "correct_refined_recovered_base_misses_050": int(
                correct["recovered_base_misses_050"]
            ),
            "gate_definition": (
                "Refined learned references must add >=2 recall@0.50 points "
                "over the base, exceed wrong/zero-P2 refinement by >=1 point, "
                "and recover more base misses than the dense reference alone."
            ),
        },
    }, images_seen


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.requires_grad_(False)
    model = model.to(device).eval()
    matcher = build_matcher(cfg)
    head = model.structured_query_head
    if head is None:
        raise ValueError("learned-reference analysis requires structured queries")
    group_size = int(head.num_instances) // max(int(head.num_groups), 1)
    input_w = int(model_cfg.get("input_w", head.input_w))
    input_h = int(model_cfg.get("input_h", 288))
    dim = int(model_cfg.get("dim", head.dim))
    dense_probe, dense_metadata = _load_dense_probe(
        path=args.dense_probe_checkpoint,
        in_dim=dim,
        state_dim=dim,
        hidden_dim=int(args.dense_hidden_dim),
        num_rows=int(head.num_rows),
        evidence_width=int(args.dense_evidence_width),
        input_w=input_w,
        device=device,
    )
    common_channels = max(
        int(model.encoder.backbone.out_channels["c2"]),
        int(model_cfg.get("fpn_channels", dim)),
        dim,
    )
    p2_probe, p2_args, p2_parameter_count = _load_p2_probe(
        path=args.p2_probe_checkpoint,
        common_channels=common_channels,
        device=device,
    )
    offsets = torch.tensor(
        p2_args["offsets_px"],
        device=device,
        dtype=torch.float32,
    )
    row_indices = _selected_row_indices(
        int(head.num_rows),
        int(p2_args["probe_rows"]),
        device,
    )
    state_source = str(dense_metadata.get("state_source", "final"))

    loader = build_dataloader(cfg, split="val", training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=str(args.eval_sample_strategy),
        max_batches=int(args.eval_max_batches),
        num_workers=int(args.num_workers),
    )
    evaluation, images_seen = _evaluate(
        model=model,
        dense_probe=dense_probe,
        p2_probe=p2_probe,
        matcher=matcher,
        loader=loader,
        device=device,
        amp_dtype=amp_dtype,
        group_size=group_size,
        line_width=float(args.line_width),
        row_indices=row_indices,
        offsets=offsets,
        input_w=input_w,
        input_h=input_h,
        state_source=state_source,
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "The detector and both probes are frozen. This staged composition "
            "tests capture range; it is not a jointly trained final model."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "dense_probe_checkpoint": args.dense_probe_checkpoint,
        "dense_probe_metadata": dense_metadata,
        "p2_probe_checkpoint": args.p2_probe_checkpoint,
        "p2_probe_parameter_count": int(p2_parameter_count),
        "images": int(images_seen),
        "sample_strategy": str(args.eval_sample_strategy),
        "sampled_dataset_indices": sampled_indices,
        "evaluation": evaluation,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    compact = dict(payload)
    compact.pop("sampled_dataset_indices")
    print(json.dumps(compact, indent=2))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
