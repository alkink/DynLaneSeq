from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import ProposalRecallStats
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.analyze_attention_acquisition import (
    AssignedGeometryStats,
    LaneRecord,
    RecallStats,
    _best_iou_by_gt,
    _build_corridor_indicator,
    _build_records,
    _structured_forward_with_attention,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causally audit horizontal-position use and the key-to-state "
            "coordinate-transport interface of a frozen structured decoder. "
            "GT corridor interventions are diagnostic oracles, never benchmark "
            "predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--corridor-radius-px", type=float, default=16.0)
    parser.add_argument("--target-layer", type=int, default=3)
    parser.add_argument(
        "--layer-mode",
        choices=("single", "cascade"),
        default="cascade",
    )
    parser.add_argument("--bias-strength", type=float, default=2.0)
    parser.add_argument(
        "--feedback-scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help=(
            "L2 magnitude multipliers for the classifier-derived coordinate "
            "code added to each attended row state."
        ),
    )
    parser.add_argument(
        "--fusion-alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help=(
            "Mixture weights for directly fusing the selected attention-x "
            "distribution into the final row-coordinate distribution."
        ),
    )
    parser.add_argument(
        "--position-modes",
        nargs="+",
        choices=(
            "zero",
            "reverse",
            "shuffle",
            "roll_left",
            "roll_right",
            "half_scale",
        ),
        default=[
            "zero",
            "reverse",
            "shuffle",
            "roll_left",
            "roll_right",
        ],
    )
    parser.add_argument("--roll-columns", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--skip-position-sensitivity",
        action="store_true",
        help="Skip x-token perturbations when only direct fusion is required.",
    )
    parser.add_argument(
        "--skip-code-feedback",
        action="store_true",
        help="Skip classifier-code state feedback while retaining oracle bias.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


@dataclass
class AbsoluteDifferenceStats:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0

    def update(self, before: torch.Tensor, after: torch.Tensor) -> None:
        delta = (after.detach().float() - before.detach().float()).abs()
        finite = torch.isfinite(delta)
        if not bool(finite.any()):
            return
        delta = delta[finite]
        self.count += int(delta.numel())
        self.total += float(delta.sum())
        self.maximum = max(self.maximum, float(delta.max()))

    def summary(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_abs_change_px": self.total / max(self.count, 1),
            "max_abs_change_px": self.maximum,
        }


class ConditionStats:
    def __init__(self) -> None:
        self.raw_recall = ProposalRecallStats(thresholds=(0.5, 0.7))
        self.transitions = RecallStats()
        self.assigned_all = AssignedGeometryStats()
        self.assigned_hits = AssignedGeometryStats()
        self.assigned_misses = AssignedGeometryStats()
        self.output_change = AbsoluteDifferenceStats()

    def update(
        self,
        *,
        baseline_x: torch.Tensor,
        after_x: torch.Tensor,
        baseline_maps: list[dict[int, float]],
        after_maps: list[dict[int, float]],
        targets: list[dict[str, torch.Tensor]],
        records: list[LaneRecord],
        group_size: int,
        line_width: float,
    ) -> None:
        self.output_change.update(
            baseline_x[:, :group_size],
            after_x[:, :group_size],
        )
        for before_map, after_map in zip(baseline_maps, after_maps):
            for gt_index, before_iou in before_map.items():
                after_iou = after_map.get(gt_index, 0.0)
                self.raw_recall.update(after_iou)
                self.transitions.update(before_iou, after_iou)
        for record in records:
            kwargs = {
                "before_x": baseline_x,
                "after_x": after_x,
                "target": targets[record.image_index],
                "record": record,
                "line_width": float(line_width),
            }
            self.assigned_all.update(**kwargs)
            if record.final_iou >= 0.5:
                self.assigned_hits.update(**kwargs)
            else:
                self.assigned_misses.update(**kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            "raw_group0_proposal_recall": self.raw_recall.summary(),
            "best_proposal_transitions": self.transitions.summary(),
            "querywise_output_sensitivity": self.output_change.summary(),
            "assigned_geometry_all": self.assigned_all.summary(),
            "assigned_geometry_baseline_hits": self.assigned_hits.summary(),
            "assigned_geometry_baseline_misses": self.assigned_misses.summary(),
        }


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _position_variant(
    original: torch.Tensor,
    mode: str,
    *,
    roll_columns: int,
    seed: int,
) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(original)
    if mode == "reverse":
        return original.flip(0)
    if mode == "shuffle":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        order = torch.randperm(
            int(original.shape[0]),
            generator=generator,
        ).to(original.device)
        return original[order]
    if mode == "roll_left":
        return torch.roll(original, shifts=-int(roll_columns), dims=0)
    if mode == "roll_right":
        return torch.roll(original, shifts=int(roll_columns), dims=0)
    if mode == "half_scale":
        return original * 0.5
    raise ValueError(f"Unknown position mode: {mode}")


def _classifier_coordinate_codebook(
    *,
    row_classifier_weight: torch.Tensor,
    evidence_bins: int,
) -> torch.Tensor:
    """Map each evidence column to a unit classifier-recognizable x code."""

    output_bins, channels = row_classifier_weight.shape
    output_index = torch.floor(
        (
            torch.arange(
                evidence_bins,
                device=row_classifier_weight.device,
                dtype=torch.float32,
            )
            + 0.5
        )
        * float(output_bins)
        / float(evidence_bins)
    ).long()
    output_index.clamp_(0, int(output_bins) - 1)
    codebook = row_classifier_weight.detach().float()[output_index]
    # Remove the position-independent classifier component, then normalize
    # every position code to unit L2 norm.  The feedback scale is therefore
    # interpretable as an approximate residual-vector magnitude.
    codebook = codebook - codebook.mean(dim=0, keepdim=True)
    codebook = F.normalize(codebook, p=2.0, dim=-1, eps=1e-8)
    if tuple(codebook.shape) != (int(evidence_bins), int(channels)):
        raise RuntimeError("coordinate codebook construction failed")
    return codebook


def _embedding_audit(position_tokens: torch.Tensor) -> dict[str, float | int]:
    values = position_tokens.detach().float()
    normalized = F.normalize(values, p=2.0, dim=-1, eps=1e-8)
    adjacent = (normalized[:-1] * normalized[1:]).sum(dim=-1)
    half_shift = max(int(values.shape[0]) // 2, 1)
    distant = (
        normalized[:-half_shift] * normalized[half_shift:]
    ).sum(dim=-1)
    return {
        "bins": int(values.shape[0]),
        "dim": int(values.shape[1]),
        "component_std": float(values.std()),
        "mean_l2_norm": float(values.norm(dim=-1).mean()),
        "mean_adjacent_cosine": float(adjacent.mean()),
        "mean_half_width_cosine": float(distant.mean()),
        "mean_adjacent_l2_distance": float(
            (values[1:] - values[:-1]).norm(dim=-1).mean()
        ),
    }


def _condition_name(
    *,
    bias: bool,
    aligned: bool | None,
    scale: float = 0.0,
) -> str:
    prefix = "oracle_bias" if bias else "natural"
    if aligned is None:
        return f"{prefix}_only"
    direction = "aligned" if aligned else "reversed"
    scale_text = f"{float(scale):g}".replace(".", "p")
    return f"{prefix}_{direction}_feedback_{scale_text}"


def _attention_position_fusion(
    *,
    row_logits: torch.Tensor,
    attention: torch.Tensor,
    input_w: float,
    alpha: float,
    reverse_attention: bool = False,
) -> torch.Tensor:
    """Return expected x after direct model/attention probability fusion.

    This deliberately bypasses the row-state interface.  A positive oracle
    result means selected x information is useful when it is transferred; it
    does not define the final architecture.
    """

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("attention fusion alpha must be in [0, 1]")
    # Attention is [B, R, H, N, E]; row logits are [B, N, R, K].
    attention_probability = attention.detach().float().mean(dim=2)
    attention_probability = attention_probability.permute(0, 2, 1, 3)
    if reverse_attention:
        attention_probability = attention_probability.flip(-1)
    batch, instances, rows, evidence_bins = attention_probability.shape
    output_bins = int(row_logits.shape[-1])
    if tuple(row_logits.shape[:3]) != (batch, instances, rows):
        raise ValueError(
            "row-logit/attention shape mismatch: "
            f"{tuple(row_logits.shape)} vs {tuple(attention.shape)}"
        )
    if evidence_bins != output_bins:
        attention_probability = F.interpolate(
            attention_probability.reshape(
                batch * instances * rows,
                1,
                evidence_bins,
            ),
            size=output_bins,
            mode="linear",
            align_corners=False,
        ).view(batch, instances, rows, output_bins)
    attention_probability = attention_probability.clamp_min(0.0)
    attention_probability = attention_probability / attention_probability.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    model_probability = torch.softmax(row_logits.detach().float(), dim=-1)
    fused = (
        (1.0 - float(alpha)) * model_probability
        + float(alpha) * attention_probability
    )
    centers = (
        torch.arange(
            output_bins,
            device=fused.device,
            dtype=torch.float32,
        )
        + 0.5
    ) * float(input_w) / float(output_bins)
    return (fused * centers.view(1, 1, 1, -1)).sum(dim=-1)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg.setdefault("dataloader", {})["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    channels_last = (
        bool(cfg.get("training", {}).get("channels_last", False))
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    head = model.structured_query_head
    if head is None or head.x_tokens is None:
        raise ValueError(
            "Position-transport audit requires structured_query.use_x_pos=true"
        )

    layer_count = len(head.layers)
    target_layer_index = int(args.target_layer) - 1
    if not 0 <= target_layer_index < layer_count:
        raise ValueError(
            f"--target-layer must be in [1, {layer_count}], "
            f"got {args.target_layer}"
        )
    if args.layer_mode == "single":
        active_layers = (target_layer_index,)
    else:
        active_layers = tuple(range(target_layer_index, layer_count))
    if any(float(scale) <= 0.0 for scale in args.feedback_scales):
        raise ValueError("--feedback-scales must be positive")
    if any(
        not 0.0 < float(alpha) <= 1.0 for alpha in args.fusion_alphas
    ):
        raise ValueError("--fusion-alphas must be in (0, 1]")
    if int(args.roll_columns) <= 0:
        raise ValueError("--roll-columns must be positive")

    matcher = build_matcher(cfg)
    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    num_instances = int(head.num_instances)
    num_groups = int(head.num_groups)
    group_size = num_instances // max(num_groups, 1)
    input_w = float(head.input_w)

    original_position = head.x_tokens.weight.detach().clone()
    aligned_codebook = _classifier_coordinate_codebook(
        row_classifier_weight=head.row_x.weight,
        evidence_bins=int(head.evidence_x_bins),
    )
    reversed_codebook = aligned_codebook.flip(0)

    position_stats = (
        {}
        if bool(args.skip_position_sensitivity)
        else {mode: ConditionStats() for mode in args.position_modes}
    )
    transport_names = [_condition_name(bias=True, aligned=None)]
    if not bool(args.skip_code_feedback):
        transport_names.append(
            _condition_name(
                bias=False,
                aligned=True,
                scale=float(args.feedback_scales[-1]),
            )
        )
        for scale in args.feedback_scales:
            transport_names.extend(
                [
                    _condition_name(
                        bias=True,
                        aligned=True,
                        scale=float(scale),
                    ),
                    _condition_name(
                        bias=True,
                        aligned=False,
                        scale=float(scale),
                    ),
                ]
            )
    transport_stats = {
        name: ConditionStats() for name in transport_names
    }
    fusion_names: list[str] = []
    for alpha in args.fusion_alphas:
        alpha_text = f"{float(alpha):g}".replace(".", "p")
        fusion_names.extend(
            [
                f"natural_attention_fusion_{alpha_text}",
                f"oracle_bias_attention_fusion_{alpha_text}",
                f"oracle_bias_reversed_attention_fusion_{alpha_text}",
            ]
        )
    fusion_stats = {name: ConditionStats() for name in fusion_names}

    images_seen = 0
    lanes_without_group0_assignment = 0
    feature_component_rms_sum = 0.0
    feature_batches = 0
    total = len(loader)
    if int(args.max_batches) > 0:
        total = min(total, int(args.max_batches))
    progress = tqdm(
        enumerate(loader),
        total=total,
        desc="position-transport audit",
        ncols=100,
    )

    try:
        for batch_index, (images, targets, _metas) in progress:
            if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
                break
            images = images.to(
                device,
                non_blocking=True,
                memory_format=(
                    torch.channels_last
                    if channels_last
                    else torch.contiguous_format
                ),
            )
            targets = nested_to_device(targets, device)
            with _amp_context(device, amp_dtype):
                features = model.encoder.forward_features(
                    images,
                    inference_only=True,
                    structured_only=True,
                )["features"]
                head.x_tokens.weight.copy_(original_position)
                baseline, stages, baseline_attention = (
                    _structured_forward_with_attention(head, features)
                )
                row_values, _row_keys = head._row_features(features)
            feature_component_rms_sum += float(
                row_values.detach().float().square().mean().sqrt()
            )
            feature_batches += 1

            matches = matcher(baseline, targets)
            records = _build_records(
                stages=stages,
                targets=targets,
                matches=matches,
                group_size=group_size,
                line_width=float(args.line_width),
            )
            expected_lanes = sum(
                int((target["valid_mask"].sum(dim=1) >= 5).sum())
                for target in targets
            )
            lanes_without_group0_assignment += expected_lanes - len(records)
            baseline_maps = [
                _best_iou_by_gt(
                    baseline["pred_x_rows"][
                        image_index, :group_size
                    ].float(),
                    target,
                    line_width=float(args.line_width),
                )
                for image_index, target in enumerate(targets)
            ]

            if not bool(args.skip_position_sensitivity):
                for mode in args.position_modes:
                    head.x_tokens.weight.copy_(
                        _position_variant(
                            original_position,
                            mode,
                            roll_columns=int(args.roll_columns),
                            seed=int(args.seed),
                        )
                    )
                    with _amp_context(device, amp_dtype):
                        changed, _changed_stages, _changed_attention = (
                            _structured_forward_with_attention(head, features)
                        )
                    changed_maps = [
                        _best_iou_by_gt(
                            changed["pred_x_rows"][
                                image_index, :group_size
                            ].float(),
                            target,
                            line_width=float(args.line_width),
                        )
                        for image_index, target in enumerate(targets)
                    ]
                    position_stats[mode].update(
                        baseline_x=baseline["pred_x_rows"],
                        after_x=changed["pred_x_rows"],
                        baseline_maps=baseline_maps,
                        after_maps=changed_maps,
                        targets=targets,
                        records=records,
                        group_size=group_size,
                        line_width=float(args.line_width),
                    )
            head.x_tokens.weight.copy_(original_position)

            corridor = _build_corridor_indicator(
                targets=targets,
                matches=matches,
                batch=int(images.shape[0]),
                instances=num_instances,
                rows=int(head.num_rows),
                x_bins=int(head.evidence_x_bins),
                group_size=group_size,
                input_w=input_w,
                radius_px=float(args.corridor_radius_px),
                device=features.device,
                dtype=features.dtype,
            )
            bias_by_layer = {
                layer_index: corridor * float(args.bias_strength)
                for layer_index in active_layers
            }

            conditions: list[
                tuple[
                    str,
                    dict[int, torch.Tensor] | None,
                    dict[int, tuple[torch.Tensor, float]] | None,
                ]
            ] = [
                (
                    _condition_name(bias=True, aligned=None),
                    bias_by_layer,
                    None,
                )
            ]
            oracle_bias_output: dict[str, torch.Tensor] | None = None
            oracle_bias_attention: list[torch.Tensor] | None = None
            if not bool(args.skip_code_feedback):
                natural_scale = float(args.feedback_scales[-1])
                conditions.append(
                    (
                        _condition_name(
                            bias=False,
                            aligned=True,
                            scale=natural_scale,
                        ),
                        None,
                        {
                            layer_index: (
                                aligned_codebook,
                                natural_scale,
                            )
                            for layer_index in active_layers
                        },
                    )
                )
                for scale in args.feedback_scales:
                    scale = float(scale)
                    conditions.extend(
                        [
                            (
                                _condition_name(
                                    bias=True,
                                    aligned=True,
                                    scale=scale,
                                ),
                                bias_by_layer,
                                {
                                    layer_index: (
                                        aligned_codebook,
                                        scale,
                                    )
                                    for layer_index in active_layers
                                },
                            ),
                            (
                                _condition_name(
                                    bias=True,
                                    aligned=False,
                                    scale=scale,
                                ),
                                bias_by_layer,
                                {
                                    layer_index: (
                                        reversed_codebook,
                                        scale,
                                    )
                                    for layer_index in active_layers
                                },
                            ),
                        ]
                    )

            for name, condition_bias, condition_feedback in conditions:
                with _amp_context(device, amp_dtype):
                    changed, _changed_stages, changed_attention = (
                        _structured_forward_with_attention(
                            head,
                            features,
                            cross_bias_by_layer=condition_bias,
                            coordinate_feedback_by_layer=condition_feedback,
                        )
                    )
                if name == _condition_name(bias=True, aligned=None):
                    oracle_bias_output = changed
                    oracle_bias_attention = changed_attention
                changed_maps = [
                    _best_iou_by_gt(
                        changed["pred_x_rows"][
                            image_index, :group_size
                        ].float(),
                        target,
                        line_width=float(args.line_width),
                    )
                    for image_index, target in enumerate(targets)
                ]
                transport_stats[name].update(
                    baseline_x=baseline["pred_x_rows"],
                    after_x=changed["pred_x_rows"],
                    baseline_maps=baseline_maps,
                    after_maps=changed_maps,
                    targets=targets,
                    records=records,
                    group_size=group_size,
                    line_width=float(args.line_width),
                )
            if oracle_bias_output is None or oracle_bias_attention is None:
                raise RuntimeError("oracle-bias condition was not evaluated")

            attention_layer_index = int(active_layers[-1])
            fusion_sources = (
                (
                    "natural_attention",
                    baseline,
                    baseline_attention[attention_layer_index],
                    False,
                ),
                (
                    "oracle_bias_attention",
                    oracle_bias_output,
                    oracle_bias_attention[attention_layer_index],
                    False,
                ),
                (
                    "oracle_bias_reversed_attention",
                    oracle_bias_output,
                    oracle_bias_attention[attention_layer_index],
                    True,
                ),
            )
            for source_name, source_output, source_attention, reverse in (
                fusion_sources
            ):
                for alpha in args.fusion_alphas:
                    alpha = float(alpha)
                    alpha_text = f"{alpha:g}".replace(".", "p")
                    name = f"{source_name}_fusion_{alpha_text}"
                    fused_x = _attention_position_fusion(
                        row_logits=source_output["row_x_logits"],
                        attention=source_attention,
                        input_w=input_w,
                        alpha=alpha,
                        reverse_attention=reverse,
                    )
                    fused_maps = [
                        _best_iou_by_gt(
                            fused_x[image_index, :group_size],
                            target,
                            line_width=float(args.line_width),
                        )
                        for image_index, target in enumerate(targets)
                    ]
                    fusion_stats[name].update(
                        baseline_x=baseline["pred_x_rows"],
                        after_x=fused_x,
                        baseline_maps=baseline_maps,
                        after_maps=fused_maps,
                        targets=targets,
                        records=records,
                        group_size=group_size,
                        line_width=float(args.line_width),
                    )
            images_seen += int(images.shape[0])
    finally:
        head.x_tokens.weight.copy_(original_position)

    position_rms = float(
        original_position.detach().float().square().mean().sqrt()
    )
    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "oracle_warning": (
            "GT lanes and frozen-model Hungarian assignments define the soft "
            "corridor bias. Classifier-derived coordinate feedback is a causal "
            "mechanistic intervention, not a deployable prediction or benchmark "
            "result."
        ),
        "interpretation": {
            "position_sensitivity": (
                "Large output/recall changes after zeroing, reversing, shuffling, "
                "or rolling x embeddings prove that key-side position is used. "
                "Small changes would support positional-key neglect. Neither "
                "outcome alone proves that selected x is adequately transported "
                "to the row state."
            ),
            "coordinate_transport": (
                "Compare oracle_bias_only with aligned and reversed feedback at "
                "the same scale. A consistent aligned-over-reversed advantage "
                "shows that attention can benefit from explicitly carrying the "
                "selected x identity. No advantage rejects this particular "
                "classifier-code intervention, but cannot prove every jointly "
                "trained reference-x design ineffective."
            ),
        },
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "num_instances": num_instances,
        "num_groups": num_groups,
        "group_size": group_size,
        "decoder_layers": layer_count,
        "target_layer": int(args.target_layer),
        "layer_mode": args.layer_mode,
        "active_zero_based_layers": list(active_layers),
        "corridor_radius_px": float(args.corridor_radius_px),
        "bias_strength": float(args.bias_strength),
        "feedback_scales": [float(value) for value in args.feedback_scales],
        "fusion_alphas": [float(value) for value in args.fusion_alphas],
        "roll_columns": int(args.roll_columns),
        "roll_pixels": (
            float(args.roll_columns)
            * input_w
            / float(head.evidence_x_bins)
        ),
        "lanes_without_group0_assignment": lanes_without_group0_assignment,
        "position_embedding_audit": _embedding_audit(original_position),
        "position_to_feature_rms": {
            "position_component_rms": position_rms,
            "projected_feature_component_rms": (
                feature_component_rms_sum / max(feature_batches, 1)
            ),
            "position_over_feature_rms": position_rms
            / max(feature_component_rms_sum / max(feature_batches, 1), 1e-12),
        },
        "position_key_sensitivity": {
            name: stats.summary() for name, stats in position_stats.items()
        },
        "coordinate_transport_interventions": {
            name: stats.summary() for name, stats in transport_stats.items()
        },
        "direct_attention_position_fusion": {
            name: stats.summary() for name, stats in fusion_stats.items()
        },
    }
    print(json.dumps(payload, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
