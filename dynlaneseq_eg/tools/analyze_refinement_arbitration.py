from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import line_iou_against_gt
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.analyze_attention_acquisition import (
    RecallStats,
    _best_iou_by_gt,
    _build_corridor_indicator,
    _build_records,
    _structured_forward_with_attention,
)
from dynlaneseq_eg.tools.analyze_position_transport import (
    _attention_position_fusion,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether deployable uncertainty signals can safely arbitrate "
            "small attention-to-coordinate corrections. Natural attention and "
            "GT-biased oracle attention are reported separately."
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=32)
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
        "--fusion-alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.025, 0.05, 0.1],
    )
    parser.add_argument(
        "--gate-budgets",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Number of most-uncertain candidates refined per image.",
    )
    parser.add_argument(
        "--eligible-top-k",
        type=int,
        default=4,
        help="Only the highest model-score candidates enter deployable gates.",
    )
    parser.add_argument("--quality-power", type=float, default=0.5)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@dataclass
class GateStats:
    transitions: RecallStats
    images: int = 0
    modified_candidates: int = 0

    @classmethod
    def create(cls) -> "GateStats":
        return cls(transitions=RecallStats())

    def update(
        self,
        *,
        before_maps: list[dict[int, float]],
        after_maps: list[dict[int, float]],
        gate: torch.Tensor,
    ) -> None:
        self.images += int(gate.shape[0])
        self.modified_candidates += int(gate.sum())
        for before_map, after_map in zip(before_maps, after_maps):
            for gt_index, before_iou in before_map.items():
                self.transitions.update(
                    before_iou,
                    after_map.get(gt_index, 0.0),
                )

    def summary(self) -> dict[str, Any]:
        payload = self.transitions.summary()
        payload["images"] = self.images
        payload["modified_candidates"] = self.modified_candidates
        payload["mean_modified_candidates_per_image"] = (
            self.modified_candidates / max(self.images, 1)
        )
        return payload


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=values.dtype)
    return (values * mask_float).sum(dim=-1) / mask_float.sum(
        dim=-1
    ).clamp_min(1.0)


def _candidate_uncertainty_signals(
    *,
    outputs: dict[str, torch.Tensor],
    attention: torch.Tensor,
    input_w: float,
    quality_power: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return deployable lane-level signals, all oriented high=uncertain."""

    row_logits = outputs["row_x_logits"].detach().float()
    batch, instances, rows, output_bins = row_logits.shape
    row_probability = torch.softmax(row_logits, dim=-1)
    row_entropy = -(
        row_probability
        * row_probability.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(float(output_bins))
    row_peak = row_probability.max(dim=-1).values

    # [B,R,H,N,E] -> [B,N,R,E]
    attention_probability = attention.detach().float().mean(dim=2)
    attention_probability = attention_probability.permute(0, 2, 1, 3)
    evidence_bins = int(attention_probability.shape[-1])
    attention_probability = attention_probability / attention_probability.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    attention_entropy = -(
        attention_probability
        * attention_probability.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(float(evidence_bins))
    attention_peak = attention_probability.max(dim=-1).values
    evidence_centers = (
        torch.arange(
            evidence_bins,
            device=row_logits.device,
            dtype=torch.float32,
        )
        + 0.5
    ) * float(input_w) / float(evidence_bins)
    attention_x = (
        attention_probability
        * evidence_centers.view(1, 1, 1, evidence_bins)
    ).sum(dim=-1)

    range_norm = outputs["range_norm"].detach().float()
    row_coordinate = (
        torch.arange(rows, device=row_logits.device, dtype=torch.float32)
        + 0.5
    ) / float(rows)
    visible = (
        row_coordinate.view(1, 1, rows) >= range_norm[..., 0, None]
    ) & (
        row_coordinate.view(1, 1, rows) <= range_norm[..., 1, None]
    )
    empty = ~visible.any(dim=-1)
    if bool(empty.any()):
        visible = torch.where(
            empty[..., None],
            torch.ones_like(visible),
            visible,
        )

    exist_probability = torch.softmax(
        outputs["exist_logits"].detach().float(),
        dim=-1,
    )[..., 0]
    quality_probability = torch.sigmoid(
        outputs["quality_logits"].detach().float()
    )
    model_score = exist_probability * quality_probability.clamp_min(
        1e-8
    ).pow(float(quality_power))
    disagreement_px = _masked_mean(
        (
            attention_x
            - outputs["pred_x_rows"].detach().float()
        ).abs(),
        visible,
    )
    normalized_disagreement = (
        disagreement_px / 64.0
    ).clamp(max=1.0)

    signals = {
        "low_quality": 1.0 - quality_probability,
        "low_model_score": 1.0 - model_score,
        "row_entropy": _masked_mean(row_entropy, visible),
        "attention_entropy": _masked_mean(
            attention_entropy,
            visible,
        ),
        "attention_disagreement": normalized_disagreement,
        "low_attention_peak": 1.0
        - _masked_mean(attention_peak, visible),
        "quality_disagreement": 0.5 * (1.0 - quality_probability)
        + 0.5 * normalized_disagreement,
        "row_disagreement": 0.5
        * _masked_mean(row_entropy, visible)
        + 0.5 * normalized_disagreement,
        "low_row_peak": 1.0 - _masked_mean(row_peak, visible),
    }
    expected_shape = (batch, instances)
    for name, value in signals.items():
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"signal {name} has shape {tuple(value.shape)}, "
                f"expected {expected_shape}"
            )
    return signals, model_score


def _uncertainty_gate(
    uncertainty: torch.Tensor,
    model_score: torch.Tensor,
    *,
    eligible_top_k: int,
    budget: int,
) -> torch.Tensor:
    if tuple(uncertainty.shape) != tuple(model_score.shape):
        raise ValueError("uncertainty/model-score shape mismatch")
    instances = int(uncertainty.shape[-1])
    eligible_top_k = min(max(int(eligible_top_k), 1), instances)
    budget = min(max(int(budget), 1), eligible_top_k)
    eligible = model_score.topk(eligible_top_k, dim=-1).indices
    eligible_uncertainty = uncertainty.gather(1, eligible)
    selected_within = eligible_uncertainty.topk(budget, dim=-1).indices
    selected = eligible.gather(1, selected_within)
    gate = torch.zeros_like(uncertainty, dtype=torch.bool)
    gate.scatter_(1, selected, True)
    return gate


def _assigned_iou(
    x_rows: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    gt_index: int,
    pred_index: int,
    line_width: float,
) -> float:
    gt_x = target["x_rows"][gt_index].to(
        device=x_rows.device,
        dtype=x_rows.dtype,
    )
    valid = target["valid_mask"][gt_index].to(
        device=x_rows.device,
    ).bool()
    value = line_iou_against_gt(
        x_rows[pred_index].unsqueeze(0),
        gt_x,
        valid,
        line_width=float(line_width),
    )
    return float(value[0]) if value.numel() else 0.0


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / float(len(positives) * len(negatives))


def _binary_signal_summary(
    scores: list[float],
    labels: list[int],
) -> dict[str, float | int | None]:
    if len(scores) != len(labels):
        raise ValueError("score/label length mismatch")
    count = len(labels)
    positives = int(sum(labels))
    order = sorted(range(count), key=lambda index: scores[index], reverse=True)
    top_count = max(1, int(math.ceil(0.25 * count))) if count else 0
    top_positives = (
        sum(labels[index] for index in order[:top_count])
        if top_count
        else 0
    )
    prevalence = positives / max(count, 1)
    top_precision = top_positives / max(top_count, 1)
    return {
        "samples": count,
        "positives": positives,
        "prevalence": prevalence,
        "roc_auc": _roc_auc(scores, labels),
        "top_quartile_precision": top_precision,
        "top_quartile_lift": top_precision / max(prevalence, 1e-12),
    }


def _condition_key(
    source: str,
    alpha: float,
    signal: str,
    budget: int,
) -> str:
    alpha_text = f"{float(alpha):g}".replace(".", "p")
    return (
        f"{source}_alpha{alpha_text}_{signal}_budget{int(budget)}"
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if any(
        not 0.0 < float(alpha) <= 1.0 for alpha in args.fusion_alphas
    ):
        raise ValueError("--fusion-alphas must be in (0, 1]")
    if any(int(value) <= 0 for value in args.gate_budgets):
        raise ValueError("--gate-budgets must be positive")

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
    if head is None:
        raise ValueError("refinement arbitration requires structured_query")

    layer_count = len(head.layers)
    target_layer_index = int(args.target_layer) - 1
    if not 0 <= target_layer_index < layer_count:
        raise ValueError("--target-layer is outside the decoder")
    if args.layer_mode == "single":
        active_layers = (target_layer_index,)
    else:
        active_layers = tuple(range(target_layer_index, layer_count))
    attention_layer_index = int(active_layers[-1])

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
    group_size = num_instances // max(int(head.num_groups), 1)
    input_w = float(head.input_w)
    eligible_top_k = min(int(args.eligible_top_k), group_size)
    budgets = sorted(
        {
            min(int(value), eligible_top_k)
            for value in args.gate_budgets
        }
    )

    signal_names = (
        "low_quality",
        "low_model_score",
        "row_entropy",
        "attention_entropy",
        "attention_disagreement",
        "low_attention_peak",
        "quality_disagreement",
        "row_disagreement",
        "low_row_peak",
    )
    gate_stats: dict[str, GateStats] = {}
    gt_miss_gate_stats: dict[str, GateStats] = {}
    signal_scores: dict[str, list[float]] = {
        name: [] for name in signal_names
    }
    labels: dict[str, list[int]] = {
        "raw_miss_050": [],
        "raw_miss_070": [],
        "assigned_bad_050": [],
        "assigned_bad_070": [],
    }
    benefit_labels: dict[str, list[int]] = {}
    images_seen = 0
    lanes_without_group0_assignment = 0

    total = len(loader)
    if int(args.max_batches) > 0:
        total = min(total, int(args.max_batches))
    progress = tqdm(
        enumerate(loader),
        total=total,
        desc="refinement arbitration audit",
        ncols=100,
    )
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
            baseline, stages, baseline_attention = (
                _structured_forward_with_attention(head, features)
            )
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
                baseline["pred_x_rows"][image_index, :group_size].float(),
                target,
                line_width=float(args.line_width),
            )
            for image_index, target in enumerate(targets)
        ]

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
        with _amp_context(device, amp_dtype):
            oracle_output, _oracle_stages, oracle_attention = (
                _structured_forward_with_attention(
                    head,
                    features,
                    cross_bias_by_layer=bias_by_layer,
                )
            )

        signals, model_score = _candidate_uncertainty_signals(
            outputs=baseline,
            attention=baseline_attention[attention_layer_index],
            input_w=input_w,
            quality_power=float(args.quality_power),
        )
        signals = {
            name: value[:, :group_size] for name, value in signals.items()
        }
        model_score = model_score[:, :group_size]

        fused_by_source: dict[str, dict[float, torch.Tensor]] = {
            "natural": {},
            "oracle_bias": {},
        }
        for alpha in args.fusion_alphas:
            alpha = float(alpha)
            fused_by_source["natural"][alpha] = _attention_position_fusion(
                row_logits=baseline["row_x_logits"],
                attention=baseline_attention[attention_layer_index],
                input_w=input_w,
                alpha=alpha,
            )[:, :group_size]
            fused_by_source["oracle_bias"][alpha] = (
                _attention_position_fusion(
                    row_logits=oracle_output["row_x_logits"],
                    attention=oracle_attention[attention_layer_index],
                    input_w=input_w,
                    alpha=alpha,
                )[:, :group_size]
            )

        baseline_group_x = baseline["pred_x_rows"][:, :group_size].float()
        for source, fused_by_alpha in fused_by_source.items():
            for alpha, fused_x in fused_by_alpha.items():
                # Non-deployable upper bound: alter only the Hungarian query
                # corresponding to a baseline raw miss.
                gt_gate = torch.zeros(
                    (int(images.shape[0]), group_size),
                    device=device,
                    dtype=torch.bool,
                )
                for record in records:
                    if record.final_iou < 0.5:
                        gt_gate[record.image_index, record.pred_index] = True
                gt_changed = torch.where(
                    gt_gate[..., None],
                    fused_x,
                    baseline_group_x,
                )
                gt_after_maps = [
                    _best_iou_by_gt(
                        gt_changed[image_index],
                        target,
                        line_width=float(args.line_width),
                    )
                    for image_index, target in enumerate(targets)
                ]
                gt_key = f"{source}_alpha{alpha:g}".replace(".", "p")
                gt_stats = gt_miss_gate_stats.setdefault(
                    gt_key,
                    GateStats.create(),
                )
                gt_stats.update(
                    before_maps=baseline_maps,
                    after_maps=gt_after_maps,
                    gate=gt_gate,
                )

                for signal_name, uncertainty in signals.items():
                    for budget in budgets:
                        gate = _uncertainty_gate(
                            uncertainty,
                            model_score,
                            eligible_top_k=eligible_top_k,
                            budget=budget,
                        )
                        changed_x = torch.where(
                            gate[..., None],
                            fused_x,
                            baseline_group_x,
                        )
                        after_maps = [
                            _best_iou_by_gt(
                                changed_x[image_index],
                                target,
                                line_width=float(args.line_width),
                            )
                            for image_index, target in enumerate(targets)
                        ]
                        key = _condition_key(
                            source,
                            alpha,
                            signal_name,
                            budget,
                        )
                        stats = gate_stats.setdefault(
                            key,
                            GateStats.create(),
                        )
                        stats.update(
                            before_maps=baseline_maps,
                            after_maps=after_maps,
                            gate=gate,
                        )

        for record in records:
            target = targets[record.image_index]
            before_assigned = _assigned_iou(
                baseline_group_x[record.image_index],
                target,
                gt_index=record.gt_index,
                pred_index=record.pred_index,
                line_width=float(args.line_width),
            )
            for signal_name in signal_names:
                signal_scores[signal_name].append(
                    float(
                        signals[signal_name][
                            record.image_index,
                            record.pred_index,
                        ]
                    )
                )
            labels["raw_miss_050"].append(int(record.final_iou < 0.5))
            labels["raw_miss_070"].append(int(record.final_iou < 0.7))
            labels["assigned_bad_050"].append(
                int(before_assigned < 0.5)
            )
            labels["assigned_bad_070"].append(
                int(before_assigned < 0.7)
            )
            for source, fused_by_alpha in fused_by_source.items():
                for alpha, fused_x in fused_by_alpha.items():
                    after_assigned = _assigned_iou(
                        fused_x[record.image_index],
                        target,
                        gt_index=record.gt_index,
                        pred_index=record.pred_index,
                        line_width=float(args.line_width),
                    )
                    benefit_key = (
                        f"{source}_alpha{alpha:g}_benefit_002"
                    ).replace(".", "p")
                    benefit_labels.setdefault(benefit_key, []).append(
                        int(after_assigned >= before_assigned + 0.02)
                    )
        images_seen += int(images.shape[0])

    signal_separation: dict[str, dict[str, Any]] = {}
    for signal_name, scores in signal_scores.items():
        signal_separation[signal_name] = {
            label_name: _binary_signal_summary(scores, values)
            for label_name, values in labels.items()
        }
        signal_separation[signal_name]["fusion_benefit"] = {
            label_name: _binary_signal_summary(scores, values)
            for label_name, values in benefit_labels.items()
        }

    gate_payload = {
        key: stats.summary() for key, stats in gate_stats.items()
    }
    natural_conditions = {
        key: value
        for key, value in gate_payload.items()
        if key.startswith("natural_")
    }
    oracle_conditions = {
        key: value
        for key, value in gate_payload.items()
        if key.startswith("oracle_bias_")
    }

    def best_condition(
        conditions: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not conditions:
            return None
        key, value = max(
            conditions.items(),
            key=lambda item: (
                float(item[1]["net_gain_050_points"]),
                float(item[1]["net_gain_070_points"]),
                float(item[1]["mean_best_iou_delta"]),
            ),
        )
        return {"condition": key, **value}

    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "oracle_warning": (
            "The oracle-bias and GT-miss gates use GT assignments and are upper "
            "bounds only. Natural-attention gates use deployable model signals, "
            "but this validation diagnostic is not a benchmark result."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "lanes_without_group0_assignment": lanes_without_group0_assignment,
        "group_size": group_size,
        "eligible_top_k": eligible_top_k,
        "gate_budgets": budgets,
        "fusion_alphas": [float(value) for value in args.fusion_alphas],
        "quality_power": float(args.quality_power),
        "target_layer": int(args.target_layer),
        "layer_mode": args.layer_mode,
        "signal_orientation": (
            "Every signal is oriented so larger values mean more uncertain."
        ),
        "signal_separation": signal_separation,
        "deployable_natural_gated_fusion": natural_conditions,
        "oracle_attention_gated_fusion": oracle_conditions,
        "gt_miss_gate_upper_bound": {
            key: stats.summary()
            for key, stats in gt_miss_gate_stats.items()
        },
        "decision_summary": {
            "best_deployable_natural_gate": best_condition(
                natural_conditions
            ),
            "best_oracle_attention_gate": best_condition(
                oracle_conditions
            ),
            "interpretation_rule": (
                "A positive natural gate with limited losses supports existing "
                "signals as refinement arbitration. Oracle-only gains implicate "
                "natural curve acquisition. If even oracle/GT-miss gates have "
                "little net gain, the tested attention-coordinate correction "
                "lacks sufficient proposal value."
            ),
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
