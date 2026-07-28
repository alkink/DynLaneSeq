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
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.analyze_attention_acquisition import (
    RecallStats,
    _best_iou_by_gt,
    _build_corridor_indicator,
    _structured_forward_with_attention,
)
from dynlaneseq_eg.tools.analyze_position_transport import (
    _attention_position_fusion,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether attention-derived curves add raw proposal coverage "
            "when retained as alternatives instead of replacing established "
            "LaneRowNet proposals."
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
        default=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
    parser.add_argument(
        "--active-top-k",
        type=int,
        default=4,
        help=(
            "Number of high-score proposals retained and used to generate "
            "alternatives in the fixed-candidate parity condition."
        ),
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
class AlternativeStats:
    transitions: RecallStats
    images: int = 0
    candidate_total: int = 0

    @classmethod
    def create(cls) -> "AlternativeStats":
        return cls(transitions=RecallStats())

    def update(
        self,
        *,
        before_maps: list[dict[int, float]],
        after_maps: list[dict[int, float]],
        candidates_per_image: int,
    ) -> None:
        self.images += len(before_maps)
        self.candidate_total += len(before_maps) * int(candidates_per_image)
        for before_map, after_map in zip(before_maps, after_maps):
            for gt_index, before_iou in before_map.items():
                self.transitions.update(
                    before_iou,
                    after_map.get(gt_index, 0.0),
                )

    def summary(self) -> dict[str, Any]:
        payload = self.transitions.summary()
        payload["images"] = self.images
        payload["candidates_per_image"] = (
            self.candidate_total / max(self.images, 1)
        )
        return payload


def _attention_curves(
    attention: torch.Tensor,
    *,
    input_w: float,
) -> dict[str, torch.Tensor]:
    """Convert [B,R,H,N,E] attention into alternative [B,N,R] curves."""

    probability = attention.detach().float().mean(dim=2)
    probability = probability.permute(0, 2, 1, 3)
    evidence_bins = int(probability.shape[-1])
    probability = probability / probability.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    centers = (
        torch.arange(
            evidence_bins,
            device=probability.device,
            dtype=torch.float32,
        )
        + 0.5
    ) * float(input_w) / float(evidence_bins)

    def expected(values: torch.Tensor) -> torch.Tensor:
        return (values * centers.view(1, 1, 1, -1)).sum(dim=-1)

    def peak(values: torch.Tensor) -> torch.Tensor:
        return centers[values.argmax(dim=-1)]

    batch, instances, rows, _bins = probability.shape

    def vertical_average(kernel: int) -> torch.Tensor:
        values = probability.reshape(
            batch * instances,
            1,
            rows,
            evidence_bins,
        )
        values = F.pad(
            values,
            (0, 0, kernel // 2, kernel // 2),
            mode="replicate",
        )
        values = F.avg_pool2d(
            values,
            kernel_size=(kernel, 1),
            stride=1,
        )
        values = values.view(batch, instances, rows, evidence_bins)
        return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    smooth5 = vertical_average(5)
    smooth11 = vertical_average(11)
    return {
        "attention_expected": expected(probability),
        "attention_peak": peak(probability),
        "attention_expected_v5": expected(smooth5),
        "attention_peak_v5": peak(smooth5),
        "attention_expected_v11": expected(smooth11),
        "attention_peak_v11": peak(smooth11),
    }


def _model_score(
    outputs: dict[str, torch.Tensor],
    *,
    quality_power: float,
) -> torch.Tensor:
    exist = torch.softmax(
        outputs["exist_logits"].detach().float(),
        dim=-1,
    )[..., 0]
    quality = torch.sigmoid(outputs["quality_logits"].detach().float())
    return exist * quality.clamp_min(1e-8).pow(float(quality_power))


def _select_curves(
    curves: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if curves.ndim != 3 or indices.ndim != 2:
        raise ValueError("curves/indices must be [B,N,R] and [B,K]")
    return curves.gather(
        1,
        indices[..., None].expand(-1, -1, int(curves.shape[-1])),
    )


def _maps_for_candidates(
    candidates: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    *,
    line_width: float,
) -> list[dict[int, float]]:
    return [
        _best_iou_by_gt(
            candidates[image_index],
            target,
            line_width=float(line_width),
        )
        for image_index, target in enumerate(targets)
    ]


def _method_key(source: str, method: str, condition: str) -> str:
    return f"{source}_{method}_{condition}".replace(".", "p")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if any(
        not 0.0 < float(alpha) <= 1.0 for alpha in args.fusion_alphas
    ):
        raise ValueError("--fusion-alphas must be in (0, 1]")

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
        raise ValueError("alternative-proposal audit requires structured_query")

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
    active_top_k = min(max(int(args.active_top_k), 1), group_size)
    input_w = float(head.input_w)

    stats: dict[str, AlternativeStats] = {}
    baseline_topk_stats = AlternativeStats.create()
    images_seen = 0

    total = len(loader)
    if int(args.max_batches) > 0:
        total = min(total, int(args.max_batches))
    progress = tqdm(
        enumerate(loader),
        total=total,
        desc="attention alternative-proposal audit",
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
            baseline, _stages, baseline_attention = (
                _structured_forward_with_attention(head, features)
            )
        matches = matcher(baseline, targets)
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

        baseline_x = baseline["pred_x_rows"][:, :group_size].float()
        baseline_maps = _maps_for_candidates(
            baseline_x,
            targets,
            line_width=float(args.line_width),
        )
        score = _model_score(
            baseline,
            quality_power=float(args.quality_power),
        )[:, :group_size]
        active_indices = score.topk(active_top_k, dim=-1).indices
        active_baseline = _select_curves(baseline_x, active_indices)
        active_maps = _maps_for_candidates(
            active_baseline,
            targets,
            line_width=float(args.line_width),
        )
        baseline_topk_stats.update(
            before_maps=baseline_maps,
            after_maps=active_maps,
            candidates_per_image=active_top_k,
        )

        source_outputs = {
            "natural": (
                baseline,
                baseline_attention[attention_layer_index],
            ),
            "oracle_bias": (
                oracle_output,
                oracle_attention[attention_layer_index],
            ),
        }
        for source, (source_output, source_attention) in source_outputs.items():
            methods = _attention_curves(
                source_attention,
                input_w=input_w,
            )
            for alpha in args.fusion_alphas:
                alpha = float(alpha)
                alpha_text = f"{alpha:g}".replace(".", "p")
                methods[f"fusion_{alpha_text}"] = (
                    _attention_position_fusion(
                        row_logits=source_output["row_x_logits"],
                        attention=source_attention,
                        input_w=input_w,
                        alpha=alpha,
                    )
                )

            all_active_alternatives: list[torch.Tensor] = []
            for method_name, all_curves in methods.items():
                all_curves = all_curves[:, :group_size].float()
                active_alternative = _select_curves(
                    all_curves,
                    active_indices,
                )
                all_active_alternatives.append(active_alternative)
                candidate_sets = {
                    # Exactly eight candidates: retain the four strongest
                    # baseline proposals and repurpose the four low-score slots.
                    "replace_bottom4": torch.cat(
                        [active_baseline, active_alternative],
                        dim=1,
                    ),
                    # Upper bound with the original group retained.
                    "base8_union_top4": torch.cat(
                        [baseline_x, active_alternative],
                        dim=1,
                    ),
                    # Larger oracle diagnostic: one alternative per query.
                    "base8_union_all8": torch.cat(
                        [baseline_x, all_curves],
                        dim=1,
                    ),
                    "alternative_top4_only": active_alternative,
                }
                for condition_name, candidates in candidate_sets.items():
                    after_maps = _maps_for_candidates(
                        candidates,
                        targets,
                        line_width=float(args.line_width),
                    )
                    key = _method_key(
                        source,
                        method_name,
                        condition_name,
                    )
                    condition_stats = stats.setdefault(
                        key,
                        AlternativeStats.create(),
                    )
                    condition_stats.update(
                        before_maps=baseline_maps,
                        after_maps=after_maps,
                        candidates_per_image=int(candidates.shape[1]),
                    )

            all_active = torch.cat(all_active_alternatives, dim=1)
            all_union = torch.cat([baseline_x, all_active], dim=1)
            all_parity = torch.cat(
                [
                    active_baseline,
                    # The best four alternatives are unknown without GT; this
                    # condition intentionally keeps only the first method
                    # (attention expected) for fixed-budget parity.  The full
                    # union below is the method-bank upper bound.
                    all_active[:, :active_top_k],
                ],
                dim=1,
            )
            for condition_name, candidates in (
                ("method_bank_union", all_union),
                ("expected_parity_control", all_parity),
            ):
                after_maps = _maps_for_candidates(
                    candidates,
                    targets,
                    line_width=float(args.line_width),
                )
                key = _method_key(source, "all_methods", condition_name)
                condition_stats = stats.setdefault(
                    key,
                    AlternativeStats.create(),
                )
                condition_stats.update(
                    before_maps=baseline_maps,
                    after_maps=after_maps,
                    candidates_per_image=int(candidates.shape[1]),
                )
        images_seen += int(images.shape[0])

    summaries = {key: value.summary() for key, value in stats.items()}

    def subset(prefix: str, suffix: str) -> dict[str, dict[str, Any]]:
        return {
            key: value
            for key, value in summaries.items()
            if key.startswith(prefix) and key.endswith(suffix)
        }

    def best(values: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        if not values:
            return None
        key, value = max(
            values.items(),
            key=lambda item: (
                float(item[1]["net_gain_050_points"]),
                float(item[1]["net_gain_070_points"]),
                float(item[1]["mean_best_iou_delta"]),
            ),
        )
        return {"condition": key, **value}

    natural_parity = subset("natural_", "_replace_bottom4")
    oracle_parity = subset("oracle_bias_", "_replace_bottom4")
    natural_union = subset("natural_", "_base8_union_top4")
    oracle_union = subset("oracle_bias_", "_base8_union_top4")
    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "oracle_warning": (
            "Oracle-bias alternatives use GT corridors and assignments. Raw "
            "union recall is a proposal-capacity diagnostic, not deployable "
            "post-processing or a benchmark result."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "split": args.split,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": images_seen,
        "group_size": group_size,
        "active_top_k": active_top_k,
        "fusion_alphas": [float(value) for value in args.fusion_alphas],
        "baseline_topk_control": baseline_topk_stats.summary(),
        "conditions": summaries,
        "decision_summary": {
            "best_natural_fixed8_replacement": best(natural_parity),
            "best_oracle_fixed8_replacement": best(oracle_parity),
            "best_natural_base8_plus4_union": best(natural_union),
            "best_oracle_base8_plus4_union": best(oracle_union),
            "interpretation_rule": (
                "A natural fixed-eight gain supports repurposing dormant slots "
                "for alternative curves. Oracle-only union gain means useful "
                "alternatives exist only after GT-like acquisition and supports "
                "jointly trained reference/denoising. No union gain rejects "
                "attention-derived alternatives as the missing proposal source."
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
