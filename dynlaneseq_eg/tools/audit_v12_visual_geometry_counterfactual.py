from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.losses.matcher_s0 import HungarianMatcherS0
from dynlaneseq_eg.modeling.common import fixed_row_fractions, sort_range_norm
from dynlaneseq_eg.modeling.four_slot_selection import (
    structured_unique_route_marginals,
)
from dynlaneseq_eg.tools.audit_v11_causal_replay import (
    _accumulate_record,
    _evaluate_records,
    _finalize_metric_tree,
    _image_id,
    _metric_delta,
    _new_metric_tree,
    _plain_meta,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.train import seed_everything


POLICIES = (
    "v7_deployment",
    "v12_first_visual_direct",
    "v12_visual_direct",
    "wrong_p2_visual_direct",
    "zero_p2_visual_direct",
    "visual_distance_hard_proposal",
    "visual_distance_soft_proposal",
    "learned_attention_hard_proposal",
    "learned_attention_soft_proposal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether V12's generalizable visual curve should own "
            "final geometry or be mapped back to a proposal identity."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--list-path", required=True)
    parser.add_argument(
        "--sample-strategy",
        choices=("sequential", "uniform"),
        default="sequential",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=(0.50, 0.75)
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg.setdefault("dataset", {}).setdefault("lists", {})[args.split] = str(
        Path(args.list_path).expanduser().resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _inference(model: torch.nn.Module, images: torch.Tensor) -> dict[str, Any]:
    if bool(getattr(model, "supports_inference_only", False)):
        return model(images, inference_only=True)
    return model(images)


def _required(outputs: dict[str, Any], name: str) -> torch.Tensor:
    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"required V12 output is unavailable: {name}")
    return value


def _curve_distance(
    visual_x: torch.Tensor,
    proposal_x: torch.Tensor,
    proposal_range: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Mean row distance from each visual lane to every coherent proposal."""
    rows = int(proposal_x.shape[-1])
    row_y = fixed_row_fractions(
        rows, device=proposal_x.device, dtype=torch.float32
    ).view(1, 1, rows)
    proposal_range = sort_range_norm(proposal_range.float())
    visible = (
        (row_y >= proposal_range[..., :1])
        & (row_y <= proposal_range[..., 1:])
        & torch.isfinite(proposal_x)
    )
    weight = visible[:, None].float()
    distance = (
        (proposal_x[:, None].float() - visual_x.unsqueeze(2).float()).abs()
        * weight
    ).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    return distance.masked_fill(~candidate_valid[:, None].bool(), 1.0e9)


def _hard_unique_indices(
    cost: torch.Tensor, candidate_valid: torch.Tensor
) -> torch.Tensor:
    batch, slots, _candidates = cost.shape
    result = torch.full(
        (batch, slots), -1, dtype=torch.long, device=cost.device
    )
    for batch_index in range(batch):
        valid_ids = torch.nonzero(
            candidate_valid[batch_index].bool(), as_tuple=False
        ).flatten()
        if valid_ids.numel() == 0:
            continue
        local = cost[batch_index].index_select(1, valid_ids).detach().cpu()
        slot_ids, local_ids = HungarianMatcherS0._linear_sum_assignment(local)
        result[batch_index, slot_ids.to(result.device)] = valid_ids.index_select(
            0, local_ids.to(valid_ids.device)
        )
    return result


def _gather_proposals(
    indices: torch.Tensor,
    proposal_x: torch.Tensor,
    proposal_range: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(proposal_x.shape[-1])
    safe = indices.clamp(min=0, max=max(int(proposal_x.shape[1]) - 1, 0))
    x = proposal_x.gather(1, safe.unsqueeze(-1).expand(-1, -1, rows))
    lane_range = proposal_range.gather(
        1, safe.unsqueeze(-1).expand(-1, -1, 2)
    )
    valid = indices >= 0
    x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
    lane_range = torch.where(
        valid.unsqueeze(-1), lane_range, torch.zeros_like(lane_range)
    )
    return x, sort_range_norm(lane_range.float())


def _soft_proposals(
    probability: torch.Tensor,
    proposal_x: torch.Tensor,
    proposal_range: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.einsum("bsn,bnr->bsr", probability.float(), proposal_x.float())
    lane_range = torch.einsum(
        "bsn,bnd->bsd", probability.float(), proposal_range.float()
    )
    return x, sort_range_norm(lane_range)


def _policy_deltas(
    metrics: dict[str, Any], thresholds: tuple[float, ...]
) -> dict[str, Any]:
    comparisons = OrderedDict(
        (
            ("visual_direct_minus_v7", ("v12_visual_direct", "v7_deployment")),
            (
                "correct_visual_minus_wrong_p2",
                ("v12_visual_direct", "wrong_p2_visual_direct"),
            ),
            (
                "correct_visual_minus_zero_p2",
                ("v12_visual_direct", "zero_p2_visual_direct"),
            ),
            (
                "distance_hard_proposal_minus_v7",
                ("visual_distance_hard_proposal", "v7_deployment"),
            ),
            (
                "distance_soft_proposal_minus_v7",
                ("visual_distance_soft_proposal", "v7_deployment"),
            ),
            (
                "learned_hard_proposal_minus_v7",
                ("learned_attention_hard_proposal", "v7_deployment"),
            ),
            (
                "learned_soft_proposal_minus_v7",
                ("learned_attention_soft_proposal", "v7_deployment"),
            ),
        )
    )
    return {
        label: {
            mode: {
                f"{threshold:.2f}": _metric_delta(
                    metrics,
                    treatment,
                    control,
                    mode,
                    f"{threshold:.2f}",
                )
                for threshold in thresholds
            }
            for mode in ("neural_active", "writer_valid")
        }
        for label, (treatment, control) in comparisons.items()
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    thresholds = tuple(float(value) for value in args.iou_thresholds)
    list_path = Path(args.list_path).expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f"evaluation list does not exist: {list_path}")
    cfg = _prepare_config(args)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    model.eval()
    selector = model.structured_query_head.set_selection_head
    module = selector.visual_first_association
    if module is None:
        raise ValueError("V12 visual-first module is unavailable")

    loader = build_dataloader(cfg, split=args.split, training=False)
    max_batches = (
        0
        if int(args.max_images) <= 0
        else math.ceil(int(args.max_images) / int(args.eval_batch_size))
    )
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=max_batches,
        num_workers=int(args.num_workers),
    )
    records: list[dict[str, Any]] = []
    image_count = 0
    for batch_index, (images, _targets, metas) in enumerate(
        tqdm(loader, desc="V12 visual geometry replay", ncols=90)
    ):
        images = images.to(device, non_blocking=True)
        captured: dict[str, torch.Tensor] = {}

        def capture(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            captured.update(kwargs)

        handle = module.register_forward_pre_hook(capture, with_kwargs=True)
        outputs = _inference(model, images)
        handle.remove()
        if not captured:
            raise RuntimeError("V12 replay captured no module inputs")
        wrong_kwargs = dict(captured)
        wrong_kwargs["row_value_features"] = torch.roll(
            captured["row_value_features"], shifts=1, dims=0
        )
        zero_kwargs = dict(captured)
        zero_kwargs["row_value_features"] = torch.zeros_like(
            captured["row_value_features"]
        )
        wrong = module(**wrong_kwargs)
        zero = module(**zero_kwargs)

        proposal_x = captured["proposal_x_rows"].detach().float()
        proposal_range = captured["proposal_range_norm"].detach().float()
        candidate_valid = captured["candidate_valid"].detach().bool()
        visual_x = _required(
            outputs, "selection_slot_v12_visual_x_rows"
        ).float()
        distance = _curve_distance(
            visual_x, proposal_x, proposal_range, candidate_valid
        )
        distance_indices = _hard_unique_indices(distance, candidate_valid)
        distance_hard_x, distance_hard_range = _gather_proposals(
            distance_indices, proposal_x, proposal_range
        )
        distance_probability = structured_unique_route_marginals(
            -module.proposal_curve_distance_scale
            * distance
            / float(max(module.input_w - 1, 1)),
            candidate_valid,
            temperature=module.proposal_attention_temperature,
            iterations=module.proposal_attention_sinkhorn_iterations,
        )
        distance_soft_x, distance_soft_range = _soft_proposals(
            distance_probability, proposal_x, proposal_range
        )

        learned_logits = _required(
            outputs, "selection_slot_v12_proposal_logits"
        ).float()
        learned_indices = _hard_unique_indices(-learned_logits, candidate_valid)
        learned_hard_x, learned_hard_range = _gather_proposals(
            learned_indices, proposal_x, proposal_range
        )
        learned_probability = _required(
            outputs, "selection_slot_v12_proposal_attention"
        ).float()
        learned_soft_x, learned_soft_range = _soft_proposals(
            learned_probability, proposal_x, proposal_range
        )

        v7_range = _required(outputs, "selection_slot_range_norm").float()
        active = _required(outputs, "selection_slot_active").bool()
        batch_policies = OrderedDict(
            (
                (
                    "v7_deployment",
                    (
                        _required(outputs, "selection_slot_pred_x_rows").float(),
                        v7_range,
                    ),
                ),
                (
                    "v12_first_visual_direct",
                    (
                        _required(
                            outputs, "selection_slot_v12_first_visual_x_rows"
                        ).float(),
                        v7_range,
                    ),
                ),
                ("v12_visual_direct", (visual_x, v7_range)),
                (
                    "wrong_p2_visual_direct",
                    (wrong["selection_slot_v12_visual_x_rows"].float(), v7_range),
                ),
                (
                    "zero_p2_visual_direct",
                    (zero["selection_slot_v12_visual_x_rows"].float(), v7_range),
                ),
                (
                    "visual_distance_hard_proposal",
                    (distance_hard_x, distance_hard_range),
                ),
                (
                    "visual_distance_soft_proposal",
                    (distance_soft_x, distance_soft_range),
                ),
                (
                    "learned_attention_hard_proposal",
                    (learned_hard_x, learned_hard_range),
                ),
                (
                    "learned_attention_soft_proposal",
                    (learned_soft_x, learned_soft_range),
                ),
            )
        )
        if tuple(batch_policies) != POLICIES:
            raise RuntimeError("V12 counterfactual policy order changed")

        take = len(metas)
        if int(args.max_images) > 0:
            take = min(take, int(args.max_images) - image_count)
        for bi, meta in enumerate(metas[:take]):
            geometry: list[torch.Tensor] = []
            ranges: list[torch.Tensor] = []
            layout: dict[str, tuple[int, int]] = {}
            active_by_policy: dict[str, torch.Tensor] = {}
            cursor = 0
            for name, (x, lane_range) in batch_policies.items():
                geometry.append(x[bi].detach().cpu())
                ranges.append(lane_range[bi].detach().cpu())
                count = int(x.shape[1])
                layout[name] = (cursor, cursor + count)
                active_by_policy[name] = active[bi].detach().cpu()
                cursor += count
            records.append(
                {
                    "image_id": _image_id(
                        meta, f"v12_{batch_index:06d}_{bi}"
                    ),
                    "record": {
                        "meta": _plain_meta(meta),
                        "stages": {
                            "combined": {
                                "pred_x_rows": torch.cat(geometry, dim=0),
                                "range_norm": torch.cat(ranges, dim=0),
                            }
                        },
                    },
                    "layout": layout,
                    "active_by_policy": active_by_policy,
                }
            )
        image_count += take
        if int(args.max_images) > 0 and image_count >= int(args.max_images):
            break

    evaluated = _evaluate_records(
        records,
        line_width=float(args.line_width),
        min_valid_rows=int(args.min_valid_rows),
        workers=int(args.metric_workers),
    )
    tree = _new_metric_tree(POLICIES, thresholds)
    for item, result in zip(records, evaluated):
        _accumulate_record(
            tree,
            result,
            item["layout"],
            item["active_by_policy"],
            thresholds,
        )
    metrics = _finalize_metric_tree(tree)
    report = {
        "experiment": "V12 visual-geometry ownership counterfactual",
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "iteration": iteration,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "split": args.split,
        "list_path": str(list_path),
        "list_sha256": sha256_file(list_path),
        "sample_strategy": args.sample_strategy,
        "sampled_indices": sampled_indices[: len(records)],
        "images": len(records),
        "thresholds": list(thresholds),
        "line_width": float(args.line_width),
        "min_valid_rows": int(args.min_valid_rows),
        "policies": list(POLICIES),
        "metrics": metrics,
        "deltas": _policy_deltas(metrics, thresholds),
        "optimizer_steps": 0,
        "test_set_used": False,
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output.resolve()),
                "images": len(records),
                "iteration": iteration,
                "writer_valid": {
                    name: metrics[name]["writer_valid"]["thresholds"]
                    for name in POLICIES
                },
                "deltas": report["deltas"],
            },
            indent=2,
        )
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
