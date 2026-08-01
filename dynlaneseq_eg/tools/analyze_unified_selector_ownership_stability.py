from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    official_proposal_gt_iou_matrix,
)
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    _frozen_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the GT-to-query owner selected by unique "
            "official-IoU assignment remains stable across unified-selector "
            "training checkpoints."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--stable-iou-floor", type=float, default=0.30)
    parser.add_argument("--near-tie-margin", type=float, default=0.02)
    parser.add_argument("--min-owner-retention", type=float, default=0.75)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, name: str):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(name)
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg.setdefault("model", {}).setdefault("structured_query", {})[
        "intermediate_supervision"
    ] = False
    return cfg


def _assignment(
    official_iou: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> tuple[list[int], list[float]]:
    gt_count, candidate_count = official_iou.shape
    owners = [-1 for _ in range(int(gt_count))]
    qualities = [0.0 for _ in range(int(gt_count))]
    if int(gt_count) == 0 or int(candidate_count) == 0:
        return owners, qualities
    valid_ids = [
        index
        for index in range(int(candidate_count))
        if bool(candidate_valid[index])
    ]
    if not valid_ids:
        return owners, qualities
    matrix = official_iou[:, valid_ids].detach().cpu().numpy()
    gt_ids, local_candidate_ids = linear_sum_assignment(1.0 - matrix)
    for gt_index, local_index in zip(gt_ids, local_candidate_ids):
        candidate_index = int(valid_ids[int(local_index)])
        owners[int(gt_index)] = candidate_index
        qualities[int(gt_index)] = float(
            official_iou[int(gt_index), candidate_index]
        )
    return owners, qualities


def _candidate_valid(stage: dict[str, torch.Tensor]) -> torch.Tensor:
    count = int(stage["pred_x_rows"].shape[0])
    return torch.ones(count, dtype=torch.bool)


@torch.no_grad()
def _collect_checkpoint(
    model: torch.nn.Module,
    matcher: Any,
    cfg: dict[str, Any],
    checkpoint: str,
    *,
    device: torch.device,
    args: argparse.Namespace,
    channels_last: bool,
) -> dict[str, Any]:
    iteration = load_checkpoint(checkpoint, model, strict=False)
    model.eval()
    base_loader = build_dataloader(cfg, split="val", training=False)
    loader, dataset_indices = select_diagnostic_loader(
        base_loader,
        strategy="uniform",
        max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
    records: list[dict[str, Any]] = []
    raw_hits = {0.5: 0, 0.7: 0}
    oracle_hits = {0.5: 0, 0.7: 0}
    selected_hits = {0.5: 0, 0.7: 0}
    gt_total = 0
    near_ties = 0
    eligible_ties = 0
    training_official_agreement = 0
    training_official_pairs = 0
    for images, batch_targets, metas in tqdm(
        loader,
        desc=f"owner trace {Path(checkpoint).stem}",
        ncols=90,
    ):
        if channels_last:
            images = images.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images.to(device, non_blocking=True)
        with _amp_context(device, args.amp_dtype):
            outputs = _frozen_outputs(model, images)
        device_targets = nested_to_device(batch_targets, device)
        training_matches = matcher(outputs, device_targets)
        if "selection_logits" not in outputs:
            raise ValueError("checkpoint does not expose unified selection logits")
        for batch_index, meta in enumerate(metas):
            stage = {
                name: outputs[name][batch_index].detach().float().cpu()
                for name in (
                    "pred_x_rows",
                    "range_norm",
                    "exist_logits",
                    "quality_logits",
                    "selection_logits",
                )
                if isinstance(outputs.get(name), torch.Tensor)
            }
            record_for_metric = {"stages": {"main": stage}, "meta": meta}
            official_iou, candidate_valid = official_proposal_gt_iou_matrix(
                record_for_metric,
                "main",
                line_width=float(args.line_width),
                min_valid_rows=int(args.min_valid_rows),
                row_visibility_thresh=float(args.row_visibility_thresh),
            )
            if candidate_valid.numel() == 0:
                candidate_valid = _candidate_valid(stage)
            owners, owner_qualities = _assignment(
                official_iou,
                candidate_valid,
            )
            training_owners = [-1 for _ in range(int(official_iou.shape[0]))]
            training_owner_qualities = [
                0.0 for _ in range(int(official_iou.shape[0]))
            ]
            match = training_matches[batch_index]
            for pred_index, gt_index in zip(
                match["pred_indices"].detach().cpu().tolist(),
                match["gt_indices"].detach().cpu().tolist(),
            ):
                if int(gt_index) >= len(training_owners):
                    raise ValueError("matcher GT index exceeds official GT count")
                training_owners[int(gt_index)] = int(pred_index)
                training_owner_qualities[int(gt_index)] = float(
                    official_iou[int(gt_index), int(pred_index)]
                )
            for official_owner, training_owner in zip(owners, training_owners):
                if int(official_owner) < 0 or int(training_owner) < 0:
                    continue
                training_official_pairs += 1
                training_official_agreement += int(
                    int(official_owner) == int(training_owner)
                )
            gt_total += int(official_iou.shape[0])
            valid_ids = [
                index
                for index in range(int(candidate_valid.shape[0]))
                if bool(candidate_valid[index])
            ]
            scores = torch.sigmoid(stage["selection_logits"])
            selected_ids = sorted(
                valid_ids,
                key=lambda index: float(scores[index]),
                reverse=True,
            )[: int(args.top_k)]
            for threshold in (0.5, 0.7):
                raw_hits[threshold] += int(
                    evaluator_hungarian_assignment(
                        official_iou,
                        valid_ids,
                        threshold=threshold,
                    ).hit_count
                )
                selected_hits[threshold] += int(
                    evaluator_hungarian_assignment(
                        official_iou,
                        selected_ids,
                        threshold=threshold,
                    ).hit_count
                )
                oracle_hits[threshold] += int(
                    cardinality_oracle_assignment(
                        official_iou,
                        threshold=threshold,
                        top_k=int(args.top_k),
                        candidate_valid=candidate_valid,
                    ).hit_count
                )
            for gt_index in range(int(official_iou.shape[0])):
                values = official_iou[gt_index, candidate_valid]
                if int(values.numel()) < 2:
                    continue
                top_two = torch.topk(values, k=2).values
                if float(top_two[0]) >= float(args.stable_iou_floor):
                    eligible_ties += 1
                    near_ties += int(
                        float(top_two[0] - top_two[1])
                        <= float(args.near_tie_margin)
                    )
            records.append(
                {
                    "image_path": str(meta.get("image_path", "")),
                    "owners": owners,
                    "owner_qualities": owner_qualities,
                    "training_owners": training_owners,
                    "training_owner_qualities": training_owner_qualities,
                    "official_iou": official_iou,
                    "candidate_valid": candidate_valid,
                }
            )
    return {
        "checkpoint": checkpoint,
        "iteration": int(iteration),
        "dataset_indices": dataset_indices,
        "records": records,
        "summary": {
            "images": len(records),
            "gt_lanes": int(gt_total),
            "raw_recall_050": raw_hits[0.5] / float(max(gt_total, 1)),
            "raw_recall_070": raw_hits[0.7] / float(max(gt_total, 1)),
            "selection_top4_recall_050": selected_hits[0.5]
            / float(max(gt_total, 1)),
            "selection_top4_recall_070": selected_hits[0.7]
            / float(max(gt_total, 1)),
            "oracle_top4_recall_050": oracle_hits[0.5]
            / float(max(gt_total, 1)),
            "oracle_top4_recall_070": oracle_hits[0.7]
            / float(max(gt_total, 1)),
            "near_tie_fraction_among_recoverable_gt": near_ties
            / float(max(eligible_ties, 1)),
            "near_tie_gt": int(near_ties),
            "recoverable_gt_for_tie_test": int(eligible_ties),
            "training_vs_official_owner_agreement": training_official_agreement
            / float(max(training_official_pairs, 1)),
        },
    }


def _compare(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    stable_iou_floor: float,
    near_tie_margin: float,
    owner_field: str = "owners",
    quality_field: str = "owner_qualities",
) -> dict[str, Any]:
    if before["dataset_indices"] != after["dataset_indices"]:
        raise ValueError("checkpoint traces use different dataset indices")
    if len(before["records"]) != len(after["records"]):
        raise ValueError("checkpoint traces have different image counts")
    all_count = 0
    same_all = 0
    eligible = 0
    same_eligible = 0
    changed_eligible = 0
    changed_near_equivalent = 0
    image_jaccard_sum = 0.0
    for record_a, record_b in zip(before["records"], after["records"]):
        if record_a["image_path"] != record_b["image_path"]:
            raise ValueError("checkpoint traces are not image aligned")
        owners_a = record_a[owner_field]
        owners_b = record_b[owner_field]
        if len(owners_a) != len(owners_b):
            raise ValueError("GT lane count changed across checkpoints")
        set_a = {int(value) for value in owners_a if int(value) >= 0}
        set_b = {int(value) for value in owners_b if int(value) >= 0}
        union = set_a | set_b
        image_jaccard_sum += len(set_a & set_b) / float(max(len(union), 1))
        matrix_a = record_a["official_iou"]
        matrix_b = record_b["official_iou"]
        for gt_index, (owner_a, owner_b) in enumerate(zip(owners_a, owners_b)):
            if int(owner_a) < 0 or int(owner_b) < 0:
                continue
            all_count += 1
            same = int(owner_a) == int(owner_b)
            same_all += int(same)
            quality_a = float(record_a[quality_field][gt_index])
            quality_b = float(record_b[quality_field][gt_index])
            if min(quality_a, quality_b) < float(stable_iou_floor):
                continue
            eligible += 1
            same_eligible += int(same)
            if same:
                continue
            changed_eligible += 1
            # A changed owner is near-equivalent when both competing query
            # identities are within the requested IoU margin in both snapshots.
            cross_a = float(matrix_a[gt_index, int(owner_b)])
            cross_b = float(matrix_b[gt_index, int(owner_a)])
            changed_near_equivalent += int(
                abs(quality_a - cross_a) <= float(near_tie_margin)
                and abs(quality_b - cross_b) <= float(near_tie_margin)
            )
    images = max(len(before["records"]), 1)
    return {
        "before_iteration": int(before["iteration"]),
        "after_iteration": int(after["iteration"]),
        "assigned_gt_pairs": int(all_count),
        "owner_retention_all": same_all / float(max(all_count, 1)),
        "eligible_gt_both_iou_ge_floor": int(eligible),
        "owner_retention_recoverable": same_eligible / float(max(eligible, 1)),
        "owner_change_recoverable": changed_eligible / float(max(eligible, 1)),
        "changed_owner_near_equivalent_fraction": changed_near_equivalent
        / float(max(changed_eligible, 1)),
        "mean_positive_query_set_jaccard": image_jaccard_sum / float(images),
    }


def _public_checkpoint(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": row["checkpoint"],
        "iteration": row["iteration"],
        "summary": row["summary"],
    }


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise ValueError("at least two checkpoints are required")
    if int(args.eval_batch_size) < 1 or int(args.max_batches) < 1:
        raise ValueError("eval_batch_size and max_batches must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    cfg = _prepare_config(args)
    device = torch.device(args.device)
    model = build_model(cfg)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if model.structured_query_head is None:
        raise ValueError("ownership analysis requires a structured query head")
    model.structured_query_head.intermediate_supervision = False
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    matcher = build_matcher(cfg)

    checkpoints = [
        _collect_checkpoint(
            model,
            matcher,
            cfg,
            checkpoint,
            device=device,
            args=args,
            channels_last=channels_last,
        )
        for checkpoint in args.checkpoints
    ]
    checkpoints.sort(key=lambda row: int(row["iteration"]))
    def comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        return {
            "before_iteration": int(before["iteration"]),
            "after_iteration": int(after["iteration"]),
            "training_assignment": _compare(
                before,
                after,
                stable_iou_floor=args.stable_iou_floor,
                near_tie_margin=args.near_tie_margin,
                owner_field="training_owners",
                quality_field="training_owner_qualities",
            ),
            "official_assignment": _compare(
                before,
                after,
                stable_iou_floor=args.stable_iou_floor,
                near_tie_margin=args.near_tie_margin,
            ),
        }

    consecutive = [
        comparison(before, after)
        for before, after in zip(checkpoints[:-1], checkpoints[1:])
    ]
    first_to_last = comparison(checkpoints[0], checkpoints[-1])
    weighted_eligible = sum(
        int(row["training_assignment"]["eligible_gt_both_iou_ge_floor"])
        for row in consecutive
    )
    weighted_retention = sum(
        float(row["training_assignment"]["owner_retention_recoverable"])
        * int(row["training_assignment"]["eligible_gt_both_iou_ge_floor"])
        for row in consecutive
    ) / float(max(weighted_eligible, 1))
    ownership_unstable = weighted_retention < float(args.min_owner_retention)
    result = {
        "diagnostic_only": True,
        "warning": (
            "Query ownership is an offline Hungarian diagnostic over fixed "
            "images. It measures target identity churn, not benchmark F1."
        ),
        "config": args.config,
        "sample_strategy": "uniform",
        "images": len(checkpoints[0]["records"]),
        "stable_iou_floor": float(args.stable_iou_floor),
        "near_tie_margin": float(args.near_tie_margin),
        "checkpoints": [_public_checkpoint(row) for row in checkpoints],
        "consecutive": consecutive,
        "first_to_last": first_to_last,
        "gate": {
            "min_owner_retention": float(args.min_owner_retention),
            "weighted_consecutive_training_owner_retention_recoverable": weighted_retention,
            "ownership_unstable": bool(ownership_unstable),
            "interpretation": (
                "one_to_one_target_identity_moves_across_training"
                if ownership_unstable
                else "query_ownership_is_sufficiently_stable"
            ),
        },
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {path}")


if __name__ == "__main__":
    main()
