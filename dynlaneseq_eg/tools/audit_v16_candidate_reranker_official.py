from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.audit_v11_causal_replay import (
    _accumulate_record,
    _evaluate_records,
    _finalize_metric_tree,
    _image_id,
    _metric_delta,
    _new_metric_tree,
    _plain_meta,
)
from dynlaneseq_eg.tools.train import seed_everything


POLICIES = (
    "v7_anchor_reference",
    "v7_refined",
    "v16_selected_reference",
    "cross_clip_wrong_p2_selected_reference",
    "zero_p2_selected_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official V16 held-out/validation hard-reranking replay."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--wrong-list-path", required=True)
    parser.add_argument("--cross-clip-report", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _config(args: argparse.Namespace, list_path: str) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg["dataset"].setdefault("lists", {})[args.split] = str(
        Path(list_path).expanduser().resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _required(outputs: dict[str, Any], name: str) -> torch.Tensor:
    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"missing V16 output: {name}")
    return value


def _deltas(metrics: dict[str, Any]) -> dict[str, Any]:
    comparisons = OrderedDict(
        (
            (
                "v16_minus_v7_anchor_reference",
                ("v16_selected_reference", "v7_anchor_reference"),
            ),
            (
                "v16_minus_v7_refined",
                ("v16_selected_reference", "v7_refined"),
            ),
            (
                "correct_minus_cross_clip_wrong_p2",
                (
                    "v16_selected_reference",
                    "cross_clip_wrong_p2_selected_reference",
                ),
            ),
            (
                "correct_minus_zero_p2",
                ("v16_selected_reference", "zero_p2_selected_reference"),
            ),
        )
    )
    return {
        label: {
            mode: {
                threshold: _metric_delta(
                    metrics, treatment, control, mode, threshold
                )
                for threshold in ("0.50", "0.75")
            }
            for mode in ("neural_active", "writer_valid")
        }
        for label, (treatment, control) in comparisons.items()
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    crossclip = json.loads(Path(args.cross_clip_report).expanduser().read_text())
    if not (
        crossclip.get("passed") is True
        and int(crossclip.get("same_image_partner_count", -1)) == 0
        and int(crossclip.get("same_clip_partner_count", -1)) == 0
    ):
        raise ValueError("invalid V16 cross-clip negative-control report")
    cfg = _config(args, args.list_path)
    wrong_cfg = _config(args, args.wrong_list_path)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    selector = model.structured_query_head.set_selection_head
    module = selector.candidate_aligned_reranker
    if module is None:
        raise ValueError("V16 official audit requires candidate reranker")
    loader = build_dataloader(cfg, split=args.split, training=False)
    wrong_loader = build_dataloader(wrong_cfg, split=args.split, training=False)
    if len(loader.dataset) != len(wrong_loader.dataset):
        raise ValueError("V16 cross-clip lists are not aligned")

    records: list[dict[str, Any]] = []
    same_image = 0
    same_clip = 0
    writer_slots = 0
    selection_changed_from_anchor = 0
    wrong_selection_changed = 0
    zero_selection_changed = 0
    selected_duplicate_count = 0
    score_change_wrong: list[float] = []
    score_change_zero: list[float] = []
    group_sizes: list[float] = []
    for batch_index, (correct_batch, wrong_batch) in enumerate(
        tqdm(
            zip(loader, wrong_loader),
            total=len(loader),
            desc="V16 official",
            ncols=90,
        )
    ):
        images, _targets, metas = correct_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        images = images.to(device, non_blocking=True)
        wrong_images = wrong_images.to(device, non_blocking=True)
        kwargs: dict[str, torch.Tensor] = {}
        wrong_kwargs: dict[str, torch.Tensor] = {}

        def capture(_module, _args, values):
            kwargs.update(values)

        def capture_wrong(_module, _args, values):
            wrong_kwargs.update(values)

        handle = module.register_forward_pre_hook(capture, with_kwargs=True)
        outputs = model(images)
        handle.remove()
        wrong_handle = module.register_forward_pre_hook(
            capture_wrong, with_kwargs=True
        )
        model(wrong_images)
        wrong_handle.remove()
        if not kwargs or not wrong_kwargs:
            raise RuntimeError("V16 audit failed to capture module inputs")
        for meta, wrong_meta in zip(metas, wrong_metas):
            left = str(meta.get("image_path", ""))
            right = str(wrong_meta.get("image_path", ""))
            same_image += int(left == right)
            same_clip += int(str(Path(left).parent) == str(Path(right).parent))

        wrong_variant = module(
            **{
                **kwargs,
                "row_value_features": wrong_kwargs["row_value_features"],
            }
        )
        zero_variant = module(**kwargs, feature_policy="zero_content")
        writer_valid = _required(outputs, "selection_slot_v16_writer_valid").bool()
        anchor_ids = _required(outputs, "selection_slot_v16_anchor_indices").long()
        selected_ids = _required(
            outputs, "selection_slot_v16_selected_indices"
        ).long()
        wrong_ids = _required(
            wrong_variant, "selection_slot_v16_selected_indices"
        ).long()
        zero_ids = _required(
            zero_variant, "selection_slot_v16_selected_indices"
        ).long()
        writer_slots += int(writer_valid.sum())
        selection_changed_from_anchor += int(
            ((selected_ids != anchor_ids) & writer_valid).sum()
        )
        wrong_selection_changed += int(
            ((selected_ids != wrong_ids) & writer_valid).sum()
        )
        zero_selection_changed += int(
            ((selected_ids != zero_ids) & writer_valid).sum()
        )
        for image_ids, active_ids in zip(selected_ids, writer_valid):
            values = image_ids[active_ids]
            selected_duplicate_count += int(
                values.numel() - values.unique().numel()
            )
        group_sizes.extend(
            _required(outputs, "selection_slot_v16_group_size")[writer_valid]
            .float()
            .cpu()
            .tolist()
        )
        mask = _required(outputs, "selection_slot_v16_group_mask").bool()
        correct_scores = _required(
            outputs, "selection_slot_v16_candidate_scores"
        ).float()
        wrong_scores = _required(
            wrong_variant, "selection_slot_v16_candidate_scores"
        ).float()
        zero_scores = _required(
            zero_variant, "selection_slot_v16_candidate_scores"
        ).float()
        score_change_wrong.append(
            float((correct_scores - wrong_scores)[mask].abs().mean().cpu())
        )
        score_change_zero.append(
            float((correct_scores - zero_scores)[mask].abs().mean().cpu())
        )

        active = writer_valid
        policies = OrderedDict(
            (
                (
                    "v7_anchor_reference",
                    (
                        _required(
                            outputs,
                            "selection_slot_v16_anchor_reference_x_rows",
                        ),
                        _required(
                            outputs,
                            "selection_slot_v16_anchor_reference_range_norm",
                        ),
                    ),
                ),
                (
                    "v7_refined",
                    (
                        _required(outputs, "selection_slot_v16_anchor_x_rows"),
                        _required(
                            outputs, "selection_slot_v16_anchor_range_norm"
                        ),
                    ),
                ),
                (
                    "v16_selected_reference",
                    (
                        _required(
                            outputs, "selection_slot_v16_selected_x_rows"
                        ),
                        _required(
                            outputs, "selection_slot_v16_selected_range_norm"
                        ),
                    ),
                ),
                (
                    "cross_clip_wrong_p2_selected_reference",
                    (
                        _required(
                            wrong_variant,
                            "selection_slot_v16_selected_x_rows",
                        ),
                        _required(
                            wrong_variant,
                            "selection_slot_v16_selected_range_norm",
                        ),
                    ),
                ),
                (
                    "zero_p2_selected_reference",
                    (
                        _required(
                            zero_variant,
                            "selection_slot_v16_selected_x_rows",
                        ),
                        _required(
                            zero_variant,
                            "selection_slot_v16_selected_range_norm",
                        ),
                    ),
                ),
            )
        )
        for batch_item, meta in enumerate(metas):
            geometry_rows = []
            range_rows = []
            layout = {}
            active_by_policy = {}
            cursor = 0
            for name, (x_rows, lane_range) in policies.items():
                geometry_rows.append(x_rows[batch_item].detach().float().cpu())
                range_rows.append(lane_range[batch_item].detach().float().cpu())
                count = int(x_rows.shape[1])
                layout[name] = (cursor, cursor + count)
                active_by_policy[name] = active[batch_item].detach().cpu()
                cursor += count
            records.append(
                {
                    "image_id": _image_id(
                        meta, f"v16_{batch_index:06d}_{batch_item}"
                    ),
                    "record": {
                        "meta": _plain_meta(meta),
                        "stages": {
                            "combined": {
                                "pred_x_rows": torch.cat(geometry_rows, dim=0),
                                "range_norm": torch.cat(range_rows, dim=0),
                            }
                        },
                    },
                    "layout": layout,
                    "active_by_policy": active_by_policy,
                }
            )
    if same_image or same_clip:
        raise ValueError("runtime V16 cross-clip pairing was contaminated")
    evaluated = _evaluate_records(
        records,
        line_width=30.0,
        min_valid_rows=5,
        workers=int(args.metric_workers),
    )
    tree = _new_metric_tree(POLICIES, (0.50, 0.75))
    for item, result in zip(records, evaluated):
        _accumulate_record(
            tree,
            result,
            item["layout"],
            item["active_by_policy"],
            (0.50, 0.75),
        )
    metrics = _finalize_metric_tree(tree)
    report = {
        "experiment": "V16 candidate-aligned hard-reranker official replay",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "iteration": iteration,
        "split": args.split,
        "list_path": str(Path(args.list_path).resolve()),
        "list_sha256": sha256_file(Path(args.list_path)),
        "wrong_list_path": str(Path(args.wrong_list_path).resolve()),
        "images": len(records),
        "cross_clip_runtime_same_image": same_image,
        "cross_clip_runtime_same_clip": same_clip,
        "writer_valid_slots": writer_slots,
        "selection_changed_from_anchor_fraction": (
            float(selection_changed_from_anchor) / float(max(writer_slots, 1))
        ),
        "correct_vs_wrong_p2_selection_change_fraction": (
            float(wrong_selection_changed) / float(max(writer_slots, 1))
        ),
        "correct_vs_zero_p2_selection_change_fraction": (
            float(zero_selection_changed) / float(max(writer_slots, 1))
        ),
        "selected_duplicate_count": selected_duplicate_count,
        "mean_abs_score_change": {
            "cross_clip_wrong_p2": sum(score_change_wrong)
            / max(len(score_change_wrong), 1),
            "zero_p2": sum(score_change_zero) / max(len(score_change_zero), 1),
        },
        "group_size": {
            "mean": sum(group_sizes) / max(len(group_sizes), 1),
            "min": min(group_sizes) if group_sizes else 0.0,
            "max": max(group_sizes) if group_sizes else 0.0,
        },
        "coordinate_averaging_used": False,
        "fixed_k_or_padding_used": False,
        "activity_count_scores_source": "exact_v7",
        "metrics": metrics,
        "deltas": _deltas(metrics),
        "optimizer_steps_during_audit": 0,
        "test_set_used": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output_json": str(destination), "deltas": report["deltas"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
