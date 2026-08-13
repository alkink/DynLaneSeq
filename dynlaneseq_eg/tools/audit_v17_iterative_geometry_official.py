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
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    evaluator_hungarian_assignment,
    sha256_file,
)
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
    "v7_anchor",
    "v17_stage1",
    "v17_stage2",
    "v17_final",
    "cross_clip_wrong_image_final",
    "zero_image_final",
    "no_proposal_context_final",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official V17 held-out/validation causal replay."
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
    cfg.setdefault("dataset", {})["root"] = str(Path(args.dataset_root).expanduser())
    cfg["dataset"].setdefault("lists", {})[args.split] = str(Path(list_path).resolve())
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _required(outputs: dict[str, Any], name: str) -> torch.Tensor:
    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"missing V17 output: {name}")
    return value


def _deltas(metrics: dict[str, Any]) -> dict[str, Any]:
    comparisons = OrderedDict(
        (
            ("v17_minus_v7", ("v17_final", "v7_anchor")),
            ("stage1_minus_v7", ("v17_stage1", "v7_anchor")),
            ("stage2_minus_stage1", ("v17_stage2", "v17_stage1")),
            ("stage3_minus_stage2", ("v17_final", "v17_stage2")),
            (
                "correct_minus_cross_clip_wrong_image",
                ("v17_final", "cross_clip_wrong_image_final"),
            ),
            ("correct_minus_zero_image", ("v17_final", "zero_image_final")),
            (
                "correct_minus_no_proposal_context",
                ("v17_final", "no_proposal_context_final"),
            ),
        )
    )
    return {
        label: {
            mode: {
                threshold: _metric_delta(metrics, treatment, control, mode, threshold)
                for threshold in ("0.50", "0.75")
            }
            for mode in ("neural_active", "writer_valid")
        }
        for label, (treatment, control) in comparisons.items()
    }


def _replace_image_inputs(
    kwargs: dict[str, Any],
    *,
    row_value_features: torch.Tensor,
    multi_scale_features: dict[str, torch.Tensor],
) -> dict[str, Any]:
    values = dict(kwargs)
    values["row_value_features"] = row_value_features
    scales = dict(kwargs["multi_scale_features"])
    for name in ("p3", "p4"):
        scales[name] = multi_scale_features[name]
    values["multi_scale_features"] = scales
    return values


def _zero_image_inputs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _replace_image_inputs(
        kwargs,
        row_value_features=torch.zeros_like(kwargs["row_value_features"]),
        multi_scale_features={
            name: torch.zeros_like(kwargs["multi_scale_features"][name])
            for name in ("p3", "p4")
        },
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    crossclip = json.loads(Path(args.cross_clip_report).read_text())
    if not (
        crossclip.get("passed") is True
        and int(crossclip.get("same_image_partner_count", -1)) == 0
        and int(crossclip.get("same_clip_partner_count", -1)) == 0
    ):
        raise ValueError("invalid cross-clip negative-control report")
    cfg = _config(args, args.list_path)
    wrong_cfg = _config(args, args.wrong_list_path)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    module = model.structured_query_head.set_selection_head.iterative_slot_geometry
    if module is None:
        raise ValueError("V17 official audit requires iterative geometry")
    loader = build_dataloader(cfg, split=args.split, training=False)
    wrong_loader = build_dataloader(wrong_cfg, split=args.split, training=False)
    if len(loader.dataset) != len(wrong_loader.dataset):
        raise ValueError("cross-clip lists are not aligned")

    records: list[dict[str, Any]] = []
    same_image = 0
    same_clip = 0
    intervention_changes: dict[str, list[float]] = {
        "cross_clip_wrong_image": [],
        "zero_image": [],
        "no_proposal_context": [],
    }
    for batch_index, (correct_batch, wrong_batch) in enumerate(
        tqdm(zip(loader, wrong_loader), total=len(loader), desc="V17 official", ncols=90)
    ):
        images, _targets, metas = correct_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        images = images.to(device, non_blocking=True)
        wrong_images = wrong_images.to(device, non_blocking=True)
        kwargs: dict[str, Any] = {}
        wrong_kwargs: dict[str, Any] = {}

        def capture(_module, _args, values):
            kwargs.update(values)

        def capture_wrong(_module, _args, values):
            wrong_kwargs.update(values)

        handle = module.register_forward_pre_hook(capture, with_kwargs=True)
        outputs = model(images)
        handle.remove()
        wrong_handle = module.register_forward_pre_hook(capture_wrong, with_kwargs=True)
        model(wrong_images)
        wrong_handle.remove()
        if not kwargs or not wrong_kwargs:
            raise RuntimeError("V17 audit failed to capture module inputs")
        for meta, wrong_meta in zip(metas, wrong_metas):
            left = str(meta.get("image_path", ""))
            right = str(wrong_meta.get("image_path", ""))
            same_image += int(left == right)
            same_clip += int(str(Path(left).parent) == str(Path(right).parent))

        variants = {
            "cross_clip_wrong_image": module(
                **_replace_image_inputs(
                    kwargs,
                    row_value_features=wrong_kwargs["row_value_features"],
                    multi_scale_features=wrong_kwargs["multi_scale_features"],
                )
            ),
            "zero_image": module(**_zero_image_inputs(kwargs)),
            "no_proposal_context": module(**kwargs, proposal_context_enabled=False),
        }
        final_x = _required(outputs, "selection_slot_pred_x_rows")
        for name, variant in variants.items():
            intervention_changes[name].append(
                float((variant["selection_slot_pred_x_rows"] - final_x).abs().mean().cpu())
            )

        anchor_x = _required(outputs, "selection_slot_v17_anchor_x_rows")
        anchor_range = _required(outputs, "selection_slot_v17_anchor_range_norm")
        stages_x = _required(outputs, "selection_slot_v17_stage_x_rows")
        stages_range = _required(outputs, "selection_slot_v17_stage_range_norm")
        active = _required(outputs, "selection_slot_active").bool()
        policies = OrderedDict(
            (
                ("v7_anchor", (anchor_x, anchor_range)),
                ("v17_stage1", (stages_x[:, 0], stages_range[:, 0])),
                ("v17_stage2", (stages_x[:, 1], stages_range[:, 1])),
                ("v17_final", (stages_x[:, 2], stages_range[:, 2])),
                (
                    "cross_clip_wrong_image_final",
                    (
                        variants["cross_clip_wrong_image"]["selection_slot_pred_x_rows"],
                        variants["cross_clip_wrong_image"]["selection_slot_range_norm"],
                    ),
                ),
                (
                    "zero_image_final",
                    (
                        variants["zero_image"]["selection_slot_pred_x_rows"],
                        variants["zero_image"]["selection_slot_range_norm"],
                    ),
                ),
                (
                    "no_proposal_context_final",
                    (
                        variants["no_proposal_context"]["selection_slot_pred_x_rows"],
                        variants["no_proposal_context"]["selection_slot_range_norm"],
                    ),
                ),
            )
        )
        for batch_item, meta in enumerate(metas):
            geometry_rows: list[torch.Tensor] = []
            range_rows: list[torch.Tensor] = []
            layout: dict[str, tuple[int, int]] = {}
            active_by_policy: dict[str, torch.Tensor] = {}
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
                    "image_id": _image_id(meta, f"v17_{batch_index:06d}_{batch_item}"),
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
        raise ValueError("runtime cross-clip pairing was contaminated")

    evaluated = _evaluate_records(records, line_width=30.0, min_valid_rows=5, workers=int(args.metric_workers))
    tree = _new_metric_tree(POLICIES, (0.50, 0.75))
    paired = {
        threshold: {"improved": 0, "worsened": 0, "tied": 0, "rows": []}
        for threshold in ("0.50", "0.75")
    }
    for item, result in zip(records, evaluated):
        _accumulate_record(tree, result, item["layout"], item["active_by_policy"], (0.50, 0.75))
        quality = result["quality"].float()
        valid = result["valid"].bool()
        for threshold in (0.50, 0.75):
            hits: dict[str, int] = {}
            for policy in ("v7_anchor", "v17_final"):
                start, stop = item["layout"][policy]
                selected = item["active_by_policy"][policy].bool() & valid[start:stop]
                ids = torch.nonzero(selected, as_tuple=False).flatten().tolist()
                hits[policy] = int(
                    evaluator_hungarian_assignment(
                        quality[:, start:stop], ids, threshold=float(threshold)
                    ).hit_count
                )
            delta = hits["v17_final"] - hits["v7_anchor"]
            label = "improved" if delta > 0 else "worsened" if delta < 0 else "tied"
            paired[f"{threshold:.2f}"][label] += 1
            paired[f"{threshold:.2f}"]["rows"].append(
                {"image_id": item["image_id"], "source_tp": hits["v7_anchor"], "v17_tp": hits["v17_final"], "delta_tp": delta}
            )
    metrics = _finalize_metric_tree(tree)
    report = {
        "experiment": "V17 iterative multi-scale geometry official causal replay",
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
        "activity_score_route_source": "exact_v7",
        "hard_proposal_id_used_for_final_geometry": False,
        "proposal_coordinate_average_used_for_final_geometry": False,
        "proposal_detector_frozen": True,
        "intervention_mean_abs_x_change_px": {
            name: sum(values) / max(len(values), 1)
            for name, values in intervention_changes.items()
        },
        "metrics": metrics,
        "deltas": _deltas(metrics),
        "paired_image_effects": paired,
        "optimizer_steps_during_audit": 0,
        "test_set_used": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(destination), "deltas": report["deltas"]}, indent=2))


if __name__ == "__main__":
    main()
