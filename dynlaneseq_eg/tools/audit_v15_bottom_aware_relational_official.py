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
    "v7_anchor",
    "v15_final",
    "cross_clip_wrong_p2_final",
    "zero_content_p2_final",
    "identity_graph_final",
    "geometry_shuffled_graph_final",
    "zero_proposal_content_final",
    "no_proposal_context_final",
    "v15_x_v7_range",
    "v7_x_v15_range",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official V15 held-out/validation causal replay."
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
        Path(list_path).resolve()
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
        raise KeyError(f"missing V15 output: {name}")
    return value


def _deltas(metrics: dict[str, Any]) -> dict[str, Any]:
    comparisons = OrderedDict(
        (
            ("v15_minus_v7", ("v15_final", "v7_anchor")),
            (
                "correct_minus_cross_clip_wrong_p2",
                ("v15_final", "cross_clip_wrong_p2_final"),
            ),
            (
                "correct_minus_zero_content_p2",
                ("v15_final", "zero_content_p2_final"),
            ),
            (
                "correct_minus_identity_graph",
                ("v15_final", "identity_graph_final"),
            ),
            (
                "correct_minus_geometry_shuffled_graph",
                ("v15_final", "geometry_shuffled_graph_final"),
            ),
            (
                "correct_minus_zero_proposal_content",
                ("v15_final", "zero_proposal_content_final"),
            ),
            (
                "correct_minus_no_proposal_context",
                ("v15_final", "no_proposal_context_final"),
            ),
            ("v15_x_only_minus_v7", ("v15_x_v7_range", "v7_anchor")),
            ("v15_range_only_minus_v7", ("v7_x_v15_range", "v7_anchor")),
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
    report_crossclip = json.loads(Path(args.cross_clip_report).read_text())
    if not (
        report_crossclip.get("passed") is True
        and int(report_crossclip.get("same_image_partner_count", -1)) == 0
        and int(report_crossclip.get("same_clip_partner_count", -1)) == 0
    ):
        raise ValueError("invalid cross-clip negative-control report")
    cfg = _config(args, args.list_path)
    wrong_cfg = _config(args, args.wrong_list_path)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    selector = model.structured_query_head.set_selection_head
    module = selector.bottom_aware_relational_geometry
    if module is None:
        raise ValueError("V15 official audit requires the relational module")
    loader = build_dataloader(cfg, split=args.split, training=False)
    wrong_loader = build_dataloader(wrong_cfg, split=args.split, training=False)
    if len(loader.dataset) != len(wrong_loader.dataset):
        raise ValueError("cross-clip lists are not aligned")

    records: list[dict[str, Any]] = []
    same_image = 0
    same_clip = 0
    intervention_changes: dict[str, list[float]] = {
        name: []
        for name in (
            "cross_clip_wrong_p2",
            "zero_content_p2",
            "identity_graph",
            "geometry_shuffled_graph",
            "zero_proposal_content",
            "no_proposal_context",
        )
    }
    for batch_index, (correct_batch, wrong_batch) in enumerate(
        tqdm(
            zip(loader, wrong_loader),
            total=len(loader),
            desc="V15 official",
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
            raise RuntimeError("V15 audit failed to capture module inputs")
        for meta, wrong_meta in zip(metas, wrong_metas):
            left = str(meta.get("image_path", ""))
            right = str(wrong_meta.get("image_path", ""))
            same_image += int(left == right)
            same_clip += int(str(Path(left).parent) == str(Path(right).parent))

        variants = {
            "cross_clip_wrong_p2": module(
                **{
                    **kwargs,
                    "row_value_features": wrong_kwargs["row_value_features"],
                }
            ),
            "zero_content_p2": module(**kwargs, feature_policy="zero_content"),
            "identity_graph": module(**kwargs, graph_policy="identity"),
            "geometry_shuffled_graph": module(
                **kwargs, graph_policy="geometry_shuffled"
            ),
            "zero_proposal_content": module(
                **{
                    **kwargs,
                    "proposal_row_tokens": torch.zeros_like(
                        kwargs["proposal_row_tokens"]
                    ),
                }
            ),
            "no_proposal_context": module(
                **kwargs, context_policy="none"
            ),
        }
        final_x = _required(outputs, "selection_slot_pred_x_rows")
        for name, variant in variants.items():
            changed = variant["selection_slot_pred_x_rows"]
            intervention_changes[name].append(
                float((changed - final_x).abs().mean().cpu())
            )

        anchor_x = _required(outputs, "selection_slot_v15_anchor_x_rows")
        anchor_range = _required(outputs, "selection_slot_v15_anchor_range_norm")
        final_range = _required(outputs, "selection_slot_range_norm")
        active = _required(outputs, "selection_slot_active").bool()
        policies = OrderedDict(
            (
                ("v7_anchor", (anchor_x, anchor_range)),
                ("v15_final", (final_x, final_range)),
                (
                    "cross_clip_wrong_p2_final",
                    (
                        variants["cross_clip_wrong_p2"][
                            "selection_slot_pred_x_rows"
                        ],
                        variants["cross_clip_wrong_p2"][
                            "selection_slot_range_norm"
                        ],
                    ),
                ),
                (
                    "zero_content_p2_final",
                    (
                        variants["zero_content_p2"][
                            "selection_slot_pred_x_rows"
                        ],
                        variants["zero_content_p2"][
                            "selection_slot_range_norm"
                        ],
                    ),
                ),
                (
                    "identity_graph_final",
                    (
                        variants["identity_graph"][
                            "selection_slot_pred_x_rows"
                        ],
                        variants["identity_graph"][
                            "selection_slot_range_norm"
                        ],
                    ),
                ),
                (
                    "geometry_shuffled_graph_final",
                    (
                        variants["geometry_shuffled_graph"][
                            "selection_slot_pred_x_rows"
                        ],
                        variants["geometry_shuffled_graph"][
                            "selection_slot_range_norm"
                        ],
                    ),
                ),
                (
                    "zero_proposal_content_final",
                    (
                        variants["zero_proposal_content"][
                            "selection_slot_pred_x_rows"
                        ],
                        variants["zero_proposal_content"][
                            "selection_slot_range_norm"
                        ],
                    ),
                ),
                (
                    "no_proposal_context_final",
                    (
                        variants["no_proposal_context"][
                            "selection_slot_pred_x_rows"
                        ],
                        variants["no_proposal_context"][
                            "selection_slot_range_norm"
                        ],
                    ),
                ),
                ("v15_x_v7_range", (final_x, anchor_range)),
                ("v7_x_v15_range", (anchor_x, final_range)),
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
                        meta, f"v15_{batch_index:06d}_{batch_item}"
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
        raise ValueError("runtime cross-clip pairing was contaminated")
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
        "experiment": "V15 bottom-aware relational official causal replay",
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
        "hard_cluster_or_prototype_used": False,
        "proposal_id_supervision_used": False,
        "proposal_detector_frozen": True,
        "intervention_mean_abs_x_change_px": {
            name: sum(values) / max(len(values), 1)
            for name, values in intervention_changes.items()
        },
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
