from __future__ import annotations

import argparse
from collections import OrderedDict
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
    "v7_anchor",
    "v13_final",
    "wrong_p2_v13_final",
    "zero_p2_v13_final",
    "v13_x_v7_range",
    "v7_x_v13_range",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Official held-out replay for V13 direct visual-precision geometry."
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


def _config(args: argparse.Namespace) -> dict[str, Any]:
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


def _required(outputs: dict[str, Any], name: str) -> torch.Tensor:
    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"required V13 output is unavailable: {name}")
    return value


def _deltas(
    metrics: dict[str, Any], thresholds: tuple[float, ...]
) -> dict[str, Any]:
    comparisons = OrderedDict(
        (
            ("v13_minus_v7", ("v13_final", "v7_anchor")),
            (
                "correct_minus_wrong_p2",
                ("v13_final", "wrong_p2_v13_final"),
            ),
            (
                "correct_minus_zero_p2",
                ("v13_final", "zero_p2_v13_final"),
            ),
            ("v13_x_only_minus_v7", ("v13_x_v7_range", "v7_anchor")),
            ("v13_range_only_minus_v7", ("v7_x_v13_range", "v7_anchor")),
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
    cfg = _config(args)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    selector = model.structured_query_head.set_selection_head
    visual = selector.visual_first_association
    precision = selector.visual_precision_geometry
    if visual is None or precision is None:
        raise ValueError("V13 official audit requires V12 and V13 modules")

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
        tqdm(loader, desc="V13 official replay", ncols=90)
    ):
        images = images.to(device, non_blocking=True)
        v12_kwargs: dict[str, torch.Tensor] = {}
        v13_kwargs: dict[str, torch.Tensor] = {}

        def capture_v12(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            v12_kwargs.update(kwargs)

        def capture_v13(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            v13_kwargs.update(kwargs)

        h12 = visual.register_forward_pre_hook(capture_v12, with_kwargs=True)
        h13 = precision.register_forward_pre_hook(capture_v13, with_kwargs=True)
        outputs = model(images)
        h12.remove()
        h13.remove()
        if not v12_kwargs or not v13_kwargs:
            raise RuntimeError("V13 official audit captured incomplete inputs")

        variants: dict[str, dict[str, torch.Tensor]] = {}
        for name, feature in (
            (
                "wrong",
                torch.roll(v12_kwargs["row_value_features"], 1, dims=0),
            ),
            ("zero", torch.zeros_like(v12_kwargs["row_value_features"])),
        ):
            changed_v12 = dict(v12_kwargs)
            changed_v12["row_value_features"] = feature
            visual_output = visual(**changed_v12)
            changed_v13 = dict(v13_kwargs)
            changed_v13["visual_state"] = visual_output[
                "selection_slot_v12_visual_state"
            ]
            changed_v13["visual_x_rows"] = visual_output[
                "selection_slot_v12_visual_x_rows"
            ]
            changed_v13["row_value_features"] = feature
            variants[name] = precision(**changed_v13)

        anchor_x = _required(outputs, "selection_slot_v13_anchor_x_rows")
        anchor_range = _required(
            outputs, "selection_slot_v13_anchor_range_norm"
        )
        final_x = _required(outputs, "selection_slot_pred_x_rows")
        final_range = _required(outputs, "selection_slot_range_norm")
        active = _required(outputs, "selection_slot_active").bool()
        policies = OrderedDict(
            (
                ("v7_anchor", (anchor_x, anchor_range)),
                ("v13_final", (final_x, final_range)),
                (
                    "wrong_p2_v13_final",
                    (
                        variants["wrong"]["selection_slot_pred_x_rows"],
                        variants["wrong"]["selection_slot_range_norm"],
                    ),
                ),
                (
                    "zero_p2_v13_final",
                    (
                        variants["zero"]["selection_slot_pred_x_rows"],
                        variants["zero"]["selection_slot_range_norm"],
                    ),
                ),
                ("v13_x_v7_range", (final_x, anchor_range)),
                ("v7_x_v13_range", (anchor_x, final_range)),
            )
        )
        take = len(metas)
        if int(args.max_images) > 0:
            take = min(take, int(args.max_images) - image_count)
        for bi, meta in enumerate(metas[:take]):
            geometry: list[torch.Tensor] = []
            ranges: list[torch.Tensor] = []
            layout: dict[str, tuple[int, int]] = {}
            active_by_policy: dict[str, torch.Tensor] = {}
            cursor = 0
            for name, (x, lane_range) in policies.items():
                geometry.append(x[bi].detach().float().cpu())
                ranges.append(lane_range[bi].detach().float().cpu())
                count = int(x.shape[1])
                layout[name] = (cursor, cursor + count)
                active_by_policy[name] = active[bi].detach().cpu()
                cursor += count
            records.append(
                {
                    "image_id": _image_id(
                        meta, f"v13_{batch_index:06d}_{bi}"
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
        "experiment": "V13 visual-precision official replay",
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "iteration": iteration,
        "split": args.split,
        "list_path": str(list_path),
        "list_sha256": sha256_file(list_path),
        "sample_strategy": args.sample_strategy,
        "sampled_indices": sampled_indices[: len(records)],
        "images": len(records),
        "thresholds": list(thresholds),
        "line_width": float(args.line_width),
        "min_valid_rows": int(args.min_valid_rows),
        "hard_proposal_id_produces_final_geometry": False,
        "activity_and_score_source": "exact_v7",
        "metrics": metrics,
        "deltas": _deltas(metrics, thresholds),
        "optimizer_steps": 0,
        "test_set_used": False,
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_json": str(destination.resolve()),
                "iteration": iteration,
                "images": len(records),
                "writer_valid": {
                    name: metrics[name]["writer_valid"]["thresholds"]
                    for name in POLICIES
                },
                "deltas": report["deltas"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
