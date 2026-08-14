from __future__ import annotations

import argparse
from collections import Counter
import gc
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
from dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_official import (
    _required,
)
from dynlaneseq_eg.tools.train import seed_everything


POLICIES = ("source_v7", "v20_deployed", "v20_context_masked")
THRESHOLDS = (0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact official-raster V20 fixed endpoint audit."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _config(path: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})[args.split] = str(
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


def _coverage(
    quality: torch.Tensor,
    valid: torch.Tensor,
    start: int,
    stop: int,
    active: torch.Tensor,
    threshold: float,
) -> tuple[int, set[int]]:
    local = quality[:, start:stop]
    local_valid = valid[start:stop]
    selected = torch.nonzero(active.bool() & local_valid.bool(), as_tuple=False).flatten().tolist()
    assignment = evaluator_hungarian_assignment(
        local, selected, threshold=float(threshold)
    )
    return int(assignment.hit_count), {int(gt) for gt, _pred in assignment.pairs}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = _config(args.config, args)
    source_cfg = _config(args.source_config, args)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    source_model = build_model(source_cfg).to(device)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source_model, strict=False)
    )
    source_model.eval()
    loader = build_dataloader(cfg, split=args.split, training=False)
    head = model.structured_query_head.set_selection_head.slot_owned_safe_replacement
    if head is None:
        raise ValueError("V20 endpoint has no replacement head")
    normal_mode = head.context_mode

    records: list[dict[str, Any]] = []
    parity = {
        "proposal_x_max_difference": 0.0,
        "proposal_range_max_difference": 0.0,
        "active_mask_mismatch": 0,
        "prediction_count_mismatch": 0,
        "images": 0,
        "edited_images": 0,
    }
    for batch_index, (images, _targets, metas) in enumerate(
        tqdm(loader, desc="V20 fixed endpoint forward", ncols=90)
    ):
        images = images.to(device, non_blocking=True)
        source = source_model(images)
        head.context_mode = normal_mode
        endpoint = model(images)
        head.context_mode = "masked"
        masked = model(images)
        head.context_mode = normal_mode
        parity["proposal_x_max_difference"] = max(
            float(parity["proposal_x_max_difference"]),
            float(
                (
                    _required(endpoint, "pred_x_rows")
                    - _required(source, "pred_x_rows")
                ).abs().max().cpu()
            ),
        )
        parity["proposal_range_max_difference"] = max(
            float(parity["proposal_range_max_difference"]),
            float(
                (
                    _required(endpoint, "range_norm")
                    - _required(source, "range_norm")
                ).abs().max().cpu()
            ),
        )
        source_active = _required(source, "selection_slot_active").bool()
        endpoint_active = _required(endpoint, "selection_slot_active").bool()
        masked_active = _required(masked, "selection_slot_active").bool()
        parity["active_mask_mismatch"] += int(
            (source_active != endpoint_active).sum().cpu()
        )
        parity["prediction_count_mismatch"] += int(
            (
                source_active.sum(dim=-1) != endpoint_active.sum(dim=-1)
            ).sum().cpu()
        )
        parity["images"] += int(images.shape[0])
        parity["edited_images"] += int(
            _required(endpoint, "selection_slot_v20_edit_count").sum().cpu()
        )
        for item, meta in enumerate(metas):
            geometries = []
            ranges = []
            layouts: dict[str, tuple[int, int]] = {}
            actives: dict[str, torch.Tensor] = {}
            cursor = 0
            for name, output, active in (
                ("source_v7", source, source_active[item]),
                ("v20_deployed", endpoint, endpoint_active[item]),
                ("v20_context_masked", masked, masked_active[item]),
            ):
                x = _required(output, "selection_slot_pred_x_rows")[item]
                rho = _required(output, "selection_slot_range_norm")[item]
                geometries.append(x.detach().float().cpu())
                ranges.append(rho.detach().float().cpu())
                layouts[name] = (cursor, cursor + int(x.shape[0]))
                cursor += int(x.shape[0])
                actives[name] = active.detach().bool().cpu()
            records.append(
                {
                    "image_id": _image_id(
                        meta, f"v20_official_{batch_index:06d}_{item}"
                    ),
                    "record": {
                        "meta": _plain_meta(meta),
                        "stages": {
                            "combined": {
                                "pred_x_rows": torch.cat(geometries, dim=0),
                                "range_norm": torch.cat(ranges, dim=0),
                            }
                        },
                    },
                    "layout": layouts,
                    "active": actives,
                    "edit": int(
                        _required(endpoint, "selection_slot_v20_edit_count")[
                            item
                        ].cpu()
                    ),
                    "replace_slot": int(
                        _required(endpoint, "selection_slot_v20_replace_slot")[
                            item
                        ].cpu()
                    ),
                    "replace_candidate": int(
                        _required(
                            endpoint, "selection_slot_v20_replace_candidate"
                        )[item].cpu()
                    ),
                }
            )

    del model, source_model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluated = _evaluate_records(
        records,
        line_width=30.0,
        min_valid_rows=5,
        workers=int(args.metric_workers),
    )
    tree = _new_metric_tree(POLICIES, THRESHOLDS)
    edits = Counter()
    degradation = Counter()
    for item, result in zip(records, evaluated):
        _accumulate_record(
            tree,
            result,
            item["layout"],
            item["active"],
            THRESHOLDS,
        )
        quality = result["quality"].float()
        valid = result["valid"].bool()
        lex_delta: tuple[int, int] | None = None
        for threshold in THRESHOLDS:
            source_tp, source_gt = _coverage(
                quality,
                valid,
                *item["layout"]["source_v7"],
                item["active"]["source_v7"],
                threshold,
            )
            endpoint_tp, endpoint_gt = _coverage(
                quality,
                valid,
                *item["layout"]["v20_deployed"],
                item["active"]["v20_deployed"],
                threshold,
            )
            delta = endpoint_tp - source_tp
            degradation[f"abandoned_gt_{threshold:.2f}"] += len(
                source_gt - endpoint_gt
            )
            degradation[f"source_tp_{threshold:.2f}"] += source_tp
            degradation[f"tp_delta_{threshold:.2f}"] += delta
            if threshold == 0.50:
                lex_delta = (delta, 0)
            else:
                lex_delta = (int(lex_delta[0]), delta)
        if int(item["edit"]):
            edits["selected"] += 1
            if lex_delta is not None and (
                lex_delta[0] > 0 or (lex_delta[0] == 0 and lex_delta[1] > 0)
            ):
                edits["beneficial"] += 1
            elif lex_delta is not None and (
                lex_delta[0] < 0 or (lex_delta[0] == 0 and lex_delta[1] < 0)
            ):
                edits["harmful"] += 1
            else:
                edits["neutral"] += 1
        else:
            edits["keep"] += 1
    metrics = _finalize_metric_tree(tree)
    comparisons = {
        "v20_minus_v7": {
            mode: {
                f"{threshold:.2f}": _metric_delta(
                    metrics,
                    "v20_deployed",
                    "source_v7",
                    mode,
                    f"{threshold:.2f}",
                )
                for threshold in THRESHOLDS
            }
            for mode in ("neural_active", "writer_valid")
        },
        "correct_context_minus_masked_context": {
            mode: {
                f"{threshold:.2f}": _metric_delta(
                    metrics,
                    "v20_deployed",
                    "v20_context_masked",
                    mode,
                    f"{threshold:.2f}",
                )
                for threshold in THRESHOLDS
            }
            for mode in ("neural_active", "writer_valid")
        },
    }
    selected = int(edits["selected"])
    report = {
        "experiment": "V20 slot-owned safe replacement official fixed endpoint",
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "iteration": iteration,
        "source_iteration": source_iteration,
        "split": args.split,
        "list_path": str(Path(args.list_path).resolve()),
        "list_sha256": sha256_file(args.list_path),
        "images": len(records),
        "configured_context_mode": normal_mode,
        "parity": {
            **parity,
            "passed": (
                float(parity["proposal_x_max_difference"]) == 0.0
                and float(parity["proposal_range_max_difference"]) == 0.0
                and int(parity["active_mask_mismatch"]) == 0
                and int(parity["prediction_count_mismatch"]) == 0
            ),
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "replacement": {
            **dict(edits),
            "precision": int(edits["beneficial"]) / max(selected, 1),
            "selection_rate": selected / max(len(records), 1),
        },
        "degradation": {
            **dict(degradation),
            "source_correct_loss_fraction_50": int(
                degradation["abandoned_gt_0.50"]
            )
            / max(int(degradation["source_tp_0.50"]), 1),
            "source_correct_loss_fraction_75": int(
                degradation["abandoned_gt_0.75"]
            )
            / max(int(degradation["source_tp_0.75"]), 1),
        },
        "checkpoint_selection_performed": False,
        "threshold_search_performed": False,
        "nms_search_performed": False,
        "full_validation_executed": False,
        "test_set_used": False,
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

