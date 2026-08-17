from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_metric import load_culane_img_data
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_s0 import DynLaneSeqS0
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.modeling.v23_ordered_slot_cost_volume import canonicalize_v7_slots
from dynlaneseq_eg.modeling.v25_dual_energy_multi_path import diverse_viterbi_paths
from dynlaneseq_eg.modeling.v25_s1_immutable_selector import build_selector_features
from dynlaneseq_eg.tools.evaluate_v25_g1b_multi_path_capacity import (
    _model_lane_to_original,
)
from dynlaneseq_eg.tools.evaluate_v25_v7_top3_union_oracle import (
    _quantize_writer_lane,
    official_iou_matrix,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v25_s1_single_edit_labels import (
    ALTERNATIVES,
    SLOTS,
    score_single_edit_iou_bank,
)


ROWS = 160
ACTIONS = SLOTS * ALTERNATIVES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache immutable exact-V7/G0-top3 banks, path-native selector "
            "features, and exact official-raster single-edit labels."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--g0-config", required=True)
    parser.add_argument("--g0-checkpoint", required=True)
    parser.add_argument("--v7-config", required=True)
    parser.add_argument("--v7-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wrong-image-list", default="")
    parser.add_argument("--retain-bank", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--label-workers", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _configured(
    config_path: str,
    *,
    root: Path,
    list_path: Path,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(config_path))
    cfg.setdefault("dataset", {})["root"] = str(root)
    cfg["dataset"].setdefault("lists", {})["val"] = str(list_path)
    cfg["dataset"]["load_targets"] = False
    cfg["dataset"]["infer_seg_labels"] = False
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(workers)
    cfg["dataloader"]["persistent_workers"] = workers > 0
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _open_arrays(root: Path, count: int, *, wrong: bool, retain_bank: bool):
    root.mkdir(parents=True, exist_ok=True)
    specifications: dict[str, tuple[Any, tuple[int, ...]]] = {
        "row_evidence": (np.float16, (count, ACTIONS, ROWS, 6)),
        "row_geometry": (np.float16, (count, ACTIONS, ROWS, 6)),
        "global_evidence": (np.float16, (count, ACTIONS, 26)),
        "global_geometry": (np.float16, (count, ACTIONS, 38)),
        "action_valid": (np.bool_, (count, ACTIONS)),
        "action_outcome": (np.uint8, (count, ACTIONS)),
        "target_action": (np.uint8, (count,)),
        "raw_oracle_action": (np.uint8, (count,)),
        "action_tp50": (np.int8, (count, ACTIONS)),
        "action_tp75": (np.int8, (count, ACTIONS)),
        "action_lost50": (np.int8, (count, ACTIONS)),
        "action_lost75": (np.int8, (count, ACTIONS)),
        "source_tp50": (np.int8, (count,)),
        "source_tp75": (np.int8, (count,)),
        "source_prediction_count": (np.int8, (count,)),
        "gt_count": (np.int8, (count,)),
    }
    if wrong:
        specifications.update(
            {
                "wrong_row_evidence": (np.float16, (count, ACTIONS, ROWS, 6)),
                "wrong_global_evidence": (np.float16, (count, ACTIONS, 26)),
            }
        )
    if retain_bank:
        specifications.update(
            {
                "source_x": (np.float32, (count, SLOTS, ROWS)),
                "source_range": (np.float32, (count, SLOTS, 2)),
                "source_active": (np.bool_, (count, SLOTS)),
                "hypotheses": (np.float32, (count, SLOTS, ALTERNATIVES, ROWS)),
                "hypothesis_range": (np.float32, (count, SLOTS, 2)),
            }
        )
    return {
        name: np.lib.format.open_memmap(root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in specifications.items()
    }


def _candidate_valid(ranges: torch.Tensor, hypotheses: torch.Tensor) -> torch.Tensor:
    rows = int(hypotheses.shape[-1])
    fraction = torch.arange(rows, device=ranges.device, dtype=torch.float32) / float(max(rows - 1, 1))
    visible = (
        (fraction.view(1, 1, rows) >= ranges[..., :1])
        & (fraction.view(1, 1, rows) <= ranges[..., 1:])
    )
    base = visible.sum(-1) >= 5
    finite = torch.isfinite(hypotheses).all(-1)
    return base.unsqueeze(-1).expand(-1, -1, ALTERNATIVES) & finite


def _label_task(task):
    source_lanes, candidate_lanes, target_lanes, source_active, candidate_valid = task
    flat: list[list[tuple[float, float]]] = []
    locations: dict[tuple[int, int], int] = {}
    for slot in range(SLOTS):
        if source_active[slot]:
            lane = source_lanes[slot]
            if lane is None:
                raise RuntimeError("active V7 slot serialized to an empty lane")
            locations[(slot, 0)] = len(flat)
            flat.append(lane)
        for path in range(ALTERNATIVES):
            lane = candidate_lanes[slot][path]
            if candidate_valid[slot, path] and lane is not None:
                locations[(slot, path + 1)] = len(flat)
                flat.append(lane)
    matrix = official_iou_matrix(flat, target_lanes)
    bank = np.zeros((SLOTS, 1 + ALTERNATIVES, len(target_lanes)), dtype=np.float32)
    for location, flat_index in locations.items():
        bank[location] = matrix[flat_index]
    return score_single_edit_iou_bank(
        bank,
        source_active=np.asarray(source_active, dtype=bool),
        candidate_valid=np.asarray(candidate_valid, dtype=bool),
    )


def _host(value: torch.Tensor, dtype) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    seed_everything(3407)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    list_path = Path(args.list_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = output_dir / "bank_report.json"
    if report_path.is_file():
        print(json.dumps({"status": "already_complete", "report": str(report_path)}), flush=True)
        return
    rows = _nonempty_lines(list_path)
    count = len(rows)
    if count <= 0:
        raise ValueError("immutable-bank list is empty")
    wrong_path = Path(args.wrong_image_list).expanduser().resolve() if args.wrong_image_list else None
    if wrong_path is not None and len(_nonempty_lines(wrong_path)) != count:
        raise ValueError("wrong-image list must be one-to-one with the primary list")

    cfg = _configured(
        args.g0_config,
        root=dataset_root,
        list_path=list_path,
        batch_size=args.batch_size,
        workers=args.num_workers,
    )
    loader = build_dataloader(cfg, split="val", training=False)
    if len(loader.dataset) != count:
        raise ValueError("bank loader changed the supplied list population")
    wrong_loader = None
    if wrong_path is not None:
        wrong_cfg = _configured(
            args.g0_config,
            root=dataset_root,
            list_path=wrong_path,
            batch_size=args.batch_size,
            workers=args.num_workers,
        )
        wrong_loader = build_dataloader(wrong_cfg, split="val", training=False)

    g0 = build_model(cfg)
    if not isinstance(g0, DynLaneSeqV25):
        raise TypeError("G0 config did not build DynLaneSeqV25")
    g0_iteration = int(load_checkpoint(args.g0_checkpoint, g0, strict=True))
    v7_cfg = _configured(
        args.v7_config,
        root=dataset_root,
        list_path=list_path,
        batch_size=args.batch_size,
        workers=args.num_workers,
    )
    v7 = DynLaneSeqS0(v7_cfg)
    v7_iteration = int(load_checkpoint(args.v7_checkpoint, v7, strict=True))
    v7.prepare_for_inference()

    device = torch.device(args.device)
    channels_last = bool(cfg.get("training", {}).get("channels_last", False))
    g0.requires_grad_(False).eval().to(device)
    v7.requires_grad_(False).eval().to(device)
    if channels_last and device.type == "cuda":
        g0.to(memory_format=torch.channels_last)
        v7.to(memory_format=torch.channels_last)
    arrays = _open_arrays(
        output_dir,
        count,
        wrong=wrong_loader is not None,
        retain_bank=bool(args.retain_bank),
    )
    image_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    label_pool = ThreadPoolExecutor(max_workers=max(1, int(args.label_workers)))
    wrong_iterator = iter(wrong_loader) if wrong_loader is not None else None
    offset = 0
    started = time.perf_counter()
    hist = {"keep": 0, "safe_edit": 0, "raw_edit": 0, "beneficial": 0, "harmful": 0}
    try:
        for batch_index, (images, _targets, metas) in enumerate(loader, start=1):
            batch = int(images.shape[0])
            images = images.to(device, non_blocking=True)
            if channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, enabled=False):
                g0_output = g0(images.float())
                v7_output = v7(images.float(), inference_only=True)
            source = canonicalize_v7_slots(v7_output)
            diverse = diverse_viterbi_paths(
                g0_output["unary_logits"],
                num_hypotheses=ALTERNATIVES,
                transition_radius_bins=g0.detector.transition_radius_bins,
                transition_penalty=g0.detector.transition_penalty,
                suppression_radius_bins=5,
                suppression_penalty=8.0,
            )
            bin_width = float(g0.detector.input_w) / float(g0.detector.x_bins)
            hypotheses = (diverse.indices.float() + 0.5) * bin_width
            candidate_valid = _candidate_valid(g0_output["range_norm"], hypotheses)
            evidence = {
                name: g0_output[name]
                for name in ("unary_logits", "quality50_logits", "quality75_logits", "exist_logits")
            }
            features = build_selector_features(
                source_x=source["x_rows"],
                source_range=source["range_norm"],
                source_active=source["active"],
                hypotheses=hypotheses,
                hypothesis_range=g0_output["range_norm"],
                hypothesis_valid=candidate_valid,
                evidence=evidence,
                input_w=g0.detector.input_w,
            )

            wrong_features = None
            if wrong_iterator is not None:
                wrong_images, _wrong_targets, wrong_metas = next(wrong_iterator)
                if int(wrong_images.shape[0]) != batch:
                    raise RuntimeError("wrong-image batch alignment drifted")
                wrong_images = wrong_images.to(device, non_blocking=True)
                if channels_last and device.type == "cuda":
                    wrong_images = wrong_images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, enabled=False):
                    wrong_output = g0(wrong_images.float())
                wrong_features = build_selector_features(
                    source_x=source["x_rows"],
                    source_range=source["range_norm"],
                    source_active=source["active"],
                    hypotheses=hypotheses,
                    hypothesis_range=g0_output["range_norm"],
                    hypothesis_valid=candidate_valid,
                    evidence={
                        name: wrong_output[name]
                        for name in ("unary_logits", "quality50_logits", "quality75_logits", "exist_logits")
                    },
                    input_w=g0.detector.input_w,
                )
                for meta, wrong_meta in zip(metas, wrong_metas):
                    if str(meta["image_path"]) == str(wrong_meta["image_path"]):
                        raise RuntimeError("wrong-image causal control contains an identical image")

            label_tasks = []
            actual_valid = _host(candidate_valid, np.bool_)
            source_active = _host(source["active"], np.bool_)
            for local, meta in enumerate(metas):
                image_ids.append(str(meta["image_path"]))
                metadata.append(
                    {
                        name: meta[name]
                        for name in (
                            "image_path",
                            "anno_path",
                            "orig_h",
                            "orig_w",
                            "input_h",
                            "input_w",
                            "scale_x",
                            "scale_y",
                            "crop_x",
                            "crop_y",
                        )
                    }
                )
                source_lanes: list[Any] = [None] * SLOTS
                candidates: list[list[Any]] = [[None] * ALTERNATIVES for _ in range(SLOTS)]
                for slot in range(SLOTS):
                    if source_active[local, slot]:
                        lane = _model_lane_to_original(
                            source["x_rows"][local, slot], source["range_norm"][local, slot], meta
                        )
                        source_lanes[slot] = _quantize_writer_lane(lane) if len(lane) >= 2 else None
                    for path in range(ALTERNATIVES):
                        if not actual_valid[local, slot, path]:
                            continue
                        lane = _model_lane_to_original(
                            hypotheses[local, slot, path], g0_output["range_norm"][local, slot], meta
                        )
                        if len(lane) >= 2:
                            candidates[slot][path] = _quantize_writer_lane(lane)
                        else:
                            actual_valid[local, slot, path] = False
                label_tasks.append(
                    (
                        source_lanes,
                        candidates,
                        load_culane_img_data(meta["anno_path"]),
                        source_active[local],
                        actual_valid[local],
                    )
                )
            labels = list(label_pool.map(_label_task, label_tasks))

            sl = slice(offset, offset + batch)
            for name in ("row_evidence", "row_geometry", "global_evidence", "global_geometry"):
                arrays[name][sl] = _host(features[name], np.float16)
            arrays["action_valid"][sl] = np.stack([label["action_valid"] for label in labels])
            arrays["action_outcome"][sl] = np.stack([label["action_outcome"] for label in labels])
            arrays["target_action"][sl] = np.asarray([label["target_action"] for label in labels], dtype=np.uint8)
            arrays["raw_oracle_action"][sl] = np.asarray([label["raw_oracle_action"] for label in labels], dtype=np.uint8)
            for array_name, label_name in (
                ("action_tp50", "tp50"),
                ("action_tp75", "tp75"),
                ("action_lost50", "lost50"),
                ("action_lost75", "lost75"),
            ):
                arrays[array_name][sl] = np.stack([label[label_name] for label in labels])
            arrays["source_tp50"][sl] = np.asarray([label["source_tp50"] for label in labels], dtype=np.int8)
            arrays["source_tp75"][sl] = np.asarray([label["source_tp75"] for label in labels], dtype=np.int8)
            arrays["source_prediction_count"][sl] = source_active.sum(axis=1).astype(np.int8)
            arrays["gt_count"][sl] = np.asarray(
                [len(task[2]) for task in label_tasks], dtype=np.int8
            )
            if wrong_features is not None:
                arrays["wrong_row_evidence"][sl] = _host(wrong_features["row_evidence"], np.float16)
                arrays["wrong_global_evidence"][sl] = _host(wrong_features["global_evidence"], np.float16)
            if args.retain_bank:
                arrays["source_x"][sl] = _host(source["x_rows"], np.float32)
                arrays["source_range"][sl] = _host(source["range_norm"], np.float32)
                arrays["source_active"][sl] = source_active
                arrays["hypotheses"][sl] = _host(hypotheses, np.float32)
                arrays["hypothesis_range"][sl] = _host(g0_output["range_norm"], np.float32)

            target_actions = np.asarray([label["target_action"] for label in labels])
            raw_actions = np.asarray([label["raw_oracle_action"] for label in labels])
            outcomes = np.stack([label["action_outcome"] for label in labels])
            hist["keep"] += int((target_actions == 0).sum())
            hist["safe_edit"] += int((target_actions > 0).sum())
            hist["raw_edit"] += int((raw_actions > 0).sum())
            hist["beneficial"] += int((outcomes == 1).sum())
            hist["harmful"] += int((outcomes == 2).sum())
            offset += batch
            if batch_index == 1 or (args.log_interval > 0 and batch_index % args.log_interval == 0):
                elapsed = max(time.perf_counter() - started, 1.0e-9)
                print(
                    json.dumps(
                        {
                            "phase": "cache_v25_s1_immutable_bank",
                            "images": offset,
                            "images_per_second": offset / elapsed,
                            **hist,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        label_pool.shutdown(wait=True)
    if offset != count or len(image_ids) != count:
        raise RuntimeError("immutable-bank cache did not consume the exact supplied population")
    for value in arrays.values():
        value.flush()
    (output_dir / "image_ids.json").write_text(json.dumps(image_ids) + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    report = {
        "experiment": "V25-S1 immutable exact-V7/G0-top3 single-edit bank",
        "population": {"list": str(list_path), "rows": count, "list_sha256": sha256_file(list_path)},
        "checkpoints": {
            "v7": str(Path(args.v7_checkpoint).resolve()),
            "v7_sha256": sha256_file(args.v7_checkpoint),
            "v7_iteration": v7_iteration,
            "g0": str(Path(args.g0_checkpoint).resolve()),
            "g0_sha256": sha256_file(args.g0_checkpoint),
            "g0_iteration": g0_iteration,
        },
        "histogram": hist,
        "contract": {
            "members": ["exact_v7", "g0_path0", "g0_path1", "g0_path2"],
            "actions": "KEEP or exactly one of 12 immutable replacements",
            "coordinate_blending": False,
            "geometry_mutation": False,
            "activity_count": "exact V7",
            "source_on_threshold_tie": True,
            "safe_target_forbids_loss_of_v7_correct_gt": True,
            "gt_used_only_for_offline_training_labels": True,
            "test_set_used": False,
        },
        "wrong_image_evidence_cached": wrong_loader is not None,
        "bank_geometry_retained": bool(args.retain_bank),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(report_path), **hist}, indent=2), flush=True)


if __name__ == "__main__":
    main()
