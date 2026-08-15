from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import cv2
import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    official_proposal_gt_iou_matrix,
    sha256_file,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.v20_slot_owned_replacement import (
    _gather_candidate,
    complete_curve_relations,
)
from dynlaneseq_eg.tools.audit_v11_causal_replay import (
    _image_id,
    _plain_meta,
)
from dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_official import (
    _required,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v20_replacement_targets import (
    build_one_edit_action_targets,
)


FEATURE_FIELDS = (
    "candidate_state",
    "p50",
    "p75",
    "expected_iou",
    "legacy_route_logits",
    "curve_relations",
    "counterfactual_valid",
    "source_route",
    "source_active",
)
TARGET_FIELDS = (
    "action_valid",
    "full_action_valid",
    "delta50_class",
    "delta75_class",
    "delta_iou",
    "duplicate",
    "abandon",
    "policy_target",
    "ownership",
    "source_tp50",
    "source_tp75",
    "best_one_edit_tp50",
    "best_one_edit_tp75",
    "gt_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache immutable V7/V19 features and exact official-raster one-edit labels."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "Reuse already written exact-raster shards, replay only the cheap "
            "zero-step model contract, and write the missing manifest."
        ),
    )
    return parser.parse_args()


def _config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})[args.split] = str(
        Path(args.list_path).expanduser().resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    augmentation = cfg.get("augmentation", {})
    expected = {
        "horizontal_flip_prob": 0.0,
        "color_jitter": False,
        "channel_shuffle_prob": 0.0,
        "hue_saturation_prob": 0.0,
        "blur_prob": 0.0,
        "affine_prob": 0.0,
        "affine_translate_x": 0.0,
        "affine_translate_y": 0.0,
        "affine_rotate_deg": 0.0,
        "affine_scale_min": 1.0,
        "affine_scale_max": 1.0,
        "random_shadow_prob": 0.0,
    }
    if any(augmentation.get(name) != value for name, value in expected.items()):
        raise ValueError("V20 feature/label cache requires exact zero augmentation")
    return cfg


def _to_cache_dtype(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    value = value.detach().cpu().contiguous()
    return value.to(dtype=dtype) if value.is_floating_point() else value


def _label_record(
    raw: dict[str, Any],
    policy_iou_support_delta: float,
) -> dict[str, Any]:
    slots, candidates, rows = raw["counterfactual_x"].shape
    record = {
        "meta": raw["meta"],
        "stages": {
            "counterfactual": {
                "pred_x_rows": raw["counterfactual_x"].reshape(
                    slots * candidates, rows
                ),
                "range_norm": raw["counterfactual_range"].reshape(
                    slots * candidates, 2
                ),
            }
        },
    }
    quality_flat, raster_valid = official_proposal_gt_iou_matrix(
        record,
        "counterfactual",
        line_width=30.0,
        min_valid_rows=5,
        row_visibility_thresh=0.0,
    )
    quality = quality_flat.reshape(-1, slots, candidates)
    valid = raw["counterfactual_valid"].bool() & raster_valid.reshape(
        slots, candidates
    )
    targets = build_one_edit_action_targets(
        quality,
        valid,
        raw["source_route"],
        raw["source_active"],
        policy_iou_support_delta=float(policy_iou_support_delta),
    )
    source_x = _gather_candidate(
        raw["counterfactual_x"].unsqueeze(0),
        raw["source_route"].unsqueeze(0),
    )
    source_range = _gather_candidate(
        raw["counterfactual_range"].unsqueeze(0),
        raw["source_route"].unsqueeze(0),
    )
    relations = complete_curve_relations(
        raw["counterfactual_x"].unsqueeze(0),
        raw["counterfactual_range"].unsqueeze(0),
        source_x,
        source_range,
        input_w=int(raw["meta"].get("input_w", 1600)),
    ).squeeze(0)
    return {
        **raw,
        "curve_relations": relations,
        **{name: targets[name] for name in TARGET_FIELDS},
    }


def _label_record_process(
    payload: tuple[dict[str, Any], float],
) -> dict[str, Any]:
    """Process-isolated exact raster labeling.

    SciPy/FITPACK's spline interpolation serializes competing Python threads.
    V20 has 128 counterfactual curves per image, so the former thread pool
    behaved almost exactly like one worker.  Spawned processes preserve the
    evaluator bit path while allowing independent images to use separate CPU
    cores.  Explicit one-thread settings prevent nested BLAS/OpenCV
    oversubscription.
    """

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    raw, policy_iou_support_delta = payload
    return _label_record(raw, policy_iou_support_delta)


def _write_shard(
    records: list[dict[str, Any]],
    output: Path,
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_ids": [str(record["image_id"]) for record in records],
    }
    for name in FEATURE_FIELDS + TARGET_FIELDS:
        payload[name] = torch.stack(
            [_to_cache_dtype(record[name], dtype) for record in records]
        )
    torch.save(payload, output)
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "images": len(records),
    }


def _validated_existing_shards(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    paths = sorted(output_dir.glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"no existing exact-raster shards found in {output_dir}"
        )
    shards: list[dict[str, Any]] = []
    image_ids: list[str] = []
    for index, path in enumerate(paths):
        expected_name = f"shard_{index:04d}.pt"
        if path.name != expected_name:
            raise ValueError(
                f"non-contiguous cache shards: expected {expected_name}, "
                f"found {path.name}"
            )
        payload = torch.load(path, map_location="cpu")
        shard_ids = [str(value) for value in payload.get("image_ids", [])]
        if not shard_ids:
            raise ValueError(f"empty or missing image_ids in {path}")
        for name in FEATURE_FIELDS + TARGET_FIELDS:
            value = payload.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"missing tensor {name} in {path}")
            if int(value.shape[0]) != len(shard_ids):
                raise ValueError(
                    f"cache batch mismatch for {name} in {path}: "
                    f"{value.shape[0]} vs {len(shard_ids)}"
                )
        image_ids.extend(shard_ids)
        shards.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "images": len(shard_ids),
            }
        )
    return shards, image_ids


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = _config(args)
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(manifest_path.read_text(encoding="utf-8"))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("shard_*.pt"):
        if args.overwrite and not args.finalize_existing:
            stale.unlink()

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    selector = model.structured_query_head.set_selection_head
    if selector.counterfactual_fidelity is None:
        raise ValueError("V20 cache requires loaded V19 fidelity")
    if selector.slot_owned_safe_replacement is None:
        raise ValueError("V20 cache requires the safe replacement module")
    loader = build_dataloader(cfg, split=args.split, training=False)
    dtype_name = str(cfg.get("v20", {}).get("cache_dtype", "float16"))
    dtype = {"float16": torch.float16, "float32": torch.float32}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported V20 cache dtype: {dtype_name}")
    shard_size = max(int(args.shard_size), 1)
    policy_delta = float(
        cfg.get("v20", {}).get("policy_iou_support_delta", 0.01)
    )
    workers = max(int(args.metric_workers), 0)
    raw_buffer: list[dict[str, Any]] = []
    if args.finalize_existing:
        shards, cached_image_ids = _validated_existing_shards(output_dir)
    else:
        shards, cached_image_ids = [], []
    replayed_image_ids: list[str] = []
    contract = {
        "nonzero_zero_step_edits": 0,
        "route_mismatch": 0,
        "active_mismatch": 0,
        "cache_population_order_mismatch": 0,
    }

    metric_executor = (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        )
        if workers > 1 and not args.finalize_existing
        else None
    )

    def flush() -> None:
        if not raw_buffer:
            return
        previous = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            if metric_executor is not None:
                labeled = list(
                    metric_executor.map(
                        _label_record_process,
                        ((record, policy_delta) for record in raw_buffer),
                        chunksize=1,
                    )
                )
            else:
                labeled = [
                    _label_record(record, policy_delta) for record in raw_buffer
                ]
        finally:
            cv2.setNumThreads(previous)
        shard_path = output_dir / f"shard_{len(shards):04d}.pt"
        shards.append(_write_shard(labeled, shard_path, dtype=dtype))
        raw_buffer.clear()

    try:
        description = (
            "V20 cache zero-step finalization"
            if args.finalize_existing
            else "V20 frozen feature cache"
        )
        for batch_index, (images, _targets, metas) in enumerate(
            tqdm(loader, desc=description, ncols=90)
        ):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            source_route = _required(
                outputs, "selection_slot_v20_v7_geometry_route_indices"
            )
            source_active = _required(
                outputs, "selection_slot_v20_v7_active"
            ).bool()
            deployed_route = _required(
                outputs, "selection_slot_geometry_route_indices"
            )
            deployed_active = _required(outputs, "selection_slot_active").bool()
            contract["nonzero_zero_step_edits"] += int(
                _required(outputs, "selection_slot_v20_edit_count").sum().cpu()
            )
            contract["route_mismatch"] += int(
                (source_route != deployed_route).sum().cpu()
            )
            contract["active_mismatch"] += int(
                (source_active != deployed_active).sum().cpu()
            )
            if args.finalize_existing:
                replayed_image_ids.extend(
                    _image_id(meta, f"v20_cache_{batch_index:06d}_{item}")
                    for item, meta in enumerate(metas)
                )
                continue
            for item, meta in enumerate(metas):
                raw_buffer.append(
                    {
                        "image_id": _image_id(
                            meta, f"v20_cache_{batch_index:06d}_{item}"
                        ),
                        "meta": _plain_meta(meta),
                        "candidate_state": _required(
                            outputs, "selection_slot_v19_candidate_state"
                        )[item].float().cpu(),
                        "p50": _required(outputs, "selection_slot_v19_p50")[
                            item
                        ].float().cpu(),
                        "p75": _required(outputs, "selection_slot_v19_p75")[
                            item
                        ].float().cpu(),
                        "expected_iou": _required(
                            outputs, "selection_slot_v19_expected_iou"
                        )[item].float().cpu(),
                        "legacy_route_logits": _required(
                            outputs, "selection_slot_v19_v7_real_route_logits"
                        )[item].float().cpu(),
                        "counterfactual_x": _required(
                            outputs, "selection_slot_v19_counterfactual_x_rows"
                        )[item].float().cpu(),
                        "counterfactual_range": _required(
                            outputs,
                            "selection_slot_v19_counterfactual_range_norm",
                        )[item].float().cpu(),
                        "counterfactual_valid": _required(
                            outputs, "selection_slot_v19_counterfactual_valid"
                        )[item].bool().cpu(),
                        "source_route": source_route[item].long().cpu(),
                        "source_active": source_active[item].bool().cpu(),
                    }
                )
                if len(raw_buffer) >= shard_size:
                    flush()
        if args.finalize_existing:
            contract["cache_population_order_mismatch"] = int(
                replayed_image_ids != cached_image_ids
            )
        else:
            flush()
    finally:
        if metric_executor is not None:
            metric_executor.shutdown(wait=True)
    passed = all(int(value) == 0 for value in contract.values())
    manifest = {
        "experiment": "V20 frozen exact-raster slot-owned replacement cache",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "iteration": iteration,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": args.split,
        "list_path": str(Path(args.list_path).resolve()),
        "list_sha256": sha256_file(args.list_path),
        "images": sum(int(item["images"]) for item in shards),
        "existing_shards_reused": bool(args.finalize_existing),
        "cache_dtype": dtype_name,
        "augmentation": "exactly_disabled",
        "official_raster": {
            "line_width": 30.0,
            "thresholds": [0.50, 0.75],
            "min_valid_rows": 5,
        },
        "policy_iou_support_delta": policy_delta,
        "contract": {**contract, "passed": passed},
        "shards": shards,
        "test_set_used": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("V20 zero-step cache parity failed")


if __name__ == "__main__":
    main()
