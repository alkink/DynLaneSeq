from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_compatible_model_weights
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.losses.loss_s0 import build_four_slot_cluster_targets
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    _frozen_outputs,
)
from dynlaneseq_eg.tools.train import seed_everything


TARGET_CONTRACTS: dict[str, dict[str, float]] = {
    # Failed V6-A copied the official-IoU threshold into row-strip space.
    "literal_official_threshold": {
        "representable_min": 0.50,
        "cluster_min": 0.30,
        "cluster_delta": 0.05,
        "temperature": 0.03,
    },
    # Narrow candidate support while using the previously calibrated
    # official-.50 representability boundary in row-strip space.
    "calibrated_narrow": {
        "representable_min": 0.20,
        "cluster_min": 0.20,
        "cluster_delta": 0.05,
        "temperature": 0.03,
    },
    # Existing V4.5 target contract, retained as a known production control.
    "calibrated_v4_5": {
        "representable_min": 0.20,
        "cluster_min": 0.20,
        "cluster_delta": 0.10,
        "temperature": 0.03,
    },
    "intermediate_narrow": {
        "representable_min": 0.30,
        "cluster_min": 0.20,
        "cluster_delta": 0.05,
        "temperature": 0.03,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _amp_context(device: torch.device, name: str):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name)
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure four-slot row-strip target cardinality under clean and "
            "augmented frozen V5.1 proposal distributions without training."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _new_accumulator() -> dict[str, Any]:
    return {
        "images": 0,
        "gt_lanes": 0.0,
        "representable": 0.0,
        "support_weighted_sum": 0.0,
        "entropy_weighted_sum": 0.0,
        "quality_weighted_sum": 0.0,
        "count_histogram": Counter(),
    }


def _finish_accumulator(value: dict[str, Any], *, slots: int) -> dict[str, Any]:
    images = int(value["images"])
    representable = float(value["representable"])
    gt_lanes = float(value["gt_lanes"])
    slot_total = float(max(images * int(slots), 1))
    cluster_denominator = max(representable, 1.0)
    return {
        "images": images,
        "gt_lanes": int(round(gt_lanes)),
        "mean_gt_lanes": gt_lanes / max(images, 1),
        "representable_lanes": int(round(representable)),
        "mean_representable_lanes": representable / max(images, 1),
        "representable_gt_fraction": representable / max(gt_lanes, 1.0),
        "expected_dustbin_fraction": (
            slot_total - representable
        )
        / slot_total,
        "mean_support_size": float(value["support_weighted_sum"])
        / cluster_denominator,
        "mean_target_entropy": float(value["entropy_weighted_sum"])
        / cluster_denominator,
        "mean_target_quality": float(value["quality_weighted_sum"])
        / cluster_denominator,
        "representable_count_histogram": {
            str(index): int(value["count_histogram"].get(index, 0))
            for index in range(int(slots) + 1)
        },
    }


@torch.no_grad()
def _audit_mode(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    split: str,
    augmented: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[int]]:
    seed_everything(int(args.seed))
    loader = build_dataloader(cfg, split=split, training=bool(augmented))
    loader, indices = select_diagnostic_loader(
        loader,
        strategy="uniform",
        max_batches=int(args.max_batches),
        num_workers=int(args.num_workers),
    )
    accumulators = {
        name: _new_accumulator() for name in TARGET_CONTRACTS
    }
    loss_cfg = cfg.get("loss", {})
    selection_cfg = (
        cfg.get("model", {})
        .get("structured_query", {})
        .get("set_selection", {})
    )
    slots = int(selection_cfg.get("four_slot_num_slots", 4))
    min_valid_rows = int(loss_cfg.get("four_slot_min_valid_rows", 5))
    line_width = float(loss_cfg.get("four_slot_line_width", 30.0))
    input_h = int(cfg.get("model", {}).get("input_h", 640))
    channels_last = bool(cfg.get("training", {}).get("channels_last", True))
    description = f"V6-A targets {split} {'aug' if augmented else 'clean'}"
    for images, targets, _metas in tqdm(loader, desc=description, ncols=92):
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        targets = nested_to_device(targets, device)
        with _amp_context(device, args.amp_dtype):
            outputs = _frozen_outputs(model, images)
        batch_size = int(images.shape[0])
        gt_count = sum(
            int(
                (
                    target["valid_mask"].bool().sum(dim=-1)
                    >= min_valid_rows
                ).sum()
            )
            for target in targets
        )
        for name, contract in TARGET_CONTRACTS.items():
            built = build_four_slot_cluster_targets(
                outputs,
                targets,
                num_slots=slots,
                input_h=input_h,
                line_width=line_width,
                min_valid_rows=min_valid_rows,
                representable_min=float(contract["representable_min"]),
                cluster_min=float(contract["cluster_min"]),
                cluster_delta=float(contract["cluster_delta"]),
                temperature=float(contract["temperature"]),
            )
            counts = built["representable_count"].detach().float().cpu()
            cluster_count = float(counts.sum())
            accumulator = accumulators[name]
            accumulator["images"] += batch_size
            accumulator["gt_lanes"] += float(gt_count)
            accumulator["representable"] += cluster_count
            accumulator["support_weighted_sum"] += float(
                built["mean_support_size"].detach().float().cpu()
            ) * cluster_count
            accumulator["entropy_weighted_sum"] += float(
                built["mean_entropy"].detach().float().cpu()
            ) * cluster_count
            accumulator["quality_weighted_sum"] += float(
                built["mean_target_quality"].detach().float().cpu()
            ) * cluster_count
            accumulator["count_histogram"].update(
                int(value) for value in counts.tolist()
            )
    return (
        {
            name: {
                "parameters": TARGET_CONTRACTS[name],
                **_finish_accumulator(accumulator, slots=slots),
            }
            for name, accumulator in accumulators.items()
        },
        indices,
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_stats = load_compatible_model_weights(checkpoint_path, model)
    # Target statistics require the frozen proposal/ownership tensors but not
    # a randomly initialized V6 selector forward.
    model.structured_query_head.set_selection_head = None
    model.eval()

    modes: dict[str, dict[str, Any]] = {}
    mode_indices: dict[str, list[int]] = {}
    for name, split, augmented in (
        ("train_clean", "train", False),
        ("train_augmented", "train", True),
        ("val_clean", "val", False),
    ):
        values, indices = _audit_mode(
            model,
            cfg,
            split=split,
            augmented=augmented,
            args=args,
            device=device,
        )
        modes[name] = values
        mode_indices[name] = indices
    same_train_indices = mode_indices["train_clean"] == mode_indices[
        "train_augmented"
    ]
    report = {
        "experiment": "V6-A row-strip target distribution audit",
        "diagnostic_only": True,
        "training_started": False,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "load_stats": load_stats,
        "sample": {
            "batch_size": int(args.batch_size),
            "max_batches": int(args.max_batches),
            "seed": int(args.seed),
            "same_uniform_train_indices": bool(same_train_indices),
            "train_indices": mode_indices["train_clean"],
            "val_indices": mode_indices["val_clean"],
        },
        "augmentation": cfg.get("augmentation"),
        "geometry_contract": {
            "num_slots": int(
                cfg.get("model", {})
                .get("structured_query", {})
                .get("set_selection", {})
                .get("four_slot_num_slots", 4)
            ),
            "line_width": float(
                cfg.get("loss", {}).get("four_slot_line_width", 30.0)
            ),
            "min_valid_rows": int(
                cfg.get("loss", {}).get("four_slot_min_valid_rows", 5)
            ),
        },
        "contracts": TARGET_CONTRACTS,
        "modes": modes,
        "checks": {
            "same_uniform_train_indices": bool(same_train_indices),
            "source_weights_loaded": int(load_stats.get("loaded", 0)) > 0,
            "all_modes_nonempty": all(
                next(iter(values.values()))["images"] > 0
                for values in modes.values()
            ),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact = {
        "checks": report["checks"],
        "modes": {
            mode: {
                contract: {
                    key: values[key]
                    for key in (
                        "mean_gt_lanes",
                        "mean_representable_lanes",
                        "representable_gt_fraction",
                        "expected_dustbin_fraction",
                        "mean_support_size",
                        "mean_target_entropy",
                    )
                }
                for contract, values in contracts.items()
            }
            for mode, contracts in modes.items()
        },
    }
    print(json.dumps(compact, indent=2))
    print(f"output_json: {output}")
    if not all(report["checks"].values()):
        raise SystemExit("V6-A target distribution audit failed")


if __name__ == "__main__":
    main()
