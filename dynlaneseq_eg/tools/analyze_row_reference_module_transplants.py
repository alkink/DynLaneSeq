from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.evaluation.proposal_recall import (
    line_iou_against_gt,
    select_candidates,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Localize row-reference checkpoint drift by transplanting coherent "
            "module groups between a good and a later degraded checkpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--good-checkpoint", required=True)
    parser.add_argument("--bad-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
    )
    parser.add_argument(
        "--group-scope",
        choices=("coarse", "fine", "all"),
        default="coarse",
        help="Choose coarse module groups, fine-grained groups, or both.",
    )
    parser.add_argument(
        "--direction",
        choices=("both", "restore", "inject"),
        default="both",
        help=(
            "restore: insert good modules into the bad checkpoint; "
            "inject: insert bad modules into the good checkpoint."
        ),
    )
    parser.add_argument(
        "--group-names",
        default="",
        help=(
            "Optional comma-separated subset of names from the selected "
            "group scope."
        ),
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _best_iou(
    candidates: torch.Tensor,
    gt_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    line_width: float,
) -> float:
    if int(valid.sum()) < 5 or int(candidates.shape[0]) == 0:
        return 0.0
    iou = line_iou_against_gt(
        candidates,
        gt_x,
        valid,
        line_width=float(line_width),
    )
    return float(iou.max()) if iou.numel() else 0.0


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
    return cfg


def _coarse_group_predicates() -> dict[str, Callable[[str], bool]]:
    structured = "structured_query_head."
    initial_prefixes = (
        f"{structured}instance_tokens.",
        f"{structured}row_tokens.",
        f"{structured}x_tokens.",
        f"{structured}feature_proj.",
        f"{structured}reference_",
    )
    row_geometry_prefixes = (
        f"{structured}row_norm.",
        f"{structured}row_x.",
    )
    scoring_prefixes = (
        f"{structured}lane_norm.",
        f"{structured}exist.",
        f"{structured}range.",
        f"{structured}quality.",
    )

    predicates: dict[str, Callable[[str], bool]] = {
        "encoder_p2": lambda name: name.startswith(
            ("encoder.backbone.", "encoder.fpn.", "encoder.proj.")
        ),
        "initial_reference": lambda name: name.startswith(initial_prefixes),
        "decoder_layers": lambda name: name.startswith(
            f"{structured}layers."
        ),
        "row_geometry_head": lambda name: name.startswith(
            row_geometry_prefixes
        ),
        "scoring_heads": lambda name: name.startswith(scoring_prefixes),
        "iterative_geometry": lambda name: name.startswith(
            (
                f"{structured}layers.",
                *row_geometry_prefixes,
            )
        ),
        "full_geometry_path": lambda name: name.startswith(
            (
                *initial_prefixes,
                f"{structured}layers.",
                *row_geometry_prefixes,
            )
        ),
        "structured_head": lambda name: name.startswith(structured),
    }
    return predicates


def _fine_group_predicates() -> dict[str, Callable[[str], bool]]:
    structured = "structured_query_head."
    layers = f"{structured}layers."
    token_prefixes = (
        f"{structured}instance_tokens.",
        f"{structured}row_tokens.",
        f"{structured}x_tokens.",
    )
    reference_initializer_prefixes = (
        f"{structured}feature_proj.",
        f"{structured}reference_",
    )
    local_markers = (
        ".relative_offset_bias",
        ".offsets_px",
        ".local_query.",
        ".local_key.",
        ".local_value.",
        ".local_out.",
        ".coordinate_proj.",
        ".norm_cross.",
    )
    inter_markers = (".inter_attn.", ".norm_inter.")
    intra_markers = (".intra_attn.", ".norm_intra.")
    ffn_markers = (".ffn.", ".norm_ffn.")
    batch_norm_buffer_suffixes = (
        ".running_mean",
        ".running_var",
        ".num_batches_tracked",
    )
    is_batch_norm_buffer = lambda name: name.endswith(
        batch_norm_buffer_suffixes
    )

    predicates: dict[str, Callable[[str], bool]] = {
        "encoder_backbone": lambda name: name.startswith("encoder.backbone."),
        "encoder_backbone_parameters": lambda name: name.startswith(
            "encoder.backbone."
        )
        and not is_batch_norm_buffer(name),
        "encoder_backbone_bn_buffers": lambda name: name.startswith(
            "encoder.backbone."
        )
        and is_batch_norm_buffer(name),
        "encoder_fpn": lambda name: name.startswith("encoder.fpn."),
        "encoder_fpn_parameters": lambda name: name.startswith("encoder.fpn.")
        and not is_batch_norm_buffer(name),
        "encoder_fpn_bn_buffers": lambda name: name.startswith("encoder.fpn.")
        and is_batch_norm_buffer(name),
        "encoder_projection": lambda name: name.startswith("encoder.proj."),
        "initial_tokens": lambda name: name.startswith(token_prefixes),
        "reference_initializer": lambda name: name.startswith(
            reference_initializer_prefixes
        ),
        "decoder_local_evidence": lambda name: name.startswith(layers)
        and any(marker in name for marker in local_markers),
        "decoder_inter_instance": lambda name: name.startswith(layers)
        and any(marker in name for marker in inter_markers),
        "decoder_intra_lane": lambda name: name.startswith(layers)
        and any(marker in name for marker in intra_markers),
        "decoder_ffn": lambda name: name.startswith(layers)
        and any(marker in name for marker in ffn_markers),
    }
    for layer_index in range(4):
        predicates[f"decoder_layer_{layer_index + 1}"] = (
            lambda name, index=layer_index: name.startswith(
                f"{layers}{index}."
            )
        )
    return predicates


def _group_predicates(scope: str) -> dict[str, Callable[[str], bool]]:
    if scope == "coarse":
        return _coarse_group_predicates()
    if scope == "fine":
        return _fine_group_predicates()
    return {
        **_coarse_group_predicates(),
        **_fine_group_predicates(),
    }


def _transplant(
    destination: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    predicate: Callable[[str], bool],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    output = dict(destination)
    transplanted = []
    for name, tensor in source.items():
        if predicate(name):
            output[name] = tensor
            transplanted.append(name)
    if not transplanted:
        raise ValueError("module transplant selected no state-dict entries")
    return output, transplanted


def _evaluate_state(
    *,
    cfg: dict[str, Any],
    state: dict[str, torch.Tensor],
    loader,
    sampled_indices: list[int],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    line_width: float,
    top_k: int,
    label: str,
) -> list[dict[str, float | int]]:
    model = build_model(cfg)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    records: list[dict[str, float | int]] = []
    image_offset = 0
    with torch.inference_mode():
        for images, targets, _metas in tqdm(loader, desc=label):
            if channels_last:
                images = images.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                images = images.to(device, non_blocking=True)
            with _amp_context(device, amp_dtype):
                outputs = model(images, inference_only=True)
            for batch_index, target in enumerate(targets):
                all_candidates = outputs["pred_x_rows"][batch_index]
                top_candidates = select_candidates(
                    outputs,
                    batch_index,
                    top_k=int(top_k),
                    rank_by="score_quality",
                )
                gt_x = target["x_rows"].to(
                    device=all_candidates.device,
                    dtype=all_candidates.dtype,
                )
                valid = target["valid_mask"].to(
                    device=all_candidates.device
                ).bool()
                dataset_index = int(
                    sampled_indices[image_offset + batch_index]
                )
                for lane_index in range(int(gt_x.shape[0])):
                    if int(valid[lane_index].sum()) < 5:
                        continue
                    records.append(
                        {
                            "dataset_index": dataset_index,
                            "lane_index": lane_index,
                            "all_iou": _best_iou(
                                all_candidates,
                                gt_x[lane_index],
                                valid[lane_index],
                                line_width=line_width,
                            ),
                            "top_iou": _best_iou(
                                top_candidates,
                                gt_x[lane_index],
                                valid[lane_index],
                                line_width=line_width,
                            ),
                        }
                    )
            image_offset += int(images.shape[0])

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def _recall(records: list[dict[str, float | int]], key: str, threshold: float):
    return sum(float(record[key]) >= threshold for record in records) / float(
        max(len(records), 1)
    )


def _summarize_records(
    records: list[dict[str, float | int]],
) -> dict[str, float | int]:
    return {
        "lanes": len(records),
        "mean_best_iou": sum(
            float(record["all_iou"]) for record in records
        )
        / float(max(len(records), 1)),
        "raw_recall_050": _recall(records, "all_iou", 0.5),
        "raw_recall_070": _recall(records, "all_iou", 0.7),
        "top4_recall_050": _recall(records, "top_iou", 0.5),
        "top4_recall_070": _recall(records, "top_iou", 0.7),
    }


def _paired_delta(
    reference: list[dict[str, float | int]],
    candidate: list[dict[str, float | int]],
) -> dict[str, float | int]:
    reference_ids = [
        (int(record["dataset_index"]), int(record["lane_index"]))
        for record in reference
    ]
    candidate_ids = [
        (int(record["dataset_index"]), int(record["lane_index"]))
        for record in candidate
    ]
    if reference_ids != candidate_ids:
        raise ValueError("variant lane identities are not aligned")
    output: dict[str, float | int] = {}
    for threshold in (0.5, 0.7):
        recovered = sum(
            float(old["all_iou"]) < threshold <= float(new["all_iou"])
            for old, new in zip(reference, candidate)
        )
        lost = sum(
            float(new["all_iou"]) < threshold <= float(old["all_iou"])
            for old, new in zip(reference, candidate)
        )
        output[f"raw_gain_{int(threshold * 100):03d}_points"] = 100.0 * (
            _recall(candidate, "all_iou", threshold)
            - _recall(reference, "all_iou", threshold)
        )
        output[f"top4_gain_{int(threshold * 100):03d}_points"] = 100.0 * (
            _recall(candidate, "top_iou", threshold)
            - _recall(reference, "top_iou", threshold)
        )
        output[f"recovered_{int(threshold * 100):03d}"] = recovered
        output[f"lost_{int(threshold * 100):03d}"] = lost
    output["mean_iou_gain"] = (
        sum(float(record["all_iou"]) for record in candidate)
        - sum(float(record["all_iou"]) for record in reference)
    ) / float(max(len(reference), 1))
    return output


def main() -> None:
    args = parse_args()
    cfg = _prepare_config(args)
    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]

    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=int(args.max_batches),
        num_workers=int(args.num_workers),
    )
    good_payload = torch.load(args.good_checkpoint, map_location="cpu")
    bad_payload = torch.load(args.bad_checkpoint, map_location="cpu")
    good_state = good_payload["model"]
    bad_state = bad_payload["model"]
    if good_state.keys() != bad_state.keys():
        raise ValueError("good and bad checkpoints have different state keys")

    predicates = _group_predicates(args.group_scope)
    if args.group_names.strip():
        requested = {
            name.strip()
            for name in args.group_names.split(",")
            if name.strip()
        }
        unknown = sorted(requested.difference(predicates))
        if unknown:
            raise ValueError(f"unknown transplant groups: {unknown}")
        predicates = {
            name: predicate
            for name, predicate in predicates.items()
            if name in requested
        }
    variants: dict[str, tuple[dict[str, torch.Tensor], list[str]]] = {
        "good": (good_state, []),
        "bad": (bad_state, []),
    }
    for group_name, predicate in predicates.items():
        keys = []
        if args.direction in ("both", "restore"):
            restored, keys = _transplant(bad_state, good_state, predicate)
            variants[f"bad_restore_{group_name}"] = (restored, keys)
        if args.direction in ("both", "inject"):
            injected, keys = _transplant(good_state, bad_state, predicate)
            variants[f"good_inject_{group_name}"] = (injected, keys)

    records_by_variant = {}
    module_key_counts = {}
    for label, (state, keys) in variants.items():
        module_key_counts[label] = len(keys)
        records_by_variant[label] = _evaluate_state(
            cfg=cfg,
            state=state,
            loader=loader,
            sampled_indices=sampled_indices,
            device=device,
            amp_dtype=amp_dtype,
            line_width=float(args.line_width),
            top_k=int(args.top_k),
            label=label,
        )

    results = {}
    for label, records in records_by_variant.items():
        results[label] = {
            "metrics": _summarize_records(records),
            "delta_vs_good": _paired_delta(
                records_by_variant["good"], records
            ),
            "delta_vs_bad": _paired_delta(
                records_by_variant["bad"], records
            ),
            "transplanted_state_entries": module_key_counts[label],
        }

    payload = {
        "diagnostic_only": True,
        "warning": (
            "Module transplants are frozen-checkpoint causal diagnostics, not "
            "deployable models or benchmark results. Interactions can make "
            "single-group restoration non-additive."
        ),
        "config": args.config,
        "good_checkpoint": args.good_checkpoint,
        "bad_checkpoint": args.bad_checkpoint,
        "good_iteration": int(good_payload.get("iteration", 0)),
        "bad_iteration": int(bad_payload.get("iteration", 0)),
        "split": args.split,
        "group_scope": args.group_scope,
        "group_names": sorted(predicates),
        "direction": args.direction,
        "sample_strategy": args.sample_strategy,
        "sampled_dataset_indices": sampled_indices,
        "images": len(sampled_indices),
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
