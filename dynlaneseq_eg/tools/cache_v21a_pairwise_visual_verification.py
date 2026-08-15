from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.v20_slot_owned_replacement import _gather_candidate
from dynlaneseq_eg.tools.audit_v11_causal_replay import _image_id
from dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_official import _required
from dynlaneseq_eg.tools.audit_v20_decision_sufficiency import _load_head, score_cache
from dynlaneseq_eg.tools.probe_curve_aligned_visual_verification import (
    sample_curve_aligned_profiles,
    sampled_range_weights,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v20_cached_replacement import _load_cache


CASE_FIELDS = (
    "source_profile_correct",
    "candidate_profile_correct",
    "source_profile_wrong",
    "candidate_profile_wrong",
    "source_row_weight",
    "candidate_row_weight",
    "source_state",
    "candidate_state",
    "source_scalar",
    "candidate_scalar",
    "candidate_to_source_relation",
    "candidate_valid",
    "beneficial",
    "harmful",
    "neutral",
    "policy_target",
    "candidate_ids",
    "source_candidate_id",
    "image_index",
    "slot_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache V21A oracle-slot current-vs-top5 visual verification cases. "
            "The cache reuses immutable V20 exact labels and never evaluates test."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--geometry-checkpoint",
        required=True,
        help="Exact frozen V20 initialization used to build the official cache.",
    )
    parser.add_argument(
        "--scoring-checkpoint",
        required=True,
        help="V20 treatment endpoint used only to rank each slot's shortlist.",
    )
    parser.add_argument("--v20-cache-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--wrong-list-path", required=True)
    parser.add_argument("--wrong-list-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--curve-samples", type=int, default=24)
    parser.add_argument(
        "--offsets-px",
        type=float,
        nargs="+",
        default=(-32.0, -16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0, 32.0),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _configured(
    args: argparse.Namespace,
    *,
    list_path: str,
) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})[args.split] = str(
        Path(list_path).expanduser().resolve()
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
        raise ValueError("V21A visual cache requires exactly zero augmentation")
    return cfg


def _gather_topk(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.ndim < 3 or indices.shape[:2] != value.shape[:2]:
        raise ValueError("V21A top-k gather shape mismatch")
    suffix = value.shape[3:]
    index = indices.long().view(*indices.shape, *([1] * len(suffix))).expand(
        *indices.shape, *suffix
    )
    return value.gather(2, index)


def _same_slot_relation(relations: torch.Tensor) -> torch.Tensor:
    if relations.ndim != 5 or relations.shape[-1] != 11:
        raise ValueError("V21A curve relation cache has invalid shape")
    batch, slots, candidates, _reference_slots, relation_dim = relations.shape
    index = torch.arange(slots, device=relations.device).view(1, slots, 1, 1, 1)
    index = index.expand(batch, slots, candidates, 1, relation_dim)
    return relations.gather(3, index).squeeze(3)


def _lexicographic_masks(
    delta50_class: torch.Tensor,
    delta75_class: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta50 = delta50_class.long() - 1
    delta75 = delta75_class.long() - 1
    beneficial = (delta50 > 0) | ((delta50 == 0) & (delta75 > 0))
    harmful = (delta50 < 0) | ((delta50 == 0) & (delta75 < 0))
    return beneficial, harmful, ~(beneficial | harmful)


def _write_shard(
    records: list[dict[str, Any]],
    path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_ids": [str(record["image_id"]) for record in records],
    }
    for name in CASE_FIELDS:
        values = [record[name].detach().cpu().contiguous() for record in records]
        stacked = torch.stack(values)
        if stacked.is_floating_point():
            stacked = stacked.to(torch.float16)
        payload[name] = stacked
    torch.save(payload, path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "cases": len(records),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 1:
        raise ValueError("V21A cache batch_size must be positive")
    if int(args.curve_samples) < 5:
        raise ValueError("V21A requires at least five curve rows")
    offsets = tuple(float(value) for value in args.offsets_px)
    if len(offsets) < 3 or offsets != tuple(sorted(set(offsets))) or 0.0 not in offsets:
        raise ValueError("V21A offsets must be unique, sorted and contain zero")
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(manifest_path.read_text(encoding="utf-8"))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for stale in output_dir.glob("shard_*.pt"):
            stale.unlink()

    source_list = Path(args.list_path).expanduser().resolve()
    wrong_list = Path(args.wrong_list_path).expanduser().resolve()
    wrong_report_path = Path(args.wrong_list_report).expanduser().resolve()
    wrong_report = json.loads(wrong_report_path.read_text(encoding="utf-8"))
    if wrong_report.get("passed") is not True:
        raise ValueError("V21A cross-clip derangement report did not pass")
    if str(wrong_report.get("input_sha256")) != sha256_file(source_list):
        raise ValueError("V21A cross-clip source-list digest mismatch")
    if str(wrong_report.get("output_sha256")) != sha256_file(wrong_list):
        raise ValueError("V21A cross-clip wrong-list digest mismatch")
    if int(wrong_report.get("same_image_partner_count", -1)) != 0 or int(
        wrong_report.get("same_clip_partner_count", -1)
    ) != 0:
        raise ValueError("V21A cross-clip pairing is contaminated")

    cfg = _configured(args, list_path=str(source_list))
    wrong_cfg = _configured(args, list_path=str(wrong_list))
    v20_manifest_path = Path(args.v20_cache_manifest).expanduser().resolve()
    cache, v20_manifest = _load_cache(v20_manifest_path)
    if sha256_file(source_list) != str(v20_manifest.get("list_sha256")):
        raise ValueError("V21A source list does not match the V20 exact cache")

    model = build_model(cfg).to(device)
    geometry_iteration = int(
        load_checkpoint(args.geometry_checkpoint, model, strict=False)
    )
    model.requires_grad_(False).eval()
    head, scoring_iteration = _load_head(
        args.config, args.scoring_checkpoint, device
    )
    scored = score_cache(
        head,
        cache,
        device=device,
        batch_size=int(args.score_batch_size),
        context_mode="treatment",
    )
    images_total = int(cache["source_route"].shape[0])
    replacement_scores = scored["raw_action_scores"][:, 1:].reshape(
        images_total,
        *cache["action_valid"].shape[1:],
    )
    action_valid_all = cache["action_valid"].bool()
    beneficial_all, harmful_all, neutral_all = _lexicographic_masks(
        cache["delta50_class"][:, 1:].reshape_as(action_valid_all),
        cache["delta75_class"][:, 1:].reshape_as(action_valid_all),
    )
    policy_all = cache["policy_target"][:, 1:].reshape_as(
        action_valid_all
    ).float()
    positive_slot_all = (beneficial_all & action_valid_all).any(dim=2)

    correct_loader = build_dataloader(cfg, split=args.split, training=False)
    wrong_loader = build_dataloader(wrong_cfg, split=args.split, training=False)
    if len(correct_loader.dataset) != images_total or len(wrong_loader.dataset) != images_total:
        raise ValueError("V21A dataloader/cache image counts differ")
    model_cfg = cfg["model"]
    input_h = int(model_cfg["input_h"])
    input_w = int(model_cfg["input_w"])
    total_rows = int(model_cfg["num_rows"])
    curve_samples = min(int(args.curve_samples), total_rows)
    row_indices = torch.linspace(0, total_rows - 1, curve_samples).round().long().to(device)
    offsets_px = torch.tensor(offsets, device=device, dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    def capture_level1(_module, _inputs, output):
        captured["level1"] = output.detach()

    handle = model.encoder.backbone.level1.register_forward_hook(capture_level1)
    records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    global_index = 0
    cases_total = 0
    cases_with_positive_shortlist = 0
    contract = {
        "image_id_mismatch": 0,
        "same_image_wrong_partner": int(wrong_report["same_image_partner_count"]),
        "same_clip_wrong_partner": int(wrong_report["same_clip_partner_count"]),
        "runtime_same_image_wrong_partner": 0,
        "runtime_same_clip_wrong_partner": 0,
    }
    replay_diagnostics = {
        "live_source_route_mismatch": 0,
        "live_source_active_mismatch": 0,
        "live_action_valid_mismatch": 0,
    }

    def flush() -> None:
        if not records:
            return
        path = output_dir / f"shard_{len(shards):04d}.pt"
        shards.append(_write_shard(records, path))
        records.clear()

    progress = tqdm(
        zip(correct_loader, wrong_loader),
        total=len(correct_loader),
        desc="V21A visual pair cache",
        ncols=90,
    )
    for correct_batch, wrong_batch in progress:
        images, _targets, metas = correct_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        batch = int(images.shape[0])
        start, stop = global_index, global_index + batch
        expected_ids = [str(value) for value in v20_manifest["image_ids"][start:stop]]
        actual_ids = [
            _image_id(meta, f"v21a_{start + item:06d}")
            for item, meta in enumerate(metas)
        ]
        for meta, wrong_meta in zip(metas, wrong_metas):
            correct_path = str(meta.get("image_path", ""))
            wrong_path = str(wrong_meta.get("image_path", ""))
            contract["runtime_same_image_wrong_partner"] += int(
                correct_path == wrong_path
            )
            contract["runtime_same_clip_wrong_partner"] += int(
                str(Path(correct_path).parent) == str(Path(wrong_path).parent)
            )
        contract["image_id_mismatch"] += sum(
            left != right for left, right in zip(expected_ids, actual_ids)
        )
        images = images.to(device, non_blocking=True)
        wrong_images = wrong_images.to(device, non_blocking=True)
        captured.clear()
        # Match the V20 exact-raster cache producer bit-for-bit: its frozen
        # forward is FP32 even though the original training config enables AMP.
        outputs = model(images)
        correct_level1 = captured.get("level1")
        if not isinstance(correct_level1, torch.Tensor):
            raise RuntimeError("V21A failed to capture DLA stride-2 features")
        captured.clear()
        wrong_level1 = model.encoder.backbone.level1(
            model.encoder.backbone.level0(
                model.encoder.backbone.base_layer(wrong_images)
            )
        )

        cached = {
            name: value[start:stop].to(device, non_blocking=True)
            for name, value in cache.items()
        }
        live_source_route = _required(
            outputs, "selection_slot_v20_v7_geometry_route_indices"
        ).long()
        live_source_active = _required(
            outputs, "selection_slot_v20_v7_active"
        ).bool()
        # The exact-raster cache is the immutable population contract.  CUDA
        # replay can flip a handful of near-tied argmaxes even with the same
        # checkpoint, so never let a second forward redefine source ownership.
        source_route = cached["source_route"].long()
        source_active = cached["source_active"].bool()
        action_valid = cached["action_valid"].bool()
        replay_diagnostics["live_source_route_mismatch"] += int(
            (live_source_route != source_route).sum().cpu()
        )
        replay_diagnostics["live_source_active_mismatch"] += int(
            (live_source_active != source_active).sum().cpu()
        )
        replay_diagnostics["live_action_valid_mismatch"] += int(
            (
                _required(outputs, "selection_slot_v20_action_valid").bool()
                != action_valid
            ).sum().cpu()
        )

        local_scores = replacement_scores[start:stop].to(device)
        ranked_scores = local_scores.masked_fill(~action_valid, -1.0e4)
        top_ids = torch.topk(ranked_scores, k=5, dim=2).indices
        top_valid = _gather_topk(action_valid.unsqueeze(-1), top_ids).squeeze(-1)
        cf_x = _required(outputs, "selection_slot_v19_counterfactual_x_rows").float()
        cf_range = _required(
            outputs, "selection_slot_v19_counterfactual_range_norm"
        ).float()
        current_x = _gather_candidate(cf_x, source_route)
        current_range = _gather_candidate(cf_range, source_route)
        candidate_x = _gather_topk(cf_x, top_ids)
        candidate_range = _gather_topk(cf_range, top_ids)
        all_x = torch.cat((current_x.unsqueeze(2), candidate_x), dim=2)
        all_range = torch.cat((current_range.unsqueeze(2), candidate_range), dim=2)
        slots = int(all_x.shape[1])
        curves = int(all_x.shape[2])
        flat_x = all_x.reshape(batch, slots * curves, total_rows)
        correct_c1 = sample_curve_aligned_profiles(
            correct_level1,
            flat_x,
            row_indices=row_indices,
            offsets_px=offsets_px,
            input_h=input_h,
            input_w=input_w,
        )
        correct_rgb = sample_curve_aligned_profiles(
            images,
            flat_x,
            row_indices=row_indices,
            offsets_px=offsets_px,
            input_h=input_h,
            input_w=input_w,
        )
        wrong_c1 = sample_curve_aligned_profiles(
            wrong_level1,
            flat_x,
            row_indices=row_indices,
            offsets_px=offsets_px,
            input_h=input_h,
            input_w=input_w,
        )
        wrong_rgb = sample_curve_aligned_profiles(
            wrong_images,
            flat_x,
            row_indices=row_indices,
            offsets_px=offsets_px,
            input_h=input_h,
            input_w=input_w,
        )
        profile_shape = (
            batch,
            slots,
            curves,
            curve_samples,
            len(offsets),
            -1,
        )
        correct_profile = torch.cat((correct_rgb, correct_c1), dim=-1).reshape(
            *profile_shape
        )
        wrong_profile = torch.cat((wrong_rgb, wrong_c1), dim=-1).reshape(
            *profile_shape
        )
        row_weight = sampled_range_weights(
            all_range.reshape(batch, slots * curves, 2),
            row_indices=row_indices,
            total_rows=total_rows,
            temperature=0.03,
        ).reshape(batch, slots, curves, curve_samples)

        candidate_state_all = cached["candidate_state"].float()
        source_state = _gather_candidate(candidate_state_all, source_route)
        candidate_state = _gather_topk(candidate_state_all, top_ids)
        quality = torch.stack(
            (cached["p50"], cached["p75"], cached["expected_iou"]), dim=-1
        ).float()
        route_logp = torch.log_softmax(
            cached["legacy_route_logits"].float(), dim=-1
        ).unsqueeze(-1)
        scalar = torch.cat((quality, route_logp), dim=-1)
        source_scalar = _gather_candidate(scalar, source_route)
        candidate_scalar = _gather_topk(scalar, top_ids)
        own_relation = _same_slot_relation(cached["curve_relations"].float())
        candidate_relation = _gather_topk(own_relation, top_ids)
        beneficial = _gather_topk(
            beneficial_all[start:stop].to(device).unsqueeze(-1), top_ids
        ).squeeze(-1)
        harmful = _gather_topk(
            harmful_all[start:stop].to(device).unsqueeze(-1), top_ids
        ).squeeze(-1)
        neutral = _gather_topk(
            neutral_all[start:stop].to(device).unsqueeze(-1), top_ids
        ).squeeze(-1)
        policy_target = _gather_topk(
            policy_all[start:stop].to(device).unsqueeze(-1), top_ids
        ).squeeze(-1)
        positive_slot = positive_slot_all[start:stop].to(device)

        for image in range(batch):
            for slot in torch.nonzero(
                positive_slot[image], as_tuple=False
            ).flatten().tolist():
                cases_total += 1
                cases_with_positive_shortlist += int(
                    bool((beneficial[image, slot] & top_valid[image, slot]).any())
                )
                records.append(
                    {
                        "image_id": actual_ids[image],
                        "source_profile_correct": correct_profile[image, slot, 0],
                        "candidate_profile_correct": correct_profile[image, slot, 1:],
                        "source_profile_wrong": wrong_profile[image, slot, 0],
                        "candidate_profile_wrong": wrong_profile[image, slot, 1:],
                        "source_row_weight": row_weight[image, slot, 0],
                        "candidate_row_weight": row_weight[image, slot, 1:],
                        "source_state": source_state[image, slot],
                        "candidate_state": candidate_state[image, slot],
                        "source_scalar": source_scalar[image, slot],
                        "candidate_scalar": candidate_scalar[image, slot],
                        "candidate_to_source_relation": candidate_relation[image, slot],
                        "candidate_valid": top_valid[image, slot],
                        "beneficial": beneficial[image, slot],
                        "harmful": harmful[image, slot],
                        "neutral": neutral[image, slot],
                        "policy_target": policy_target[image, slot],
                        "candidate_ids": top_ids[image, slot],
                        "source_candidate_id": source_route[image, slot],
                        "image_index": torch.tensor(start + image, device=device),
                        "slot_index": torch.tensor(slot, device=device),
                    }
                )
                if len(records) >= int(args.shard_size):
                    flush()
        global_index = stop
    handle.remove()
    flush()
    if global_index != images_total:
        raise RuntimeError("V21A did not consume the complete fixed image list")
    passed = all(int(value) == 0 for value in contract.values())
    manifest = {
        "experiment": "V21A oracle-slot top5 pairwise visual verification cache",
        "config": str(Path(args.config).expanduser().resolve()),
        "geometry_checkpoint": str(
            Path(args.geometry_checkpoint).expanduser().resolve()
        ),
        "geometry_checkpoint_sha256": sha256_file(args.geometry_checkpoint),
        "geometry_iteration": geometry_iteration,
        "scoring_checkpoint": str(
            Path(args.scoring_checkpoint).expanduser().resolve()
        ),
        "scoring_checkpoint_sha256": sha256_file(args.scoring_checkpoint),
        "scoring_iteration": scoring_iteration,
        "v20_cache_manifest": str(v20_manifest_path),
        "v20_cache_manifest_sha256": sha256_file(v20_manifest_path),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "split": args.split,
        "list_path": str(source_list),
        "list_sha256": sha256_file(source_list),
        "wrong_list_path": str(wrong_list),
        "wrong_list_sha256": sha256_file(wrong_list),
        "wrong_list_report": str(wrong_report_path),
        "wrong_list_report_sha256": sha256_file(wrong_report_path),
        "images": images_total,
        "cases": cases_total,
        "cases_with_beneficial_in_top5": cases_with_positive_shortlist,
        "beneficial_top5_case_coverage": (
            float(cases_with_positive_shortlist) / float(max(cases_total, 1))
        ),
        "shortlist": {
            "oracle_slot_definition": "slot_has_any_lexicographically_beneficial_action",
            "candidate_count": 5,
            "ranking_source": "V20 treatment correct-context policy score",
        },
        "visual_evidence": {
            "sources": ["normalized_RGB_stride1", "frozen_DLA_level1_stride2"],
            "profile_channels": int(correct_profile.shape[-1]),
            "curve_samples": curve_samples,
            "offsets_px": list(offsets),
        },
        "cache_dtype": "float16",
        "augmentation": "exactly_disabled",
        "cross_clip": {
            "same_image_partner_count": int(
                wrong_report["same_image_partner_count"]
            ),
            "same_clip_partner_count": int(
                wrong_report["same_clip_partner_count"]
            ),
            "passed": bool(wrong_report["passed"]),
        },
        "contract": {**contract, "passed": passed},
        "live_replay_diagnostics_not_consumed": replay_diagnostics,
        "shards": shards,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "full_validation_executed": False,
        "test_set_used": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("V21A visual cache contract failed")


if __name__ == "__main__":
    main()
