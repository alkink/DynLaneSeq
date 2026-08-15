from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.v20_slot_owned_replacement import _gather_candidate
from dynlaneseq_eg.modeling.v22_lane_field import (
    V22LaneFieldStageA,
    sample_lane_field_rows,
    score_candidates_from_lane_field,
)
from dynlaneseq_eg.tools.audit_v11_causal_replay import _image_id
from dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_official import _required
from dynlaneseq_eg.tools.audit_v20_decision_sufficiency import _load_head, score_cache
from dynlaneseq_eg.tools.cache_v21a_pairwise_visual_verification import (
    _gather_topk,
    _lexicographic_masks,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v20_cached_replacement import _load_cache
from dynlaneseq_eg.tools.v22_official_protocol import official_culane_list_contract


FIXED_SEED = 3407
FIXED_TOP1_MINIMUM = 0.65
FIXED_ADVANTAGE_MINIMUM = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the V22 Stage-A lane field on the immutable V20 oracle-slot "
            "top-5 diagnostic. This tool is not a deployable detector evaluation."
        )
    )
    parser.add_argument("--field-config", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--v20-config", required=True)
    parser.add_argument("--v20-geometry-checkpoint", required=True)
    parser.add_argument("--v20-scoring-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        help="NAME|SPLIT|SOURCE_LIST|V20_CACHE|WRONG_LIST|WRONG_REPORT",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    return parser.parse_args()


def _parse_domains(rows: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        parts = row.split("|")
        if len(parts) != 6:
            raise ValueError(f"invalid V22 Stage-A domain specification: {row!r}")
        name, split, source, cache, wrong, report = parts
        if not name or name in result or split not in {"train", "val"}:
            raise ValueError(f"invalid/duplicate V22 domain: {name!r}")
        result[name] = {
            "split": split,
            "source": str(Path(source).expanduser().resolve()),
            "cache": str(Path(cache).expanduser().resolve()),
            "wrong": str(Path(wrong).expanduser().resolve()),
            "wrong_report": str(Path(report).expanduser().resolve()),
        }
    return result


def _zero_augmentation(cfg: dict[str, Any]) -> None:
    cfg["augmentation"] = {
        "cut_height": int(cfg.get("dataset", {}).get("cut_height", 270)),
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


def _dataset_config(
    base: dict[str, Any],
    *,
    dataset_root: str,
    split: str,
    list_path: str,
    num_workers: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})[split] = str(
        Path(list_path).expanduser().resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = 1
    cfg["dataloader"]["num_workers"] = int(num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    _zero_augmentation(cfg)
    return cfg


def _load_field(
    config: dict[str, Any], checkpoint_path: str, device: torch.device
) -> tuple[V22LaneFieldStageA, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = config["model"]
    stage_cfg = config["v22_stage_a"]
    model = V22LaneFieldStageA(
        input_h=int(model_cfg["input_h"]),
        input_w=int(model_cfg["input_w"]),
        num_rows=int(model_cfg["num_rows"]),
        x_bins=int(model_cfg["x_bins"]),
        fpn_channels=int(model_cfg["fpn_channels"]),
        hidden_dim=int(stage_cfg["hidden_dim"]),
        distance_limit_px=float(stage_cfg["distance_limit_px"]),
        freeze_batch_norm_stats=bool(stage_cfg["freeze_batch_norm_stats"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, payload


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().flatten().cpu()
    labels = labels.detach().bool().flatten().cpu()
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order].float()
    precision = ranked.cumsum(0) / torch.arange(
        1, ranked.numel() + 1, dtype=torch.float32
    )
    return float((precision * ranked).sum() / float(positives))


def _summarize_scores(
    score: torch.Tensor,
    *,
    valid: torch.Tensor,
    beneficial: torch.Tensor,
    harmful: torch.Tensor,
    neutral: torch.Tensor,
) -> dict[str, Any]:
    valid = valid.bool()
    beneficial = beneficial.bool() & valid
    harmful = harmful.bool() & valid
    neutral = neutral.bool() & valid
    positive_case = beneficial.any(dim=1)
    selected = score.float().masked_fill(~valid, -1.0e4).argmax(dim=1)
    selected_beneficial = beneficial.gather(1, selected[:, None]).squeeze(1)
    selected_harmful = harmful.gather(1, selected[:, None]).squeeze(1)
    selected_neutral = neutral.gather(1, selected[:, None]).squeeze(1)
    good = int((selected_beneficial & positive_case).sum())
    bad = int((selected_harmful & positive_case).sum())
    neutral_count = int((selected_neutral & positive_case).sum())
    decisive = good + bad
    return {
        "cases": int(score.shape[0]),
        "cases_with_beneficial_in_top5": int(positive_case.sum()),
        "beneficial_top5_case_coverage": float(positive_case.float().mean()),
        "conditional_top1_beneficial": (
            float(selected_beneficial[positive_case].float().mean())
            if bool(positive_case.any())
            else float("nan")
        ),
        "conditional_decisive_precision": float(good) / float(decisive) if decisive else 0.0,
        "conditional_top1_outcome": {
            "beneficial": good,
            "harmful": bad,
            "neutral": neutral_count,
        },
        "beneficial_action_average_precision": _average_precision(
            score[valid], beneficial[valid]
        ),
        "v20_shortlist_first_conditional_top1": (
            float(beneficial[positive_case, 0].float().mean())
            if bool(positive_case.any())
            else float("nan")
        ),
    }


def _empty_row_metrics() -> dict[str, float]:
    return {
        "valid_rows": 0.0,
        "center_probability_sum": 0.0,
        "distance_abs_sum_px": 0.0,
        "support_probability_sum": 0.0,
        "local_peak_error_sum_px": 0.0,
    }


@torch.no_grad()
def _field_row_metrics(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    input_w: int,
    distance_limit_px: float,
) -> dict[str, float]:
    aggregate = _empty_row_metrics()
    offsets = torch.arange(
        -32.0,
        32.0 + 1.0e-6,
        2.0,
        device=outputs["centerline_logits"].device,
    )
    for batch_index, target in enumerate(targets):
        x = target["x_rows"].to(
            device=outputs["centerline_logits"].device, dtype=torch.float32
        )
        valid = target["valid_mask"].to(
            device=outputs["centerline_logits"].device
        ).bool()
        if x.numel() == 0:
            continue
        rows = int(outputs["centerline_logits"].shape[-2])
        x = x[:, :rows]
        valid = valid[:, :rows] & (x >= 0.0) & (x < float(input_w))
        count = int(valid.sum())
        if count == 0:
            continue
        exact_x = x.unsqueeze(0)
        center = sample_lane_field_rows(
            outputs["centerline_logits"][batch_index : batch_index + 1],
            exact_x,
            input_w=input_w,
        )[0, ..., 0].sigmoid()
        distance = sample_lane_field_rows(
            torch.tanh(outputs["distance_raw"][batch_index : batch_index + 1])
            * float(distance_limit_px),
            exact_x,
            input_w=input_w,
        )[0, ..., 0].abs()
        support = sample_lane_field_rows(
            outputs["support_logits"][batch_index : batch_index + 1],
            exact_x,
            input_w=input_w,
        )[0, ..., 0].sigmoid()

        local_x = x[:, None, :] + offsets.view(1, -1, 1)
        lanes, choices, row_count = local_x.shape
        local = sample_lane_field_rows(
            outputs["centerline_logits"][batch_index : batch_index + 1],
            local_x.reshape(1, lanes * choices, row_count),
            input_w=input_w,
        )[0, ..., 0].reshape(lanes, choices, row_count)
        local_valid = (local_x >= 0.0) & (local_x < float(input_w))
        choice = local.masked_fill(~local_valid, -1.0e4).argmax(dim=1)
        peak_error = offsets.abs()[choice]
        aggregate["valid_rows"] += float(count)
        aggregate["center_probability_sum"] += float(center[valid].sum())
        aggregate["distance_abs_sum_px"] += float(distance[valid].sum())
        aggregate["support_probability_sum"] += float(support[valid].sum())
        aggregate["local_peak_error_sum_px"] += float(peak_error[valid].sum())
    return aggregate


def _finalize_row_metrics(aggregate: dict[str, float]) -> dict[str, float | int]:
    count = max(float(aggregate["valid_rows"]), 1.0)
    return {
        "valid_rows": int(aggregate["valid_rows"]),
        "center_probability_at_gt": aggregate["center_probability_sum"] / count,
        "distance_abs_at_gt_px": aggregate["distance_abs_sum_px"] / count,
        "support_probability_at_gt": aggregate["support_probability_sum"] / count,
        "local_peak_error_px": aggregate["local_peak_error_sum_px"] / count,
    }


def _accumulate(left: dict[str, float], right: dict[str, float]) -> None:
    for name, value in right.items():
        left[name] += float(value)


@torch.no_grad()
def evaluate_domain(
    *,
    name: str,
    spec: dict[str, str],
    base_cfg: dict[str, Any],
    dataset_root: str,
    num_workers: int,
    field_model: V22LaneFieldStageA,
    v20_model: torch.nn.Module,
    v20_head: torch.nn.Module,
    device: torch.device,
    input_w: int,
    distance_limit_px: float,
) -> dict[str, Any]:
    source_path = Path(spec["source"])
    wrong_path = Path(spec["wrong"])
    wrong_report_path = Path(spec["wrong_report"])
    wrong_report = json.loads(wrong_report_path.read_text(encoding="utf-8"))
    if wrong_report.get("passed") is not True:
        raise ValueError(f"V22 wrong-image contract failed for {name}")
    if str(wrong_report.get("input_sha256")) != sha256_file(source_path):
        raise ValueError(f"V22 source/wrong report mismatch for {name}")
    if str(wrong_report.get("output_sha256")) != sha256_file(wrong_path):
        raise ValueError(f"V22 wrong-list/report mismatch for {name}")
    if int(wrong_report.get("same_image_partner_count", -1)) != 0 or int(
        wrong_report.get("same_clip_partner_count", -1)
    ) != 0:
        raise ValueError(f"V22 wrong-image pairing is contaminated for {name}")

    cache, cache_manifest = _load_cache(Path(spec["cache"]))
    if str(cache_manifest.get("list_sha256")) != sha256_file(source_path):
        raise ValueError(f"V22 source list/cache mismatch for {name}")
    images_total = int(cache["source_route"].shape[0])
    scored = score_cache(
        v20_head,
        cache,
        device=device,
        batch_size=64,
        context_mode="treatment",
    )
    replacement_scores = scored["raw_action_scores"][:, 1:].reshape_as(
        cache["action_valid"]
    )
    action_valid_all = cache["action_valid"].bool()
    beneficial_all, harmful_all, neutral_all = _lexicographic_masks(
        cache["delta50_class"][:, 1:].reshape_as(action_valid_all),
        cache["delta75_class"][:, 1:].reshape_as(action_valid_all),
    )
    positive_slot_all = (beneficial_all & action_valid_all).any(dim=2)

    source_cfg = _dataset_config(
        base_cfg,
        dataset_root=dataset_root,
        split=spec["split"],
        list_path=spec["source"],
        num_workers=num_workers,
    )
    wrong_cfg = _dataset_config(
        base_cfg,
        dataset_root=dataset_root,
        split=spec["split"],
        list_path=spec["wrong"],
        num_workers=num_workers,
    )
    source_loader = build_dataloader(source_cfg, split=spec["split"], training=False)
    wrong_loader = build_dataloader(wrong_cfg, split=spec["split"], training=False)
    if len(source_loader.dataset) != images_total or len(wrong_loader.dataset) != images_total:
        raise ValueError(f"V22 loader/cache size mismatch for {name}")

    correct_scores: list[torch.Tensor] = []
    wrong_scores: list[torch.Tensor] = []
    geometry_scores: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    beneficial_rows: list[torch.Tensor] = []
    harmful_rows: list[torch.Tensor] = []
    neutral_rows: list[torch.Tensor] = []
    correct_field_metrics = _empty_row_metrics()
    wrong_field_metrics = _empty_row_metrics()
    contract = {
        "image_id_mismatch": 0,
        "runtime_same_image_wrong_partner": 0,
        "runtime_same_clip_wrong_partner": 0,
    }
    replay = {
        "live_source_route_mismatch": 0,
        "live_source_active_mismatch": 0,
        "live_action_valid_mismatch": 0,
    }
    global_index = 0
    for source_batch, wrong_batch in zip(source_loader, wrong_loader):
        images, targets, metas = source_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        batch = int(images.shape[0])
        start, stop = global_index, global_index + batch
        expected_ids = [str(value) for value in cache_manifest["image_ids"][start:stop]]
        actual_ids = [
            _image_id(meta, f"v22_{name}_{start + item:06d}")
            for item, meta in enumerate(metas)
        ]
        contract["image_id_mismatch"] += sum(
            left != right for left, right in zip(expected_ids, actual_ids)
        )
        for meta, wrong_meta in zip(metas, wrong_metas):
            source_image = str(meta.get("image_path", ""))
            wrong_image = str(wrong_meta.get("image_path", ""))
            contract["runtime_same_image_wrong_partner"] += int(source_image == wrong_image)
            contract["runtime_same_clip_wrong_partner"] += int(
                str(Path(source_image).parent) == str(Path(wrong_image).parent)
            )
        images = images.to(device, non_blocking=True)
        wrong_images = wrong_images.to(device, non_blocking=True)
        v20_outputs = v20_model(images)
        correct_field = field_model(images)
        wrong_field = field_model(wrong_images)
        cached = {
            key: value[start:stop].to(device, non_blocking=True)
            for key, value in cache.items()
        }
        source_route = cached["source_route"].long()
        source_active = cached["source_active"].bool()
        action_valid = cached["action_valid"].bool()
        replay["live_source_route_mismatch"] += int(
            (
                _required(v20_outputs, "selection_slot_v20_v7_geometry_route_indices").long()
                != source_route
            ).sum().cpu()
        )
        replay["live_source_active_mismatch"] += int(
            (
                _required(v20_outputs, "selection_slot_v20_v7_active").bool()
                != source_active
            ).sum().cpu()
        )
        replay["live_action_valid_mismatch"] += int(
            (
                _required(v20_outputs, "selection_slot_v20_action_valid").bool()
                != action_valid
            ).sum().cpu()
        )
        local_rank = replacement_scores[start:stop].to(device).masked_fill(
            ~action_valid, -1.0e4
        )
        top_ids = torch.topk(local_rank, k=5, dim=2).indices
        top_valid = _gather_topk(action_valid.unsqueeze(-1), top_ids).squeeze(-1)
        cf_x = _required(v20_outputs, "selection_slot_v19_counterfactual_x_rows").float()
        cf_range = _required(
            v20_outputs, "selection_slot_v19_counterfactual_range_norm"
        ).float()
        source_x = _gather_candidate(cf_x, source_route)
        source_range = _gather_candidate(cf_range, source_route)
        candidate_x = _gather_topk(cf_x, top_ids)
        candidate_range = _gather_topk(cf_range, top_ids)
        correct_scored = score_candidates_from_lane_field(
            correct_field,
            source_x=source_x,
            source_range=source_range,
            candidate_x=candidate_x,
            candidate_range=candidate_range,
            candidate_valid=top_valid,
            input_w=input_w,
            distance_limit_px=distance_limit_px,
        )
        wrong_scored = score_candidates_from_lane_field(
            wrong_field,
            source_x=source_x,
            source_range=source_range,
            candidate_x=candidate_x,
            candidate_range=candidate_range,
            candidate_valid=top_valid,
            input_w=input_w,
            distance_limit_px=distance_limit_px,
        )
        beneficial = _gather_topk(
            beneficial_all[start:stop].to(device).unsqueeze(-1), top_ids
        ).squeeze(-1)
        harmful = _gather_topk(
            harmful_all[start:stop].to(device).unsqueeze(-1), top_ids
        ).squeeze(-1)
        neutral = _gather_topk(
            neutral_all[start:stop].to(device).unsqueeze(-1), top_ids
        ).squeeze(-1)
        positive_slot = positive_slot_all[start:stop].to(device)
        for image in range(batch):
            selected_slots = torch.nonzero(positive_slot[image], as_tuple=False).flatten()
            if selected_slots.numel() == 0:
                continue
            correct_scores.append(correct_scored["field_score"][image, selected_slots].cpu())
            wrong_scores.append(wrong_scored["field_score"][image, selected_slots].cpu())
            geometry_scores.append(correct_scored["geometry_score"][image, selected_slots].cpu())
            valid_rows.append(top_valid[image, selected_slots].cpu())
            beneficial_rows.append(beneficial[image, selected_slots].cpu())
            harmful_rows.append(harmful[image, selected_slots].cpu())
            neutral_rows.append(neutral[image, selected_slots].cpu())
        _accumulate(
            correct_field_metrics,
            _field_row_metrics(
                correct_field,
                targets,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
            ),
        )
        _accumulate(
            wrong_field_metrics,
            _field_row_metrics(
                wrong_field,
                targets,
                input_w=input_w,
                distance_limit_px=distance_limit_px,
            ),
        )
        global_index = stop
    if global_index != images_total:
        raise RuntimeError(f"V22 did not consume complete domain {name}")
    if not correct_scores:
        raise RuntimeError(f"V22 domain {name} contains no oracle-positive slots")
    score_correct = torch.cat(correct_scores)
    score_wrong = torch.cat(wrong_scores)
    score_geometry = torch.cat(geometry_scores)
    valid = torch.cat(valid_rows).bool()
    beneficial = torch.cat(beneficial_rows).bool()
    harmful = torch.cat(harmful_rows).bool()
    neutral = torch.cat(neutral_rows).bool()
    correct_summary = _summarize_scores(
        score_correct,
        valid=valid,
        beneficial=beneficial,
        harmful=harmful,
        neutral=neutral,
    )
    wrong_summary = _summarize_scores(
        score_wrong,
        valid=valid,
        beneficial=beneficial,
        harmful=harmful,
        neutral=neutral,
    )
    geometry_summary = _summarize_scores(
        score_geometry,
        valid=valid,
        beneficial=beneficial,
        harmful=harmful,
        neutral=neutral,
    )
    passed_contract = all(int(value) == 0 for value in contract.values())
    return {
        "images_evaluated": images_total,
        "source_list": str(source_path),
        "source_list_sha256": sha256_file(source_path),
        "v20_cache_manifest": spec["cache"],
        "v20_cache_manifest_sha256": sha256_file(spec["cache"]),
        "correct_image_field": correct_summary,
        "cross_clip_wrong_image_field": wrong_summary,
        "source_geometry_control": geometry_summary,
        "field_row_evidence": {
            "correct_image": _finalize_row_metrics(correct_field_metrics),
            "wrong_image": _finalize_row_metrics(wrong_field_metrics),
        },
        "deltas": {
            "correct_minus_wrong_top1_points": (
                correct_summary["conditional_top1_beneficial"]
                - wrong_summary["conditional_top1_beneficial"]
            ),
            "correct_minus_geometry_top1_points": (
                correct_summary["conditional_top1_beneficial"]
                - geometry_summary["conditional_top1_beneficial"]
            ),
        },
        "contract": {**contract, "passed": passed_contract},
        "live_replay_diagnostics_not_consumed": replay,
    }


def main() -> None:
    args = parse_args()
    if int(args.seed) != FIXED_SEED:
        raise ValueError(f"V22 Stage-A fixed gate requires seed {FIXED_SEED}")
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    domains = _parse_domains(args.domain)
    required = {"official_val"}
    if set(domains) != required:
        raise ValueError(
            "V22 Stage-A primary gate accepts exactly one population: "
            "the complete official_val split"
        )
    official_val_population = official_culane_list_contract(
        args.dataset_root,
        split="val",
        supplied_path=domains["official_val"]["source"],
    )
    if domains["official_val"]["split"] != "val":
        raise ValueError("official_val must use the CULane val loader")
    field_cfg: dict[str, Any] = load_config(args.field_config)
    v20_cfg: dict[str, Any] = load_config(args.v20_config)
    v20_cfg.setdefault("model", {})["pretrained_backbone"] = False
    v20_cfg["model"]["require_pretrained_backbone"] = False
    _zero_augmentation(v20_cfg)
    field_model, field_payload = _load_field(
        field_cfg, args.field_checkpoint, device
    )
    v20_model = build_model(v20_cfg).to(device)
    geometry_iteration = int(
        load_checkpoint(args.v20_geometry_checkpoint, v20_model, strict=False)
    )
    v20_model.requires_grad_(False).eval()
    v20_head, scoring_iteration = _load_head(
        args.v20_config, args.v20_scoring_checkpoint, device
    )
    model_cfg = field_cfg["model"]
    distance_limit = float(field_cfg["v22_stage_a"]["distance_limit_px"])
    results: dict[str, Any] = {}
    for name, spec in domains.items():
        results[name] = evaluate_domain(
            name=name,
            spec=spec,
            base_cfg=v20_cfg,
            dataset_root=args.dataset_root,
            num_workers=int(args.num_workers),
            field_model=field_model,
            v20_model=v20_model,
            v20_head=v20_head,
            device=device,
            input_w=int(model_cfg["input_w"]),
            distance_limit_px=distance_limit,
        )
    gate_checks: dict[str, bool] = {}
    for name in sorted(required):
        item = results[name]
        correct = float(
            item["correct_image_field"]["conditional_top1_beneficial"]
        )
        gate_checks[f"{name}_top1_at_least_0p65"] = correct >= FIXED_TOP1_MINIMUM
        gate_checks[f"{name}_correct_over_wrong_at_least_0p10"] = (
            float(item["deltas"]["correct_minus_wrong_top1_points"])
            >= FIXED_ADVANTAGE_MINIMUM
        )
        gate_checks[f"{name}_correct_over_geometry_at_least_0p10"] = (
            float(item["deltas"]["correct_minus_geometry_top1_points"])
            >= FIXED_ADVANTAGE_MINIMUM
        )
        gate_checks[f"{name}_data_contract"] = item["contract"]["passed"] is True
        gate_checks[f"{name}_all_9675_images_evaluated"] = (
            int(item["images_evaluated"])
            == int(official_val_population["expected_nonempty_rows"])
        )
    passed = all(gate_checks.values())
    report = {
        "experiment": "V22 Stage-A trainable global lane-field sufficiency gate",
        "field_config": str(Path(args.field_config).expanduser().resolve()),
        "field_config_sha256": sha256_file(args.field_config),
        "field_checkpoint": str(Path(args.field_checkpoint).expanduser().resolve()),
        "field_checkpoint_sha256": sha256_file(args.field_checkpoint),
        "field_training_iteration": int(field_payload["iteration"]),
        "v20_config": str(Path(args.v20_config).expanduser().resolve()),
        "v20_geometry_checkpoint": str(
            Path(args.v20_geometry_checkpoint).expanduser().resolve()
        ),
        "v20_geometry_checkpoint_sha256": sha256_file(
            args.v20_geometry_checkpoint
        ),
        "v20_geometry_iteration": geometry_iteration,
        "v20_scoring_checkpoint": str(
            Path(args.v20_scoring_checkpoint).expanduser().resolve()
        ),
        "v20_scoring_checkpoint_sha256": sha256_file(
            args.v20_scoring_checkpoint
        ),
        "v20_scoring_iteration": scoring_iteration,
        "fixed_gate": {
            "conditional_top1_minimum": FIXED_TOP1_MINIMUM,
            "correct_over_wrong_minimum_points": FIXED_ADVANTAGE_MINIMUM,
            "correct_over_geometry_minimum_points": FIXED_ADVANTAGE_MINIMUM,
        },
        "domains": results,
        "official_validation_population_contract": official_val_population,
        "gate_checks": gate_checks,
        "passed": passed,
        "recommendation": (
            "eligible_for_user_review_before_v22_stage_b"
            if passed
            else "stop_trainable_lane_field_stage_a_failed"
        ),
        "contract": {
            "diagnostic_only_not_deployable": True,
            "oracle_beneficial_slot_used": True,
            "candidate_shortlist": 5,
            "learned_candidate_reranker": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "full_validation_executed": True,
            "official_validation_rows": int(
                official_val_population["observed_nonempty_rows"]
            ),
            "validation_subset_used": False,
            "validation_rows_removed": 0,
            "validation_deduplication_performed": False,
            "test_set_used": False,
            "v22_stage_b_started": False,
        },
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        "V22 Stage-A diagnostic complete on all 9,675 official validation "
        "images. No slot decoder, test, checkpoint selection, or threshold "
        "search was started."
    )


if __name__ == "__main__":
    main()
