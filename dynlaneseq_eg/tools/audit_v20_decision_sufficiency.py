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
from dynlaneseq_eg.factory import build_model
from dynlaneseq_eg.tools.train_v20_cached_replacement import _load_cache


FIXED_RISK_COVERAGE = (0.01, 0.02, 0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free V20 per-action decision-sufficiency autopsy. "
            "No threshold or checkpoint is selected."
        )
    )
    parser.add_argument("--treatment-config", required=True)
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--treatment-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        help="NAME=cache_manifest.json; may be repeated",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary average precision without choosing a decision threshold."""

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


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if float(denominator) == 0.0 else float(numerator) / float(denominator)


def _positive_prevalence(labels: torch.Tensor) -> float:
    labels = labels.detach().bool().flatten()
    return float(labels.float().mean()) if labels.numel() else float("nan")


def _ap_lift(average_precision_value: float, labels: torch.Tensor) -> float:
    prevalence = _positive_prevalence(labels)
    if not math.isfinite(prevalence) or prevalence <= 0.0:
        return float("nan")
    return float(average_precision_value) / prevalence


def _score_summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().flatten().cpu()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "q10": float("nan"),
            "q50": float("nan"),
            "q90": float("nan"),
        }
    quantiles = torch.quantile(finite, finite.new_tensor((0.10, 0.50, 0.90)))
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean()),
        "q10": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q90": float(quantiles[2]),
    }


def _lexicographic_labels(
    delta50: torch.Tensor,
    delta75: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beneficial = (delta50 > 0) | ((delta50 == 0) & (delta75 > 0))
    harmful = (delta50 < 0) | ((delta50 == 0) & (delta75 < 0))
    neutral = ~(beneficial | harmful)
    return beneficial, harmful, neutral


def _gather_action(value: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return value.gather(1, action.long().unsqueeze(1)).squeeze(1)


def _topk_policy_recall(
    action_scores: torch.Tensor,
    policy_positive: torch.Tensor,
    opportunity: torch.Tensor,
    k: int,
) -> float:
    if int(opportunity.sum()) == 0:
        return float("nan")
    top = torch.topk(
        action_scores[:, 1:],
        k=min(int(k), int(action_scores.shape[1]) - 1),
        dim=1,
    ).indices + 1
    hit = policy_positive.gather(1, top).any(dim=1)
    return float(hit[opportunity].float().mean())


def _slot_candidate_decomposition(
    replacement_scores: torch.Tensor,
    action_valid: torch.Tensor,
    positive_actions: torch.Tensor,
) -> dict[str, float | int]:
    batch, slots, candidates = replacement_scores.shape
    positive = positive_actions[:, 1:].reshape(batch, slots, candidates).bool()
    opportunity = positive.flatten(1).any(dim=1)
    valid_score = replacement_scores.masked_fill(~action_valid, -1.0e4)
    slot_scores = valid_score.max(dim=2).values
    predicted_slot = slot_scores.argmax(dim=1)
    positive_slot = positive.any(dim=2)
    slot_hit = positive_slot.gather(1, predicted_slot.unsqueeze(1)).squeeze(1)
    valid_slot = action_valid.any(dim=2)
    random_slot_hit = (
        positive_slot.sum(dim=1).float()
        / valid_slot.sum(dim=1).float().clamp_min(1.0)
    )

    ranks: list[int] = []
    random_candidate_top1: list[float] = []
    random_candidate_top5: list[float] = []
    for image in torch.nonzero(opportunity, as_tuple=False).flatten().tolist():
        for slot in torch.nonzero(positive_slot[image], as_tuple=False).flatten().tolist():
            valid = action_valid[image, slot]
            positives = positive[image, slot]
            if not bool(positives.any()):
                continue
            candidate_ids = torch.nonzero(valid, as_tuple=False).flatten()
            ordered = candidate_ids[
                torch.argsort(
                    replacement_scores[image, slot, candidate_ids],
                    descending=True,
                    stable=True,
                )
            ]
            positive_positions = torch.nonzero(
                positives[ordered], as_tuple=False
            ).flatten()
            if positive_positions.numel():
                ranks.append(int(positive_positions[0]) + 1)
                valid_count = int(candidate_ids.numel())
                positive_count = int(positives[candidate_ids].sum())
                random_candidate_top1.append(
                    _safe_ratio(positive_count, valid_count)
                )
                sample_count = min(5, valid_count)
                misses = valid_count - positive_count
                miss_probability = (
                    0.0
                    if misses < sample_count
                    else float(math.comb(misses, sample_count))
                    / float(math.comb(valid_count, sample_count))
                )
                random_candidate_top5.append(1.0 - miss_probability)
    rank_tensor = torch.tensor(ranks, dtype=torch.float32)
    return {
        "opportunity_images": int(opportunity.sum()),
        "slot_top1_hit": (
            float(slot_hit[opportunity].float().mean())
            if bool(opportunity.any())
            else float("nan")
        ),
        "slot_uniform_random_top1_hit": (
            float(random_slot_hit[opportunity].mean())
            if bool(opportunity.any())
            else float("nan")
        ),
        "candidate_cases": len(ranks),
        "candidate_top1_hit": _safe_ratio(sum(rank == 1 for rank in ranks), len(ranks)),
        "candidate_top5_recall": _safe_ratio(sum(rank <= 5 for rank in ranks), len(ranks)),
        "candidate_uniform_random_top1_hit": (
            sum(random_candidate_top1) / len(random_candidate_top1)
            if random_candidate_top1
            else float("nan")
        ),
        "candidate_uniform_random_top5_recall": (
            sum(random_candidate_top5) / len(random_candidate_top5)
            if random_candidate_top5
            else float("nan")
        ),
        "candidate_mean_best_positive_rank": (
            float(rank_tensor.mean()) if rank_tensor.numel() else float("nan")
        ),
        "candidate_median_best_positive_rank": (
            float(rank_tensor.median()) if rank_tensor.numel() else float("nan")
        ),
    }


def decision_metrics(
    cache: dict[str, torch.Tensor],
    scored: dict[str, torch.Tensor],
) -> dict[str, Any]:
    full_valid = cache["full_action_valid"].bool()
    action_valid = cache["action_valid"].bool()
    policy_target = cache["policy_target"].float()
    delta50 = cache["delta50_class"].long() - 1
    delta75 = cache["delta75_class"].long() - 1
    beneficial, harmful, neutral = _lexicographic_labels(delta50, delta75)
    policy_positive = policy_target > 0.0
    opportunity = beneficial[:, 1:].any(dim=1)
    if not torch.equal(opportunity, policy_target[:, 0] < 0.5):
        raise RuntimeError("V20 beneficial-opportunity and policy-target disagree")

    action_scores = scored["raw_action_scores"].float()
    replacement_scores = action_scores[:, 1:].reshape_as(action_valid)
    best_replacement_score, best_replacement_flat = replacement_scores.flatten(1).max(dim=1)
    best_raw_action = best_replacement_flat + 1
    raw_edit = best_replacement_score > 0.0
    deployed_action = scored["deployed_action"].long()
    deployed_edit = deployed_action > 0

    deployed_beneficial = _gather_action(beneficial, deployed_action) & deployed_edit
    deployed_harmful = _gather_action(harmful, deployed_action) & deployed_edit
    deployed_neutral = _gather_action(neutral, deployed_action) & deployed_edit
    deployed_policy_hit = (
        _gather_action(policy_positive, deployed_action) & deployed_edit
    )
    selected = int(deployed_edit.sum())
    selected_beneficial = int(deployed_beneficial.sum())
    selected_harmful = int(deployed_harmful.sum())
    selected_neutral = int(deployed_neutral.sum())
    decisive = selected_beneficial + selected_harmful

    flat_valid = full_valid[:, 1:].flatten()
    flat_scores = action_scores[:, 1:].flatten()[flat_valid]
    flat_beneficial = beneficial[:, 1:].flatten()[flat_valid]
    flat_policy = policy_positive[:, 1:].flatten()[flat_valid]
    flat_harmful = harmful[:, 1:].flatten()[flat_valid]
    flat_neutral = neutral[:, 1:].flatten()[flat_valid]
    flat_delta50_score = scored["expected_delta50"].flatten()[flat_valid]
    flat_delta75_score = scored["expected_delta75"].flatten()[flat_valid]
    flat_duplicate_score = scored["duplicate_probability"].flatten()[flat_valid]
    flat_abandon_score = scored["abandon_probability"].flatten()[flat_valid]
    flat_delta50_positive = (delta50[:, 1:] > 0).flatten()[flat_valid]
    flat_delta75_positive = (delta75[:, 1:] > 0).flatten()[flat_valid]
    flat_duplicate_target = cache["duplicate"][:, 1:].bool().flatten()[flat_valid]
    flat_abandon_target = cache["abandon"][:, 1:].bool().flatten()[flat_valid]

    raw_best_beneficial = _gather_action(beneficial, best_raw_action)
    raw_best_harmful = _gather_action(harmful, best_raw_action)
    raw_best_policy = _gather_action(policy_positive, best_raw_action)
    order = torch.argsort(best_replacement_score, descending=True, stable=True)
    risk_coverage: dict[str, Any] = {}
    images = int(best_replacement_score.numel())
    for fraction in FIXED_RISK_COVERAGE:
        count = max(1, int(math.ceil(float(fraction) * images)))
        chosen = order[:count]
        positive_count = int(raw_best_beneficial[chosen].sum())
        harmful_count = int(raw_best_harmful[chosen].sum())
        target_count = int(raw_best_policy[chosen].sum())
        risk_coverage[f"{fraction:.2f}"] = {
            "images": count,
            "beneficial": positive_count,
            "harmful": harmful_count,
            "neutral": count - positive_count - harmful_count,
            "beneficial_precision": _safe_ratio(positive_count, count),
            "decisive_precision": _safe_ratio(
                positive_count, positive_count + harmful_count
            ),
            "policy_target_hit": _safe_ratio(target_count, count),
            "opportunity_fraction": float(opportunity[chosen].float().mean()),
        }

    raw_true_positive = raw_edit & opportunity
    raw_false_positive = raw_edit & ~opportunity
    deployed_true_positive = deployed_edit & opportunity
    deployed_false_positive = deployed_edit & ~opportunity
    score_groups = {
        "beneficial": _score_summary(flat_scores[flat_beneficial]),
        "harmful": _score_summary(flat_scores[flat_harmful]),
        "neutral": _score_summary(flat_scores[flat_neutral]),
    }
    beneficial_ap = average_precision(flat_scores, flat_beneficial)
    policy_ap = average_precision(flat_scores, flat_policy)
    edit_ap = average_precision(best_replacement_score, opportunity)
    delta50_ap = average_precision(flat_delta50_score, flat_delta50_positive)
    delta75_ap = average_precision(flat_delta75_score, flat_delta75_positive)
    duplicate_ap = average_precision(flat_duplicate_score, flat_duplicate_target)
    abandon_ap = average_precision(flat_abandon_score, flat_abandon_target)
    return {
        "population": {
            "images": images,
            "valid_replacement_actions": int(flat_valid.sum()),
            "opportunity_images": int(opportunity.sum()),
            "opportunity_fraction": float(opportunity.float().mean()),
            "beneficial_actions": int(flat_beneficial.sum()),
            "harmful_actions": int(flat_harmful.sum()),
            "policy_target_actions": int(flat_policy.sum()),
        },
        "action_ranking": {
            "beneficial_average_precision": beneficial_ap,
            "beneficial_prevalence": _positive_prevalence(flat_beneficial),
            "beneficial_average_precision_lift": _ap_lift(
                beneficial_ap, flat_beneficial
            ),
            "policy_target_average_precision": policy_ap,
            "policy_target_prevalence": _positive_prevalence(flat_policy),
            "policy_target_average_precision_lift": _ap_lift(
                policy_ap, flat_policy
            ),
            "top1_beneficial_precision_all_images": float(
                raw_best_beneficial.float().mean()
            ),
            "top1_policy_hit_on_opportunity": (
                float(raw_best_policy[opportunity].float().mean())
                if bool(opportunity.any())
                else float("nan")
            ),
            "top5_policy_recall_on_opportunity": _topk_policy_recall(
                action_scores, policy_positive, opportunity, 5
            ),
            "score_groups": score_groups,
        },
        "individual_head_separability": {
            "policy_logit_beneficial_average_precision": beneficial_ap,
            "delta50_positive_average_precision": delta50_ap,
            "delta50_positive_prevalence": _positive_prevalence(
                flat_delta50_positive
            ),
            "delta50_positive_average_precision_lift": _ap_lift(
                delta50_ap, flat_delta50_positive
            ),
            "delta75_positive_average_precision": delta75_ap,
            "delta75_positive_prevalence": _positive_prevalence(
                flat_delta75_positive
            ),
            "delta75_positive_average_precision_lift": _ap_lift(
                delta75_ap, flat_delta75_positive
            ),
            "duplicate_risk_average_precision": duplicate_ap,
            "duplicate_risk_prevalence": _positive_prevalence(
                flat_duplicate_target
            ),
            "duplicate_risk_average_precision_lift": _ap_lift(
                duplicate_ap, flat_duplicate_target
            ),
            "abandon_risk_average_precision": abandon_ap,
            "abandon_risk_prevalence": _positive_prevalence(
                flat_abandon_target
            ),
            "abandon_risk_average_precision_lift": _ap_lift(
                abandon_ap, flat_abandon_target
            ),
        },
        "edit_detection": {
            "average_precision": edit_ap,
            "positive_prevalence": _positive_prevalence(opportunity),
            "average_precision_lift": _ap_lift(edit_ap, opportunity),
            "raw_keep_zero_rule": {
                "selected": int(raw_edit.sum()),
                "precision": _safe_ratio(
                    int(raw_true_positive.sum()), int(raw_edit.sum())
                ),
                "recall": _safe_ratio(
                    int(raw_true_positive.sum()), int(opportunity.sum())
                ),
                "no_op_false_edit_rate": _safe_ratio(
                    int(raw_false_positive.sum()), int((~opportunity).sum())
                ),
            },
            "configured_deployment": {
                "selected": selected,
                "precision_for_opportunity": _safe_ratio(
                    int(deployed_true_positive.sum()), selected
                ),
                "opportunity_recall": _safe_ratio(
                    int(deployed_true_positive.sum()), int(opportunity.sum())
                ),
                "no_op_false_edit_rate": _safe_ratio(
                    int(deployed_false_positive.sum()), int((~opportunity).sum())
                ),
            },
        },
        "slot_and_candidate": {
            "any_beneficial_action": _slot_candidate_decomposition(
                replacement_scores, action_valid, beneficial
            ),
            "exact_policy_target": _slot_candidate_decomposition(
                replacement_scores, action_valid, policy_positive
            ),
        },
        "configured_deployment_outcome": {
            "selected": selected,
            "beneficial": selected_beneficial,
            "harmful": selected_harmful,
            "neutral": selected_neutral,
            "beneficial_precision": _safe_ratio(selected_beneficial, selected),
            "decisive_precision": _safe_ratio(selected_beneficial, decisive),
            "policy_target_hit": _safe_ratio(
                int(deployed_policy_hit.sum()), selected
            ),
        },
        "fixed_risk_coverage": risk_coverage,
    }


def _load_head(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    cfg = load_config(config_path)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    model = build_model(cfg)
    iteration = int(load_checkpoint(checkpoint_path, model, strict=False))
    source = model.structured_query_head.set_selection_head.slot_owned_safe_replacement
    if source is None:
        raise ValueError("V20 replacement head is missing")
    head = copy.deepcopy(source).to(device).eval()
    del model
    return head, iteration


@torch.no_grad()
def score_cache(
    head: torch.nn.Module,
    cache: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    context_mode: str,
) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {
        "raw_action_scores": [],
        "deployed_action": [],
        "expected_delta50": [],
        "expected_delta75": [],
        "duplicate_probability": [],
        "abandon_probability": [],
    }
    images = int(cache["source_route"].shape[0])
    required_fields = (
        "candidate_state",
        "p50",
        "p75",
        "expected_iou",
        "legacy_route_logits",
        "action_valid",
        "source_route",
        "source_active",
        "curve_relations",
        "full_action_valid",
    )
    for start in range(0, images, int(batch_size)):
        stop = min(start + int(batch_size), images)
        batch = {
            name: value[start:stop].to(device, non_blocking=True)
            for name, value in cache.items()
            if name in required_fields
        }
        output = head(
            candidate_state=batch["candidate_state"],
            p50=batch["p50"],
            p75=batch["p75"],
            expected_iou=batch["expected_iou"],
            legacy_route_logits=batch["legacy_route_logits"],
            counterfactual_valid=batch["action_valid"],
            source_route=batch["source_route"],
            source_active=batch["source_active"],
            precomputed_relations=batch["curve_relations"],
            force_context_mode=context_mode,
        )
        if not torch.equal(output["action_valid"], batch["action_valid"].bool()):
            raise RuntimeError("V20 autopsy action mask parity failed")
        local_batch, slots, candidates = output["policy_logits"].shape
        policy = output["policy_logits"].float()
        raw = torch.cat(
            (
                policy.new_zeros((local_batch, 1)),
                policy.reshape(local_batch, slots * candidates),
            ),
            dim=1,
        )
        full_valid = batch["full_action_valid"].bool()
        raw = raw.masked_fill(~full_valid, -1.0e4)
        deployed = torch.zeros(local_batch, dtype=torch.long, device=device)
        replace = output["replace"].bool()
        deployed[replace] = (
            1
            + output["replace_slot"][replace].long() * candidates
            + output["replace_candidate"][replace].long()
        )
        collected["raw_action_scores"].append(raw.cpu())
        collected["deployed_action"].append(deployed.cpu())
        collected["expected_delta50"].append(
            output["expected_delta50"].float().cpu()
        )
        collected["expected_delta75"].append(
            output["expected_delta75"].float().cpu()
        )
        collected["duplicate_probability"].append(
            torch.sigmoid(output["duplicate_logits"].float()).cpu()
        )
        collected["abandon_probability"].append(
            torch.sigmoid(output["abandon_logits"].float()).cpu()
        )
    return {name: torch.cat(values, dim=0) for name, values in collected.items()}


def context_comparison(
    correct: dict[str, torch.Tensor],
    masked: dict[str, torch.Tensor],
    full_valid: torch.Tensor,
) -> dict[str, float | int]:
    valid = full_valid[:, 1:].bool()
    correct_delta50 = correct["expected_delta50"].flatten(1)
    masked_delta50 = masked["expected_delta50"].flatten(1)
    correct_delta75 = correct["expected_delta75"].flatten(1)
    masked_delta75 = masked["expected_delta75"].flatten(1)
    correct_duplicate = correct["duplicate_probability"].flatten(1)
    masked_duplicate = masked["duplicate_probability"].flatten(1)
    correct_abandon = correct["abandon_probability"].flatten(1)
    masked_abandon = masked["abandon_probability"].flatten(1)
    difference = (
        correct["raw_action_scores"][:, 1:]
        - masked["raw_action_scores"][:, 1:]
    ).abs()
    correct_top = correct["raw_action_scores"][:, 1:].argmax(dim=1)
    masked_top = masked["raw_action_scores"][:, 1:].argmax(dim=1)
    return {
        "valid_action_mean_absolute_policy_delta": float(difference[valid].mean()),
        "valid_action_max_absolute_policy_delta": float(difference[valid].max()),
        "valid_action_mean_absolute_delta50_prediction_change": float(
            (correct_delta50 - masked_delta50).abs()[valid].mean()
        ),
        "valid_action_mean_absolute_delta75_prediction_change": float(
            (correct_delta75 - masked_delta75).abs()[valid].mean()
        ),
        "valid_action_mean_absolute_duplicate_probability_change": float(
            (correct_duplicate - masked_duplicate).abs()[valid].mean()
        ),
        "valid_action_mean_absolute_abandon_probability_change": float(
            (correct_abandon - masked_abandon).abs()[valid].mean()
        ),
        "raw_top_action_changed_fraction": float((correct_top != masked_top).float().mean()),
        "deployed_action_changed_fraction": float(
            (correct["deployed_action"] != masked["deployed_action"]).float().mean()
        ),
    }


def _parse_domains(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --domain value: {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in seen:
            raise ValueError(f"invalid or duplicate domain name: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(name)
        parsed.append((name, path))
    return parsed


def _fmt_percent(value: float) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{100.0 * float(value):.1f}%"


def _markdown(report: dict[str, Any]) -> str:
    rows = [
        "# V20 Decision-Sufficiency Autopsy",
        "",
        "Training-free fixed-endpoint diagnostic. No threshold/checkpoint selection was performed.",
        "",
        "| Domain | Edit AP | Action AP | Slot top-1 | Candidate top-1/top-5 | Deployed B/H/N | Deployed beneficial precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, payload in report["domains"].items():
        metrics = payload["variants"]["treatment_correct_context"]
        outcome = metrics["configured_deployment_outcome"]
        split = metrics["slot_and_candidate"]["any_beneficial_action"]
        rows.append(
            "| {domain} | {edit_ap:.3f} | {action_ap:.3f} | {slot} | {candidate1}/{candidate5} | {beneficial}/{harmful}/{neutral} | {precision} |".format(
                domain=domain,
                edit_ap=metrics["edit_detection"]["average_precision"],
                action_ap=metrics["action_ranking"]["beneficial_average_precision"],
                slot=_fmt_percent(split["slot_top1_hit"]),
                candidate1=_fmt_percent(split["candidate_top1_hit"]),
                candidate5=_fmt_percent(split["candidate_top5_recall"]),
                beneficial=outcome["beneficial"],
                harmful=outcome["harmful"],
                neutral=outcome["neutral"],
                precision=_fmt_percent(outcome["beneficial_precision"]),
            )
        )
    rows.extend(
        (
            "",
            "Fixed risk-coverage values (1%, 2%, 5%, 10%, 20%) are diagnostic only and were not used to choose a deployment threshold.",
            "",
            "The JSON report contains treatment-correct, treatment-masked, and separately trained control-masked results plus context-edge perturbations.",
            "",
        )
    )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 1:
        raise ValueError("batch size must be positive")
    device = torch.device(args.device)
    domains = _parse_domains(args.domain)
    treatment, treatment_iteration = _load_head(
        args.treatment_config, args.treatment_checkpoint, device
    )
    control, control_iteration = _load_head(
        args.control_config, args.control_checkpoint, device
    )
    report: dict[str, Any] = {
        "experiment": "V20 training-free per-action decision-sufficiency autopsy",
        "treatment_config": str(Path(args.treatment_config).resolve()),
        "control_config": str(Path(args.control_config).resolve()),
        "treatment_checkpoint": str(Path(args.treatment_checkpoint).resolve()),
        "control_checkpoint": str(Path(args.control_checkpoint).resolve()),
        "treatment_checkpoint_sha256": sha256_file(args.treatment_checkpoint),
        "control_checkpoint_sha256": sha256_file(args.control_checkpoint),
        "treatment_iteration": treatment_iteration,
        "control_iteration": control_iteration,
        "fixed_risk_coverage": list(FIXED_RISK_COVERAGE),
        "contract": {
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "nms_search_performed": False,
            "full_validation_executed": False,
            "test_set_used": False,
        },
        "domains": {},
    }
    for name, manifest_path in domains:
        cache, manifest = _load_cache(manifest_path)
        treatment_correct = score_cache(
            treatment,
            cache,
            device=device,
            batch_size=int(args.batch_size),
            context_mode="treatment",
        )
        treatment_masked = score_cache(
            treatment,
            cache,
            device=device,
            batch_size=int(args.batch_size),
            context_mode="masked",
        )
        control_masked = score_cache(
            control,
            cache,
            device=device,
            batch_size=int(args.batch_size),
            context_mode="masked",
        )
        report["domains"][name] = {
            "cache_manifest": str(manifest_path),
            "cache_manifest_sha256": sha256_file(manifest_path),
            "images": int(manifest["images"]),
            "variants": {
                "treatment_correct_context": decision_metrics(
                    cache, treatment_correct
                ),
                "treatment_masked_context": decision_metrics(
                    cache, treatment_masked
                ),
                "control_masked_context": decision_metrics(
                    cache, control_masked
                ),
            },
            "treatment_context_edge": context_comparison(
                treatment_correct,
                treatment_masked,
                cache["full_action_valid"],
            ),
        }
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
