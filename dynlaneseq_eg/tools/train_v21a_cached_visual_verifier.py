from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn import functional as F

from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.modeling.v21a_pairwise_visual_verifier import (
    PairwiseVisualLaneVerifier,
)
from dynlaneseq_eg.tools.cache_v21a_pairwise_visual_verification import CASE_FIELDS
from dynlaneseq_eg.tools.train import seed_everything


FIXED_STEPS = 4000
FIXED_BATCH_SIZE = 64
FIXED_SEED = 3407
FIXED_TOP1_MINIMUM = 0.65
FIXED_DECISIVE_PRECISION_MINIMUM = 0.70
FIXED_VISUAL_ADVANTAGE_MINIMUM = 0.10
FIXED_ANTISYMMETRY_MINIMUM = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the fixed V21A oracle-slot top5 visual verifier and its "
            "equal-parameter geometry-only control. This is not deployment."
        )
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument(
        "--eval-domain",
        action="append",
        required=True,
        help="NAME=manifest.json; may be repeated",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=FIXED_STEPS)
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def _resolve_shard(path: str, manifest_path: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    local = manifest_path.parent / candidate.name
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError(f"V21A cache shard is unavailable: {path}")


def load_cache(manifest_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract", {}).get("passed") is not True:
        raise ValueError("V21A cache contract did not pass")
    pieces: dict[str, list[torch.Tensor]] = {name: [] for name in CASE_FIELDS}
    image_ids: list[str] = []
    for item in manifest.get("shards", []):
        path = _resolve_shard(str(item["path"]), manifest_path)
        if sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"V21A cache shard digest mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        image_ids.extend(str(value) for value in payload["image_ids"])
        for name in CASE_FIELDS:
            pieces[name].append(payload[name])
    cache = {name: torch.cat(values, dim=0) for name, values in pieces.items()}
    if int(cache["slot_index"].shape[0]) != int(manifest["cases"]):
        raise ValueError("V21A cache case count mismatch")
    manifest["case_image_ids"] = image_ids
    return cache, manifest


def _batch_indices(cases: int, batch_size: int, step: int, seed: int) -> torch.Tensor:
    batches = math.ceil(float(cases) / float(batch_size))
    epoch = int(step) // batches
    position = int(step) % batches
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1_000_003 * epoch)
    permutation = torch.randperm(cases, generator=generator)
    start = position * int(batch_size)
    selected = permutation[start : min(start + int(batch_size), cases)]
    if int(selected.numel()) < int(batch_size):
        selected = torch.cat(
            (selected, permutation[: int(batch_size) - int(selected.numel())])
        )
    return selected


def _to_device(
    cache: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value.index_select(0, indices).to(device, non_blocking=True)
        for name, value in cache.items()
    }


def _forward(
    model: PairwiseVisualLaneVerifier,
    batch: dict[str, torch.Tensor],
    *,
    profile_mode: str,
    return_swapped: bool = False,
) -> dict[str, torch.Tensor]:
    if profile_mode == "correct":
        source_profile = batch["source_profile_correct"]
        candidate_profile = batch["candidate_profile_correct"]
    elif profile_mode == "wrong":
        source_profile = batch["source_profile_wrong"]
        candidate_profile = batch["candidate_profile_wrong"]
    elif profile_mode == "zero":
        source_profile = torch.zeros_like(batch["source_profile_correct"])
        candidate_profile = torch.zeros_like(batch["candidate_profile_correct"])
    else:
        raise ValueError(f"unknown V21A profile mode: {profile_mode!r}")
    return model(
        source_profile=source_profile,
        candidate_profile=candidate_profile,
        source_row_weight=batch["source_row_weight"],
        candidate_row_weight=batch["candidate_row_weight"],
        source_state=batch["source_state"],
        candidate_state=batch["candidate_state"],
        source_scalar=batch["source_scalar"],
        candidate_scalar=batch["candidate_scalar"],
        candidate_to_source_relation=batch["candidate_to_source_relation"],
        candidate_valid=batch["candidate_valid"].bool(),
        return_swapped=return_swapped,
    )


def verifier_loss(
    score: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = batch["candidate_valid"].bool()
    positive = batch["beneficial"].bool() & valid
    negative = ~positive & valid
    positive_case = positive.any(dim=1)
    masked = score.float().masked_fill(~valid, -1.0e4)
    positive_score = masked.masked_fill(~positive, -1.0e4)
    listwise_per_case = torch.logsumexp(masked, dim=1) - torch.logsumexp(
        positive_score, dim=1
    )
    listwise = (
        listwise_per_case[positive_case].mean()
        if bool(positive_case.any())
        else masked.sum() * 0.0
    )

    hard_positive = positive_score.max(dim=1).values
    hard_negative = masked.masked_fill(~negative, -1.0e4).max(dim=1).values
    rank_case = positive_case & negative.any(dim=1)
    hard_rank = (
        F.softplus(0.5 + hard_negative[rank_case] - hard_positive[rank_case]).mean()
        if bool(rank_case.any())
        else masked.sum() * 0.0
    )

    binary_loss = F.binary_cross_entropy_with_logits(
        score.float(), positive.float(), reduction="none"
    )
    positive_binary = (
        binary_loss[positive].mean() if bool(positive.any()) else masked.sum() * 0.0
    )
    negative_binary = (
        binary_loss[negative].mean() if bool(negative.any()) else masked.sum() * 0.0
    )
    binary = 0.5 * (positive_binary + negative_binary)
    total = listwise + 0.5 * hard_rank + 0.25 * binary
    with torch.no_grad():
        selected = masked.argmax(dim=1)
        selected_positive = positive.gather(1, selected.unsqueeze(1)).squeeze(1)
        diagnostics = {
            "total": total.detach(),
            "listwise": listwise.detach(),
            "hard_rank": hard_rank.detach(),
            "binary": binary.detach(),
            "positive_case_fraction": positive_case.float().mean(),
            "conditional_top1": (
                selected_positive[positive_case].float().mean()
                if bool(positive_case.any())
                else total.detach().new_tensor(float("nan"))
            ),
        }
    return total, diagnostics


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


@torch.no_grad()
def evaluate_mode(
    model: PairwiseVisualLaneVerifier,
    cache: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    profile_mode: str,
    audit_antisymmetry: bool,
) -> dict[str, Any]:
    model.eval()
    scores: list[torch.Tensor] = []
    swapped_scores: list[torch.Tensor] = []
    cases = int(cache["slot_index"].shape[0])
    for start in range(0, cases, int(batch_size)):
        stop = min(start + int(batch_size), cases)
        indices = torch.arange(start, stop)
        batch = _to_device(cache, indices, device)
        output = _forward(
            model,
            batch,
            profile_mode=profile_mode,
            return_swapped=audit_antisymmetry,
        )
        scores.append(output["score"].float().cpu())
        if audit_antisymmetry:
            swapped_scores.append(output["swapped_score"].float().cpu())
    score = torch.cat(scores)
    valid = cache["candidate_valid"].bool()
    beneficial = cache["beneficial"].bool() & valid
    harmful = cache["harmful"].bool() & valid
    neutral = cache["neutral"].bool() & valid
    positive_case = beneficial.any(dim=1)
    selected = score.masked_fill(~valid, -1.0e4).argmax(dim=1)
    selected_beneficial = beneficial.gather(1, selected.unsqueeze(1)).squeeze(1)
    selected_harmful = harmful.gather(1, selected.unsqueeze(1)).squeeze(1)
    selected_neutral = neutral.gather(1, selected.unsqueeze(1)).squeeze(1)
    conditional = positive_case
    beneficial_count = int((selected_beneficial & conditional).sum())
    harmful_count = int((selected_harmful & conditional).sum())
    neutral_count = int((selected_neutral & conditional).sum())
    decisive = beneficial_count + harmful_count
    result: dict[str, Any] = {
        "cases": cases,
        "cases_with_beneficial_in_top5": int(positive_case.sum()),
        "beneficial_top5_case_coverage": float(positive_case.float().mean()),
        "conditional_top1_beneficial": (
            float(selected_beneficial[conditional].float().mean())
            if bool(conditional.any())
            else float("nan")
        ),
        "overall_beneficial_recovery": float(selected_beneficial.float().mean()),
        "conditional_top1_outcome": {
            "beneficial": beneficial_count,
            "harmful": harmful_count,
            "neutral": neutral_count,
        },
        "conditional_decisive_precision": (
            float(beneficial_count) / float(decisive) if decisive else 0.0
        ),
        "beneficial_action_average_precision": _average_precision(
            score[valid], beneficial[valid]
        ),
        "v20_shortlist_first_conditional_top1": (
            float(beneficial[conditional, 0].float().mean())
            if bool(conditional.any())
            else float("nan")
        ),
    }
    if audit_antisymmetry:
        swapped = torch.cat(swapped_scores)
        error = (score + swapped).abs()[valid]
        result["swap_antisymmetry"] = {
            "max_absolute_error": float(error.max()) if error.numel() else 0.0,
            "fraction_within_1e_6": (
                float((error <= 1.0e-6).float().mean()) if error.numel() else 1.0
            ),
        }
    return result


def _build_model(manifest: dict[str, Any], cache: dict[str, torch.Tensor]) -> PairwiseVisualLaneVerifier:
    return PairwiseVisualLaneVerifier(
        profile_channels=int(cache["source_profile_correct"].shape[-1]),
        state_dim=int(cache["source_state"].shape[-1]),
        scalar_dim=int(cache["source_scalar"].shape[-1]),
        relation_dim=int(cache["candidate_to_source_relation"].shape[-1]),
        rows=int(manifest["visual_evidence"]["curve_samples"]),
        offsets=len(manifest["visual_evidence"]["offsets_px"]),
        curve_dim=96,
        state_hidden_dim=64,
        scalar_hidden_dim=32,
        relation_hidden_dim=48,
        pair_hidden_dim=192,
        num_heads=4,
        ff_dim=256,
        vertical_layers=2,
    )


def _parse_domains(rows: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"invalid V21A eval-domain: {row!r}")
        name, path = row.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid/duplicate V21A domain: {name!r}")
        result[name] = Path(path).expanduser().resolve()
    return result


def main() -> None:
    args = parse_args()
    if int(args.steps) != FIXED_STEPS or int(args.batch_size) != FIXED_BATCH_SIZE:
        raise ValueError("V21A fixed gate requires exactly 4000 steps and batch 64")
    if int(args.seed) != FIXED_SEED:
        raise ValueError("V21A fixed gate requires seed 3407")
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    train_manifest_path = Path(args.train_cache).expanduser().resolve()
    train_cache, train_manifest = load_cache(train_manifest_path)
    eval_paths = _parse_domains(args.eval_domain)
    eval_caches: dict[str, tuple[dict[str, torch.Tensor], dict[str, Any]]] = {
        name: load_cache(path) for name, path in eval_paths.items()
    }
    expected_evidence = train_manifest["visual_evidence"]
    for name, (_cache, manifest) in eval_caches.items():
        if manifest["visual_evidence"] != expected_evidence:
            raise ValueError(f"V21A visual contract differs for domain {name}")

    seed_everything(int(args.seed))
    treatment = _build_model(train_manifest, train_cache).to(device)
    control = copy.deepcopy(treatment).to(device)
    treatment.train()
    control.train()
    treatment_optimizer = torch.optim.AdamW(
        treatment.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    control_optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_metrics.jsonl"
    if log_path.exists():
        log_path.unlink()
    cases = int(train_cache["slot_index"].shape[0])
    start_time = time.monotonic()
    for step in range(int(args.steps)):
        indices = _batch_indices(cases, int(args.batch_size), step, int(args.seed))
        batch = _to_device(train_cache, indices, device)
        rows: dict[str, dict[str, torch.Tensor]] = {}
        for name, model, optimizer, profile_mode in (
            ("treatment", treatment, treatment_optimizer, "correct"),
            ("geometry_control", control, control_optimizer, "zero"),
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = _forward(model, batch, profile_mode=profile_mode)
                loss, diagnostics = verifier_loss(output["score"], batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite V21A {name} loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError(f"non-finite V21A {name} gradient")
            optimizer.step()
            rows[name] = {**diagnostics, "gradient_norm": gradient.detach()}
        completed = step + 1
        if completed == 1 or completed % int(args.log_interval) == 0:
            elapsed = time.monotonic() - start_time
            record: dict[str, Any] = {
                "step": completed,
                "cases_per_second_per_arm": (
                    completed * int(args.batch_size) / max(elapsed, 1.0e-9)
                ),
            }
            for arm, metrics in rows.items():
                for key, value in metrics.items():
                    record[f"{arm}_{key}"] = float(value.detach().cpu())
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

    treatment_path = output_dir / "treatment_endpoint.pt"
    control_path = output_dir / "geometry_control_endpoint.pt"
    torch.save(
        {
            "model": treatment.state_dict(),
            "steps": int(args.steps),
            "seed": int(args.seed),
        },
        treatment_path,
    )
    torch.save(
        {
            "model": control.state_dict(),
            "steps": int(args.steps),
            "seed": int(args.seed),
        },
        control_path,
    )

    domains: dict[str, Any] = {}
    for name, (cache, manifest) in eval_caches.items():
        correct = evaluate_mode(
            treatment,
            cache,
            device=device,
            batch_size=int(args.eval_batch_size),
            profile_mode="correct",
            audit_antisymmetry=True,
        )
        wrong = evaluate_mode(
            treatment,
            cache,
            device=device,
            batch_size=int(args.eval_batch_size),
            profile_mode="wrong",
            audit_antisymmetry=False,
        )
        geometry = evaluate_mode(
            control,
            cache,
            device=device,
            batch_size=int(args.eval_batch_size),
            profile_mode="zero",
            audit_antisymmetry=True,
        )
        correct_top1 = float(correct["conditional_top1_beneficial"])
        domains[name] = {
            "cache_manifest": str(eval_paths[name]),
            "cache_manifest_sha256": sha256_file(eval_paths[name]),
            "correct_image_treatment": correct,
            "cross_clip_wrong_image_replay": wrong,
            "geometry_only_control": geometry,
            "deltas": {
                "correct_minus_wrong_top1_points": correct_top1
                - float(wrong["conditional_top1_beneficial"]),
                "correct_minus_geometry_control_top1_points": correct_top1
                - float(geometry["conditional_top1_beneficial"]),
            },
        }

    unseen_names = ("heldout256", "validation256")
    if any(name not in domains for name in unseen_names):
        raise ValueError("V21A formal gate requires heldout256 and validation256")
    gate_checks: dict[str, bool] = {}
    for name in unseen_names:
        row = domains[name]
        correct = row["correct_image_treatment"]
        gate_checks[f"{name}_top1_at_least_0p65"] = (
            float(correct["conditional_top1_beneficial"]) >= FIXED_TOP1_MINIMUM
        )
        gate_checks[f"{name}_decisive_precision_at_least_0p70"] = (
            float(correct["conditional_decisive_precision"])
            >= FIXED_DECISIVE_PRECISION_MINIMUM
        )
        gate_checks[f"{name}_correct_over_wrong_at_least_0p10"] = (
            float(row["deltas"]["correct_minus_wrong_top1_points"])
            >= FIXED_VISUAL_ADVANTAGE_MINIMUM
        )
        gate_checks[f"{name}_correct_over_geometry_at_least_0p10"] = (
            float(row["deltas"]["correct_minus_geometry_control_top1_points"])
            >= FIXED_VISUAL_ADVANTAGE_MINIMUM
        )
        gate_checks[f"{name}_swap_antisymmetry_at_least_0p99"] = (
            float(correct["swap_antisymmetry"]["fraction_within_1e_6"])
            >= FIXED_ANTISYMMETRY_MINIMUM
        )
    passed = all(gate_checks.values())
    report = {
        "experiment": "V21A oracle-slot top5 pairwise visual verification sufficiency gate",
        "contract": {
            "diagnostic_only_not_deployable": True,
            "frozen_v7_v19_v20": True,
            "oracle_beneficial_slot_used": True,
            "candidate_shortlist": 5,
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "full_validation_executed": False,
            "test_set_used": False,
        },
        "fixed_gate": {
            "conditional_top1_minimum": FIXED_TOP1_MINIMUM,
            "conditional_decisive_precision_minimum": FIXED_DECISIVE_PRECISION_MINIMUM,
            "correct_over_wrong_top1_minimum_points": FIXED_VISUAL_ADVANTAGE_MINIMUM,
            "correct_over_geometry_top1_minimum_points": FIXED_VISUAL_ADVANTAGE_MINIMUM,
            "swap_antisymmetry_minimum": FIXED_ANTISYMMETRY_MINIMUM,
        },
        "train_cache_manifest": str(train_manifest_path),
        "train_cache_manifest_sha256": sha256_file(train_manifest_path),
        "treatment_checkpoint": str(treatment_path),
        "treatment_checkpoint_sha256": sha256_file(treatment_path),
        "geometry_control_checkpoint": str(control_path),
        "geometry_control_checkpoint_sha256": sha256_file(control_path),
        "domains": domains,
        "gate_checks": gate_checks,
        "passed": passed,
        "recommendation": (
            "stop_and_review_before_any_deployment_v21b"
            if passed
            else "stop_pairwise_single_frame_visual_verification_failed"
        ),
    }
    report_path = output_dir / "v21a_pairwise_visual_verification_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        "V21A diagnostic complete. No deployment, full validation, test, "
        "checkpoint selection or threshold search was started."
    )


if __name__ == "__main__":
    main()
