from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v28 import DynLaneSeqV28
from dynlaneseq_eg.modeling.v23_ordered_slot_cost_volume import (
    build_v23_owned_targets,
)
from dynlaneseq_eg.modeling.v28_refined_belief_router import (
    build_v28_refined_route_targets,
)
from dynlaneseq_eg.tools.evaluate_v28_refined_belief_gate import (
    SEED,
    _belief_public,
    _configured,
    _move_images,
    _validate_endpoint,
    _validate_v29_support,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free V28 arm-C audit: test whether the selected-vs-source "
            "score margin separates beneficial and harmful route changes."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm-b-checkpoint", required=True)
    parser.add_argument("--arm-b-training-report", required=True)
    parser.add_argument("--arm-c-router-checkpoint", required=True)
    parser.add_argument("--arm-c-training-report", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--oof-fold", choices=("a", "b"), default="")
    parser.add_argument("--train-list-contract", default="")
    parser.add_argument("--expected-v7-checkpoint", default="")
    parser.add_argument("--expected-v7-iteration", type=int, default=112_500)
    parser.add_argument("--support-training-report", default="")
    return parser.parse_args()


def _binary_auc(score: np.ndarray, positive: np.ndarray) -> float | None:
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    finite = np.isfinite(score)
    score = score[finite]
    positive = positive[finite]
    positives = int(positive.sum())
    negatives = int((~positive).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(score, method="average")
    value = (
        float(ranks[positive].sum()) - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)
    return float(value)


def _margin_auc(
    rows: list[dict[str, Any]],
    *,
    positive_key: str,
    negative_key: str,
) -> dict[str, Any]:
    relevant = [
        row
        for row in rows
        if bool(row[positive_key]) or bool(row[negative_key])
    ]
    scores = np.asarray([row["selected_source_margin"] for row in relevant])
    labels = np.asarray([bool(row[positive_key]) for row in relevant])
    return {
        "auc": _binary_auc(scores, labels),
        "beneficial": int(labels.sum()),
        "harmful": int((~labels).sum()),
        "population": len(relevant),
    }


def _risk_curve(
    rows: list[dict[str, Any]], *, threshold: float
) -> dict[str, Any]:
    matched = [row for row in rows if bool(row["matched"])]
    source_positive = sum(float(row["source_quality"]) >= threshold for row in matched)
    changed = [row for row in matched if bool(row["changed"])]
    margins = np.asarray(
        [float(row["selected_source_margin"]) for row in changed],
        dtype=np.float64,
    )
    gain = np.asarray(
        [
            float(row["source_quality"]) < threshold
            and float(row["selected_quality"]) >= threshold
            for row in changed
        ],
        dtype=np.int64,
    )
    loss = np.asarray(
        [
            float(row["source_quality"]) >= threshold
            and float(row["selected_quality"]) < threshold
            for row in changed
        ],
        dtype=np.int64,
    )
    all_gain = int(gain.sum())
    best: dict[str, Any] = {
        "margin_threshold": None,
        "switched": 0,
        "gained": 0,
        "lost": 0,
        "net": 0,
        "harmful_rate": 0.0,
        "beneficial_recall_of_arm_c_gains": 0.0,
    }
    if margins.size:
        order = np.argsort(-margins, kind="stable")
        margins = margins[order]
        cumulative_gain = np.cumsum(gain[order])
        cumulative_loss = np.cumsum(loss[order])
        group_end = np.flatnonzero(
            np.r_[margins[1:] != margins[:-1], True]
        )
        # Each point applies all changes whose margin is at least the score at
        # that tie-group boundary.  This computes the same diagnostic envelope
        # as the former nested loop in O(N log N), dominated by sorting.
        for index in group_end.tolist():
            gained = int(cumulative_gain[index])
            lost = int(cumulative_loss[index])
            harmful_rate = lost / max(source_positive, 1)
            candidate = {
                "margin_threshold": float(margins[index]),
                "switched": int(index + 1),
                "gained": gained,
                "lost": lost,
                "net": gained - lost,
                "harmful_rate": harmful_rate,
                "beneficial_recall_of_arm_c_gains": gained / max(all_gain, 1),
            }
            if harmful_rate <= 0.01 and (
                (candidate["net"], candidate["gained"])
                > (best["net"], best["gained"])
            ):
                best = candidate
    return {
        "threshold": threshold,
        "source_positive": source_positive,
        "arm_c_available_gains": all_gain,
        "best_validation_diagnostic_at_harmful_rate_le_1pct": best,
        "warning": (
            "Post-hoc validation diagnostic only; threshold must not be deployed "
            "or reported as a new validation result."
        ),
    }


@torch.inference_mode()
def _collect(
    model: DynLaneSeqV28,
    arm_c_router: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    channels_last: bool,
    input_h: int,
    input_w: int,
    log_interval: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.requires_grad_(False).eval().prepare_for_inference()
    arm_c_router.requires_grad_(False).eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    images_seen = 0
    for batch_index, (images, targets, metas) in enumerate(loader, start=1):
        images = _move_images(images, device=device, channels_last=channels_last)
        with torch.autocast(device_type=device.type, enabled=False):
            _teacher, bank = model._frozen_bank(images.float())
            _public, output = _belief_public(
                model, arm_c_router, images.float(), bank
            )
            owned = build_v23_owned_targets(
                targets,
                source_x=bank["source_x"],
                source_range=bank["source_range"],
                source_active=bank["source_active"],
                input_h=input_h,
                input_w=input_w,
                minimum_valid_rows=5,
            )
            target = build_v28_refined_route_targets(
                candidate_x=bank["candidate_x"],
                candidate_range=bank["candidate_range"],
                candidate_valid=bank["candidate_valid"],
                owned_x=owned["x_rows"],
                owned_valid=owned["valid_mask"],
                owned_matched=owned["matched"],
                input_h=input_h,
                line_width=30.0,
                minimum_valid_rows=5,
                temperature=0.05,
                support_delta=0.05,
                support_floor=0.0,
            )

        score = output["candidate_scores"].float()
        probability = score.masked_fill(~target["valid"], -1.0e4).softmax(dim=-1)
        selected = output["selected_route"]
        source = bank["source_route"].clamp_min(0)
        best = target["quality"].masked_fill(~target["valid"], -1.0e4).argmax(-1)
        selected_score = score.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        source_score = score.gather(-1, source.unsqueeze(-1)).squeeze(-1)
        selected_quality = target["quality"].gather(
            -1, selected.unsqueeze(-1)
        ).squeeze(-1)
        source_quality = target["quality"].gather(
            -1, source.unsqueeze(-1)
        ).squeeze(-1)
        best_quality = target["quality"].gather(
            -1, best.unsqueeze(-1)
        ).squeeze(-1)
        source_probability = probability.gather(
            -1, source.unsqueeze(-1)
        ).squeeze(-1)
        selected_probability = probability.gather(
            -1, selected.unsqueeze(-1)
        ).squeeze(-1)

        for batch_offset, meta in enumerate(metas):
            for slot in range(int(selected.shape[1])):
                matched = bool(target["matched"][batch_offset, slot].item())
                src_q = float(source_quality[batch_offset, slot].item())
                sel_q = float(selected_quality[batch_offset, slot].item())
                changed = bool(
                    selected[batch_offset, slot].item()
                    != source[batch_offset, slot].item()
                )
                rows.append(
                    {
                        "image_path": str(meta["image_path"]),
                        "slot": slot,
                        "matched": matched,
                        "changed": changed,
                        "source_route": int(source[batch_offset, slot].item()),
                        "selected_route": int(selected[batch_offset, slot].item()),
                        "best_route": int(best[batch_offset, slot].item()),
                        "source_quality": src_q,
                        "selected_quality": sel_q,
                        "best_quality": float(
                            best_quality[batch_offset, slot].item()
                        ),
                        "selected_source_margin": float(
                            selected_score[batch_offset, slot].item()
                            - source_score[batch_offset, slot].item()
                        ),
                        "selected_probability": float(
                            selected_probability[batch_offset, slot].item()
                        ),
                        "source_probability": float(
                            source_probability[batch_offset, slot].item()
                        ),
                        "quality_delta": sel_q - src_q,
                        "quality_beneficial_0p01": changed
                        and sel_q > src_q + 0.01,
                        "quality_harmful_0p01": changed
                        and sel_q < src_q - 0.01,
                        "beneficial_0p50": changed
                        and src_q < 0.50
                        and sel_q >= 0.50,
                        "harmful_0p50": changed
                        and src_q >= 0.50
                        and sel_q < 0.50,
                        "beneficial_0p75": changed
                        and src_q < 0.75
                        and sel_q >= 0.75,
                        "harmful_0p75": changed
                        and src_q >= 0.75
                        and sel_q < 0.75,
                    }
                )
        images_seen += int(images.shape[0])
        if batch_index == 1 or (
            log_interval > 0 and batch_index % log_interval == 0
        ):
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            print(
                json.dumps(
                    {
                        "phase": "v28_switch_confidence_audit",
                        "batches": batch_index,
                        "images": images_seen,
                        "images_per_second": images_seen / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return rows, {
        "images": images_seen,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(dataset_root, split="val")
    source_list = Path(population["list_path"])
    expected = int(population["expected_nonempty_rows"])
    arm_b_checkpoint = Path(args.arm_b_checkpoint).expanduser().resolve()
    arm_c_checkpoint = Path(args.arm_c_router_checkpoint).expanduser().resolve()
    oof_values = (
        bool(args.oof_fold),
        bool(args.train_list_contract),
        bool(args.expected_v7_checkpoint),
        bool(args.support_training_report),
    )
    if any(oof_values) and not all(oof_values):
        raise ValueError(
            "V29 OOF audit requires --oof-fold, --train-list-contract, "
            "--expected-v7-checkpoint, and --support-training-report together"
        )
    oof_contract = None
    endpoint_kwargs: dict[str, Any] = {}
    if all(oof_values):
        fold_contract_path = Path(args.train_list_contract).expanduser().resolve()
        expected_v7 = Path(args.expected_v7_checkpoint).expanduser().resolve()
        oof_contract = _validate_v29_support(
            checkpoint=expected_v7,
            report_path=Path(args.support_training_report).expanduser().resolve(),
            fold_contract_path=fold_contract_path,
            belief_fold=str(args.oof_fold),
            expected_iteration=int(args.expected_v7_iteration),
        )
        endpoint_kwargs = {
            "oof_fold": str(args.oof_fold),
            "fold_contract_sha256": sha256_file(fold_contract_path),
            "expected_v7_sha256": sha256_file(expected_v7),
            "expected_v7_iteration": int(args.expected_v7_iteration),
        }
    endpoint_contracts = {
        "arm_b": _validate_endpoint(
            arm_b_checkpoint,
            Path(args.arm_b_training_report).expanduser().resolve(),
            arm="B",
            router_only=False,
            **endpoint_kwargs,
        ),
        "arm_c": _validate_endpoint(
            arm_c_checkpoint,
            Path(args.arm_c_training_report).expanduser().resolve(),
            arm="C",
            router_only=True,
            **endpoint_kwargs,
        ),
    }

    base_cfg: dict[str, Any] = load_config(args.config)
    cfg = _configured(
        base_cfg,
        dataset_root=dataset_root,
        list_path=source_list,
        batch_size=int(args.eval_batch_size),
        workers=int(args.num_workers),
        load_targets=True,
    )
    loader = build_dataloader(cfg, split="val", training=False)
    if len(loader.dataset) != expected:
        raise ValueError("V28 switch audit altered official validation population")

    device = torch.device(args.device)
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV28):
        raise TypeError("V28 switch audit did not build DynLaneSeqV28")
    if int(load_checkpoint(arm_b_checkpoint, model, strict=True)) != 6_000:
        raise ValueError("V28 switch audit requires the fixed B endpoint")
    arm_c_router = copy.deepcopy(model.router)
    if int(load_checkpoint(arm_c_checkpoint, arm_c_router, strict=True)) != 6_000:
        raise ValueError("V28 switch audit requires the fixed C endpoint")
    model.to(device)
    arm_c_router.to(device)
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model.to(memory_format=torch.channels_last)
        arm_c_router.to(memory_format=torch.channels_last)

    rows, runtime = _collect(
        model,
        arm_c_router,
        loader,
        device=device,
        channels_last=channels_last,
        input_h=int(cfg["model"]["input_h"]),
        input_w=int(cfg["model"]["input_w"]),
        log_interval=int(args.log_interval),
    )
    matched = [row for row in rows if bool(row["matched"])]
    changed = [row for row in matched if bool(row["changed"])]
    quality_relevant = [
        row
        for row in changed
        if bool(row["quality_beneficial_0p01"])
        or bool(row["quality_harmful_0p01"])
    ]
    report = {
        "experiment": "V28 arm-C switch-confidence audit",
        "contract": {
            "training_free": True,
            "full_official_validation": True,
            "threshold_selection_performed": False,
            "test_set_used": False,
            "posthoc_risk_curve_is_diagnostic_only": True,
        },
        "endpoint_contracts": endpoint_contracts,
        "oof_support_contract": oof_contract,
        "official_validation_population_contract": population,
        "runtime": runtime,
        "population": {
            "slot_rows": len(rows),
            "matched_slots": len(matched),
            "changed_slots": len(changed),
            "changed_fraction": len(changed) / max(len(matched), 1),
        },
        "margin_auc": {
            "quality_delta_0p01": _margin_auc(
                quality_relevant,
                positive_key="quality_beneficial_0p01",
                negative_key="quality_harmful_0p01",
            ),
            "threshold_0p50": _margin_auc(
                changed,
                positive_key="beneficial_0p50",
                negative_key="harmful_0p50",
            ),
            "threshold_0p75": _margin_auc(
                changed,
                positive_key="beneficial_0p75",
                negative_key="harmful_0p75",
            ),
        },
        "risk_curve": {
            "0.50": _risk_curve(matched, threshold=0.50),
            "0.75": _risk_curve(matched, threshold=0.75),
        },
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "switch_confidence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "switch_confidence_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
