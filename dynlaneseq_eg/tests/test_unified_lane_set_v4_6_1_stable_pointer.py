from __future__ import annotations

import json
import sys
import warnings

import pytest
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_scheduler
from dynlaneseq_eg.tools.summarize_v4_6_1_stable_pointer_gate import (
    main as summarize_gate,
)


SOURCE_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_"
    "v4_6_remaining_cluster_mixture_pointer.yaml"
)
STABLE_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_"
    "v4_6_1_stable_pointer.yaml"
)


def test_v4_6_1_changes_only_optimizer_schedule_and_run_length() -> None:
    source = load_config(SOURCE_CONFIG)
    stable = load_config(STABLE_CONFIG)

    assert stable["model"] == source["model"]
    assert stable["loss"] == source["loss"]
    assert stable["matcher"] == source["matcher"]
    assert stable["augmentation"] == source["augmentation"]
    assert stable["dataset"] == source["dataset"]

    groups = {
        group["name"]: group for group in stable["optimizer"]["parameter_groups"]
    }
    assert groups["set_selection"]["lr"] == pytest.approx(3e-5)
    assert groups["final_semantic_score"]["lr"] == pytest.approx(1e-5)
    assert stable["scheduler"] == {
        "name": "cosine",
        "total_iters": 10000,
        "warmup_iters": 500,
        "min_lr_ratio": 0.10,
    }
    assert stable["training"]["max_iters"] == 10000
    assert stable["training"]["checkpoint_interval"] == 1000
    assert stable["training"]["checkpoint_include_optimizer"] is True
    assert stable["training"]["trainable_parameter_prefixes"] == source[
        "training"
    ]["trainable_parameter_prefixes"]


def test_v4_6_1_scheduler_is_a_local_warmup_and_cosine() -> None:
    cfg = load_config(STABLE_CONFIG)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 3e-5}])
    scheduler = build_scheduler(cfg, optimizer, total_iters=10000)
    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5 / 500.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler.step(499)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
        scheduler.step(10000)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)


def _coverage(f1_050: float, f1_075: float) -> dict:
    def metric(f1: float) -> dict:
        return {
            "f1": f1,
            "precision": f1,
            "recall": f1,
            "mean_selected_per_image": 3.2,
            "false_positive_breakdown": {
                "duplicate_fp": {"fraction_of_fp": 0.0}
            },
        }

    return {
        "methods": {
            "pointer_greedy": {
                "0.50": metric(f1_050),
                "0.75": metric(f1_075),
            }
        },
        "capacity": {
            "0.50": {"all_candidate_oracle": {"recall": 0.94}},
            "0.75": {"all_candidate_oracle": {"recall": 0.85}},
        },
    }


def _policy(early_hit: float) -> dict:
    return {
        "teacher_contract": {"target_mode": "remaining_cluster_mixture"},
        "audit": {
            "teacher_prefix_policy": {
                str(step): {
                    "candidate_support_hit_rate": (
                        early_hit if step < 4 else 0.47
                    ),
                    "mean_probability_mass_on_candidate_support": 0.35,
                    "mean_soft_target_cross_entropy": 2.0,
                }
                for step in range(1, 5)
            },
            "teacher_support": {
                "support_size": {"mean": 4.0},
                "target_entropy": {"mean": 1.0},
            },
        },
    }


def test_v4_6_1_summary_requires_a_stable_final_plateau(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.json"
    source_policy = tmp_path / "source_policy.json"
    aggressive = tmp_path / "aggressive.json"
    gradient = tmp_path / "gradient.json"
    output = tmp_path / "summary.json"
    source.write_text(json.dumps(_coverage(0.823, 0.633)), encoding="utf-8")
    source_policy.write_text(json.dumps(_policy(0.47)), encoding="utf-8")
    aggressive.write_text(
        json.dumps(
            {
                "best_diagnostic_checkpoint": {
                    "iteration": 107500,
                    "f1_050": 0.830,
                    "f1_075": 0.651,
                }
            }
        ),
        encoding="utf-8",
    )
    gradient.write_text(json.dumps({"passed": True}), encoding="utf-8")

    f1_050 = [0.826, 0.828, 0.829, 0.8295, 0.8290, 0.8292, 0.8290, 0.8291, 0.8293, 0.8292]
    f1_075 = [0.640, 0.644, 0.647, 0.650, 0.649, 0.649, 0.648, 0.648, 0.649, 0.648]
    trajectory_args: list[str] = []
    policy_args: list[str] = []
    for offset, iteration in enumerate(range(106000, 115001, 1000)):
        metric_path = tmp_path / f"metric_{iteration}.json"
        policy_path = tmp_path / f"policy_{iteration}.json"
        metric_path.write_text(
            json.dumps(_coverage(f1_050[offset], f1_075[offset])),
            encoding="utf-8",
        )
        policy_path.write_text(json.dumps(_policy(0.52)), encoding="utf-8")
        trajectory_args.append(f"{iteration}={metric_path}")
        policy_args.append(f"{iteration}={policy_path}")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--source",
            str(source),
            "--source-policy",
            str(source_policy),
            "--aggressive-summary",
            str(aggressive),
            "--gradient-audit",
            str(gradient),
            "--trajectory",
            *trajectory_args,
            "--policy-trajectory",
            *policy_args,
            "--output-json",
            str(output),
        ],
    )
    summarize_gate()
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["selected_full_validation_iteration"] == 115000
    assert summary["tail_plateau_iterations"] == [113000, 114000, 115000]
    assert summary["best_diagnostic_checkpoint"]["iteration"] == 109000
