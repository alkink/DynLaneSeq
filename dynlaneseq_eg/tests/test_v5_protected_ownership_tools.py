from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from dynlaneseq_eg.tools.analyze_v5_ownership_representatives import (
    main as representative_main,
)
from dynlaneseq_eg.tools.summarize_v5_protected_ownership_gate import (
    main as summary_main,
)


def test_representative_analyzer_uses_direct_ownership_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "cache.pt"
    torch.save(
        {
            "records": [
                {
                    "stages": {
                        "main": {
                            "pred_x_rows": torch.zeros(4, 8),
                            "official_candidate_valid": torch.ones(
                                4,
                                dtype=torch.bool,
                            ),
                            "official_iou": torch.tensor(
                                [
                                    [0.90, 0.80, 0.05, 0.05],
                                    [0.05, 0.05, 0.85, 0.70],
                                ]
                            ),
                            "exist_logits": torch.tensor(
                                [
                                    [4.0, 0.0],
                                    [1.0, 0.0],
                                    [3.0, 0.0],
                                    [0.5, 0.0],
                                ]
                            ),
                        }
                    }
                }
            ]
        },
        cache_path,
    )
    report_path = tmp_path / "oracle.json"
    report_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "cache_path": str(cache_path),
                    "checkpoint": "iter_0025000.pt",
                    "num_records": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "representatives.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_v5_ownership_representatives",
            "--oracle-report",
            str(report_path),
            "--output-json",
            str(output_path),
        ],
    )
    representative_main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["representable_clusters"] == 2
    assert result["representative"]["top1_rate"] == 1.0
    assert result["representative"]["mean_official_iou_regret"] == 0.0


def _trajectory_report(path: Path, iteration: int, f1: float) -> None:
    rows = []
    for iou in (0.5, 0.75):
        for score_threshold in (0.5, -1.0):
            row = {
                "stage": "main",
                "strategy": "model_topk_nms",
                "top_k": 4,
                "iou_threshold": iou,
                "score_threshold": score_threshold,
                "recall": 0.80,
                f"official_iou_{iou:g}": {
                    "tp": 80,
                    "fp": 20,
                    "fn": 20,
                    "precision": 0.80,
                    "recall": 0.80,
                    "f1": f1,
                },
            }
            rows.append(row)
        rows.append(
            {
                "stage": "main",
                "strategy": "oracle_topk",
                "top_k": 4,
                "iou_threshold": iou,
                "score_threshold": None,
                "recall": 0.95,
            }
        )
    path.write_text(
        json.dumps(
            {
                "metadata": {"checkpoint": f"iter_{iteration:07d}.pt"},
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def _representative_report(
    path: Path,
    iteration: int,
    top1: float,
    *,
    gate: bool,
) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint": f"iter_{iteration:07d}.pt",
                "representative": {
                    "top1_rate": top1,
                    "top2_rate": 0.75 if gate else 0.60,
                    "mean_official_iou_regret": 0.09 if gate else 0.14,
                },
                "predeclared_gate": {
                    "top1_at_least_0p40": gate,
                    "top2_at_least_0p70": gate,
                    "mean_regret_at_most_0p10": gate,
                },
            }
        ),
        encoding="utf-8",
    )


def test_summary_promotes_only_when_representation_and_causal_gates_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    control_reports = []
    assignment_reports = []
    control_representatives = []
    assignment_representatives = []
    for iteration in (5000, 10000, 15000, 20000, 25000):
        control_report = tmp_path / f"control_{iteration}.json"
        assignment_report = tmp_path / f"assignment_{iteration}.json"
        control_representative = tmp_path / f"control_rep_{iteration}.json"
        assignment_representative = tmp_path / f"assignment_rep_{iteration}.json"
        _trajectory_report(control_report, iteration, 0.80)
        after_ramp = iteration > 10000
        _trajectory_report(
            assignment_report,
            iteration,
            0.81 if after_ramp else 0.80,
        )
        _representative_report(
            control_representative,
            iteration,
            0.35,
            gate=False,
        )
        _representative_report(
            assignment_representative,
            iteration,
            0.42 if after_ramp else 0.35,
            gate=after_ramp,
        )
        control_reports.append(str(control_report))
        assignment_reports.append(str(assignment_report))
        control_representatives.append(str(control_representative))
        assignment_representatives.append(str(assignment_representative))

    control_stability = tmp_path / "control_stability.json"
    assignment_stability = tmp_path / "assignment_stability.json"
    stability = {
        "gate": {"geometry_stable": True},
        "checkpoints": [
            {"summary": {"training_vs_official_owner_agreement": 0.5}}
        ],
    }
    control_stability.write_text(json.dumps(stability), encoding="utf-8")
    assignment_stability.write_text(json.dumps(stability), encoding="utf-8")
    output_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_v5_protected_ownership_gate",
            "--control-reports",
            *control_reports,
            "--assignment-reports",
            *assignment_reports,
            "--control-representatives",
            *control_representatives,
            "--assignment-representatives",
            *assignment_representatives,
            "--control-stability",
            str(control_stability),
            "--assignment-stability",
            str(assignment_stability),
            "--output-json",
            str(output_path),
        ],
    )
    summary_main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["gates"]["assignment_representation_pass"] is True
    assert result["gates"]["pre_ramp_equivalence_pass"] is True
    assert result["gates"]["causal_assignment_pass"] is True
    assert result["verdict"] == "promote_v5_b_to_full_validation"
