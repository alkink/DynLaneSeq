from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import V25LossWeights
from dynlaneseq_eg.tools.summarize_v33_private_head_gate import build
from dynlaneseq_eg.tools.train_v33_private_auxiliary_head import (
    auxiliary_only_weights,
    private_parameter_partition,
)


def _weights() -> V25LossWeights:
    return V25LossWeights(
        existence=2.0,
        row_distribution=5.0,
        point=1.0,
        strip_iou=2.0,
        range=1.0,
        quality50=0.0,
        quality75=0.0,
        smoothness=0.25,
        order=0.25,
        duplicate=0.25,
        visibility=0.0,
        proposal_coverage=1.0,
    )


def test_private_auxiliary_weights_and_parameter_ownership() -> None:
    weights = auxiliary_only_weights(_weights())
    assert weights.proposal_coverage == 1.0
    assert weights.row_distribution == 0.0
    assert weights.existence == 0.0

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detector = torch.nn.Module()
            self.detector.proposal_memory = torch.nn.Linear(2, 2)
            self.detector.backbone = torch.nn.Linear(2, 2)

    model = Tiny()
    private, frozen = private_parameter_partition(model)
    assert private
    assert frozen
    assert all(parameter.requires_grad for _name, parameter in private)
    assert all(not parameter.requires_grad for _name, parameter in frozen)
    assert all(name.startswith("detector.proposal_memory") for name, _ in private)


def _row(f1: float, tp: int) -> dict[str, float | int]:
    return {
        "F1": f1,
        "TP": tp,
        "FP": 1000 - tp,
        "FN": 1200 - tp,
        "Precision": 0.7,
        "Recall": 0.7,
    }


def _pair(control50: float, control75: float, treatment50: float, treatment75: float) -> dict:
    return {
        "metrics": {
            "control": {
                "0.5": _row(control50, 700),
                "0.75": _row(control75, 500),
            },
            "image_ownership": {
                "0.5": _row(treatment50, 710),
                "0.75": _row(treatment75, 510),
            },
        }
    }


def test_summary_passes_only_when_mechanism_and_f1_pass() -> None:
    a_vs_d = _pair(0.700, 0.500, 0.705, 0.505)
    b_vs_d = _pair(0.699, 0.504, 0.705, 0.505)
    # D must replay exactly in both pair reports.
    b_vs_d["metrics"]["image_ownership"] = a_vs_d["metrics"]["image_ownership"]
    private = {
        "frozen_parent_state_exact": True,
        "shared_updates": 0,
        "gradient_ownership_gate": {"passed": True},
    }
    audit = {
        "hybrid_parent_with_trained_private_auxiliary": {
            "verdict": {"label": "aligned_or_redundant"},
            "summary": {
                "gradient_groups": {
                    "all_shared": {"cosine": {"median": 0.65}}
                },
                "cross_batch_virtual_steps": {
                    "1.0e-04": {"auxiliary_direction_harm_fraction": 0.55}
                },
            },
        }
    }
    result = build(a_vs_d, b_vs_d, private, audit)
    assert result["mechanism_gate"]["passed"] is True
    assert result["f1_gate"]["passed"] is True
    assert result["verdict"] == "private_head_firewall_improves_primary"

