from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_refinement_arbitration import (
    _binary_signal_summary,
    _candidate_uncertainty_signals,
    _roc_auc,
    _uncertainty_gate,
)


def test_uncertainty_gate_respects_model_score_eligibility() -> None:
    uncertainty = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
    model_score = torch.tensor([[0.1, 0.9, 0.8, 0.7]])
    gate = _uncertainty_gate(
        uncertainty,
        model_score,
        eligible_top_k=3,
        budget=2,
    )
    # Candidate 0 is most uncertain but is outside the model-score top three.
    assert gate.tolist() == [[False, True, True, False]]


def test_roc_auc_and_top_quartile_summary_detect_separation() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    assert _roc_auc(scores, labels) == 1.0
    summary = _binary_signal_summary(scores, labels)
    assert summary["roc_auc"] == 1.0
    assert summary["top_quartile_precision"] == 1.0
    assert summary["top_quartile_lift"] == 2.0


def test_candidate_uncertainty_signals_have_lane_shape() -> None:
    torch.manual_seed(17)
    batch, instances, rows, output_bins, evidence_bins = 1, 3, 2, 4, 2
    outputs = {
        "row_x_logits": torch.randn(batch, instances, rows, output_bins),
        "pred_x_rows": torch.rand(batch, instances, rows) * 80.0,
        "range_norm": torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]
        ),
        "exist_logits": torch.randn(batch, instances, 2),
        "quality_logits": torch.randn(batch, instances),
    }
    attention = torch.softmax(
        torch.randn(batch, rows, 1, instances, evidence_bins),
        dim=-1,
    )
    signals, model_score = _candidate_uncertainty_signals(
        outputs=outputs,
        attention=attention,
        input_w=80.0,
        quality_power=0.5,
    )
    assert model_score.shape == (batch, instances)
    assert "attention_disagreement" in signals
    assert "quality_disagreement" in signals
    for values in signals.values():
        assert values.shape == (batch, instances)
        assert bool(torch.isfinite(values).all())
