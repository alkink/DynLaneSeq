from __future__ import annotations

import pytest

from dynlaneseq_eg.tools.audit_hierarchical_probe_generalization import (
    diagnose_generalization,
    evaluation_gains,
    verify_validation_replay,
)


def _mode(tp50: int, tp70: int, *, gt: int = 100, selected: int = 100) -> dict:
    def row(tp: int) -> tuple[float, float, float]:
        precision = tp / selected
        recall = tp / gt
        f1 = 2.0 * precision * recall / (precision + recall)
        return precision, recall, f1

    p50, r50, f50 = row(tp50)
    p70, r70, f70 = row(tp70)
    return {
        "gt_lanes": gt,
        "selected_predictions": selected,
        "tp_050": tp50,
        "precision_050": p50,
        "recall_050": r50,
        "f1_050": f50,
        "tp_070": tp70,
        "precision_070": p70,
        "recall_070": r70,
        "f1_070": f70,
    }


def _evaluation(cluster: tuple[int, int] = (76, 61)) -> dict:
    return {
        "modes": {
            "source_nms": _mode(70, 55),
            "learned_cluster_source_representative": _mode(*cluster),
            "source_cluster_learned_representative": _mode(72, 60),
            "learned_hierarchical": _mode(77, 62),
        },
        "oracle_top4": {
            "0.50": {"tp": 95, "gt_lanes": 100, "recall": 0.95},
            "0.70": {"tp": 85, "gt_lanes": 100, "recall": 0.85},
        },
    }


def test_evaluation_gains_reports_oracle_gap_fraction() -> None:
    gains = evaluation_gains(_evaluation())
    cluster = gains["arms"]["learned_cluster_source_representative"]
    assert cluster["gain_recall_050_points"] == pytest.approx(6.0)
    assert cluster["gain_recall_070_points"] == pytest.approx(6.0)
    assert cluster["oracle_gap_recovered_050_fraction"] == pytest.approx(0.24)
    assert cluster["oracle_gap_recovered_070_fraction"] == pytest.approx(0.20)


def test_validation_replay_rejects_a_different_saved_result() -> None:
    recorded = _evaluation()
    replayed = _evaluation(cluster=(75, 61))
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_validation_replay(replayed, recorded)


def test_generalization_diagnosis_separates_train_only_fit() -> None:
    train_gate = {
        "cluster": {"positive": True},
        "representative": {"positive": True},
        "hierarchical": {"positive": True},
        "dual_head_positive": True,
    }
    val_gate = {
        "cluster": {"positive": False},
        "representative": {"positive": False},
        "hierarchical": {"positive": False},
        "dual_head_positive": False,
    }
    decision = diagnose_generalization(train_gate, val_gate)
    assert (
        decision["diagnosis"]
        == "hierarchical_contract_fits_train_but_does_not_generalize"
    )


def test_generalization_diagnosis_detects_no_in_sample_fit() -> None:
    gate = {
        "cluster": {"positive": False},
        "representative": {"positive": False},
        "hierarchical": {"positive": False},
        "dual_head_positive": False,
    }
    decision = diagnose_generalization(gate, gate)
    assert (
        decision["diagnosis"]
        == "saved_best_probe_does_not_fit_the_contract_even_in_sample"
    )
