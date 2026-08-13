from __future__ import annotations

from dynlaneseq_eg.tools.summarize_geometry_proposal_clustering import summarize


def _metric(tp: int, predictions: int = 8, gt: int = 8) -> dict:
    return {
        "tp": tp,
        "fp": predictions - tp,
        "fn": gt - tp,
        "predictions": predictions,
        "gt": gt,
        "f1": 2.0 * tp / float(predictions + gt),
        "exact_count_parity": True,
        "count_mismatch_images": 0,
    }


def _report(
    *,
    precision: float,
    recall: float,
    catastrophic: float,
    multi: float,
    retention: float,
    treatment_tp_050: int,
    treatment_tp_075: int,
) -> dict:
    policies = (
        "flat_complete_48",
        "perspective_balanced_48",
        "perspective_conservative_36",
    )
    structural = {
        name: {
            "pair_merge_precision": precision
            if name == "perspective_balanced_48"
            else 0.90,
            "same_gt_pair_recall": recall
            if name == "perspective_balanced_48"
            else 0.20,
            "catastrophic_cluster_fraction": catastrophic
            if name == "perspective_balanced_48"
            else 0.10,
            "candidate_multi_member_fraction": multi
            if name == "perspective_balanced_48"
            else 0.20,
        }
        for name in policies
    }
    metrics = {
        "current_v7_refined/0.50": _metric(4),
        "current_v7_refined/0.75": _metric(4),
        "perspective_balanced_48/medoid/slot_cluster_mass_unique/0.50": _metric(
            treatment_tp_050
        ),
        "perspective_balanced_48/medoid/slot_cluster_mass_unique/0.75": _metric(
            treatment_tp_075
        ),
        "perspective_balanced_48/medoid/oracle_same_count/0.50": {
            **_metric(8),
            "all32_tp_retention": retention,
        },
        "perspective_balanced_48/medoid/oracle_same_count/0.75": {
            **_metric(8),
            "all32_tp_retention": retention,
        },
    }
    def distribute(total: int) -> list[int]:
        values: list[int] = []
        remaining = int(total)
        for _ in range(4):
            value = min(2, remaining)
            values.append(value)
            remaining -= value
        assert remaining == 0
        return values

    treatment_hits_050 = distribute(treatment_tp_050)
    treatment_hits_075 = distribute(treatment_tp_075)
    per_image = []
    for index in range(4):
        per_image.append(
            {
                "gt_count": 2,
                "current_writer_count": 2,
                "current_v7_refined": {"hits_0.50": 1, "hits_0.75": 1},
                "policies": {
                    "perspective_balanced_48": {
                        "prototypes": {
                            "medoid": {
                                "slot_cluster_mass_unique_hits_0.50": treatment_hits_050[
                                    index
                                ],
                                "slot_cluster_mass_unique_predictions_0.50": 2,
                                "slot_cluster_mass_unique_hits_0.75": treatment_hits_075[
                                    index
                                ],
                                "slot_cluster_mass_unique_predictions_0.75": 2,
                            }
                        }
                    }
                },
            }
        )
    return {
        "scope": {"test_set_used": False, "optimizer_steps": 0},
        "primary_confirmatory_contract": {
            "policy": "perspective_balanced_48",
            "prototype": "medoid",
            "selection": "slot_cluster_mass_unique",
        },
        "metadata": {"cache_path": "cache.pt"},
        "structural_clustering": structural,
        "official_metrics": metrics,
        "bottom_guard_ablation": {
            "different_gt_precision_among_labeled_removed": 0.95,
        },
        "per_image": per_image,
    }


def test_summary_passes_only_when_fixed_primary_improves_and_transfers() -> None:
    calibration = _report(
        precision=0.99,
        recall=0.60,
        catastrophic=0.01,
        multi=0.65,
        retention=0.98,
        treatment_tp_050=4,
        treatment_tp_075=4,
    )
    validation = _report(
        precision=0.985,
        recall=0.55,
        catastrophic=0.015,
        multi=0.60,
        retention=0.96,
        treatment_tp_050=7,
        treatment_tp_075=4,
    )
    result = summarize(calibration, validation)
    assert result["passed"] is True
    assert result["predeclared_primary"]["policy"] == "perspective_balanced_48"
    assert result["validation"]["primary_deployable"]["0.50"]["delta_tp"] == 3
    assert result["new_model_version_authorized"] is False


def test_summary_fails_when_validation_pair_precision_does_not_transfer() -> None:
    calibration = _report(
        precision=0.99,
        recall=0.60,
        catastrophic=0.01,
        multi=0.65,
        retention=0.98,
        treatment_tp_050=4,
        treatment_tp_075=4,
    )
    validation = _report(
        precision=0.90,
        recall=0.55,
        catastrophic=0.015,
        multi=0.60,
        retention=0.98,
        treatment_tp_050=7,
        treatment_tp_075=4,
    )
    result = summarize(calibration, validation)
    assert result["passed"] is False
    assert result["long_training_authorized"] is False


def test_summary_does_not_posthoc_select_a_better_secondary_policy() -> None:
    calibration = _report(
        precision=0.99,
        recall=0.60,
        catastrophic=0.01,
        multi=0.65,
        retention=0.98,
        treatment_tp_050=4,
        treatment_tp_075=4,
    )
    validation = _report(
        precision=0.99,
        recall=0.60,
        catastrophic=0.01,
        multi=0.65,
        retention=0.98,
        treatment_tp_050=6,
        treatment_tp_075=4,
    )
    result = summarize(calibration, validation)
    assert result["passed"] is False
    assert result["selection_contract"]["secondary_policy_selection_performed"] is False
