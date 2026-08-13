from __future__ import annotations

from dynlaneseq_eg.tools.summarize_geometry_proposal_clustering import summarize


def _report(*, precision: float, recall: float, catastrophic: float, multi: float, retention: float):
    policy_names = (
        "flat_complete_48",
        "perspective_balanced_48",
        "perspective_conservative_36",
    )
    structural = {
        name: {
            "pair_merge_precision": precision if name == "perspective_balanced_48" else 0.90,
            "same_gt_pair_recall": recall if name == "perspective_balanced_48" else 0.20,
            "catastrophic_cluster_fraction": catastrophic if name == "perspective_balanced_48" else 0.10,
            "candidate_multi_member_fraction": multi if name == "perspective_balanced_48" else 0.20,
        }
        for name in policy_names
    }
    metrics = {}
    for policy in policy_names:
        for prototype in ("medoid", "median", "mean"):
            for threshold in ("0.50", "0.75"):
                value = retention if policy == "perspective_balanced_48" else 0.80
                metrics[f"{policy}/{prototype}/oracle_same_count/{threshold}"] = {
                    "all32_tp_retention": value,
                }
    return {
        "scope": {"test_set_used": False},
        "metadata": {"cache_path": "cache.pt"},
        "structural_clustering": structural,
        "official_metrics": metrics,
        "bottom_guard_ablation": {
            "different_gt_precision_among_labeled_removed": 0.95,
        },
    }


def test_summary_passes_only_when_locked_policy_transfers() -> None:
    calibration = _report(
        precision=0.99,
        recall=0.60,
        catastrophic=0.01,
        multi=0.65,
        retention=0.98,
    )
    validation = _report(
        precision=0.985,
        recall=0.55,
        catastrophic=0.015,
        multi=0.60,
        retention=0.96,
    )
    result = summarize(calibration, validation)
    assert result["passed"] is True
    assert result["selection_contract"]["selected_policy"] == "perspective_balanced_48"
    assert result["new_model_version_authorized"] is False


def test_summary_fails_when_validation_pair_precision_does_not_transfer() -> None:
    calibration = _report(
        precision=0.99,
        recall=0.60,
        catastrophic=0.01,
        multi=0.65,
        retention=0.98,
    )
    validation = _report(
        precision=0.90,
        recall=0.55,
        catastrophic=0.015,
        multi=0.60,
        retention=0.98,
    )
    result = summarize(calibration, validation)
    assert result["passed"] is False
    assert result["long_training_authorized"] is False
