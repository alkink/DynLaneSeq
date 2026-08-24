from __future__ import annotations

import pytest

from dynlaneseq_eg.tools.audit_v38_direct_primary_autopsy import (
    _binary_auc,
    _policy_metrics,
    classify_threshold_bottleneck,
)


def test_binary_auc_handles_ties_exactly() -> None:
    assert _binary_auc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert _binary_auc([False, True], [0.5, 0.5]) == 0.5
    assert _binary_auc([True, True], [0.1, 0.9]) is None


def test_policy_metrics_use_official_count_contract() -> None:
    metrics = _policy_metrics(tp=8, fp=2, fn=2)
    assert metrics["Precision"] == pytest.approx(0.8)
    assert metrics["Recall"] == pytest.approx(0.8)
    assert metrics["F1"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("deployed", "support", "v7", "decision"),
    [
        (80, 105, 100, "EXISTENCE_OR_SET_CONVERSION_LIMITED"),
        (80, 85, 100, "DIRECT_GEOMETRY_CAPACITY_LIMITED"),
        (80, 92, 100, "MIXED_GEOMETRY_AND_CONVERSION_LIMITED"),
    ],
)
def test_bottleneck_classifier(
    deployed: int, support: int, v7: int, decision: str
) -> None:
    result = classify_threshold_bottleneck(
        deployed_tp=deployed,
        support_tp=support,
        v7_tp=v7,
    )
    assert result["decision"] == decision

