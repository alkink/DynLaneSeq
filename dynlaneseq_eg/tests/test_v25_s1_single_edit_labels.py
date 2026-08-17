import numpy as np

from dynlaneseq_eg.tools.v25_s1_single_edit_labels import (
    OUTCOME_BENEFICIAL,
    OUTCOME_HARMFUL,
    score_single_edit_iou_bank,
)


def _bank() -> np.ndarray:
    bank = np.zeros((4, 4, 4), dtype=np.float32)
    for slot in range(4):
        bank[slot, 0, slot] = 0.9
        bank[slot, 1:, slot] = 0.9
    return bank


def test_source_on_tie_keeps_v7() -> None:
    result = score_single_edit_iou_bank(
        _bank(),
        source_active=np.ones(4, dtype=bool),
        candidate_valid=np.ones((4, 3), dtype=bool),
    )
    assert result["target_action"] == 0
    assert result["raw_oracle_action"] == 0


def test_safe_target_selects_threshold_gain() -> None:
    bank = _bank()
    bank[0, 0] = 0.0
    bank[0, 1:] = 0.0
    bank[0, 1, 0] = 0.8
    result = score_single_edit_iou_bank(
        bank,
        source_active=np.ones(4, dtype=bool),
        candidate_valid=np.ones((4, 3), dtype=bool),
    )
    assert result["target_action"] == 1
    assert result["action_outcome"][0] == OUTCOME_BENEFICIAL


def test_source_correct_loss_is_harmful_and_never_safe_target() -> None:
    bank = _bank()
    bank[0, 1] = 0.0
    bank[0, 1, 1] = 0.95
    result = score_single_edit_iou_bank(
        bank,
        source_active=np.ones(4, dtype=bool),
        candidate_valid=np.ones((4, 3), dtype=bool),
    )
    assert result["action_outcome"][0] == OUTCOME_HARMFUL
    assert result["target_action"] == 0
