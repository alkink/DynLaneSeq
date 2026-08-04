from __future__ import annotations

from dynlaneseq_eg.config import load_config


CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_v4_4_set_oracle_pointer.yaml"
)


def test_v4_4_config_changes_only_the_pointer_acquisition_contract() -> None:
    cfg = load_config(CONFIG)
    selection = cfg["model"]["structured_query"]["set_selection"]
    loss = cfg["loss"]
    assert selection["candidate_interaction"] == "sequential_pointer"
    assert selection["pointer_teacher_mode"] == "permutation_invariant_set"
    assert selection["detach_geometry_features"] is True
    assert loss["pointer_unary_target_mode"] == "unique_representative"
    assert loss["set_selection_focal_beta"] == 0.0
    assert loss["w_pointer_selection"] == 1.0
    assert loss["w_set_selection"] == 0.0
    assert cfg["postprocess"]["score_mode"] == "pointer"
