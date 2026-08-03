from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.structured_queries import SetAwareLaneSelectionHead


CONFIG_PREFIX = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_v4_2_score_"
)


def _selector() -> SetAwareLaneSelectionHead:
    return SetAwareLaneSelectionHead(
        8,
        input_w=64,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        curve_samples=4,
        unified_score=True,
        detach_geometry_features=True,
        candidate_interaction="relation_transformer",
        relation_sigma_px=20.0,
        relation_hidden_dim=8,
    )


def test_pairwise_curve_relations_are_symmetric_and_distance_aware() -> None:
    selector = _selector()
    pred_x = torch.tensor(
        [[[10.0] * 8, [12.0] * 8, [45.0] * 8]],
        requires_grad=True,
    )
    outputs = {
        "pred_x_rows": pred_x,
        "range_norm": torch.tensor([[[0.0, 1.0]] * 3]),
    }
    relation = selector.build_pairwise_relations(outputs)
    assert relation.shape == (1, 3, 3, 6)
    assert relation.requires_grad is False
    torch.testing.assert_close(relation, relation.transpose(1, 2))
    assert float(relation[0, 0, 1, 0]) < float(relation[0, 0, 2, 0])
    assert float(relation[0, 0, 1, 5]) > float(relation[0, 0, 2, 5])


def test_relation_transformer_is_candidate_permutation_equivariant() -> None:
    torch.manual_seed(607)
    selector = _selector().eval()
    feature_dim = int(selector.input_norm.normalized_shape[0])
    features = torch.randn(2, 4, feature_dim)
    relations = torch.randn(2, 4, 4, selector.relation_dim)
    permutation = torch.tensor([2, 0, 3, 1])
    original = selector.score_selection_features(features, relations)
    permuted = selector.score_selection_features(
        features[:, permutation],
        relations[:, permutation][:, :, permutation],
    )
    torch.testing.assert_close(
        permuted,
        original[:, permutation],
        atol=1e-5,
        rtol=1e-5,
    )


def _set_loss(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    criterion = S0Criterion(
        LossConfig(
            input_w=64,
            input_h=32,
            w_set_selection=1.0,
            set_selection_line_width=12.0,
            set_selection_min_valid_rows=2,
            set_selection_share_matcher_assignment=False,
            set_selection_negative_weight=2.0,
            set_selection_rank_weight=0.25,
            set_selection_coverage_weight=0.5,
            set_selection_duplicate_weight=1.0,
            set_selection_winner_weight=0.25,
            set_selection_count_weight=0.1,
            set_selection_duplicate_quality_min=0.3,
            set_selection_winner_quality_min=0.3,
        )
    )
    outputs = {
        "selection_logits": logits,
        "pred_x_rows": torch.tensor(
            [[[10.0] * 6, [12.0] * 6, [48.0] * 6]]
        ),
        "range_norm": torch.tensor([[[0.0, 1.0]] * 3]),
    }
    targets = [
        {
            "x_rows": torch.tensor([[10.0] * 6]),
            "valid_mask": torch.ones(1, 6, dtype=torch.bool),
        }
    ]
    return criterion.compute_set_selection_loss(outputs, targets)


def test_relation_set_losses_are_finite_and_suppress_duplicate_coactivation() -> None:
    high_duplicate = torch.tensor([[2.0, 2.0, -2.0]], requires_grad=True)
    low_duplicate = torch.tensor([[2.0, -2.0, -2.0]], requires_grad=True)
    high = _set_loss(high_duplicate)
    low = _set_loss(low_duplicate)
    for key in ("total", "coverage", "duplicate", "winner", "count"):
        assert torch.isfinite(high[key])
    assert float(high["duplicate"]) > float(low["duplicate"])
    high["total"].backward()
    assert high_duplicate.grad is not None
    assert bool(torch.isfinite(high_duplicate.grad).all())


def test_v4_2_configs_encode_the_causal_r0_to_r3_matrix() -> None:
    configs = [
        load_config(CONFIG_PREFIX + "r0_generic_frozen.yaml"),
        load_config(CONFIG_PREFIX + "r1_generic_semantic.yaml"),
        load_config(CONFIG_PREFIX + "r2_relation_semantic.yaml"),
        load_config(CONFIG_PREFIX + "r3_relation_setloss.yaml"),
    ]
    assert [
        cfg["model"]["structured_query"]["set_selection"][
            "candidate_interaction"
        ]
        for cfg in configs
    ] == [
        "transformer",
        "transformer",
        "relation_transformer",
        "relation_transformer",
    ]
    for index, cfg in enumerate(configs):
        training = cfg["training"]
        assert training["checkpoint_interval"] == 500
        assert training["checkpoint_include_optimizer"] is True
        assert cfg["model"]["structured_query"]["set_selection"][
            "detach_geometry_features"
        ] is True
        prefixes = training["trainable_parameter_prefixes"]
        if index == 0:
            assert prefixes == ["structured_query_head.set_selection_head"]
        else:
            assert any("semantic_attention" in item for item in prefixes)
    assert configs[2]["loss"]["set_selection_duplicate_weight"] == 0.0
    assert configs[3]["loss"]["set_selection_duplicate_weight"] == 1.0

