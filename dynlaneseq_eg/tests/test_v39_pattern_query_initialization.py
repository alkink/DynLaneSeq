from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import (
    ImageConditionedPatternQueryInitializer,
    V25ImageMediatedLaneObjects,
)
from dynlaneseq_eg.tools.train_v39_pattern_query_initialization import (
    FIXED_EFFECTIVE_BATCH,
    FIXED_ENDPOINT,
    FIXED_SCHEDULE,
    SOURCE_ITERATION,
    validate_v39_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "dynlaneseq_eg/configs/culane_v39_pqi_control_65k.yaml"
TREATMENT = ROOT / "dynlaneseq_eg/configs/culane_v39_pqi_treatment_65k.yaml"


@pytest.mark.parametrize(
    ("path", "arm", "enabled"),
    ((CONTROL, "control", False), (TREATMENT, "treatment", True)),
)
def test_v39_exact_pair_config_contract(
    path: Path, arm: str, enabled: bool
) -> None:
    cfg = load_config(path)
    contract = validate_v39_contract(cfg)
    assert contract["passed"], contract
    assert contract["arm"] == arm
    assert contract["effective_batch"] == FIXED_EFFECTIVE_BATCH == 16
    assert cfg["v25"]["pattern_query_initializer"]["enabled"] is enabled
    assert cfg["v39"]["source_iteration"] == SOURCE_ITERATION == 50_000
    assert cfg["v39"]["endpoint_iteration"] == FIXED_ENDPOINT == 65_000
    assert cfg["v39"]["scheduler_total_iters"] == FIXED_SCHEDULE == 278_000


def test_pattern_initializer_is_exactly_canonical_at_step_zero() -> None:
    module = ImageConditionedPatternQueryInitializer(
        channels=8,
        num_slots=4,
        num_rows=16,
        pattern_count=4,
        pooled_rows=2,
        pooled_columns=3,
        projection_channels=4,
        hidden_dim=12,
    )
    features = torch.randn(2, 8, 8, 12)
    canonical = torch.linspace(0.16, 0.84, 4)
    output = module(features, canonical)
    expected = canonical.view(1, 4, 1).expand(2, 4, 16)
    assert torch.equal(output["query_anchor_x_rows"], expected)
    assert torch.count_nonzero(output["pattern_gate"]) == 0


def test_pattern_initializer_gets_geometry_gradient_without_moving_source() -> None:
    module = ImageConditionedPatternQueryInitializer(
        channels=8,
        num_slots=4,
        num_rows=16,
        pattern_count=4,
        pooled_rows=2,
        pooled_columns=3,
        projection_channels=4,
        hidden_dim=12,
    )
    features = torch.randn(2, 8, 8, 12, requires_grad=True)
    canonical = torch.linspace(0.16, 0.84, 4)
    anchor = module(features, canonical)["query_anchor_x_rows"]
    row_weight = torch.linspace(-1.0, 1.0, 16).view(1, 1, 16)
    (anchor * row_weight).sum().backward()
    gradient = module.pattern_and_gate.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_pattern_bank_contract_rejects_wrong_shape_and_range() -> None:
    module = ImageConditionedPatternQueryInitializer(
        channels=4,
        num_slots=4,
        num_rows=8,
        pattern_count=3,
        pooled_rows=2,
        pooled_columns=2,
        projection_channels=2,
        hidden_dim=4,
    )
    with pytest.raises(ValueError, match="shape"):
        module.set_pattern_bank(torch.zeros(4, 2, 8))
    invalid = torch.zeros(4, 3, 8)
    invalid[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        module.set_pattern_bank(invalid)


def test_v39_zero_initializer_preserves_full_v38_forward_and_then_can_move() -> None:
    torch.manual_seed(17)
    model = V25ImageMediatedLaneObjects(
        input_h=64,
        input_w=128,
        num_rows=8,
        x_bins=32,
        fpn_channels=32,
        hidden_dim=32,
        decoder_layers=1,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        transition_radius_bins=2,
        pretrained_backbone=False,
        require_pretrained_backbone=False,
        enable_pattern_query_initializer=True,
        pattern_query_count=4,
        pattern_query_pooled_rows=2,
        pattern_query_pooled_columns=4,
        pattern_query_projection_channels=4,
        pattern_query_hidden_dim=16,
    ).eval()
    images = torch.randn(1, 3, 64, 128)
    canonical = model.canonical_slot_centres.view(1, 4, 1).expand(1, 4, 8)
    treatment = model(images)
    source = model(images, query_anchor_x_rows=canonical)
    for name in (
        "exist_logits",
        "pred_x_rows",
        "range_norm",
        "unary_logits",
        "path_logits",
    ):
        assert torch.equal(treatment[name], source[name]), name
    initializer = model.pattern_query_initializer
    assert initializer is not None
    with torch.no_grad():
        initializer.pattern_and_gate.bias[4] = 0.5
    moved = model(images)
    assert not torch.equal(moved["query_anchor_x_rows"], canonical)
    assert not torch.equal(moved["unary_logits"], source["unary_logits"])
