import torch

from dynlaneseq_eg.modeling.v25_s1_immutable_selector import (
    ACTIONS,
    GLOBAL_EVIDENCE_DIM,
    GLOBAL_GEOMETRY_DIM,
    OUTCOME_BENEFICIAL,
    OUTCOME_NEUTRAL,
    ROW_EVIDENCE_DIM,
    ROW_GEOMETRY_DIM,
    ImmutableBankSingleEditSelector,
    build_selector_features,
    immutable_selector_loss,
)


def _inputs(batch: int = 2):
    rows = 160
    bins = 80
    source = torch.linspace(180.0, 1400.0, 4).view(1, 4, 1).expand(batch, 4, rows)
    offsets = torch.tensor((-16.0, 8.0, 24.0)).view(1, 1, 3, 1)
    hypotheses = source.unsqueeze(2) + offsets
    lane_range = torch.tensor((0.20, 0.95)).view(1, 1, 2).expand(batch, 4, 2)
    unary = torch.randn(batch, 4, rows, bins)
    evidence = {
        "unary_logits": unary,
        "exist_logits": torch.randn(batch, 4, 2),
        "quality50_logits": torch.randn(batch, 4),
        "quality75_logits": torch.randn(batch, 4),
    }
    return source, hypotheses, lane_range, evidence


def test_selector_feature_contract_and_wrong_evidence_is_geometry_immutable() -> None:
    source, hypotheses, lane_range, evidence = _inputs()
    active = torch.tensor(((True, True, True, False), (True, True, True, True)))
    correct = build_selector_features(
        source_x=source,
        source_range=lane_range,
        source_active=active,
        hypotheses=hypotheses,
        hypothesis_range=lane_range,
        evidence=evidence,
        input_w=1600,
    )
    wrong_evidence = dict(evidence)
    wrong_evidence["unary_logits"] = torch.randn_like(evidence["unary_logits"])
    wrong = build_selector_features(
        source_x=source,
        source_range=lane_range,
        source_active=active,
        hypotheses=hypotheses,
        hypothesis_range=lane_range,
        evidence=wrong_evidence,
        input_w=1600,
    )
    assert correct["row_evidence"].shape == (2, ACTIONS, 160, ROW_EVIDENCE_DIM)
    assert correct["row_geometry"].shape == (2, ACTIONS, 160, ROW_GEOMETRY_DIM)
    assert correct["global_evidence"].shape == (2, ACTIONS, GLOBAL_EVIDENCE_DIM)
    assert correct["global_geometry"].shape == (2, ACTIONS, GLOBAL_GEOMETRY_DIM)
    assert torch.equal(correct["row_geometry"], wrong["row_geometry"])
    assert torch.equal(correct["global_geometry"], wrong["global_geometry"])
    assert not torch.equal(correct["row_evidence"], wrong["row_evidence"])
    assert not correct["action_valid"][0, 9:].any()


def test_selector_is_hard_keep_or_single_edit_and_gets_gradient() -> None:
    source, hypotheses, lane_range, evidence = _inputs()
    active = torch.ones(2, 4, dtype=torch.bool)
    features = build_selector_features(
        source_x=source,
        source_range=lane_range,
        source_active=active,
        hypotheses=hypotheses,
        hypothesis_range=lane_range,
        evidence=evidence,
        input_w=1600,
    )
    model = ImmutableBankSingleEditSelector(hidden_dim=32, dropout=0.0)
    outputs = model(features)
    assert outputs["action_scores"].shape == (2, 13)
    assert torch.equal(outputs["action_scores"][:, 0], torch.zeros(2))
    assert bool(((outputs["selected_action"] >= 0) & (outputs["selected_action"] <= 12)).all())

    target_action = torch.tensor((0, 4))
    outcomes = torch.full((2, 12), OUTCOME_NEUTRAL, dtype=torch.long)
    outcomes[1, 3] = OUTCOME_BENEFICIAL
    loss, diagnostics = immutable_selector_loss(
        outputs,
        target_action=target_action,
        action_outcome=outcomes,
        action_valid=features["action_valid"],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["target_edit_fraction"].item() == 0.5
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_geometry_control_has_same_parameters_but_ignores_evidence_values() -> None:
    source, hypotheses, lane_range, evidence = _inputs(batch=1)
    active = torch.ones(1, 4, dtype=torch.bool)
    first = build_selector_features(
        source_x=source,
        source_range=lane_range,
        source_active=active,
        hypotheses=hypotheses,
        hypothesis_range=lane_range,
        evidence=evidence,
        input_w=1600,
    )
    changed = dict(first)
    changed["row_evidence"] = torch.randn_like(first["row_evidence"])
    changed["global_evidence"] = torch.randn_like(first["global_evidence"])
    model = ImmutableBankSingleEditSelector(hidden_dim=32, dropout=0.0).eval()
    left = model(first, evidence_enabled=False)
    right = model(changed, evidence_enabled=False)
    assert torch.equal(left["action_scores"], right["action_scores"])
