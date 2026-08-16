from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from dynlaneseq_eg.modeling.v23_ordered_slot_cost_volume import (
    V23OrderedSlotCostVolume,
    _shifted_transition,
    build_v23_owned_targets,
    canonicalize_v7_slots,
    soft_viterbi_marginals,
    v23_ordered_cost_volume_loss,
)
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


def _teacher(batch: int = 1, rows: int = 16, proposals: int = 8):
    curves = torch.stack(
        (
            torch.linspace(90.0, 100.0, rows),
            torch.linspace(10.0, 20.0, rows),
            torch.linspace(65.0, 75.0, rows),
            torch.linspace(35.0, 45.0, rows),
        )
    ).unsqueeze(0).expand(batch, -1, -1).clone()
    proposal_curves = torch.stack(
        [torch.linspace(5.0 + 3.0 * index, 15.0 + 3.0 * index, rows) for index in range(proposals)]
    ).unsqueeze(0).expand(batch, -1, -1).clone()
    return {
        "selection_slot_pred_x_rows": curves,
        "selection_slot_range_norm": torch.tensor(
            [[[0.0, 0.9]] * 4], dtype=torch.float32
        ).expand(batch, -1, -1).clone(),
        "selection_slot_active": torch.tensor(
            [[True, True, True, False]]
        ).expand(batch, -1).clone(),
        "pred_x_rows": proposal_curves,
        "range_norm": torch.tensor(
            [[[0.0, 0.9]] * proposals], dtype=torch.float32
        ).expand(batch, -1, -1).clone(),
    }


def _model(rows: int = 16, bins: int = 32) -> V23OrderedSlotCostVolume:
    return V23OrderedSlotCostVolume(
        input_h=64,
        input_w=128,
        num_rows=rows,
        x_bins=bins,
        fpn_channels=32,
        hidden_dim=32,
        query_dim=32,
        vertical_layers=1,
        num_heads=4,
        ff_dim=64,
        transition_radius_bins=2,
    )


def test_teacher_slots_are_canonical_left_to_right() -> None:
    result = canonicalize_v7_slots(_teacher())
    assert result["source_slot_indices"].tolist() == [[1, 2, 0, 3]]
    assert result["active"].tolist() == [[True, True, True, False]]
    assert result["x_rows"][0, :3, -2].tolist() == sorted(
        result["x_rows"][0, :3, -2].tolist()
    )


def test_owned_assignment_preserves_order_when_counts_differ() -> None:
    source = canonicalize_v7_slots(_teacher())
    rows = int(source["x_rows"].shape[-1])
    target_x = torch.stack(
        (
            torch.linspace(11.0, 21.0, rows),
            torch.linspace(36.0, 46.0, rows),
            torch.linspace(66.0, 76.0, rows),
            torch.linspace(96.0, 106.0, rows),
        )
    )
    owned = build_v23_owned_targets(
        [{"x_rows": target_x, "valid_mask": torch.ones_like(target_x, dtype=torch.bool)}],
        source_x=source["x_rows"],
        source_range=source["range_norm"],
        source_active=source["active"],
        input_h=64,
        input_w=128,
    )
    assert owned["matched"].tolist() == [[True, True, True, False]]
    # V7 has no lane around x=40. The monotonic assignment skips precisely
    # that unmatched GT instead of shifting all identities after it.
    assert owned["target_indices"][0, :3].tolist() == [0, 2, 3]


def test_zero_step_is_exact_v7_but_student_receives_gradients() -> None:
    torch.manual_seed(7)
    model = _model().train()
    teacher = _teacher()
    output = model(torch.randn(1, 3, 64, 128), teacher)
    canonical = canonicalize_v7_slots(teacher)
    assert torch.equal(output["pred_x_rows"], teacher["selection_slot_pred_x_rows"])
    assert torch.equal(output["range_norm"], teacher["selection_slot_range_norm"])
    assert torch.equal(output["source_active"], canonical["active"])
    target_x = canonical["x_rows"][0, :3] + 1.0
    total, diagnostics = v23_ordered_cost_volume_loss(
        output,
        [{"x_rows": target_x, "valid_mask": torch.ones_like(target_x, dtype=torch.bool)}],
        input_h=64,
        input_w=128,
    )
    total.backward()
    assert torch.isfinite(total)
    assert diagnostics["matched_slots"].item() == 3
    groups = (model.backbone, model.fpn, model.fine_stem, model.vertical_encoder)
    for group in groups:
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in group.parameters()
        )
    assert model.geometry_gate.grad is not None
    assert model.geometry_gate.grad.abs().sum() > 0


def test_soft_viterbi_prefers_coherent_path() -> None:
    unary = torch.full((1, 1, 6, 12), -4.0)
    unary[0, 0, :, 5] = 4.0
    unary[0, 0, 3, 10] = 6.0
    marginal = soft_viterbi_marginals(
        unary, transition_radius_bins=2, transition_penalty=1.0
    )
    assert marginal[0, 0, 3].argmax().item() == 5


def test_vectorized_transition_matches_offset_reference() -> None:
    torch.manual_seed(11)
    value = torch.randn(2, 4, 31)
    radius = 4
    transition_penalty = 0.17
    candidates = []
    bins = int(value.shape[-1])
    for offset in range(-radius, radius + 1):
        if offset < 0:
            shifted = F.pad(value[..., -offset:], (0, -offset), value=-1.0e4)
        elif offset > 0:
            shifted = F.pad(value[..., : bins - offset], (offset, 0), value=-1.0e4)
        else:
            shifted = value
        candidates.append(shifted - transition_penalty * abs(offset))
    reference = torch.logsumexp(torch.stack(candidates, dim=-2), dim=-2)
    vectorized = _shifted_transition(
        value,
        radius=radius,
        transition_penalty=transition_penalty,
    )
    torch.testing.assert_close(vectorized, reference, rtol=1.0e-6, atol=1.0e-6)


def test_official_v23_protocol_requires_train_txt(tmp_path: Path) -> None:
    root = tmp_path / "CULane"
    (root / "list").mkdir(parents=True)
    # The production count is deliberately tested through a small monkeypatch
    # so this unit test does not generate an 88,880-line fixture.
    import dynlaneseq_eg.tools.v23_official_protocol as protocol

    original = protocol.OFFICIAL_V23_CULANE_LISTS
    protocol.OFFICIAL_V23_CULANE_LISTS = {"train": ("list/train.txt", 2)}
    try:
        (root / "list/train.txt").write_text("/a.jpg\n/a.jpg\n", encoding="utf-8")
        report = official_v23_culane_list_contract(root, split="train")
        assert report["list_relative_path"] == "list/train.txt"
        assert report["repeated_row_count_preserved"] == 1
        assert report["rows_removed"] == 0
        with pytest.raises(ValueError):
            official_v23_culane_list_contract(
                root, split="train", supplied_path=root / "list/train_gt.txt"
            )
    finally:
        protocol.OFFICIAL_V23_CULANE_LISTS = original
