from __future__ import annotations

from pathlib import Path

import numpy as np

from dynlaneseq_eg.tools.audit_v34_temporal_candidate_observability import (
    WarpResult,
    bilinear_sample_flow,
    build_temporal_manifest,
    candidate_support_score,
    curve_soft_iou,
    summarize_pairs,
    warp_curve,
)


def test_manifest_is_clip_disjoint_and_uses_real_neighbors(tmp_path: Path) -> None:
    rows = []
    for clip in ("clip_a.MP4", "clip_b.MP4", "clip_c.MP4", "clip_d.MP4"):
        for frame in (0, 30, 60, 90, 120):
            rows.append(f"/driver/{clip}/{frame:05d}.jpg")
    val = tmp_path / "val.txt"
    val.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = build_temporal_manifest(
        val, sample_per_fold=4, frame_step=30, seed=3407
    )
    samples = manifest["samples"]
    clips = {
        fold: {row["clip"] for row in samples if row["fold"] == fold}
        for fold in ("a", "b")
    }
    assert clips["a"].isdisjoint(clips["b"])
    assert manifest["selected_by_fold"] == {"a": 4, "b": 4}
    for row in samples:
        target = int(Path(row["target"]).stem)
        assert int(Path(row["previous"]).stem) == target - 30
        assert int(Path(row["following"]).stem) == target + 30
        assert Path(row["wrong_context"]).parent != Path(row["target"]).parent


def test_bilinear_flow_and_forward_backward_warp_are_exact() -> None:
    flow = np.zeros((8, 12, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    flow[..., 1] = -1.0
    sampled, valid = bilinear_sample_flow(
        flow,
        np.asarray([2.5, 8.25], dtype=np.float32),
        np.asarray([3.0, 5.5], dtype=np.float32),
    )
    np.testing.assert_allclose(sampled, [[2.0, -1.0], [2.0, -1.0]], atol=1e-6)
    assert valid.all()

    backward = -flow
    result = warp_curve(
        np.asarray([4.0, 8.0], dtype=np.float32),
        np.asarray([4.0, 8.0], dtype=np.float32),
        np.asarray([True, True]),
        flow,
        backward,
        input_w=24,
        input_h=16,
        fb_threshold=0.01,
    )
    np.testing.assert_allclose(result.x, [8.0, 12.0], atol=1e-6)
    np.testing.assert_allclose(result.y, [2.0, 6.0], atol=1e-6)
    assert result.valid.all()


def test_soft_iou_prefers_aligned_neighbor_curve() -> None:
    rows = 40
    y = np.arange(rows, dtype=np.float32) * 4.0
    source = WarpResult(
        x=np.full(rows, 100.0, dtype=np.float32),
        y=y,
        valid=np.ones(rows, dtype=bool),
        forward_backward_error=np.zeros(rows, dtype=np.float32),
    )
    mask = np.ones(rows, dtype=bool)
    aligned = np.full(rows, 101.0, dtype=np.float32)
    shifted = np.full(rows, 125.0, dtype=np.float32)
    assert curve_soft_iou(
        source, aligned, y, mask, line_width=30.0, min_rows=20
    ) > curve_soft_iou(source, shifted, y, mask, line_width=30.0, min_rows=20)
    score = candidate_support_score(
        source,
        np.stack((shifted, aligned)),
        np.stack((mask, mask)),
        y,
        line_width=30.0,
        min_rows=20,
    )
    np.testing.assert_allclose(
        score,
        curve_soft_iou(source, aligned, y, mask, line_width=30.0, min_rows=20),
    )


def test_summary_requires_both_folds_and_both_thresholds() -> None:
    rows = []
    for fold in ("a", "b"):
        for index in range(30):
            row = {
                "fold": fold,
                "clip": f"{fold}_{index % 3}",
                "thresholds": [0.5, 0.75],
            }
            for prefix in (
                "previous_selected",
                "following_selected",
                "bidirectional_selected",
                "bidirectional_bank",
                "bidirectional_gt",
                "identity_bidirectional_selected",
                "identity_bidirectional_gt",
            ):
                row[f"{prefix}_good"] = 0.9
                row[f"{prefix}_wrong"] = 0.1
            for prefix in ("wrong_context_selected", "wrong_context_bank"):
                row[f"{prefix}_good"] = float(index % 2)
                row[f"{prefix}_wrong"] = float((index + 1) % 2)
            rows.append(row)
    summary = summarize_pairs(rows, bootstrap_reps=20, seed=3)
    assert summary["passed"] is True
    assert len(summary["checks"]) == 16
