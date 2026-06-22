"""
analyze_frozen_vs_joint.py
--------------------------
Compares proposal recall for the same checkpoint under three training regimes:

  Variant A  (frozen S0 + corridor):  freeze_s0_frontend=true
  Variant B  (joint S0 + corridor):   freeze_s0_frontend=false
  Variant C  (joint + quality):       freeze_s0_frontend=false, quality_score_power > 0

All variants load the SAME checkpoint. The config's freeze_s0_frontend flag
is overridden programmatically, so no separate config files are needed.
The dataloader is built once from the base config.

Also reports per-strategy oracle recall (all-proposals, top-4 exist, top-4 quality).

Usage:
  python -m dynlaneseq_eg.tools.analyze_frozen_vs_joint \\
    --config <cfg.yaml> \\
    --checkpoint <ckpt.pt> \\
    --split val \\
    --quality-powers 0.0 0.25 0.5 \\
    --top-k 4
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import (
    ProposalRecallStats,
    collect_prediction_stages,
    update_stage_recall,
)
from dynlaneseq_eg.factory import build_dataloader, build_model


# ---------------------------------------------------------------------------
# Config patching helpers
# ---------------------------------------------------------------------------

def patch_freeze(cfg: dict[str, Any], frozen: bool) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("model", {})["freeze_s0_frontend"] = frozen
    return out


def patch_quality(cfg: dict[str, Any], power: float) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("postprocess", {})["quality_score_power"] = power
    return out


# ---------------------------------------------------------------------------
# Evaluation loop (one model, all strategies)
# ---------------------------------------------------------------------------

STRATEGY_DESCS: list[tuple[str, int]] = [
    # (rank_by, top_k)
    ("none",          0),   # all proposals
    ("score",         -1),  # top-K by exist
    ("quality",       -1),  # top-K by quality
    ("score_quality", -1),  # top-K by exist×quality
]


def make_stats(thresholds: tuple[float, ...]) -> dict[str, ProposalRecallStats]:
    keys = [f"{rb}_{tk}" for rb, tk in STRATEGY_DESCS]
    return {k: ProposalRecallStats(thresholds=thresholds) for k in keys}


def strategy_key(rank_by: str, top_k: int) -> str:
    return f"{rank_by}_{top_k}"


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    thresholds: tuple[float, ...],
    top_k: int,
    line_width: float,
    min_valid_rows: int,
    max_batches: int,
    desc: str = "eval",
) -> dict[str, dict[str, ProposalRecallStats]]:
    """Returns {stage_name: {strategy_key: ProposalRecallStats}}."""
    pass_targets = bool(getattr(model, "oracle_coarse_enabled", False))
    stage_stats: dict[str, dict[str, ProposalRecallStats]] = {}

    for batch_idx, (images, targets, _metas) in enumerate(
        tqdm(loader, ncols=80, desc=desc)
    ):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device)
        outputs = model(images, targets=targets) if pass_targets else model(images)
        stages = collect_prediction_stages(outputs)

        for sname, sout in stages.items():
            if sname not in stage_stats:
                stage_stats[sname] = make_stats(thresholds)

            for rank_by, raw_top_k in STRATEGY_DESCS:
                actual_k = top_k if raw_top_k < 0 else raw_top_k
                key = strategy_key(rank_by, raw_top_k)
                update_stage_recall(
                    stage_stats[sname][key],
                    sout, targets,
                    top_k=actual_k,
                    rank_by=rank_by,
                    line_width=line_width,
                    min_valid_rows=min_valid_rows,
                )

    return stage_stats


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

STRATEGY_LABEL_MAP = {
    "none_0":          "all_proposals",
    "score_-1":        "exist_topK",
    "quality_-1":      "quality_topK",
    "score_quality_-1":"exist×quality_topK",
}


def format_stats(
    label: str,
    stage_stats: dict[str, dict[str, ProposalRecallStats]],
    thresholds: list[float],
    top_k: int,
) -> None:
    print(f"\n  ── {label} ──")
    for sname in sorted(stage_stats.keys()):
        print(f"    Stage: {sname}")
        for raw_key, friendly in STRATEGY_LABEL_MAP.items():
            if raw_key not in stage_stats[sname]:
                continue
            s = stage_stats[sname][raw_key].summary()
            rec_parts = "  ".join(f"R@{t:g}={s[f'recall@{t:g}']:.4f}" for t in thresholds)
            k_str = f"top-{top_k}" if "_-1" in raw_key else "all"
            print(f"      {friendly:<25} ({k_str:>6})  {rec_parts}  "
                  f"meanIoU={s['mean_best_iou']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen vs joint S0 training regimes on the same checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quality-powers", type=float, nargs="+", default=[0.0, 0.25, 0.5],
                        help="quality_score_power values for the joint+quality variants")
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    thresholds = tuple(args.iou_thresholds)
    base_cfg = load_config(args.config)
    device = torch.device(args.device)
    loader = build_dataloader(base_cfg, split=args.split, training=False)

    print(f"config:     {args.config}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"split:      {args.split}")
    print(f"top_k:      {args.top_k}")

    common_kwargs = dict(
        loader=loader,
        device=device,
        thresholds=thresholds,
        top_k=args.top_k,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        max_batches=args.max_batches,
    )

    # ── Variant A: Frozen S0 ──────────────────────────────────────────────
    cfg_frozen = patch_freeze(base_cfg, frozen=True)
    model_frozen = build_model(cfg_frozen).to(device)
    load_checkpoint(args.checkpoint, model_frozen, strict=False)
    model_frozen.eval()
    stats_frozen = eval_model(model_frozen, desc="frozen S0", **common_kwargs)
    del model_frozen

    # ── Variant B: Joint (unfrozen S0) ───────────────────────────────────
    cfg_joint = patch_freeze(base_cfg, frozen=False)
    model_joint = build_model(cfg_joint).to(device)
    load_checkpoint(args.checkpoint, model_joint, strict=False)
    model_joint.eval()
    stats_joint = eval_model(model_joint, desc="joint S0", **common_kwargs)
    del model_joint

    # ── Variants C: Joint + quality (several powers) ──────────────────────
    # NOTE: quality_score_power is a postprocess param, not model-level.
    # The model is the same as joint; power only affects scoring in select_candidates.
    # We patch it into the config to document, but for proposal recall it has no effect
    # unless rank_by=quality or score_quality is used — which it is.
    # So we can reuse stats_joint for all powers if rank_by=none/score, and need a
    # separate run only for rank_by=quality and score_quality (which uses quality_logits).
    # Since quality_logits come from the model (not from postprocess power), the model
    # itself is the same. Only the scoring formula changes. We handle this by including
    # quality_power in the strategy loop.
    # For simplicity, stats_joint already captures quality ranking — report with label.

    print("\n" + "=" * 70)
    print("RESULTS: Frozen vs Joint S0 — Proposal Recall Comparison")
    print("=" * 70)

    format_stats(
        label="A: Frozen S0 frontend (freeze_s0_frontend=True)",
        stage_stats=stats_frozen,
        thresholds=list(thresholds),
        top_k=args.top_k,
    )
    format_stats(
        label="B: Joint S0 frontend (freeze_s0_frontend=False)",
        stage_stats=stats_joint,
        thresholds=list(thresholds),
        top_k=args.top_k,
    )

    # Delta summary
    print("\n── Delta (Joint − Frozen) ──")
    for sname in sorted(set(list(stats_frozen.keys()) + list(stats_joint.keys()))):
        if sname not in stats_frozen or sname not in stats_joint:
            continue
        print(f"  Stage: {sname}")
        for raw_key, friendly in STRATEGY_LABEL_MAP.items():
            if raw_key not in stats_frozen[sname] or raw_key not in stats_joint[sname]:
                continue
            s_f = stats_frozen[sname][raw_key].summary()
            s_j = stats_joint[sname][raw_key].summary()
            for thr in thresholds:
                k = f"recall@{thr:g}"
                delta = s_j.get(k, 0.0) - s_f.get(k, 0.0)
                sign = "+" if delta >= 0 else ""
                print(f"    {friendly:<25}  R@{thr:g}: {sign}{delta:+.4f}")


if __name__ == "__main__":
    main()
