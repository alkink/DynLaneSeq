from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.active_corridor_diagnostics import ActiveCorridorAccumulator, matched_index_pairs
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure whether Active Corridor uses image evidence correctly.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--shuffle-modes", nargs="*", choices=["offset_reverse", "lane_roll"], default=[])
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _intervene_offset_samples(samples: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "offset_reverse":
        return samples.flip(dims=(3,))
    if mode == "lane_roll":
        b, n, p, o, c = samples.shape
        flat = samples.reshape(b * n, p, o, c)
        return torch.roll(flat, shifts=1, dims=0).reshape(b, n, p, o, c)
    raise ValueError(f"Unsupported intervention mode: {mode}")


@torch.no_grad()
def _forward_with_intervention(model, images: torch.Tensor, mode: str):
    module = getattr(model, "active_corridor", None)
    if module is None:
        raise RuntimeError("Model does not expose an active_corridor module")

    if hasattr(module, "score_coarse") and hasattr(module, "score_fine"):
        original_coarse = module.score_coarse
        original_fine = module.score_fine

        def intervene_priors(priors: torch.Tensor | None) -> torch.Tensor | None:
            return None if priors is None else _intervene_offset_samples(priors, mode)

        def score_coarse(samples, priors, queries, row_embedding):
            return original_coarse(
                _intervene_offset_samples(samples, mode),
                intervene_priors(priors),
                queries,
                row_embedding,
            )

        def score_fine(samples, priors, queries, row_embedding):
            return original_fine(
                _intervene_offset_samples(samples, mode),
                intervene_priors(priors),
                queries,
                row_embedding,
            )

        module.score_coarse = score_coarse
        module.score_fine = score_fine
        try:
            return model(images, sampler_alpha=0.0)
        finally:
            module.score_coarse = original_coarse
            module.score_fine = original_fine

    def hook(_module, inputs):
        samples, queries, row_embedding = inputs
        return _intervene_offset_samples(samples, mode), queries, row_embedding

    handle = module.register_forward_pre_hook(hook)
    try:
        return model(images, sampler_alpha=0.0)
    finally:
        handle.remove()


def _require_active_outputs(outputs: dict) -> dict[str, torch.Tensor]:
    evidence = outputs.get("evidence", {})
    required = (
        "active_center_x_rows",
        "active_pred_delta_x_rows",
        "active_offset_logits",
        "active_offsets_px",
    )
    missing = [key for key in required if key not in evidence]
    if missing:
        raise RuntimeError(f"Checkpoint/config does not expose Active Corridor tensors: {missing}")
    return evidence


def _print_summary(name: str, result: dict[str, object]) -> None:
    print(f"\n{name}:")
    keys = (
        "matched_lanes",
        "valid_rows",
        "corridor_coverage",
        "coarse_mae_px",
        "active_mae_px",
        "final_mae_px",
        "oracle_discrete_mae_px",
        "active_lane_improvement_rate",
        "final_lane_improvement_rate",
        "offset_target_pearson",
        "offset_direction_accuracy",
        "offset_top1_accuracy",
    )
    for key in keys:
        value = result[key]
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    for mode, stats in result["evidence_interventions"].items():
        print(
            f"  intervention={mode}: shuffled_active_mae={stats['shuffled_active_mae_px']:.4f} "
            f"shuffled_final_mae={stats['shuffled_final_mae_px']:.4f} "
            f"delta_change={stats['pred_delta_mean_abs_change_px']:.4f} "
            f"normal_final_better={stats['normal_final_better_rate']:.4f}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    if args.num_workers is not None:
        cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if args.list_path:
        cfg.setdefault("dataset", {}).setdefault("lists", {})[args.split] = str(Path(args.list_path).resolve())

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()
    matcher = build_matcher(cfg)
    loader = build_dataloader(cfg, split=args.split, training=False)
    accumulators = {
        "best_per_gt": ActiveCorridorAccumulator(),
        "all_matches": ActiveCorridorAccumulator(),
    }
    image_count = 0

    for batch_index, (images, targets, _metas) in enumerate(tqdm(loader, desc="active corridor diagnostics")):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        outputs = model(images, sampler_alpha=0.0)
        evidence = _require_active_outputs(outputs)
        matches = matcher(outputs["coarse"], targets)
        intervened_outputs = {mode: _forward_with_intervention(model, images, mode) for mode in args.shuffle_modes}

        for batch_pos, (target, match) in enumerate(zip(targets, matches)):
            for name, best_per_gt in (("best_per_gt", True), ("all_matches", False)):
                pred_idx, gt_idx = matched_index_pairs(
                    evidence["active_center_x_rows"][batch_pos],
                    target,
                    match,
                    best_per_gt=best_per_gt,
                )
                if pred_idx.numel() == 0:
                    continue
                interventions = {}
                for mode, intervention_output in intervened_outputs.items():
                    intervention_evidence = _require_active_outputs(intervention_output)
                    interventions[mode] = (
                        intervention_evidence["active_pred_delta_x_rows"][batch_pos, pred_idx],
                        intervention_output["final"]["pred_x_rows"][batch_pos, pred_idx],
                    )
                accumulators[name].update(
                    center_x=evidence["active_center_x_rows"][batch_pos, pred_idx],
                    pred_delta=evidence["active_pred_delta_x_rows"][batch_pos, pred_idx],
                    final_x=outputs["final"]["pred_x_rows"][batch_pos, pred_idx],
                    target_x=target["x_rows"][gt_idx],
                    valid_mask=target["valid_mask"][gt_idx],
                    offsets=evidence["active_offsets_px"],
                    logits=evidence["active_offset_logits"][batch_pos, pred_idx],
                    interventions=interventions,
                )
        image_count += int(images.shape[0])

    results = {name: accumulator.as_dict() for name, accumulator in accumulators.items()}
    payload = {
        "metadata": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "list_path": str(Path(args.list_path).resolve()) if args.list_path else str(loader.dataset.list_path),
            "images": image_count,
            "eval_batch_size": int(args.eval_batch_size),
            "shuffle_modes": list(args.shuffle_modes),
        },
        "results": results,
    }
    for name, result in results.items():
        _print_summary(name, result)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
