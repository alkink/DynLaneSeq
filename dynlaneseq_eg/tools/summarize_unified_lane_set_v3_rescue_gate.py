from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


READOUT_PREFIXES = (
    "structured_query_head.row_norm.",
    "structured_query_head.row_x.",
)
LANE_STATE_PREFIXES = ("structured_query_head.lane_state_layers.",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched 25k->30k V3 collapse-rescue arms without "
            "selecting a test-set threshold."
        )
    )
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        nargs=3,
        metavar=("NAME", "AUDIT_JSON", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError("missing audit field: " + ".".join(keys))
        value = value[key]
    return value


def audit_metrics(payload: dict[str, Any]) -> dict[str, float]:
    alignment = payload["score_official_iou_alignment"]
    count = payload["count_calibration"]
    return {
        "all_candidates_recall_050": float(
            nested(payload, "capacity", "0.50", "all_candidates_recall")
        ),
        "all_candidates_recall_075": float(
            nested(payload, "capacity", "0.75", "all_candidates_recall")
        ),
        "direct_top4_recall_050": float(
            nested(payload, "capacity", "0.50", "direct_topk_recall")
        ),
        "direct_top4_recall_075": float(
            nested(payload, "capacity", "0.75", "direct_topk_recall")
        ),
        "oracle_top4_recall_050": float(
            nested(payload, "capacity", "0.50", "oracle_topk_recall")
        ),
        "oracle_top4_recall_075": float(
            nested(payload, "capacity", "0.75", "oracle_topk_recall")
        ),
        "matched_mean_best_official_iou": float(
            alignment["matched_mean_best_official_iou"]
        ),
        "matched_mean_score": float(alignment["matched_mean_score"]),
        "unmatched_mean_score": float(alignment["unmatched_mean_score"]),
        "score_iou_pearson": float(
            alignment["pearson_score_vs_best_official_iou"]
        ),
        "mean_foreground_probability_mass": float(
            count["mean_foreground_probability_mass"]
        ),
        "f1_050_score_0p20": float(
            nested(
                payload,
                "deployed_operating_points",
                "score_0.20",
                "0.50",
                "f1",
            )
        ),
    }


def model_state(path: str) -> tuple[int, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"checkpoint has no model state: {path}")
    return int(payload.get("iteration", -1)), payload["model"]


def selected_norms(state: dict[str, torch.Tensor]) -> dict[str, float]:
    required = (
        "structured_query_head.row_norm.weight",
        "structured_query_head.row_norm.bias",
        "structured_query_head.row_x.weight",
        "structured_query_head.row_x.bias",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise KeyError("checkpoint misses V3 readout tensors: " + ", ".join(missing))
    return {
        name[len("structured_query_head.") :] + "_l2": float(
            state[name].detach().float().norm()
        )
        for name in required
    }


def prefix_relative_delta(
    source: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> float:
    keys = sorted(
        key
        for key, value in source.items()
        if value.is_floating_point() and key.startswith(prefixes)
    )
    if not keys:
        raise ValueError(f"no floating tensors matched prefixes {prefixes}")
    missing = [key for key in keys if key not in candidate]
    if missing:
        raise KeyError("candidate checkpoint misses tensors: " + ", ".join(missing))
    numerator = 0.0
    denominator = 0.0
    for key in keys:
        before = source[key].detach().float()
        after = candidate[key].detach().float()
        if before.shape != after.shape:
            raise ValueError(f"shape mismatch for {key}: {before.shape} vs {after.shape}")
        numerator += float((after - before).square().sum())
        denominator += float(before.square().sum())
    return math.sqrt(numerator / max(denominator, 1e-30))


def preserved(source: dict[str, float], arm: dict[str, float]) -> bool:
    return bool(
        arm["all_candidates_recall_050"]
        >= max(0.85, source["all_candidates_recall_050"] - 0.05)
        and arm["all_candidates_recall_075"]
        >= max(0.60, source["all_candidates_recall_075"] - 0.10)
        and arm["matched_mean_best_official_iou"]
        >= max(0.65, source["matched_mean_best_official_iou"] - 0.10)
    )


def infer_next_step(rows: dict[str, dict[str, Any]]) -> dict[str, str]:
    statuses = {
        name: bool(row["geometry_preserved"])
        for name, row in rows.items()
    }
    control = statuses.get("control")
    contract = statuses.get("contract")
    scale = statuses.get("scale")
    if control is True:
        return {
            "signal": "collapse_not_yet_visible_at_30k",
            "action": (
                "Continue the same three arms to 35k/40k; do not choose a "
                "winner while the exact control remains healthy."
            ),
        }
    if control is False and contract is True and scale is False:
        return {
            "signal": "training_contract_intervention_is_locally_sufficient",
            "action": (
                "Factor the contract arm into matcher, intermediate-score, "
                "and cardinality/margin sub-ablations before a from-scratch run."
            ),
        }
    if control is False and scale is True and contract is False:
        return {
            "signal": "decoder_scale_intervention_is_locally_sufficient",
            "action": (
                "Extend the scale arm and audit norm/geometry slopes; then tune "
                "only the two isolated LR groups from scratch."
            ),
        }
    if control is False and contract is True and scale is True:
        return {
            "signal": "both_intervention_families_prevent_the_local_collapse",
            "action": (
                "Extend both to 40k and prefer the arm with higher strict-IoU "
                "capacity and bounded readout norms; test a combined arm only "
                "after their separate effects remain positive."
            ),
        }
    if control is False and contract is False and scale is False:
        return {
            "signal": "neither_minimal_intervention_is_sufficient",
            "action": (
                "Run the combined contract+scale arm; if it also fails, move "
                "to the structural global-acquisition/persistent-identity gate."
            ),
        }
    return {
        "signal": "incomplete_or_unexpected_arm_set",
        "action": "Inspect individual rows; no architecture decision is licensed.",
    }


def main() -> None:
    args = parse_args()
    source_audit = load_json(args.source_audit)
    source_iteration, source_state = model_state(args.source_checkpoint)
    if source_iteration != int(source_audit["checkpoint_iteration"]):
        raise ValueError("source audit/checkpoint iteration mismatch")
    source_indices = source_audit.get("sampled_dataset_indices")
    source_metrics = audit_metrics(source_audit)
    source_norms = selected_norms(source_state)

    rows: dict[str, dict[str, Any]] = {}
    for name, audit_path, checkpoint_path in args.arm:
        if name in rows:
            raise ValueError(f"duplicate arm name: {name}")
        audit = load_json(audit_path)
        if audit.get("sampled_dataset_indices") != source_indices:
            raise ValueError(f"sample indices differ for arm {name}")
        iteration, state = model_state(checkpoint_path)
        if iteration != int(audit["checkpoint_iteration"]):
            raise ValueError(f"audit/checkpoint iteration mismatch for arm {name}")
        metrics = audit_metrics(audit)
        norms = selected_norms(state)
        rows[name] = {
            "iteration": iteration,
            "audit": audit_path,
            "checkpoint": checkpoint_path,
            "metrics": metrics,
            "delta_from_source": {
                key: metrics[key] - source_metrics[key]
                for key in metrics
            },
            "parameter_norms": norms,
            "parameter_norm_ratios_from_source": {
                key: norms[key] / max(source_norms[key], 1e-30)
                for key in norms
            },
            "readout_relative_parameter_delta": prefix_relative_delta(
                source_state, state, READOUT_PREFIXES
            ),
            "lane_state_relative_parameter_delta": prefix_relative_delta(
                source_state, state, LANE_STATE_PREFIXES
            ),
            "geometry_preserved": preserved(source_metrics, metrics),
        }
        del state

    decision = infer_next_step(rows)
    payload = {
        "diagnostic_only": True,
        "warning": (
            "This matched 5k continuation gate identifies a local collapse "
            "mechanism. It is not a final model selection or benchmark result."
        ),
        "source": {
            "iteration": source_iteration,
            "audit": args.source_audit,
            "checkpoint": args.source_checkpoint,
            "metrics": source_metrics,
            "parameter_norms": source_norms,
        },
        "arms": rows,
        "decision": decision,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
