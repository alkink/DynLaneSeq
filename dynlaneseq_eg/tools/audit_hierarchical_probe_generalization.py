from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.tools.probe_four_slot_coverage_selector import (
    _load_cache,
    _load_source_selector,
    _validate_cache,
    source_scores,
)
from dynlaneseq_eg.tools.probe_hierarchical_cluster_selector import (
    ClusterExistenceProbe,
    RepresentativeQualityProbe,
    _representative_ids,
    _selected_cluster_ids,
    build_hierarchical_supervision,
    evaluate_hierarchy,
    hierarchy_gate,
)


MODES = (
    "source_nms",
    "learned_cluster_source_representative",
    "source_cluster_learned_representative",
    "learned_hierarchical",
)
LEARNED_MODES = MODES[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved frozen hierarchical selector on its disjoint "
            "train and validation caches without further optimization."
        )
    )
    parser.add_argument("--probe-report", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--train-cache", default="")
    parser.add_argument("--val-cache", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load_torch(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _required_path(override: str, report: dict[str, Any], key: str) -> str:
    value = str(override or report.get(key, ""))
    if not value:
        raise ValueError(f"missing {key}; provide an explicit override")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"missing {key}: {path}")
    return str(path)


def _model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    names = (
        "hidden_dim",
        "num_layers",
        "num_heads",
        "ff_dim",
        "dropout",
    )
    missing = [name for name in names if name not in config]
    if missing:
        raise ValueError(f"probe config is missing model fields: {missing}")
    return {name: config[name] for name in names}


def _build_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "input_h": "input_h",
        "input_w": "input_w",
        "min_valid_rows": "min_valid_rows",
        "row_visibility_thresh": "row_visibility_thresh",
        "nms_distance": "nms_distance",
        "nms_min_overlap_points": "nms_min_overlap_points",
        "top_k": "top_k",
        "positive_iou": "positive_iou",
    }
    missing = [source for source in mapping if source not in config]
    if missing:
        raise ValueError(f"probe config is missing hierarchy fields: {missing}")
    return {target: config[source] for source, target in mapping.items()}


def _gate_kwargs(config: dict[str, Any]) -> dict[str, float]:
    names = (
        "cluster_min_gain_050",
        "representative_min_gain_070",
        "hierarchical_min_gain_050",
        "hierarchical_min_gain_070",
    )
    missing = [name for name in names if name not in config]
    if missing:
        raise ValueError(f"probe config is missing gate fields: {missing}")
    return {name: float(config[name]) for name in names}


def verify_validation_replay(
    replayed: dict[str, Any],
    recorded: dict[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, int] = {}
    for mode in MODES:
        for suffix in ("050", "070"):
            key = f"tp_{suffix}"
            delta = int(replayed["modes"][mode][key]) - int(
                recorded["modes"][mode][key]
            )
            comparisons[f"{mode}.{key}_delta"] = delta
            if delta != 0:
                raise ValueError(
                    f"saved validation replay mismatch for {mode}/{key}: {delta}"
                )
    for threshold in ("0.50", "0.70"):
        delta = int(replayed["oracle_top4"][threshold]["tp"]) - int(
            recorded["oracle_top4"][threshold]["tp"]
        )
        comparisons[f"oracle_top4.{threshold}.tp_delta"] = delta
        if delta != 0:
            raise ValueError(
                f"saved validation oracle mismatch at {threshold}: {delta}"
            )
    return {"matched": True, "comparisons": comparisons}


def evaluation_gains(evaluation: dict[str, Any]) -> dict[str, Any]:
    source = evaluation["modes"]["source_nms"]
    result: dict[str, Any] = {
        "source": {
            "recall_050": float(source["recall_050"]),
            "recall_070": float(source["recall_070"]),
            "f1_050": float(source["f1_050"]),
            "f1_070": float(source["f1_070"]),
        },
        "arms": {},
    }
    for mode in LEARNED_MODES:
        row = evaluation["modes"][mode]
        arm: dict[str, float] = {}
        for suffix, threshold in (("050", "0.50"), ("070", "0.70")):
            gain = float(row[f"recall_{suffix}"]) - float(
                source[f"recall_{suffix}"]
            )
            oracle_gap = float(
                evaluation["oracle_top4"][threshold]["recall"]
            ) - float(source[f"recall_{suffix}"])
            arm[f"gain_recall_{suffix}_points"] = 100.0 * gain
            arm[f"gain_f1_{suffix}_points"] = 100.0 * (
                float(row[f"f1_{suffix}"]) - float(source[f"f1_{suffix}"])
            )
            arm[f"oracle_gap_recovered_{suffix}_fraction"] = (
                gain / oracle_gap if oracle_gap > 1e-12 else 0.0
            )
        result["arms"][mode] = arm
    return result


def _pearson(probability: torch.Tensor, target: torch.Tensor) -> float | None:
    if int(probability.numel()) < 2:
        return None
    probability = probability.double()
    target = target.double()
    probability = probability - probability.mean()
    target = target - target.mean()
    denominator = probability.square().sum().sqrt() * target.square().sum().sqrt()
    if float(denominator) <= 1e-12:
        return None
    return float((probability * target).sum() / denominator)


def _association_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    positive_iou: float,
) -> dict[str, Any]:
    probability = torch.sigmoid(logits[valid.bool()].float())
    target = targets[valid.bool()].float()
    positive = target >= float(positive_iou)
    negative = ~positive

    def mean_or_none(values: torch.Tensor) -> float | None:
        return float(values.mean()) if int(values.numel()) else None

    return {
        "items": int(target.numel()),
        "positive_items": int(positive.sum()),
        "negative_items": int(negative.sum()),
        "positive_fraction": float(positive.float().mean())
        if int(target.numel())
        else 0.0,
        "pearson_probability_vs_target": _pearson(probability, target),
        "mean_absolute_error": mean_or_none((probability - target).abs()),
        "mean_probability_positive": mean_or_none(probability[positive]),
        "mean_probability_negative": mean_or_none(probability[negative]),
        "positive_negative_probability_margin": (
            mean_or_none(probability[positive]) - mean_or_none(probability[negative])
            if bool(positive.any()) and bool(negative.any())
            else None
        ),
    }


@torch.no_grad()
def descriptor_target_fit(
    cluster_probe: ClusterExistenceProbe,
    representative_probe: RepresentativeQualityProbe,
    cache: dict[str, Any],
    hierarchy: dict[str, Any],
    scores: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    positive_iou: float,
    top_k: int,
) -> dict[str, Any]:
    cluster_probe.eval()
    representative_probe.eval()
    cluster_rows: list[torch.Tensor] = []
    representative_rows: list[torch.Tensor] = []
    for start in range(0, int(cache["features"].shape[0]), int(batch_size)):
        stop = start + int(batch_size)
        features = cache["features"][start:stop].to(device, dtype=torch.float32)
        candidate_valid = cache["candidate_valid"][start:stop].to(device)
        source = scores[start:stop].to(device)
        cluster_rows.append(
            cluster_probe(
                features,
                candidate_valid,
                hierarchy["membership"][start:stop].to(device),
                hierarchy["cluster_valid"][start:stop].to(device),
                source,
            ).cpu()
        )
        representative_rows.append(
            representative_probe(features, candidate_valid, source).cpu()
        )
    cluster_logits = torch.cat(cluster_rows)
    representative_logits = torch.cat(representative_rows)

    source_cluster_target = 0.0
    learned_cluster_target = 0.0
    oracle_cluster_target = 0.0
    source_representative_target = 0.0
    learned_representative_target = 0.0
    oracle_representative_target = 0.0
    cluster_selections = 0
    representative_selections = 0
    learned_representative_is_best = 0
    for image_index, record in enumerate(hierarchy["records"]):
        cluster_valid = hierarchy["cluster_valid"][image_index]
        learned_clusters = _selected_cluster_ids(
            cluster_logits[image_index],
            cluster_valid,
            top_k=top_k,
        )
        source_clusters = record["source_selected_cluster_ids"]
        valid_targets = hierarchy["cluster_targets"][image_index][cluster_valid]
        oracle_cluster_target += float(
            valid_targets.topk(min(int(top_k), int(valid_targets.numel()))).values.sum()
        ) if int(valid_targets.numel()) else 0.0
        source_cluster_target += sum(
            float(hierarchy["cluster_targets"][image_index, cluster_index])
            for cluster_index in source_clusters
        )
        learned_cluster_target += sum(
            float(hierarchy["cluster_targets"][image_index, cluster_index])
            for cluster_index in learned_clusters
        )
        cluster_selections += len(source_clusters)

        representative_target = hierarchy["representative_targets"][image_index]
        source_ids = _representative_ids(
            source_clusters,
            record,
            representative_logits[image_index],
            learned=False,
        )
        learned_ids = _representative_ids(
            source_clusters,
            record,
            representative_logits[image_index],
            learned=True,
        )
        for cluster_index, source_id, learned_id in zip(
            source_clusters,
            source_ids,
            learned_ids,
        ):
            members = record["members"][int(cluster_index)]
            oracle_value = max(float(representative_target[index]) for index in members)
            source_representative_target += float(representative_target[source_id])
            learned_value = float(representative_target[learned_id])
            learned_representative_target += learned_value
            oracle_representative_target += oracle_value
            learned_representative_is_best += int(
                learned_value >= oracle_value - 1e-7
            )
            representative_selections += 1

    return {
        "cluster": {
            **_association_summary(
                cluster_logits,
                hierarchy["cluster_targets"],
                hierarchy["cluster_valid"],
                positive_iou=positive_iou,
            ),
            "mean_selected_target_source": source_cluster_target
            / float(max(cluster_selections, 1)),
            "mean_selected_target_learned": learned_cluster_target
            / float(max(cluster_selections, 1)),
            "mean_selected_target_oracle": oracle_cluster_target
            / float(max(cluster_selections, 1)),
        },
        "representative": {
            **_association_summary(
                representative_logits,
                hierarchy["representative_targets"],
                cache["candidate_valid"],
                positive_iou=positive_iou,
            ),
            "mean_selected_target_source": source_representative_target
            / float(max(representative_selections, 1)),
            "mean_selected_target_learned": learned_representative_target
            / float(max(representative_selections, 1)),
            "mean_selected_target_oracle": oracle_representative_target
            / float(max(representative_selections, 1)),
            "learned_exact_best_fraction": learned_representative_is_best
            / float(max(representative_selections, 1)),
        },
    }


def diagnose_generalization(
    train_gate: dict[str, Any],
    val_gate: dict[str, Any],
) -> dict[str, Any]:
    train_arms = {
        name: bool(train_gate[name]["positive"])
        for name in ("cluster", "representative", "hierarchical")
    }
    val_arms = {
        name: bool(val_gate[name]["positive"])
        for name in ("cluster", "representative", "hierarchical")
    }
    if bool(train_gate["dual_head_positive"]) and bool(
        val_gate["dual_head_positive"]
    ):
        diagnosis = "hierarchical_contract_fits_and_generalizes"
    elif bool(train_gate["dual_head_positive"]):
        diagnosis = "hierarchical_contract_fits_train_but_does_not_generalize"
    elif any(train_arms.values()) and not any(val_arms.values()):
        diagnosis = "partial_training_cache_fit_without_validation_transfer"
    elif not any(train_arms.values()) and not any(val_arms.values()):
        diagnosis = "saved_best_probe_does_not_fit_the_contract_even_in_sample"
    elif any(val_arms.values()):
        diagnosis = "validation_signal_exists_without_full_training_fit"
    else:
        diagnosis = "partial_hierarchical_signal"
    return {
        "diagnosis": diagnosis,
        "train_positive_arms": train_arms,
        "validation_positive_arms": val_arms,
        "train_dual_head_positive": bool(train_gate["dual_head_positive"]),
        "validation_dual_head_positive": bool(val_gate["dual_head_positive"]),
        "limits": (
            "The checkpoint contains the validation-selected 250-step states, "
            "not the discarded 3000-step terminal states. This audit measures "
            "generalization of the saved useful probe and cannot prove whether "
            "a later overfit state memorized the training cache."
        ),
    }


def _gain_gap(
    train_gains: dict[str, Any],
    val_gains: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in LEARNED_MODES:
        output[mode] = {}
        for suffix in ("050", "070"):
            key = f"gain_recall_{suffix}_points"
            output[mode][f"train_minus_validation_{key}"] = float(
                train_gains["arms"][mode][key]
            ) - float(val_gains["arms"][mode][key])
    return output


def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 1:
        raise ValueError("batch_size must be positive")
    report_path = Path(args.probe_report)
    checkpoint_path = Path(args.probe_checkpoint)
    if not report_path.is_file():
        raise FileNotFoundError(f"missing probe report: {report_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing probe checkpoint: {checkpoint_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload = _load_torch(checkpoint_path)
    probe_config = dict(report.get("probe_config", {}))
    saved_config = dict(payload.get("config", {}))
    for key, value in probe_config.items():
        if key in saved_config and saved_config[key] != value:
            raise ValueError(f"probe report/checkpoint config mismatch at {key}")

    config_path = _required_path(args.config, report, "config")
    source_checkpoint = _required_path(
        args.source_checkpoint,
        report,
        "source_checkpoint",
    )
    train_cache_path = _required_path(args.train_cache, report, "train_cache")
    val_cache_path = _required_path(args.val_cache, report, "val_cache")
    train_cache = _load_cache(train_cache_path)
    val_cache = _load_cache(val_cache_path)
    _validate_cache(train_cache, "train")
    _validate_cache(val_cache, "validation")
    overlap = set(train_cache["metadata"]["image_paths"]) & set(
        val_cache["metadata"]["image_paths"]
    )
    if overlap:
        raise ValueError(f"train/validation cache overlap: {sorted(overlap)[0]}")

    device = torch.device(args.device)
    selector, source_iteration = _load_source_selector(
        config_path,
        source_checkpoint,
    )
    if int(source_iteration) != int(report["source_iteration"]):
        raise ValueError("source checkpoint iteration differs from probe report")
    train_scores = source_scores(
        selector,
        train_cache,
        device=device,
        batch_size=args.batch_size,
    )
    val_scores = source_scores(
        selector,
        val_cache,
        device=device,
        batch_size=args.batch_size,
    )
    hierarchy_kwargs = _build_kwargs(probe_config)
    train_hierarchy = build_hierarchical_supervision(
        train_cache,
        train_scores,
        **hierarchy_kwargs,
    )
    val_hierarchy = build_hierarchical_supervision(
        val_cache,
        val_scores,
        **hierarchy_kwargs,
    )
    feature_dim = int(train_cache["features"].shape[-1])
    if int(val_cache["features"].shape[-1]) != feature_dim:
        raise ValueError("train/validation descriptor dimensions differ")
    model_kwargs = _model_kwargs(probe_config)
    cluster_probe = ClusterExistenceProbe(feature_dim, **model_kwargs).to(device)
    representative_probe = RepresentativeQualityProbe(
        feature_dim,
        **model_kwargs,
    ).to(device)
    cluster_probe.load_state_dict(payload["cluster_model"], strict=True)
    representative_probe.load_state_dict(
        payload["representative_model"],
        strict=True,
    )

    evaluation_kwargs = {
        "device": device,
        "batch_size": int(args.batch_size),
        "top_k": int(probe_config["top_k"]),
    }
    train_evaluation = evaluate_hierarchy(
        cluster_probe,
        representative_probe,
        train_cache,
        train_hierarchy,
        train_scores,
        **evaluation_kwargs,
    )
    val_evaluation = evaluate_hierarchy(
        cluster_probe,
        representative_probe,
        val_cache,
        val_hierarchy,
        val_scores,
        **evaluation_kwargs,
    )
    replay = verify_validation_replay(val_evaluation, report["evaluation"])
    gate_kwargs = _gate_kwargs(probe_config)
    train_gate = hierarchy_gate(train_evaluation, **gate_kwargs)
    val_gate = hierarchy_gate(val_evaluation, **gate_kwargs)
    train_gains = evaluation_gains(train_evaluation)
    val_gains = evaluation_gains(val_evaluation)
    fit_kwargs = {
        "device": device,
        "batch_size": int(args.batch_size),
        "positive_iou": float(probe_config["positive_iou"]),
        "top_k": int(probe_config["top_k"]),
    }
    train_fit = descriptor_target_fit(
        cluster_probe,
        representative_probe,
        train_cache,
        train_hierarchy,
        train_scores,
        **fit_kwargs,
    )
    val_fit = descriptor_target_fit(
        cluster_probe,
        representative_probe,
        val_cache,
        val_hierarchy,
        val_scores,
        **fit_kwargs,
    )
    result = {
        "diagnostic_only": True,
        "warning": (
            "No model was optimized. This replays the saved validation-selected "
            "probe on disjoint frozen train and validation caches."
        ),
        "probe_report": str(report_path),
        "probe_checkpoint": str(checkpoint_path),
        "source_iteration": int(source_iteration),
        "train_images": int(train_cache["features"].shape[0]),
        "val_images": int(val_cache["features"].shape[0]),
        "saved_best_steps": payload.get("best_steps"),
        "validation_replay": replay,
        "train": {
            "evaluation": train_evaluation,
            "gains": train_gains,
            "gate": train_gate,
            "descriptor_target_fit": train_fit,
        },
        "validation": {
            "evaluation": val_evaluation,
            "gains": val_gains,
            "gate": val_gate,
            "descriptor_target_fit": val_fit,
        },
        "train_validation_gain_gap": _gain_gap(train_gains, val_gains),
        "decision": diagnose_generalization(train_gate, val_gate),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
