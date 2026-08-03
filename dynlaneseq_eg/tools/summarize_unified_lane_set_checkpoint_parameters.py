from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict

import torch


StateDict = Dict[str, torch.Tensor]


def _config_signature(cfg: dict[str, Any]) -> str | None:
    if not isinstance(cfg, dict) or not cfg:
        return None
    contract = {
        "model": cfg.get("model"),
        "matcher": cfg.get("matcher"),
        "loss": cfg.get("loss"),
        "optimizer": cfg.get("optimizer"),
        "scheduler": cfg.get("scheduler"),
        "seed": cfg.get("training", {}).get("seed"),
    }
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optimizer_steps(optimizer: Any) -> list[float]:
    if not isinstance(optimizer, dict):
        return []
    steps: list[float] = []
    for state in optimizer.get("state", {}).values():
        if not isinstance(state, dict) or state.get("step") is None:
            continue
        value = state["step"]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        steps.append(float(value))
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize module-wise parameter drift and query/anchor diversity "
            "over unified lane-set checkpoints without running the dataset."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--input-width", type=int, default=1600)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def _load(path: str) -> tuple[StateDict, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    state = dict(payload["model"])
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"model state contains non-tensor entries: {path}")
    optimizer = payload.get("optimizer", {})
    scheduler = payload.get("scheduler", {})
    optimizer_steps = _optimizer_steps(optimizer)
    optimizer_groups = (
        optimizer.get("param_groups", []) if isinstance(optimizer, dict) else []
    )
    return state, {
        "checkpoint": path,
        "iteration": int(payload.get("iteration", -1)),
        "model_tensors": len(state),
        "model_elements": int(sum(value.numel() for value in state.values())),
        "config_contract_sha256": _config_signature(payload.get("cfg")),
        "optimizer_groups": len(optimizer_groups),
        "optimizer_state_entries": len(optimizer.get("state", {}))
        if isinstance(optimizer, dict)
        else 0,
        "optimizer_step_min": min(optimizer_steps) if optimizer_steps else None,
        "optimizer_step_max": max(optimizer_steps) if optimizer_steps else None,
        "optimizer_step_unique": sorted(set(optimizer_steps)),
        "optimizer_lrs": [group.get("lr") for group in optimizer_groups],
        "optimizer_initial_lrs": [
            group.get("initial_lr") for group in optimizer_groups
        ],
        "scheduler_last_epoch": scheduler.get("last_epoch")
        if isinstance(scheduler, dict)
        else None,
        "scheduler_step_count": scheduler.get("_step_count")
        if isinstance(scheduler, dict)
        else None,
    }


def _prefixes(*prefixes: str) -> Callable[[str], bool]:
    return lambda key: key.startswith(prefixes)


def _fpn_running_buffer(key: str) -> bool:
    return key.startswith("encoder.fpn.") and key.endswith(
        ("running_mean", "running_var", "num_batches_tracked")
    )


def _fpn_nonrunning_state(key: str) -> bool:
    return key.startswith(("encoder.fpn.", "encoder.proj.")) and not (
        _fpn_running_buffer(key)
    )


GROUP_RULES = {
    "backbone": _prefixes("encoder.backbone."),
    "fpn_nonrunning_state": _fpn_nonrunning_state,
    "fpn_running_buffers": _fpn_running_buffer,
    "row_readout": _prefixes(
        "structured_query_head.row_norm.",
        "structured_query_head.row_x.",
        "structured_query_head.row_delta_heads.",
    ),
    "lane_state_core": _prefixes("structured_query_head.lane_state_layers."),
    "exist_head": _prefixes("structured_query_head.exist."),
    "range_head": _prefixes("structured_query_head.range."),
    "row_reference_path": _prefixes(
        "structured_query_head.layers.",
        "structured_query_head.feature_proj.",
        "structured_query_head.reference_",
        "structured_query_head.instance_tokens.",
        "structured_query_head.row_tokens.",
        "structured_query_head.x_tokens.",
    ),
}


TRACKED_TENSORS = (
    "structured_query_head.row_norm.weight",
    "structured_query_head.row_norm.bias",
    "structured_query_head.row_x.weight",
    "structured_query_head.row_x.bias",
    "structured_query_head.row_delta_heads.0.weight",
    "structured_query_head.row_delta_heads.1.weight",
    "structured_query_head.row_delta_heads.2.weight",
    "structured_query_head.row_delta_heads.3.weight",
    "structured_query_head.reference_logit_scale",
)


def _selected(
    state: StateDict,
    predicate: Callable[[str], bool],
) -> list[torch.Tensor]:
    return [
        value.detach().float()
        for key, value in state.items()
        if predicate(key) and torch.is_floating_point(value)
    ]


def _norm(tensors: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float((value * value).sum()) for value in tensors))


def _delta_norm(
    state: StateDict,
    reference: StateDict,
    predicate: Callable[[str], bool],
) -> tuple[float, float | None]:
    keys = [
        key
        for key, value in state.items()
        if predicate(key) and torch.is_floating_point(value)
    ]
    delta_sq = 0.0
    reference_sq = 0.0
    for key in keys:
        value = state[key].detach().float()
        baseline = reference[key].detach().float()
        delta = value - baseline
        delta_sq += float((delta * delta).sum())
        reference_sq += float((baseline * baseline).sum())
    delta_norm = math.sqrt(delta_sq)
    relative = delta_norm / math.sqrt(reference_sq) if reference_sq > 0.0 else None
    return delta_norm, relative


def _candidate_diversity(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().flatten(1)
    normalized = torch.nn.functional.normalize(flat, dim=1, eps=1e-12)
    cosine = normalized @ normalized.transpose(0, 1)
    mask = ~torch.eye(cosine.shape[0], dtype=torch.bool)
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    pairwise = torch.pdist(flat, p=2)
    return {
        "off_diagonal_cosine": float(cosine[mask].mean()),
        "effective_rank": float(effective_rank),
        "mean_pairwise_l2": float(pairwise.mean()) if pairwise.numel() else 0.0,
        "mean_candidate_std": float(flat.std(dim=0, unbiased=False).mean()),
    }


def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float()
    return {
        "norm": float(value.norm()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def main() -> None:
    args = parse_args()
    if int(args.input_width) < 2:
        raise ValueError("input-width must be at least two")
    loaded = [_load(path) for path in args.inputs]
    loaded.sort(key=lambda item: int(item[1]["iteration"]))
    iterations = [int(metadata["iteration"]) for _state, metadata in loaded]
    if any(iteration < 0 for iteration in iterations):
        raise ValueError(f"checkpoint iteration missing: {iterations}")
    if len(iterations) != len(set(iterations)):
        raise ValueError(f"duplicate checkpoint iterations: {iterations}")
    signatures = [metadata["config_contract_sha256"] for _state, metadata in loaded]
    if any(signature is None for signature in signatures):
        raise ValueError("one or more checkpoints have no embedded config contract")
    if len(set(signatures)) != 1:
        raise ValueError(f"checkpoint config contracts differ: {signatures}")
    reference = loaded[0][0]
    reference_keys = set(reference)

    rows: list[dict[str, Any]] = []
    for state, metadata in loaded:
        if set(state) != reference_keys:
            raise ValueError(
                f"checkpoint keys differ at iteration {metadata['iteration']}"
            )
        row: dict[str, Any] = dict(metadata)
        row["finite"] = all(
            bool(torch.isfinite(value).all())
            for value in state.values()
            if torch.is_floating_point(value)
        )
        row["groups"] = {}
        for name, predicate in GROUP_RULES.items():
            tensors = _selected(state, predicate)
            delta_norm, relative = _delta_norm(state, reference, predicate)
            row["groups"][name] = {
                "tensors": len(tensors),
                "elements": int(sum(value.numel() for value in tensors)),
                "norm": _norm(tensors),
                "delta_norm_from_first": delta_norm,
                "relative_delta_from_first": relative,
            }
        row["tracked_tensors"] = {
            key: _tensor_summary(state[key])
            for key in TRACKED_TENSORS
            if key in state
        }
        instance_key = "structured_query_head.instance_tokens.weight"
        anchor_key = "structured_query_head.reference_anchor_logits"
        row["instance_token_diversity"] = _candidate_diversity(
            state[instance_key]
        )
        anchors_px = torch.sigmoid(state[anchor_key].detach().float()) * float(
            int(args.input_width) - 1
        )
        row["reference_anchor_diversity_px"] = _candidate_diversity(anchors_px)
        rows.append(row)

    payload = {
        "diagnostic_only": True,
        "warning": (
            "Static parameter diversity does not prove runtime activation "
            "diversity; use it to locate drift, not to estimate F1."
        ),
        "reference_iteration": int(rows[0]["iteration"]),
        "config_contract_sha256": signatures[0],
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {
            "iteration": row["iteration"],
            "checkpoint": row["checkpoint"],
            "finite": row["finite"],
        }
        for group, values in row["groups"].items():
            flat[f"{group}_relative_delta"] = values["relative_delta_from_first"]
        for key, values in row["tracked_tensors"].items():
            prefix = "structured_query_head."
            label = (
                key[len(prefix) :] if key.startswith(prefix) else key
            ).replace(".", "_")
            flat[f"{label}_norm"] = values["norm"]
        flat["instance_effective_rank"] = row["instance_token_diversity"][
            "effective_rank"
        ]
        flat["anchor_effective_rank"] = row["reference_anchor_diversity_px"][
            "effective_rank"
        ]
        flat["anchor_candidate_std_px"] = row["reference_anchor_diversity_px"][
            "mean_candidate_std"
        ]
        flat_rows.append(flat)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_json}")
    print(f"output_csv: {output_csv}")


if __name__ == "__main__":
    main()
