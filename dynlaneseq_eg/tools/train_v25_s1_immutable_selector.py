from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.modeling.v25_s1_immutable_selector import (
    ImmutableBankSingleEditSelector,
    immutable_selector_loss,
)
from dynlaneseq_eg.tools.train import seed_everything


FEATURE_NAMES = (
    "row_evidence",
    "row_geometry",
    "global_evidence",
    "global_geometry",
    "action_valid",
)
DIAGNOSTIC_NAMES = (
    "action_tp50",
    "action_tp75",
    "action_lost50",
    "action_lost75",
    "source_tp50",
    "source_tp75",
    "source_prediction_count",
    "gt_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fixed-endpoint V25-S1 hard selector.")
    parser.add_argument("--train-cache", action="append", required=True)
    parser.add_argument("--eval-cache", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", choices=("treatment", "geometry_control"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


class ImmutableCacheDataset(Dataset):
    def __init__(self, roots: list[str | Path]) -> None:
        self.roots = [Path(root).expanduser().resolve() for root in roots]
        self.parts: list[dict[str, np.ndarray]] = []
        self.ends: list[int] = []
        total = 0
        for root in self.roots:
            report = root / "bank_report.json"
            if not report.is_file():
                raise FileNotFoundError(report)
            arrays = {
                name: np.load(root / f"{name}.npy", mmap_mode="r")
                for name in (*FEATURE_NAMES, *DIAGNOSTIC_NAMES, "action_outcome", "target_action")
            }
            length = int(arrays["target_action"].shape[0])
            if any(int(value.shape[0]) != length for value in arrays.values()):
                raise ValueError(f"cache tensor population mismatch in {root}")
            self.parts.append(arrays)
            total += length
            self.ends.append(total)

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int):
        part = bisect_right(self.ends, int(index))
        start = 0 if part == 0 else self.ends[part - 1]
        local = int(index) - start
        arrays = self.parts[part]
        result: dict[str, torch.Tensor] = {}
        for name in FEATURE_NAMES:
            value = arrays[name][local]
            # DataLoader collates these views into writable pinned tensors; no
            # mutation is performed on the mmap itself.
            result[name] = torch.from_numpy(np.asarray(value))
        result["action_outcome"] = torch.from_numpy(np.asarray(arrays["action_outcome"][local]))
        for name in DIAGNOSTIC_NAMES:
            result[name] = torch.from_numpy(np.asarray(arrays[name][local]))
        result["target_action"] = torch.as_tensor(int(arrays["target_action"][local]), dtype=torch.long)
        return result


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.inference_mode()
def evaluate(
    model: ImmutableBankSingleEditSelector,
    loader: DataLoader,
    *,
    device: torch.device,
    evidence_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    count = edits = target_edits = correct = selected_beneficial = selected_harmful = 0
    available_beneficial = 0
    loss_sum = 0.0
    source_tp50 = source_tp75 = selected_tp50 = selected_tp75 = 0
    prediction_count = gt_count = source_lost50 = source_lost75 = 0
    for batch in loader:
        batch = _move(batch, device)
        outputs = model(batch, evidence_enabled=evidence_enabled)
        loss, _ = immutable_selector_loss(
            outputs,
            target_action=batch["target_action"],
            action_outcome=batch["action_outcome"],
            action_valid=batch["action_valid"],
        )
        selected = outputs["selected_action"]
        local_count = int(selected.numel())
        count += local_count
        loss_sum += float(loss.item()) * local_count
        edits += int((selected > 0).sum().item())
        target_edits += int((batch["target_action"] > 0).sum().item())
        correct += int((selected == batch["target_action"]).sum().item())
        available_beneficial += int((batch["action_outcome"] == 1).any(dim=1).sum().item())
        edit_index = (selected - 1).clamp_min(0)
        outcome = batch["action_outcome"].gather(1, edit_index.unsqueeze(1)).squeeze(1)
        selected_beneficial += int(((selected > 0) & (outcome == 1)).sum().item())
        selected_harmful += int(((selected > 0) & (outcome == 2)).sum().item())
        source_tp50 += int(batch["source_tp50"].sum().item())
        source_tp75 += int(batch["source_tp75"].sum().item())
        prediction_count += int(batch["source_prediction_count"].sum().item())
        gt_count += int(batch["gt_count"].sum().item())
        chosen50 = batch["source_tp50"].clone()
        chosen75 = batch["source_tp75"].clone()
        chosen_lost50 = torch.zeros_like(chosen50)
        chosen_lost75 = torch.zeros_like(chosen75)
        edited_rows = torch.nonzero(selected > 0, as_tuple=False).flatten()
        if int(edited_rows.numel()) > 0:
            columns = selected[edited_rows] - 1
            chosen50[edited_rows] = batch["action_tp50"][edited_rows, columns]
            chosen75[edited_rows] = batch["action_tp75"][edited_rows, columns]
            chosen_lost50[edited_rows] = batch["action_lost50"][edited_rows, columns]
            chosen_lost75[edited_rows] = batch["action_lost75"][edited_rows, columns]
        selected_tp50 += int(chosen50.sum().item())
        selected_tp75 += int(chosen75.sum().item())
        source_lost50 += int(chosen_lost50.sum().item())
        source_lost75 += int(chosen_lost75.sum().item())
    decisive = selected_beneficial + selected_harmful
    denominator = max(prediction_count + gt_count, 1)
    source_f1_50 = 2.0 * source_tp50 / denominator
    source_f1_75 = 2.0 * source_tp75 / denominator
    selected_f1_50 = 2.0 * selected_tp50 / denominator
    selected_f1_75 = 2.0 * selected_tp75 / denominator
    return {
        "images": count,
        "loss": loss_sum / max(count, 1),
        "action_accuracy": correct / max(count, 1),
        "selected_edit_fraction": edits / max(count, 1),
        "target_edit_fraction": target_edits / max(count, 1),
        "selected_beneficial": selected_beneficial,
        "selected_harmful": selected_harmful,
        "decisive_precision": selected_beneficial / max(decisive, 1),
        "beneficial_image_recall": selected_beneficial / max(available_beneficial, 1),
        "source_f1_50": source_f1_50,
        "source_f1_75": source_f1_75,
        "selected_f1_50": selected_f1_50,
        "selected_f1_75": selected_f1_75,
        "gain_f1_50_points": 100.0 * (selected_f1_50 - source_f1_50),
        "gain_f1_75_points": 100.0 * (selected_f1_75 - source_f1_75),
        "source_tp50": source_tp50,
        "source_tp75": source_tp75,
        "selected_tp50": selected_tp50,
        "selected_tp75": selected_tp75,
        "source_correct_tp50_lost": source_lost50,
        "source_correct_tp75_lost": source_lost75,
    }


def main() -> None:
    args = parse_args()
    seed_everything(3407)
    device = torch.device(args.device)
    treatment = args.arm == "treatment"
    train_dataset = ImmutableCacheDataset(args.train_cache)
    eval_dataset = ImmutableCacheDataset(args.eval_cache) if args.eval_cache else None
    generator = torch.Generator().manual_seed(3407)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        if eval_dataset is not None
        else None
    )
    model = ImmutableBankSingleEditSelector(hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(len(train_loader) * int(args.epochs), 1)
    step = 0
    started = time.perf_counter()
    for epoch in range(int(args.epochs)):
        model.train()
        for batch in train_loader:
            step += 1
            progress = float(step) / float(total_steps)
            learning_rate = float(args.learning_rate) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, evidence_enabled=treatment)
            loss, diagnostics = immutable_selector_loss(
                outputs,
                target_action=batch["target_action"],
                action_outcome=batch["action_outcome"],
                action_valid=batch["action_valid"],
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("V25-S1 selector loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if step == 1 or (args.log_interval > 0 and step % args.log_interval == 0):
                elapsed = max(time.perf_counter() - started, 1.0e-9)
                print(
                    json.dumps(
                        {
                            "phase": "train_v25_s1_selector",
                            "arm": args.arm,
                            "epoch": epoch + 1,
                            "step": step,
                            "images_per_second": step * args.batch_size / elapsed,
                            "learning_rate": learning_rate,
                            "gradient_norm": float(gradient_norm),
                            **{name: float(value.detach()) for name, value in diagnostics.items()},
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_metrics = evaluate(model, train_eval_loader, device=device, evidence_enabled=treatment)
    eval_metrics = (
        evaluate(model, eval_loader, device=device, evidence_enabled=treatment)
        if eval_loader is not None
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = output_dir / "selector_endpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "arm": args.arm,
            "epochs": int(args.epochs),
            "seed": 3407,
            "training_cache_roots": [str(root) for root in train_dataset.roots],
        },
        endpoint,
    )
    report = {
        "experiment": "V25-S1 fixed-endpoint immutable-bank selector",
        "arm": args.arm,
        "endpoint": str(endpoint),
        "endpoint_sha256": sha256_file(endpoint),
        "epochs": int(args.epochs),
        "steps": step,
        "train_population": len(train_dataset),
        "eval_population": 0 if eval_dataset is None else len(eval_dataset),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "contract": {
            "fixed_endpoint": True,
            "checkpoint_selection": False,
            "threshold_selection": False,
            "hard_categorical_output": True,
            "coordinate_blending": False,
            "test_set_used": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "selector_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(report_path), "eval": eval_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
