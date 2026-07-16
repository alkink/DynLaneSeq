from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class TuSimpleLaneEval:
    """Reference-compatible implementation of the official TuSimple metric."""

    pixel_thresh = 20.0
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs: np.ndarray, y_samples: np.ndarray) -> float:
        valid = xs >= 0
        xs_valid = xs[valid].astype(np.float64, copy=False)
        ys_valid = y_samples[valid].astype(np.float64, copy=False)
        if xs_valid.size <= 1:
            return 0.0
        y_centered = ys_valid - ys_valid.mean()
        denominator = float(np.dot(y_centered, y_centered))
        if denominator <= 0.0:
            return 0.0
        slope = float(np.dot(y_centered, xs_valid - xs_valid.mean()) / denominator)
        return float(np.arctan(slope))

    @staticmethod
    def line_accuracy(pred: list[float], gt: list[float], threshold: float) -> float:
        pred_array = np.asarray([p if p >= 0 else -100 for p in pred], dtype=np.float64)
        gt_array = np.asarray([g if g >= 0 else -100 for g in gt], dtype=np.float64)
        return float(np.mean(np.abs(pred_array - gt_array) < threshold))

    @classmethod
    def bench(
        cls,
        pred: list[list[float]],
        gt: list[list[float]],
        y_samples: list[int],
        running_time: float,
    ) -> tuple[float, float, float]:
        if any(len(lane) != len(y_samples) for lane in pred):
            raise ValueError("Every predicted lane must have one x value per h_sample")
        if running_time > 200.0 or len(pred) > len(gt) + 2:
            return 0.0, 0.0, 1.0

        y_array = np.asarray(y_samples, dtype=np.float64)
        angles = [cls.get_angle(np.asarray(lane), y_array) for lane in gt]
        thresholds = [cls.pixel_thresh / np.cos(angle) for angle in angles]
        line_accuracies: list[float] = []
        false_negative = 0.0
        matched = 0
        for gt_lane, threshold in zip(gt, thresholds):
            accuracies = [cls.line_accuracy(pred_lane, gt_lane, threshold) for pred_lane in pred]
            max_accuracy = max(accuracies) if accuracies else 0.0
            if max_accuracy < cls.pt_thresh:
                false_negative += 1.0
            else:
                matched += 1
            line_accuracies.append(max_accuracy)

        false_positive = float(len(pred) - matched)
        if len(gt) > 4 and false_negative > 0:
            false_negative -= 1.0
        accuracy_sum = float(sum(line_accuracies))
        if len(gt) > 4 and line_accuracies:
            accuracy_sum -= min(line_accuracies)
        accuracy = accuracy_sum / max(min(4.0, float(len(gt))), 1.0)
        fp_rate = false_positive / float(len(pred)) if pred else 0.0
        fn_rate = false_negative / max(min(float(len(gt)), 4.0), 1.0)
        return accuracy, fp_rate, fn_rate

    @classmethod
    def evaluate(cls, prediction_file: str | Path, ground_truth_file: str | Path) -> dict[str, float | int]:
        predictions = cls._read_json_lines(prediction_file)
        ground_truth = cls._read_json_lines(ground_truth_file)
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Expected {len(ground_truth)} predictions, received {len(predictions)}"
            )
        gt_by_file = {str(item["raw_file"]): item for item in ground_truth}
        if len(gt_by_file) != len(ground_truth):
            raise ValueError("Ground-truth file contains duplicate raw_file entries")

        accuracy_sum = 0.0
        fp_sum = 0.0
        fn_sum = 0.0
        seen: set[str] = set()
        for prediction in predictions:
            for required in ("raw_file", "lanes", "run_time"):
                if required not in prediction:
                    raise ValueError(f"Prediction is missing required field {required!r}")
            raw_file = str(prediction["raw_file"])
            if raw_file in seen:
                raise ValueError(f"Duplicate prediction for raw_file={raw_file!r}")
            if raw_file not in gt_by_file:
                raise ValueError(f"Prediction raw_file not present in ground truth: {raw_file}")
            seen.add(raw_file)
            gt = gt_by_file[raw_file]
            accuracy, fp, fn = cls.bench(
                prediction["lanes"],
                gt["lanes"],
                gt["h_samples"],
                float(prediction["run_time"]),
            )
            accuracy_sum += accuracy
            fp_sum += fp
            fn_sum += fn

        count = len(ground_truth)
        accuracy = accuracy_sum / max(count, 1)
        fp = fp_sum / max(count, 1)
        fn = fn_sum / max(count, 1)
        true_positive_proxy = 1.0 - fp
        precision = true_positive_proxy / max(true_positive_proxy + fp, 1e-12)
        recall = true_positive_proxy / max(true_positive_proxy + fn, 1e-12)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        return {
            "Accuracy": float(accuracy),
            "F1_score": float(f1),
            "FP": float(fp),
            "FN": float(fn),
            "num_samples": int(count),
        }

    @staticmethod
    def _read_json_lines(path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def format_tusimple_results(results: dict[str, float | int]) -> str:
    return (
        f"Accuracy={float(results['Accuracy']):.6f} "
        f"F1={float(results['F1_score']):.6f} "
        f"FP={float(results['FP']):.6f} "
        f"FN={float(results['FN']):.6f} "
        f"samples={int(results['num_samples'])}"
    )
