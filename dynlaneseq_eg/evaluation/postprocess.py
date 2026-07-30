from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm


def _lane_mean_abs_distance(lane_a: list[tuple[float, float]], lane_b: list[tuple[float, float]]) -> tuple[float, int]:
    xa_by_y = {round(y, 4): x for x, y in lane_a}
    diffs = []
    for x_b, y_b in lane_b:
        x_a = xa_by_y.get(round(y_b, 4))
        if x_a is not None:
            diffs.append(abs(x_a - x_b))
    if not diffs:
        return float("inf"), 0
    return float(sum(diffs) / len(diffs)), len(diffs)


def lane_nms(
    candidates: list[tuple[float, list[tuple[float, float]]]],
    distance_thresh_px: float = 20.0,
    min_overlap_points: int = 5,
    top_k: int = 0,
) -> list[list[tuple[float, float]]]:
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    if distance_thresh_px <= 0 or len(candidates) <= 1:
        lanes = [lane for _, lane in candidates]
        return lanes[:top_k] if top_k > 0 else lanes
    kept: list[tuple[float, list[tuple[float, float]]]] = []
    for score, lane in candidates:
        suppress = False
        for _, kept_lane in kept:
            distance, overlap = _lane_mean_abs_distance(lane, kept_lane)
            if overlap >= min_overlap_points and distance < distance_thresh_px:
                suppress = True
                break
        if not suppress:
            kept.append((score, lane))
            if top_k > 0 and len(kept) >= top_k:
                break
    return [lane for _, lane in kept]


@torch.no_grad()
def predictions_to_lanes(
    outputs: dict[str, torch.Tensor],
    score_thresh: float = 0.5,
    min_pred_points: int = 5,
    input_w: int = 800,
    input_h: int = 288,
    nms_distance_thresh_px: float = 0.0,
    nms_min_overlap_points: int = 5,
    top_k: int = 0,
    row_visibility_thresh: float = 0.0,
    quality_score_power: float = 0.0,
    score_mode: str = "exist_quality",
) -> list[list[list[tuple[float, float]]]]:
    if "stage2" in outputs:
        outputs = outputs["stage2"]
    elif "final" in outputs:
        outputs = outputs["final"]
    exist_score = torch.softmax(outputs["exist_logits"], dim=-1)[..., 0]
    quality = (
        torch.sigmoid(outputs["quality_logits"]).clamp(min=1e-6)
        if "quality_logits" in outputs
        else None
    )
    score_mode = str(score_mode).strip().lower()
    if score_mode in {"exist", "existence"}:
        p_lane = exist_score
    elif score_mode in {"quality", "iou"}:
        if quality is None:
            raise ValueError("postprocess score_mode='quality' requires quality_logits")
        quality_power = float(quality_score_power) if quality_score_power > 0 else 1.0
        p_lane = quality.pow(quality_power)
    elif score_mode in {"exist_quality", "product"}:
        p_lane = exist_score
        if quality_score_power > 0 and quality is not None:
            p_lane = p_lane * quality.pow(float(quality_score_power))
    elif score_mode in {"selection", "set_selection", "set"}:
        selection_logits = outputs.get("selection_logits")
        if selection_logits is None:
            raise ValueError(
                "postprocess score_mode='selection' requires selection_logits"
            )
        p_lane = torch.sigmoid(selection_logits.float())
    else:
        raise ValueError(f"Unsupported postprocess score_mode: {score_mode!r}")
    pred_x = outputs["pred_x_rows"].clamp(0, input_w - 1)
    row_visibility = None
    if row_visibility_thresh > 0 and "row_visibility_logits" in outputs:
        row_visibility = torch.sigmoid(outputs["row_visibility_logits"]) >= float(row_visibility_thresh)
    ranges = sort_range_norm(outputs["range_norm"])
    y_rows = fixed_y_rows(pred_x.shape[-1], input_h, device=pred_x.device, dtype=pred_x.dtype)
    # Transfer each result tensor once per batch.  The previous per-sample
    # .cpu() calls introduced four CUDA synchronizations for every image.
    p_lane_cpu = p_lane.detach().cpu()
    pred_x_cpu = pred_x.detach().cpu()
    ranges_cpu = ranges.detach().cpu()
    y_rows_cpu = y_rows.detach().cpu()
    row_visibility_cpu = row_visibility.detach().cpu() if row_visibility is not None else None
    batch_lanes: list[list[list[tuple[float, float]]]] = []
    for b in range(pred_x.shape[0]):
        p_lane_b = p_lane_cpu[b]
        pred_x_b = pred_x_cpu[b]
        ranges_b = ranges_cpu[b]
        row_visibility_b = row_visibility_cpu[b] if row_visibility_cpu is not None else None
        candidates: list[tuple[float, list[tuple[float, float]]]] = []
        for n in range(pred_x.shape[1]):
            score = float(p_lane_b[n])
            if score < score_thresh:
                continue
            y_min = float(ranges_b[n, 0] * input_h)
            y_max = float(ranges_b[n, 1] * input_h)
            mask = (y_rows_cpu >= y_min) & (y_rows_cpu <= y_max)
            if row_visibility_b is not None:
                mask = mask & row_visibility_b[n]
            if int(mask.sum().item()) < min_pred_points:
                continue
            lane = [(float(x), float(y)) for x, y in zip(pred_x_b[n, mask], y_rows_cpu[mask])]
            candidates.append((score, lane))
        batch_lanes.append(
            lane_nms(
                candidates,
                distance_thresh_px=nms_distance_thresh_px,
                min_overlap_points=nms_min_overlap_points,
                top_k=top_k,
            )
        )
    return batch_lanes
