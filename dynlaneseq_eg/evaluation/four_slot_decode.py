from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


@torch.no_grad()
def decode_four_slot_logits(
    logits: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Decode slots with globally unique proposals and private dustbins.

    ``logits`` has one shared dustbin class in the neural output.  For the
    discrete assignment it is expanded to one private dustbin per slot, so
    any number of slots may be empty while each real proposal can be owned by
    at most one slot.
    """

    squeeze = False
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
        candidate_valid = candidate_valid.unsqueeze(0)
        squeeze = True
    if logits.ndim != 3:
        raise ValueError("four-slot logits must have shape [B,S,N+1]")
    batch, slots, classes = logits.shape
    candidates = int(classes) - 1
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("four-slot candidate_valid must have shape [B,N]")

    logits_cpu = logits.detach().float().cpu()
    valid_cpu = candidate_valid.detach().bool().cpu()
    probability_cpu = torch.softmax(logits_cpu, dim=-1)
    selected = torch.full((batch, slots), -1, dtype=torch.long)
    scores = torch.zeros((batch, slots), dtype=torch.float32)
    repair_count = torch.zeros((batch,), dtype=torch.long)
    raw_collision_count = torch.zeros((batch,), dtype=torch.long)

    for batch_index in range(int(batch)):
        row_logits = logits_cpu[batch_index]
        matrix = np.full(
            (int(slots), candidates + int(slots)),
            -1.0e9,
            dtype=np.float64,
        )
        matrix[:, :candidates] = row_logits[:, :candidates].numpy()
        invalid = torch.nonzero(
            ~valid_cpu[batch_index], as_tuple=False
        ).flatten()
        if invalid.numel():
            matrix[:, invalid.numpy()] = -1.0e9
        for slot_index in range(int(slots)):
            matrix[slot_index, candidates + slot_index] = float(
                row_logits[slot_index, candidates]
            )

        row_ids, column_ids = linear_sum_assignment(-matrix)
        assigned_columns = [-1] * int(slots)
        for slot_index, column_index in zip(
            row_ids.tolist(), column_ids.tolist()
        ):
            assigned_columns[int(slot_index)] = int(column_index)

        raw_class = row_logits.argmax(dim=-1)
        raw_candidates = torch.where(
            raw_class < candidates,
            raw_class,
            raw_class.new_full(raw_class.shape, -1),
        )
        active_raw = raw_candidates[raw_candidates >= 0]
        raw_collision_count[batch_index] = int(active_raw.numel()) - int(
            active_raw.unique().numel()
        )

        for slot_index, column_index in enumerate(assigned_columns):
            assigned_candidate = (
                int(column_index) if int(column_index) < candidates else -1
            )
            selected[batch_index, slot_index] = assigned_candidate
            if assigned_candidate >= 0:
                scores[batch_index, slot_index] = probability_cpu[
                    batch_index, slot_index, assigned_candidate
                ]
            else:
                scores[batch_index, slot_index] = probability_cpu[
                    batch_index, slot_index, candidates
                ]
            if assigned_candidate != int(raw_candidates[slot_index]):
                repair_count[batch_index] += 1

    result = {
        "indices": selected,
        "scores": scores,
        "raw_collision_count": raw_collision_count,
        "repair_count": repair_count,
    }
    if squeeze:
        return {name: value[0] for name, value in result.items()}
    return result

