from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .dynlaneseq_s0 import DynLaneSeqS0
from .v19_counterfactual_fidelity import frozen_v7_counterfactual_anchors
from .v23_ordered_slot_cost_volume import canonicalize_v7_slots
from .v28_refined_belief_router import (
    V28RefinedBeliefRouter,
    decode_v28_unique_routes,
    gather_canonical_slots,
    gather_slot_candidates,
    restore_public_slots,
)


def _required(outputs: dict[str, Any], name: str) -> torch.Tensor:
    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"V28 requires frozen V7 output {name!r}")
    return value


class DynLaneSeqV28(nn.Module):
    """Frozen V7 support bank plus a trainable selection-only belief tower.

    The teacher remains the sole owner of proposal geometry, counterfactual
    refinement, lane activity, and visible range.  The student can only rank
    the resulting immutable ``4 x 32`` final-ready curves.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.get("model", cfg)
        v28_cfg = cfg.get("v28", {})
        self.teacher = DynLaneSeqS0(cfg)
        self.router = V28RefinedBeliefRouter(
            input_h=int(model_cfg.get("input_h", 640)),
            input_w=int(model_cfg.get("input_w", 1600)),
            num_rows=int(model_cfg.get("num_rows", 160)),
            x_bins=int(model_cfg.get("x_bins", 800)),
            fpn_channels=int(model_cfg.get("fpn_channels", 256)),
            hidden_dim=int(v28_cfg.get("hidden_dim", 96)),
            query_dim=int(v28_cfg.get("query_dim", 96)),
            vertical_layers=int(v28_cfg.get("vertical_layers", 2)),
            num_heads=int(v28_cfg.get("num_heads", 4)),
            ff_dim=int(v28_cfg.get("ff_dim", 256)),
            source_prior_sigma_px=float(
                v28_cfg.get("source_prior_sigma_px", 64.0)
            ),
            source_prior_weight=float(
                v28_cfg.get("source_prior_weight", 0.25)
            ),
            freeze_batch_norm_stats=bool(
                v28_cfg.get("freeze_batch_norm_stats", True)
            ),
        )
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        selection_head = self.teacher.structured_query_head.set_selection_head
        self._teacher_refiner = selection_head.slot_refinement
        if self._teacher_refiner is None:
            raise RuntimeError("V28 requires the frozen V7 slot refiner")
        self._captured_refiner_inputs: dict[str, torch.Tensor] = {}
        # The hook only observes the already-frozen V7 call. It neither alters
        # teacher execution nor creates a second proposal path.
        self._refiner_hook = self._teacher_refiner.register_forward_pre_hook(
            self._capture_refiner_inputs,
            with_kwargs=True,
        )
        self.supports_inference_only = True

    def _capture_refiner_inputs(self, _module, _args, kwargs) -> None:
        self._captured_refiner_inputs.clear()
        self._captured_refiner_inputs.update(
            {
                key: value
                for key, value in kwargs.items()
                if isinstance(value, torch.Tensor)
            }
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def prepare_for_inference(self) -> None:
        self.teacher.prepare_for_inference()

    @torch.no_grad()
    def _frozen_bank(
        self,
        images: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._captured_refiner_inputs.clear()
        with torch.autocast(device_type=images.device.type, enabled=False):
            teacher_outputs = self.teacher(images.float(), inference_only=True)
        required_inputs = {
            "slot_states",
            "proposal_row_tokens",
            "proposal_x_rows",
            "proposal_range_norm",
            "candidate_valid",
            "row_value_features",
        }
        missing = required_inputs - set(self._captured_refiner_inputs)
        if missing:
            raise RuntimeError(
                f"V28 refiner hook missed frozen inputs: {sorted(missing)}"
            )
        captured = self._captured_refiner_inputs
        with torch.autocast(device_type=images.device.type, enabled=False):
            counterfactual = frozen_v7_counterfactual_anchors(
                self._teacher_refiner,
                slot_states=captured["slot_states"].float(),
                proposal_row_tokens=captured["proposal_row_tokens"].float(),
                proposal_x_rows=captured["proposal_x_rows"].float(),
                proposal_range_norm=captured["proposal_range_norm"].float(),
                candidate_valid=captured["candidate_valid"],
                row_value_features=captured["row_value_features"].float(),
            )
        # The counterfactual refiner invocation triggers the same observer
        # with an expanded private slot axis. None of those teacher tensors is
        # needed after the immutable bank has been materialised.
        self._captured_refiner_inputs.clear()
        source = canonicalize_v7_slots(teacher_outputs)
        order = source["source_slot_indices"]
        source_routes = gather_canonical_slots(
            _required(
                teacher_outputs, "selection_slot_geometry_route_indices"
            ).long(),
            order,
        )
        candidate_x = gather_canonical_slots(
            counterfactual["x_rows"], order
        ).detach().float().clone()
        candidate_range = gather_canonical_slots(
            counterfactual["range_norm"], order
        ).detach().float().clone()
        candidate_valid = gather_canonical_slots(
            counterfactual["valid"], order
        ).detach().bool().clone()
        # A second frozen-refiner call can differ from the already-deployed
        # source by a few thousandths of a pixel due to kernel execution
        # order. Replace precisely the source proposal entry with V7's first
        # forward tensors. Thus "belief chose the V7 ID" is byte-identical to
        # V7, while all 31 alternatives remain counterfactual-refined.
        source_x = source["x_rows"].detach().float().clone()
        source_range = source["range_norm"].detach().float().clone()
        source_active = source["active"].detach().bool().clone()
        batch_ids = torch.arange(
            source_routes.shape[0], device=source_routes.device
        ).view(-1, 1).expand_as(source_routes)
        slot_ids = torch.arange(
            source_routes.shape[1], device=source_routes.device
        ).view(1, -1).expand_as(source_routes)
        route_ids = source_routes.clamp_min(0)
        candidate_x[batch_ids, slot_ids, route_ids] = source_x
        candidate_range[batch_ids, slot_ids, route_ids] = source_range
        candidate_valid[batch_ids, slot_ids, route_ids] = True
        bank = {
            # Clone inference tensors before the trainable router saves them
            # for backward. Geometry remains detached and immutable.
            "candidate_x": candidate_x,
            "candidate_range": candidate_range,
            "candidate_valid": candidate_valid,
            "source_x": source_x,
            "source_range": source_range,
            "source_active": source_active,
            "source_route": source_routes.detach().long().clone(),
            "source_slot_indices": order.detach().long().clone(),
        }
        return teacher_outputs, bank

    @staticmethod
    def _public_output(
        *,
        selected_x: torch.Tensor,
        selected_range: torch.Tensor,
        source_active: torch.Tensor,
        source_slot_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        public_x = restore_public_slots(selected_x, source_slot_indices)
        public_range = restore_public_slots(
            selected_range, source_slot_indices
        )
        public_active = restore_public_slots(
            source_active, source_slot_indices
        )
        positive = torch.full_like(public_x[..., 0], 20.0)
        negative = torch.full_like(public_x[..., 0], -20.0)
        exist_logits = torch.stack(
            (
                torch.where(public_active, positive, negative),
                torch.where(public_active, negative, positive),
            ),
            dim=-1,
        )
        return {
            "exist_logits": exist_logits,
            "pred_x_rows": public_x,
            "range_norm": public_range,
            # The first V28 gate deliberately inherits V7 count/activity and
            # never uses a learned quality threshold at the writer boundary.
            "quality_logits": torch.zeros_like(public_x[..., 0]),
        }

    def forward(
        self,
        images: torch.Tensor,
        targets=None,
        return_features: bool = False,
        inference_only: bool = False,
        deployment_policy: str = "belief",
    ) -> dict[str, torch.Tensor]:
        del targets, return_features
        teacher_outputs, bank = self._frozen_bank(images)
        router_output = self.router(
            images,
            source_x=bank["source_x"],
            source_range=bank["source_range"],
            source_active=bank["source_active"],
            candidate_x=bank["candidate_x"],
            candidate_range=bank["candidate_range"],
            candidate_valid=bank["candidate_valid"],
        )
        policy = str(deployment_policy).strip().lower()
        if policy == "source":
            routes = bank["source_route"]
            # Gate-zero/source replay uses the deployed V7 tensors directly,
            # rather than relying on numerical parity of a second refiner
            # invocation. This makes writer parity bit-exact by construction.
            selected_x = bank["source_x"]
            selected_range = bank["source_range"]
        elif policy == "belief":
            routes = decode_v28_unique_routes(
                router_output["candidate_scores"],
                router_output["candidate_valid"],
            )
            selected_x = gather_slot_candidates(bank["candidate_x"], routes)
            selected_range = gather_slot_candidates(
                bank["candidate_range"], routes
            )
        else:
            raise ValueError("V28 deployment_policy must be source or belief")
        public = self._public_output(
            selected_x=selected_x,
            selected_range=selected_range,
            source_active=bank["source_active"],
            source_slot_indices=bank["source_slot_indices"],
        )
        if inference_only:
            return public
        return {
            **public,
            **router_output,
            **bank,
            "selected_route": routes,
            "selected_x_rows": selected_x,
            "selected_range_norm": selected_range,
            "teacher_source_x_rows": _required(
                teacher_outputs, "selection_slot_pred_x_rows"
            ),
            "teacher_source_range_norm": _required(
                teacher_outputs, "selection_slot_range_norm"
            ),
        }
