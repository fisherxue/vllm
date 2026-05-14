# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable
import json
import os

import numpy as np
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if current_platform.is_cuda_alike():

    @triton.jit
    def _eplb_map_and_record_i32_kernel(
        topk_ids_ptr,
        logical_replica_count_ptr,
        logical_to_physical_ptr,
        out_ids_ptr,
        out_ptr,
        record_enabled_ptr,
        num_logical_experts,
        map_slots,
        out_size,
        numel,
        num_active_experts,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int64)
        valid_expert = (expert_id >= 0) & (expert_id < num_logical_experts)
        safe_expert_id = tl.where(valid_expert, expert_id, 0)

        # 1. Convert the logical expert ids to physical expert ids
        replica_count = tl.load(
            logical_replica_count_ptr + safe_expert_id,
            mask=mask & valid_expert,
            other=1,
        )
        # Avoid invalid modulo/div by forcing at least 1.
        replica_count = tl.maximum(replica_count, 1)
        # floor(2^32 / phi), classic Knuth multiplicative hash multiplier.
        KNUTH_MULTIPLIER = 2654435769
        token_idx = (offs // num_active_experts).to(tl.int64)
        hashed = (token_idx * KNUTH_MULTIPLIER) & 0xFFFFFFFF
        replica_idx = hashed % replica_count

        # 2. Record expert load metrics.

        # TODO(bowen): When using `FusedMoEModularKernel`, this
        # can be done in a more unified way, since
        # `FusedMoEPrepareAndFinalize` will return the expert
        # token count, in some cases directly from the kernel.
        # However, now there are many code paths not using
        # the modular kernel, e.g. calling `fused_experts`,
        # so we decide to keep the logic here.
        #
        # If later refactor moved all the MoE kernel calls
        # to the modular kernel, we can move this logic there
        # to achieve better efficiency.
        map_index = safe_expert_id * map_slots + replica_idx
        physical_id = tl.load(
            logical_to_physical_ptr + map_index,
            mask=mask & valid_expert,
            other=-1,
        )
        tl.store(out_ids_ptr + offs, physical_id, mask=mask)

        record_enabled = tl.load(record_enabled_ptr) != 0
        valid = mask & record_enabled & (physical_id >= 0) & (physical_id < out_size)
        safe_physical_id = tl.where(physical_id >= 0, physical_id, 0)
        tl.atomic_add(out_ptr + safe_physical_id, 1, mask=valid)

    def _eplb_map_and_record_triton(
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        expert_load_view: torch.Tensor,
        record_enabled: torch.Tensor,
    ) -> torch.Tensor:
        topk_ids_in = topk_ids.contiguous().to(dtype=torch.int32)
        numel = topk_ids_in.numel()
        if numel == 0:
            return topk_ids
        num_active_experts = topk_ids_in.shape[-1]
        out_flat = torch.empty((numel,), device=topk_ids.device, dtype=topk_ids.dtype)
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        assert expert_load_view.is_contiguous()
        _eplb_map_and_record_i32_kernel[grid](
            topk_ids_in,
            logical_replica_count.contiguous(),
            logical_to_physical_map.contiguous(),
            out_flat,
            expert_load_view,
            record_enabled,
            logical_replica_count.shape[0],
            logical_to_physical_map.shape[1],
            expert_load_view.shape[0],
            numel,
            num_active_experts,
            BLOCK_SIZE=256,
        )
        return out_flat.reshape(topk_ids.shape)

    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_enabled: torch.Tensor,
    ) -> torch.Tensor:
        # Fused triton implementation: mapping + optional recording in one kernel.
        return _eplb_map_and_record_triton(
            topk_ids=topk_ids,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_view=expert_load_view,
            record_enabled=record_enabled,
        )
else:

    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_enabled: torch.Tensor,
    ) -> torch.Tensor:
        return topk_ids


_override_config_cache: dict[str, dict] = {}
_override_layer_counter: int = 0
_capture_buffers: dict[int, dict] = {}
_capture_output_dir: str | None = None
_capture_flush_interval: int = 4096


def _flush_capture_buffers() -> None:
    """Append accumulated capture buffers to per-layer binary files.

    Each worker appends to two files per layer:
      {output_dir}/L{lid:03d}_ids_w{pid}.bin   — int16, width K
      {output_dir}/L{lid:03d}_logits_w{pid}.bin — float16, width E

    This produces O(layers × workers) files instead of O(tokens × layers).
    """
    if not _capture_output_dir or not _capture_buffers:
        return
    os.makedirs(_capture_output_dir, exist_ok=True)
    pid = os.getpid()
    for lid, data in _capture_buffers.items():
        if not data["ids"]:
            continue
        ids = np.concatenate(data["ids"], axis=0)
        logits = np.concatenate(data["logits"], axis=0)
        with open(os.path.join(
            _capture_output_dir, f"L{lid:03d}_ids_w{pid}.bin"
        ), "ab") as f:
            f.write(ids.tobytes())
        with open(os.path.join(
            _capture_output_dir, f"L{lid:03d}_logits_w{pid}.bin"
        ), "ab") as f:
            f.write(logits.tobytes())
        data["ids"].clear()
        data["logits"].clear()


import atexit
atexit.register(_flush_capture_buffers)


def _build_override_for_layer(cfg_path: str, layer_id: int) -> Callable:
    """Build a per-layer override function from a JSON config file.

    Supports policies:
    - "capture": pass-through, saves expert IDs + logits to disk
    - "two_tier": keep top-1, substitute from hot set
    - "prune": substitute all slots from hot set
    """
    global _capture_output_dir

    if cfg_path not in _override_config_cache:
        with open(cfg_path) as f:
            cfg = json.load(f)
        policy = cfg.get("policy", "two_tier")
        _override_config_cache[cfg_path] = {"policy": policy, "cfg": cfg}

        if policy == "capture":
            _capture_output_dir = cfg.get("output_dir", "/tmp/moe_capture")
            os.makedirs(_capture_output_dir, exist_ok=True)

        if policy != "capture":
            hot_mask = np.load(cfg["hot_mask_path"])
            _override_config_cache[cfg_path].update({
                "hot_tensor": torch.from_numpy(hot_mask).bool(),
                "per_layer": hot_mask.ndim == 2,
                "start_slot": 1 if policy == "two_tier" else 0,
                "normalize": cfg.get("normalize", "softmax"),
            })

    c = _override_config_cache[cfg_path]
    policy = c["policy"]

    # ── Capture-only mode: save data, return unchanged ──
    if policy == "capture":
        _lid = layer_id
        _capture_buffers[_lid] = {"ids": [], "logits": []}

        def capture_fn(topk_weights, topk_ids, router_logits):
            buf = _capture_buffers[_lid]
            buf["ids"].append(topk_ids.cpu().to(torch.int16).numpy())
            buf["logits"].append(router_logits.cpu().to(torch.float16).numpy())
            if sum(len(b["ids"]) for b in _capture_buffers.values()) >= _capture_flush_interval:
                _flush_capture_buffers()
            return topk_weights, topk_ids

        return capture_fn

    # ── Override modes: two_tier / prune ──
    hot_tensor = c["hot_tensor"]
    per_layer = c["per_layer"]
    start_slot = c["start_slot"]

    if per_layer:
        lid = min(layer_id, hot_tensor.shape[0] - 1)
        layer_mask = hot_tensor[lid]
    else:
        layer_mask = hot_tensor

    if c["normalize"] == "softmax":
        norm_fn = lambda x: torch.softmax(x, dim=-1)
    elif c["normalize"] == "sigmoid":
        norm_fn = torch.sigmoid
    else:
        raise ValueError(f"Unknown normalize={c['normalize']!r}")

    _gpu_mask: dict[str, torch.Tensor] = {}

    def override_fn(topk_weights, topk_ids, router_logits):
        dev_key = str(topk_ids.device)
        if dev_key not in _gpu_mask:
            _gpu_mask[dev_key] = layer_mask.to(topk_ids.device)
        hot = _gpu_mask[dev_key]

        T, K = topk_ids.shape
        new_ids = topk_ids.clone()
        new_weights = topk_weights.clone()
        scores = norm_fn(router_logits)
        arange_T = torch.arange(T, device=topk_ids.device)

        for slot in range(start_slot, K):
            expert_id = new_ids[:, slot]
            is_miss = ~hot[expert_id]
            if not is_miss.any():
                continue
            candidate = scores.clone()
            candidate[:, ~hot] = -1.0
            for s in range(K):
                if s == slot:
                    continue
                candidate[arange_T, new_ids[:, s]] = -1.0
            best = candidate.argmax(dim=1)
            best_w = scores[arange_T, best]
            new_ids[:, slot] = torch.where(is_miss, best, new_ids[:, slot])
            new_weights[:, slot] = torch.where(
                is_miss, best_w, new_weights[:, slot]
            )
        return new_weights, new_ids

    return override_fn


class BaseRouter(FusedMoERouter):
    """
    Base router class that provides common functionality for all router implementations.

    This class implements the template method pattern where select_experts() handles
    common pre-processing and post-processing, delegating the actual routing logic
    to the abstract _compute_routing() method.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        enable_eplb: bool = False,
        # TODO(bnell): Once the MK is constructed at layer init time, we
        # can make this a plain value instead of a callback.
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        """
        Note: the indices dtype might not be available at router construction
        time, so we need to supply a callback to get it at runtime.  This is
        because the indices type is supplied by modular kernels which are
        created after MoE layer/router construction.
        """
        super().__init__()
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.eplb_state = eplb_state
        self.enable_eplb = enable_eplb
        self.indices_type_getter = indices_type_getter
        self.capture_fn: Callable[[torch.Tensor], None] | None = None
        self._logits_capture_fn: Callable[[torch.Tensor], None] | None = None
        self._override_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ] | None = None
        self._override_checked = False

    def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None:
        """Set a capture callback for logical routed expert IDs."""
        self.capture_fn = capture_fn

    def set_logits_capture_fn(
        self, logits_capture_fn: Callable[[torch.Tensor], None] | None
    ) -> None:
        """Set a capture callback for full router logits (all experts)."""
        self._logits_capture_fn = logits_capture_fn

    def set_override_fn(
        self,
        override_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None,
    ) -> None:
        """Set a routing override function.

        override_fn(topk_weights, topk_ids, router_logits)
            -> (new_topk_weights, new_topk_ids)

        Called after capture (original routing is logged) but before
        EPLB mapping, so the model executes with modified routing while
        the capture buffer retains the original decisions.
        """
        self._override_fn = override_fn

    def _validate_eplb_state(self) -> None:
        """Validate that EPLB state is properly initialized if EPLB is enabled."""
        if self.enable_eplb:
            if self.eplb_state.expert_load_view is None:
                raise ValueError("enable_eplb=True requires expert_load_view != None")
            if self.eplb_state.logical_to_physical_map is None:
                raise ValueError(
                    "enable_eplb=True requires logical_to_physical_map != None"
                )
            if self.eplb_state.logical_replica_count is None:
                raise ValueError(
                    "enable_eplb=True requires logical_replica_count != None"
                )
            if self.eplb_state.should_record_tensor is None:
                raise ValueError(
                    "enable_eplb=True requires should_record_tensor != None"
                )

    def _get_indices_type(self) -> torch.dtype | None:
        """Get the desired indices dtype from the getter function."""
        return (
            self.indices_type_getter() if self.indices_type_getter is not None else None
        )

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Apply EPLB mapping to convert logical expert IDs to physical expert IDs."""
        if self.enable_eplb:
            assert self.eplb_state.expert_load_view is not None
            assert self.eplb_state.logical_to_physical_map is not None
            assert self.eplb_state.logical_replica_count is not None
            assert self.eplb_state.should_record_tensor is not None
            return eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                logical_to_physical_map=self.eplb_state.logical_to_physical_map,
                logical_replica_count=self.eplb_state.logical_replica_count,
                expert_load_view=self.eplb_state.expert_load_view,
                record_enabled=self.eplb_state.should_record_tensor,
            )
        return topk_ids

    def _convert_indices_dtype(
        self, topk_ids: torch.Tensor, indices_type: torch.dtype | None
    ) -> torch.Tensor:
        """Convert topk_ids to the desired dtype if needed."""
        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None
        return topk_ids

    @abstractmethod
    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the actual routing logic.

        This method must be implemented by subclasses to provide the specific
        routing algorithm (e.g., grouped_topk, fused_topk, custom routing, etc.).

        Args:
            hidden_states: Input hidden states
            router_logits: Router logits for expert selection
            indices_type: Desired dtype for expert indices (may be None)

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        raise NotImplementedError

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        This method implements the template method pattern:
        1. Validates EPLB state
        2. Gets indices type
        3. Calls _compute_routing() to get topk_weights and topk_ids
        4. Applies EPLB mapping if enabled
        5. Converts indices dtype if needed

        Returns:
            (topk_weights, topk_ids)
            (tuple[torch.Tensor, torch.Tensor]):
            The weights and expert ids computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        # Step 1: Validate EPLB state
        self._validate_eplb_state()

        # Capture full router logits before top-k / softmax.
        if self._logits_capture_fn is not None:
            self._logits_capture_fn(router_logits)

        # Step 2: Get indices type.
        indices_type = self._get_indices_type()

        # Step 3: Compute routing (delegated to subclass)
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, indices_type, input_ids=input_ids
        )

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # Lazy-load override from env var on first call (TP-safe:
        # every worker's router does this independently).
        # Layer ID is assigned via a monotonic counter — routers are
        # called in layer order during the first forward pass.
        if not self._override_checked:
            self._override_checked = True
            cfg_path = os.environ.get("MOE_ROUTING_OVERRIDE")
            if cfg_path and os.path.isfile(cfg_path):
                global _override_layer_counter
                layer_id = _override_layer_counter
                _override_layer_counter += 1
                self._override_fn = _build_override_for_layer(
                    cfg_path, layer_id
                )

        # Override routing decisions (original already captured above).
        if self._override_fn is not None:
            orig_shape = topk_ids.shape
            orig_device = topk_ids.device
            topk_weights, topk_ids = self._override_fn(
                topk_weights, topk_ids, router_logits
            )
            assert topk_ids.shape == orig_shape, (
                f"override changed topk_ids shape: {orig_shape} -> {topk_ids.shape}"
            )
            assert topk_weights.shape == orig_shape, (
                f"override changed topk_weights shape: {orig_shape} -> {topk_weights.shape}"
            )
            assert topk_ids.device == orig_device, (
                f"override moved topk_ids to {topk_ids.device}, expected {orig_device}"
            )
            assert (topk_ids >= 0).all() and (topk_ids < self.global_num_experts).all(), (
                "override produced out-of-range expert IDs"
            )

        # Step 4: Apply EPLB mapping
        topk_ids = self._apply_eplb_mapping(topk_ids)

        # Step 5: Convert indices dtype
        topk_ids = self._convert_indices_dtype(topk_ids, indices_type)

        return topk_weights, topk_ids
