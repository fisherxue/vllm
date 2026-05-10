# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import contextlib
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import torch
import torch.distributed

from vllm.config.model import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom op for routing capture -- traceable by torch.compile / Dynamo.
#
# Registered as a formal custom op so that torch.compile traces through it
# cleanly without graph breaks.  ALL TP ranks call this op with a real
# device buffer to ensure identical CUDA graph structure (symmetry).
# Non-rank-0 buffers are written but never read for D2H.
# ---------------------------------------------------------------------------


@torch.library.custom_op("vllm::capture_routing", mutates_args={"buffer"})
def capture_routing_op(
    buffer: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_id: int,
    batch_size: int,
) -> None:
    buffer[layer_id, :batch_size, :].copy_(
        topk_ids[:batch_size].to(buffer.dtype), non_blocking=True
    )


@capture_routing_op.register_fake
def _capture_routing_op_fake(
    buffer: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_id: int,
    batch_size: int,
) -> None:
    pass


_MB = 1024 * 1024


class _RoutedExpertsDeviceCache:
    """Per-device (GPU) cache for capturing routed expert IDs during forward
    pass.  Always writes at row 0 so that CUDA graph replay sees the same
    addresses that were recorded at capture time.
    """

    DTYPE = torch.int16

    def __init__(
        self,
        max_num_batched_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        device: str,
        num_experts: int = 0,
    ) -> None:
        # Layout: (L, N, K) so that buffer[layer_id] is a contiguous (N, K)
        # view — required by the FlashInfer routing-replay kernel which
        # writes expert IDs assuming contiguous row-major memory.
        self.num_hidden_layers = num_hidden_layers
        self.buffer = torch.zeros(
            (num_hidden_layers, max_num_batched_tokens, num_experts_per_tok),
            dtype=self.DTYPE,
            device=device,
        )
        # Optional logits buffer: (L, N, E) float16 for full router logits.
        if num_experts > 0:
            self.logits_buffer: torch.Tensor | None = torch.zeros(
                (num_hidden_layers, max_num_batched_tokens, num_experts),
                dtype=torch.float16,
                device=device,
            )
        else:
            self.logits_buffer = None
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        size = self.buffer.nbytes
        if self.logits_buffer is not None:
            size += self.logits_buffer.nbytes
        return size

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[layer_id, :batch, :].copy_(topk_ids, non_blocking=True)

    def _finalize_allocation_log(self):
        buf_mb = self.get_buffer_size_bytes() / _MB
        logger.info(
            "Routing experts device buffer allocated. shape=%s, size=%.2f MB",
            tuple(self.buffer.shape),
            buf_mb,
        )


class _RoutedExpertsHostCache:
    """Host (CPU) cache using numpy arrays for per-request routing data.

    Numpy arrays avoid torch dispatcher overhead for scatter operations.
    Lazy per-request allocation avoids a massive up-front buffer.
    """

    DTYPE = np.int16

    def __init__(
        self,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        max_model_len: int,
    ) -> None:
        self.max_model_len = max_model_len
        self.num_hidden_layers = num_hidden_layers
        self.num_experts_per_tok = num_experts_per_tok

        self._req_buffers: dict[str, np.ndarray] = {}
        self._filled_len: dict[str, int] = {}
        self._total_allocated_bytes = 0

        self._finalize_allocation_log()

    def get_buffer_size_bytes(self) -> int:
        return self._total_allocated_bytes

    def get_or_grow_buffer(self, req_id: str, max_pos: int) -> np.ndarray:
        required_len = max_pos + 1

        if req_id not in self._req_buffers:
            buf = np.zeros(
                (required_len, self.num_hidden_layers, self.num_experts_per_tok),
                dtype=self.DTYPE,
            )
            self._req_buffers[req_id] = buf
            self._total_allocated_bytes += buf.nbytes
            return buf

        buf = self._req_buffers[req_id]
        if buf.shape[0] >= required_len:
            return buf

        new_len = min(max(required_len, buf.shape[0] * 2), self.max_model_len)
        new_buf = np.zeros(
            (new_len, self.num_hidden_layers, self.num_experts_per_tok),
            dtype=self.DTYPE,
        )
        new_buf[: buf.shape[0]] = buf
        self._total_allocated_bytes += new_buf.nbytes - buf.nbytes
        self._req_buffers[req_id] = new_buf
        return new_buf

    def get_buffer(self, req_id: str) -> np.ndarray | None:
        return self._req_buffers.get(req_id)

    def update_filled_len(self, req_id: str, max_pos: int) -> None:
        new_len = max_pos + 1
        self._filled_len[req_id] = max(self._filled_len.get(req_id, 0), new_len)

    def get_filled_len(self, req_id: str) -> int:
        return self._filled_len.get(req_id, 0)

    def free_request(self, req_id: str) -> None:
        if req_id in self._req_buffers:
            self._total_allocated_bytes -= self._req_buffers.pop(req_id).nbytes
        self._filled_len.pop(req_id, None)

    def _finalize_allocation_log(self):
        logger.info(
            "Routing experts host cache initialized (lazy allocation). "
            "max_model_len=%s, layers=%s, experts_per_tok=%s",
            self.max_model_len,
            self.num_hidden_layers,
            self.num_experts_per_tok,
        )


class _RoutedExpertsDiskCache:
    """Disk-backed cache for per-request router logits.

    Uses pre-allocated memory-mapped numpy files for progressive writes.
    On request completion, compacts to a final .npy file and deletes the
    temp mmap.  Internal filenames are counter-based (not raw req_id) to
    avoid path-injection from user-controlled request IDs.
    """

    DTYPE = np.float16

    def __init__(
        self,
        output_dir: str,
        num_hidden_layers: int,
        num_experts: int,
        max_model_len: int,
    ) -> None:
        self.output_dir = output_dir
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.max_model_len = max_model_len
        os.makedirs(output_dir, exist_ok=True)

        self._req_files: dict[str, str] = {}
        self._req_mmaps: dict[str, np.memmap] = {}
        self._filled_len: dict[str, int] = {}
        self._file_counter = 0
        self._pid = os.getpid()

    def _make_temp_path(self) -> str:
        self._file_counter += 1
        return os.path.join(
            self.output_dir,
            f"_tmp_{self._pid}_{self._file_counter:08d}.logits.mmap",
        )

    def get_or_create_mmap(self, req_id: str) -> np.memmap:
        if req_id not in self._req_mmaps:
            path = self._make_temp_path()
            mmap = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=self.DTYPE,
                shape=(self.max_model_len, self.num_hidden_layers, self.num_experts),
            )
            self._req_files[req_id] = path
            self._req_mmaps[req_id] = mmap
        return self._req_mmaps[req_id]

    def write_chunk(
        self, req_id: str, positions: np.ndarray, logits_chunk: np.ndarray
    ) -> None:
        mmap = self.get_or_create_mmap(req_id)
        mmap[positions] = logits_chunk
        max_pos = int(positions.max()) + 1
        self._filled_len[req_id] = max(self._filled_len.get(req_id, 0), max_pos)

    _COPY_CHUNK = 4096

    def finalize(self, req_id: str) -> str | None:
        """Write compact final .npy via chunked mmap-to-mmap copy."""
        if req_id not in self._req_mmaps:
            return None
        src = self._req_mmaps.pop(req_id)
        filled = self._filled_len.pop(req_id, 0)
        temp_path = self._req_files.pop(req_id)
        final_path = temp_path.replace("_tmp_", "").replace(".mmap", ".npy")
        dst = np.lib.format.open_memmap(
            final_path,
            mode="w+",
            dtype=self.DTYPE,
            shape=(filled, self.num_hidden_layers, self.num_experts),
        )
        for start in range(0, filled, self._COPY_CHUNK):
            end = min(start + self._COPY_CHUNK, filled)
            dst[start:end] = src[start:end]
        dst.flush()
        del dst, src
        os.unlink(temp_path)
        return final_path

    def free_request(self, req_id: str) -> None:
        """Discard without saving (preemption)."""
        mmap = self._req_mmaps.pop(req_id, None)
        temp_path = self._req_files.pop(req_id, None)
        self._filled_len.pop(req_id, None)
        if mmap is not None:
            del mmap
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        device: str,
        shared_host_cache: _RoutedExpertsHostCache | None = None,
        skip_host_cache: bool = False,
        num_experts: int = 0,
        router_logits_output_dir: str | None = None,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                max_num_batched_tokens=max_num_batched_tokens,
                num_fused_shared_experts=num_fused_shared_experts,
                max_model_len=max_model_len,
                device=device,
                shared_host_cache=shared_host_cache,
                skip_host_cache=skip_host_cache,
                num_experts=num_experts,
                router_logits_output_dir=router_logits_output_dir,
            )
        return _RoutedExpertsCapturerNoop()

    @abstractmethod
    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self, req_id: str, seqlen: int | None = None, free_slot: bool = True
    ):
        raise NotImplementedError

    def sync_fwd_experts_buffer_DtoH(
        self,
        positions: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ):
        raise NotImplementedError

    def finalize_pending_copy(self):
        raise NotImplementedError

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


def _count_moe_layers(hf_config) -> int:
    """Count the number of MoE layers in a model.

    Resolves three known config shapes:
    - Nemotron-style: an explicit ``layers_block_type`` list with "moe" entries.
    - Qwen3MoE / DeepSeek-style sparse: ``decoder_sparse_step > 1`` with optional
      ``mlp_only_layers`` exclusions.
    - Default: every layer is MoE except those listed in ``mlp_only_layers``.
    """
    layers_block_type = getattr(hf_config, "layers_block_type", None)
    if layers_block_type is not None:
        return layers_block_type.count("moe")
    n = hf_config.num_hidden_layers
    mlp_only = getattr(hf_config, "mlp_only_layers", None) or []
    step = getattr(hf_config, "decoder_sparse_step", 1) or 1
    if step > 1:
        return sum(1 for i in range(n) if (i + 1) % step == 0 and i not in mlp_only)
    return n - sum(1 for i in mlp_only if 0 <= i < n)


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer with GPU device cache and CPU host cache.

    Performance strategy -- async D2H with optimized host-cache scatter:

    Every decode step we issue a non-blocking D2H copy on a dedicated
    CUDA stream.  The scatter into per-request host-cache buffers is
    deferred to the start of the NEXT step (by which time the copy has
    finished).  The scatter loop is optimized with direct scalar access
    to avoid numpy slice views, int() conversions, and .max() calls.

    At extraction time (when a request finishes), data is already in a
    contiguous host buffer -- just a numpy slice, no concatenation.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        max_num_batched_tokens: int,
        num_fused_shared_experts: int,
        max_model_len: int,
        device: str,
        shared_host_cache: _RoutedExpertsHostCache | None = None,
        skip_host_cache: bool = False,
        num_experts: int = 0,
        router_logits_output_dir: str | None = None,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = _count_moe_layers(model_config.hf_text_config)
        hf_cfg = model_config.hf_text_config
        for _attr in ("num_experts_per_tok", "top_k_experts", "moe_topk",
                       "num_experts_per_token"):
            _val = getattr(hf_cfg, _attr, None)
            if _val is not None and _val > 0:
                self.num_experts_per_tok = _val
                break
        else:
            raise ValueError(
                "Could not determine num_experts_per_tok from model config. "
                f"Checked attributes on {type(hf_cfg).__name__}."
            )
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self._skip_host_cache = skip_host_cache
        self._enable_logits = num_experts > 0 and router_logits_output_dir is not None

        if skip_host_cache:
            self.host_cache = None
            logger.info("Skipping host cache for device %s (non-rank-0)", device)
        elif shared_host_cache is not None:
            self.host_cache = shared_host_cache
        else:
            self.host_cache = _RoutedExpertsHostCache(
                num_hidden_layers=self.num_hidden_layers,
                num_experts_per_tok=self.num_experts_per_tok,
                max_model_len=self.max_model_len,
            )

        self.device_cache = _RoutedExpertsDeviceCache(
            max_num_batched_tokens=self.max_num_batched_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            device=device,
            num_experts=num_experts if self._enable_logits else 0,
        )

        # ---- Disk cache for logits (rank-0 only) ----
        if self._enable_logits and not skip_host_cache:
            assert router_logits_output_dir is not None
            self.disk_cache: _RoutedExpertsDiskCache | None = (
                _RoutedExpertsDiskCache(
                    output_dir=router_logits_output_dir,
                    num_hidden_layers=self.num_hidden_layers,
                    num_experts=num_experts,
                    max_model_len=self.max_model_len,
                )
            )
        else:
            self.disk_cache = None

        # ---- Async D2H pipeline (rank-0 only) ----
        # Non-rank-0 workers only need the device buffer for symmetric
        # CUDA graph capture; they skip the D2H pipeline entirely.
        self._has_pending_copy = False
        self._pending_positions: np.ndarray | None = None
        self._pending_num_scheduled: dict[str, int] | None = None
        self._pending_total_tokens: int = 0

        if not skip_host_cache:
            # Same (L, N, K) layout as device_cache.buffer.
            self._pinned_staging = torch.zeros(
                (
                    self.num_hidden_layers,
                    max_num_batched_tokens,
                    self.num_experts_per_tok,
                ),
                dtype=_RoutedExpertsDeviceCache.DTYPE,
                pin_memory=True,
            )
            # Private device snapshot: source for the async D2H. Decouples
            # the in-flight copy from device_cache.buffer, which the next
            # step's MoE writes overwrite in place on main_stream.
            self._device_staging = torch.empty_like(self.device_cache.buffer)
            self._copy_stream = torch.cuda.Stream(device=device)
            self._copy_event = torch.cuda.Event()

            # Logits staging buffers (only when logits capture is enabled).
            if self._enable_logits and self.device_cache.logits_buffer is not None:
                logits_shape = self.device_cache.logits_buffer.shape
                logits_dtype = self.device_cache.logits_buffer.dtype
                self._pinned_logits_staging: torch.Tensor | None = torch.zeros(
                    logits_shape, dtype=logits_dtype, pin_memory=True,
                )
                self._device_logits_staging: torch.Tensor | None = (
                    torch.empty_like(self.device_cache.logits_buffer)
                )
            else:
                self._pinned_logits_staging = None
                self._device_logits_staging = None

            pinned_mb = self._pinned_staging.nbytes / _MB
            if self._pinned_logits_staging is not None:
                pinned_mb += self._pinned_logits_staging.nbytes / _MB
            logger.info(
                "Routing experts pinned staging buffer allocated. "
                "size=%.2f MB (logits=%s)",
                pinned_mb,
                self._enable_logits,
            )
        else:
            self._pinned_staging = None
            self._device_staging = None
            self._copy_stream = None
            self._copy_event = None
            self._pinned_logits_staging = None
            self._device_logits_staging = None
            logger.info(
                "Routing experts device-only capturer (rank != 0). "
                "Device buffer shape=%s",
                tuple(self.device_cache.buffer.shape),
            )

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    # ------------------------------------------------------------------
    # sync_fwd_experts_buffer_DtoH -- called AFTER the forward pass
    # ------------------------------------------------------------------

    def sync_fwd_experts_buffer_DtoH(
        self,
        positions: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ):
        if self.host_cache is None:
            return

        # 1. Finalize previous async copy -- the copy had an entire
        #    forward pass to complete so event.synchronize() is ~free.
        if self._has_pending_copy:
            self._copy_event.synchronize()
            self._scatter_to_host()
            self._has_pending_copy = False

        total_tokens = sum(num_scheduled_tokens.values())
        if total_tokens == 0:
            return

        # 2. Snapshot the device buffer on main_stream into a private
        #    staging buffer, then issue the D2H from the staging buffer
        #    on a dedicated copy stream. The snapshot serializes after
        #    the current step's MoE writes (same stream) and is private
        #    from the next step's MoE writes, so the in-flight D2H is
        #    not aliased by step N+1's forward under async scheduling.
        main_stream = torch.cuda.current_stream(self._copy_stream.device)
        self._device_staging[:, :total_tokens, :].copy_(
            self.device_cache.buffer[:, :total_tokens, :], non_blocking=True
        )
        # Snapshot logits device buffer on main stream (if enabled).
        if (
            self._device_logits_staging is not None
            and self.device_cache.logits_buffer is not None
        ):
            self._device_logits_staging[:, :total_tokens, :].copy_(
                self.device_cache.logits_buffer[:, :total_tokens, :],
                non_blocking=True,
            )
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(main_stream)
            self._pinned_staging[:, :total_tokens, :].copy_(
                self._device_staging[:, :total_tokens, :], non_blocking=True
            )
            # D2H logits on same stream (if enabled).
            if (
                self._pinned_logits_staging is not None
                and self._device_logits_staging is not None
            ):
                self._pinned_logits_staging[:, :total_tokens, :].copy_(
                    self._device_logits_staging[:, :total_tokens, :],
                    non_blocking=True,
                )
            self._copy_event.record()

        # 3. Save metadata for deferred scatter.
        self._pending_positions = positions.numpy().copy()
        self._pending_num_scheduled = num_scheduled_tokens
        self._pending_total_tokens = total_tokens
        self._has_pending_copy = True

    # ------------------------------------------------------------------
    # Optimized scatter into pre-allocated host-cache buffers
    # ------------------------------------------------------------------

    def _scatter_to_host(self):
        """Scatter D2H data into per-request host cache buffers.

        Staging layout is (L, N, K).  Host cache layout is (seq_len, L, K).
        We transpose the staging slice to (N, L, K) before scattering so
        that indexing by token position naturally yields (L, K) rows.
        """
        # Transpose (L, N, K) -> (N, L, K) for the active token range.
        host_values = (
            self._pinned_staging[:, : self._pending_total_tokens, :]
            .numpy()
            .transpose(1, 0, 2)
        )
        positions_np = self._pending_positions
        host_cache = self.host_cache
        assert self._pending_num_scheduled is not None
        assert positions_np is not None
        assert host_cache is not None

        # Transpose logits (L, N, E) -> (N, L, E) if enabled.
        logits_values = None
        if self._pinned_logits_staging is not None and self.disk_cache is not None:
            logits_values = (
                self._pinned_logits_staging[:, : self._pending_total_tokens, :]
                .numpy()
                .transpose(1, 0, 2)
            )

        offset = 0
        for req_id, n_tokens in self._pending_num_scheduled.items():
            if n_tokens == 0:
                continue

            if n_tokens == 1:
                pos_val = int(positions_np[offset])
                buf = host_cache.get_or_grow_buffer(req_id, pos_val)
                buf[pos_val] = host_values[offset]
                host_cache.update_filled_len(req_id, pos_val)
            else:
                pos = positions_np[offset : offset + n_tokens]
                max_pos = int(pos[-1]) if n_tokens > 0 else 0
                if n_tokens > 1:
                    max_pos = int(pos.max())
                buf = host_cache.get_or_grow_buffer(req_id, max_pos)
                buf[pos] = host_values[offset : offset + n_tokens]
                host_cache.update_filled_len(req_id, max_pos)

            # Scatter logits to disk (same position mapping).
            if logits_values is not None and self.disk_cache is not None:
                if n_tokens == 1:
                    pos_arr = positions_np[offset : offset + 1]
                else:
                    pos_arr = pos  # type: ignore[possibly-undefined]
                self.disk_cache.write_chunk(
                    req_id, pos_arr, logits_values[offset : offset + n_tokens]
                )

            offset += n_tokens

        self._pending_positions = None
        self._pending_num_scheduled = None
        self._pending_total_tokens = 0

    # ------------------------------------------------------------------
    # finalize_pending_copy -- call before reading host cache
    # ------------------------------------------------------------------

    def finalize_pending_copy(self):
        """Ensure the most recent async D2H copy has been scattered into
        host cache buffers.  Call before get_routed_experts."""
        if self._has_pending_copy:
            self._copy_event.synchronize()
            self._scatter_to_host()
            self._has_pending_copy = False

    # ------------------------------------------------------------------
    # Extraction -- O(1), just a numpy slice
    # ------------------------------------------------------------------

    def get_routed_experts(
        self,
        req_id: str,
        seqlen: int | None = None,
        free_slot: bool = True,
    ):
        if self.host_cache is None:
            return None
        buf = self.host_cache.get_buffer(req_id)
        if buf is None:
            return None
        filled = self.host_cache.get_filled_len(req_id)
        if filled <= 0:
            return None
        effective_len = min(filled, seqlen) if seqlen is not None else filled
        result = buf[:effective_len].copy()
        if free_slot:
            self.host_cache.free_request(req_id)
        return result

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def get_routed_experts(self, req_id: str, seqlen=None, free_slot=True):
        return None

    def sync_fwd_experts_buffer_DtoH(self, positions, num_scheduled_tokens):
        pass

    def finalize_pending_copy(self):
        pass

    def get_host_cache(self):
        return None

    def get_device_cache(self):
        pass


# Global capturer instance (per-process)
_global_expert_capturer: RoutedExpertsCapturer | None = _RoutedExpertsCapturerNoop()
_shared_host_cache: _RoutedExpertsHostCache | None = None


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def extract_routed_experts_for_current_batch(
    req_ids: list[str],
    requests: dict,
    req_id_to_index: dict[str, int],
    num_tokens_no_spec: np.ndarray,
    max_model_len: int,
) -> dict[str, np.ndarray] | None:
    """Extract routed experts for requests predicted to finish this step.

    Checks all stop conditions the scheduler will check (max_tokens,
    EOS token, stop tokens, max_model_len) so that every finished
    request gets its routing data attached to the ModelRunnerOutput.

    Args:
        req_ids: Ordered request IDs for the current batch.
        requests: Map of req_id to CachedRequestState (read-only).
        req_id_to_index: Map of req_id to input batch index.
        num_tokens_no_spec: Array of total token counts per request index.
        max_model_len: Maximum model sequence length.
    """
    capturer = get_global_experts_capturer()
    if capturer is None:
        return None, None
    host_cache = capturer.get_host_cache()
    if host_cache is None:
        return None, None

    finishing_req_ids: list[str] = []
    for req_id in req_ids:
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        sp = req_state.sampling_params
        if sp is None:
            continue
        output_ids = req_state.output_token_ids
        if not output_ids:
            continue
        if len(output_ids) < sp.min_tokens:
            continue

        finishing = False
        last_token = output_ids[-1]

        # EOS token (mirrors check_stop: eos_token_id is None
        # when ignore_eos=True, so this naturally respects that)
        if last_token == sp.eos_token_id:
            finishing = True

        # Explicit stop token IDs
        if not finishing and sp.stop_token_ids and last_token in sp.stop_token_ids:
            finishing = True

        # max_tokens / max_model_len length cap
        if not finishing:
            if sp.max_tokens is not None and len(output_ids) >= sp.max_tokens:
                finishing = True
            else:
                req_idx = req_id_to_index.get(req_id)
                if req_idx is not None:
                    total = num_tokens_no_spec[req_idx]
                    if total >= max_model_len:
                        finishing = True

        if finishing:
            finishing_req_ids.append(req_id)

    if not finishing_req_ids:
        return None, None

    # At least one request is finishing: ensure the latest async D2H
    # copy has been scattered into the host cache.
    capturer.finalize_pending_copy()

    result: dict[str, np.ndarray] = {}
    logits_paths: dict[str, str] = {}
    for req_id in finishing_req_ids:
        seqlen = host_cache.get_filled_len(req_id)
        if seqlen <= 0:
            continue
        experts = capturer.get_routed_experts(req_id, seqlen=seqlen, free_slot=False)
        if experts is not None:
            result[req_id] = experts
        # Finalize logits to disk if disk cache is active.
        disk_cache = getattr(capturer, "disk_cache", None)
        if disk_cache is not None:
            path = disk_cache.finalize(req_id)
            if path is not None:
                logits_paths[req_id] = path

    return (
        result if result else None,
        logits_paths if logits_paths else None,
    )


def free_routing_buffers(
    finished_req_ids: set[str],
    preempted_req_ids: set[str] | None = None,
) -> None:
    """Free host cache buffers for finished and preempted requests.

    Finished requests had their routing data extracted in the previous
    step.

    Preempted requests are re-prefilled from scratch when they resume,
    so their host-cache buffer is freed here. This means any routing
    already accumulated in the host cache for the preempted request is
    dropped without being emitted on a ``ModelRunnerOutput`` --
    consumers see ``routed_experts=None`` for those requests with no
    other signal. Partial-rollout / async-RL pipelines that depend on
    receiving routing for preempted requests should treat preemption
    as a routing-data loss event and either keep preemption disabled
    or reconstruct routing on the resumed prefill.
    """
    capturer = get_global_experts_capturer()
    if capturer is None:
        return
    host_cache = capturer.get_host_cache()
    if host_cache is None:
        return

    disk_cache = getattr(capturer, "disk_cache", None)

    for req_id in finished_req_ids:
        host_cache.free_request(req_id)
        if disk_cache is not None:
            disk_cache.free_request(req_id)
    if preempted_req_ids:
        for req_id in preempted_req_ids:
            host_cache.free_request(req_id)
            if disk_cache is not None:
                disk_cache.free_request(req_id)


def issue_routing_d2h_copy(
    input_batch_req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    positions: torch.Tensor,
    positions_cpu: torch.Tensor,
) -> None:
    """Issue async D2H copy of routed experts after the forward pass.

    Called EARLY in the execute_model epilogue so the copy overlaps with
    eplb, kv_connector finalization, and draft work.
    finalize_pending_copy() + get_routed_experts() happen later in
    extract_routed_experts_for_current_batch().
    """
    capturer = get_global_experts_capturer()
    if capturer is None:
        return

    ordered = {
        req_id: num_scheduled_tokens[req_id]
        for req_id in input_batch_req_ids
        if req_id in num_scheduled_tokens
    }
    n = sum(ordered.values())
    positions_cpu[:n].copy_(positions[:n])
    capturer.sync_fwd_experts_buffer_DtoH(
        positions=positions_cpu[:n],
        num_scheduled_tokens=ordered,
    )


def split_routed_experts(
    routed_experts: np.ndarray,
    prompt_len: int,
    num_output_tokens: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split routing data into prompt and generation portions.

    Args:
        routed_experts: Full routing array of shape (seq_len, L, K).
        prompt_len: Number of prompt tokens for the request.
        num_output_tokens: Actual number of generated tokens (from
            detokenizer).  When provided, the generation portion is
            clipped to this length — necessary with MTP where the model
            runner may capture routing for more tokens than the final
            output contains.

    Returns:
        (prompt_routed_experts, gen_routed_experts) numpy arrays, either
        of which may be None if the corresponding portion is empty.
    """
    prompt_routed_experts = routed_experts[:prompt_len]
    gen_routed_experts = routed_experts[prompt_len:]

    # Clip generation routing to match actual output tokens.
    if (
        num_output_tokens is not None
        and gen_routed_experts.shape[0] > num_output_tokens
        and num_output_tokens > 0
    ):
        gen_routed_experts = gen_routed_experts[:num_output_tokens]

    if prompt_routed_experts.size == 0:
        prompt_routed_experts = None
    if gen_routed_experts.size == 0:
        gen_routed_experts = None

    return prompt_routed_experts, gen_routed_experts


def get_shared_host_cache() -> _RoutedExpertsHostCache | None:
    return _shared_host_cache


def create_shared_host_cache(
    model_config: ModelConfig,
    max_model_len: int,
) -> _RoutedExpertsHostCache:
    global _shared_host_cache
    num_hidden_layers = _count_moe_layers(model_config.hf_text_config)
    num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
    _shared_host_cache = _RoutedExpertsHostCache(
        num_hidden_layers=num_hidden_layers,
        num_experts_per_tok=num_experts_per_tok,
        max_model_len=max_model_len,
    )
    return _shared_host_cache


def init_routed_experts_capturer_with_shared_cache(
    enable: bool,
    model_config: ModelConfig,
    num_fused_shared_experts: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    device: str,
    rank: int = 0,
    world_size: int = 1,
    num_experts: int = 0,
    router_logits_output_dir: str | None = None,
) -> RoutedExpertsCapturer:
    """Initialize capturer with rank-aware handling (only rank 0 captures)."""
    if not enable:
        capturer = _RoutedExpertsCapturerNoop()
        set_global_experts_capturer(capturer)
        return capturer

    if world_size > 1 and rank != 0:
        # Non-rank-0 workers get a device-only capturer (no host cache,
        # no D2H pipeline) so that ALL ranks have a real device buffer.
        # This ensures the custom op call in every MoE layer produces
        # identical CUDA graph structure across TP ranks.
        logger.info("Creating device-only routed experts capturer for rank %s", rank)
        capturer = RoutedExpertsCapturer.create(
            enable=True,
            model_config=model_config,
            num_fused_shared_experts=num_fused_shared_experts,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            device=device,
            skip_host_cache=True,
            num_experts=num_experts,
        )
        set_global_experts_capturer(capturer)
        return capturer

    capturer = RoutedExpertsCapturer.create(
        enable=True,
        model_config=model_config,
        num_fused_shared_experts=num_fused_shared_experts,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        device=device,
        skip_host_cache=False,
        num_experts=num_experts,
        router_logits_output_dir=router_logits_output_dir,
    )
    set_global_experts_capturer(capturer)
    return capturer


def bind_routing_capture_to_model(model) -> None:
    """Bind routing capture buffers to all FusedMoE layers in the model.

    Must be called AFTER init_routed_experts_capturer_with_shared_cache()
    and BEFORE CUDA graph capture.  All TP ranks get a real buffer so
    that the custom op call produces identical graph structure.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    capturer = get_global_experts_capturer()
    device_cache = capturer.get_device_cache()
    if device_cache is None:
        return  # routing capture not enabled

    buffer = device_cache.buffer

    # Mark the buffer so CUDA graphs do NOT snapshot/restore its contents.
    if hasattr(torch.compiler, "cudagraph_mark_tensor_static"):
        torch.compiler.cudagraph_mark_tensor_static(buffer)
    elif hasattr(torch._C, "_set_static_address_tag"):
        torch._C._set_static_address_tag(buffer, True)
    with contextlib.suppress(Exception):
        torch._dynamo.mark_static_address(buffer)

    bound = 0
    for module in model.modules():
        if isinstance(module, FusedMoE) and hasattr(module, "moe_layer_id"):
            # Per-FusedMoE configurations not yet validated for routing
            # capture. These signals are only set after model init, so a
            # config-level guard cannot see them.
            if module.moe_config.is_sequence_parallel:
                raise NotImplementedError(
                    "routed-experts capture is not yet validated with "
                    "sequence parallelism on the FusedMoE layer "
                    "(moe_config.is_sequence_parallel=True)."
                )
            if (
                module.moe_config.dp_size > 1
                and not module.quant_method.supports_internal_mk
            ):
                raise NotImplementedError(
                    "routed-experts capture is not yet validated with "
                    "naive DP dispatch (non-modular quant method "
                    f"{type(module.quant_method).__name__}, "
                    f"dp_size={module.moe_config.dp_size})."
                )

            if module.quant_method.is_monolithic:
                raise NotImplementedError(
                    "routed-experts capture requires the non-monolithic "
                    "(Triton) MoE path so that select_experts() runs in "
                    "Python and logical expert IDs can be captured. "
                    f"Layer {module.moe_layer_id} uses monolithic quant "
                    f"method {type(module.quant_method).__name__}. "
                    "Disable FlashInfer MoE or set "
                    "enable_return_routed_experts=False."
                )

            layer_id = module.moe_layer_id
            layer_buf = buffer[layer_id]  # (N_max, K)
            module._routing_replay_out = layer_buf
            # Mark each per-layer view as static so CUDA graphs don't
            # snapshot/restore or relocate the buffer during replay.
            if hasattr(torch.compiler, "cudagraph_mark_tensor_static"):
                torch.compiler.cudagraph_mark_tensor_static(layer_buf)
            with contextlib.suppress(Exception):
                torch._dynamo.mark_static_address(layer_buf)

            # Wire capture_fn to write logical (pre-EPLB) expert IDs
            # into the device buffer. capture_fn fires inside
            # select_experts() before _apply_eplb_mapping(), so the
            # buffer receives logical IDs. The runner's post-
            # select_experts() write is guarded to avoid overwriting
            # with physical IDs when capture_fn is set.
            if hasattr(module, "router"):
                _buf = layer_buf

                def _capture_logical_ids(
                    topk_ids: torch.Tensor, buf: torch.Tensor = _buf
                ) -> None:
                    buf[: topk_ids.shape[0]].copy_(topk_ids.to(buf.dtype))

                module.router.set_capture_fn(_capture_logical_ids)

            # Wire logits capture (if logits buffer is allocated).
            logits_buffer = device_cache.logits_buffer
            if (
                logits_buffer is not None
                and hasattr(module, "router")
            ):
                logits_layer_buf = logits_buffer[layer_id]
                if hasattr(torch.compiler, "cudagraph_mark_tensor_static"):
                    torch.compiler.cudagraph_mark_tensor_static(logits_layer_buf)
                with contextlib.suppress(Exception):
                    torch._dynamo.mark_static_address(logits_layer_buf)

                _lbuf = logits_layer_buf

                def _capture_logits(
                    router_logits: torch.Tensor, buf: torch.Tensor = _lbuf
                ) -> None:
                    buf[: router_logits.shape[0]].copy_(
                        router_logits.to(buf.dtype)
                    )

                module.router.set_logits_capture_fn(_capture_logits)

            bound += 1

    logger.info(
        "Bound routing capture buffer to %s FusedMoE layers. Buffer shape=%s",
        bound,
        tuple(buffer.shape),
    )

    _maybe_bind_override_from_env(model)


# ---------------------------------------------------------------------------
# Routing override: env-var-based config for TP-safe initialization
# ---------------------------------------------------------------------------

_ROUTING_OVERRIDE_ENV = "MOE_ROUTING_OVERRIDE"


def _maybe_bind_override_from_env(model) -> None:
    """Load routing override config from env var and bind to model.

    Called from bind_routing_capture_to_model() which runs in EVERY
    worker process BEFORE CUDA graph capture, so this is both TP-safe
    and graph-safe.
    """
    config_path = os.environ.get(_ROUTING_OVERRIDE_ENV)
    if not config_path:
        return

    import json

    if not os.path.isfile(config_path):
        logger.warning(
            "%s=%s but file does not exist, skipping override",
            _ROUTING_OVERRIDE_ENV, config_path,
        )
        return

    with open(config_path) as f:
        cfg = json.load(f)

    hot_mask_path = cfg["hot_mask_path"]
    hot_mask = np.load(hot_mask_path)
    policy = cfg.get("policy", "two_tier")
    normalize = cfg.get("normalize", "softmax")

    import torch as _torch

    hot_tensor = _torch.from_numpy(hot_mask).bool()
    per_layer = hot_tensor.ndim == 2

    if normalize == "softmax":
        norm_fn = lambda x: _torch.softmax(x, dim=-1)
    elif normalize == "sigmoid":
        norm_fn = _torch.sigmoid
    else:
        raise ValueError(f"Unknown normalize={normalize!r} in {config_path}")

    keep_top1 = policy == "two_tier"
    start_slot = 1 if keep_top1 else 0
    _gpu_cache: dict[int, _torch.Tensor] = {}

    def override_fn(topk_weights, topk_ids, router_logits, layer_idx):
        if layer_idx not in _gpu_cache:
            src = hot_tensor[layer_idx] if per_layer else hot_tensor
            _gpu_cache[layer_idx] = src.to(topk_ids.device)
        hot = _gpu_cache[layer_idx]

        T, K = topk_ids.shape
        new_ids = topk_ids.clone()
        new_weights = topk_weights.clone()
        scores = norm_fn(router_logits)
        arange_T = _torch.arange(T, device=topk_ids.device)

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
            new_ids[:, slot] = _torch.where(is_miss, best, new_ids[:, slot])
            new_weights[:, slot] = _torch.where(
                is_miss, best_w, new_weights[:, slot]
            )

        return new_weights, new_ids

    n = bind_routing_override_to_model(model, override_fn)
    logger.info(
        "Loaded routing override from %s: policy=%s, normalize=%s, "
        "hot_mask shape=%s, bound to %s layers",
        config_path, policy, normalize, hot_mask.shape, n,
    )


def bind_routing_override_to_model(
    model,
    override_fn: Callable,
) -> int:
    """Bind a routing override function to all FusedMoE layers.

    override_fn(topk_weights, topk_ids, router_logits, layer_idx)
        -> (new_topk_weights, new_topk_ids)

    The per-router hook receives (topk_weights, topk_ids, router_logits)
    without layer_idx; this function wraps the user's 4-arg function into
    per-layer 3-arg closures that bake in the layer index.

    Returns the number of layers bound.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    bound = 0
    for module in model.modules():
        if not isinstance(module, FusedMoE) or not hasattr(module, "moe_layer_id"):
            continue
        if not hasattr(module, "router"):
            continue

        layer_id = module.moe_layer_id

        def _make_override(lid: int):
            def _override(topk_weights, topk_ids, router_logits):
                return override_fn(topk_weights, topk_ids, router_logits, lid)
            return _override

        module.router.set_override_fn(_make_override(layer_id))
        bound += 1

    logger.info("Bound routing override to %s FusedMoE layers.", bound)
    return bound
