# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpu_test


def test_bind_routing_capture_to_model_sets_layer_view(monkeypatch):
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec_mod

    class _DummyMoEConfig:
        is_sequence_parallel = False
        dp_size = 1

    class _DummyQuantMethod:
        supports_internal_mk = True
        is_monolithic = False

    class _DummyRouter:
        def __init__(self):
            self.capture_fn = None

        def set_capture_fn(self, fn):
            self.capture_fn = fn

    class DummyFusedMoE:
        _routing_replay_out: torch.Tensor

        def __init__(self, moe_layer_id):
            self.moe_layer_id = moe_layer_id
            self.moe_config = _DummyMoEConfig()
            self.quant_method = _DummyQuantMethod()
            self.router = _DummyRouter()

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    num_layers, num_tokens, top_k = 4, 8, 2
    buffer = torch.zeros((num_layers, num_tokens, top_k), dtype=torch.int16)

    class DummyDeviceCache:
        def __init__(self, buf):
            self.buffer = buf
            self.logits_buffer = None

    class DummyCapturer:
        def get_device_cache(self):
            return DummyDeviceCache(buffer)

    monkeypatch.setattr(rec_mod, "get_global_experts_capturer", lambda: DummyCapturer())

    m0 = DummyFusedMoE(moe_layer_id=0)
    m2 = DummyFusedMoE(moe_layer_id=2)

    class DummyModel:
        def modules(self):
            return iter([m0, m2])

    rec_mod.bind_routing_capture_to_model(DummyModel())

    assert torch.equal(m0._routing_replay_out, buffer[0])
    assert torch.equal(m2._routing_replay_out, buffer[2])

    # capture_fn should be wired to write logical IDs to the buffer
    assert m0.router.capture_fn is not None
    assert m2.router.capture_fn is not None


def test_bind_routing_capture_to_model_noop_when_disabled(monkeypatch):
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec_mod

    class DummyCapturer:
        def get_device_cache(self):
            return None

    monkeypatch.setattr(rec_mod, "get_global_experts_capturer", lambda: DummyCapturer())

    class DummyModel:
        def modules(self):
            return iter([])

    rec_mod.bind_routing_capture_to_model(DummyModel())


# =========================================================================
# Tests for device-cache routing replay architecture
# =========================================================================


class TestRoutedExpertsDeviceCache:
    """Tests for _RoutedExpertsDeviceCache (GPU buffer for routing data)."""

    def test_allocation_shape_and_dtype(self):
        """Device cache allocates (L, N, K) int16 buffer."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDeviceCache,
        )

        cache = _RoutedExpertsDeviceCache(
            num_hidden_layers=40,
            max_num_batched_tokens=8192,
            num_experts_per_tok=8,
            device="cpu",
        )
        assert cache.buffer.shape == (40, 8192, 8)
        assert cache.buffer.dtype == torch.int16

    def test_per_layer_view_is_contiguous(self):
        """buffer[layer_id] gives contiguous (N, K) view for FlashInfer."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDeviceCache,
        )

        cache = _RoutedExpertsDeviceCache(
            num_hidden_layers=40,
            max_num_batched_tokens=8192,
            num_experts_per_tok=8,
            device="cpu",
        )
        layer_view = cache.buffer[0]
        assert layer_view.is_contiguous()
        assert layer_view.shape == (8192, 8)


class TestRoutedExpertsHostCache:
    """Tests for _RoutedExpertsHostCache (per-request numpy buffer)."""

    def test_sentinel_initialization(self):
        """Host cache initializes with zeros by default."""
        import numpy as np

        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsHostCache,
        )

        cache = _RoutedExpertsHostCache(
            num_hidden_layers=40,
            num_experts_per_tok=8,
            max_model_len=1024,
        )
        buf = cache.get_or_grow_buffer("req1", max_pos=100)
        assert buf.dtype == np.int16
        assert (buf == 0).all(), "Host cache must initialize with zeros"

    def test_grow_preserves_existing_data(self):
        """Growing the buffer preserves previously written data."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsHostCache,
        )

        cache = _RoutedExpertsHostCache(
            num_hidden_layers=40,
            num_experts_per_tok=8,
            max_model_len=1024,
        )
        buf = cache.get_or_grow_buffer("req1", max_pos=50)
        buf[0, 0, 0] = 42
        buf2 = cache.get_or_grow_buffer("req1", max_pos=200)
        assert buf2[0, 0, 0] == 42, "Data lost during buffer grow"

    def test_free_request_removes_buffer(self):
        """Freeing a request removes its buffer."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsHostCache,
        )

        cache = _RoutedExpertsHostCache(
            num_hidden_layers=40,
            num_experts_per_tok=8,
            max_model_len=1024,
        )
        cache.get_or_grow_buffer("req1", max_pos=50)
        cache.free_request("req1")
        assert cache.get_buffer("req1") is None


# =========================================================================
# Tests for logical-ID capture via capture_fn
# =========================================================================


def test_capture_fn_writes_logical_ids_to_buffer(monkeypatch):
    """capture_fn should write topk_ids into the device buffer."""
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec_mod

    class _DummyMoEConfig:
        is_sequence_parallel = False
        dp_size = 1

    class _DummyQuantMethod:
        supports_internal_mk = True
        is_monolithic = False

    class _DummyRouter:
        def __init__(self):
            self.capture_fn = None

        def set_capture_fn(self, fn):
            self.capture_fn = fn

    class DummyFusedMoE:
        def __init__(self, moe_layer_id):
            self.moe_layer_id = moe_layer_id
            self.moe_config = _DummyMoEConfig()
            self.quant_method = _DummyQuantMethod()
            self.router = _DummyRouter()

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    num_layers, num_tokens, top_k = 4, 8, 2
    buffer = torch.zeros((num_layers, num_tokens, top_k), dtype=torch.int16)

    class DummyDeviceCache:
        def __init__(self, buf):
            self.buffer = buf
            self.logits_buffer = None

    class DummyCapturer:
        def get_device_cache(self):
            return DummyDeviceCache(buffer)

    monkeypatch.setattr(rec_mod, "get_global_experts_capturer", lambda: DummyCapturer())

    m0 = DummyFusedMoE(moe_layer_id=0)

    class DummyModel:
        def modules(self):
            return iter([m0])

    rec_mod.bind_routing_capture_to_model(DummyModel())

    # Simulate calling capture_fn with logical IDs
    logical_ids = torch.tensor([[3, 5], [1, 7]], dtype=torch.int32)
    m0.router.capture_fn(logical_ids)

    # Buffer should contain the logical IDs
    expected = torch.tensor([[3, 5], [1, 7]], dtype=torch.int16)
    assert torch.equal(buffer[0, :2, :], expected)


def test_monolithic_layers_raise_error(monkeypatch):
    """Monolithic quant methods should raise NotImplementedError."""
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec_mod

    class _DummyMoEConfig:
        is_sequence_parallel = False
        dp_size = 1

    class _MonolithicQuantMethod:
        supports_internal_mk = True
        is_monolithic = True

    class _DummyRouter:
        def __init__(self):
            self.capture_fn = None

        def set_capture_fn(self, fn):
            self.capture_fn = fn

    class DummyFusedMoE:
        def __init__(self, moe_layer_id):
            self.moe_layer_id = moe_layer_id
            self.moe_config = _DummyMoEConfig()
            self.quant_method = _MonolithicQuantMethod()
            self.router = _DummyRouter()

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    num_layers, num_tokens, top_k = 4, 8, 2
    buffer = torch.zeros((num_layers, num_tokens, top_k), dtype=torch.int16)

    class DummyDeviceCache:
        def __init__(self, buf):
            self.buffer = buf
            self.logits_buffer = None

    class DummyCapturer:
        def get_device_cache(self):
            return DummyDeviceCache(buffer)

    monkeypatch.setattr(rec_mod, "get_global_experts_capturer", lambda: DummyCapturer())

    m_mono = DummyFusedMoE(moe_layer_id=0)

    class DummyModel:
        def modules(self):
            return iter([m_mono])

    with pytest.raises(NotImplementedError, match="monolithic"):
        rec_mod.bind_routing_capture_to_model(DummyModel())


# =========================================================================
# Tests for _RoutedExpertsDiskCache (router logits to disk)
# =========================================================================


class TestRoutedExpertsDiskCache:
    def test_write_chunk_and_finalize(self, tmp_path):
        """Write chunks at known positions and verify the final .npy file."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDiskCache,
        )

        cache = _RoutedExpertsDiskCache(
            output_dir=str(tmp_path),
            num_hidden_layers=2,
            num_experts=4,
            max_model_len=16,
        )
        chunk = np.ones((3, 2, 4), dtype=np.float16) * 0.5
        positions = np.array([0, 1, 2])
        cache.write_chunk("req1", positions, chunk)

        chunk2 = np.ones((2, 2, 4), dtype=np.float16) * 0.75
        positions2 = np.array([3, 4])
        cache.write_chunk("req1", positions2, chunk2)

        path = cache.finalize("req1")
        assert path is not None
        assert path.endswith(".npy")

        data = np.load(path)
        assert data.shape == (5, 2, 4)
        assert data.dtype == np.float16
        assert np.allclose(data[:3], 0.5)
        assert np.allclose(data[3:5], 0.75)

    def test_free_request_deletes_files(self, tmp_path):
        """Freeing a request should delete the temp mmap file."""
        import os

        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDiskCache,
        )

        cache = _RoutedExpertsDiskCache(
            output_dir=str(tmp_path),
            num_hidden_layers=2,
            num_experts=4,
            max_model_len=16,
        )
        chunk = np.ones((2, 2, 4), dtype=np.float16)
        cache.write_chunk("req1", np.array([0, 1]), chunk)

        # Temp file should exist
        assert len(os.listdir(tmp_path)) == 1

        cache.free_request("req1")
        assert len(os.listdir(tmp_path)) == 0

    def test_finalize_nonexistent_returns_none(self, tmp_path):
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDiskCache,
        )

        cache = _RoutedExpertsDiskCache(
            output_dir=str(tmp_path),
            num_hidden_layers=2,
            num_experts=4,
            max_model_len=16,
        )
        assert cache.finalize("nonexistent") is None

    def test_counter_based_filenames(self, tmp_path):
        """Filenames use internal counter, not raw request IDs."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDiskCache,
        )

        cache = _RoutedExpertsDiskCache(
            output_dir=str(tmp_path),
            num_hidden_layers=1,
            num_experts=2,
            max_model_len=4,
        )
        chunk = np.ones((1, 1, 2), dtype=np.float16)
        cache.write_chunk("../../etc/passwd", np.array([0]), chunk)
        path = cache.finalize("../../etc/passwd")
        assert path is not None
        # Path should be inside tmp_path, not escaped
        assert str(tmp_path) in path
        assert "passwd" not in path


    def test_finalize_chunked_copy_does_not_load_full_tensor(self, tmp_path):
        """Finalize uses chunked mmap-to-mmap copy, not np.array()."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDiskCache,
        )

        cache = _RoutedExpertsDiskCache(
            output_dir=str(tmp_path),
            num_hidden_layers=2,
            num_experts=4,
            max_model_len=1024,
        )
        # Write at positions 0-9 and 500-509
        chunk = np.arange(10 * 2 * 4, dtype=np.float16).reshape(10, 2, 4)
        cache.write_chunk("req1", np.arange(10), chunk)
        chunk2 = np.ones((10, 2, 4), dtype=np.float16) * 99.0
        cache.write_chunk("req1", np.arange(500, 510), chunk2)

        path = cache.finalize("req1")
        assert path is not None
        data = np.load(path)
        # Filled to position 510
        assert data.shape == (510, 2, 4)
        assert np.allclose(data[0:10], chunk)
        assert np.allclose(data[500:510], 99.0)
        # Positions 10-499 should be zero (unfilled)
        assert np.allclose(data[10:500], 0.0)

    def test_pid_prefix_in_filenames(self, tmp_path):
        """Filenames include PID to avoid collisions across restarts."""
        import os

        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDiskCache,
        )

        cache = _RoutedExpertsDiskCache(
            output_dir=str(tmp_path),
            num_hidden_layers=1,
            num_experts=2,
            max_model_len=4,
        )
        chunk = np.ones((1, 1, 2), dtype=np.float16)
        cache.write_chunk("req1", np.array([0]), chunk)
        path = cache.finalize("req1")
        assert path is not None
        assert str(os.getpid()) in os.path.basename(path)


# =========================================================================
# Tests for extract_routed_experts_for_current_batch return type
# =========================================================================


def test_extract_returns_tuple_when_capturer_disabled(monkeypatch):
    """extract_routed_experts_for_current_batch returns (None, None)
    when capture is disabled, not bare None."""
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec_mod

    class NoopCapturer:
        def get_host_cache(self):
            return None

        def get_device_cache(self):
            return None

    monkeypatch.setattr(rec_mod, "get_global_experts_capturer", lambda: NoopCapturer())

    result = rec_mod.extract_routed_experts_for_current_batch(
        req_ids=["r1"],
        requests={},
        req_id_to_index={},
        num_tokens_no_spec=np.array([0]),
        max_model_len=1024,
    )
    # Must be a tuple, not bare None — caller unpacks as (ids, paths)
    assert isinstance(result, tuple)
    assert result == (None, None)


def test_extract_returns_tuple_when_capturer_is_none(monkeypatch):
    """extract returns (None, None) when global capturer is None."""
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec_mod

    monkeypatch.setattr(rec_mod, "get_global_experts_capturer", lambda: None)

    result = rec_mod.extract_routed_experts_for_current_batch(
        req_ids=[],
        requests={},
        req_id_to_index={},
        num_tokens_no_spec=np.array([]),
        max_model_len=1024,
    )
    assert isinstance(result, tuple)
    assert result == (None, None)


# =========================================================================
# Tests for RequestOutput.add propagation
# =========================================================================


def test_request_output_add_propagates_router_logits_path():
    """RequestOutput.add() should merge router_logits_path from next output."""
    from vllm.outputs import CompletionOutput, RequestOutput

    base = RequestOutput(
        request_id="r1",
        prompt="hello",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[CompletionOutput(index=0, text="a", token_ids=[3],
                                  cumulative_logprob=None, logprobs=None)],
        finished=False,
    )
    assert base.router_logits_path is None

    next_out = RequestOutput(
        request_id="r1",
        prompt="hello",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[CompletionOutput(index=0, text="b", token_ids=[4],
                                  cumulative_logprob=None, logprobs=None)],
        finished=True,
        router_logits_path="/tmp/logits.npy",
    )
    base.add(next_out, aggregate=True)
    assert base.router_logits_path == "/tmp/logits.npy"


def test_request_output_add_does_not_overwrite_with_none():
    """If next output has no path, existing path should be preserved."""
    from vllm.outputs import CompletionOutput, RequestOutput

    base = RequestOutput(
        request_id="r1",
        prompt="hello",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[CompletionOutput(index=0, text="a", token_ids=[3],
                                  cumulative_logprob=None, logprobs=None)],
        finished=False,
        router_logits_path="/tmp/existing.npy",
    )

    next_out = RequestOutput(
        request_id="r1",
        prompt="hello",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[CompletionOutput(index=0, text="b", token_ids=[4],
                                  cumulative_logprob=None, logprobs=None)],
        finished=True,
    )
    base.add(next_out, aggregate=True)
    assert base.router_logits_path == "/tmp/existing.npy"


# =========================================================================
# Integration test: EPLB logical vs physical ID capture (requires GPU)
# =========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for EPLB Triton kernel")
def test_capture_fn_returns_logical_ids_with_eplb():
    """With EPLB enabled, capture_fn captures logical IDs while
    select_experts() returns physical (remapped) IDs."""
    from vllm.distributed.eplb.eplb_state import EplbLayerState
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        BaseRouter,
    )

    device = "cuda"
    num_experts = 4
    num_physical = 4
    top_k = 2
    num_tokens = 3

    # Known EPLB mapping: logical i → physical (i+1)%4
    # logical 0 → physical 1
    # logical 1 → physical 2
    # logical 2 → physical 3
    # logical 3 → physical 0
    logical_to_physical = torch.tensor(
        [[1], [2], [3], [0]], dtype=torch.int32, device=device
    )
    replica_count = torch.ones(num_experts, dtype=torch.int32, device=device)
    load_view = torch.zeros(num_physical, dtype=torch.int32, device=device)
    record_enabled = torch.tensor(False, device=device)

    eplb_state = EplbLayerState()
    eplb_state.logical_to_physical_map = logical_to_physical
    eplb_state.logical_replica_count = replica_count
    eplb_state.expert_load_view = load_view
    eplb_state.should_record_tensor = record_enabled

    # Router subclass that returns fixed logical IDs.
    fixed_logical_ids = torch.tensor(
        [[0, 1], [2, 3], [1, 0]], dtype=torch.int32, device=device
    )
    fixed_weights = torch.tensor(
        [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], dtype=torch.float32, device=device
    )

    class _FixedRouter(BaseRouter):
        @property
        def routing_method_type(self):
            return RoutingMethodType.Default

        def _compute_routing(self, hidden_states, router_logits,
                             indices_type, *, input_ids=None):
            return fixed_weights.clone(), fixed_logical_ids.clone()

    router = _FixedRouter(
        top_k=top_k,
        global_num_experts=num_experts,
        eplb_state=eplb_state,
        enable_eplb=True,
    )

    # Set up capture buffer (simulates what bind_routing_capture_to_model does)
    capture_buf = torch.zeros(num_tokens, top_k, dtype=torch.int16, device=device)

    def _capture(topk_ids, buf=capture_buf):
        buf[: topk_ids.shape[0]].copy_(topk_ids.to(buf.dtype))

    router.set_capture_fn(_capture)

    # Run select_experts
    dummy_hidden = torch.randn(num_tokens, 64, device=device)
    dummy_logits = torch.randn(num_tokens, num_experts, device=device)

    returned_weights, returned_ids = router.select_experts(
        hidden_states=dummy_hidden,
        router_logits=dummy_logits,
    )

    # capture_buf should contain LOGICAL IDs (pre-EPLB)
    expected_logical = torch.tensor(
        [[0, 1], [2, 3], [1, 0]], dtype=torch.int16, device=device
    )
    assert torch.equal(capture_buf, expected_logical), (
        f"capture_fn should record logical IDs.\n"
        f"  Expected: {expected_logical}\n"
        f"  Got:      {capture_buf}"
    )

    # returned_ids should contain PHYSICAL IDs (post-EPLB remap)
    # logical 0→phys 1, logical 1→phys 2, logical 2→phys 3, logical 3→phys 0
    expected_physical = torch.tensor(
        [[1, 2], [3, 0], [2, 1]], dtype=torch.int32, device=device
    )
    assert torch.equal(returned_ids, expected_physical), (
        f"select_experts() should return physical IDs.\n"
        f"  Expected: {expected_physical}\n"
        f"  Got:      {returned_ids}"
    )
