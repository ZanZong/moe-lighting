"""Unit tests for long-context offloading policy and KV cache manager.

These tests exercise the analytical models in:
  * fastmoe.backend.long_context_policy
  * fastmoe.backend.kv_cache_manager
  * fastmoe.backend.memory  (new session-aware methods)
  * fastmoe.backend.task    (new session fields on Req)

No GPU is required for the majority of tests.  Tests that require CUDA or
optional dependencies are automatically skipped when unavailable.
"""

import sys
import unittest
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fastmoe.backend.kv_cache_manager import (
    ConversationSession,
    MultiTurnKVCacheManager,
    estimate_prefix_reuse_savings,
)
from fastmoe.backend.long_context_policy import (
    AttentionStrategy,
    LongContextAnalysis,
    analyze_long_context,
    compute_offload_threshold,
    sweep_context_lengths,
)
from fastmoe.backend.task import Req
from fastmoe.backend.utils import HardwareConfig

GB = 1 << 30
T = 1e12

CUDA_AVAILABLE = torch.cuda.is_available()

try:
    import pulp as _pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hardware_config() -> HardwareConfig:
    return HardwareConfig(
        gmem=24 * GB,
        cmem=192 * GB,
        ctog_bdw=16 * GB,
        g_bdw=300 * GB,
        c_bdw=76 * GB,
        gpu_flops=104 * T,
        cpu_flops=1.6 * T,
        tp_size=1,
    )


class _FakeModelConfig:
    """Minimal model config resembling Mixtral-8x7B."""
    num_hidden_layers = 32
    hidden_size = 4096
    intermediate_size = 14336
    num_attention_heads = 32
    num_key_value_heads = 8
    num_local_experts = 8
    topk = 2
    context_len = 32768
    vocab_size = 32000


# ---------------------------------------------------------------------------
# Tests: Req session fields
# ---------------------------------------------------------------------------

class TestReqSessionFields(unittest.TestCase):

    def test_default_single_turn(self):
        req = Req("r1", "hello", [1, 2, 3])
        self.assertIsNone(req.session_id)
        self.assertEqual(req.turn_id, 0)
        self.assertEqual(req.cached_prefix_len, 0)

    def test_multi_turn_fields(self):
        req = Req("r2", "turn2", [10, 11, 12],
                  session_id="sess-abc", turn_id=2, cached_prefix_len=50)
        self.assertEqual(req.session_id, "sess-abc")
        self.assertEqual(req.turn_id, 2)
        self.assertEqual(req.cached_prefix_len, 50)
        self.assertEqual(req.input_len, 3)

    def test_backward_compat_no_kwargs(self):
        """Old call sites that pass only (rid, text, ids) must still work."""
        req = Req("r3", "text", list(range(10)))
        self.assertIsNone(req.session_id)


# ---------------------------------------------------------------------------
# Tests: TokenToKVPool session methods (skipped without CUDA)
# ---------------------------------------------------------------------------

@unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available - skipping GPU pool tests")
class TestTokenToKVPoolSession(unittest.TestCase):

    def _make_pool(self, gpu_size=64, cpu_size=512):
        from fastmoe.backend.memory import TokenToKVPool
        return TokenToKVPool(
            gpu_size=gpu_size,
            cpu_size=cpu_size,
            dtype=torch.float16,
            head_num=8,
            head_dim=128,
            layer_num=2,
        )

    def test_has_session_initial_false(self):
        pool = self._make_pool()
        self.assertFalse(pool.has_session("sess-1"))

    def test_save_and_load_session(self):
        pool = self._make_pool()
        layer = 0
        num_tokens = 4
        pool.kv_data_cpu[layer][:num_tokens, :, :, :] = torch.ones(
            num_tokens, 2, 8, 128, dtype=torch.float16
        )
        pool.save_session("sess-1", start_loc=0, num_tokens=num_tokens, layer=layer)
        self.assertTrue(pool.has_session("sess-1", layer=layer))
        pool.kv_data_cpu[layer][:num_tokens] = 0
        hit = pool.load_session("sess-1", dst_start=0,
                                num_tokens=num_tokens, layer=layer)
        self.assertTrue(hit)
        restored = pool.kv_data_cpu[layer][:num_tokens]
        self.assertTrue(torch.all(restored == 1.0))

    def test_load_session_miss(self):
        pool = self._make_pool()
        hit = pool.load_session("nonexistent", dst_start=0, num_tokens=4, layer=0)
        self.assertFalse(hit)

    def test_evict_session(self):
        pool = self._make_pool()
        pool.save_session("sess-2", start_loc=0, num_tokens=4, layer=0)
        self.assertTrue(pool.has_session("sess-2", layer=0))
        pool.evict_session("sess-2")
        self.assertFalse(pool.has_session("sess-2", layer=0))

    def test_clear_removes_sessions(self):
        pool = self._make_pool()
        pool.save_session("sess-3", start_loc=0, num_tokens=4, layer=0)
        pool.clear()
        self.assertFalse(pool.has_session("sess-3", layer=0))


# ---------------------------------------------------------------------------
# Tests: ConversationSession
# ---------------------------------------------------------------------------

class TestConversationSession(unittest.TestCase):

    def test_kv_bytes(self):
        s = ConversationSession(
            session_id="s",
            cpu_start_loc=0,
            num_cached_tokens=100,
            layer_num=4,
            head_num=4,
            head_dim=64,
            dtype_bytes=2,
        )
        # 100 tokens * 2 (K+V) * 4 heads * 64 dim * 2 bytes * 4 layers
        expected = 100 * 2 * 4 * 64 * 2 * 4
        self.assertEqual(s.kv_bytes(), expected)

    def test_touch_updates_timestamp(self):
        import time
        s = ConversationSession("s", 0, 10)
        old_ts = s.last_access
        time.sleep(0.01)
        s.touch()
        self.assertGreater(s.last_access, old_ts)


# ---------------------------------------------------------------------------
# Tests: MultiTurnKVCacheManager
# ---------------------------------------------------------------------------

class TestMultiTurnKVCacheManager(unittest.TestCase):

    def _make_manager(self, max_bytes=10 * 1024 * 1024):
        return MultiTurnKVCacheManager(
            max_cpu_bytes=max_bytes,
            layer_num=4,
            head_num=4,
            head_dim=64,
            dtype_bytes=2,
        )

    def test_register_and_lookup(self):
        mgr = self._make_manager()
        mgr.register_session("s1", cpu_start_loc=0, num_cached_tokens=64)
        sess = mgr.get_session("s1")
        self.assertIsNotNone(sess)
        self.assertEqual(sess.num_cached_tokens, 64)
        self.assertTrue(mgr.has_session("s1"))

    def test_update_existing_session(self):
        mgr = self._make_manager()
        mgr.register_session("s1", cpu_start_loc=0, num_cached_tokens=64)
        mgr.register_session("s1", cpu_start_loc=0, num_cached_tokens=128)
        sess = mgr.get_session("s1")
        self.assertEqual(sess.num_cached_tokens, 128)
        self.assertEqual(sess.turn_id, 1)

    def test_evict_explicit(self):
        mgr = self._make_manager()
        mgr.register_session("s1", cpu_start_loc=0, num_cached_tokens=64)
        evicted = mgr.evict_session("s1")
        self.assertTrue(evicted)
        self.assertFalse(mgr.has_session("s1"))

    def test_evict_nonexistent(self):
        mgr = self._make_manager()
        self.assertFalse(mgr.evict_session("ghost"))

    def test_lru_eviction_on_overflow(self):
        # Each session = 64 tokens * 2 * 4 heads * 64 dim * 2 bytes * 4 layers
        per_session = 64 * 2 * 4 * 64 * 2 * 4
        mgr = self._make_manager(max_bytes=2 * per_session + 1)
        mgr.register_session("s1", 0, 64)
        mgr.register_session("s2", 100, 64)
        mgr.get_session("s1")  # make s1 more-recently-used
        mgr.register_session("s3", 200, 64)
        self.assertTrue(mgr.has_session("s1"))
        self.assertFalse(mgr.has_session("s2"))
        self.assertTrue(mgr.has_session("s3"))

    def test_prefix_reuse_info_hit(self):
        mgr = self._make_manager()
        mgr.register_session("s1", 0, 100)
        cached, new = mgr.prefix_reuse_info("s1", list(range(50)))
        self.assertEqual(cached, 100)
        self.assertEqual(new, 50)

    def test_prefix_reuse_info_miss(self):
        mgr = self._make_manager()
        cached, new = mgr.prefix_reuse_info("unknown", list(range(30)))
        self.assertEqual(cached, 0)
        self.assertEqual(new, 30)

    def test_num_sessions(self):
        mgr = self._make_manager()
        self.assertEqual(mgr.num_sessions, 0)
        mgr.register_session("s1", 0, 10)
        self.assertEqual(mgr.num_sessions, 1)

    def test_used_cpu_bytes_tracking(self):
        mgr = self._make_manager()
        mgr.register_session("s1", 0, 64)
        b1 = mgr.used_cpu_bytes
        mgr.register_session("s2", 100, 64)
        b2 = mgr.used_cpu_bytes
        self.assertGreater(b2, b1)
        mgr.evict_session("s1")
        self.assertEqual(mgr.used_cpu_bytes, b2 - b1)


# ---------------------------------------------------------------------------
# Tests: estimate_prefix_reuse_savings
# ---------------------------------------------------------------------------

class TestEstimatePrefixReuseSavings(unittest.TestCase):

    def test_returns_positive_savings(self):
        s = estimate_prefix_reuse_savings(
            cached_tokens=512,
            new_tokens=64,
            batch_size=1,
            layer_num=32,
            head_num=8,
            head_dim=128,
            cpu_flops=1.6e12,
            c_bdw=76 * GB,
        )
        self.assertGreater(s["saved_attn_flops"], 0)
        self.assertGreater(s["saved_kv_bytes"], 0)
        self.assertGreater(s["saved_attn_time"], 0)
        self.assertEqual(s["total_context"], 512 + 64)

    def test_zero_cached_tokens(self):
        s = estimate_prefix_reuse_savings(
            cached_tokens=0,
            new_tokens=128,
            batch_size=2,
            layer_num=32,
            head_num=8,
            head_dim=128,
            cpu_flops=1.6e12,
            c_bdw=76 * GB,
        )
        self.assertEqual(s["saved_attn_flops"], 0)
        self.assertEqual(s["saved_kv_bytes"], 0)
        self.assertEqual(s["saved_attn_time"], 0)


# ---------------------------------------------------------------------------
# Tests: long_context_policy
# ---------------------------------------------------------------------------

class TestLongContextPolicy(unittest.TestCase):

    def setUp(self):
        self.hw = _make_hardware_config()
        self.model = _FakeModelConfig()

    def test_analyze_returns_correct_type(self):
        a = analyze_long_context(1024, 1, self.model, self.hw)
        self.assertIsInstance(a, LongContextAnalysis)
        self.assertEqual(a.context_len, 1024)

    def test_strategy_is_valid_enum(self):
        a = analyze_long_context(512, 8, self.model, self.hw)
        self.assertIn(a.strategy, list(AttentionStrategy))

    def test_cpu_attn_grows_with_context(self):
        a_short = analyze_long_context(512, 1, self.model, self.hw)
        a_long = analyze_long_context(4096, 1, self.model, self.hw)
        self.assertLess(a_short.cpu_attn_time, a_long.cpu_attn_time)

    def test_latency_positive(self):
        a = analyze_long_context(2048, 4, self.model, self.hw)
        self.assertGreater(a.latency_cpu_offload, 0)
        self.assertGreater(a.latency_gpu_only, 0)
        self.assertGreater(a.latency_sparse_cpu, 0)

    def test_compute_offload_threshold_positive(self):
        threshold = compute_offload_threshold(self.model, self.hw, batch_size=1)
        self.assertGreater(threshold, 0)

    def test_sweep_context_lengths(self):
        ctxs = [512, 1024, 2048]
        results = sweep_context_lengths(ctxs, 1, self.model, self.hw)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, LongContextAnalysis)

    def test_threshold_monotonic_with_batch(self):
        """Larger batches mean more attention work, so threshold should decrease."""
        t1 = compute_offload_threshold(self.model, self.hw, batch_size=1)
        t8 = compute_offload_threshold(self.model, self.hw, batch_size=8)
        self.assertGreaterEqual(t1, t8)


# ---------------------------------------------------------------------------
# Tests: optimizer.analyze_long_context_offloading
# ---------------------------------------------------------------------------

@unittest.skipUnless(PULP_AVAILABLE, "pulp not installed - skipping optimizer tests")
class TestOptimizerLongContextAnalysis(unittest.TestCase):

    def test_returns_list_of_dicts(self):
        from fastmoe.backend.optimizer import analyze_long_context_offloading
        hw = _make_hardware_config()
        model = _FakeModelConfig()
        results = analyze_long_context_offloading(
            model, hw, context_lengths=[512, 2048], batch_size=1
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("context_len", r)
            self.assertIn("strategy", r)
            self.assertIn("latency_cpu_offload_ms", r)
            self.assertIn("latency_gpu_only_ms", r)


if __name__ == "__main__":
    unittest.main()
