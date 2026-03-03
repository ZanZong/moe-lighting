"""Multi-turn KV cache manager for long-context offloading scenarios.

In multi-turn dialogues and large-text multi-query workloads, successive
requests to the same conversation share a common KV-cache prefix
(the conversation history).  Re-computing this prefix every turn wastes
both compute and memory bandwidth, and causes the CPU KV pool to grow
unboundedly.

This module provides:

  * :class:`ConversationSession` – per-session metadata (turn count, cached
    prefix length, last-access timestamp).
  * :class:`MultiTurnKVCacheManager` – an LRU-eviction cache of session KV
    data stored on CPU.  On a cache hit for a new turn, only the *delta*
    (new tokens since the last cached turn) needs to be computed and stored;
    the shared prefix is fetched directly without re-running attention.
  * :func:`estimate_prefix_reuse_savings` – a utility that estimates the
    compute and bandwidth savings from prefix reuse at a given context length.

The manager integrates with :class:`~fastmoe.backend.memory.TokenToKVPool`
to perform the actual CPU tensor copies.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ConversationSession:
    """State for a single multi-turn conversation session.

    Attributes
    ----------
    session_id:        Unique string identifier for this conversation.
    cpu_start_loc:     Starting token index inside the CPU KV pool where
                       this session's KV data begins.
    num_cached_tokens: Number of tokens whose KV pairs are currently cached
                       on CPU (i.e. the shared prefix for the next turn).
    turn_id:           Zero-based index of the most recently completed turn.
    last_access:       Wall-clock time of the last cache access (for LRU).
    layer_num:         Number of transformer layers (needed for memory maths).
    head_num:          Number of KV heads per layer (after TP sharding).
    head_dim:          Dimension per head.
    dtype_bytes:       Bytes per element (2 for fp16, 4 for fp32).
    """

    session_id: str
    cpu_start_loc: int
    num_cached_tokens: int
    turn_id: int = 0
    last_access: float = field(default_factory=time.monotonic)
    layer_num: int = 32
    head_num: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2

    def kv_bytes(self) -> int:
        """Total CPU memory (bytes) occupied by this session's cached KV."""
        # 2 = key + value tensors
        return (
            self.num_cached_tokens
            * 2
            * self.head_num
            * self.head_dim
            * self.dtype_bytes
            * self.layer_num
        )

    def touch(self) -> None:
        """Update the last-access timestamp (used by LRU eviction)."""
        self.last_access = time.monotonic()


class MultiTurnKVCacheManager:
    """Session-level KV cache manager for multi-turn long-context workloads.

    The manager maintains a dictionary of active :class:`ConversationSession`
    objects, keyed by ``session_id``.  When the total cached bytes exceed
    ``max_cpu_bytes``, LRU eviction removes the least-recently-used sessions.

    Parameters
    ----------
    max_cpu_bytes:
        Upper bound on aggregate CPU memory (bytes) used to store cached KV
        data across all sessions.  When exceeded, sessions are evicted in
        LRU order.
    layer_num, head_num, head_dim, dtype_bytes:
        Model / hardware parameters shared across all sessions; forwarded to
        :class:`ConversationSession`.
    """

    def __init__(
        self,
        max_cpu_bytes: int,
        layer_num: int = 32,
        head_num: int = 8,
        head_dim: int = 128,
        dtype_bytes: int = 2,
    ) -> None:
        self.max_cpu_bytes = max_cpu_bytes
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes

        # Ordered by insertion / last-access for LRU tracking
        self._sessions: OrderedDict[str, ConversationSession] = OrderedDict()
        self._used_cpu_bytes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_session(self, session_id: str) -> bool:
        """Return True if *session_id* has a cached KV prefix."""
        return session_id in self._sessions

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a session and update its LRU timestamp, or None."""
        if session_id not in self._sessions:
            return None
        session = self._sessions[session_id]
        session.touch()
        # Move to end (most-recently-used)
        self._sessions.move_to_end(session_id)
        return session

    def register_session(
        self,
        session_id: str,
        cpu_start_loc: int,
        num_cached_tokens: int,
    ) -> ConversationSession:
        """Register a newly completed turn's KV cache for future reuse.

        If a session with the same *session_id* already exists, its metadata
        is updated in-place (extending the cached prefix length).  Otherwise
        a new :class:`ConversationSession` is created.

        Memory is not copied here — the caller is responsible for ensuring
        the CPU KV pool already contains the data at ``cpu_start_loc``.

        Returns the (updated or new) :class:`ConversationSession`.
        """
        if session_id in self._sessions:
            old = self._sessions[session_id]
            self._used_cpu_bytes -= old.kv_bytes()
            old.cpu_start_loc = cpu_start_loc
            old.num_cached_tokens = num_cached_tokens
            old.turn_id += 1
            old.touch()
            self._used_cpu_bytes += old.kv_bytes()
            self._sessions.move_to_end(session_id)
            return old
        else:
            session = ConversationSession(
                session_id=session_id,
                cpu_start_loc=cpu_start_loc,
                num_cached_tokens=num_cached_tokens,
                layer_num=self.layer_num,
                head_num=self.head_num,
                head_dim=self.head_dim,
                dtype_bytes=self.dtype_bytes,
            )
            self._sessions[session_id] = session
            self._used_cpu_bytes += session.kv_bytes()
            self._evict_if_needed()
            return session

    def evict_session(self, session_id: str) -> bool:
        """Explicitly remove a session from the cache.

        Returns True if the session existed and was removed, False otherwise.
        """
        if session_id not in self._sessions:
            return False
        session = self._sessions.pop(session_id)
        self._used_cpu_bytes -= session.kv_bytes()
        return True

    def prefix_reuse_info(
        self, session_id: str, new_input_ids: List[int]
    ) -> Tuple[int, int]:
        """Return ``(cached_tokens, new_tokens)`` for the given new turn.

        ``cached_tokens`` is the number of tokens from the previous turn(s)
        that are already stored in the CPU KV pool and do NOT need to be
        re-computed.  ``new_tokens`` is the number of new tokens that must be
        processed during the prefill phase.

        If the session is not in the cache, ``cached_tokens`` is 0 and
        ``new_tokens`` is ``len(new_input_ids)``.
        """
        session = self.get_session(session_id)
        if session is None:
            return 0, len(new_input_ids)
        cached = session.num_cached_tokens
        new = len(new_input_ids)
        return cached, new

    @property
    def num_sessions(self) -> int:
        return len(self._sessions)

    @property
    def used_cpu_bytes(self) -> int:
        return self._used_cpu_bytes

    def session_ids(self) -> List[str]:
        """Return all active session IDs (LRU order, oldest first)."""
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict the least-recently-used session(s) until under the limit."""
        while self._used_cpu_bytes > self.max_cpu_bytes and self._sessions:
            # Pop from the front (least-recently-used)
            lru_id, lru_session = next(iter(self._sessions.items()))
            self._sessions.pop(lru_id)
            self._used_cpu_bytes -= lru_session.kv_bytes()


# ---------------------------------------------------------------------------
# Utility: savings estimation
# ---------------------------------------------------------------------------

def estimate_prefix_reuse_savings(
    cached_tokens: int,
    new_tokens: int,
    batch_size: int,
    layer_num: int,
    head_num: int,
    head_dim: int,
    cpu_flops: float,
    c_bdw: float,
    dtype_bytes: int = 2,
) -> dict:
    """Estimate compute and bandwidth savings from prefix KV reuse.

    Parameters
    ----------
    cached_tokens:  Number of prefix tokens already in the CPU KV cache.
    new_tokens:     Number of new tokens in this turn (excluding prefix).
    batch_size:     Number of concurrent requests that share this prefix.
    layer_num:      Number of transformer layers.
    head_num:       KV heads per layer (post-TP).
    head_dim:       Head dimension.
    cpu_flops:      CPU peak FLOPS (ops/s).
    c_bdw:          CPU memory bandwidth (bytes/s).
    dtype_bytes:    Bytes per tensor element.

    Returns
    -------
    dict with keys:
      ``saved_attn_flops``: FLOPs saved per decode step by skipping re-computation.
      ``saved_kv_bytes``:   CPU memory bytes saved by not re-fetching old KV.
      ``saved_attn_time``:  Approximate seconds saved per decode step.
      ``memory_overhead``:  Extra CPU bytes to store the cached KV.
    """
    total_ctx = cached_tokens + new_tokens
    per_token_kv = 2 * head_num * head_dim * dtype_bytes  # key + value

    # FLOPs for QK^T dot-product over the cached prefix tokens
    # (2 * batch * 1 * cached_tokens * head_num * head_dim)
    saved_flops_per_layer = (
        2 * batch_size * 1 * cached_tokens * head_num * head_dim * 2
    )
    saved_attn_flops = saved_flops_per_layer * layer_num

    # Memory traffic for loading the cached KV
    saved_kv_bytes_per_layer = batch_size * cached_tokens * per_token_kv
    saved_kv_bytes = saved_kv_bytes_per_layer * layer_num

    # Approximate time savings (dominated by memory bandwidth for long ctx)
    saved_attn_time = saved_kv_bytes / c_bdw

    # Memory overhead: we keep the cached prefix KV on CPU
    memory_overhead = cached_tokens * per_token_kv * layer_num

    return {
        "saved_attn_flops": saved_attn_flops,
        "saved_kv_bytes": saved_kv_bytes,
        "saved_attn_time": saved_attn_time,
        "memory_overhead": memory_overhead,
        "cached_tokens": cached_tokens,
        "new_tokens": new_tokens,
        "total_context": total_ctx,
    }
