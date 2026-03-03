"""Adaptive offloading policy for long-context scenarios.

In multi-turn dialogue and large-text multi-query settings, as the context
length grows, the CPU attention computation time increases quadratically
while the GPU FFN time (which is parallelized with CPU attention via
CGOPipe) may no longer hide the CPU latency.  This module provides:

  1. A break-even analysis that computes the context-length threshold
     beyond which CPU offloading is no longer beneficial.
  2. An adaptive strategy selector that returns the recommended attention
     execution mode for a given context length.
  3. A throughput model that estimates the per-step decode throughput for
     long-context workloads under different offloading decisions.

These tools allow downstream components (ExecutionEngine, optimizer) to
make data-driven decisions instead of always offloading attention.
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum, auto
from typing import Optional

from fastmoe.backend.utils import (
    HardwareConfig,
    attention_bytes,
    attention_flops,
    MLP_flops,
    MLP_bytes,
)
from fastmoe.utils.model_config import ModelConfig


class AttentionStrategy(Enum):
    """Recommended attention computation strategy."""

    CPU_OFFLOAD = auto()   # Classic CGOPipe: attention runs on CPU
    GPU_COMPUTE = auto()   # Move attention back to GPU (no offloading)
    SPARSE_CPU = auto()    # Sparse/approximate CPU attention for very long ctx


@dataclasses.dataclass
class LongContextAnalysis:
    """Result of a long-context offloading analysis for one configuration."""

    context_len: int
    batch_size: int
    strategy: AttentionStrategy

    # Time estimates (seconds) for one decode layer
    cpu_attn_time: float
    gpu_attn_time: float
    gpu_ffn_time: float

    # End-to-end per-layer latency under each strategy
    latency_cpu_offload: float   # max(gpu_ffn, cpu_attn, ctog)
    latency_gpu_only: float      # gpu_ffn + gpu_attn (sequential on GPU)
    latency_sparse_cpu: float    # max(gpu_ffn, sparse_cpu_attn, ctog)

    # KV cache sizes
    kv_cache_gpu_bytes: int
    kv_cache_cpu_bytes: int

    # Whether the current hardware can hold the full KV cache on GPU
    fits_on_gpu: bool

    @property
    def recommended_latency(self) -> float:
        if self.strategy == AttentionStrategy.CPU_OFFLOAD:
            return self.latency_cpu_offload
        elif self.strategy == AttentionStrategy.GPU_COMPUTE:
            return self.latency_gpu_only
        else:
            return self.latency_sparse_cpu


def _cpu_attention_time(
    context_len: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> float:
    """Estimate CPU attention time for a single decode step (one layer)."""
    num_q_heads = model_config.num_attention_heads // tp_size
    num_kv_heads = model_config.num_key_value_heads // tp_size
    head_dim = model_config.hidden_size // model_config.num_attention_heads

    flops = attention_flops(batch_size, 1, context_len, num_q_heads, head_dim)
    bw_bytes = attention_bytes(
        batch_size, 1, context_len, num_q_heads, num_kv_heads, head_dim
    )
    comp_time = flops / hardware_config.cpu_flops
    mem_time = bw_bytes / hardware_config.c_bdw
    return max(comp_time, mem_time)


def _gpu_attention_time(
    context_len: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> float:
    """Estimate GPU attention time for a single decode step (one layer)."""
    num_q_heads = model_config.num_attention_heads // tp_size
    num_kv_heads = model_config.num_key_value_heads // tp_size
    head_dim = model_config.hidden_size // model_config.num_attention_heads

    flops = attention_flops(batch_size, 1, context_len, num_q_heads, head_dim)
    bw_bytes = attention_bytes(
        batch_size, 1, context_len, num_q_heads, num_kv_heads, head_dim
    )
    comp_time = flops / hardware_config.gpu_flops
    mem_time = bw_bytes / hardware_config.g_bdw
    return max(comp_time, mem_time)


def _gpu_ffn_time(
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> float:
    """Estimate GPU FFN (MoE) time for one decode layer with one micro-batch."""
    h1 = model_config.hidden_size
    h2 = model_config.intermediate_size // tp_size
    topk = model_config.topk
    num_experts = model_config.num_local_experts

    flops = MLP_flops(h1, h2, batch_size, topk)
    bw_bytes = MLP_bytes(h1, h2, batch_size, num_experts)
    comp_time = flops / hardware_config.gpu_flops
    mem_time = bw_bytes / hardware_config.g_bdw
    return max(comp_time, mem_time)


def _ctog_transfer_time(
    context_len: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> float:
    """Estimate CPU→GPU KV cache transfer time for attention output."""
    num_kv_heads = model_config.num_key_value_heads // tp_size
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    # QKV offload + hidden reload per step
    kv_bytes = 2 * batch_size * context_len * num_kv_heads * head_dim * 2  # fp16
    return kv_bytes / hardware_config.ctog_bdw


def _kv_cache_bytes(
    context_len: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> tuple[int, int]:
    """Return (gpu_kv_bytes, cpu_kv_bytes) for the full KV cache."""
    num_kv_heads = model_config.num_key_value_heads // tp_size
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    num_layers = model_config.num_hidden_layers
    # 2 = key + value; 2 bytes for fp16
    per_token = 2 * num_kv_heads * head_dim * 2
    total = batch_size * context_len * per_token * num_layers
    # GPU only holds a double-buffered working set in the current design
    gpu_kv = 2 * batch_size * context_len * per_token
    cpu_kv = total
    return gpu_kv, cpu_kv


def _sparse_attention_time(
    context_len: int,
    batch_size: int,
    sparsity: float,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> float:
    """Estimate CPU sparse-attention time using a fixed top-k token sparsity.

    The effective attended context length is ``context_len * (1 - sparsity)``.
    """
    effective_len = max(1, int(context_len * (1.0 - sparsity)))
    return _cpu_attention_time(
        effective_len, batch_size, model_config, hardware_config, tp_size
    )


def compute_offload_threshold(
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    batch_size: int = 1,
    tp_size: int = 1,
) -> int:
    """Binary-search for the context length at which CPU offloading breaks even.

    Returns the context length (in tokens) beyond which CPU offloading is
    *slower* than GPU-only attention (i.e., CPU attention time exceeds the
    GPU FFN time that would otherwise hide it).

    A context shorter than this threshold benefits from CGOPipe offloading.
    A context longer than this threshold is better served by GPU attention.
    """
    gpu_ffn = _gpu_ffn_time(batch_size, model_config, hardware_config, tp_size)

    lo, hi = 1, 1
    # Find an upper bound
    while _cpu_attention_time(hi, batch_size, model_config, hardware_config, tp_size) <= gpu_ffn:
        hi *= 2
        if hi > 1_000_000:
            # CPU attention always fits — no crossover point in practical range
            return hi

    # Binary search
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if _cpu_attention_time(mid, batch_size, model_config, hardware_config, tp_size) <= gpu_ffn:
            lo = mid
        else:
            hi = mid

    return hi


def analyze_long_context(
    context_len: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
    sparse_sparsity: float = 0.75,
) -> LongContextAnalysis:
    """Produce a full analysis for a given context length and batch size.

    Parameters
    ----------
    context_len:      Total KV context length (prompt + generated tokens).
    batch_size:       Number of concurrent decode requests.
    model_config:     Model architecture parameters.
    hardware_config:  CPU/GPU hardware performance parameters.
    tp_size:          Tensor-parallel degree.
    sparse_sparsity:  Fraction of KV tokens dropped in sparse attention mode.
    """
    cpu_attn = _cpu_attention_time(
        context_len, batch_size, model_config, hardware_config, tp_size
    )
    gpu_attn = _gpu_attention_time(
        context_len, batch_size, model_config, hardware_config, tp_size
    )
    gpu_ffn = _gpu_ffn_time(
        batch_size, model_config, hardware_config, tp_size
    )
    ctog = _ctog_transfer_time(
        context_len, batch_size, model_config, hardware_config, tp_size
    )
    sparse_attn = _sparse_attention_time(
        context_len, batch_size, sparse_sparsity,
        model_config, hardware_config, tp_size
    )

    # Per-layer latency under each strategy
    # CGOPipe: GPU does FFN while CPU does attention (and QKV transfer)
    latency_cpu_offload = max(gpu_ffn, cpu_attn, ctog)
    # GPU-only: attention and FFN run sequentially
    latency_gpu_only = gpu_ffn + gpu_attn
    # Sparse CPU: GPU does FFN while CPU does (cheaper) sparse attention
    latency_sparse_cpu = max(gpu_ffn, sparse_attn, ctog)

    # KV cache footprint
    gpu_kv, cpu_kv = _kv_cache_bytes(
        context_len, batch_size, model_config, hardware_config, tp_size
    )
    fits_on_gpu = gpu_kv + cpu_kv <= hardware_config.gmem * 0.5  # rough check

    # Choose best strategy
    if fits_on_gpu:
        # If everything fits on GPU, avoid the PCIe overhead entirely
        strategy = AttentionStrategy.GPU_COMPUTE
    elif latency_cpu_offload <= latency_gpu_only and latency_cpu_offload <= latency_sparse_cpu:
        strategy = AttentionStrategy.CPU_OFFLOAD
    elif latency_sparse_cpu <= latency_gpu_only:
        strategy = AttentionStrategy.SPARSE_CPU
    else:
        strategy = AttentionStrategy.GPU_COMPUTE

    return LongContextAnalysis(
        context_len=context_len,
        batch_size=batch_size,
        strategy=strategy,
        cpu_attn_time=cpu_attn,
        gpu_attn_time=gpu_attn,
        gpu_ffn_time=gpu_ffn,
        latency_cpu_offload=latency_cpu_offload,
        latency_gpu_only=latency_gpu_only,
        latency_sparse_cpu=latency_sparse_cpu,
        kv_cache_gpu_bytes=gpu_kv,
        kv_cache_cpu_bytes=cpu_kv,
        fits_on_gpu=fits_on_gpu,
    )


def sweep_context_lengths(
    context_lengths: list[int],
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
    sparse_sparsity: float = 0.75,
) -> list[LongContextAnalysis]:
    """Run :func:`analyze_long_context` for a list of context lengths."""
    return [
        analyze_long_context(
            ctx, batch_size, model_config, hardware_config,
            tp_size, sparse_sparsity
        )
        for ctx in context_lengths
    ]


def analyze_long_context_offloading(
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    context_lengths: Optional[list] = None,
    batch_size: int = 1,
) -> list:
    """Analyse when CPU offloading remains beneficial as context grows.

    For each context length this function computes the per-layer decode
    latency under:

    * **CPU offload** (CGOPipe): GPU runs FFN while CPU runs attention.
    * **GPU only**: Both attention and FFN run sequentially on GPU.

    Returns a list of dicts, one per context length, containing timing
    breakdowns and the recommended strategy.

    This entry-point is used by the long-context benchmark in
    ``benchmarks/long_context/bench_long_context.py``.
    """
    GB = 1 << 30
    if context_lengths is None:
        context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    results = []
    for ctx in context_lengths:
        a = analyze_long_context(
            ctx, batch_size, model_config, hardware_config,
            tp_size=hardware_config.tp_size,
        )
        results.append({
            "context_len": ctx,
            "batch_size": batch_size,
            "strategy": a.strategy.name,
            "cpu_attn_time_ms": a.cpu_attn_time * 1e3,
            "gpu_attn_time_ms": a.gpu_attn_time * 1e3,
            "gpu_ffn_time_ms": a.gpu_ffn_time * 1e3,
            "latency_cpu_offload_ms": a.latency_cpu_offload * 1e3,
            "latency_gpu_only_ms": a.latency_gpu_only * 1e3,
            "latency_sparse_cpu_ms": a.latency_sparse_cpu * 1e3,
            "kv_cache_cpu_gb": a.kv_cache_cpu_bytes / GB,
            "fits_on_gpu": a.fits_on_gpu,
        })
    return results
