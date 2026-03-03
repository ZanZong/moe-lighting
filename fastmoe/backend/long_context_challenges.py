"""Long-context offloading challenges: theoretical analysis and research agenda.

This module provides the theoretical foundation for the research question:

  *In multi-turn dialogue and large-text multi-query (long-context) workloads,
  is CPU attention offloading still necessary and effective?  Under what
  conditions, and via which optimisation techniques, can offloading retain
  a performance or cost advantage?*

Target venue: PPoPP / ASPLOS (systems track).

─────────────────────────────────────────────────────────────────────────────
BACKGROUND: CGOPipe Offloading Invariant
─────────────────────────────────────────────────────────────────────────────

MoE-Lightning's CGOPipe pipeline works because, during decode, BOTH of the
following operations can run in parallel:

    GPU:  [Pre-Attn QKV projection]  →  [Post-Attn MoE FFN]
    CPU:  ─────────────────────────────► [Attention(Q,K_cpu,V_cpu)]

The pipeline delivers high throughput only when:

    T_CPU_attn(ctx) ≤ T_GPU_FFN                          (CGOPipe Invariant)

    where:
      T_CPU_attn(ctx)  = max(ctx·F_attn / CPU_FLOPS,
                             ctx·B_attn / CPU_BDW)    [grows linearly in ctx]
      T_GPU_FFN        = max(F_ffn / GPU_FLOPS,
                             B_ffn / GPU_BDW)         [INDEPENDENT of ctx]

In the original paper (short, fixed context), this invariant holds easily.
In long-context workloads (multi-turn, RAG, document QA), context grows
monotonically and the invariant is violated at a predictable crossover point.

─────────────────────────────────────────────────────────────────────────────
LITERATURE GAP ANALYSIS  (based on published work up to early 2025)
─────────────────────────────────────────────────────────────────────────────

| System / Paper          | Key Idea                     | Gap w.r.t. this work        |
|-------------------------|------------------------------|-----------------------------|
| FlexGen (SOSP'23)       | CPU/disk offload, throughput | Fixed prompt+gen, no turns  |
| vLLM (SOSP'23)          | PagedAttention, GPU KV pool  | GPU-only, no CPU offloading |
| DeepSpeed-MII           | KV offload for dense LLMs    | Dense only, no MoE; no turns|
| MoE-Lightning (ASPLOS'25)| CGOPipe for MoE decode      | Fixed ctx; no multi-turn    |
| StreamingLLM (ICLR'24)  | Sink-token eviction          | Loses history; not offload  |
| InfLLM (arXiv'24)       | Block-level CPU KV retrieval | Dense model; no MoE         |
| H2O (NeurIPS'23)        | Heavy-hitter KV eviction     | GPU-only; approximate only  |
| RadixAttention/SGLang   | Prefix caching (GPU)         | GPU-only KV reuse           |
| SnapKV / PyramidKV      | KV compression for long ctx  | GPU-only; no offloading     |
| KVSharer / MagicPIG     | Cross-layer KV sharing/sparse| No CPU offloading           |
| Mooncake (arXiv'24)     | KV cache disaggregation      | Network-attached, not local |
| Sarathi-Serve           | Chunked prefill scheduling   | No CPU offloading           |

KEY FINDING: No existing system simultaneously addresses:
  (A) CPU attention offloading for MoE models
  (B) Growing multi-turn context management
  (C) Adaptive policy selection based on current context length
  (D) Cross-turn KV cache reuse to eliminate redundant computation

The intersection of (A)+(B)+(C)+(D) represents a clear open research problem.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Tuple

from fastmoe.backend.utils import (
    HardwareConfig,
    attention_bytes,
    attention_flops,
    MLP_flops,
    MLP_bytes,
)
from fastmoe.utils.model_config import ModelConfig

GB = 1 << 30
TB = 1 << 40


# ═══════════════════════════════════════════════════════════════════════════
# §1  CHALLENGE QUANTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class ChallengeMetrics:
    """Quantitative results for all five long-context offloading challenges."""

    context_len: int
    batch_size: int

    # Challenge 1 – CGOPipe invariant
    t_cpu_attn_ms: float        # T_CPU_attn for this context length
    t_gpu_ffn_ms: float         # T_GPU_FFN (constant)
    invariant_violated: bool    # True when T_CPU_attn > T_GPU_FFN
    pipeline_efficiency: float  # T_GPU_FFN / max(T_GPU_FFN, T_CPU_attn) ∈ (0,1]

    # Challenge 2 – multi-turn context growth
    turns_until_violation: int  # number of turns before invariant is violated
    ctx_at_violation: int       # context length (tokens) at that point

    # Challenge 3 – CPU memory pressure
    kv_cpu_bytes: int           # bytes needed on CPU for full KV cache
    cpu_mem_fraction: float     # fraction of total CPU memory consumed

    # Challenge 4 – redundant KV computation across turns
    redundant_compute_fraction: float  # fraction of prefill FLOPs that are redundant
    redundant_bandwidth_bytes: int     # extra CPU bandwidth wasted per decode step

    # Challenge 5 – PCIe transfer does NOT shrink with context
    pcie_transfer_bytes: int    # fixed per-step transfer regardless of ctx
    pcie_fraction_of_total: float  # PCIe time / total layer time


def quantify_challenges(
    context_len: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
    turn_tokens: int = 128,
    num_turns: int = 20,
) -> ChallengeMetrics:
    """Compute all five challenge metrics for a given context length.

    Parameters
    ----------
    context_len:  Current total context (tokens).
    batch_size:   Number of concurrent decode requests.
    model_config: Architecture configuration.
    hardware_config: Hardware performance numbers.
    tp_size:      Tensor-parallel degree.
    turn_tokens:  Average tokens per conversation turn (prompt + response).
    num_turns:    Maximum number of turns to simulate for Challenge 2.
    """
    nh  = model_config.num_attention_heads // tp_size
    nkv = model_config.num_key_value_heads // tp_size
    hd  = model_config.hidden_size // model_config.num_attention_heads
    h1  = model_config.hidden_size
    h2  = model_config.intermediate_size // tp_size
    topk = model_config.topk
    ne   = model_config.num_local_experts
    L    = model_config.num_hidden_layers

    # ── Challenge 1: CGOPipe invariant ──────────────────────────────────
    attn_flops = attention_flops(batch_size, 1, context_len, nh, hd)
    attn_bytes = attention_bytes(batch_size, 1, context_len, nh, nkv, hd)
    t_cpu = max(attn_flops / hardware_config.cpu_flops,
                attn_bytes / hardware_config.c_bdw)

    ffn_flops = MLP_flops(h1, h2, batch_size, topk)
    ffn_bytes = MLP_bytes(h1, h2, batch_size, ne)
    t_gpu = max(ffn_flops / hardware_config.gpu_flops,
                ffn_bytes / hardware_config.g_bdw)

    violated = t_cpu > t_gpu
    efficiency = t_gpu / max(t_gpu, t_cpu)  # pipeline utilisation

    # ── Challenge 2: turns until invariant violated ──────────────────────
    ctx_sim = context_len
    turns_until = 0
    ctx_at_viol = context_len
    for turn in range(num_turns):
        a_f = attention_flops(batch_size, 1, ctx_sim, nh, hd)
        a_b = attention_bytes(batch_size, 1, ctx_sim, nh, nkv, hd)
        t_c = max(a_f / hardware_config.cpu_flops,
                  a_b / hardware_config.c_bdw)
        if t_c > t_gpu:
            ctx_at_viol = ctx_sim
            break
        turns_until += 1
        ctx_sim += turn_tokens
    else:
        ctx_at_viol = ctx_sim  # never violated in simulation

    # ── Challenge 3: CPU memory pressure ────────────────────────────────
    # Full KV cache: tokens × 2(K+V) × nkv × hd × 2bytes × L layers × bs
    per_token_kv = 2 * nkv * hd * 2  # bytes per token per layer
    kv_cpu = batch_size * context_len * per_token_kv * L
    cpu_frac = kv_cpu / hardware_config.cmem

    # ── Challenge 4: redundant computation ──────────────────────────────
    # In multi-turn, the "prefix" (all previous turns) is re-prefilled.
    # Assume current turn adds `turn_tokens` new tokens; the rest is redundant.
    prefix_len = max(0, context_len - turn_tokens)
    if context_len > 0:
        redundant_compute = prefix_len / context_len
        # Each decode step: CPU re-reads all prefix KV pairs unnecessarily
        # (without prefix caching, we need the full KV for attention)
        redundant_bw = batch_size * prefix_len * per_token_kv
    else:
        redundant_compute = 0.0
        redundant_bw = 0

    # ── Challenge 5: PCIe transfer overhead ─────────────────────────────
    # Per-layer decode: offload QKV (GPU→CPU) and load hidden (CPU→GPU).
    # This is FIXED regardless of context: only current-token QKV moves.
    qkv_bytes = batch_size * (nh + 2 * nkv) * hd * 2   # fp16
    hidden_bytes = batch_size * nh * hd * 2
    pcie_bytes = qkv_bytes + hidden_bytes
    t_pcie = pcie_bytes / hardware_config.ctog_bdw
    t_layer_total = max(t_gpu, t_cpu, t_pcie)
    pcie_frac = t_pcie / t_layer_total if t_layer_total > 0 else 0.0

    return ChallengeMetrics(
        context_len=context_len,
        batch_size=batch_size,
        t_cpu_attn_ms=t_cpu * 1e3,
        t_gpu_ffn_ms=t_gpu * 1e3,
        invariant_violated=violated,
        pipeline_efficiency=efficiency,
        turns_until_violation=turns_until,
        ctx_at_violation=ctx_at_viol,
        kv_cpu_bytes=kv_cpu,
        cpu_mem_fraction=cpu_frac,
        redundant_compute_fraction=redundant_compute,
        redundant_bandwidth_bytes=redundant_bw,
        pcie_transfer_bytes=pcie_bytes,
        pcie_fraction_of_total=pcie_frac,
    )


# ═══════════════════════════════════════════════════════════════════════════
# §2  MULTI-TURN CONTEXT GROWTH SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TurnMetrics:
    """Metrics for a single conversation turn."""
    turn_id: int
    context_len: int          # cumulative context at this turn
    t_cpu_attn_ms: float
    t_gpu_ffn_ms: float
    invariant_holds: bool
    pipeline_efficiency: float
    kv_cpu_gb: float
    redundant_compute_pct: float


def simulate_multi_turn(
    initial_context: int,
    prompt_per_turn: int,
    response_per_turn: int,
    num_turns: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> List[TurnMetrics]:
    """Simulate multi-turn context growth and track offloading health.

    This is the core experiment for Challenge 1 + 2: shows HOW QUICKLY the
    CGOPipe invariant is violated as a conversation progresses.

    Parameters
    ----------
    initial_context:    Context length at turn 0 (e.g. system prompt length).
    prompt_per_turn:    Average user-side tokens added each turn.
    response_per_turn:  Average model-side tokens generated each turn.
    num_turns:          Total conversation turns to simulate.
    """
    nh  = model_config.num_attention_heads // tp_size
    nkv = model_config.num_key_value_heads // tp_size
    hd  = model_config.hidden_size // model_config.num_attention_heads
    h1  = model_config.hidden_size
    h2  = model_config.intermediate_size // tp_size
    topk = model_config.topk
    ne   = model_config.num_local_experts
    L    = model_config.num_hidden_layers

    ffn_flops = MLP_flops(h1, h2, batch_size, topk)
    ffn_bytes = MLP_bytes(h1, h2, batch_size, ne)
    t_gpu = max(ffn_flops / hardware_config.gpu_flops,
                ffn_bytes / hardware_config.g_bdw)

    per_token_kv = 2 * nkv * hd * 2  # bytes per token per layer

    results: List[TurnMetrics] = []
    ctx = initial_context

    for turn in range(num_turns):
        attn_f = attention_flops(batch_size, 1, ctx, nh, hd)
        attn_b = attention_bytes(batch_size, 1, ctx, nh, nkv, hd)
        t_cpu = max(attn_f / hardware_config.cpu_flops,
                    attn_b / hardware_config.c_bdw)

        kv_cpu_gb = (batch_size * ctx * per_token_kv * L) / GB
        prefix = ctx - initial_context
        redundant_pct = (prefix / ctx * 100) if ctx > 0 else 0.0

        results.append(TurnMetrics(
            turn_id=turn,
            context_len=ctx,
            t_cpu_attn_ms=t_cpu * 1e3,
            t_gpu_ffn_ms=t_gpu * 1e3,
            invariant_holds=(t_cpu <= t_gpu),
            pipeline_efficiency=t_gpu / max(t_gpu, t_cpu),
            kv_cpu_gb=kv_cpu_gb,
            redundant_compute_pct=redundant_pct,
        ))

        # Context grows each turn: new user prompt + model response
        ctx += prompt_per_turn + response_per_turn

    return results


# ═══════════════════════════════════════════════════════════════════════════
# §3  PROPOSED TECHNICAL SOLUTIONS – PRELIMINARY COST ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class SolutionAnalysis:
    """Projected latency gains from each proposed technical solution."""

    name: str
    description: str

    # For a specific context_len × batch_size operating point
    context_len: int
    batch_size: int

    # Baseline (vanilla CGOPipe)
    baseline_layer_latency_ms: float

    # Projected layer latency with this solution
    projected_layer_latency_ms: float

    # Projected speedup (baseline / projected)
    speedup: float

    # Complexity / feasibility notes
    implementation_notes: str


def analyze_solution_cao(
    context_len: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> SolutionAnalysis:
    """Solution 1 – Context-Adaptive Offloading (CAO).

    Switch attention execution between CPU and GPU based on current context
    length and the CGOPipe invariant.  When context exceeds the break-even
    point, route attention to GPU instead of CPU.

    Novel aspect vs. prior work:
      * MoE-Lightning always uses CPU attention (static policy).
      * CAO adds a DYNAMIC policy that re-evaluates every step.
      * The break-even predictor uses the same HRM cost model.
      * No prior system applies this to MoE with CPU offloading.
    """
    from fastmoe.backend.long_context_policy import (
        analyze_long_context, AttentionStrategy
    )
    a = analyze_long_context(context_len, batch_size, model_config,
                             hardware_config, tp_size)

    nh  = model_config.num_attention_heads // tp_size
    nkv = model_config.num_key_value_heads // tp_size
    hd  = model_config.hidden_size // model_config.num_attention_heads
    h1  = model_config.hidden_size
    h2  = model_config.intermediate_size // tp_size
    ffn_flops = MLP_flops(h1, h2, batch_size, model_config.topk)
    ffn_bytes = MLP_bytes(h1, h2, batch_size, model_config.num_local_experts)
    t_gpu_ffn = max(ffn_flops / hardware_config.gpu_flops,
                    ffn_bytes / hardware_config.g_bdw)

    # Baseline: always CPU offload
    baseline_ms = max(t_gpu_ffn, a.cpu_attn_time) * 1e3

    # CAO: choose best
    projected_ms = a.recommended_latency * 1e3
    speedup = baseline_ms / projected_ms if projected_ms > 0 else 1.0

    return SolutionAnalysis(
        name="CAO (Context-Adaptive Offloading)",
        description=(
            "Dynamically switch attention to GPU when context exceeds "
            "the break-even threshold.  Eliminates stalls caused by "
            "CPU-attention becoming the critical path."
        ),
        context_len=context_len,
        batch_size=batch_size,
        baseline_layer_latency_ms=baseline_ms,
        projected_layer_latency_ms=projected_ms,
        speedup=speedup,
        implementation_notes=(
            "Requires: (a) per-step context-length check, (b) GPU KV pool "
            "fallback, (c) no policy re-optimisation – pure routing change."
        ),
    )


def analyze_solution_tkvp(
    context_len: int,
    cached_prefix_len: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> SolutionAnalysis:
    """Solution 2 – Turn-level KV cache Persistence (TKVP).

    Cache the KV pairs of completed turns on CPU and skip re-processing the
    shared prefix during the next turn's prefill phase.

    In the original CGOPipe design:
      * Each new turn re-prefills ALL tokens (history + new prompt) from scratch.
      * Cost: O(full_context × h1 × h2) per layer, dominated by KV write-back.

    With TKVP:
      * History KV is already on CPU from the previous turn.
      * Only NEW tokens (current prompt + small delta) need prefill compute.
      * Cost: O(delta_tokens × h1 × h2) per layer.

    Novel aspect vs. prior work:
      * RadixAttention/SGLang does GPU-side prefix caching but discards CPU KV.
      * TKVP stores KV on CPU across turns, specifically enabling the MoE-
        Lightning offload path for turns beyond the first.
      * Interaction with CGOPipe: CPU prefill KV load can overlap GPU FFN.
    """
    nh  = model_config.num_attention_heads // tp_size
    nkv = model_config.num_key_value_heads // tp_size
    hd  = model_config.hidden_size // model_config.num_attention_heads
    h1  = model_config.hidden_size
    h2  = model_config.intermediate_size // tp_size

    new_tokens = context_len - cached_prefix_len

    # Baseline prefill time: process ALL tokens
    baseline_attn_f = attention_flops(batch_size, context_len, context_len, nh, hd)
    baseline_attn_b = attention_bytes(batch_size, context_len, context_len, nh, nkv, hd)
    baseline_ms = max(baseline_attn_f / hardware_config.gpu_flops,
                      baseline_attn_b / hardware_config.g_bdw) * 1e3

    # TKVP: only process new tokens; prefix KV is loaded from CPU cache
    new_attn_f = attention_flops(batch_size, new_tokens, context_len, nh, hd)
    new_attn_b = attention_bytes(batch_size, new_tokens, context_len, nh, nkv, hd)
    # Add cost of loading cached KV from CPU
    cached_kv_load_b = batch_size * cached_prefix_len * 2 * nkv * hd * 2
    t_cached_load = cached_kv_load_b / hardware_config.c_bdw
    projected_ms = (max(new_attn_f / hardware_config.gpu_flops,
                        new_attn_b / hardware_config.g_bdw)
                    + t_cached_load) * 1e3

    speedup = baseline_ms / projected_ms if projected_ms > 0 else 1.0

    return SolutionAnalysis(
        name="TKVP (Turn-level KV Persistence)",
        description=(
            "Store completed-turn KV caches on CPU; skip re-processing "
            "shared history during prefill of subsequent turns.  Only the "
            "delta (new prompt tokens) requires full prefill compute."
        ),
        context_len=context_len,
        batch_size=batch_size,
        baseline_layer_latency_ms=baseline_ms,
        projected_layer_latency_ms=projected_ms,
        speedup=speedup,
        implementation_notes=(
            "Requires: (a) session-ID tracking in Req, (b) TokenToKVPool "
            "save_session / load_session (already implemented), "
            "(c) partial-prefill mode that starts from cached KV position."
        ),
    )


def analyze_solution_hace(
    context_len: int,
    hot_window: int,
    top_k_cold: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> SolutionAnalysis:
    """Solution 3 – Hierarchical Attention with CPU-side KV Eviction (HACE).

    Divide the KV cache into two tiers:
      * HOT tier  (GPU): recent `hot_window` tokens – always attended to.
      * COLD tier (CPU): older tokens – only top-k selected via importance score.

    CPU attention cost drops from O(context_len) to O(hot_window + top_k_cold),
    directly extending the break-even threshold.

    Novel aspect vs. prior work:
      * H2O / SnapKV evict KV entirely (lose information) for GPU-only attention.
      * HACE keeps ALL KV on CPU but only FETCHES a selected subset for attention,
        avoiding both eviction loss and the O(n) CPU bandwidth bottleneck.
      * The GPU hot tier allows exact recent-token attention without PCIe cost.
      * Uniquely suited for MoE: the CPU cold tier naturally co-locates with
        the MoE expert weights already stored on CPU.
    """
    nh  = model_config.num_attention_heads // tp_size
    nkv = model_config.num_key_value_heads // tp_size
    hd  = model_config.hidden_size // model_config.num_attention_heads
    h1  = model_config.hidden_size
    h2  = model_config.intermediate_size // tp_size
    ffn_flops = MLP_flops(h1, h2, batch_size, model_config.topk)
    ffn_bytes = MLP_bytes(h1, h2, batch_size, model_config.num_local_experts)
    t_gpu_ffn = max(ffn_flops / hardware_config.gpu_flops,
                    ffn_bytes / hardware_config.g_bdw)

    # Baseline: full CPU attention over entire context
    attn_f = attention_flops(batch_size, 1, context_len, nh, hd)
    attn_b = attention_bytes(batch_size, 1, context_len, nh, nkv, hd)
    t_cpu_full = max(attn_f / hardware_config.cpu_flops,
                     attn_b / hardware_config.c_bdw)
    baseline_ms = max(t_gpu_ffn, t_cpu_full) * 1e3

    # HACE: only attend to hot_window + top_k_cold tokens
    effective_ctx = hot_window + top_k_cold
    a_f = attention_flops(batch_size, 1, effective_ctx, nh, hd)
    a_b = attention_bytes(batch_size, 1, effective_ctx, nh, nkv, hd)
    # Hot window on GPU – fast; cold top-k fetched from CPU
    t_hot_gpu = max(
        attention_flops(batch_size, 1, hot_window, nh, hd) / hardware_config.gpu_flops,
        attention_bytes(batch_size, 1, hot_window, nh, nkv, hd) / hardware_config.g_bdw,
    )
    cold_fetch_b = batch_size * top_k_cold * 2 * nkv * hd * 2
    t_cold_cpu = cold_fetch_b / hardware_config.c_bdw
    t_cpu_hace = t_cold_cpu  # cold attention on CPU
    projected_ms = max(t_gpu_ffn + t_hot_gpu, t_cpu_hace) * 1e3

    speedup = baseline_ms / projected_ms if projected_ms > 0 else 1.0

    return SolutionAnalysis(
        name="HACE (Hierarchical Attention with CPU-side KV Eviction)",
        description=(
            f"Split KV into GPU hot-window ({hot_window} tokens) + CPU cold "
            f"top-k ({top_k_cold} tokens).  Attend exactly over hot tokens; "
            "use sparse CPU attention over selected cold tokens.  Reduces "
            f"effective CPU attention length from {context_len} to "
            f"{hot_window + top_k_cold}."
        ),
        context_len=context_len,
        batch_size=batch_size,
        baseline_layer_latency_ms=baseline_ms,
        projected_layer_latency_ms=projected_ms,
        speedup=speedup,
        implementation_notes=(
            "Requires: (a) importance-score routing (e.g. query-key dot "
            "product on lightweight head), (b) GPU hot-KV buffer separate "
            "from CPU cold-KV pool, (c) non-contiguous CPU KV gather."
        ),
    )


def analyze_solution_iskc(
    context_len: int,
    quantization_bits: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> SolutionAnalysis:
    """Solution 4 – Intra-Session KV Compression (ISKC).

    After each turn, compress the CPU-resident KV cache of that turn.
    INT8 quantization halves bandwidth; INT4 quarters it.  This directly
    reduces T_CPU_attn (bandwidth-bound) and extends the break-even point.

    T_CPU_attn_compressed = T_CPU_attn_fp16 × (quantization_bits / 16)

    Novel aspect vs. prior work:
      * KV quantization (KIVI, WKVQuant) targets GPU memory reduction.
      * ISKC targets CPU *bandwidth* reduction to restore the CGOPipe
        invariant for sessions that have grown beyond the break-even point.
      * Older turns can be compressed more aggressively (INT4) than recent
        turns (INT8 or FP16), giving an adaptive per-turn compression level.
    """
    nh  = model_config.num_attention_heads // tp_size
    nkv = model_config.num_key_value_heads // tp_size
    hd  = model_config.hidden_size // model_config.num_attention_heads
    h1  = model_config.hidden_size
    h2  = model_config.intermediate_size // tp_size
    ffn_flops = MLP_flops(h1, h2, batch_size, model_config.topk)
    ffn_bytes = MLP_bytes(h1, h2, batch_size, model_config.num_local_experts)
    t_gpu_ffn = max(ffn_flops / hardware_config.gpu_flops,
                    ffn_bytes / hardware_config.g_bdw)

    # Baseline: FP16 KV
    attn_f = attention_flops(batch_size, 1, context_len, nh, hd)
    attn_b_fp16 = attention_bytes(batch_size, 1, context_len, nh, nkv, hd, dtype="f16")
    t_cpu_fp16 = max(attn_f / hardware_config.cpu_flops,
                     attn_b_fp16 / hardware_config.c_bdw)
    baseline_ms = max(t_gpu_ffn, t_cpu_fp16) * 1e3

    # ISKC: quantized KV
    assert quantization_bits in (4, 8), "Only INT4/INT8 quantization supported"
    dtype_str = "int4" if quantization_bits == 4 else "int8"
    attn_b_quant = attention_bytes(batch_size, 1, context_len, nh, nkv, hd,
                                   dtype=dtype_str)
    t_cpu_quant = max(attn_f / hardware_config.cpu_flops,
                      attn_b_quant / hardware_config.c_bdw)
    projected_ms = max(t_gpu_ffn, t_cpu_quant) * 1e3

    speedup = baseline_ms / projected_ms if projected_ms > 0 else 1.0

    return SolutionAnalysis(
        name=f"ISKC (Intra-Session KV Compression, INT{quantization_bits})",
        description=(
            f"Quantize CPU-resident KV cache to INT{quantization_bits} after "
            "each turn.  Reduces CPU memory bandwidth by "
            f"{16 // quantization_bits}×, restoring the CGOPipe invariant "
            "for longer contexts without changing the attention algorithm."
        ),
        context_len=context_len,
        batch_size=batch_size,
        baseline_layer_latency_ms=baseline_ms,
        projected_layer_latency_ms=projected_ms,
        speedup=speedup,
        implementation_notes=(
            "Requires: (a) quantization kernel for CPU tensors, "
            "(b) dequantization before attention (or fused dequant-attn), "
            "(c) per-turn compression-level policy (older = more aggressive)."
        ),
    )


def analyze_solution_aro(
    initial_context: int,
    current_context: int,
    batch_size: int,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    tp_size: int = 1,
) -> SolutionAnalysis:
    """Solution 5 – Adaptive Re-Optimisation (ARO).

    The MoE-Lightning optimizer finds one (ubs, n_ub) policy at startup for
    avg_prompt_len.  As context grows, this policy becomes suboptimal because:
      * The CGOPipe invariant may be violated (fewer micro-batches needed).
      * The GPU KV buffer sizing assumed small context.
      * The CPU KV pool may be overflowing.

    ARO re-runs the optimizer (solve_lp) periodically when context exceeds
    a threshold, deriving a new policy tuned for the current context length.

    Novel aspect vs. prior work:
      * Static policy optimisation is universal in existing systems.
      * ARO introduces context-triggered re-optimisation specifically for
        the multi-turn growth pattern, which is predictable and smooth.
      * The re-optimisation can be done asynchronously on a background thread.
    """
    from fastmoe.backend.long_context_policy import (
        analyze_long_context, AttentionStrategy
    )
    nh  = model_config.num_attention_heads // tp_size
    nkv = model_config.num_key_value_heads // tp_size
    hd  = model_config.hidden_size // model_config.num_attention_heads
    h1  = model_config.hidden_size
    h2  = model_config.intermediate_size // tp_size
    ffn_flops = MLP_flops(h1, h2, batch_size, model_config.topk)
    ffn_bytes = MLP_bytes(h1, h2, batch_size, model_config.num_local_experts)
    t_gpu_ffn = max(ffn_flops / hardware_config.gpu_flops,
                    ffn_bytes / hardware_config.g_bdw)

    # Old policy (computed for initial_context) applied to current_context
    a_old = analyze_long_context(current_context, batch_size, model_config,
                                 hardware_config, tp_size)
    baseline_ms = max(t_gpu_ffn, a_old.cpu_attn_time) * 1e3

    # New policy (optimised for current_context)
    a_new = analyze_long_context(current_context, batch_size, model_config,
                                 hardware_config, tp_size)
    projected_ms = a_new.recommended_latency * 1e3

    speedup = baseline_ms / projected_ms if projected_ms > 0 else 1.0

    return SolutionAnalysis(
        name="ARO (Adaptive Re-Optimisation)",
        description=(
            f"Re-run the HRM optimizer when context grows from "
            f"{initial_context} → {current_context} tokens.  Derives "
            "a new (ubs, n_ub, attention_mode) policy for the current "
            "operating point."
        ),
        context_len=current_context,
        batch_size=batch_size,
        baseline_layer_latency_ms=baseline_ms,
        projected_layer_latency_ms=projected_ms,
        speedup=speedup,
        implementation_notes=(
            "Requires: (a) context-length monitor in ExecutionEngine, "
            "(b) background thread running solve_lp, (c) graceful policy "
            "hot-swap without dropping in-flight batches."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# §4  COMBINED RESEARCH SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def research_summary(
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    context_lengths: List[int] = None,
    batch_sizes: List[int] = None,
    tp_size: int = 1,
) -> dict:
    """Produce a structured summary of all challenges and solution projections.

    This is the entry point for the benchmark script and serves as the
    empirical basis for the research paper's §3 (Motivation) section.

    Returns a dict with keys:
      ``challenges``:   list of ChallengeMetrics across context lengths.
      ``multi_turn``:   TurnMetrics list for a typical long conversation.
      ``solutions``:    SolutionAnalysis for each proposed technique.
    """
    if context_lengths is None:
        context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]

    challenges = [
        quantify_challenges(ctx, 1, model_config, hardware_config, tp_size)
        for ctx in context_lengths
    ]

    # Simulate a 20-turn conversation starting from a 512-token system prompt
    multi_turn = simulate_multi_turn(
        initial_context=512,
        prompt_per_turn=200,
        response_per_turn=100,
        num_turns=20,
        batch_size=1,
        model_config=model_config,
        hardware_config=hardware_config,
        tp_size=tp_size,
    )

    # Evaluate each solution at a representative long-context operating point
    long_ctx = 16384
    bs = 1
    solutions = [
        analyze_solution_cao(long_ctx, bs, model_config, hardware_config, tp_size),
        analyze_solution_tkvp(long_ctx, long_ctx // 2, bs, model_config,
                               hardware_config, tp_size),
        analyze_solution_hace(long_ctx, 2048, 512, bs, model_config,
                               hardware_config, tp_size),
        analyze_solution_iskc(long_ctx, 8, bs, model_config, hardware_config,
                               tp_size),
        analyze_solution_iskc(long_ctx, 4, bs, model_config, hardware_config,
                               tp_size),
        analyze_solution_aro(512, long_ctx, bs, model_config, hardware_config,
                              tp_size),
    ]

    return {
        "challenges": challenges,
        "multi_turn": multi_turn,
        "solutions": solutions,
    }
