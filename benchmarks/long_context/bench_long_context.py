"""Long-context offloading analysis benchmark.

This script sweeps over a range of context lengths and reports the
estimated per-layer decode latency and recommended attention strategy for
a given model / hardware configuration.

Usage example (no GPU required – runs the analytical model only)::

    python bench_long_context.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --batch-size 8 \\
        --gpu-mem 24 \\
        --cpu-mem 192 \\
        --cpu-bdw 76 \\
        --output-file results.jsonl

The benchmark also computes the **break-even context length**: the token
count beyond which offloading attention to CPU degrades end-to-end latency
compared to running attention on GPU.

For multi-turn workloads it additionally estimates the **prefix-reuse
savings**: how much compute and bandwidth can be saved by caching the KV
pairs of previous conversation turns.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastmoe.backend.long_context_policy import (
    analyze_long_context,
    analyze_long_context_offloading,
    compute_offload_threshold,
    sweep_context_lengths,
    AttentionStrategy,
)
from fastmoe.backend.kv_cache_manager import estimate_prefix_reuse_savings
from fastmoe.backend.utils import HardwareConfig
from fastmoe.utils.model_config import ModelConfig

GB = 1 << 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse long-context offloading trade-offs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="HuggingFace model path or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Decode batch size.",
    )
    parser.add_argument(
        "--gpu-mem",
        type=int,
        default=24,
        help="GPU memory in GB.",
    )
    parser.add_argument(
        "--cpu-mem",
        type=int,
        default=192,
        help="CPU memory in GB.",
    )
    parser.add_argument(
        "--cpu-bdw",
        type=float,
        default=76.0,
        help="CPU memory bandwidth in GB/s.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor-parallel degree.",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        help="List of context lengths to evaluate.",
    )
    parser.add_argument(
        "--turn-cached-tokens",
        type=int,
        default=1024,
        help="For multi-turn analysis: number of tokens cached from prior turns.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="If given, write JSONL results to this file.",
    )
    return parser.parse_args()


def print_separator(char="─", width=90):
    print(char * width)


def main():
    args = parse_args()

    # Build configs
    try:
        model_config = ModelConfig(args.model)
    except Exception as exc:
        print(
            f"[warn] Could not load model config from '{args.model}': {exc}\n"
            "       Using Mixtral-8×7B defaults instead."
        )

        class _FakeModelConfig:
            num_hidden_layers = 32
            hidden_size = 4096
            intermediate_size = 14336
            num_attention_heads = 32
            num_key_value_heads = 8
            num_local_experts = 8
            topk = 2
            context_len = 32768
            vocab_size = 32000

        model_config = _FakeModelConfig()

    hw = HardwareConfig(
        gmem=args.gpu_mem * GB,
        cmem=args.cpu_mem * GB,
        ctog_bdw=16 * GB,
        g_bdw=300 * GB,
        c_bdw=int(args.cpu_bdw * GB),
        gpu_flops=104e12,
        cpu_flops=1.6e12,
        tp_size=args.tp_size,
    )

    # ------------------------------------------------------------------ #
    # 1.  Break-even context length                                        #
    # ------------------------------------------------------------------ #
    threshold = compute_offload_threshold(
        model_config, hw, batch_size=args.batch_size, tp_size=args.tp_size
    )
    print()
    print_separator("═")
    print("  Long-Context CPU Offloading Analysis")
    print_separator("═")
    print(f"  Model      : {args.model}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  GPU mem    : {args.gpu_mem} GB")
    print(f"  CPU bdw    : {args.cpu_bdw} GB/s")
    print()
    print(
        f"  ► Break-even context length: {threshold:,} tokens\n"
        f"    (CPU offloading is beneficial for contexts shorter than this)"
    )
    print_separator()

    # ------------------------------------------------------------------ #
    # 2.  Per-context-length breakdown                                     #
    # ------------------------------------------------------------------ #
    print()
    print(f"  {'Context':>10}  {'Strategy':>14}  "
          f"{'CPU-attn(ms)':>14}  {'GPU-ffn(ms)':>12}  "
          f"{'Off-load(ms)':>13}  {'GPU-only(ms)':>13}  "
          f"{'KV CPU(GB)':>11}")
    print_separator()

    all_results = []
    for ctx in args.context_lengths:
        a = analyze_long_context(
            ctx, args.batch_size, model_config, hw, tp_size=args.tp_size
        )
        kv_gb = a.kv_cache_cpu_bytes / GB
        print(
            f"  {ctx:>10,}  {a.strategy.name:>14}  "
            f"{a.cpu_attn_time*1e3:>14.3f}  {a.gpu_ffn_time*1e3:>12.3f}  "
            f"{a.latency_cpu_offload*1e3:>13.3f}  "
            f"{a.latency_gpu_only*1e3:>13.3f}  "
            f"{kv_gb:>11.3f}"
        )
        all_results.append(
            {
                "context_len": ctx,
                "strategy": a.strategy.name,
                "cpu_attn_ms": a.cpu_attn_time * 1e3,
                "gpu_attn_ms": a.gpu_attn_time * 1e3,
                "gpu_ffn_ms": a.gpu_ffn_time * 1e3,
                "latency_cpu_offload_ms": a.latency_cpu_offload * 1e3,
                "latency_gpu_only_ms": a.latency_gpu_only * 1e3,
                "latency_sparse_cpu_ms": a.latency_sparse_cpu * 1e3,
                "kv_cache_cpu_gb": kv_gb,
                "fits_on_gpu": a.fits_on_gpu,
            }
        )

    print_separator()

    # ------------------------------------------------------------------ #
    # 3.  Multi-turn prefix-reuse savings                                  #
    # ------------------------------------------------------------------ #
    print()
    print("  Multi-turn prefix reuse analysis")
    print_separator()
    num_kv_heads = getattr(
        model_config, "num_key_value_heads", 8
    ) // args.tp_size
    head_dim = (
        model_config.hidden_size // model_config.num_attention_heads
    )

    for new_tokens in [64, 128, 256, 512]:
        savings = estimate_prefix_reuse_savings(
            cached_tokens=args.turn_cached_tokens,
            new_tokens=new_tokens,
            batch_size=args.batch_size,
            layer_num=model_config.num_hidden_layers,
            head_num=num_kv_heads,
            head_dim=head_dim,
            cpu_flops=hw.cpu_flops,
            c_bdw=hw.c_bdw,
        )
        print(
            f"  cached={args.turn_cached_tokens:,}  new={new_tokens:>5}  "
            f"saved_attn_time={savings['saved_attn_time']*1e3:.2f}ms  "
            f"mem_overhead={savings['memory_overhead']/1e6:.1f}MB"
        )
    print_separator()

    # ------------------------------------------------------------------ #
    # 4.  Observations & recommendations                                   #
    # ------------------------------------------------------------------ #
    print()
    print("  Key findings:")
    print(
        f"  1. CPU offloading is beneficial up to ~{threshold:,} tokens;\n"
        "     beyond this the CPU attention bottleneck dominates latency."
    )
    print(
        "  2. Sparse CPU attention (75% sparsity) extends the useful range\n"
        "     of CPU offloading at the cost of approximate results."
    )
    print(
        "  3. Multi-turn prefix caching avoids re-computing shared KV\n"
        "     history, reducing both CPU bandwidth and attention FLOPs."
    )
    print()

    # ------------------------------------------------------------------ #
    # 5.  Optional output file                                             #
    # ------------------------------------------------------------------ #
    if args.output_file:
        with open(args.output_file, "w") as f:
            for row in all_results:
                f.write(json.dumps(row) + "\n")
        print(f"  Results written to {args.output_file}")
        print()


if __name__ == "__main__":
    main()
