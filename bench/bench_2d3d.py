from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grouped_mm_bf16 import grouped_mm
from grouped_mm_bf16.kernels import grouped_mm_2d3d_backward


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP required")
    return torch.device("cuda")


def _time_forward(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor, iters: int, warmup: int
) -> float:
    for _ in range(warmup):
        grouped_mm(a, b, offs=offs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        grouped_mm(a, b, offs=offs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters * 1e3


def _time_backward_da_db(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    compute_grad_a: bool,
    compute_grad_b: bool,
    iters: int,
    warmup: int,
) -> float:
    if compute_grad_a:
        grad_a = torch.empty_like(a)
    else:
        grad_a = None

    if compute_grad_b:
        grad_b = torch.empty_like(b)
    else:
        grad_b = None

    for _ in range(warmup):
        grouped_mm_2d3d_backward(a, b, offs, grad_out, grad_a, grad_b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        grouped_mm_2d3d_backward(a, b, offs, grad_out, grad_a, grad_b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=64, help="Number of experts (groups)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Token batch size (used with --active-experts-per-token to generate uneven expert loads)",
    )
    parser.add_argument(
        "--active-experts-per-token",
        type=int,
        default=None,
        help="Top-k routing: number of experts each token is sent to (used with --batch-size)",
    )
    parser.add_argument(
        "--tokens-per-expert",
        type=int,
        default=2048,
        help="Uniform expert load (rows per expert) when routing is not generated",
    )
    parser.add_argument("--embedding-size", type=int, default=32, help="Input feature size (K)")
    parser.add_argument("--output-features", type=int, default=64, help="Output feature size (N)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible benchmarks")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "backward", "both"])
    parser.add_argument("--backward", type=str, default="both", choices=["da", "db", "both"])
    parser.add_argument("--impl", type=str, default="both", choices=["prologue", "rowwise", "both"])
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--rowwise-block-n", type=int, default=None)
    parser.add_argument("--rowwise-block-k", type=int, default=None)
    parser.add_argument("--rowwise-num-warps", type=int, default=None)
    args = parser.parse_args()

    dev = _device()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    torch.manual_seed(int(args.seed))

    experts = args.experts
    embedding_size = args.embedding_size
    output_features = args.output_features

    if (args.batch_size is None) != (args.active_experts_per_token is None):
        raise SystemExit("Provide both --batch-size and --active-experts-per-token, or neither")

    if args.batch_size is not None:
        batch_size = args.batch_size
        active_experts_per_token = args.active_experts_per_token
        if active_experts_per_token > experts:
            raise SystemExit(
                f"--active-experts-per-token ({active_experts_per_token}) cannot exceed --experts ({experts})"
            )
        total_assignments = batch_size * active_experts_per_token
        expert_ids = torch.randint(0, experts, (total_assignments,), device=dev, dtype=torch.int64)
        sizes = torch.bincount(expert_ids, minlength=experts).to(torch.int32)
    else:
        tokens_per_expert = args.tokens_per_expert
        sizes = torch.full((experts,), tokens_per_expert, device=dev, dtype=torch.int32)

    offs = sizes.cumsum(0).to(torch.int32)
    total_routed_tokens = int(offs[-1].item())

    a = torch.randn((total_routed_tokens, embedding_size), device=dev, dtype=dtype)
    b = torch.randn((experts, embedding_size, output_features), device=dev, dtype=dtype)
    grad_out = torch.randn((total_routed_tokens, output_features), device=dev, dtype=dtype)

    impls = [args.impl] if args.impl != "both" else ["rowwise", "prologue"]
    for impl in impls:
        for key in (
            "GROUPED_MM_2D3D_IMPL",
            "GROUPED_MM_2D3D_BLOCK_M",
            "GROUPED_MM_2D3D_BLOCK_N",
            "GROUPED_MM_2D3D_BLOCK_K",
            "GROUPED_MM_2D3D_NUM_WARPS",
            "GROUPED_MM_2D3D_ROWWISE_BLOCK_N",
            "GROUPED_MM_2D3D_ROWWISE_BLOCK_K",
            "GROUPED_MM_2D3D_ROWWISE_NUM_WARPS",
        ):
            os.environ.pop(key, None)

        os.environ["GROUPED_MM_2D3D_IMPL"] = impl
        if impl == "tiled":
            if args.block_m is not None:
                os.environ["GROUPED_MM_2D3D_BLOCK_M"] = str(args.block_m)
            if args.block_n is not None:
                os.environ["GROUPED_MM_2D3D_BLOCK_N"] = str(args.block_n)
            if args.block_k is not None:
                os.environ["GROUPED_MM_2D3D_BLOCK_K"] = str(args.block_k)
            if args.num_warps is not None:
                os.environ["GROUPED_MM_2D3D_NUM_WARPS"] = str(args.num_warps)
        if impl == "rowwise":
            if args.rowwise_block_n is not None:
                os.environ["GROUPED_MM_2D3D_ROWWISE_BLOCK_N"] = str(args.rowwise_block_n)
            if args.rowwise_block_k is not None:
                os.environ["GROUPED_MM_2D3D_ROWWISE_BLOCK_K"] = str(args.rowwise_block_k)
            if args.rowwise_num_warps is not None:
                os.environ["GROUPED_MM_2D3D_ROWWISE_NUM_WARPS"] = str(args.rowwise_num_warps)

        routing_desc = (
            f"batch_size={args.batch_size}, active_experts_per_token={args.active_experts_per_token}"
            if args.batch_size is not None
            else f"tokens_per_expert={args.tokens_per_expert}"
        )

        common = (
            f"(experts={experts}, total_routed_tokens={total_routed_tokens}, {routing_desc}, "
            f"embedding_size={embedding_size}, output_features={output_features}, dtype={args.dtype}, seed={args.seed})"
        )

        if args.mode in ("forward", "both"):
            ms = _time_forward(a, b, offs, iters=args.iters, warmup=args.warmup)
            flops = 2 * total_routed_tokens * embedding_size * output_features
            tflops = flops / (ms / 1e3) / 1e12
            print(f"{impl:7s}  forward  {ms:9.4f} ms  {tflops:7.3f} TFLOPs   {common}")

        if args.mode in ("backward", "both"):
            compute_grad_a = args.backward in ("da", "both")
            compute_grad_b = args.backward in ("db", "both")
            ms = _time_backward_da_db(
                a,
                b,
                offs,
                grad_out,
                compute_grad_a=compute_grad_a,
                compute_grad_b=compute_grad_b,
                iters=args.iters,
                warmup=args.warmup,
            )
            # Report FLOPs for the subset being computed.
            flops = 0
            if compute_grad_a:
                flops += 2 * total_routed_tokens * embedding_size * output_features
            if compute_grad_b:
                flops += 2 * total_routed_tokens * embedding_size * output_features
            tflops = flops / (ms / 1e3) / 1e12 if flops else 0.0
            print(f"{impl:7s}  backward {ms:9.4f} ms  {tflops:7.3f} TFLOPs   {common}  backward={args.backward}")


if __name__ == "__main__":
    main()
