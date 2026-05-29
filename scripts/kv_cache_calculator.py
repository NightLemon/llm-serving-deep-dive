#!/usr/bin/env python3
"""Small KV cache capacity calculator for the exercises.

The goal is to make the arithmetic in the notes reproducible without pulling in
framework dependencies. Units are binary units for memory output: KiB/MiB/GiB.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import floor
from typing import Iterable


BYTES_PER_KIB = 1024
BYTES_PER_MIB = 1024**2
BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class MixItem:
    name: str
    weight: float
    tokens: int


def bytes_human(value: float) -> str:
    units = [
        (BYTES_PER_GIB, "GiB"),
        (BYTES_PER_MIB, "MiB"),
        (BYTES_PER_KIB, "KiB"),
    ]
    for divisor, unit in units:
        if abs(value) >= divisor:
            return f"{value / divisor:,.3f} {unit}"
    return f"{value:,.0f} bytes"


def parse_mix(raw_items: Iterable[str]) -> list[MixItem]:
    items: list[MixItem] = []
    for raw in raw_items:
        parts = raw.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"invalid mix item {raw!r}; expected name:weight:tokens"
            )
        name, weight_raw, tokens_raw = parts
        if not name:
            raise argparse.ArgumentTypeError("mix item name cannot be empty")
        try:
            weight = float(weight_raw)
            tokens = int(tokens_raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid mix item {raw!r}; weight must be float and tokens int"
            ) from exc
        if weight < 0:
            raise argparse.ArgumentTypeError("mix item weight must be non-negative")
        if tokens <= 0:
            raise argparse.ArgumentTypeError("mix item tokens must be positive")
        items.append(MixItem(name=name, weight=weight, tokens=tokens))
    return items


def kv_per_token_bytes(args: argparse.Namespace) -> tuple[float, list[str]]:
    warnings: list[str] = []
    dtype_bytes = args.dtype_bytes
    if args.layers <= 0:
        raise ValueError("--layers must be positive")
    if args.head_dim <= 0:
        raise ValueError("--head-dim must be positive")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")
    if dtype_bytes <= 0:
        raise ValueError("--dtype-bytes must be positive")

    if args.attention == "mla":
        if args.mla_latent_dim is None or args.mla_rope_dim is None:
            raise ValueError("MLA mode requires --mla-latent-dim and --mla-rope-dim")
        shard_factor = args.kv_shard_factor if args.kv_shard_factor else 1
        if shard_factor <= 0:
            raise ValueError("--kv-shard-factor must be positive")
        return (
            args.layers
            * ((args.mla_latent_dim + args.mla_rope_dim) / shard_factor)
            * dtype_bytes,
            warnings,
        )

    if args.kv_heads is None:
        raise ValueError("standard attention modes require --kv-heads")
    if args.kv_heads <= 0:
        raise ValueError("--kv-heads must be positive")
    shard_factor = (
        args.kv_shard_factor if args.kv_shard_factor else args.tensor_parallel_size
    )
    if shard_factor <= 0:
        raise ValueError("KV shard factor must be positive")
    if args.kv_heads % shard_factor != 0:
        warnings.append(
            "total KV heads is not evenly divisible by the shard factor; "
            "pass --local-kv-heads or --kv-shard-factor to match your runtime exactly"
        )
    local_kv_heads = (
        args.local_kv_heads
        if args.local_kv_heads is not None
        else args.kv_heads / shard_factor
    )
    if local_kv_heads <= 0:
        raise ValueError("local KV heads must be positive")
    return 2 * args.layers * local_kv_heads * args.head_dim * dtype_bytes, warnings


def available_kv_bytes(args: argparse.Namespace) -> float | None:
    provided = [
        args.gpu_memory_gb is not None,
        args.model_weights_gb_per_gpu is not None,
    ]
    if not any(provided):
        return None
    if not all(provided):
        raise ValueError(
            "capacity calculation requires both --gpu-memory-gb and "
            "--model-weights-gb-per-gpu"
        )
    if args.gpu_memory_gb <= 0:
        raise ValueError("--gpu-memory-gb must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.model_weights_gb_per_gpu < 0 or args.overhead_gb_per_gpu < 0:
        raise ValueError("weight and overhead memory must be non-negative")
    available_gb = (
        args.gpu_memory_gb * args.gpu_memory_utilization
        - args.model_weights_gb_per_gpu
        - args.overhead_gb_per_gpu
    )
    if available_gb <= 0:
        raise ValueError("available KV memory is not positive; check capacity arguments")
    return available_gb * BYTES_PER_GIB


def format_capacity_line(prefix: str, kv_bytes: float, available_bytes: float | None) -> str:
    if available_bytes is None:
        return f"{prefix}: {bytes_human(kv_bytes)}"
    exact = available_bytes / kv_bytes
    return (
        f"{prefix}: {bytes_human(kv_bytes)} | "
        f"max concurrency floor={floor(exact):,} exact={exact:,.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate per-token and per-request KV cache memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--attention", choices=["mha", "gqa", "mqa", "mla"], default="gqa")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, help="total KV heads for MHA/GQA/MQA")
    parser.add_argument(
        "--local-kv-heads", type=float, help="override local KV heads per GPU"
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype-bytes", type=float, default=2.0, help="BF16/FP16=2, FP8=1"
    )
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument(
        "--kv-shard-factor",
        type=float,
        help=(
            "override KV cache sharding divisor; default is TP for standard "
            "attention and 1 for MLA"
        ),
    )
    parser.add_argument("--mla-latent-dim", type=int, help="MLA compressed latent dimension")
    parser.add_argument("--mla-rope-dim", type=int, help="MLA RoPE key dimension stored in cache")

    token_group = parser.add_argument_group("request size")
    token_group.add_argument("--seq-len", type=int, help="total cached tokens for one request")
    token_group.add_argument("--text-tokens", type=int, default=0)
    token_group.add_argument("--visual-tokens", type=int, default=0)
    token_group.add_argument("--output-tokens", type=int, default=0)
    token_group.add_argument(
        "--mix",
        action="append",
        default=[],
        metavar="NAME:WEIGHT:TOKENS",
        help="repeat for traffic mix rows, for example low:0.4:1274",
    )

    capacity_group = parser.add_argument_group("per-GPU capacity")
    capacity_group.add_argument("--gpu-memory-gb", type=float, help="per-GPU total memory")
    capacity_group.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    capacity_group.add_argument("--model-weights-gb-per-gpu", type=float)
    capacity_group.add_argument("--overhead-gb-per-gpu", type=float, default=0.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        per_token, warnings = kv_per_token_bytes(args)
        available = available_kv_bytes(args)
        mix_items = parse_mix(args.mix)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))

    token_parts = [
        args.text_tokens,
        args.visual_tokens,
        args.output_tokens,
    ]
    if args.seq_len is not None and args.seq_len <= 0:
        parser.error("--seq-len must be positive when provided")
    if any(part < 0 for part in token_parts):
        parser.error("token counts must be non-negative")

    print("KV cache estimate")
    print(f"attention: {args.attention}")
    print(f"layers: {args.layers:,}")
    if args.attention == "mla":
        shard_factor = args.kv_shard_factor if args.kv_shard_factor else 1
        print(f"MLA dims cached: {args.mla_latent_dim:,} + {args.mla_rope_dim:,}")
        print(f"KV shard factor: {shard_factor:g}")
    else:
        shard_factor = (
            args.kv_shard_factor if args.kv_shard_factor else args.tensor_parallel_size
        )
        local_heads = (
            args.local_kv_heads
            if args.local_kv_heads is not None
            else args.kv_heads / shard_factor
        )
        print(f"KV heads total/local: {args.kv_heads:g} / {local_heads:g}")
        print(f"head dim: {args.head_dim:,}")
        print(f"KV shard factor: {shard_factor:g}")
    print(f"dtype bytes: {args.dtype_bytes:g}")
    print(f"KV per token per GPU: {per_token:,.0f} bytes ({bytes_human(per_token)})")

    if available is not None:
        print(f"available KV per GPU: {bytes_human(available)}")

    seq_len = args.seq_len or args.text_tokens + args.visual_tokens + args.output_tokens
    if seq_len > 0:
        print(
            format_capacity_line(
                f"request tokens={seq_len:,}", per_token * seq_len, available
            )
        )

    if mix_items:
        weight_sum = sum(item.weight for item in mix_items)
        if weight_sum <= 0:
            parser.error("mix weights must sum to a positive value")
        print("traffic mix:")
        weighted_tokens = 0.0
        for item in mix_items:
            normalized_weight = item.weight / weight_sum
            weighted_tokens += normalized_weight * item.tokens
            label = f"  {item.name} weight={normalized_weight:.3f} tokens={item.tokens:,}"
            print(format_capacity_line(label, per_token * item.tokens, available))
        weighted_kv = per_token * weighted_tokens
        print(
            format_capacity_line(
                f"weighted average tokens={weighted_tokens:,.1f}", weighted_kv, available
            )
        )

    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
