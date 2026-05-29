#!/usr/bin/env python3
"""Estimate KV cache transfer size and transfer time.

The numbers are first-order estimates for disaggregated prefill/decode studies.
They model the KV shard moved by one worker by default when --kv-shard-factor is
set. Use --kv-shard-factor 1 for the full logical KV cache.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable


BYTES_PER_MIB = 1024**2


@dataclass(frozen=True)
class ModelConfig:
    name: str
    layers: int
    kv_heads: int | None = None
    head_dim: int = 128
    dtype_bytes: int = 2
    mla_latent_dim: int | None = None
    mla_rope_dim: int | None = None

    @property
    def is_mla(self) -> bool:
        return self.mla_latent_dim is not None and self.mla_rope_dim is not None


@dataclass(frozen=True)
class NetworkConfig:
    name: str
    bandwidth_gb_s: float


MODELS: dict[str, ModelConfig] = {
    "Llama-3-8B": ModelConfig("Llama-3-8B", layers=32, kv_heads=8),
    "Llama-3-70B": ModelConfig("Llama-3-70B", layers=80, kv_heads=8),
    "Llama-3.1-405B": ModelConfig("Llama-3.1-405B", layers=126, kv_heads=8),
    "Qwen-2.5-72B": ModelConfig("Qwen-2.5-72B", layers=80, kv_heads=8),
    "Qwen-2.5-7B": ModelConfig("Qwen-2.5-7B", layers=28, kv_heads=4),
    "DeepSeek-V3": ModelConfig(
        "DeepSeek-V3", layers=61, mla_latent_dim=512, mla_rope_dim=64
    ),
}

NETWORKS: dict[str, NetworkConfig] = {
    "NVLink-H100": NetworkConfig("NVLink-H100", 450),
    "NVLink-A100": NetworkConfig("NVLink-A100", 300),
    "IB-NDR-400G": NetworkConfig("IB-NDR-400G", 46),
    "IB-HDR-200G": NetworkConfig("IB-HDR-200G", 23),
    "RoCE-100G": NetworkConfig("RoCE-100G", 11),
    "TCP-25G": NetworkConfig("TCP-25G", 2.8),
}


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def kv_bytes(model: ModelConfig, seq_len: int, kv_shard_factor: float = 1.0) -> float:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if kv_shard_factor <= 0:
        raise ValueError("kv_shard_factor must be positive")
    if model.is_mla:
        assert model.mla_latent_dim is not None
        assert model.mla_rope_dim is not None
        per_token = model.layers * (model.mla_latent_dim + model.mla_rope_dim)
    else:
        assert model.kv_heads is not None
        per_token = 2 * model.layers * model.kv_heads * model.head_dim
    return per_token * seq_len * model.dtype_bytes / kv_shard_factor


def transfer_time_ms(size_bytes: float, network: NetworkConfig) -> float:
    if network.bandwidth_gb_s <= 0:
        raise ValueError("network bandwidth must be positive")
    return size_bytes / (network.bandwidth_gb_s * 1e9) * 1000


def print_size_table(models: Iterable[ModelConfig], seq_lens: list[int], shard: float) -> None:
    print("KV cache shard size (MiB)")
    print(f"kv_shard_factor={shard:g}")
    print(f"{'Model':<18}" + "".join(f"{seq:>12,}" for seq in seq_lens))
    for model in models:
        cells = [kv_bytes(model, seq, shard) / BYTES_PER_MIB for seq in seq_lens]
        print(f"{model.name:<18}" + "".join(f"{value:>12.1f}" for value in cells))


def print_transfer_table(
    model: ModelConfig,
    networks: Iterable[NetworkConfig],
    seq_lens: list[int],
    shard: float,
) -> None:
    print()
    print(f"Transfer time for {model.name} (ms)")
    print(f"kv_shard_factor={shard:g}")
    print(f"{'Network':<18}" + "".join(f"{seq:>12,}" for seq in seq_lens))
    for network in networks:
        cells = [transfer_time_ms(kv_bytes(model, seq, shard), network) for seq in seq_lens]
        print(f"{network.name:<18}" + "".join(f"{value:>12.2f}" for value in cells))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate KV cache transfer size and time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODELS), default="Llama-3-70B")
    parser.add_argument("--network", choices=sorted(NETWORKS), default="IB-NDR-400G")
    parser.add_argument(
        "--seq-lens",
        type=parse_int_list,
        default=parse_int_list("512,1024,4096,16384,32768,131072"),
    )
    parser.add_argument(
        "--kv-shard-factor",
        type=float,
        default=1.0,
        help="divide logical KV by this factor to estimate one transferred shard",
    )
    parser.add_argument("--table", action="store_true", help="print preset tables")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.kv_shard_factor <= 0:
        raise SystemExit("--kv-shard-factor must be positive")
    model = MODELS[args.model]
    network = NETWORKS[args.network]
    if args.table:
        print_size_table(MODELS.values(), args.seq_lens, args.kv_shard_factor)
        print_transfer_table(model, NETWORKS.values(), args.seq_lens, args.kv_shard_factor)
        return 0

    print(f"model: {model.name}")
    print(f"network: {network.name} ({network.bandwidth_gb_s:g} GB/s)")
    print(f"kv_shard_factor: {args.kv_shard_factor:g}")
    print(f"{'seq_len':>12} {'size_mib':>12} {'transfer_ms':>14}")
    for seq_len in args.seq_lens:
        size = kv_bytes(model, seq_len, args.kv_shard_factor)
        print(
            f"{seq_len:>12,} {size / BYTES_PER_MIB:>12.1f} "
            f"{transfer_time_ms(size, network):>14.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
