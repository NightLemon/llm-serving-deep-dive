#!/usr/bin/env python3
"""Rough Tensor Parallel decode-step communication model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params_b: float
    layers: int
    hidden_size: int
    kv_heads: int
    head_dim: int = 128


@dataclass(frozen=True)
class HardwareConfig:
    name: str
    hbm_tb_s: float
    link_gb_s: float
    allreduce_latency_us: float


MODELS: dict[str, ModelConfig] = {
    "Llama-3-8B": ModelConfig("Llama-3-8B", 8, 32, 4096, 8),
    "Llama-3-70B": ModelConfig("Llama-3-70B", 70, 80, 8192, 8),
    "Llama-3.1-405B": ModelConfig("Llama-3.1-405B", 405, 126, 16384, 8),
    "Qwen-2.5-72B": ModelConfig("Qwen-2.5-72B", 72, 80, 8192, 8),
}

HARDWARE: dict[str, HardwareConfig] = {
    "H100-NVLink": HardwareConfig("H100-NVLink", hbm_tb_s=3.35, link_gb_s=450, allreduce_latency_us=5),
    "A100-NVLink": HardwareConfig("A100-NVLink", hbm_tb_s=2.0, link_gb_s=300, allreduce_latency_us=8),
    "PCIe-Gen4": HardwareConfig("PCIe-Gen4", hbm_tb_s=1.6, link_gb_s=32, allreduce_latency_us=20),
}


def allreduce_ms(bytes_per_rank: float, tp_size: int, hw: HardwareConfig) -> float:
    if tp_size <= 1:
        return 0.0
    ring_bytes = 2 * (tp_size - 1) / tp_size * bytes_per_rank
    bandwidth_ms = ring_bytes / (hw.link_gb_s * 1e9) * 1000
    return hw.allreduce_latency_us / 1000 + bandwidth_ms


def decode_breakdown(
    model: ModelConfig,
    hw: HardwareConfig,
    tp_size: int,
    pp_size: int,
    batch_size: int,
    seq_len: int,
    dtype_bytes: int,
) -> dict[str, float]:
    if min(tp_size, pp_size, batch_size, seq_len, dtype_bytes) <= 0:
        raise ValueError("tp, pp, batch, seq_len, and dtype_bytes must be positive")
    layers_local = model.layers / pp_size
    params_bytes_local = model.params_b * 1e9 * dtype_bytes / (tp_size * pp_size)
    weight_read_ms = params_bytes_local / (hw.hbm_tb_s * 1e12) * 1000

    local_kv_heads = model.kv_heads / tp_size
    kv_bytes_local = (
        2 * layers_local * local_kv_heads * model.head_dim * batch_size * seq_len * dtype_bytes
    )
    kv_read_ms = kv_bytes_local / (hw.hbm_tb_s * 1e12) * 1000

    allreduce_payload = batch_size * model.hidden_size * dtype_bytes
    # A standard TP transformer layer has one attention-output and one MLP-output reduction.
    tp_comm_ms = 2 * layers_local * allreduce_ms(allreduce_payload, tp_size, hw)

    pp_payload = batch_size * model.hidden_size * dtype_bytes
    pp_comm_ms = 0.0
    if pp_size > 1:
        pp_comm_ms = (pp_size - 1) * pp_payload / (hw.link_gb_s * 1e9) * 1000

    total_ms = weight_read_ms + kv_read_ms + tp_comm_ms + pp_comm_ms
    comm_ms = tp_comm_ms + pp_comm_ms
    return {
        "weight_read_ms": weight_read_ms,
        "kv_read_ms": kv_read_ms,
        "tp_comm_ms": tp_comm_ms,
        "pp_comm_ms": pp_comm_ms,
        "total_ms": total_ms,
        "comm_pct": comm_ms / total_ms * 100 if total_ms else 0,
    }


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate decode-step TP communication cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODELS), default="Llama-3-70B")
    parser.add_argument("--hardware", choices=sorted(HARDWARE), default="H100-NVLink")
    parser.add_argument("--tp-sizes", type=parse_int_list, default=parse_int_list("1,2,4,8"))
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = MODELS[args.model]
    hw = HARDWARE[args.hardware]
    print("TP decode-step estimate")
    print(
        f"model={model.name} hardware={hw.name} pp={args.pp_size} "
        f"batch={args.batch_size} seq_len={args.seq_len} dtype_bytes={args.dtype_bytes}"
    )
    print(
        f"{'TP':>3} {'weight_ms':>11} {'kv_ms':>9} {'tp_comm_ms':>12} "
        f"{'pp_comm_ms':>12} {'total_ms':>10} {'comm_%':>8}"
    )
    for tp_size in args.tp_sizes:
        row = decode_breakdown(
            model=model,
            hw=hw,
            tp_size=tp_size,
            pp_size=args.pp_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            dtype_bytes=args.dtype_bytes,
        )
        print(
            f"{tp_size:>3} {row['weight_read_ms']:>11.2f} {row['kv_read_ms']:>9.2f} "
            f"{row['tp_comm_ms']:>12.2f} {row['pp_comm_ms']:>12.2f} "
            f"{row['total_ms']:>10.2f} {row['comm_pct']:>7.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
