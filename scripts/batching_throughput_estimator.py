#!/usr/bin/env python3
"""Estimate static vs continuous batching token waste from length distributions."""

from __future__ import annotations

import argparse
import random
from statistics import mean


def generate_lengths(distribution: str, num_prompts: int, rng: random.Random) -> list[int]:
    if distribution == "fixed":
        return [150] * num_prompts
    if distribution == "uniform":
        return [rng.randint(50, 300) for _ in range(num_prompts)]
    if distribution == "exponential":
        return [min(int(rng.expovariate(1 / 150)) + 10, 500) for _ in range(num_prompts)]
    if distribution == "bimodal":
        lengths: list[int] = []
        for _ in range(num_prompts):
            if rng.random() < 0.5:
                lengths.append(rng.randint(20, 60))
            else:
                lengths.append(rng.randint(200, 400))
        return lengths
    raise ValueError(f"unknown distribution: {distribution}")


def summarize(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        raise ValueError("length list cannot be empty")
    total_work = sum(lengths)
    static_work = len(lengths) * max(lengths)
    wasted = static_work - total_work
    return {
        "count": len(lengths),
        "mean": mean(lengths),
        "max": max(lengths),
        "total_work": total_work,
        "static_work": static_work,
        "wasted": wasted,
        "wasted_pct": wasted / static_work * 100 if static_work else 0,
        "ideal_gain": static_work / total_work if total_work else 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate ideal continuous batching gain from output lengths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--distributions",
        default="fixed,uniform,exponential,bimodal",
        help="comma-separated subset of fixed,uniform,exponential,bimodal",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_prompts <= 0:
        raise SystemExit("--num-prompts must be positive")
    rng = random.Random(args.seed)
    distributions = [item.strip() for item in args.distributions.split(",") if item.strip()]

    print("Static vs continuous batching length estimate")
    print(f"num_prompts={args.num_prompts:,} seed={args.seed}")
    print(
        f"{'distribution':<14} {'mean':>8} {'max':>8} {'Lmax/Lmean':>11} "
        f"{'static_tokens':>14} {'work_tokens':>12} {'waste_%':>9}"
    )
    for distribution in distributions:
        lengths = generate_lengths(distribution, args.num_prompts, rng)
        row = summarize(lengths)
        print(
            f"{distribution:<14} {row['mean']:>8.1f} {row['max']:>8.0f} "
            f"{row['ideal_gain']:>11.2f} {row['static_work']:>14,.0f} "
            f"{row['total_work']:>12,.0f} {row['wasted_pct']:>8.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
