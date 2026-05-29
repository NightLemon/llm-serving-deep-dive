#!/usr/bin/env python3
"""Pure-Python speculative decoding rejection-sampling simulator."""

from __future__ import annotations

import argparse
import random


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("cannot normalize empty or zero distribution")
    return [value / total for value in values]


def random_distribution(vocab_size: int, rng: random.Random) -> list[float]:
    return normalize([rng.gammavariate(1.0, 1.0) for _ in range(vocab_size)])


def mix_distribution(
    target: list[float], similarity: float, rng: random.Random
) -> tuple[list[float], float]:
    noise = random_distribution(len(target), rng)
    draft = [similarity * p + (1 - similarity) * n for p, n in zip(target, noise)]
    draft = normalize(draft)
    alpha = sum(min(p, q) for p, q in zip(target, draft))
    return draft, alpha


def sample_index(distribution: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(distribution):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(distribution) - 1


def correction_distribution(target: list[float], draft: list[float]) -> list[float]:
    residual = [max(0.0, p - q) for p, q in zip(target, draft)]
    total = sum(residual)
    if total == 0:
        return target
    return [value / total for value in residual]


def rejection_sample_one(
    token: int, target: list[float], draft: list[float], rng: random.Random
) -> bool:
    accept_prob = min(1.0, target[token] / max(draft[token], 1e-15))
    return rng.random() < accept_prob


def simulate_round(
    gamma: int, vocab_size: int, similarity: float, rng: random.Random
) -> tuple[int, float]:
    alphas: list[float] = []
    for position in range(gamma):
        target = random_distribution(vocab_size, rng)
        draft, alpha = mix_distribution(target, similarity, rng)
        alphas.append(alpha)
        token = sample_index(draft, rng)
        if rejection_sample_one(token, target, draft, rng):
            continue
        _ = sample_index(correction_distribution(target, draft), rng)
        return position + 1, sum(alphas) / len(alphas)

    # If all draft tokens are accepted, the target model contributes one bonus token.
    bonus_target = random_distribution(vocab_size, rng)
    _ = sample_index(bonus_target, rng)
    return gamma + 1, sum(alphas) / len(alphas)


def theoretical_expected(alpha: float, gamma: int) -> float:
    if abs(1 - alpha) < 1e-12:
        return gamma + 1
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def parse_number_list(raw: str, value_type: type) -> list:
    values = [value_type(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate speculative decoding acceptance counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gammas", type=lambda raw: parse_number_list(raw, int), default="1,3,5,7")
    parser.add_argument(
        "--similarities",
        type=lambda raw: parse_number_list(raw, float),
        default="0.5,0.8,0.9,0.95",
    )
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--vocab-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.trials <= 0 or args.vocab_size <= 1:
        raise SystemExit("--trials must be positive and --vocab-size must be > 1")
    rng = random.Random(args.seed)
    gammas: list[int] = args.gammas if isinstance(args.gammas, list) else parse_number_list(args.gammas, int)
    similarities: list[float] = (
        args.similarities
        if isinstance(args.similarities, list)
        else parse_number_list(args.similarities, float)
    )
    if any(gamma <= 0 for gamma in gammas):
        raise SystemExit("all gamma values must be positive")
    if any(sim < 0 or sim > 1 for sim in similarities):
        raise SystemExit("similarity values must be in [0, 1]")

    print("Speculative decoding simulation")
    print(f"trials={args.trials:,} vocab_size={args.vocab_size:,} seed={args.seed}")
    print(f"{'similarity':>10} {'gamma':>5} {'alpha':>8} {'theory':>10} {'actual':>10} {'error_%':>9}")
    for similarity in similarities:
        for gamma in gammas:
            accepted_total = 0
            alpha_total = 0.0
            for _ in range(args.trials):
                accepted, alpha = simulate_round(gamma, args.vocab_size, similarity, rng)
                accepted_total += accepted
                alpha_total += alpha
            mean_alpha = alpha_total / args.trials
            actual = accepted_total / args.trials
            theory = theoretical_expected(mean_alpha, gamma)
            error_pct = abs(actual - theory) / theory * 100
            print(
                f"{similarity:>10.2f} {gamma:>5} {mean_alpha:>8.3f} "
                f"{theory:>10.3f} {actual:>10.3f} {error_pct:>9.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
