#!/usr/bin/env python3
"""Run lightweight checks for repository helper scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(script: str, *args: str, expect: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"{script} failed with exit code {result.returncode}")
    if expect not in result.stdout:
        print(result.stdout)
        raise SystemExit(f"{script} output did not include expected text: {expect!r}")
    print(f"ok: {script}")


def main() -> int:
    run_script(
        "kv_cache_calculator.py",
        "--layers",
        "80",
        "--kv-heads",
        "8",
        "--head-dim",
        "128",
        "--tp",
        "4",
        "--seq-len",
        "16384",
        expect="1.250 GiB",
    )
    run_script(
        "kv_transfer_calculator.py",
        "--model",
        "Llama-3-70B",
        "--network",
        "IB-NDR-400G",
        "--seq-lens",
        "4096",
        "--kv-shard-factor",
        "8",
        expect="160.0",
    )
    run_script(
        "speculative_decoding_simulator.py",
        "--gammas",
        "3",
        "--similarities",
        "0.8",
        "--trials",
        "200",
        expect="Speculative decoding simulation",
    )
    run_script(
        "batching_throughput_estimator.py",
        "--num-prompts",
        "50",
        expect="Lmax/Lmean",
    )
    run_script(
        "tp_comm_estimator.py",
        "--model",
        "Llama-3-70B",
        "--tp-sizes",
        "2,4",
        expect="TP decode-step estimate",
    )
    run_script("freshness_check.py", "--help", expect="PyPI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
