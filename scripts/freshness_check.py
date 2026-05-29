#!/usr/bin/env python3
"""Check current PyPI package versions against the recorded learning baseline.

This is intentionally optional and is not part of the default CI gate because it
uses the network. Run it before experiments to spot framework version drift.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PACKAGES = ["vllm", "sglang", "lmcache"]
BASELINE_NAMES = {
    "vllm": "vllm",
    "sglang": "sglang",
    "lmcache": "lmcache",
}


@dataclass(frozen=True)
class PackageResult:
    package: str
    latest: str | None
    uploaded_at: str | None
    baseline: str | None
    status: str
    error: str | None = None


def normalize_version(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().removeprefix("v")


def parse_baseline_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    baseline: dict[str, str] = {}
    version_pattern = re.compile(r"`v?([^`]+)`")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 2:
            continue
        name = columns[0].lower()
        package = BASELINE_NAMES.get(name)
        if package is None:
            continue
        match = version_pattern.search(columns[1])
        if match:
            baseline[package] = normalize_version(match.group(1)) or match.group(1)
    return baseline


def parse_baseline_args(items: list[str]) -> dict[str, str]:
    baselines: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError("baseline entries must look like package=version")
        package, version = item.split("=", 1)
        package = package.strip().lower()
        version = normalize_version(version)
        if not package or not version:
            raise argparse.ArgumentTypeError("baseline package and version cannot be empty")
        baselines[package] = version
    return baselines


def fetch_pypi(package: str, timeout: float) -> tuple[str, str | None]:
    url = f"https://pypi.org/pypi/{package}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "llm-serving-deep-dive freshness_check"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    latest = payload["info"]["version"]
    files = payload.get("releases", {}).get(latest, [])
    uploaded_at = None
    if files:
        uploaded_at = files[0].get("upload_time_iso_8601") or files[0].get("upload_time")
    return latest, uploaded_at


def check_package(package: str, baseline: str | None, timeout: float) -> PackageResult:
    try:
        latest, uploaded_at = fetch_pypi(package, timeout)
    except urllib.error.HTTPError as exc:
        return PackageResult(package, None, None, baseline, "error", f"HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        return PackageResult(package, None, None, baseline, "error", str(exc))

    normalized_latest = normalize_version(latest)
    normalized_baseline = normalize_version(baseline)
    if normalized_baseline is None:
        status = "no baseline"
    elif normalized_latest == normalized_baseline:
        status = "matches baseline"
    else:
        status = "differs from baseline"
    return PackageResult(package, latest, uploaded_at, baseline, status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check current PyPI package versions against the repo baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--package", action="append", dest="packages", help="PyPI package to check; repeatable")
    parser.add_argument("--baseline", action="append", default=[], help="override baseline, e.g. vllm=0.21.0")
    parser.add_argument("--baseline-file", default="resources/version-baseline.md")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    packages = [package.lower() for package in (args.packages or DEFAULT_PACKAGES)]
    baseline = parse_baseline_file(Path(args.baseline_file))
    baseline.update(parse_baseline_args(args.baseline))

    print("PyPI freshness check")
    print(f"baseline_file={args.baseline_file}")
    print(f"{'package':<12} {'baseline':<16} {'latest':<16} {'uploaded_at':<24} {'status'}")
    exit_code = 0
    for package in packages:
        result = check_package(package, baseline.get(package), args.timeout)
        if result.status in {"differs from baseline", "error"}:
            exit_code = 1
        uploaded_at = result.uploaded_at or "-"
        latest = result.latest or "-"
        baseline_version = result.baseline or "-"
        status = result.status if result.error is None else f"{result.status}: {result.error}"
        print(f"{package:<12} {baseline_version:<16} {latest:<16} {uploaded_at:<24} {status}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
