#!/usr/bin/env python3
"""Run lightweight content checks for Markdown pages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".harness", ".venv", "build", "dist", "docs", "node_modules", "site", "venv"}

BOLD_ONLY_RE = re.compile(r"^(?P<prefix>(?:>[ \t]?)*?)\*\*[^\n]+\*\*[ \t]*$")
LIST_ITEM_RE = re.compile(r"^(?P<prefix>(?:>[ \t]?)*?)(?:[-*+] |\d+[.)] )")
FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")


def markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files)


def line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return "\n"


def check_file(path: Path, *, fix: bool) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.readlines()

    issues = 0
    changed = False
    output: list[str] = []
    in_fence = False
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.rstrip("\r\n")

        if FENCE_RE.match(stripped):
            in_fence = not in_fence

        output.append(line)
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        next_stripped = next_line.rstrip("\r\n")

        bold_match = BOLD_ONLY_RE.match(stripped)
        list_match = LIST_ITEM_RE.match(next_stripped)

        if not in_fence and bold_match and list_match and bold_match["prefix"] == list_match["prefix"]:
            issues += 1
            if fix:
                blank_prefix = bold_match["prefix"].rstrip()
                output.append(f"{blank_prefix}{line_ending(line)}")
                changed = True
            else:
                rel = path.relative_to(ROOT)
                print(f"{rel}:{index + 2}: add a blank line before the list item")

        index += 1

    if changed:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("".join(output))

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Markdown content conventions.")
    parser.add_argument("--fix", action="store_true", help="rewrite files to fix safe content issues")
    args = parser.parse_args()

    issues = sum(check_file(path, fix=args.fix) for path in markdown_files())
    if issues and not args.fix:
        print(f"content checks found {issues} issue(s)")
        return 1
    if issues:
        print(f"fixed: {issues} markdown list spacing issue(s)")
        return 0
    print("ok: content checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
