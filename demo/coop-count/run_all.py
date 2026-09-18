# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Run one case through the scripted, claude and qwen seats, keeping each journal."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parents[1]

SEATS = ("scripted", "claude", "qwen")

#: The tape and the handler are written around these two numbers.
SAID, NUMBER = 15, 51

#: The scripted tape stops before this on its own; a live seat is cut off here.
MAX_TICKS = 400


def environment(path: Path) -> dict[str, str]:
    """This process's environment, with a ``.env``'s settings added under it."""
    env = dict(os.environ)
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key, value = key.strip(), value.strip().strip("'\"")
        if value and key not in env:
            env[key] = value
    return env


def run(command: list[str], env: dict[str, str]) -> int:
    print(f"\n$ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=DEMO, env=env).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seats", nargs="+", choices=SEATS, default=list(SEATS))
    parser.add_argument(
        "--case",
        default=str(DEMO / "cases" / "coop-count-scripted"),
        help="the case every seat runs; it needs the tape for the scripted seat "
        "and the handler all three interrupt into",
    )
    parser.add_argument(
        "--runs",
        default=str(Path.cwd() / "runs"),
        help="directory for the journals (default: runs/ in the current directory)",
    )
    parser.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="thread count for the qwen seat; it changes sampling, so pin it",
    )
    args = parser.parse_args(argv)

    env = environment(ROOT / ".env")

    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for seat in args.seats:
        journal = runs / f"{seat}.jsonl"
        if seat == "qwen" and run(["uv", "run", "zeos-count", "fetch-model"], env):
            failed.append(seat)
            continue

        command = [
            "uv",
            "run",
            "zeos-count",
            "run",
            args.case,
            "--machine",
            seat,
            "--journal",
            str(journal),
            "--interrupt",
            str(SAID),
            str(NUMBER),
            "--seed",
            str(args.seed),
            "--max-ticks",
            str(args.max_ticks),
        ]
        if seat == "qwen" and args.threads is not None:
            command += ["--threads", str(args.threads)]
        if run(command, env):
            failed.append(seat)

    print(f"\n{'-' * 60}")
    for seat in args.seats:
        journal = runs / f"{seat}.jsonl"
        if seat in failed:
            print(f"  {seat:<10} FAILED")
        else:
            print(f"  {seat:<10} {journal} ({journal.stat().st_size} bytes)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
