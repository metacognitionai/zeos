# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Run one case through every machine seat, keeping a journal of each.

    uv run python demo/counter/run_all.py
    uv run python demo/counter/run_all.py --seats scripted qwen
    uv run python demo/counter/run_all.py --case simple_count --runs /tmp/counter

Each seat writes ``<runs>/<case>-<seat>.jsonl``, so the three runs can be compared
against each other afterwards with ``zeos debug`` or ``zeos replay``. The qwen seat
fetches its weights first; that is a no-op once they are on disk.

The repository root's ``.env`` is passed to each run, which is how the claude seat
gets its key: the SDK reads ``ANTHROPIC_API_KEY`` from the environment and nothing
else here puts it there. An already-exported variable wins over the file.

A seat that cannot run -- no API key for ``claude``, no network for the fetch --
fails on its own and the remaining seats still run. The exit status is non-zero if
any of them failed.
"""

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
    """Run a command from the repository root, echoing it first."""
    print(f"\n$ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seats",
        nargs="+",
        choices=SEATS,
        default=list(SEATS),
        help=f"which seats to run (default: all of {', '.join(SEATS)})",
    )
    parser.add_argument(
        "--case",
        default="simple_count",
        help="which case directory in this demo to run (default: simple_count)",
    )
    parser.add_argument(
        "--runs",
        default=str(DEMO / "runs"),
        help="directory for the journals (default: demo/counter/runs)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="token boundaries per run; passed through to run.py",
    )
    args = parser.parse_args(argv)

    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    env = environment(ROOT / ".env")

    failed: list[str] = []
    for seat in args.seats:
        journal = runs / f"{args.case}-{seat}.jsonl"
        fetch = ["uv", "run", "python", "demo/counter/fetch_model.py"]
        if seat == "qwen" and run(fetch, env):
            failed.append(f"{seat} (fetch_model)")
            continue

        command = [
            "uv",
            "run",
            "python",
            "demo/counter/run.py",
            "--case",
            args.case,
            "--journal",
            str(journal),
        ]
        if seat != "scripted":
            command += ["--machine", seat]
        if args.ticks is not None:
            command += ["--ticks", str(args.ticks)]
        if run(command, env):
            failed.append(seat)

    print(f"\n{'-' * 60}")
    for seat in args.seats:
        journal = runs / f"{args.case}-{seat}.jsonl"
        if seat in failed or f"{seat} (fetch_model)" in failed:
            print(f"  {seat:<10} FAILED")
        else:
            print(f"  {seat:<10} {journal} ({journal.stat().st_size} bytes)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
