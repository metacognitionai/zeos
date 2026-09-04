# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate the golden run-shape trace.

    uv run python -m tests.replay.regenerate

Run this only after an *intentional* behavioural change, and read the diff it prints
before committing. A golden fixture that gets regenerated reflexively is worse than no
fixture, because it looks like a check and is not one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tests.replay.test_golden_traces import GOLDEN, run, shape_digest


def main() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    target = GOLDEN / "smoke.shape"
    was = target.read_text("utf-8").strip() if target.is_file() else "(none)"
    with tempfile.TemporaryDirectory() as tmp:
        kinds = run(Path(tmp) / "run.jsonl")
    now = shape_digest(kinds)
    target.write_text(now + "\n", encoding="utf-8")
    verb = "unchanged" if was == now else f"CHANGED from {was}"
    print(f"smoke.shape  {now}  ({len(kinds)} events)  {verb}")


if __name__ == "__main__":
    main()
