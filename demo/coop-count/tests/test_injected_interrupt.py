# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Tests for `--interrupt`: the keypress delivered on a condition rather than at a time."""

from __future__ import annotations

import json
from pathlib import Path

from zeos_coop_count.cli import main

CASE = Path(__file__).resolve().parent.parent / "cases" / "coop-count-scripted"


def play(tmp_path: Path, name: str, *flags: str) -> list[str]:
    journal = tmp_path / f"{name}.jsonl"
    assert main(["run", str(CASE), "--machine", "scripted", "--journal", str(journal), *flags]) == 0
    return journal.read_text().splitlines()


def test_a_said_number_presses_the_key_where_a_time_would_have(tmp_path: Path) -> None:
    """The condition lands the interrupt exactly where the case's own schedule puts it."""
    scheduled = play(tmp_path, "scheduled", "--events", str(CASE / "events.jsonl"))
    triggered = play(tmp_path, "triggered", "--interrupt", "15", "51")

    assert triggered == scheduled


def test_without_it_nothing_presses_the_key(tmp_path: Path) -> None:
    """No flag, no schedule, no keypress -- and the tape plays on regardless."""
    events = [json.loads(line) for line in play(tmp_path, "quiet")]

    kinds = {e["kind"] for e in events}
    assert "vector.fired" not in kinds and "job.preempted" not in kinds
    written = [(e["obj"], e["after"]) for e in events if e["kind"] == "world.written"]
    assert written == [("count.a", "10"), ("count.b", "60")], (
        "the tape plays the same numbers; only the reset in the middle is missing"
    )
