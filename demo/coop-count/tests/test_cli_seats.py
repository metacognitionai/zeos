# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the seats the CLI builds, with the network replaced by the case's own tape."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from zeos.core.ids import JobId
from zeos.descriptor.loader import load_case
from zeos_coop_count import claude as claude_mod
from zeos_coop_count.cli import main
from zeos_coop_count.scripted import TapeSource
from zeos_coop_count.seat import CommandSeat, Turn

CASE = Path(__file__).resolve().parent.parent / "cases" / "coop-count-scripted"

#: The seat builds a client at construction, which insists on a key it never uses here.
os.environ.setdefault("ANTHROPIC_API_KEY", "not-used-by-this-suite")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stands in for ``ClaudeSource`` so the claude path can be run without an API key."""
    tape = TapeSource(load_case(CASE).scripts)

    class Offline:
        def __init__(self, **_: Any) -> None:
            pass

        def next_command(self, turn: Turn) -> str:
            return tape.next_command(turn)

    monkeypatch.setattr(claude_mod, "ClaudeSource", Offline)


@pytest.mark.usefixtures("offline")
@pytest.mark.parametrize("flags", [(), ("--seat-model", "some-model")])
def test_the_claude_seat_runs_to_the_end(tmp_path: Path, flags: tuple[str, ...]) -> None:
    """A run with no llama model in it must not try to free one on the way out."""
    journal = tmp_path / "claude.jsonl"
    code = main(["run", str(CASE), "--machine", "claude", "--journal", str(journal), *flags])

    assert code == 0
    assert journal.read_text().splitlines(), "the run wrote no journal"


class _Reply:
    """The shape of the API response the seat reads: one text block and a stop reason."""

    type = "text"
    text = "say 41;"
    stop_reason = "end_turn"

    @property
    def content(self) -> list[_Reply]:
        return [self]


class _Recorder:
    """Stands in for the Anthropic client and keeps what the seat asked for."""

    def __init__(self) -> None:
        self.sent: dict[str, Any] = {}
        self.beta = self
        self.messages = self

    def create(self, **kwargs: Any) -> _Reply:
        self.sent = kwargs
        return _Reply()


def _asked_with(model: str | None) -> dict[str, Any]:
    from zeos_coop_count.claude import ClaudeSource

    source = ClaudeSource(**({"model": model} if model else {}))
    recorder = _Recorder()
    source._client = recorder  # pyright: ignore[reportPrivateUsage]
    source.next_command(Turn(job=JobId(1), descriptor="counter-a", transcript="one", issued=0))
    return recorder.sent


def test_the_default_model_is_asked_for_server_side_fallback() -> None:
    assert _asked_with(None)["fallbacks"] == "default"


def test_another_model_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models that do not offer fallback refuse the whole request with a 400."""
    sent = _asked_with("claude-sonnet-5")

    assert "fallbacks" not in sent and "betas" not in sent
    assert sent["model"] == "claude-sonnet-5"


def _invocation(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """What the claude-code source would run, with the process itself stubbed out."""
    from zeos_coop_count import claude_code

    captured: dict[str, Any] = {}

    class _Finished:
        returncode = 0
        stdout = "say 41;"
        stderr = ""

    def fake_run(argv: list[str], **kwargs: Any) -> _Finished:
        captured["argv"] = argv
        captured.update(kwargs)
        return _Finished()

    monkeypatch.setattr(claude_code.subprocess, "run", fake_run)
    source = claude_code.ClaudeCodeSource()
    source.next_command(Turn(job=JobId(1), descriptor="counter-a", transcript="one", issued=0))
    # The directory lives as long as the source does, so the caller gets both.
    captured["source"] = source
    return captured


def test_the_process_runs_where_no_claude_md_can_reach_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`claude --print` reads the CLAUDE.md of its working directory into the model's
    context whatever `--system-prompt` says, and this repository has one."""
    where = Path(_invocation(monkeypatch)["cwd"])

    assert where.is_dir() and not any(where.iterdir()), "the seat must start somewhere empty"
    assert not (where / "CLAUDE.md").exists()
    assert Path(__file__).resolve().parent not in where.parents


def test_the_seat_asks_for_one_command_and_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _invocation(monkeypatch)["argv"]

    assert argv[:2] == ["claude", "--print"]
    assert argv[argv.index("--allowedTools") + 1] == "", "a command is words, never a tool call"
    assert "--system-prompt" in argv


def test_a_job_is_told_the_command_it_has_just_issued() -> None:
    """Without it a model reissues what it just said: the goal's closing line stays
    literally true-looking, and the job's own working is words in the same flat run."""
    from zeos_coop_count.claude import prompt_for

    first = prompt_for(Turn(job=JobId(1), descriptor="counter-a", transcript="c", issued=0))
    later = prompt_for(
        Turn(job=JobId(1), descriptor="counter-a", transcript="c", issued=1, last="say 41")
    )

    assert "last command" not in first, "nothing has been issued yet"
    assert "`say 41;` and it is done" in later


def test_the_seat_carries_the_last_command_into_the_turn() -> None:
    """The seat knows it from the parser; the source should not have to keep its own."""
    seen: list[str] = []

    class Watching:
        def next_command(self, turn: Turn) -> str:
            seen.append(turn.last)
            return "say 41" if not turn.last else "read stdin"

    seat = CommandSeat(source=Watching())
    seat.create_context(JobId(1), "counter-a")
    for _ in range(4):
        seat.decode(JobId(1), allow_control=False)

    assert seen == ["", "say 41"]


def test_the_subscription_seat_asks_for_the_same_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    from zeos_coop_count import claude, claude_code

    argv = _invocation(monkeypatch)["argv"]

    assert argv[argv.index("--effort") + 1] == claude_code.EFFORT == claude.EFFORT
