# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A sink is the way out: a job's reply is checked and journalled on the way in, and the
driver drains it for the world, waking a writer parked on a full one."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from zeos.core.events import (
    CapabilityChecked,
    Event,
    FaultRaised,
    JobBlocked,
    PipeDrained,
    PipeWritten,
)
from zeos.core.ids import DescriptorName, JobState, PipeName
from zeos.core.kernel import Kernel, KernelConfig, KernelError
from zeos.core.pipes import PipeError, PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.lint import Severity, lint
from zeos.descriptor.loader import load_case
from zeos.descriptor.schema import Descriptor
from zeos.driver import Driver, build_kernel
from zeos.journal.codec import to_line
from zeos.machine.base import Token
from zeos.machine.seat import CommandSeat, TapeSource, Turn
from zeos.world.store import WorldStore

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "seat"
BOT = DescriptorName("bot")
REPLIES = PipeName("user.replies")


class Tape:
    def __init__(self, commands: Sequence[str]) -> None:
        self.commands = list(commands)

    def next_command(self, turn: Turn) -> str:
        return self.commands[turn.issued]


def _kernel(
    commands: Sequence[str], spec: PipeSpec, **frontmatter: object
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            BOT: Descriptor.from_frontmatter(
                {"name": str(BOT), "priority": 50, "pipes": {"stdout": str(spec.name)}}
                | frontmatter,
                body="b",
            )
        },
        machine=CommandSeat(source=Tape(commands)),
        pipes=PipeTable([spec]),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="sink", max_ticks=80),
    )
    kernel.start()
    return kernel, events


def _of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def _text(tokens: Sequence[Token]) -> str:
    return " ".join(t.text for t in tokens)


def test_a_write_to_a_sink_is_checked_and_journalled_and_a_drain_takes_it_out() -> None:
    kernel, events = _kernel(["write stdout hello there;", "exit;"], PipeSpec(REPLIES, sink=True))
    kernel.spawn(BOT)
    kernel.run_until_quiescent()

    assert [e.pipe for e in _of(events, CapabilityChecked)] == [REPLIES]
    assert [" ".join(w.text) for w in _of(events, PipeWritten)] == ["hello there"]
    assert kernel.pipes.get(REPLIES).available == 2

    drained = kernel.drain(REPLIES)

    assert _text(drained) == "hello there"
    assert kernel.pipes.get(REPLIES).available == 0
    assert [(d.pipe, d.tokens) for d in _of(events, PipeDrained)] == [(REPLIES, 2)]


def test_a_writer_parked_on_a_full_sink_is_woken_by_the_drain() -> None:
    kernel, events = _kernel(
        ["write stdout hello there;", "write stdout how are you;", "exit;"],
        PipeSpec(REPLIES, sink=True, capacity_tokens=3),
    )
    job = kernel.spawn(BOT)
    kernel.run_until_quiescent()
    assert job.state is JobState.BLOCKED and job.blocked_reason == "write-full"
    assert [b.pipe for b in _of(events, JobBlocked)] == [REPLIES]

    assert _text(kernel.drain(REPLIES)) == "hello there"
    kernel.run_until_quiescent()
    assert _text(kernel.drain(REPLIES)) == "how are you"
    assert job.state is JobState.DONE, "room was made, the second reply landed, the job finished"


def test_only_a_sink_can_be_drained() -> None:
    kernel, _ = _kernel(["exit;"], PipeSpec(REPLIES))
    with pytest.raises(KernelError, match="not a sink"):
        kernel.drain(REPLIES)


def test_a_job_reading_a_sink_is_faulted() -> None:
    kernel, events = _kernel(["read stdout;", "exit;"], PipeSpec(REPLIES, sink=True))
    kernel.spawn(BOT)
    kernel.run_until_quiescent()

    faults = _of(events, FaultRaised)
    assert faults and faults[0].pipe == REPLIES and "sink" in faults[0].detail


def test_a_pipe_cannot_be_both_a_sink_and_an_actuator() -> None:
    with pytest.raises(PipeError, match="both a sink and an actuator"):
        PipeSpec(REPLIES, sink=True, world_object="chat.reply")


def test_binding_a_sink_as_stdin_does_not_lint() -> None:
    d = Descriptor.from_frontmatter({"name": "x", "priority": 50, "pipes": {"stdin": str(REPLIES)}})
    findings = lint({d.name: d}, pipes=[PipeSpec(REPLIES, sink=True)])
    assert [f.rule for f in findings] == ["sink-is-read"]
    assert findings[0].severity is Severity.ERROR


def test_a_vector_may_fire_on_a_sink() -> None:
    """A sink write is still a pipe write, so "the bot replied" is a legitimate trigger."""
    from zeos.core.vectors import VectorSpec

    d = Descriptor.from_frontmatter(
        {"name": "x", "priority": 50, "pipes": {"stdout": str(REPLIES)}}
    )
    h = Descriptor.from_frontmatter({"name": "h", "priority": 5})
    vector = VectorSpec(name=d.name, source=REPLIES, handler=h.name, priority=5)  # type: ignore[arg-type]
    findings = lint({d.name: d, h.name: h}, pipes=[PipeSpec(REPLIES, sink=True)], vectors=[vector])
    assert not [f for f in findings if f.severity is Severity.ERROR], [f.render() for f in findings]


# -- the driver -----------------------------------------------------------------


def _drive() -> tuple[list[tuple[str, str]], list[Event]]:
    bundle = load_case(FIXTURE)
    events: list[Event] = []
    kernel, transport = build_kernel(
        bundle, machine=CommandSeat(source=TapeSource(bundle.scripts)), journal_sink=events
    )
    out: list[tuple[str, str]] = []
    driver = Driver(
        kernel,
        transport=transport,
        on_drain=lambda pipe, tokens: out.append((str(pipe), _text(tokens))),
    )
    driver.boot(bundle.boot)
    driver.run()
    return out, events


def test_the_driver_drains_sinks_between_ticks() -> None:
    out, events = _drive()

    assert out == [("ops.report", "all done")]
    assert [(d.pipe, d.tokens) for d in _of(events, PipeDrained)] == [(PipeName("ops.report"), 2)]
    # The drain follows the write it empties, never precedes it.
    assert events.index(_of(events, PipeDrained)[0]) > events.index(_of(events, PipeWritten)[0])


def test_a_run_with_drains_is_the_same_run_twice() -> None:
    _, first = _drive()
    _, again = _drive()
    assert [to_line(i, e) for i, e in enumerate(first)] == [
        to_line(i, e) for i, e in enumerate(again)
    ]


def test_the_cli_prints_what_leaves_a_sink(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from zeos.cli import main

    journal = tmp_path / "seat.jsonl"
    assert main(["run", str(FIXTURE), "--machine", "seat", "--journal", str(journal)]) == 0
    assert "ops.report ◀── all done" in capsys.readouterr().out
    assert main(["replay", str(journal), "--assert-identical"]) == 0

    assert main(["run", str(FIXTURE), "--machine", "seat", "--quiet"]) == 0
    assert "all done" not in capsys.readouterr().out
