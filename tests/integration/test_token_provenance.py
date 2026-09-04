# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What moved, and where it is now: the journal answers both, or the debugger lies.

The debugger reconstructs a pipe's buffer by replaying writes, reads and latches out
of the journal. That reconstruction is only trustworthy if the journal carries every
mutation, so the load-bearing assertion here compares the fold's idea of each pipe
against the kernel's own buffers after the same run. It is a journal-completeness
test wearing a monitor's clothes: the two ways the pipe could disagree are a write
the record does not carry and a read it does not carry, and both have happened.

Token *text* is journalled for the same reason the owner on ``JobSpawned`` is. A
fact the record does not carry is a fact nobody can check after the run, and "what
did this job actually say" was previously answerable only from a live machine.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from zeos.core.events import Decoded, Injected, PipeReadEvent, PipeWritten
from zeos.core.ids import PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.descriptor.loader import CaseBundle, load_case
from zeos.driver import Driver, ScheduledEvent, build_kernel
from zeos.journal.writer import Journal, JournalRecord
from zeos.monitor.state import PIPE_PREVIEW, SystemView, fold

SMOKE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"

#: Mid-flight for the goal job, so the run contains the interrupt whose payload the
#: vector drains -- the mutation the journal used to leave out.
ALARM = (ScheduledEvent(at_ns=3_000_000, pipe=PipeName("sensors.threshold"), text="level 42"),)


@pytest.fixture(scope="module")
def run() -> tuple[Kernel, Sequence[JournalRecord]]:
    bundle: CaseBundle = load_case(SMOKE)
    kernel, transport = build_kernel(bundle, config=KernelConfig(case=bundle.name), block_size=8)
    journal = Journal()
    driver = Driver(kernel, transport=transport, journal=journal)
    driver.boot(bundle.boot)
    driver.run(ALARM)
    return kernel, journal.records


@pytest.fixture(scope="module")
def final(run: tuple[Kernel, Sequence[JournalRecord]]) -> SystemView:
    _kernel, records = run
    return fold(r.event for r in records).final


def test_the_fold_reconstructs_every_pipe_the_kernel_holds(
    run: tuple[Kernel, Sequence[JournalRecord]], final: SystemView
) -> None:
    """The whole basis of the contents pane: replaying the journal lands on the
    buffers the kernel actually has, token for token."""
    kernel, _records = run
    for view in final.pipes:
        live = kernel.pipes.get(view.name)
        assert view.depth == live.available, f"{view.name}: depth drifted from the kernel's"
        assert list(view.contents) == [t.text for t in live.buffer][:PIPE_PREVIEW], (
            f"{view.name}: reconstructed contents differ from the kernel's buffer"
        )


def test_an_actuator_write_replaces_rather_than_accumulates(
    run: tuple[Kernel, Sequence[JournalRecord]], final: SystemView
) -> None:
    """``Pipe.latch`` clears before it writes, because an actuator pipe holds the
    current value rather than a backlog. A fold that treats every write as an append
    over-reports depth on exactly these pipes."""
    kernel, records = run
    latched = [
        r.event
        for r in records
        if isinstance(r.event, PipeWritten) and r.event.pipe == PipeName("actuators.unit_a")
    ]
    assert latched, "the fixture no longer actuates; this test has lost its subject"
    assert all(e.latched for e in latched), "a write to a world-object pipe is a latch"

    view = next(p for p in final.pipes if p.name == PipeName("actuators.unit_a"))
    assert view.written == sum(e.tokens for e in latched)
    assert view.depth == kernel.pipes.get(view.name).available


def test_the_vectors_drain_is_journalled(run: tuple[Kernel, Sequence[JournalRecord]]) -> None:
    """Firing a vector consumes its source payload. Until that read was journalled it
    was the one pipe mutation the record did not carry, and a replay left the payload
    sitting in the buffer for the rest of the run."""
    _kernel, records = run
    source = PipeName("sensors.threshold")
    reads = [
        r.event for r in records if isinstance(r.event, PipeReadEvent) and r.event.pipe == source
    ]
    assert [e.text for e in reads] == [("level", "42")]

    view = next(p for p in fold(r.event for r in records).final.pipes if p.name == source)
    assert (view.depth, view.contents) == (0, ())


def test_every_token_bearing_event_carries_its_text(
    run: tuple[Kernel, Sequence[JournalRecord]],
) -> None:
    """Counts and text must agree. A text field that silently disagreed with the
    count beside it would make the stream pane a second source of truth."""
    _kernel, records = run
    seen = 0
    for record in records:
        match record.event:
            case Decoded() | Injected() | PipeWritten() | PipeReadEvent() as event:
                assert len(event.text) == event.tokens, f"{type(event).KIND} miscounts its text"
                seen += 1
            case _:
                pass
    assert seen, "the fixture moved no tokens; this test has lost its subject"


def test_what_a_job_generated_is_answerable_from_the_journal(
    run: tuple[Kernel, Sequence[JournalRecord]],
) -> None:
    """The thing the record could not answer before: what did this job say."""
    _kernel, records = run
    said = [
        " ".join(r.event.text)
        for r in records
        if isinstance(r.event, Decoded) and int(r.event.job) == 1
    ]
    assert "reviewing the production schedule for this shift" in said


def test_an_injection_names_the_pipe_it_arrived_on(
    run: tuple[Kernel, Sequence[JournalRecord]],
) -> None:
    """Provenance is stamped at INJECT, so the tokens and the pipe they entered
    through are one fact rather than two that have to be joined by timing."""
    _kernel, records = run
    arrivals = [
        (str(r.event.pipe), " ".join(r.event.text))
        for r in records
        if isinstance(r.event, Injected)
    ]
    assert ("sensors.threshold", "level 42") in arrivals
