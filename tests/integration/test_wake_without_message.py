# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A job woken from a read does not proceed until it has actually read.

A parked operation is retried through the operation that parked it. ``_wake_readers``
wakes every reader on the pipe; when the loser reaches ``_service_pending`` its
read is retried through ``_do_read``, which finds the pipe empty and parks it again.
A ``WRITE_READ`` whose write is parked -- by backpressure or by a gate -- keeps its
read half in ``pending_write``, and the retried write performs the read when it
completes.

It used not to. The loser's ``pending_read`` was cleared and the tick went on to
decode, as if the read had completed; and a parked write-then-read remembered only
the write, so the job never waited for the reply it asked for. Issue #16.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.events import Decoded, Event, GateAnswered, JobBlocked, JobWoken, PipeReadEvent
from zeos.core.gates import ALLOW, GateSpec, GateTable
from zeos.core.ids import DescriptorName, JobState, PipeName, Principal, Ring
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

CMD = PipeName("user.cmd")
ASK = PipeName("peer.a2b")
REPLY = PipeName("peer.b2a")


def _build(
    descriptors: Sequence[Mapping[str, Any]],
    scripts: Mapping[str, list[dict[str, Any]]],
    pipes: Sequence[PipeSpec],
    gates: GateTable | None = None,
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in scripts.items()}, block_size=8),
        pipes=PipeTable(pipes),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        gates=gates,
        journal_sink=events,
        config=KernelConfig(case="wake-without-message", max_ticks=300),
    )
    kernel.start()
    return kernel, events


def _of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


PARK = {"read": {"read": str(CMD)}, "select": {"select": [str(CMD)]}}


@pytest.mark.parametrize("parked_on", sorted(PARK), ids=str)
def test_the_reader_that_lost_the_race_keeps_waiting(parked_on: str) -> None:
    reader = {"priority": 50, "pipes": {"stdin": str(CMD)}}
    kernel, events = _build(
        [{"name": "r1", **reader}, {"name": "r2", **reader, "priority": 60}],
        {
            "r1": [PARK[parked_on], {"emit": "r1-got-it"}, {"exit": True}],
            "r2": [PARK[parked_on], {"emit": "r2-got-it"}, {"exit": True}],
        },
        [PipeSpec(name=CMD)],
    )
    r1 = kernel.spawn(DescriptorName("r1"))
    r2 = kernel.spawn(DescriptorName("r2"))
    kernel.run_until_quiescent()
    assert r1.state is JobState.BLOCKED and r2.state is JobState.BLOCKED

    kernel.deliver(CMD, "hello")
    kernel.run_until_quiescent()

    reads = [(e.job, e.text) for e in _of(events, PipeReadEvent) if e.pipe == CMD]
    assert reads == [(r1.job_id, ("hello",))], "one message, one read"
    assert r1.state is JobState.DONE
    decoded_by_loser = [e for e in _of(events, Decoded) if e.job == r2.job_id]
    assert not decoded_by_loser and r2.state is JobState.BLOCKED, (
        "the second reader was woken, found the pipe empty, and carried on as if it "
        "had read something"
    )

    kernel.deliver(CMD, "hello again")
    kernel.run_until_quiescent()
    assert r2.state is JobState.DONE
    assert [e for e in _of(events, PipeReadEvent) if e.job == r2.job_id]


def test_a_delayed_write_still_waits_for_its_reply() -> None:
    """The ask pipe holds two tokens and is already full, so the write parks. A
    consumer drains it; the write goes through; the job must then block for the
    reply as it asked to."""
    asker = {"name": "asker", "priority": 50, "pipes": {"tools": str(ASK), "stdin": str(REPLY)}}
    consumer = {"name": "consumer", "priority": 90, "pipes": {"stdin": str(ASK)}}
    kernel, events = _build(
        [asker, consumer],
        {
            "asker": [
                {"write": {"pipe": str(ASK), "text": "hello there", "then_read": str(REPLY)}},
                {"emit": "asker-continued"},
                {"exit": True},
            ],
            "consumer": [{"read": str(ASK)}, {"exit": True}],
        },
        [PipeSpec(name=ASK, capacity_tokens=2), PipeSpec(name=REPLY)],
    )
    kernel.deliver(ASK, "x y")
    job = kernel.spawn(DescriptorName("asker"))
    kernel.run_until_quiescent()
    assert job.state is JobState.BLOCKED and job.blocked_reason == "write-full"

    kernel.spawn(DescriptorName("consumer"))
    kernel.run_until_quiescent()

    assert [e for e in _of(events, JobWoken) if e.job == job.job_id], "the drain woke the writer"
    assert job.state is JobState.BLOCKED and job.blocked_on == REPLY, (
        "the write went through and the job carried on without waiting for the "
        "reply it asked to read"
    )
    kernel.deliver(REPLY, "here you go")
    kernel.run_until_quiescent()
    assert job.state is JobState.DONE
    assert [e for e in _of(events, PipeReadEvent) if e.job == job.job_id and e.pipe == REPLY]


REQUESTS = PipeName("gates.ask.requests")
VERDICTS = PipeName("gates.ask.verdicts")


def test_a_gated_write_then_read_waits_for_its_reply() -> None:
    """The other way a write parks: held for a guard. The read half must survive
    the verdict too."""
    asker = {
        "name": "asker",
        "priority": 50,
        "pipes": {"tools": str(ASK), "stdin": str(REPLY)},
        "capabilities": [{"pipe": str(ASK), "min_integrity": 2}],
    }
    guard = {
        "name": "guard",
        "priority": 15,
        "pipes": {"stdin": str(REQUESTS)},
        "capabilities": [{"pipe": str(VERDICTS), "min_integrity": 2}],
    }
    kernel, events = _build(
        [asker, guard],
        {
            "asker": [
                {"write": {"pipe": str(ASK), "text": "hello there", "then_read": str(REPLY)}},
                {"emit": "asker-continued"},
                {"exit": True},
            ],
            "guard": [
                {"read": str(REQUESTS)},
                {"write": {"pipe": str(VERDICTS), "text": ALLOW}},
                {"exit": True},
            ],
        },
        [
            PipeSpec(name=ASK, ring=Ring.TRUSTED, principal=Principal.DEVICE),
            PipeSpec(name=REPLY),
            PipeSpec(name=REQUESTS, ring=Ring.KERNEL, principal=Principal.KERNEL),
            PipeSpec(name=VERDICTS, ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
        ],
        gates=GateTable([GateSpec(ASK, DescriptorName("guard"), REQUESTS, VERDICTS)]),
    )
    job = kernel.spawn(DescriptorName("asker"))
    kernel.run_until_quiescent()

    answered = _of(events, GateAnswered)
    assert answered and answered[0].allowed, "the guard was meant to allow it"
    assert job.state is JobState.BLOCKED and job.blocked_on == REPLY

    kernel.deliver(REPLY, "here you go")
    kernel.run_until_quiescent()
    assert job.state is JobState.DONE
    assert [e for e in _of(events, PipeReadEvent) if e.job == job.job_id and e.pipe == REPLY]


def test_an_unhindered_write_then_read_waits_for_its_reply() -> None:
    """The control: no backpressure, so the read half is reached and the job parks."""
    asker = {"name": "asker", "priority": 50, "pipes": {"tools": str(ASK), "stdin": str(REPLY)}}
    kernel, events = _build(
        [asker],
        {
            "asker": [
                {"write": {"pipe": str(ASK), "text": "hello there", "then_read": str(REPLY)}},
                {"emit": "asker-continued"},
                {"exit": True},
            ]
        },
        [PipeSpec(name=ASK), PipeSpec(name=REPLY)],
    )
    job = kernel.spawn(DescriptorName("asker"))
    kernel.run_until_quiescent()
    assert job.state is JobState.BLOCKED and job.blocked_on == REPLY
    assert [e for e in _of(events, JobBlocked) if e.job == job.job_id and e.pipe == REPLY]

    kernel.deliver(REPLY, "here you go")
    kernel.run_until_quiescent()
    assert job.state is JobState.DONE
