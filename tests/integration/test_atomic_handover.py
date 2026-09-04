# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Handing over and going to sleep is one operation, not two.

The universal shape of a turn in a pipeline is *hand over, then sleep* -- ``write
stdout go; read stdin;`` -- and the first of those is what makes the peer eligible to
displace the second. ``Kernel.tick`` checks ``should_preempt`` at the top of every
tick and a ``MachineRequest`` carried exactly one ``op``, so the two landed in
different ticks with a preemption check between them. A peer that outranks the running
job took the machine there, one command short of the job yielding it voluntarily.

The consequences are quiet, because the values still come out right: the job is
displaced inside its own handover on every turn, so reads it was written to make never
happen, ``RESUME_DIRTY`` becomes routine traffic rather than a signal, and the
suspension stack churns once a turn so a genuine interrupt arrives into machinery
already busy. Against a real model the job stopped performing pipe reads altogether;
the scripted peer below is more forgiving and merely misses some. At equal priority
none of it appears, because ``outranks`` is strict -- change either priority by one and
it opens on every turn.

``OpKind.WRITE_READ`` is the combined send-and-wait that message-passing kernels have
for the same reason -- L4's ``Call``, QNX's ``MsgSend``, and ``pthread_cond_wait``'s
atomic unlock-and-wait from the other direction. ZEOS has the sharper version, because
it deliberately has no yield (Appendix A rule 2): a job's only way off the machine is a
blocking read, so it must advertise its own replacement before it can get there.
"""

from __future__ import annotations

from typing import Any

from zeos.core.events import Event, JobPreempted, PipeReadEvent, PipeWritten
from zeos.core.ids import DescriptorName, JobId, JobState, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

A2B = PipeName("peers.a2b")
B2A = PipeName("peers.b2a")
TURNS = 6


def _turn(out: PipeName, back: PipeName, *, atomic: bool) -> list[dict[str, Any]]:
    """``TURNS`` turns of hand-over-then-sleep, as one op or as two."""
    steps: list[dict[str, Any]] = []
    for i in range(TURNS):
        write: dict[str, Any] = {"pipe": str(out), "text": "go"}
        if atomic:
            write["then_read"] = str(back)
            steps += [{"emit": f"count {i}"}, {"write": write}]
        else:
            steps += [{"emit": f"count {i}"}, {"write": write}, {"read": "stdin"}]
    return steps + [{"exit": True}]


def _build(
    *, atomic: bool, b_priority: int, capacity: int = 64
) -> tuple[Kernel, list[Event], dict[JobId, str]]:
    a: dict[str, Any] = {
        "name": "peer-a",
        "priority": 50,
        "pipes": {"stdin": str(B2A), "stdout": str(A2B)},
    }
    b: dict[str, Any] = {
        "name": "peer-b",
        "priority": b_priority,
        "pipes": {"stdin": str(A2B), "stdout": str(B2A)},
    }
    machine = ScriptedMachine(
        {
            "peer-a": Script.from_spec(_turn(A2B, B2A, atomic=atomic)),
            "peer-b": Script.from_spec(_turn(B2A, A2B, atomic=atomic)),
        },
        block_size=16,
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d, body=f"be {d['name']}")
            for d in (a, b)
        },
        machine=machine,
        pipes=PipeTable(
            [
                PipeSpec(name=A2B, capacity_tokens=capacity),
                PipeSpec(name=B2A, capacity_tokens=capacity),
            ]
        ),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="atomic-handover", max_ticks=600),
    )
    kernel.start()
    kernel.spawn(DescriptorName("peer-a"))
    kernel.spawn(DescriptorName("peer-b"))
    kernel.run_until_quiescent()
    names = {j.job_id: str(j.name) for j in kernel.sched.jobs()}
    return kernel, events, names


def _who(event: Event, names: dict[JobId, str]) -> str:
    job: JobId | None = getattr(event, "job", None)
    return "" if job is None else names.get(job, "")


def _count(events: list[Event], names: dict[JobId, str], cls: type[Event], name: str) -> int:
    return len([e for e in events if isinstance(e, cls) and _who(e, names) == name])


def _preemptions_inside_a_handover(events: list[Event], names: dict[JobId, str], name: str) -> int:
    """How often a preemption lands between the job's write and its own following read.

    This is the gap itself, rather than a proxy for it: a job preempted *after* it has
    read is ordinary scheduling, and closing the gap is not supposed to stop that.
    """
    inside = False
    landed = 0
    for event in events:
        if _who(event, names) != name:
            continue
        if isinstance(event, PipeWritten):
            inside = True
        elif isinstance(event, PipeReadEvent):
            inside = False
        elif isinstance(event, JobPreempted) and inside:
            landed += 1
    return landed


def test_the_split_handover_is_preempted_inside_every_turn() -> None:
    """The bug, at unequal priority. Kept as the control, since this is also what a
    backend that goes on emitting separate WRITE and READ still gets."""
    _, events, names = _build(atomic=False, b_priority=60)

    assert _preemptions_inside_a_handover(events, names, "peer-b") == TURNS
    assert _count(events, names, PipeReadEvent, "peer-b") < TURNS, "reads it never reached"
    assert _preemptions_inside_a_handover(events, names, "peer-a") == 0


def test_the_compound_handover_closes_the_gap() -> None:
    """The fix. The peer is still woken by the write and still becomes READY; it simply
    cannot be dispatched until the writer is already blocked, which is where it would
    have ended up one tick later anyway."""
    _, events, names = _build(atomic=True, b_priority=60)

    assert _preemptions_inside_a_handover(events, names, "peer-b") == 0
    assert _count(events, names, PipeReadEvent, "peer-b") == TURNS, "every read reached"


def test_equal_priority_hid_the_gap_either_way() -> None:
    """Why this went unnoticed for so long: ``outranks`` is strict, so a woken peer at
    50 does not displace a runner at 50 and the job always reaches its read."""
    for atomic in (False, True):
        _, events, names = _build(atomic=atomic, b_priority=50)
        assert _preemptions_inside_a_handover(events, names, "peer-b") == 0
        assert _preemptions_inside_a_handover(events, names, "peer-a") == 0


def test_a_write_that_cannot_land_does_not_commit_the_read() -> None:
    """The halves are ordered, not simultaneous.

    A write parked by backpressure has not happened, so the job must be waiting on the
    pipe it could not write -- not blocked on a read it never reached.
    """
    out, inbound = PipeName("narrow.out"), PipeName("narrow.in")
    descriptor = Descriptor.from_frontmatter(
        {"name": "writer", "priority": 50, "pipes": {"stdin": str(inbound), "stdout": str(out)}},
        body="write then wait",
    )
    machine = ScriptedMachine(
        {
            "writer": Script.from_spec(
                [
                    {
                        "write": {
                            "pipe": str(out),
                            "text": "far too many tokens",
                            "then_read": str(inbound),
                        }
                    }
                ]
            )
        },
        block_size=16,
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors={DescriptorName("writer"): descriptor},
        machine=machine,
        pipes=PipeTable(
            [PipeSpec(name=out, capacity_tokens=1), PipeSpec(name=inbound, capacity_tokens=8)]
        ),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="narrow-handover", max_ticks=50),
    )
    kernel.start()
    job = kernel.spawn(DescriptorName("writer"))
    kernel.run_until_quiescent()

    assert job.state is JobState.BLOCKED
    assert job.blocked_reason == "write-full", "parked by backpressure, not by the read"
    assert job.blocked_on == out
    assert job.pending_read is None, "no read was committed behind a write that never landed"
    assert not [e for e in events if isinstance(e, PipeReadEvent)]
