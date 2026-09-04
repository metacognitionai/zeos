# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``pinned: true`` means resident and prefilled, but not yet runnable.

``JobState.PINNED_IDLE`` was declared at ``ids.py:93`` and referenced nowhere else, and
``Job.is_pinned`` had no consumers, so a descriptor could ask to be pinned, see it
echoed in the journal, and get nothing. ``Transformer-OS.md`` documents the state
(:304) and its transition (:321) as real behaviour.

What the gap costs is a peer that cannot say "spawned, resident, but not my turn".
Booted READY, it is a dispatch candidate from the start, so the moment the job ahead of
it blocks it is given the machine -- with nothing yet handed to it and its first
command, a read, not yet run. A real model dispatched in that position answers from
whatever is already in its context instead of waiting.

A pinned job enters the state only where something can leave it: a write to its
declared ``stdin``. A handler binding no input is addressed by its vector, which spawns
it fresh at the moment it fires, so it boots READY exactly as before.
"""

from __future__ import annotations

from typing import Any

from zeos.core.events import (
    Event,
    JobDispatched,
    JobStateChanged,
    PipeReadEvent,
    PipeWritten,
)
from zeos.core.ids import DescriptorName, JobState, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

CMD = PipeName("user.cmd")
A2B = PipeName("peers.a2b")

#: ``a`` waits to be started, then hands over to ``b``; ``b`` only ever answers.
A_SCRIPT = [{"read": "stdin"}, {"emit": "starting"}, {"write": {"pipe": str(A2B)}}, {"exit": True}]
B_SCRIPT = [{"emit": "answering"}, {"exit": True}]


def _build(*, pin_b: bool) -> tuple[Kernel, list[Event]]:
    a: dict[str, Any] = {"name": "peer-a", "priority": 50, "pipes": {"stdin": str(CMD)}}
    b: dict[str, Any] = {"name": "peer-b", "priority": 50, "pipes": {"stdin": str(A2B)}}
    if pin_b:
        b["pinned"] = True
    machine = ScriptedMachine(
        {"peer-a": Script.from_spec(A_SCRIPT), "peer-b": Script.from_spec(B_SCRIPT)}, block_size=16
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d, body=f"be {d['name']}")
            for d in (a, b)
        },
        machine=machine,
        pipes=PipeTable([PipeSpec(name=CMD, device=True), PipeSpec(name=A2B)]),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="pinned-idle", max_ticks=400),
    )
    kernel.start()
    kernel.spawn(DescriptorName("peer-a"))
    kernel.spawn(DescriptorName("peer-b"))
    return kernel, events


def _job(kernel: Kernel, name: str) -> Any:
    return next(j for j in kernel.sched.jobs() if j.name == DescriptorName(name))


def _ran(kernel: Kernel, events: list[Event], name: str) -> bool:
    job = _job(kernel, name).job_id
    return any(isinstance(e, JobDispatched) and e.job == job for e in events)


def test_a_pinned_peer_boots_resident_but_not_runnable() -> None:
    """Resident *and* prefilled -- the body is in context before it is a candidate.

    Deferring prefill to first dispatch would make pinning pointless: the latency
    argument (Transformer-OS §5.3) is that the prompt prefix is already paid for.
    """
    kernel, _ = _build(pin_b=True)
    peer_b = _job(kernel, "peer-b")

    assert peer_b.state is JobState.PINNED_IDLE
    assert peer_b.started, "prefilled at spawn, not deferred to first dispatch"
    assert peer_b.segments.all(), "its body is resident"
    assert kernel.sched.best_ready() is _job(kernel, "peer-a"), "not a dispatch candidate"


def test_a_pinned_peer_declines_the_machine_until_something_addresses_it() -> None:
    """The behaviour the state buys. ``peer-a`` blocks with nothing handed over."""
    kernel, events = _build(pin_b=True)
    kernel.run_until_quiescent()

    assert _job(kernel, "peer-a").state is JobState.BLOCKED
    assert not _ran(kernel, events, "peer-b"), "nothing had addressed it, so it did not run"


def test_an_unpinned_peer_takes_the_machine_it_was_written_to_wait_for() -> None:
    """The control, so the fix is not credited with more than it does."""
    kernel, events = _build(pin_b=False)
    kernel.run_until_quiescent()

    assert _job(kernel, "peer-a").state is JobState.BLOCKED
    assert _ran(kernel, events, "peer-b"), "dispatched with nothing handed over"


def test_the_handover_wakes_it_and_the_wake_is_journalled() -> None:
    """And it runs after the write, not before -- replay has to see the transition."""
    kernel, events = _build(pin_b=True)
    kernel.run_until_quiescent()

    kernel.deliver(CMD, "start")
    kernel.run_until_quiescent()

    moves = [
        e
        for e in events
        if isinstance(e, JobStateChanged)
        and e.job == _job(kernel, "peer-b").job_id
        and e.from_state is JobState.PINNED_IDLE
    ]
    assert [e.to_state for e in moves] == [JobState.READY]

    order = [e for e in events if isinstance(e, JobDispatched | PipeWritten)]
    handover = next(i for i, e in enumerate(order) if isinstance(e, PipeWritten) and e.pipe == A2B)
    first_b = next(
        i
        for i, e in enumerate(order)
        if isinstance(e, JobDispatched) and e.job == _job(kernel, "peer-b").job_id
    )
    assert handover < first_b, "woken by the handover, not before it"


def test_a_case_left_holding_only_pinned_jobs_is_idle_not_stuck() -> None:
    """Pinned-idle is a resting state -- most of a resident handler's life is spent in
    it -- so a case with nothing else left must read as idle, not as one that failed to
    terminate."""
    kernel, _ = _build(pin_b=True)
    _job(kernel, "peer-a").state = JobState.DONE

    kernel.run_until_quiescent()

    assert _job(kernel, "peer-b").state is JobState.PINNED_IDLE


def test_being_addressed_delivers_the_payload_that_addressed_it() -> None:
    """Promotion out of PINNED_IDLE is a delivery, not just a state change.

    The symmetry that matters is with a vector: firing one takes the source payload and
    injects it into the handler it spawns, so the handler wakes with the message already
    in context. A pinned peer is addressed the same way -- by a write to the input its
    descriptor declared -- and must be delivered to the same way.

    Leaving the payload queued instead is not a smaller version of the same behaviour;
    it is the bug pinning exists to fix, moved one turn later. The job is dispatched at
    the right moment, does its work, and then its own closing ``read stdin;`` finds the
    token that woke it still sitting in the pipe, returns immediately rather than
    blocking, and it takes a second turn on it. Downstream that is exactly what it looked
    like: two counters that alternated correctly except for one duplicated turn at the
    very start, which pinning alone did not remove.
    """
    kernel, events = _build(pin_b=True)
    kernel.deliver(CMD, "go")
    kernel.run_until_quiescent()

    b = _job(kernel, "peer-b")
    promoted = [
        e
        for e in events
        if isinstance(e, JobStateChanged)
        and e.job == b.job_id
        and e.from_state is JobState.PINNED_IDLE
    ]
    assert promoted, "peer-b was never addressed"

    assert kernel.pipes.get(A2B).available == 0, (
        "the write that addressed the pinned job is still queued behind it, so its own "
        "first read will take it and return without blocking"
    )
    delivered = [e for e in events if isinstance(e, PipeReadEvent) and e.job == b.job_id]
    assert delivered and delivered[0].pipe == A2B, (
        "being addressed must deliver, the way firing a vector delivers to its handler"
    )
