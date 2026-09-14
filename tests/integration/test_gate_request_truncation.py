# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A gate is shown the whole action it is asked to approve, or nothing.

``_consult_gate`` checks that the rendered action fits its request pipe before
writing it. An action longer than the pipe's whole capacity can never reach the
guard intact, so the gate's failure policy decides (veto by default) and nothing is
written. An action that would fit but finds an earlier request still unread is
parked and asked again once the guard has read, so each guard instance reads
exactly one request.

It used not to. The request was written with a plain ``Pipe.write``, the accepted
count was journalled and ignored, and the held write kept the full payload: the
guard read the opening words, allowed them, and the device received the beam lift.
Issue #17.
"""

from __future__ import annotations

from typing import Any

from zeos.core.events import (
    Event,
    FaultRaised,
    GateAnswered,
    GateConsulted,
    JobBlocked,
    PipeReadEvent,
    PipeWritten,
)
from zeos.core.gates import ALLOW, GateSpec, GateTable
from zeos.core.ids import DescriptorName, FaultKind, JobState, PipeName, Principal, Ring
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import DEFAULT_CAPACITY, PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

BARRIER = PipeName("actuators.barrier")
REQUESTS = PipeName("gates.walkway.requests")
VERDICTS = PipeName("gates.walkway.verdicts")

ACTION = "open the barrier and then lift the beam over the walkway while people are under it"

ACTOR: dict[str, Any] = {
    "name": "actor",
    "priority": 50,
    "capabilities": [{"pipe": str(BARRIER), "min_integrity": 2}],
}
GUARD: dict[str, Any] = {
    "name": "walkway-guard",
    "priority": 15,
    "pipes": {"stdin": str(REQUESTS)},
    "capabilities": [{"pipe": str(VERDICTS), "min_integrity": 2}],
}


def _run(
    request_capacity: int,
    *,
    on_gate_failure: str = "veto",
    actors: int = 1,
    guard_priority: int = 15,
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName("actor"): Descriptor.from_frontmatter(ACTOR),
            DescriptorName("walkway-guard"): Descriptor.from_frontmatter(
                {**GUARD, "priority": guard_priority}
            ),
        },
        machine=ScriptedMachine(
            {
                "actor": Script.from_spec(
                    [{"write": {"pipe": str(BARRIER), "text": ACTION}}, {"exit": True}]
                ),
                "walkway-guard": Script.from_spec(
                    [
                        {"read": str(REQUESTS)},
                        {"write": {"pipe": str(VERDICTS), "text": ALLOW}},
                        {"exit": True},
                    ]
                ),
            },
            block_size=8,
        ),
        pipes=PipeTable(
            [
                PipeSpec(BARRIER, ring=Ring.TRUSTED, principal=Principal.DEVICE),
                PipeSpec(
                    REQUESTS,
                    ring=Ring.KERNEL,
                    principal=Principal.KERNEL,
                    capacity_tokens=request_capacity,
                ),
                PipeSpec(VERDICTS, ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
            ]
        ),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        gates=GateTable(
            [
                GateSpec(
                    pipe=BARRIER,
                    descriptor=DescriptorName("walkway-guard"),
                    requests=REQUESTS,
                    verdicts=VERDICTS,
                    on_gate_failure=on_gate_failure,
                )
            ]
        ),
        journal_sink=events,
        config=KernelConfig(case="gate-truncation"),
    )
    kernel.start()
    for _ in range(actors):
        kernel.spawn(DescriptorName("actor"))
    kernel.run_until_quiescent()
    return kernel, events


def _guard_saw(events: list[Event]) -> list[str]:
    """What each guard instance read, one entry per read."""
    return [" ".join(e.text) for e in events if isinstance(e, PipeReadEvent) and e.pipe == REQUESTS]


def _device_got(events: list[Event]) -> list[str]:
    return [" ".join(e.text) for e in events if isinstance(e, PipeWritten) and e.pipe == BARRIER]


def test_an_action_that_cannot_fit_is_refused_not_truncated() -> None:
    """Four tokens of room for a sixteen-token action: the guard is never asked, the
    failure policy (veto) applies, and the device receives nothing."""
    kernel, events = _run(request_capacity=4)

    assert not [e for e in events if isinstance(e, GateConsulted)]
    assert _guard_saw(events) == []
    answered = [e for e in events if isinstance(e, GateAnswered)]
    assert answered and not answered[0].allowed
    assert "16 tokens" in answered[0].reason and "holds 4" in answered[0].reason
    faults = [e for e in events if isinstance(e, FaultRaised) and e.fault is FaultKind.GATE]
    assert faults and faults[0].pipe == BARRIER
    assert _device_got(events) == []
    assert all(j.state is JobState.FAULTED for j in kernel.sched.jobs())


def test_a_fail_open_gate_lets_an_unjudgeable_action_through_on_record() -> None:
    """The declared alternative. The action proceeds, and the journal says the guard
    was not consulted and why."""
    _, events = _run(request_capacity=4, on_gate_failure=ALLOW)

    answered = [e for e in events if isinstance(e, GateAnswered)]
    assert answered and answered[0].allowed and "applying allow" in answered[0].reason
    assert not [e for e in events if isinstance(e, GateConsulted)]
    assert _device_got(events) == [ACTION]


def test_queued_actions_are_judged_one_at_a_time() -> None:
    """Room for one request, and a guard less urgent than the actors so both actions
    are raised before it reads. The second actor parks until the first guard has
    read; each guard sees exactly one whole action and both actions reach the device."""
    kernel, events = _run(request_capacity=len(ACTION.split()), actors=2, guard_priority=90)

    parked = [e for e in events if isinstance(e, JobBlocked) and e.reason == "gate-queue"]
    assert parked and parked[0].pipe == REQUESTS
    assert _guard_saw(events) == [ACTION, ACTION]
    assert len([e for e in events if isinstance(e, GateConsulted)]) == 2
    assert _device_got(events) == [ACTION, ACTION]
    assert all(j.state is JobState.DONE for j in kernel.sched.jobs())


def test_a_guard_with_room_sees_the_whole_action() -> None:
    """The control: at the default capacity the guard reads exactly what the device
    later receives."""
    _, events = _run(request_capacity=DEFAULT_CAPACITY)

    assert _guard_saw(events) == [ACTION]
    assert _device_got(events) == [ACTION]
