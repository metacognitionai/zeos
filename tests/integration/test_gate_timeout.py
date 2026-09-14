# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A guard that never answers is answered by the gate's failure policy at its deadline.

GateSpec.timeout_ticks: "Ticks the actuating job may wait before the failure policy
applies ... an unbounded wait would turn a stuck gate into a silently stalled actuator."
The kernel checks every held write against its deadline once per tick; an overdue one is
settled as a guard that faulted would be, and the stalled guard is cancelled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import Event, FaultRaised, GateAnswered, JobCancelled
from zeos.core.gates import ALLOW, GateSpec, GateTable
from zeos.core.ids import DescriptorName, JobState, ObjectName, PipeName, Principal
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.base import MachineBackend
from zeos.machine.seat import CommandSeat, Turn
from zeos.world.store import WorldStore


class Tapes:
    """One command list per descriptor, played in order; a stand-in for a model."""

    def __init__(self, by: Mapping[str, Sequence[str]]) -> None:
        self.by = {k: list(v) for k, v in by.items()}

    def next_command(self, turn: Turn) -> str:
        return self.by[turn.descriptor][turn.issued]


def _build(
    descriptors: Sequence[Mapping[str, Any]],
    machine: MachineBackend,
    pipes: Sequence[PipeSpec] = (),
    *,
    vectors: Sequence[VectorSpec] = (),
    world: Mapping[str, str] | None = None,
    gates: Any = None,
    max_ticks: int = 400,
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d, body="b")
            for d in descriptors
        },
        machine=machine,
        pipes=PipeTable(pipes),
        vectors=VectorTable(list(vectors)),
        world=WorldStore(),
        resources=ResourceTable(),
        gates=gates,
        journal_sink=events,
        config=KernelConfig(case="repro", max_ticks=max_ticks),
    )
    for obj, value in (world or {}).items():
        kernel.world.set(ObjectName(obj), value, at=kernel.clock)
    kernel.start()
    return kernel, events


def _of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


DOOR = PipeName("act.door")
GATE = GateSpec(
    pipe=DOOR,
    descriptor=DescriptorName("guard"),
    requests=PipeName("gate.req"),
    verdicts=PipeName("gate.verdict"),
    timeout_ticks=32,
)
DESCRIPTORS: list[dict[str, Any]] = [
    {"name": "mover", "priority": 50, "pipes": {"tools": str(DOOR)}},
    {"name": "guard", "priority": 10, "pipes": {"stdin": "gate.req", "stdout": "gate.verdict"}},
    {"name": "other", "priority": 60, "pipes": {"stdout": "o"}},
]
PIPES = [
    PipeSpec(DOOR, principal=Principal.DEVICE, world_object="door"),
    PipeSpec(PipeName("gate.req")),
    PipeSpec(PipeName("gate.verdict")),
    PipeSpec(PipeName("o")),
]


def _run(guard: list[str], ticks: int) -> tuple[Kernel, list[Event], JobState]:
    kernel, events = _build(
        DESCRIPTORS,
        CommandSeat(
            source=Tapes(
                {
                    "mover": ["write tools open;", "exit;"],
                    "guard": guard,
                    "other": ["say tick;"] * ticks + ["exit;"],
                }
            )
        ),
        PIPES,
        gates=GateTable([GATE]),
        world={"door": "closed"},
        max_ticks=ticks * 4,
    )
    mover = kernel.spawn(DescriptorName("mover"))
    kernel.spawn(DescriptorName("other"))
    for t in range(ticks):
        kernel.advance_time(t * 1_000_000)
        if not kernel.tick():
            break
    return kernel, events, mover.state


def test_a_silent_guard_is_answered_by_the_timeout_policy() -> None:
    kernel, events, state = _run(["read stdin;", "read stdin;", "exit;"], ticks=200)

    assert kernel.world.get(ObjectName("door")) == "closed"
    answers = _of(events, GateAnswered)
    assert [a.allowed for a in answers] == [False]
    assert "no verdict within 32 ticks" in answers[0].reason
    assert answers[0].clock.token_clock >= GATE.timeout_ticks, "not before the deadline"
    assert [f.fault.value for f in _of(events, FaultRaised)] == ["gate_veto"]
    assert state is JobState.FAULTED, "the default on_fault escalates"


def test_the_stalled_guard_is_cancelled_when_the_gate_times_out() -> None:
    kernel, events, _ = _run(["read stdin;", "read stdin;", "exit;"], ticks=200)

    cancelled = _of(events, JobCancelled)
    assert [c.policy for c in cancelled] == ["gate timed out"]
    guard = next(j for j in kernel.sched.jobs() if str(j.descriptor.name) == "guard")
    assert guard.state.is_terminal


def test_a_gate_that_fails_open_lets_the_write_through_at_the_deadline() -> None:
    open_gate = GateSpec(
        pipe=DOOR,
        descriptor=DescriptorName("guard"),
        requests=PipeName("gate.req"),
        verdicts=PipeName("gate.verdict"),
        timeout_ticks=32,
        on_gate_failure=ALLOW,
    )
    kernel, events = _build(
        DESCRIPTORS,
        CommandSeat(
            source=Tapes(
                {
                    "mover": ["write tools open;", "exit;"],
                    "guard": ["read stdin;", "read stdin;", "exit;"],
                    "other": ["say tick;"] * 200 + ["exit;"],
                }
            )
        ),
        PIPES,
        gates=GateTable([open_gate]),
        world={"door": "closed"},
        max_ticks=800,
    )
    mover = kernel.spawn(DescriptorName("mover"))
    kernel.spawn(DescriptorName("other"))
    for t in range(200):
        kernel.advance_time(t * 1_000_000)
        if not kernel.tick():
            break

    assert [a.allowed for a in _of(events, GateAnswered)] == [True]
    assert kernel.world.get(ObjectName("door")) == "open"
    assert mover.state is JobState.DONE


def test_a_guard_that_answers_resolves_the_write() -> None:
    kernel, events, _ = _run(["read stdin;", "write stdout veto;", "exit;"], ticks=60)
    assert [a.allowed for a in _of(events, GateAnswered)] == [False]
    assert kernel.world.get(ObjectName("door")) == "closed"
