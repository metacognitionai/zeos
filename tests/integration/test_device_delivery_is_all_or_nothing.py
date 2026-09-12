# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A device delivery is all or nothing: one that does not fit lands no token, is
journalled as backpressure, and raises PipeFull for the driver to decide.

A job in the same position is parked on write-full and retried by the kernel; a device
cannot be parked, and the kernel holds nothing on its behalf, so the refusal goes to the
driver. Nothing is ever truncated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.events import Event, PipeBackpressure, PipeWritten
from zeos.core.ids import DescriptorName, ObjectName, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeFull, PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.base import MachineBackend
from zeos.machine.scripted import Script, ScriptedMachine
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


IN = PipeName("sensor.in")


def _listener() -> tuple[Kernel, list[Event]]:
    kernel, events = _build(
        [{"name": "r", "priority": 50, "pipes": {"stdin": str(IN)}}],
        ScriptedMachine({"r": Script.from_spec([{"read": str(IN)}, {"exit": True}])}, block_size=8),
        [PipeSpec(IN, device=True, capacity_tokens=16)],
    )
    kernel.spawn(DescriptorName("r"))
    kernel.run_until_quiescent()
    return kernel, events


def test_a_delivery_that_does_not_fit_is_refused_whole() -> None:
    kernel, events = _listener()
    with pytest.raises(PipeFull, match="room for 16 tokens; a delivery of 40 does not fit"):
        kernel.deliver(IN, " ".join(f"w{i}" for i in range(40)))

    assert kernel.pipes.get(IN).available == 0, "not one token landed"
    assert not [w for w in _of(events, PipeWritten) if w.pipe == IN]
    back = _of(events, PipeBackpressure)
    assert [(b.pipe, b.job, b.capacity_tokens) for b in back] == [(IN, None, 16)]


def test_a_delivery_that_fits_lands_as_before() -> None:
    kernel, events = _listener()
    kernel.deliver(IN, " ".join(f"w{i}" for i in range(16)))
    kernel.run_until_quiescent()
    assert [w.tokens for w in _of(events, PipeWritten) if w.pipe == IN] == [16]
    assert not _of(events, PipeBackpressure)


def test_an_actuator_delivery_larger_than_its_capacity_is_refused_too() -> None:
    kernel, events = _build(
        [{"name": "r", "priority": 50}],
        ScriptedMachine({"r": Script.from_spec([{"exit": True}])}, block_size=8),
        [PipeSpec(PipeName("act"), device=True, capacity_tokens=4, world_object="thing")],
        world={"thing": "old"},
    )
    with pytest.raises(PipeFull):
        kernel.deliver(PipeName("act"), "one two three four five")
    assert kernel.world.get(ObjectName("thing")) == "old"
    assert _of(events, PipeBackpressure)


def test_the_driver_drops_a_refused_delivery_and_says_so() -> None:
    from zeos.driver import Driver, ScheduledEvent

    kernel, events = _build(
        [{"name": "r", "priority": 50, "pipes": {"stdin": str(IN)}}],
        ScriptedMachine({"r": Script.from_spec([{"read": str(IN)}, {"exit": True}])}, block_size=8),
        [PipeSpec(IN, device=True, capacity_tokens=4)],
    )
    driver = Driver(kernel)
    kernel.spawn(DescriptorName("r"))
    driver.run(
        [
            ScheduledEvent(at_ns=1_000_000, pipe=IN, text="a b c d e f g h"),
            ScheduledEvent(at_ns=2_000_000, pipe=IN, text="ok"),
        ]
    )

    assert driver.refused == [(IN, "a b c d e f g h")]
    assert [" ".join(w.text) for w in _of(events, PipeWritten) if w.pipe == IN] == ["ok"]


def test_a_gated_delivery_answered_late_is_dropped_loudly_not_raised() -> None:
    """Answered inside a tick, where no driver can be told, the kernel journals and drops."""
    from zeos.core.events import GateAnswered
    from zeos.core.gates import GateSpec, GateTable

    door = PipeName("act.door")
    gate = GateSpec(
        pipe=door,
        descriptor=DescriptorName("guard"),
        requests=PipeName("gate.req"),
        verdicts=PipeName("gate.verdict"),
    )
    kernel, events = _build(
        [
            {
                "name": "guard",
                "priority": 10,
                "pipes": {"stdin": "gate.req", "stdout": "gate.verdict"},
            }
        ],
        CommandSeat(source=Tapes({"guard": ["read stdin;", "write stdout allow;", "exit;"]})),
        [
            PipeSpec(door, device=True, capacity_tokens=2, world_object="door"),
            PipeSpec(PipeName("gate.req")),
            PipeSpec(PipeName("gate.verdict")),
        ],
        gates=GateTable([gate]),
        world={"door": "closed"},
    )
    kernel.deliver(door, "open wide now please")
    kernel.run_until_quiescent()

    assert [a.allowed for a in _of(events, GateAnswered)] == [True]
    assert kernel.world.get(ObjectName("door")) == "closed", "allowed, but it no longer fit"
    assert _of(events, PipeBackpressure) and not [
        w for w in _of(events, PipeWritten) if w.pipe == door
    ]


def test_a_jobs_write_that_does_not_fit_blocks_instead() -> None:
    """The control: the same overflow from a job parks the writer, losing nothing."""
    from zeos.core.events import JobBlocked

    kernel, events = _build(
        [{"name": "w", "priority": 50, "pipes": {"stdout": str(IN)}}],
        CommandSeat(
            source=Tapes(
                {"w": ["write stdout " + " ".join(f"w{i}" for i in range(40)) + ";", "exit;"]}
            )
        ),
        [PipeSpec(IN, capacity_tokens=40)],
    )
    kernel.deliver(IN, "busy")
    kernel.spawn(DescriptorName("w"))
    kernel.run_until_quiescent()
    assert [b.reason for b in _of(events, JobBlocked)] == ["write-full"]
