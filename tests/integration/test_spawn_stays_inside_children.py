# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A job may spawn only the descriptors its ``children:`` lists.

Transformer-OS §3: "children: sub-jobs this job may spawn (the hierarchy)". A spawn of
any other name is a capability fault on the asking job, as a write to an unbound pipe
is, and starts nothing. A descriptor with no ``children:`` may spawn nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import Event, FaultRaised, JobSpawned
from zeos.core.faults import FaultKind
from zeos.core.ids import DescriptorName, ObjectName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.base import MachineBackend
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.machine.seat import Turn
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


TREE: list[dict[str, Any]] = [
    {"name": "p", "priority": 50, "children": ["ok"]},
    {"name": "ok", "priority": 50},
    {"name": "secret", "priority": 1},
]


def _run(parent_steps: list[dict[str, Any]]) -> list[Event]:
    kernel, events = _build(
        TREE,
        ScriptedMachine(
            {
                "p": Script.from_spec(parent_steps),
                "ok": Script.from_spec([{"exit": True}]),
                "secret": Script.from_spec([{"emit": "pwned"}, {"exit": True}]),
            },
            block_size=8,
        ),
    )
    kernel.spawn(DescriptorName("p"))
    kernel.run_until_quiescent()
    return events


def test_a_spawn_outside_children_is_a_capability_fault_and_starts_nothing() -> None:
    events = _run([{"spawn": "secret"}, {"exit": True}])

    spawned = _of(events, JobSpawned)
    assert [str(e.descriptor) for e in spawned] == ["p"]
    faults = _of(events, FaultRaised)
    assert [(f.fault, f.job) for f in faults] == [(FaultKind.CAPABILITY, spawned[0].job)]
    assert "'secret'" in faults[0].detail, "the journal names what was asked for"


def test_a_descriptor_without_children_may_spawn_nothing() -> None:
    kernel, events = _build(
        [{"name": "lone", "priority": 50}, {"name": "ok", "priority": 50}],
        ScriptedMachine(
            {
                "lone": Script.from_spec([{"spawn": "ok"}, {"exit": True}]),
                "ok": Script.from_spec([{"exit": True}]),
            },
            block_size=8,
        ),
    )
    kernel.spawn(DescriptorName("lone"))
    kernel.run_until_quiescent()

    assert [str(e.descriptor) for e in _of(events, JobSpawned)] == ["lone"]
    assert [f.fault for f in _of(events, FaultRaised)] == [FaultKind.CAPABILITY]


def test_a_declared_child_spawns() -> None:
    events = _run([{"spawn": "ok"}, {"exit": True}])
    assert [str(e.descriptor) for e in _of(events, JobSpawned)] == ["p", "ok"]
