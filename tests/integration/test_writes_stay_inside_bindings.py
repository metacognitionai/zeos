# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A job that declares no capabilities may write only the pipes it binds.

ZEOS-MP §5.2: "The kernel checks every write." Without ``capabilities:`` the bindings
under ``pipes:`` are the grant: a write to any other name is a capability fault on the
writing job, and no pipe springs into existence from the name. ``writes:`` stays a
declaration for resume diffs. With capabilities declared, the held pipes are the grant,
as before.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import Event, FaultRaised, PipeWritten, WorldWritten
from zeos.core.ids import DescriptorName, ObjectName, PipeName, Principal
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


PEER_ACTUATOR = PipeName("count.progress_b")


def _run(commands: list[str], **frontmatter: Any) -> tuple[Kernel, list[Event]]:
    kernel, events = _build(
        [
            {"name": "a", "priority": 50, "pipes": {"stdout": "a.out"}, "writes": ["count.a"]}
            | frontmatter
        ],
        CommandSeat(source=Tapes({"a": commands})),
        [
            PipeSpec(PipeName("a.out")),
            PipeSpec(PEER_ACTUATOR, principal=Principal.DEVICE, world_object="count.b"),
        ],
        world={"count.b": "0"},
    )
    kernel.spawn(DescriptorName("a"))
    kernel.run_until_quiescent()
    return kernel, events


def test_a_job_cannot_write_a_pipe_it_did_not_bind() -> None:
    kernel, events = _run(["write count.progress_b 999;", "exit;"])

    assert kernel.world.get(ObjectName("count.b")) == "0"
    assert not [w for w in _of(events, WorldWritten) if w.obj == "count.b"]
    assert [f.fault.value for f in _of(events, FaultRaised)] == ["capability_fault"]


def test_a_job_writes_the_pipes_it_binds_without_capabilities() -> None:
    _, events = _run(["write stdout hello;", "exit;"])
    assert [str(w.pipe) for w in _of(events, PipeWritten)] == ["a.out"]
    assert not _of(events, FaultRaised)


def test_a_stray_name_creates_no_pipe() -> None:
    kernel, events = _run(["write nowhere hello;", "exit;"])

    assert not kernel.pipes.has(PipeName("nowhere"))
    assert not [w for w in _of(events, PipeWritten) if str(w.pipe) == "nowhere"]
    assert [f.fault.value for f in _of(events, FaultRaised)] == ["capability_fault"]


def test_with_capabilities_declared_the_same_write_faults() -> None:
    """The control: once a descriptor holds any capability, an unheld pipe is a fault."""
    kernel, events = _run(
        ["write count.progress_b 999;", "exit;"], capabilities=[{"pipe": "a.out"}]
    )
    assert kernel.world.get(ObjectName("count.b")) == "0"
    assert [f.fault.value for f in _of(events, FaultRaised)] == ["capability_fault"]
