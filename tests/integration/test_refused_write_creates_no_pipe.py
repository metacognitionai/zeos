# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A refused write leaves the pipe table as it found it, capabilities or not.

#47 keeps a job without capabilities inside its bindings before the pipe table is
touched; a job with capabilities is refused by the capability check, and the pipe is
fetched or created only once the write is allowed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import (
    Event,
    FaultRaised,
)
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    ObjectName,
    PipeName,
)
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
        journal_sink=events,
        config=KernelConfig(case="repro", max_ticks=max_ticks),
    )
    for obj, value in (world or {}).items():
        kernel.world.set(ObjectName(obj), value, at=kernel.clock)
    kernel.start()
    return kernel, events


def _of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


OUT = PipeName("held")
STRAY = PipeName("elsewhere")


def _run() -> tuple[Kernel, list[Event]]:
    kernel, events = _build(
        [
            {
                "name": "w",
                "priority": 50,
                "pipes": {"stdout": str(OUT)},
                "capabilities": [{"pipe": str(OUT), "min_integrity": 3}],
            }
        ],
        CommandSeat(source=Tapes({"w": [f"write {STRAY} hi;", "exit;"]})),
        [PipeSpec(OUT)],
    )
    kernel.spawn(DescriptorName("w"))
    kernel.run_until_quiescent()
    return kernel, events


def test_the_unheld_write_is_refused() -> None:
    _, events = _run()
    assert [(e.fault, e.pipe) for e in _of(events, FaultRaised)] == [(FaultKind.CAPABILITY, STRAY)]


def test_the_refused_name_creates_no_pipe() -> None:
    kernel, _ = _run()
    assert not kernel.pipes.has(STRAY)
