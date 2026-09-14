# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A write larger than its pipe's whole capacity is refused, not parked.

Backpressure is a wait for room a reader can make. A payload larger than the pipe itself
has no such room to wait for, so it is a capability fault on the job, lands nothing and
parks nothing. A write that fits an empty pipe but not the current one is still parked
and lands whole when the reader frees room.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import (
    Event,
    FaultRaised,
    JobBlocked,
    PipeWritten,
)
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    JobState,
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


OUT = PipeName("narrow")


def _run(text: str) -> tuple[Kernel, list[Event]]:
    kernel, events = _build(
        [{"name": "w", "priority": 50, "pipes": {"stdout": str(OUT)}}],
        CommandSeat(source=Tapes({"w": [f"write stdout {text};", "exit;"]})),
        [PipeSpec(OUT, capacity_tokens=4)],
    )
    kernel.spawn(DescriptorName("w"))
    kernel.run_until_quiescent()
    return kernel, events


def test_a_write_that_can_never_fit_faults_instead_of_waiting() -> None:
    kernel, events = _run("a b c d e")
    job = kernel.sched.jobs()[0]
    assert [(e.fault, e.pipe) for e in _of(events, FaultRaised)] == [(FaultKind.CAPABILITY, OUT)]
    assert not [e for e in _of(events, JobBlocked) if e.reason == "write-full"]
    assert job.state is not JobState.BLOCKED
    assert not [e for e in _of(events, PipeWritten) if e.pipe == OUT], "nothing landed"


def test_a_write_that_fits_an_empty_pipe_but_not_a_busy_one_is_parked_and_then_lands() -> None:
    kernel, events = _build(
        [
            {"name": "w", "priority": 50, "pipes": {"stdout": str(OUT)}},
            {"name": "r", "priority": 60, "pipes": {"stdin": str(OUT)}},
        ],
        CommandSeat(
            source=Tapes(
                {
                    "w": ["write stdout a b c;", "write stdout d e f;", "exit;"],
                    "r": ["read stdin;", "read stdin;", "exit;"],
                }
            )
        ),
        [PipeSpec(OUT, capacity_tokens=4)],
    )
    kernel.spawn(DescriptorName("w"))
    kernel.spawn(DescriptorName("r"))
    kernel.run_until_quiescent()

    parked = [e for e in _of(events, JobBlocked) if e.pipe == OUT and e.reason == "write-full"]
    assert len(parked) == 1
    assert [e.tokens for e in _of(events, PipeWritten) if e.pipe == OUT] == [3, 3], (
        "both landed whole"
    )
    assert not _of(events, FaultRaised)


def test_a_write_that_exactly_fills_the_pipe_lands_whole() -> None:
    _, events = _run("a b c d")
    assert [e.tokens for e in _of(events, PipeWritten) if e.pipe == OUT] == [4]
    assert not [e for e in _of(events, JobBlocked) if e.reason == "write-full"]
