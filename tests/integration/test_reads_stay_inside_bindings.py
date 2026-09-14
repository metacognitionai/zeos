# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A read stays inside the bindings, as a write does.

A read or a select that names a pipe outside the descriptor's ``pipes:`` is a capability
fault on the job, creates no pipe and parks nothing, whether or not the job holds
capabilities. A read of a bound pipe blocks and wakes as before.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.events import (
    Event,
    FaultRaised,
    JobBlocked,
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


IN = PipeName("inbox")
STRAY = PipeName("nowhere")


def _run(capabilities: bool) -> tuple[Kernel, list[Event]]:
    descriptor: dict[str, Any] = {"name": "r", "priority": 50, "pipes": {"stdin": str(IN)}}
    if capabilities:
        descriptor["capabilities"] = [{"pipe": str(IN), "min_integrity": 3}]
    kernel, events = _build(
        [descriptor],
        CommandSeat(source=Tapes({"r": [f"read {STRAY};", "say awake;", "exit;"]})),
        [PipeSpec(IN, device=True)],
    )
    kernel.spawn(DescriptorName("r"))
    kernel.run_until_quiescent()
    return kernel, events


@pytest.mark.parametrize(
    "capabilities", [False, True], ids=["no-capabilities", "with-capabilities"]
)
def test_a_read_of_an_unbound_name_faults_and_creates_no_pipe(capabilities: bool) -> None:
    kernel, events = _run(capabilities)
    job = kernel.sched.jobs()[0]
    assert not kernel.pipes.has(STRAY), "a model's word became a pipe"
    assert job.state is not JobState.BLOCKED, "parked on a pipe nobody will ever write"
    assert [e.fault for e in _of(events, FaultRaised)] == [FaultKind.CAPABILITY]


def test_a_read_of_the_bound_stdin_blocks_as_it_should() -> None:
    kernel, events = _build(
        [{"name": "r", "priority": 50, "pipes": {"stdin": str(IN)}}],
        CommandSeat(source=Tapes({"r": ["read stdin;", "say awake;", "exit;"]})),
        [PipeSpec(IN, device=True)],
    )
    kernel.spawn(DescriptorName("r"))
    kernel.run_until_quiescent()
    assert [(e.pipe, e.reason) for e in _of(events, JobBlocked)] == [(IN, "read-empty")]
    kernel.deliver(IN, "go")
    kernel.run_until_quiescent()
    assert kernel.sched.jobs()[0].state is JobState.DONE


def test_a_select_naming_an_unbound_pipe_faults_and_creates_no_pipe() -> None:
    kernel, events = _build(
        [{"name": "r", "priority": 50, "pipes": {"stdin": str(IN)}}],
        ScriptedMachine(
            {
                "r": Script.from_spec(
                    [{"select": [str(IN), str(STRAY)]}, {"emit": "awake"}, {"exit": True}]
                )
            },
            block_size=8,
        ),
        [PipeSpec(IN, device=True)],
    )
    kernel.spawn(DescriptorName("r"))
    kernel.run_until_quiescent()
    assert not kernel.pipes.has(STRAY)
    assert [e.fault for e in _of(events, FaultRaised)] == [FaultKind.CAPABILITY]
    assert not _of(events, JobBlocked)
