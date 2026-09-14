# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A finished job leaves the scheduler's live set, keeps its record, and gives its
context back.

Every per-tick scan walks the live jobs only, so a stream of interrupts costs the
kernel the same per tick after the thousandth handler as after the first. The record
stays in the table for lookups by id. The driver reaps a terminal job's context as soon
as it is terminal, unless told to keep contexts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.events import Event, JobSpawned
from zeos.core.ids import (
    DescriptorName,
    JobState,
    ObjectName,
    PipeName,
    Principal,
    Priority,
    VectorName,
    VectorPolicy,
)
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


KEYS = PipeName("keys")
VECTOR = VectorSpec(
    name=VectorName("v"),
    source=KEYS,
    handler=DescriptorName("h"),
    priority=Priority(5),
    policy=VectorPolicy.QUEUE,
)


def _storm(n: int) -> tuple[Kernel, list[Event]]:
    kernel, events = _build(
        [
            {"name": "g", "priority": 50},
            {"name": "h", "priority": 5, "pipes": {"stdin": str(KEYS)}},
        ],
        ScriptedMachine(
            {
                "g": Script.from_spec([{"emit": "w"}] * 3 + [{"exit": True}]),
                "h": Script.from_spec([{"emit": "h"}, {"exit": True}]),
            },
            block_size=8,
        ),
        [PipeSpec(KEYS, device=True, principal=Principal.USER)],
        vectors=[VECTOR],
        max_ticks=100_000,
    )
    kernel.spawn(DescriptorName("g"))
    for _ in range(n):
        kernel.deliver(KEYS, "x")
    return kernel, events


def test_finished_handlers_leave_the_live_set_and_keep_their_record() -> None:
    kernel, _ = _storm(200)
    kernel.run_until_quiescent()

    assert not kernel.sched.live(), "nothing left that can run"
    done = [j for j in kernel.sched.jobs() if j.state is JobState.DONE]
    assert len(done) == 201, "every record is still there to look up"
    assert kernel.sched.get(done[-1].job_id).state is JobState.DONE


def test_the_driver_reaps_a_finished_job_by_default() -> None:
    from zeos.driver import Driver

    kernel, _ = _storm(3)
    driver = Driver(kernel)
    driver.run()

    for job in kernel.sched.jobs():
        assert job.state is JobState.DONE
        with pytest.raises(KeyError):
            kernel.machine.stats(job.job_id)
    assert not kernel.unreaped()


def test_the_driver_can_be_told_to_keep_contexts() -> None:
    from zeos.driver import Driver

    kernel, _ = _storm(3)
    driver = Driver(kernel, reap=False)
    driver.run()

    assert len(kernel.unreaped()) == 4
    for job in kernel.sched.jobs():
        assert kernel.machine.stats(job.job_id).resident_tokens > 0


def test_every_firing_is_still_handled_in_order() -> None:
    kernel, events = _storm(200)
    kernel.run_until_quiescent()
    assert len([e for e in _of(events, JobSpawned) if str(e.descriptor) == "h"]) == 200
