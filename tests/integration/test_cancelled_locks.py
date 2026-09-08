# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A job cancelled by a completion policy must give its locks back.

``on_complete: cancel-below:N`` and ``on_complete: replace-with:<descriptor>`` end
the jobs beneath the finishing one. Ending a job means giving back what it held:
``_cancel`` and ``_complete`` both call ``_release_held_resources``, and so does
``Kernel._apply_completion_policy`` for each job it unwinds.

It used not to. A job cancelled by either policy was marked DONE and journalled
as ``JobCancelled``, and stayed the holder of every resource it had acquired. No
``ResourceReleased`` was journalled, and the next job to ask for the resource
blocked on a holder that could never release it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.events import Event, JobCancelled, ResourceBlocked, ResourceReleased
from zeos.core.ids import DescriptorName, JobState, ResourceName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import ResourceSpec, ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

DOOR = ResourceName("door.south")

#: The holder is deliberately unhurried (priority 90) so the emergency preempts it
#: while it is standing in the doorway.
SWEEPER: dict[str, Any] = {"name": "sweeper", "priority": 90, "resources": [str(DOOR)]}
FOLLOW_UP: dict[str, Any] = {"name": "follow-up", "priority": 60}
CARRIER: dict[str, Any] = {"name": "carrier-lead", "priority": 50, "resources": [str(DOOR)]}

SCRIPTS = {
    "sweeper": [
        {"acquire": str(DOOR)},
        {"emit": "sweeping"},
        {"emit": "still sweeping"},
        {"release": str(DOOR)},
        {"exit": True},
    ],
    "alarm": [{"emit": "handled"}, {"exit": True}],
    "follow-up": [{"emit": "following up"}, {"exit": True}],
    "carrier-lead": [
        {"acquire": str(DOOR)},
        {"emit": "through"},
        {"release": str(DOOR)},
        {"exit": True},
    ],
}

UNWINDING = ["cancel-below:1", "replace-with:follow-up"]


def _build(on_complete: str | None) -> tuple[Kernel, list[Event]]:
    alarm: dict[str, Any] = {"name": "alarm", "priority": 10}
    if on_complete is not None:
        alarm["on_complete"] = on_complete
    descriptors: list[Mapping[str, Any]] = [SWEEPER, alarm, FOLLOW_UP, CARRIER]
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in SCRIPTS.items()}, block_size=8),
        pipes=PipeTable(),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable([ResourceSpec(name=DOOR, capacity=1, description="doorway")]),
        journal_sink=events,
        config=KernelConfig(case="cancelled-locks", max_ticks=200),
    )
    kernel.start()
    return kernel, events


def _of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def _stage_holder_then_emergency(kernel: Kernel) -> None:
    """The sweeper takes the door, then the alarm preempts it and runs to completion."""
    kernel.spawn(DescriptorName("sweeper"))
    for _ in range(20):
        if kernel.resources.get(DOOR).holders:
            break
        kernel.tick()
    assert kernel.resources.get(DOOR).holders, "the sweeper never took the door"
    kernel.spawn(DescriptorName("alarm"))
    kernel.run_until_quiescent()


@pytest.mark.parametrize("on_complete", UNWINDING)
def test_a_job_cancelled_by_a_completion_policy_releases_its_locks(on_complete: str) -> None:
    kernel, events = _build(on_complete)
    _stage_holder_then_emergency(kernel)

    sweeper = next(j for j in kernel.sched.jobs() if j.name == DescriptorName("sweeper"))
    assert sweeper.state is JobState.DONE, "the sweeper was meant to be cancelled"
    assert [c for c in _of(events, JobCancelled) if c.job == sweeper.job_id]

    assert [e for e in _of(events, ResourceReleased) if e.job == sweeper.job_id], (
        "a job that will never run again is still the holder of the door"
    )
    assert not kernel.resources.get(DOOR).holders

    carrier = kernel.spawn(DescriptorName("carrier-lead"))
    kernel.run_until_quiescent()
    assert carrier.state is JobState.DONE, "the next job through the door blocked on a dead holder"


def test_a_job_that_resumes_after_the_emergency_releases_its_own_locks() -> None:
    """The control: the alarm returns instead of unwinding, so the sweeper resumes and
    releases the door itself, and the carrier gets through."""
    kernel, events = _build(None)
    _stage_holder_then_emergency(kernel)

    sweeper = next(j for j in kernel.sched.jobs() if j.name == DescriptorName("sweeper"))
    assert sweeper.state is JobState.DONE
    assert not _of(events, JobCancelled)
    assert [e for e in _of(events, ResourceReleased) if e.job == sweeper.job_id]
    assert not kernel.resources.get(DOOR).holders

    carrier = kernel.spawn(DescriptorName("carrier-lead"))
    kernel.run_until_quiescent()
    assert carrier.state is JobState.DONE
    assert not _of(events, ResourceBlocked), "nobody should have had to wait"
