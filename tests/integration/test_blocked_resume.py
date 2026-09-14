# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A job woken from BLOCKED is told what moved in its read set, as a preempted
job is.

The resume diff (core §6.2) is computed from a baseline. ``_preempt`` records
``suspended_at``; ``Kernel._block`` records ``blocked_at`` at every site a job
parks -- read, select, acquire, a full pipe, a gate -- and ``_resume`` diffs from
the earlier of the two. A woken job's notice opens "Waited" rather than
"Suspended", and the ``JobResumed`` event carries ``waited`` so the journal can
tell the two apart.

It used not to. Only ``_preempt`` wrote a baseline, so a job preempted while a
peer changed its read set resumed with the diff and a job blocked on a pipe
while the identical change landed was woken and dispatched with no notice at
all. Same descriptor, same read set, same write. Issue #3.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from zeos.core.events import Event, JobPreempted, JobResumed, JobWoken
from zeos.core.ids import DescriptorName, JobState, PipeName, ResourceName, ResumeKind
from zeos.core.kernel import Kernel, KernelConfig, render_resume_notice
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceSpec, ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import ObjectName, StateDelta, WorldStore

CMD = PipeName("user.cmd")
ACT = PipeName("actuators.shared")
SHARED = ObjectName("shared")
DOOR = ResourceName("door.south")

WAITER: dict[str, Any] = {
    "name": "waiter",
    "priority": 50,
    "reads": [str(SHARED)],
    "pipes": {"stdin": str(CMD)},
    "resources": [str(DOOR)],
}
#: Outranks the waiter, so spawning it while the waiter runs preempts the waiter.
HANDLER: dict[str, Any] = {
    "name": "handler",
    "priority": 5,
    "writes": [str(SHARED)],
    "pipes": {"tools": str(ACT)},
}
#: Less urgent than the waiter, so the waiter preempts it and then parks on the door.
HOLDER: dict[str, Any] = {"name": "holder", "priority": 90, "resources": [str(DOOR)]}

SCRIPTS: dict[str, list[dict[str, Any]]] = {
    "handler": [{"write": {"pipe": str(ACT), "text": "99"}}, {"exit": True}],
    "holder": [
        {"acquire": str(DOOR)},
        {"emit": "a"},
        {"emit": "b"},
        {"emit": "c"},
        {"release": str(DOOR)},
        {"exit": True},
    ],
}

#: Two ways to park on the same pipe. Both go BLOCKED -> READY through
#: ``Scheduler.wake``.
PARKED = {
    "read": [{"emit": "a"}, {"read": str(CMD)}, {"emit": "after"}, {"exit": True}],
    "select": [{"emit": "a"}, {"select": [str(CMD)]}, {"emit": "after"}, {"exit": True}],
}
ACQUIRING = [{"acquire": str(DOOR)}, {"emit": "after"}, {"release": str(DOOR)}, {"exit": True}]
RUNNING = [{"emit": "a"}, {"emit": "b"}, {"emit": "c"}, {"emit": "d"}, {"exit": True}]


def _build(waiter_script: list[dict[str, Any]]) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    world = WorldStore()
    kernel = Kernel(
        descriptors={
            DescriptorName("waiter"): Descriptor.from_frontmatter(WAITER, body="wait"),
            DescriptorName("handler"): Descriptor.from_frontmatter(HANDLER, body="set"),
            DescriptorName("holder"): Descriptor.from_frontmatter(HOLDER, body="hold"),
        },
        machine=ScriptedMachine(
            {"waiter": Script.from_spec(waiter_script)}
            | {n: Script.from_spec(s) for n, s in SCRIPTS.items()},
            block_size=8,
        ),
        pipes=PipeTable([PipeSpec(name=CMD), PipeSpec(name=ACT, world_object=str(SHARED))]),
        vectors=VectorTable(),
        world=world,
        resources=ResourceTable([ResourceSpec(name=DOOR, capacity=1, description="doorway")]),
        journal_sink=events,
        config=KernelConfig(case="blocked-resume", max_ticks=200),
    )
    world.set(SHARED, "0", at=kernel.clock)
    kernel.start()
    return kernel, events


def _of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def _dirty_resumes(events: Sequence[Event], job_id: int) -> list[JobResumed]:
    return [
        e for e in _of(events, JobResumed) if e.job == job_id and e.resume_kind is ResumeKind.DIRTY
    ]


def _run_until_blocked(kernel: Kernel, job: Any) -> None:
    for _ in range(20):
        if job.state is JobState.BLOCKED:
            return
        kernel.tick()
    raise AssertionError("the waiter never parked")


def _the_one_change(resumed: JobResumed) -> list[tuple[str, str, str]]:
    return [(str(d.obj), d.before, d.after) for d in resumed.dirty]


@pytest.mark.parametrize("parked_on", sorted(PARKED), ids=str)
def test_a_woken_job_is_told_its_read_set_moved(parked_on: str) -> None:
    kernel, events = _build(PARKED[parked_on])
    waiter = kernel.spawn(DescriptorName("waiter"))
    _run_until_blocked(kernel, waiter)

    kernel.spawn(DescriptorName("handler"))
    kernel.run_until_quiescent()
    assert kernel.world.get(SHARED) == "99", "the handler never wrote"
    assert waiter.state is JobState.BLOCKED, "the write must land while the waiter is parked"

    kernel.deliver(CMD, "go")
    kernel.run_until_quiescent()
    assert [e for e in _of(events, JobWoken) if e.job == waiter.job_id]
    assert waiter.state is JobState.DONE

    resumed = _dirty_resumes(events, waiter.job_id)
    assert resumed, "woken with a moved read set and told nothing"
    assert _the_one_change(resumed[0]) == [("shared", "0", "99")]
    assert resumed[0].waited, "it waited; it was not displaced"


def test_a_job_woken_from_acquire_is_told_its_read_set_moved() -> None:
    """Parking on a lock is parking. The holder is preempted by the waiter, which then
    blocks on the door; the handler writes while it waits; the holder's release wakes it."""
    kernel, events = _build(ACQUIRING)
    kernel.spawn(DescriptorName("holder"))
    for _ in range(20):
        if kernel.resources.get(DOOR).holders:
            break
        kernel.tick()
    assert kernel.resources.get(DOOR).holders, "the holder never took the door"

    waiter = kernel.spawn(DescriptorName("waiter"))
    _run_until_blocked(kernel, waiter)

    kernel.spawn(DescriptorName("handler"))
    kernel.run_until_quiescent()
    assert kernel.world.get(SHARED) == "99"
    assert waiter.state is JobState.DONE

    resumed = _dirty_resumes(events, waiter.job_id)
    assert resumed, "woken from acquire with a moved read set and told nothing"
    assert _the_one_change(resumed[0]) == [("shared", "0", "99")]
    assert resumed[0].waited


def test_a_preempted_job_is_told_its_read_set_moved() -> None:
    """The control: the same waiter, running instead of parked when the write lands."""
    kernel, events = _build(RUNNING)
    waiter = kernel.spawn(DescriptorName("waiter"))
    kernel.tick()
    kernel.tick()
    assert waiter.state is JobState.RUNNING

    kernel.spawn(DescriptorName("handler"))
    kernel.run_until_quiescent()
    assert kernel.world.get(SHARED) == "99"
    assert [e for e in _of(events, JobPreempted) if e.job == waiter.job_id]
    assert waiter.state is JobState.DONE

    resumed = _dirty_resumes(events, waiter.job_id)
    assert resumed
    assert _the_one_change(resumed[0]) == [("shared", "0", "99")]
    assert not resumed[0].waited, "it was displaced; it did not ask to wait"


def test_a_woken_job_is_not_told_it_was_suspended() -> None:
    """A blocked job was not displaced by anything; it asked to wait. The notice says so."""
    delta = StateDelta(obj=SHARED, before="0", after="99")
    assert render_resume_notice(1_000, [delta], waited=True).startswith("<RESUME> Waited ")
    assert render_resume_notice(1_000, [delta]).startswith("<RESUME> Suspended ")
