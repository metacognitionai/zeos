# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``job.state`` is the canonical transition record, so a job's run of them is unbroken.

The scheduler's ``block``, ``wake`` and ``preempt`` set a job's new state themselves,
and the kernel's ``_transition`` compared the job's state with the target to decide
whether anything had moved. After those calls it never had, so blocking on a resource or
a body, being woken from one, and losing a body mid-run all went unrecorded: a job's
record read ``ready -> running`` twice in a row, and anything folding the journal showed
it running while it waited.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from zeos.core.embodiment import PlatformProfile
from zeos.core.events import Event, JobStateChanged
from zeos.core.ids import DescriptorName, JobId, JobState, ResourceName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import ResourceSpec, ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

DOOR = ResourceSpec(name=ResourceName("door"), capacity=1)
BODY = PlatformProfile(name="stretch", locomotion="wheeled", tooling=frozenset({"gripper"}))


def build(
    descriptors: Sequence[dict[str, Any]], scripts: dict[str, list[dict[str, Any]]]
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(d["name"]): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in scripts.items()}, block_size=8),
        pipes=PipeTable(),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable([DOOR]),
        platforms=(BODY,),
        journal_sink=events,
        config=KernelConfig(case="states"),
    )
    kernel.start()
    return kernel, events


def unbroken(events: Sequence[Event]) -> dict[JobId, list[tuple[JobState, JobState]]]:
    """Every job's transitions, asserting each starts where the last one ended."""
    runs: dict[JobId, list[tuple[JobState, JobState]]] = {}
    for event in events:
        if isinstance(event, JobStateChanged):
            run = runs.setdefault(event.job, [])
            if run:
                assert event.from_state is run[-1][1], (
                    f"job {event.job}: {run[-1][1]} was followed by a move from {event.from_state}"
                )
            run.append((event.from_state, event.to_state))
    return runs


def test_waiting_for_a_lock_is_recorded() -> None:
    kernel, events = build(
        [
            {"name": "sweeper", "priority": 90, "resources": ["door"]},
            {"name": "carrier", "priority": 10, "resources": ["door"]},
        ],
        {
            "sweeper": [
                {"acquire": "door"},
                {"emit": "a"},
                {"emit": "b"},
                {"release": "door"},
                {"exit": True},
            ],
            "carrier": [
                {"acquire": "door"},
                {"emit": "through"},
                {"release": "door"},
                {"exit": True},
            ],
        },
    )
    kernel.spawn(DescriptorName("sweeper"))
    kernel.tick()
    kernel.tick()
    carrier = kernel.spawn(DescriptorName("carrier"))
    kernel.run_until_quiescent()

    moves = unbroken(events)[carrier.job_id]
    assert (JobState.RUNNING, JobState.BLOCKED) in moves
    assert (JobState.BLOCKED, JobState.READY) in moves


def test_losing_a_body_mid_run_is_recorded() -> None:
    wearer = {"name": "tidy", "priority": 80, "requires": {"tooling": ["gripper"]}}
    kernel, events = build(
        [wearer, {"name": "coffee", "priority": 40, "requires": {"tooling": ["gripper"]}}],
        {
            "tidy": [{"emit": w} for w in "one two three four five six".split()] + [{"exit": True}],
            "coffee": [{"emit": "done"}, {"exit": True}],
        },
    )
    tidy = kernel.spawn(DescriptorName("tidy"))
    kernel.tick()
    kernel.tick()
    coffee = kernel.spawn(DescriptorName("coffee"))
    kernel.run_until_quiescent()

    runs = unbroken(events)
    assert (JobState.RUNNING, JobState.BLOCKED) in runs[coffee.job_id], "waiting for the body"
    assert runs[tidy.job_id][-1] == (JobState.RUNNING, JobState.DONE)
