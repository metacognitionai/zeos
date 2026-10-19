# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A device's report of a job's own effect stays out of that job's resume diff.

A body that is told to move reports where it ended up. The report is a device write,
so without attribution the job that gave the order is woken with a RESUME notice
announcing its own move as news -- the thing ``dirty_for`` excludes a job's own writes
to prevent. Attributed, the move is silent for the job that caused it and still news
for every other job that depends on it.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from zeos.core.events import (
    Event,
    JobResumed,
    PipeBackpressure,
    PipeWritten,
    ReportAttributed,
    WorldWritten,
)
from zeos.core.ids import DescriptorName, JobId, ObjectName, PipeName, Principal, ResumeKind
from zeos.core.kernel import Kernel, KernelConfig, KernelError
from zeos.core.pipes import PipeFull, PipeSpec, PipeTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

REPORT = PipeName("report.position")
PIPES = [
    PipeSpec(REPORT, principal=Principal.DEVICE, world_object="robot.position"),
    PipeSpec(PipeName("done.mover"), principal=Principal.DEVICE, device=True),
    PipeSpec(PipeName("done.watcher"), principal=Principal.DEVICE, device=True),
]


def job(name: str) -> Descriptor:
    return Descriptor.from_frontmatter(
        {
            "name": name,
            "priority": 50,
            "reads": ["robot.position"],
            "pipes": {"stdin": f"done.{name}"},
        }
    )


def build() -> tuple[Kernel, list[Event], JobId, JobId]:
    """Two jobs that depend on where the robot is, both asleep on their own pipe."""
    events: list[Event] = []
    wait = [{"read": "stdin"}, {"exit": True}]
    kernel = Kernel(
        descriptors={DescriptorName(n): job(n) for n in ("mover", "watcher")},
        machine=ScriptedMachine(
            {n: Script.from_spec(wait) for n in ("mover", "watcher")}, block_size=8
        ),
        pipes=PipeTable(PIPES),
        vectors=VectorTable(),
        world=WorldStore(),
        journal_sink=events,
        config=KernelConfig(case="attribution"),
    )
    kernel.start()
    kernel.world.set(ObjectName("robot.position"), "dock", at=kernel.clock)
    mover = kernel.spawn(DescriptorName("mover"))
    watcher = kernel.spawn(DescriptorName("watcher"))
    kernel.run_until_quiescent()
    return kernel, events, mover.job_id, watcher.job_id


def wake_both(kernel: Kernel) -> None:
    kernel.deliver(PipeName("done.mover"), "done")
    kernel.deliver(PipeName("done.watcher"), "done")
    kernel.run_until_quiescent()


def resumed(events: Sequence[Event], job_id: JobId) -> JobResumed:
    return next(e for e in events if isinstance(e, JobResumed) and e.job == job_id)


def test_the_job_that_caused_a_report_is_not_told_about_it() -> None:
    kernel, events, mover, watcher = build()
    kernel.deliver(REPORT, "kitchen", on_behalf_of=mover)
    wake_both(kernel)

    assert resumed(events, mover).resume_kind is ResumeKind.CLEAN
    news = resumed(events, watcher)
    assert news.resume_kind is ResumeKind.DIRTY
    assert [(d.obj, d.before, d.after) for d in news.dirty] == [
        (ObjectName("robot.position"), "dock", "kitchen")
    ]


def test_an_unattributed_report_is_news_to_everyone() -> None:
    kernel, events, mover, watcher = build()
    kernel.deliver(REPORT, "kitchen")
    wake_both(kernel)

    assert resumed(events, mover).resume_kind is ResumeKind.DIRTY
    assert resumed(events, watcher).resume_kind is ResumeKind.DIRTY


def test_the_journal_says_who_wrote_and_who_caused_it() -> None:
    kernel, events, mover, _ = build()
    kernel.deliver(REPORT, "kitchen", on_behalf_of=mover)

    attributed = next(e for e in events if isinstance(e, ReportAttributed))
    written = next(e for e in events if isinstance(e, PipeWritten) and e.pipe == REPORT)
    world = next(e for e in events if isinstance(e, WorldWritten))
    assert (attributed.pipe, attributed.job) == (REPORT, mover)
    assert written.job is None, "the device wrote the pipe"
    assert world.job == mover, "the job caused the change"


def test_a_report_that_does_not_land_is_not_attributed() -> None:
    kernel, events, mover, _ = build()
    capacity = kernel.pipes.get(REPORT).spec.capacity_tokens
    before = len(events)
    with pytest.raises(PipeFull):
        kernel.deliver(REPORT, " ".join(["far"] * (capacity + 1)), on_behalf_of=mover)

    refused = events[before:]
    assert [type(e) for e in refused] == [PipeBackpressure]
    assert kernel.world.get(ObjectName("robot.position")) == "dock"


def test_attribution_needs_an_actuator() -> None:
    kernel, _, mover, _ = build()
    with pytest.raises(KernelError, match="only on an actuator"):
        kernel.deliver(PipeName("done.mover"), "done", on_behalf_of=mover)


def test_attribution_needs_a_job_the_kernel_knows() -> None:
    kernel, _, _, _ = build()
    with pytest.raises(KernelError, match="unknown job"):
        kernel.deliver(REPORT, "kitchen", on_behalf_of=JobId(99))
