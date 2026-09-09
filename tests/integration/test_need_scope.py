# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A NEED, and an explicit fault, are answered from the asking job's own archive.

Each eviction records the job whose context it came from, and ``resolve_need``
searches only that job's spans: a job asking for content about X is answered from
what was evicted from its own context, never from another job's archive that happens
to match the words. An explicit ``FAULT`` on a segment owned by another job answers
as if the segment did not exist.

It used not to. The pager is one per kernel; ``resolve_need`` ranked every archived
span by word overlap and was not told who was asking, so a job asking about payroll
was handed a different job's evicted context, injected with that job's provenance so
it read as the asker's own trusted output. Issue #19.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import Event, PagedIn, PageFaultRaised, SegmentEvicted
from zeos.core.ids import DescriptorName, JobState, PipeName, Principal, Ring
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

#: A window far smaller than the transcript, so the clerk's output is archived.
CLERK: dict[str, Any] = {
    "name": "payroll-clerk",
    "priority": 80,
    "context": {
        "window": 300,
        "eviction": "attention-clock",
        "high_watermark": 0.7,
        "low_watermark": 0.4,
    },
}
VISITOR: dict[str, Any] = {"name": "visitor", "priority": 70}

PRIVATE = " ".join(f"secret payroll salary figure{i} confidential" for i in range(8))
CLERK_WORK = [{"emit": PRIVATE} for _ in range(12)]


def _build(
    descriptors: Sequence[Mapping[str, Any]], scripts: Mapping[str, list[dict[str, Any]]]
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in scripts.items()}, block_size=4),
        pipes=PipeTable(
            [PipeSpec(PipeName("ops.report"), ring=Ring.TRUSTED, principal=Principal.USER)]
        ),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="need-scope"),
    )
    kernel.start()
    return kernel, events


def _of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def _clerk_evicted_segment() -> int:
    """A probe run of the clerk alone, returning a segment id it evicted.

    A script cannot name a runtime segment id, and the run is deterministic, so the
    id the clerk stubs here is the same one it stubs when the visitor runs after it.
    The visitor faulting this id is a job reaching for another job's handle.
    """
    kernel, events = _build([CLERK], {"payroll-clerk": [*CLERK_WORK, {"exit": True}]})
    kernel.spawn(DescriptorName("payroll-clerk"))
    kernel.run_until_quiescent()
    evicted = _of(events, SegmentEvicted)
    assert evicted, "probe run produced no eviction; the fixture is mis-proportioned"
    return int(evicted[0].segment)


def test_a_need_is_not_answered_from_another_job_s_archive() -> None:
    kernel, events = _build(
        [CLERK, VISITOR],
        {
            "payroll-clerk": [*CLERK_WORK, {"exit": True}],
            "visitor": [{"need": "payroll salary figures"}, {"emit": "thanks"}, {"exit": True}],
        },
    )
    clerk = kernel.spawn(DescriptorName("payroll-clerk"))
    kernel.run_until_quiescent()
    archived = {e.store for e in _of(events, SegmentEvicted) if e.job == clerk.job_id}
    assert clerk.state is JobState.DONE and archived, "the clerk's output was meant to be archived"

    visitor = kernel.spawn(DescriptorName("visitor"))
    kernel.run_until_quiescent()

    assert [e for e in _of(events, PageFaultRaised) if e.job == visitor.job_id]
    handed = [e for e in _of(events, PagedIn) if e.job == visitor.job_id]
    assert not [e for e in handed if e.store in archived], (
        "the visitor asked about payroll and was handed the clerk's archived context"
    )
    transcript = " ".join(t.text for t in kernel.machine.transcript(visitor.job_id))
    assert "confidential" not in transcript


def test_an_explicit_fault_cannot_reach_another_job_s_stub() -> None:
    """The same leak through the other door. A visitor faulting a segment id that
    belongs to the clerk's eviction is told the reference resolves to nothing, and
    the clerk's content is not paged in."""
    foreign = _clerk_evicted_segment()
    kernel, events = _build(
        [CLERK, VISITOR],
        {
            "payroll-clerk": [*CLERK_WORK, {"exit": True}],
            "visitor": [{"fault": foreign}, {"emit": "thanks"}, {"exit": True}],
        },
    )
    clerk = kernel.spawn(DescriptorName("payroll-clerk"))
    kernel.run_until_quiescent()
    archived = {e.store for e in _of(events, SegmentEvicted) if e.job == clerk.job_id}
    assert archived, "the clerk's output was meant to be archived"

    visitor = kernel.spawn(DescriptorName("visitor"))
    kernel.run_until_quiescent()

    handed = [e for e in _of(events, PagedIn) if e.job == visitor.job_id]
    assert not [e for e in handed if e.store in archived]
    transcript = " ".join(t.text for t in kernel.machine.transcript(visitor.job_id))
    assert "confidential" not in transcript


def test_a_need_is_answered_from_the_job_s_own_archive() -> None:
    """The control: the clerk asks for its own archived words and gets them back."""
    kernel, events = _build(
        [CLERK],
        {
            "payroll-clerk": [
                *CLERK_WORK,
                {"need": "payroll salary figures"},
                {"emit": "found"},
                {"exit": True},
            ]
        },
    )
    clerk = kernel.spawn(DescriptorName("payroll-clerk"))
    kernel.run_until_quiescent()

    archived = {e.store for e in _of(events, SegmentEvicted) if e.job == clerk.job_id}
    handed = [e for e in _of(events, PagedIn) if e.job == clerk.job_id]
    assert handed and handed[0].store in archived
    assert clerk.state is JobState.DONE
