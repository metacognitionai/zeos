# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A job that faults must give its body back.

A lease is recorded twice: as an ordinary resource (``body:<platform>``), so that
waiting for a body reuses blocking and inheritance, and in the lease table, which
is what the allocator consults. Every terminal path -- completion, cancellation,
and a fault that aborts or escalates -- clears both through ``Kernel._return_body``
and journals a ``Disembodied``.

The fault path used to release only the resource. The lease table went on naming
the dead job, the platform never returned to ``free_platforms``, and the next job
that needed the body neither got it nor blocked: each tick the allocator refused
it (the lease table said taken) and it acquired the body's lock (the resource
table said free), so the kernel never reached quiescence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.embodiment import PlatformProfile
from zeos.core.events import Disembodied, Embodied, Event, FaultRaised
from zeos.core.ids import DescriptorName, JobState
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

CARRIER_7 = PlatformProfile(
    name="carrier-7",
    locomotion="wheeled",
    tooling=frozenset({"gripper-std"}),
    battery=0.95,
    location="bay-3",
)

CARRY: dict[str, Any] = {
    "name": "carry-beam",
    "priority": 40,
    "requires": {"tooling": ["gripper-std"], "locomotion": "wheeled"},
}
#: Five tokens against a one-token budget: the carrier breaches on its first decode
#: and ``on_fault: abort`` terminates it FAULTED while it is wearing the body.
CRASHING_CARRY: dict[str, Any] = {**CARRY, "budget": {"tokens": 1}, "on_fault": "abort"}

WORK = [{"emit": "one two three four five"}, {"exit": True}]


def _build(descriptor: Mapping[str, Any]) -> tuple[Kernel, list[Event]]:
    """One body, and one descriptor that needs it."""
    events: list[Event] = []
    kernel = Kernel(
        descriptors={DescriptorName("carry-beam"): Descriptor.from_frontmatter(descriptor)},
        machine=ScriptedMachine({"carry-beam": Script.from_spec(WORK)}, block_size=8),
        pipes=PipeTable(),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        platforms=(CARRIER_7,),
        journal_sink=events,
        config=KernelConfig(case="faulted-lease", max_ticks=200),
    )
    kernel.start()
    return kernel, events


def _of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def test_a_faulting_job_returns_its_body() -> None:
    kernel, events = _build(CRASHING_CARRY)

    first = kernel.spawn(DescriptorName("carry-beam"))
    kernel.run_until_quiescent()
    assert first.state is JobState.FAULTED, "the carrier was meant to crash"
    assert _of(events, FaultRaised), "no fault was journalled"
    assert [e for e in _of(events, Embodied) if e.job == first.job_id], "it never wore the body"

    assert [e for e in _of(events, Disembodied) if e.job == first.job_id], (
        "a job that will never run again is still recorded as wearing the body"
    )
    assert kernel.leases.free_platforms() == ("carrier-7",), "the body did not return to the pool"

    second = kernel.spawn(DescriptorName("carry-beam"))
    kernel.run_until_quiescent()
    assert [e for e in _of(events, Embodied) if e.job == second.job_id], "the next job got no body"


def test_a_completing_job_returns_its_body() -> None:
    """The control: same body, same descriptor, a carrier that exits cleanly."""
    kernel, events = _build(CARRY)

    first = kernel.spawn(DescriptorName("carry-beam"))
    kernel.run_until_quiescent()
    assert first.state is JobState.DONE
    assert not _of(events, FaultRaised), "the carrier was meant to survive"

    assert [e for e in _of(events, Disembodied) if e.job == first.job_id]
    assert kernel.leases.free_platforms() == ("carrier-7",)

    second = kernel.spawn(DescriptorName("carry-beam"))
    kernel.run_until_quiescent()
    assert [e for e in _of(events, Embodied) if e.job == second.job_id]
    assert second.state is JobState.DONE
