# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A handler that faults must not disarm its own vector.

``VectorTable`` serialises a vector against itself with an ``active`` counter:
``mark_dispatched`` raises it, ``mark_complete`` lowers it, and while it is above
zero a ``coalesce`` vector folds new firings into the pending dispatch and a
``queue`` vector parks them behind the running instance. Both are correct
storm control -- provided the counter comes back down whenever the instance
ends, and a fault is one of the ways it ends.

It used not to. Completion and cancellation each released the vector and
``Kernel._raise_fault`` did not, so a handler that aborted left ``active`` at one
for the rest of the run and the vector was dead: nothing on that pipe was ever
dispatched again, and the journal showed the alarm arriving and being swallowed
-- as ``VectorCoalesced`` under ``coalesce``, as a queue note under ``queue`` --
which reads exactly like healthy storm control. The release now lives in
``Kernel._transition``, the one place every terminal state passes through.

Both serialising policies are exercised because the accounting is shared: a
regression that unsticks only one of them has patched a symptom.
"""

from __future__ import annotations

import pytest

from zeos.core.events import Event, FaultRaised, VectorFired
from zeos.core.ids import (
    DescriptorName,
    JobState,
    PipeName,
    Priority,
    VectorName,
    VectorPolicy,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

ALARM = PipeName("sensors.alarm")
HANDLER = DescriptorName("alarm-handler")
VECTOR = VectorName("alarm")

#: Five tokens against a one-token budget: the handler breaches on its first decode
#: and its ``on_fault: abort`` terminates it FAULTED. Any hard fault would do; the
#: budget is simply the one a scripted job can reach without a device or a segment.
CRASHES = Script.from_spec([{"emit": "one two three four five"}, {"exit": True}])
SURVIVES = Script.from_spec([{"emit": "ack"}, {"exit": True}])

SERIALISING = [VectorPolicy.COALESCE, VectorPolicy.QUEUE]


def _build(script: Script, policy: VectorPolicy) -> tuple[Kernel, list[Event]]:
    descriptor = Descriptor.from_frontmatter(
        {"name": str(HANDLER), "priority": 5, "budget": {"tokens": 1}, "on_fault": "abort"},
        body="handle the alarm",
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors={HANDLER: descriptor},
        machine=ScriptedMachine({str(HANDLER): script}, block_size=16),
        pipes=PipeTable([PipeSpec(name=ALARM, device=True, capacity_tokens=8)]),
        vectors=VectorTable(
            [
                VectorSpec(
                    name=VECTOR,
                    source=ALARM,
                    handler=HANDLER,
                    priority=Priority(5),
                    policy=policy,
                )
            ]
        ),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="dead-interrupt", max_ticks=500),
    )
    kernel.start()
    return kernel, events


def _trip(kernel: Kernel, events: list[Event]) -> list[Event]:
    """Raise the alarm once and run the firing out. Returns the events it produced."""
    before = len(events)
    kernel.deliver(ALARM, "trip")
    for _ in range(10):
        if not kernel.tick():
            break
    return events[before:]


@pytest.mark.parametrize("policy", SERIALISING, ids=str)
def test_a_faulting_handler_leaves_its_vector_armed(policy: VectorPolicy) -> None:
    """One crash must not leave the alarm deaf for the rest of the run."""
    kernel, events = _build(CRASHES, policy)

    first = _trip(kernel, events)
    assert [e for e in first if isinstance(e, VectorFired)], "the alarm never fired at all"
    assert [e for e in first if isinstance(e, FaultRaised)], "the handler was meant to crash"
    assert [j for j in kernel.sched.jobs() if j.state is JobState.FAULTED], "no job faulted"

    second = _trip(kernel, events)

    assert [e for e in second if isinstance(e, VectorFired)], (
        "the vector is latched: the crashed instance is still counted active, "
        "so this alarm was swallowed by storm control"
    )
    assert kernel.vectors.active(VECTOR) == 0, "a terminal handler is not an active instance"


@pytest.mark.parametrize("policy", SERIALISING, ids=str)
def test_a_completing_handler_leaves_its_vector_armed(policy: VectorPolicy) -> None:
    """The control: the same vector with a handler that exits cleanly.

    Same pipe, same two alarms -- only the handler's exit differs, so if the case
    above ever fails again the difference is in how the first instance ended, not in
    coalescing, throttling, or the pipe.
    """
    kernel, events = _build(SURVIVES, policy)

    first = _trip(kernel, events)
    assert not [e for e in first if isinstance(e, FaultRaised)], "the handler was meant to survive"

    second = _trip(kernel, events)

    assert [e for e in second if isinstance(e, VectorFired)], "a healthy vector re-arms"
    assert kernel.vectors.active(VECTOR) == 0
