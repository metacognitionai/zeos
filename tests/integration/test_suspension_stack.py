# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A job that faults while suspended is left on the suspension stack.

``_preempt`` pushes the displaced job onto the stack and *then* checks whether it has
been preempted past ``starvation_limit``; if it has, it raises a starvation fault on the
job it just stacked. Nothing removes it. ``drop_from_stack`` exists and is called from
exactly one place -- cancellation -- so the fault path leaks.

``pop_suspended`` tolerates this: it skips entries whose job is no longer SUSPENDED, so
nothing is mis-dispatched. Two things do not tolerate it.

``cancel_below(depth)`` pops ``depth`` entries and returns every job it finds, terminal
or not. A handler with ``on_complete: cancel-below:2`` unwinding an emergency will spend
its budget on jobs that are already dead and leave live ones running -- the opposite of
what the policy is for.

``stack_depth`` is journalled on every ``JobPreempted`` and folded by the monitor into
``max_stack_depth``. ``drop_from_stack``'s own docstring says depth accuracy matters:
"wrong ``stack_depth``, and depth is journalled and asserted on".

Found by fuzzing; this is the minimal deterministic case, with no model involved.
"""

from __future__ import annotations

from zeos.core.events import Event, FaultRaised
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    JobState,
    PipeName,
    Priority,
    VectorName,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

IRQ = PipeName("irq")


def _starve_a_job() -> tuple[Kernel, list[Event]]:
    """A job that never blocks, interrupted until it starves.

    The victim has to be one that never yields, or it would block and leave the stack
    of its own accord; the handler has to outrank it, or there would be no preemption.
    """
    victim = Descriptor.from_frontmatter({"name": "victim", "priority": 80}, body="v")
    handler = Descriptor.from_frontmatter(
        {"name": "handler", "priority": 5, "budget": {"tokens": 64}, "pipes": {"stdin": "irq"}},
        body="h",
    )
    machine = ScriptedMachine(
        {
            "victim": Script.from_spec([{"emit": "work"} for _ in range(400)] + [{"exit": True}]),
            "handler": Script.from_spec([{"emit": "ack"}, {"exit": True}]),
        },
        block_size=16,
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors={DescriptorName("victim"): victim, DescriptorName("handler"): handler},
        machine=machine,
        pipes=PipeTable([PipeSpec(name=IRQ, device=True, capacity_tokens=64)]),
        vectors=VectorTable(
            [
                VectorSpec(
                    name=VectorName("irq"),
                    source=IRQ,
                    handler=DescriptorName("handler"),
                    priority=Priority(5),
                )
            ]
        ),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="stale-stack", max_ticks=3000),
    )
    kernel.start()
    kernel.spawn(DescriptorName("victim"))
    now = 0
    for tick in range(300):
        if tick % 6 == 0:
            kernel.deliver(IRQ, "ping")
        kernel.advance_time(now)
        kernel.tick()
        now += 1_000_000
        victim_job = next(j for j in kernel.sched.jobs() if j.name == DescriptorName("victim"))
        if victim_job.state is JobState.FAULTED:
            break
    return kernel, events


def test_the_victim_starves_as_designed() -> None:
    """The half that works, pinned so the fix is not credited with more than it does."""
    kernel, events = _starve_a_job()
    faults: list[FaultRaised] = [e for e in events if isinstance(e, FaultRaised)]
    assert [f.fault for f in faults] == [FaultKind.STARVATION]
    victim = next(j for j in kernel.sched.jobs() if j.name == DescriptorName("victim"))
    assert victim.state is JobState.FAULTED


def test_a_job_that_faults_while_suspended_leaves_the_stack() -> None:
    kernel, _ = _starve_a_job()
    victim = next(j for j in kernel.sched.jobs() if j.name == DescriptorName("victim"))

    assert victim.job_id not in kernel.sched.stack, "a terminal job must not remain stacked"
    assert kernel.sched.stack_depth == 0, (
        "stack_depth is journalled on every preemption and folded into max_stack_depth"
    )


def test_cancel_below_does_not_spend_a_frame_on_a_dead_job() -> None:
    """The consequence: an emergency unwind that unwinds nothing."""
    kernel, _ = _starve_a_job()
    cancelled = kernel.sched.cancel_below(1)
    assert all(not job.state.is_terminal for job in cancelled), (
        f"cancel_below returned terminal jobs: "
        f"{[(int(j.job_id), j.state.value) for j in cancelled]}"
    )
