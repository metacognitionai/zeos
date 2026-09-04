# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""An interrupted job must not lose its place to lower-priority work.

``Kernel.tick`` used to ask ``should_preempt()`` first, which returns ``best_ready()``
whenever nothing is running, and reached ``pop_suspended()`` only if that returned
nothing. So a READY job was dispatched ahead of the suspension stack **at any
priority**: being interrupted demoted a job below every runnable job in the system,
however unimportant.

That is not what the scheduler says it does. ``Scheduler``'s docstring describes the
stack as "ordinary interrupt semantics: the most recently interrupted job resumes first
as higher-priority work drains" -- as *higher-priority* work drains. A priority-90
background job is not higher-priority work, and a priority-20 job does not stop being
priority 20 because a handler ran.

The practical shape is a priority inversion that the rest of this kernel takes great
care to avoid elsewhere: it implements priority inheritance for pipes and for resources
precisely so that important work is not held up by unimportant work. An interrupt
should not be the one path that reintroduces it.

The four cases below fix the boundaries in both directions -- the stack competes on
priority, it does not win outright.
"""

from __future__ import annotations

from zeos.core.events import Event, JobDispatched, JobPreempted
from zeos.core.ids import DescriptorName, JobId, PipeName, Priority, VectorName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

IRQ = PipeName("irq")


def _run(background_priority: int) -> tuple[Kernel, list[Event]]:
    """One job interrupted mid-work, and another that turns up while the handler runs.

    ``background`` is spawned *after* the interrupt has been delivered so that
    ``important`` is reliably the job holding the machine when the handler preempts it,
    whatever priority the caller gives the background job. It then sits READY across the
    handler's run -- which is the situation being tested: what the kernel picks when the
    handler exits and both a ready job and a suspended one are waiting.
    """
    descriptors = {
        DescriptorName("important"): Descriptor.from_frontmatter(
            {"name": "important", "priority": 20}, body="i"
        ),
        DescriptorName("background"): Descriptor.from_frontmatter(
            {"name": "background", "priority": background_priority}, body="b"
        ),
        DescriptorName("handler"): Descriptor.from_frontmatter(
            {
                "name": "handler",
                "priority": 5,
                "budget": {"tokens": 64},
                "pipes": {"stdin": "irq"},
            },
            body="h",
        ),
    }
    long_run = [{"emit": "work"} for _ in range(40)] + [{"exit": True}]
    machine = ScriptedMachine(
        {
            "important": Script.from_spec(long_run),
            "background": Script.from_spec(long_run),
            "handler": Script.from_spec([{"emit": "ack"}, {"exit": True}]),
        },
        block_size=16,
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors=descriptors,
        machine=machine,
        pipes=PipeTable([PipeSpec(name=IRQ, device=True, capacity_tokens=32)]),
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
        config=KernelConfig(case="resume-priority", max_ticks=900),
    )
    kernel.start()
    kernel.spawn(DescriptorName("important"))
    now = 0
    for tick in range(40):
        if tick == 5:
            kernel.deliver(IRQ, "go")
        if tick == 6:
            kernel.spawn(DescriptorName("background"))
        kernel.advance_time(now)
        kernel.tick()
        now += 1_000_000
    return kernel, events


def _job(kernel: Kernel, name: str) -> JobId:
    return next(j.job_id for j in kernel.sched.jobs() if j.name == DescriptorName(name))


def _after_handler(kernel: Kernel, events: list[Event]) -> list[str]:
    """Dispatch order once the handler has had the machine, as descriptor names."""
    by_id = {j.job_id: str(j.name) for j in kernel.sched.jobs()}
    order = [e.job for e in events if isinstance(e, JobDispatched)]
    handler = _job(kernel, "handler")
    assert handler in order, "the interrupt never ran"
    return [by_id[j] for j in order[order.index(handler) + 1 :]]


def test_the_handler_preempts_the_important_job() -> None:
    """The half that already worked, pinned so the fix is not credited with more."""
    kernel, events = _run(background_priority=90)
    preemptions = [e for e in events if isinstance(e, JobPreempted)]
    assert [e.job for e in preemptions] == [_job(kernel, "important")]
    assert preemptions[0].by_priority == 5


def test_the_interrupted_job_resumes_before_lower_priority_work() -> None:
    """A priority-90 job must not take the machine from a suspended priority-20 one."""
    order = _run(background_priority=90)
    assert _after_handler(*order)[0] == "important"


def test_the_interrupted_job_wins_a_tie() -> None:
    """Equal priority resolves to the suspended job: it is mid-work, the other is not.

    This is the case the counting tutorial hits -- two peers at the same priority, one
    interrupted and one made runnable behind it -- and the case a purely priority-ordered
    fix would still get wrong, since ``best_ready`` breaks ties by spawn order.
    """
    order = _run(background_priority=20)
    assert _after_handler(*order)[0] == "important"


def test_higher_priority_ready_work_still_goes_first() -> None:
    """The stack competes on priority; it does not win outright.

    The other side of the boundary. Resuming eagerly would be its own inversion, so a
    genuinely more urgent ready job must still be dispatched ahead of the stack.
    """
    order = _run(background_priority=10)
    assert _after_handler(*order)[0] == "background"
