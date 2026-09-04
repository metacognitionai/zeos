# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A blocking read must actually deliver, not just wake.

The half of core §4.1 that is easy to leave untested. ``read`` on an empty pipe
deschedules the reader and a peer's write wakes it -- both of which are covered
elsewhere. What is covered here is the step after: on its next dispatch the woken job
must *complete* the read it was parked on, so the token it waited for lands in its
context and leaves the buffer.

This regressed once and was invisible for a long time. ``_do_read`` parked the read in
``blocked_on`` / ``blocked_reason`` and ``_service_pending`` looked for exactly those,
but ``Scheduler.wake`` clears both on BLOCKED -> READY, so the branch was unreachable
for every woken reader. The three sibling operations were unaffected because each parks
itself in a field ``wake`` does not touch, which is why nothing else caught it: a SELECT
recovers correctly, and a test whose pipe is pre-filled never blocks at all.

So the case has to be exactly this shape -- the reader must reach the pipe **first**,
while it is genuinely empty -- or it tests nothing.
"""

from __future__ import annotations

from zeos.core.ids import DescriptorName, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

PIPE = PipeName("p")


def _build() -> tuple[PipeTable, ScriptedMachine, Kernel]:
    waiter = Descriptor.from_frontmatter(
        {"name": "waiter", "priority": 50, "pipes": {"stdin": "p"}}, body="w"
    )
    sender = Descriptor.from_frontmatter(
        {"name": "sender", "priority": 60, "pipes": {"stdout": "p"}}, body="s"
    )
    scripts = {
        "waiter": Script.from_spec([{"read": "stdin"}, {"emit": "resumed"}, {"exit": True}]),
        "sender": Script.from_spec(
            [{"emit": "sending"}, {"write": {"pipe": "stdout", "text": "42"}}, {"exit": True}]
        ),
    }
    machine = ScriptedMachine(scripts, block_size=16)
    pipes = PipeTable([PipeSpec(name=PIPE, capacity_tokens=64)])
    kernel = Kernel(
        descriptors={DescriptorName("waiter"): waiter, DescriptorName("sender"): sender},
        machine=machine,
        pipes=pipes,
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(),
        config=KernelConfig(case="blocking-read"),
    )
    kernel.start()
    # The waiter is spawned first so it reaches the empty pipe before anything is in it.
    kernel.spawn(DescriptorName("waiter"))
    kernel.spawn(DescriptorName("sender"))
    return pipes, machine, kernel


def test_a_woken_reader_completes_the_read_it_was_parked_on() -> None:
    pipes, machine, kernel = _build()
    kernel.run_until_quiescent(200)

    waiter = next(j for j in kernel.sched.jobs() if j.name == DescriptorName("waiter"))
    transcript = " ".join(t.text for t in machine.transcript(waiter.job_id))

    assert "42" in transcript, "the token the job blocked for must reach its context"
    assert pipes.get(PIPE).available == 0, "and must leave the buffer"
    assert waiter.pending_read is None, "the parked read is cleared once completed"


def test_the_read_lands_before_the_job_carries_on() -> None:
    """Ordering, not just arrival: a resumed job must see its input before it acts."""
    _, machine, kernel = _build()
    kernel.run_until_quiescent(200)

    waiter = next(j for j in kernel.sched.jobs() if j.name == DescriptorName("waiter"))
    texts = [t.text for t in machine.transcript(waiter.job_id)]
    assert texts.index("42") < texts.index("resumed")
