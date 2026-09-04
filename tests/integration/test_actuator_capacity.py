# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""An actuator pipe must not fill up, because nothing can drain it.

A pipe declared with ``world_object:`` is an actuator: a write to it changes world
state. It is also, today, an ordinary bounded buffer -- ``_do_write`` appends the
payload *and* applies the world write. Nothing reads it back, and the only wake for a
blocked writer is inside ``_consume_read``, so a job that actuates once per turn walks
its own buffer to the capacity line and parks there for good.

The failure is silent in the way that matters most: no fault, no deadlock detection
(``find_deadlock`` looks for resource cycles, not for blocked-forever), and
``sched.is_quiescent()`` goes true, so a driver loop simply idles. A long-running case
stops and says nothing.

Buffering is the wrong model for an actuator. A message is a thing you send to someone;
an actuation is an effect on the world, and the world already holds the result. So the
buffer keeps the *latest* value and nothing more -- a register rather than a queue --
which is also what makes reading one back mean something, since a reader wants the
current value and never the backlog.
"""

from __future__ import annotations

from zeos.core.events import Event, JobBlocked, PipeWritten
from zeos.core.ids import DescriptorName, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import ObjectName, WorldStore

ACT = PipeName("progress")
OBJ = ObjectName("count")
WRITES = 40
CAPACITY = 8


def _run() -> tuple[Kernel, list[Event]]:
    """One job actuating far more times than the buffer could ever hold."""
    descriptors = {
        DescriptorName("recorder"): Descriptor.from_frontmatter(
            {
                "name": "recorder",
                "priority": 50,
                "writes": ["count"],
                "pipes": {"tools": "progress"},
            },
            body="r",
        )
    }
    steps = [{"write": {"pipe": "progress", "text": str(n)}} for n in range(1, WRITES + 1)]
    machine = ScriptedMachine(
        {"recorder": Script.from_spec([*steps, {"exit": True}])}, block_size=16
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors=descriptors,
        machine=machine,
        pipes=PipeTable([PipeSpec(name=ACT, world_object="count", capacity_tokens=CAPACITY)]),
        vectors=VectorTable([]),
        world=WorldStore(),
        resources=ResourceTable(),
        journal_sink=events,
        config=KernelConfig(case="actuator-capacity", max_ticks=2000),
    )
    kernel.start()
    kernel.spawn(DescriptorName("recorder"))
    kernel.run_until_quiescent()
    return kernel, events


def test_an_actuator_never_backs_up_against_its_own_capacity() -> None:
    """Forty writes into a buffer that holds eight. All forty must land."""
    _, events = _run()
    stalled = [e for e in events if isinstance(e, JobBlocked) and e.reason == "write-full"]
    assert not stalled, (
        f"the writer parked against a buffer nobody can drain after "
        f"{len([e for e in events if isinstance(e, PipeWritten)])} of {WRITES} writes"
    )
    landed = [e for e in events if isinstance(e, PipeWritten) and e.pipe == ACT]
    assert len(landed) == WRITES


def test_the_world_holds_the_last_value_written() -> None:
    """The point of the write. Capacity must not silently drop the tail."""
    kernel, _ = _run()
    assert kernel.world.get(OBJ) == str(WRITES)


def test_the_buffer_holds_the_current_value_not_the_history() -> None:
    """A register, not a queue -- which is why capacity stops being a countdown.

    Asserted as an upper bound rather than an exact figure: what matters is that
    occupancy does not grow with the number of writes.
    """
    kernel, _ = _run()
    assert kernel.pipes.get(ACT).available <= CAPACITY
    assert kernel.pipes.get(ACT).peek()[-1].text == str(WRITES)
