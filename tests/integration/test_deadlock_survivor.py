# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Breaking a deadlock frees the survivor, not just the victim's locks.

``_do_acquire`` asks ``find_deadlock`` whether waiting would close a cycle before the
requester is registered as a waiter. When the victim is the other job and it dies,
the requester asks again from the top: the lock the victim held is free and it takes
it, or the lock is still held further round the cycle and the ordinary path parks it
as a waiter the eventual release will wake.

It used not to. The victim's fault released the contested lock, ``_do_release`` found
nobody waiting, and the requester was then parked on a lock that was already free.
The kernel went quiet with one job faulted and the other blocked on an available
resource. Issue #15.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import DeadlockDetected, Event
from zeos.core.ids import DescriptorName, JobState, PipeName, ResourceName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import ResourceSpec, ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

DOOR = ResourceName("door.south")
CRANE = ResourceName("crane.a")
LIFT = ResourceName("lift.a")
CMD = PipeName("user.cmd")


def _build(
    descriptors: Sequence[Mapping[str, Any]], scripts: Mapping[str, list[dict[str, Any]]]
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in scripts.items()}, block_size=8),
        pipes=PipeTable(),
        vectors=VectorTable(),
        world=WorldStore(),
        resources=ResourceTable(
            [
                ResourceSpec(name=DOOR, capacity=1),
                ResourceSpec(name=CRANE, capacity=1),
                ResourceSpec(name=LIFT, capacity=1),
            ]
        ),
        journal_sink=events,
        config=KernelConfig(case="deadlock-survivor", max_ticks=300),
    )
    kernel.start()
    return kernel, events


def _run_until_holds(kernel: Kernel, resource: ResourceName) -> None:
    for _ in range(20):
        if kernel.resources.get(resource).holders:
            return
        kernel.tick()
    raise AssertionError(f"nobody took {resource}")


def test_the_survivor_takes_the_lock_the_victim_freed() -> None:
    """The urgent job parks on a pipe, the slow one takes the crane and waits for the
    door, then the urgent job wakes and asks for the crane: a cycle whose victim is
    the slow job. The urgent job must get the crane it was waiting for."""
    urgent = {
        "name": "robot-a",
        "priority": 40,
        "resources": [str(DOOR), str(CRANE)],
        "pipes": {"stdin": str(CMD)},
    }
    slow = {"name": "robot-b", "priority": 80, "resources": [str(CRANE), str(DOOR)]}
    kernel, events = _build(
        [urgent, slow],
        {
            "robot-a": [
                {"acquire": str(DOOR)},
                {"read": str(CMD)},
                {"acquire": str(CRANE)},
                {"release": str(CRANE)},
                {"release": str(DOOR)},
                {"exit": True},
            ],
            "robot-b": [
                {"acquire": str(CRANE)},
                {"emit": "x"},
                {"acquire": str(DOOR)},
                {"release": str(DOOR)},
                {"release": str(CRANE)},
                {"exit": True},
            ],
        },
    )
    a = kernel.spawn(DescriptorName("robot-a"))
    _run_until_holds(kernel, DOOR)
    kernel.run_until_quiescent()
    b = kernel.spawn(DescriptorName("robot-b"))
    kernel.run_until_quiescent()
    assert a.state is JobState.BLOCKED and b.state is JobState.BLOCKED

    kernel.deliver(CMD, "go")
    kernel.run_until_quiescent()

    detected = [e for e in events if isinstance(e, DeadlockDetected)]
    assert detected and detected[0].victim == b.job_id, "the slow job was meant to be the victim"
    assert b.state is JobState.FAULTED
    assert not kernel.resources.get(CRANE).holders, "the victim's crane was released"
    assert a.state is JobState.DONE, (
        "the survivor is still blocked on a crane nobody holds; the deadlock was "
        "broken and the survivor was left behind"
    )


def test_the_survivor_parks_properly_when_the_victim_held_a_different_lock() -> None:
    """Three jobs round a cycle. The victim is the one holding the lock the *middle*
    job wants, not the one the requester wants, so the requester's second ask finds
    its lock still held and parks as a waiter; the unwinding then wakes it."""
    a = {
        "name": "robot-a",
        "priority": 40,
        "resources": [str(LIFT), str(DOOR)],
        "pipes": {"stdin": str(CMD)},
    }
    b = {"name": "robot-b", "priority": 60, "resources": [str(DOOR), str(CRANE)]}
    c = {"name": "robot-c", "priority": 80, "resources": [str(CRANE), str(LIFT)]}
    kernel, events = _build(
        [a, b, c],
        {
            "robot-a": [
                {"acquire": str(LIFT)},
                {"read": str(CMD)},
                {"acquire": str(DOOR)},
                {"release": str(DOOR)},
                {"release": str(LIFT)},
                {"exit": True},
            ],
            "robot-b": [
                {"acquire": str(DOOR)},
                {"emit": "x"},
                {"acquire": str(CRANE)},
                {"release": str(CRANE)},
                {"release": str(DOOR)},
                {"exit": True},
            ],
            "robot-c": [
                {"acquire": str(CRANE)},
                {"emit": "x"},
                {"acquire": str(LIFT)},
                {"release": str(LIFT)},
                {"release": str(CRANE)},
                {"exit": True},
            ],
        },
    )
    ja = kernel.spawn(DescriptorName("robot-a"))
    _run_until_holds(kernel, LIFT)
    kernel.run_until_quiescent()
    jc = kernel.spawn(DescriptorName("robot-c"))
    kernel.run_until_quiescent()
    jb = kernel.spawn(DescriptorName("robot-b"))
    kernel.run_until_quiescent()
    assert all(j.state is JobState.BLOCKED for j in (ja, jb, jc))

    kernel.deliver(CMD, "go")
    kernel.run_until_quiescent()

    detected = [e for e in events if isinstance(e, DeadlockDetected)]
    assert detected and detected[0].victim == jc.job_id
    assert jc.state is JobState.FAULTED
    assert ja.state is JobState.DONE and jb.state is JobState.DONE
    for name in (DOOR, CRANE, LIFT):
        assert not kernel.resources.get(name).holders


def test_the_cure_works_when_the_victim_is_the_requester() -> None:
    """The control: same two jobs, but the cycle closes from the slow job's side, so it
    is both requester and victim. The other job was already a registered waiter and
    the release wakes it."""
    a_desc = {"name": "robot-a", "priority": 80, "resources": [str(DOOR), str(CRANE)]}
    b_desc = {"name": "robot-b", "priority": 40, "resources": [str(CRANE), str(DOOR)]}
    kernel, events = _build(
        [a_desc, b_desc],
        {
            "robot-a": [
                {"acquire": str(DOOR)},
                {"emit": "x"},
                {"emit": "y"},
                {"acquire": str(CRANE)},
                {"release": str(CRANE)},
                {"release": str(DOOR)},
                {"exit": True},
            ],
            "robot-b": [
                {"acquire": str(CRANE)},
                {"emit": "x"},
                {"acquire": str(DOOR)},
                {"release": str(DOOR)},
                {"release": str(CRANE)},
                {"exit": True},
            ],
        },
    )
    a = kernel.spawn(DescriptorName("robot-a"))
    _run_until_holds(kernel, DOOR)
    b = kernel.spawn(DescriptorName("robot-b"))
    kernel.run_until_quiescent()

    detected = [e for e in events if isinstance(e, DeadlockDetected)]
    assert detected and detected[0].victim == a.job_id
    assert a.state is JobState.FAULTED
    assert b.state is JobState.DONE
    assert not kernel.resources.get(DOOR).holders and not kernel.resources.get(CRANE).holders
