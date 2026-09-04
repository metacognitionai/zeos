# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""R0 -- the resource table.

Three parts of the design were waiting on the same missing subsystem: the multi-threaded extension's
mutexes, ZEOS-Fleet's physical locks, and ZEOS-Fleet's embodiment leases
("a resource in the core §2.2 sense"). This is that subsystem.

The scenario throughout is Fleet's own: two robots and a doorway. It is the
smallest thing that exercises capacity, inheritance, and deadlock at once.

**One structural fact shapes every test here.** With a single running job,
contention can only arise through *preemption*: a lower-priority contender simply
never runs while the holder holds, so it never blocks. Every contention test
therefore starts the holder, lets it take the resource, and *then* introduces an
urgent job -- which is exactly the priority-inversion scenario, and exactly why
inheritance matters. Under the SMP extension contention would also arise from
parallelism; here it cannot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.events import (
    DeadlockDetected,
    Event,
    FaultRaised,
    PriorityInherited,
    PriorityRestored,
    ResourceAcquired,
    ResourceBlocked,
    ResourceReleased,
)
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    JobId,
    JobState,
    Priority,
    ResourceKind,
    ResourceName,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import (
    Deadlock,
    ResourceError,
    ResourceSpec,
    ResourceTable,
    lock_order_violations,
)
from zeos.core.vectors import VectorTable
from zeos.descriptor.lint import Severity, lint
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

DOOR = ResourceSpec(name=ResourceName("door.south"), capacity=1, description="doorway")
CORRIDOR = ResourceSpec(
    name=ResourceName("corridor.main"),
    capacity=2,
    kind=ResourceKind.SEMAPHORE,
    description="admits two",
)
CRANE = ResourceSpec(name=ResourceName("crane.a"), capacity=1)


def build(
    descriptors: Sequence[Mapping[str, Any]],
    scripts: Mapping[str, list[dict[str, Any]]],
    resources: Sequence[ResourceSpec] = (DOOR, CORRIDOR, CRANE),
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
        resources=ResourceTable(resources),
        journal_sink=events,
        config=KernelConfig(case="r0"),
    )
    kernel.start()
    return kernel, events


def of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


def carrier(name: str, priority: int, script_resources: list[str]) -> dict[str, Any]:
    return {"name": name, "priority": priority, "resources": script_resources}


def run_until_holds(kernel: Kernel, job_name: str, resource: str, limit: int = 20) -> None:
    """Tick until the named job holds the resource.

    Contention needs the holder to be *already holding* when the contender arrives,
    so tests stage it explicitly rather than hoping for an interleaving.
    """
    target = ResourceName(resource)
    for _ in range(limit):
        holders = kernel.resources.get(target).holders
        if any(kernel.sched.get(h).name == job_name for h in holders):
            return
        if not kernel.tick():
            break
    raise AssertionError(f"{job_name} never acquired {resource}")


# --- acquisition and capacity ----------------------------------------------


def test_a_mutex_admits_one_and_blocks_the_second() -> None:
    kernel, events = build(
        [carrier("sweeper", 90, ["door.south"]), carrier("carrier-lead", 10, ["door.south"])],
        {
            "sweeper": [
                {"acquire": "door.south"},
                {"emit": "sweeping"},
                {"emit": "still sweeping"},
                {"release": "door.south"},
                {"exit": True},
            ],
            "carrier-lead": [
                {"acquire": "door.south"},
                {"emit": "through"},
                {"release": "door.south"},
                {"exit": True},
            ],
        },
    )
    a = kernel.spawn(DescriptorName("sweeper"))
    run_until_holds(kernel, "sweeper", "door.south")
    b = kernel.spawn(DescriptorName("carrier-lead"))  # urgent; preempts the sweeper
    kernel.run_until_quiescent()

    assert of(events, ResourceBlocked), "the carrier must have waited on the doorway"
    assert a.state is JobState.DONE and b.state is JobState.DONE
    assert len(of(events, ResourceAcquired)) == 2, "both eventually got through"


def test_a_semaphore_admits_its_capacity() -> None:
    """A corridor might admit two. A mutex is the degenerate case, not a separate
    mechanism."""
    descriptors = [carrier(f"r{i}", 90 - i * 10, ["corridor.main"]) for i in range(3)]
    scripts = {
        f"r{i}": [
            {"acquire": "corridor.main"},
            {"emit": "in"},
            {"emit": "still in"},
            {"release": "corridor.main"},
            {"exit": True},
        ]
        for i in range(3)
    }
    kernel, events = build(descriptors, scripts)
    # Each is more urgent than the last, so each preempts and piles onto the corridor.
    for i in range(3):
        kernel.spawn(DescriptorName(f"r{i}"))
        if i < 2:
            run_until_holds(kernel, f"r{i}", "corridor.main")
    kernel.run_until_quiescent()

    acquired = of(events, ResourceAcquired)
    assert len(acquired) == 3
    assert max(a.holders for a in acquired) == 2, "never more than capacity at once"
    assert acquired[0].capacity == 2


def test_blocking_on_a_resource_costs_nothing() -> None:
    """The same deschedule as blocking on a pipe: a job waiting on a doorway costs
    exactly what a job waiting on a tool result costs."""
    from zeos.core.events import Decoded

    kernel, events = build(
        [carrier("holder", 90, ["door.south"]), carrier("waiter", 10, ["door.south"])],
        {
            "holder": [
                {"acquire": "door.south"},
                {"emit": "holding it a while"},
                {"emit": "still holding"},
                {"release": "door.south"},
                {"exit": True},
            ],
            "waiter": [
                {"acquire": "door.south"},
                {"emit": "finally"},
                {"release": "door.south"},
                {"exit": True},
            ],
        },
    )
    kernel.spawn(DescriptorName("holder"))
    run_until_holds(kernel, "holder", "door.south")
    waiter = kernel.spawn(DescriptorName("waiter"))
    kernel.run_until_quiescent()

    blocked_at = next(i for i, e in enumerate(events) if isinstance(e, ResourceBlocked))
    woken_at = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, ResourceReleased) and e.woke == waiter.job_id
    )
    decoded_while_blocked = [
        e for e in events[blocked_at:woken_at] if isinstance(e, Decoded) and e.job == waiter.job_id
    ]
    assert not decoded_while_blocked, "a blocked job must not decode"


def test_holding_a_resource_is_released_on_completion() -> None:
    """A job that died holding a lock would block every waiter forever."""
    kernel, _ = build(
        [carrier("forgetful", 50, ["door.south"])],
        {"forgetful": [{"acquire": "door.south"}, {"exit": True}]},  # never releases
    )
    job = kernel.spawn(DescriptorName("forgetful"))
    kernel.run_until_quiescent()
    assert job.state is JobState.DONE
    assert kernel.resources.get(ResourceName("door.south")).holders == []


# --- priority inheritance over a real resource ------------------------------


def test_holder_inherits_the_blocked_jobs_priority() -> None:
    """Core §2.2, now over a resource the table actually knows the holders of --
    unambiguous, unlike the pipe case which must infer counterparties."""
    kernel, events = build(
        [carrier("sweeper", 90, ["door.south"]), carrier("carrier-lead", 10, ["door.south"])],
        {
            "sweeper": [
                {"acquire": "door.south"},
                {"emit": "sweeping"},
                {"emit": "still sweeping"},
                {"release": "door.south"},
                {"exit": True},
            ],
            "carrier-lead": [
                {"acquire": "door.south"},
                {"emit": "carrying"},
                {"release": "door.south"},
                {"exit": True},
            ],
        },
    )
    sweeper = kernel.spawn(DescriptorName("sweeper"))
    run_until_holds(kernel, "sweeper", "door.south")
    lead = kernel.spawn(DescriptorName("carrier-lead"))
    kernel.run_until_quiescent()

    inherited = [e for e in of(events, PriorityInherited) if e.job == sweeper.job_id]
    assert inherited, "the sweeper must inherit the carriers' priority"
    assert inherited[0].from_priority == Priority(90)
    assert inherited[0].to_priority == Priority(10)
    assert inherited[0].blocked_job == lead.job_id
    assert inherited[0].resource == "door.south"


def test_inherited_priority_is_returned_on_release() -> None:
    kernel, events = build(
        [carrier("sweeper", 90, ["door.south"]), carrier("carrier-lead", 10, ["door.south"])],
        {
            "sweeper": [
                {"acquire": "door.south"},
                {"emit": "sweeping"},
                {"release": "door.south"},
                {"emit": "back to normal"},
                {"exit": True},
            ],
            "carrier-lead": [{"acquire": "door.south"}, {"release": "door.south"}, {"exit": True}],
        },
    )
    sweeper = kernel.spawn(DescriptorName("sweeper"))
    run_until_holds(kernel, "sweeper", "door.south")
    kernel.spawn(DescriptorName("carrier-lead"))
    kernel.run_until_quiescent()

    restored = [e for e in of(events, PriorityRestored) if e.job == sweeper.job_id]
    assert restored and restored[0].to_priority == Priority(90)
    assert sweeper.current_priority == Priority(90)


def test_waiters_wake_by_priority_not_arrival() -> None:
    """A FIFO queue would let a low-priority early arrival hold up an urgent job --
    reintroducing at the resource layer exactly the inversion inheritance prevents."""
    table = ResourceTable([DOOR])
    door = table.get(ResourceName("door.south"))
    door.acquire(JobId(1))
    door.add_waiter(JobId(2))  # arrives first, low priority
    door.add_waiter(JobId(3))  # arrives second, urgent

    nxt = table.next_waiter(
        ResourceName("door.south"),
        {JobId(2): Priority(90), JobId(3): Priority(5)},
    )
    assert nxt == JobId(3)


# --- deadlock ---------------------------------------------------------------


def test_two_robots_facing_off_in_a_corridor_is_detected() -> None:
    """The Fleet design's exact example: a deadlock on two mutexes, found in the global
    lock graph and broken by the declared victim policy."""
    kernel, events = build(
        [
            carrier("robot-a", 80, ["door.south", "crane.a"]),
            carrier("robot-b", 40, ["crane.a", "door.south"]),
        ],
        {
            # A takes door then crane; B takes crane then door. Opposite orders.
            "robot-a": [
                {"acquire": "door.south"},
                {"emit": "have door"},
                {"emit": "working"},
                {"acquire": "crane.a"},
                {"exit": True},
            ],
            "robot-b": [
                {"acquire": "crane.a"},
                {"emit": "have crane"},
                {"acquire": "door.south"},
                {"exit": True},
            ],
        },
    )
    a = kernel.spawn(DescriptorName("robot-a"))
    run_until_holds(kernel, "robot-a", "door.south")
    kernel.spawn(DescriptorName("robot-b"))  # urgent: preempts A, grabs the crane
    kernel.run_until_quiescent()

    detected = of(events, DeadlockDetected)
    assert detected, "the cycle must be found"
    assert len(detected[0].cycle) == 2
    assert set(detected[0].resources) == {"door.south", "crane.a"}
    # "Lower mission priority backs out". Compared on *base* priority --
    # by now the two have lent priority to each other, which would otherwise flatten
    # the comparison to a tie-break on job id.
    assert detected[0].victim == a.job_id


def test_the_deadlock_fault_names_the_cycle() -> None:
    """The Fleet design asks for "a loud scheduler fault naming the cycle" -- so the cycle
    is in the fault, not something an operator reconstructs from blocked jobs."""
    kernel, events = build(
        [
            carrier("robot-a", 80, ["door.south", "crane.a"]),
            carrier("robot-b", 40, ["crane.a", "door.south"]),
        ],
        {
            "robot-a": [
                {"acquire": "door.south"},
                {"emit": "x"},
                {"emit": "y"},
                {"acquire": "crane.a"},
                {"exit": True},
            ],
            "robot-b": [
                {"acquire": "crane.a"},
                {"emit": "x"},
                {"acquire": "door.south"},
                {"exit": True},
            ],
        },
    )
    kernel.spawn(DescriptorName("robot-a"))
    run_until_holds(kernel, "robot-a", "door.south")
    kernel.spawn(DescriptorName("robot-b"))
    kernel.run_until_quiescent()

    faults = [f for f in of(events, FaultRaised) if f.fault is FaultKind.DEADLOCK]
    assert faults
    assert "waits on" in faults[0].detail and "cycle" in faults[0].detail
    assert "door.south" in faults[0].detail and "crane.a" in faults[0].detail


def test_no_false_deadlock_when_orders_agree() -> None:
    """The discriminating case: same two resources, consistent order, no cycle."""
    kernel, events = build(
        [
            carrier("robot-a", 80, ["door.south", "crane.a"]),
            carrier("robot-b", 40, ["door.south", "crane.a"]),
        ],
        {
            "robot-a": [
                {"acquire": "door.south"},
                {"emit": "x"},
                {"acquire": "crane.a"},
                {"release": "crane.a"},
                {"release": "door.south"},
                {"exit": True},
            ],
            "robot-b": [
                {"acquire": "door.south"},
                {"acquire": "crane.a"},
                {"release": "crane.a"},
                {"release": "door.south"},
                {"exit": True},
            ],
        },
    )
    a = kernel.spawn(DescriptorName("robot-a"))
    run_until_holds(kernel, "robot-a", "door.south")
    b = kernel.spawn(DescriptorName("robot-b"))
    kernel.run_until_quiescent()

    assert not of(events, DeadlockDetected)
    assert a.state is JobState.DONE and b.state is JobState.DONE


def test_victim_policy_is_lowest_priority() -> None:
    table = ResourceTable([DOOR, CRANE])
    table.get(ResourceName("door.south")).acquire(JobId(1))
    table.get(ResourceName("crane.a")).acquire(JobId(2))
    table.get(ResourceName("crane.a")).add_waiter(JobId(1))
    table.get(ResourceName("door.south")).add_waiter(JobId(2))

    found = table.find_deadlock(priorities={JobId(1): Priority(10), JobId(2): Priority(99)})
    assert isinstance(found, Deadlock)
    assert found.victim == JobId(2)
    assert "cycle" in found.render()


# --- declaration and load-time checks ---------------------------------------


def test_acquiring_an_undeclared_resource_is_a_capability_fault() -> None:
    kernel, events = build(
        [carrier("sneaky", 50, ["door.south"])],
        {"sneaky": [{"acquire": "crane.a"}, {"exit": True}]},
    )
    kernel.spawn(DescriptorName("sneaky"))
    kernel.run_until_quiescent()

    faults = [f for f in of(events, FaultRaised) if f.fault is FaultKind.CAPABILITY]
    assert faults and "does not declare" in faults[0].detail


def test_declaring_a_resource_that_does_not_exist_is_a_lint_error() -> None:
    d = Descriptor.from_frontmatter({"name": "x", "priority": 50, "resources": ["door.imaginary"]})
    findings = lint({d.name: d}, resources=[DOOR])
    assert [f.rule for f in findings] == ["unknown-resource"]
    assert findings[0].severity is Severity.ERROR


def test_lock_order_disagreement_is_rejected_at_load() -> None:
    """Static deadlock prevention. A cycle knowable from the descriptors should
    never reach runtime -- the runtime detector exists for locks taken by things the
    kernel did not schedule."""
    a = Descriptor.from_frontmatter(
        {"name": "a", "priority": 50, "resources": ["door.south", "crane.a"]}
    )
    b = Descriptor.from_frontmatter(
        {"name": "b", "priority": 50, "resources": ["crane.a", "door.south"]}
    )
    findings = lint({a.name: a, b.name: b}, resources=[DOOR, CRANE])
    inversions = [f for f in findings if f.rule == "lock-order-inversion"]
    assert inversions and inversions[0].severity is Severity.ERROR
    assert "opposite order" in inversions[0].detail


def test_consistent_lock_order_passes() -> None:
    a = Descriptor.from_frontmatter(
        {"name": "a", "priority": 50, "resources": ["door.south", "crane.a"]}
    )
    b = Descriptor.from_frontmatter(
        {"name": "b", "priority": 50, "resources": ["door.south", "crane.a"]}
    )
    findings = lint({a.name: a, b.name: b}, resources=[DOOR, CRANE])
    assert not [f for f in findings if f.rule == "lock-order-inversion"]


def test_lock_order_violations_is_symmetric_and_deterministic() -> None:
    orders = {
        "a": [ResourceName("x"), ResourceName("y")],
        "b": [ResourceName("y"), ResourceName("x")],
    }
    first = lock_order_violations(orders)
    second = lock_order_violations(dict(reversed(list(orders.items()))))
    assert first == second and len(first) == 1


# --- table invariants -------------------------------------------------------


def test_capacity_below_one_is_rejected() -> None:
    with pytest.raises(ResourceError, match="capacity must be >= 1"):
        ResourceSpec(name=ResourceName("bad"), capacity=0)


def test_conflicting_capacity_declarations_are_rejected() -> None:
    table = ResourceTable([DOOR])
    with pytest.raises(ResourceError, match="conflicting capacities"):
        table.declare(ResourceSpec(name=ResourceName("door.south"), capacity=4))


def test_reacquisition_is_a_no_op_not_an_error() -> None:
    table = ResourceTable([DOOR])
    door = table.get(ResourceName("door.south"))
    assert door.acquire(JobId(1))
    assert door.acquire(JobId(1)), "re-acquiring what you hold is fine"
    assert door.holders == [JobId(1)]


def test_unknown_resource_lookup_is_loud() -> None:
    with pytest.raises(ResourceError, match="unknown resource"):
        ResourceTable().get(ResourceName("nope"))
