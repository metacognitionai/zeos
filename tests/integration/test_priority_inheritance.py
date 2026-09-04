# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Priority inheritance, and the lint that refuses unenforced frontmatter.

Both of these exist because the kernel was previously claiming more than it did.

`Scheduler.inherit_priority`, `restore_priority`, and the `PriorityInherited`
event were all defined and **never called** -- there was no resource for them to
trigger on, so core §2.2's named guarantee was scaffolding. Every test passed.
These tests assert the mechanism actually fires, which is the only kind of test
that would have caught it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import Event, PriorityInherited, PriorityRestored
from zeos.core.ids import DescriptorName, JobState, PipeName, Principal, Priority, Ring
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.lint import Severity, lint
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

PIPES = [
    PipeSpec(PipeName("work.queue"), ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
    PipeSpec(
        PipeName("narrow"), ring=Ring.TRUSTED, principal=Principal.PEER_JOB, capacity_tokens=2
    ),
]


def build(
    descriptors: Sequence[Mapping[str, Any]],
    scripts: Mapping[str, list[dict[str, Any]]],
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in scripts.items()}, block_size=8),
        pipes=PipeTable(PIPES),
        vectors=VectorTable(),
        world=WorldStore(),
        journal_sink=events,
        config=KernelConfig(case="inheritance"),
    )
    kernel.start()
    return kernel, events


def of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


# --- the classic inversion --------------------------------------------------

#: Urgent consumer. Blocks reading a pipe only the slow producer can fill.
URGENT = {
    "name": "urgent-consumer",
    "priority": 10,
    "pipes": {"stdin": "work.queue"},
}
#: Low-priority producer, holding explicit write authority for that pipe.
PRODUCER = {
    "name": "slow-producer",
    "priority": 90,
    "capabilities": [{"pipe": "work.queue", "min_integrity": 3}],
}


def test_blocking_reader_donates_its_priority_to_the_producer() -> None:
    """Core §2.2. Without this the urgent job's deadline is bounded by the
    *producer's* priority rather than its own."""
    kernel, events = build(
        [URGENT, PRODUCER],
        {
            "urgent-consumer": [{"read": "work.queue"}, {"emit": "got it"}, {"exit": True}],
            "slow-producer": [
                {"emit": "working"},
                {"write": {"pipe": "work.queue", "text": "payload"}},
                {"exit": True},
            ],
        },
    )
    urgent = kernel.spawn(DescriptorName("urgent-consumer"))
    producer = kernel.spawn(DescriptorName("slow-producer"))
    kernel.run_until_quiescent()

    inherited = of(events, PriorityInherited)
    assert inherited, "the producer must inherit the blocked reader's priority"
    donation = inherited[0]
    assert donation.job == producer.job_id
    assert donation.blocked_job == urgent.job_id
    assert donation.from_priority == Priority(90)
    assert donation.to_priority == Priority(10)
    assert donation.resource == "work.queue"


def test_inherited_priority_is_returned_once_the_waiter_wakes() -> None:
    """Inheritance is temporary by definition. A holder that kept the borrowed
    priority would quietly become as urgent as the most urgent thing that ever
    waited on it."""
    kernel, events = build(
        [URGENT, PRODUCER],
        {
            "urgent-consumer": [{"read": "work.queue"}, {"emit": "got it"}, {"exit": True}],
            "slow-producer": [
                {"emit": "working"},
                {"write": {"pipe": "work.queue", "text": "payload"}},
                {"emit": "carrying on"},
                {"exit": True},
            ],
        },
    )
    kernel.spawn(DescriptorName("urgent-consumer"))
    producer = kernel.spawn(DescriptorName("slow-producer"))
    kernel.run_until_quiescent()

    restored = of(events, PriorityRestored)
    assert restored, "the donated priority must be returned"
    assert restored[0].job == producer.job_id
    assert restored[0].to_priority == Priority(90)
    assert producer.current_priority == Priority(90)
    assert producer.inherited_from is None


def test_backpressured_writer_donates_to_the_consumer() -> None:
    """The mirror case: an urgent job blocked writing a full pipe lends its
    priority to whoever drains it."""
    urgent_writer = {
        "name": "urgent-writer",
        "priority": 10,
        "capabilities": [{"pipe": "narrow", "min_integrity": 3}],
    }
    drainer = {"name": "slow-drainer", "priority": 90, "pipes": {"stdin": "narrow"}}

    kernel, events = build(
        [urgent_writer, drainer],
        {
            "urgent-writer": [
                {"write": {"pipe": "narrow", "text": "one two three four five"}},
                {"exit": True},
            ],
            "slow-drainer": [{"emit": "idling"}, {"read": "narrow"}, {"exit": True}],
        },
    )
    kernel.spawn(DescriptorName("urgent-writer"))
    drain = kernel.spawn(DescriptorName("slow-drainer"))
    kernel.run_until_quiescent()

    inherited = [e for e in of(events, PriorityInherited) if e.job == drain.job_id]
    assert inherited, "the consumer must inherit the backpressured writer's priority"
    assert inherited[0].resource == "narrow"


def test_no_donation_when_the_holder_is_already_more_urgent() -> None:
    """Inheritance raises priority; it must never lower one."""
    fast_producer = dict(PRODUCER, name="fast-producer", priority=1)
    kernel, events = build(
        [URGENT, fast_producer],
        {
            "urgent-consumer": [{"read": "work.queue"}, {"exit": True}],
            "fast-producer": [
                {"write": {"pipe": "work.queue", "text": "payload"}},
                {"exit": True},
            ],
        },
    )
    kernel.spawn(DescriptorName("urgent-consumer"))
    fast = kernel.spawn(DescriptorName("fast-producer"))
    kernel.run_until_quiescent()

    assert not [e for e in of(events, PriorityInherited) if e.job == fast.job_id]
    assert fast.current_priority == Priority(1)


def test_donation_only_follows_declared_relationships() -> None:
    """Donating to the wrong job is not a harmless over-approximation -- it is
    priority inflation. A job with no declared write authority for the pipe gets
    nothing, even though it happens to be low priority and runnable."""
    bystander = {"name": "bystander", "priority": 95}
    kernel, events = build(
        [URGENT, bystander],
        {
            "urgent-consumer": [{"read": "work.queue"}, {"exit": True}],
            "bystander": [{"emit": "unrelated work"}, {"exit": True}],
        },
    )
    kernel.spawn(DescriptorName("urgent-consumer"))
    other = kernel.spawn(DescriptorName("bystander"))
    kernel.run_until_quiescent()

    assert not [e for e in of(events, PriorityInherited) if e.job == other.job_id]
    assert other.current_priority == Priority(95)


def test_the_urgent_job_actually_completes() -> None:
    """The end the mechanism exists for: the inversion is resolved and the urgent
    job is not left waiting behind a job it outranks."""
    kernel, _ = build(
        [URGENT, PRODUCER],
        {
            "urgent-consumer": [{"read": "work.queue"}, {"emit": "done"}, {"exit": True}],
            "slow-producer": [
                {"emit": "working"},
                {"write": {"pipe": "work.queue", "text": "payload"}},
                {"exit": True},
            ],
        },
    )
    urgent = kernel.spawn(DescriptorName("urgent-consumer"))
    kernel.spawn(DescriptorName("slow-producer"))
    kernel.run_until_quiescent()
    assert urgent.state is JobState.DONE


# --- the frontmatter lint ---------------------------------------------------


def descriptor(**overrides: Any) -> Descriptor:
    spec: dict[str, Any] = {"name": "d", "priority": 50}
    spec.update(overrides)
    return Descriptor.from_frontmatter(spec)


def test_mesh_frontmatter_is_rejected_rather_than_silently_ignored() -> None:
    """The rule that keeps the phase boundary honest, on a key that is still
    outstanding.

    ``gang``, ``requires`` and ``release_policy`` used to be here; F0 implemented
    them, so they moved out of ``UNIMPLEMENTED_KEYS``. ``mesh`` remains, because
    platform-to-platform links have no transport until F2 -- and a descriptor
    declaring one would be asserting a link that does not exist.
    """
    d = descriptor(mesh=["carrier-5"])
    findings = [f for f in lint({d.name: d}) if f.rule == "unimplemented-frontmatter"]
    assert findings and findings[0].severity is Severity.ERROR
    assert "mesh" in findings[0].detail


def test_every_unimplemented_key_is_an_error() -> None:
    from zeos.descriptor.lint import UNIMPLEMENTED_KEYS

    for key in UNIMPLEMENTED_KEYS:
        d = descriptor(**{key: "anything"})
        rules = {f.rule for f in lint({d.name: d})}
        assert "unimplemented-frontmatter" in rules, f"{key} was accepted silently"


def test_unrecognised_keys_warn_but_do_not_block() -> None:
    d = descriptor(some_future_idea={"a": 1})
    findings = [f for f in lint({d.name: d}) if f.rule == "unrecognised-frontmatter"]
    assert findings and findings[0].severity is Severity.WARNING


def test_m0_scaffolding_is_not_flagged() -> None:
    """``script:`` is deliberate M0 scaffolding, not an unimplemented promise."""
    d = descriptor(script=[{"exit": True}])
    assert not [
        f
        for f in lint({d.name: d})
        if f.rule in ("unimplemented-frontmatter", "unrecognised-frontmatter")
    ]


def test_fleet_frontmatter_is_now_interpreted_not_parked() -> None:
    """The other side of the same rule: once a mechanism exists, its key must stop
    being rejected *and* stop landing in ``extra``.

    This is the machine-checkable phase boundary. If F0 had implemented the
    behaviour but left the keys in ``UNIMPLEMENTED_KEYS``, or parsed them into
    ``extra``, this test would say so.
    """
    from zeos.core.allocator import ReleasePolicy

    d = descriptor(
        name="carry-lead",
        requires={"tooling": ["gripper-std"], "locomotion": "wheeled"},
        gang={"members": ["carry-lead"], "coupling": "loose"},
        release_policy="at-action-boundary",
    )
    assert not d.extra, "implemented keys must not be parked in extra"
    assert d.requires.tooling == frozenset({"gripper-std"})
    assert d.release_policy == ReleasePolicy.ACTION_BOUNDARY
    assert d.gang is not None and d.gang.coupling == "loose"

    unimplemented = [f for f in lint({d.name: d}) if f.rule == "unimplemented-frontmatter"]
    assert not unimplemented, [f.detail for f in unimplemented]
