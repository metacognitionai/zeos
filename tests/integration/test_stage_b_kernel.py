# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Stage B acceptance: a descriptor tree runs end to end, an interrupt preempts
within one token boundary, and the preempted job resumes with a correct dirty diff.

Every assertion here is made against the **journal**, not against transcripts.
That is the point the Programming Model makes about testing: "the alarm preempted supervision within
one boundary" and "supervision resumed with a notice naming plant.unit_a" are
structural facts, whereas anything read out of a transcript is statistical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.events import (
    Event,
    Injected,
    JobCompleted,
    JobDispatched,
    JobPreempted,
    JobResumed,
    JobSpawned,
    PipeBackpressure,
    StateDelta,
    VectorCoalesced,
    VectorFired,
    WorldWritten,
)
from zeos.core.ids import (
    DescriptorName,
    JobState,
    ObjectName,
    PipeName,
    Principal,
    Priority,
    ResumeKind,
    Ring,
    VectorName,
    VectorPolicy,
)
from zeos.core.kernel import Kernel, KernelConfig, render_resume_notice
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

# --- fixtures ---------------------------------------------------------------

SUPERVISION = {
    "name": "production-supervision",
    "priority": 80,
    "reads": ["plant.unit_a"],
    "writes": ["plant.schedule"],
}
SUPERVISION_SCRIPT = [
    {"emit": "reviewing production schedule"},
    {"emit": "checking auxiliary throughput"},
    {"emit": "still reviewing"},
    {"emit": "schedule confirmed"},
    {"exit": True},
]

GAS_ALARM = {
    "name": "threshold-alarm",
    "priority": 5,
    "pinned": True,
    "writes": ["plant.unit_a"],
}
THRESHOLD_ALARM_SCRIPT = [
    {"emit": "threshold alarm acknowledged"},
    {"write": {"pipe": "actuators.unit_a", "text": "max"}},
    {"exit": True},
]


def build(
    *,
    descriptors: Sequence[Mapping[str, Any]],
    scripts: Mapping[str, list[dict[str, Any]]],
    pipes: Sequence[PipeSpec] = (),
    vectors: Sequence[VectorSpec] = (),
    world: Mapping[str, str] | None = None,
    config: KernelConfig | None = None,
) -> tuple[Kernel, list[Event]]:
    events: list[Event] = []
    machine = ScriptedMachine(
        {name: Script.from_spec(steps) for name, steps in scripts.items()}, block_size=8
    )
    table = PipeTable(pipes)
    store = WorldStore()
    for key, value in (world or {}).items():
        from zeos.core.clock import Clock
        from zeos.core.ids import ObjectName

        store.set(ObjectName(key), value, at=Clock())

    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=machine,
        pipes=table,
        vectors=VectorTable(vectors),
        world=store,
        journal_sink=events,
        config=config or KernelConfig(case="stage-b"),
    )
    kernel.start()
    return kernel, events


def of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


# --- end to end -------------------------------------------------------------


def test_a_single_descriptor_runs_to_completion() -> None:
    kernel, events = build(
        descriptors=[SUPERVISION], scripts={"production-supervision": SUPERVISION_SCRIPT}
    )
    job = kernel.spawn(DescriptorName("production-supervision"))
    kernel.run_until_quiescent()

    assert job.state is JobState.DONE
    completed = of(events, JobCompleted)
    assert len(completed) == 1 and completed[0].job == job.job_id
    assert completed[0].tokens_used > 0


def test_a_script_without_exit_is_reported_not_looped_forever() -> None:
    from zeos.machine.scripted import ScriptExhausted

    kernel, _ = build(
        descriptors=[SUPERVISION],
        scripts={"production-supervision": [{"emit": "no exit here"}]},
    )
    kernel.spawn(DescriptorName("production-supervision"))
    with pytest.raises(ScriptExhausted):
        kernel.run_until_quiescent()


# --- preemption -------------------------------------------------------------


def test_interrupt_preempts_within_one_token_boundary() -> None:
    """The core latency claim (core §5.2), stated as a journal property.

    Between the vector firing and the handler being dispatched, the preempted job
    must not decode again. The quantum *is* one token boundary, so this is checkable
    exactly rather than approximately.
    """
    kernel, events = build(
        descriptors=[SUPERVISION, GAS_ALARM],
        scripts={
            "production-supervision": SUPERVISION_SCRIPT,
            "threshold-alarm": THRESHOLD_ALARM_SCRIPT,
        },
        pipes=[
            PipeSpec(
                PipeName("sensors.threshold"),
                ring=Ring.EXTERNAL,
                principal=Principal.DEVICE,
                device=True,
            ),
            PipeSpec(PipeName("actuators.unit_a"), world_object="plant.unit_a"),
        ],
        vectors=[
            VectorSpec(
                name=VectorName("gas"),
                source=PipeName("sensors.threshold"),
                handler=DescriptorName("threshold-alarm"),
                priority=Priority(5),
                policy=VectorPolicy.COALESCE,
            )
        ],
        world={"plant.unit_a": "idle"},
    )
    goal = kernel.spawn(DescriptorName("production-supervision"))

    kernel.tick()  # loads the descriptor body
    kernel.tick()  # one decode
    kernel.advance_time(1_000_000)
    kernel.deliver(PipeName("sensors.threshold"), "level 42")

    fired_at = len(events)
    kernel.tick()  # must preempt here, not decode the goal job

    after = events[fired_at:]
    preempted = of(after, JobPreempted)
    assert len(preempted) == 1, "the alarm must preempt on the very next quantum"
    assert preempted[0].job == goal.job_id
    assert preempted[0].by_priority == Priority(5)

    from zeos.core.events import Decoded

    assert not [d for d in of(after, Decoded) if d.job == goal.job_id], (
        "the preempted job decoded again after the interrupt arrived"
    )

    dispatched = of(after, JobDispatched)
    assert dispatched and dispatched[-1].priority == Priority(5)


def test_equal_priority_does_not_preempt() -> None:
    """Strictness matters: equal-priority preemption would let two jobs ping-pong
    at every boundary and make no progress."""
    peer = {"name": "peer", "priority": 80}
    kernel, events = build(
        descriptors=[SUPERVISION, peer],
        scripts={
            "production-supervision": SUPERVISION_SCRIPT,
            "peer": [{"emit": "peer work"}, {"exit": True}],
        },
    )
    kernel.spawn(DescriptorName("production-supervision"))
    kernel.tick()
    kernel.tick()
    kernel.spawn(DescriptorName("peer"))
    kernel.tick()
    assert not of(events, JobPreempted)


def test_unpreemptible_job_masks_interrupts() -> None:
    """``preemptible: false`` is the cli/sti equivalent (core §5.4)."""
    critical = dict(SUPERVISION, name="critical-section", preemptible=False)
    kernel, events = build(
        descriptors=[critical, GAS_ALARM],
        scripts={
            "critical-section": SUPERVISION_SCRIPT,
            "threshold-alarm": THRESHOLD_ALARM_SCRIPT,
        },
        pipes=[PipeSpec(PipeName("actuators.unit_a"), world_object="plant.unit_a")],
    )
    kernel.spawn(DescriptorName("critical-section"))
    kernel.tick()
    kernel.tick()
    kernel.spawn(DescriptorName("threshold-alarm"), priority=Priority(5))
    kernel.tick()
    assert not of(events, JobPreempted)


# --- resume -----------------------------------------------------------------


def test_resume_carries_a_dirty_diff_naming_what_changed() -> None:
    """The one genuinely new problem (core §6.2): the job's saved state contains
    beliefs, and the world moved while the handler ran."""
    kernel, events = build(
        descriptors=[SUPERVISION, GAS_ALARM],
        scripts={
            "production-supervision": SUPERVISION_SCRIPT,
            "threshold-alarm": THRESHOLD_ALARM_SCRIPT,
        },
        pipes=[
            PipeSpec(
                PipeName("sensors.threshold"),
                ring=Ring.EXTERNAL,
                principal=Principal.DEVICE,
                device=True,
            ),
            PipeSpec(PipeName("actuators.unit_a"), world_object="plant.unit_a"),
        ],
        vectors=[
            VectorSpec(
                name=VectorName("gas"),
                source=PipeName("sensors.threshold"),
                handler=DescriptorName("threshold-alarm"),
                priority=Priority(5),
            )
        ],
        world={"plant.unit_a": "idle"},
    )
    goal = kernel.spawn(DescriptorName("production-supervision"))
    kernel.tick()
    kernel.tick()
    kernel.advance_time(2_000_000_000)
    kernel.deliver(PipeName("sensors.threshold"), "level 42")
    kernel.run_until_quiescent()

    assert of(events, WorldWritten), "the handler must actually change the world"
    resumed = [r for r in of(events, JobResumed) if r.job == goal.job_id]
    assert len(resumed) == 1
    notice = resumed[0]
    assert notice.resume_kind is ResumeKind.DIRTY
    assert [d.obj for d in notice.dirty] == ["plant.unit_a"]
    assert (notice.dirty[0].before, notice.dirty[0].after) == ("idle", "max")
    assert notice.suspended_ns >= 0


def test_resume_is_clean_when_nothing_relevant_changed() -> None:
    """No dirty set means no revalidation cost -- the common case must stay cheap."""
    unrelated = dict(GAS_ALARM, name="unrelated-handler", writes=["weather.temp"])
    kernel, events = build(
        descriptors=[SUPERVISION, unrelated],
        scripts={
            "production-supervision": SUPERVISION_SCRIPT,
            "unrelated-handler": [{"emit": "noted"}, {"exit": True}],
        },
    )
    goal = kernel.spawn(DescriptorName("production-supervision"))
    kernel.tick()
    kernel.tick()
    kernel.spawn(DescriptorName("unrelated-handler"), priority=Priority(5))
    kernel.run_until_quiescent()

    resumed = [r for r in of(events, JobResumed) if r.job == goal.job_id]
    assert len(resumed) == 1
    assert resumed[0].resume_kind is ResumeKind.CLEAN
    assert resumed[0].dirty == ()


def test_resume_notice_text_names_the_object_and_both_values() -> None:
    """The notice is read by the model, so its text is behaviour under test -- a
    bare flag would not be salient enough to override stale in-context state.

    Note the duration rendering departs from the spec's illustrative "Suspended
    94s" (core §6.2): durations roll over into minutes and hours. That matters for
    the partition case, where a suspension can run for hours and "11520s" is
    materially less legible than "3h12m".
    """
    notice = render_resume_notice(
        94_000_000_000,
        [StateDelta(obj=ObjectName("plant.unit_a"), before="idle", after="max")],
    )
    assert "<RESUME>" in notice
    assert "1m34s" in notice
    assert "plant.unit_a: idle -> max" in notice
    assert "Revalidate" in notice


def test_resume_notice_renders_sub_minute_durations_plainly() -> None:
    notice = render_resume_notice(
        42_000_000_000,
        [StateDelta(obj=ObjectName("plant.unit_a"), before="idle", after="max")],
    )
    assert "Suspended 42s" in notice


def test_a_clean_resume_injects_nothing_at_all() -> None:
    """The job's beliefs are exactly as it left them, so it is told nothing.

    A notice saying "nothing changed" is not information, and it is 9 tokens the
    pager can never reclaim -- ``stub_size(9)`` is 11, so eviction is net-negative
    and declined for the life of the job.
    """
    unrelated = dict(GAS_ALARM, name="unrelated-handler", writes=["weather.temp"])
    kernel, events = build(
        descriptors=[SUPERVISION, unrelated],
        scripts={
            "production-supervision": SUPERVISION_SCRIPT,
            "unrelated-handler": [{"emit": "noted"}, {"exit": True}],
        },
    )
    goal = kernel.spawn(DescriptorName("production-supervision"))
    kernel.tick()
    kernel.tick()
    kernel.spawn(DescriptorName("unrelated-handler"), priority=Priority(5))
    kernel.run_until_quiescent()

    resumed = [r for r in of(events, JobResumed) if r.job == goal.job_id]
    assert [r.resume_kind for r in resumed] == [ResumeKind.CLEAN], "journalled either way"
    # Ring 0, not merely the kernel pipe: the descriptor body arrives that way too.
    injected = [e for e in of(events, Injected) if e.job == goal.job_id and e.ring is Ring.KERNEL]
    assert not injected, "a clean resume costs the job nothing"
    assert not [r for r in goal.segments.all() if r.ring is Ring.KERNEL], "no prose in context"


# --- pipes ------------------------------------------------------------------


def test_blocking_read_deschedules_until_a_write_arrives() -> None:
    """Blocking costs nothing: no forward passes while waiting (core §4.1)."""
    waiter = {"name": "waiter", "priority": 50, "pipes": {"stdin": "user.commands"}}
    kernel, events = build(
        descriptors=[waiter],
        scripts={"waiter": [{"read": "stdin"}, {"emit": "got it"}, {"exit": True}]},
        pipes=[PipeSpec(PipeName("user.commands"), principal=Principal.USER)],
    )
    job = kernel.spawn(DescriptorName("waiter"))
    kernel.run_until_quiescent()
    assert job.state is JobState.BLOCKED

    from zeos.core.events import Decoded

    decodes_before = len(of(events, Decoded))
    kernel.run_until_quiescent()
    assert len(of(events, Decoded)) == decodes_before, "a blocked job must not decode"

    kernel.deliver(PipeName("user.commands"), "status report")
    kernel.run_until_quiescent()
    assert job.state is JobState.DONE


def test_backpressure_parks_the_writer_without_losing_the_payload() -> None:
    """All-or-nothing: a partial write would tear the payload and dropping the
    remainder would lose data."""
    producer = {"name": "producer", "priority": 50}
    kernel, events = build(
        descriptors=[producer],
        scripts={
            "producer": [
                {"write": {"pipe": "narrow", "text": "one two three four five"}},
                {"exit": True},
            ]
        },
        pipes=[PipeSpec(PipeName("narrow"), capacity_tokens=2)],
    )
    job = kernel.spawn(DescriptorName("producer"))
    kernel.run_until_quiescent()

    assert of(events, PipeBackpressure), "a full pipe must apply backpressure"
    assert job.state is JobState.BLOCKED
    assert job.pending_write is not None, "the payload must be parked, not dropped"


# --- vectors ----------------------------------------------------------------


def test_coalescing_collapses_a_sensor_storm() -> None:
    """Level-triggered: what matters is the current reading, not how many times it
    changed on the way here (core §5.5)."""
    kernel, events = build(
        descriptors=[GAS_ALARM],
        scripts={"threshold-alarm": THRESHOLD_ALARM_SCRIPT},
        pipes=[
            PipeSpec(PipeName("sensors.threshold"), device=True),
            PipeSpec(PipeName("actuators.unit_a"), world_object="plant.unit_a"),
        ],
        vectors=[
            VectorSpec(
                name=VectorName("gas"),
                source=PipeName("sensors.threshold"),
                handler=DescriptorName("threshold-alarm"),
                priority=Priority(5),
                policy=VectorPolicy.COALESCE,
            )
        ],
    )
    for _ in range(5):
        kernel.deliver(PipeName("sensors.threshold"), "level rising")

    assert len(of(events, VectorFired)) == 1, "a storm must produce one dispatch"
    assert of(events, VectorCoalesced), "collapsed firings must be journalled"
    assert len(of(events, JobSpawned)) == 1


def test_throttled_vector_defers_rather_than_discards() -> None:
    """A throttle that dropped events would silently convert a storm into data loss."""
    kernel, events = build(
        descriptors=[GAS_ALARM],
        scripts={"threshold-alarm": THRESHOLD_ALARM_SCRIPT},
        pipes=[
            PipeSpec(PipeName("sensors.threshold"), device=True),
            PipeSpec(PipeName("actuators.unit_a"), world_object="plant.unit_a"),
        ],
        vectors=[
            VectorSpec(
                name=VectorName("gas"),
                source=PipeName("sensors.threshold"),
                handler=DescriptorName("threshold-alarm"),
                priority=Priority(5),
                policy=VectorPolicy.QUEUE,
                min_interval_ns=1_000_000_000,
            )
        ],
    )
    kernel.advance_time(0)
    kernel.deliver(PipeName("sensors.threshold"), "first")
    kernel.run_until_quiescent()

    kernel.advance_time(1)
    kernel.deliver(PipeName("sensors.threshold"), "second")
    from zeos.core.events import VectorThrottled

    assert of(events, VectorThrottled), "the second firing must be throttled"

    fired_before = len(of(events, VectorFired))
    kernel.advance_time(5_000_000_000)
    kernel.run_until_quiescent()
    assert len(of(events, VectorFired)) > fired_before, (
        "a deferred firing must come back once the interval elapses"
    )
