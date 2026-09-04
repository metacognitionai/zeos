# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""D1 -- two kernels, one process.

ZEOS-Distributed's D1 validates three things, and the reason to build it before a real
transport is stated in the spec itself:

> …with no networking involved -- **and, being deterministic, is replayable, which a
> real network is not.**

A real network reproduces a partition bug approximately. This reproduces it exactly,
and the difference is whether "the reconnection resumed with the wrong dirty set" is a
regression test or an anecdote.

The three properties: preemption does not cross the link,
replicas are stale and say so, and a partition is a suspension and a
reconnection is a resume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeos.core.events import (
    Event,
    FaultRaised,
    FrameDelivered,
    FrameSent,
    JobPreempted,
    LinkStateChanged,
    ReplicaRefreshed,
    StalenessRefused,
)
from zeos.core.ids import (
    DescriptorName,
    JobState,
    ObjectName,
    PipeName,
    Principal,
    Priority,
    Ring,
    VectorName,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.topology import LINK_STATE, LinkSpec, NodeSpec, Topology
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.federated import Federation, Node
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.transport.link import LinkTransport, LossModel
from zeos.world.store import ObjectSet, WorldStore

MS = 1_000_000

#: The pipe that crosses. Everything else is node-local.
UPLINK = PipeName("uplink.events")
DOWNLINK = PipeName("downlink.commands")

PIPES = [
    PipeSpec(UPLINK, ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
    PipeSpec(DOWNLINK, ring=Ring.TRUSTED, principal=Principal.PEER_JOB),
    PipeSpec(PipeName("sensors.local"), ring=Ring.EXTERNAL, principal=Principal.DEVICE),
]

TOPOLOGY = Topology(
    [
        NodeSpec("platform", "small-local", ObjectSet.of(["robot.*", "sensors.*"])),
        NodeSpec("offboard", "large", ObjectSet.of(["mission.*"])),
    ],
    #: 120 ms round trip. Chosen because the distribution design's worked topology uses it,
    #: and because it is comfortably above a pinned handler's tens-of-milliseconds
    #: budget -- which is the whole reason the placement rule exists.
    LinkSpec(rtt_p99_ns=120 * MS, ring=Ring.TRUSTED),
)


def kernel_for(
    node: str,
    descriptors: Sequence[Mapping[str, Any]],
    scripts: Mapping[str, list[dict[str, Any]]],
    *,
    link: LinkTransport,
    events: list[Event],
    vectors: Sequence[VectorSpec] = (),
    world: Mapping[str, str] = {},
) -> Kernel:
    store = WorldStore()
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in descriptors
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in scripts.items()}, block_size=8),
        pipes=PipeTable(PIPES),
        vectors=VectorTable(vectors),
        world=store,
        node=node,
        topology=TOPOLOGY,
        link=link,
        journal_sink=events,
        config=KernelConfig(case=f"d1-{node}"),
    )
    for obj, value in sorted(world.items()):
        store.set(ObjectName(obj), value, at=kernel.clock)
    kernel.start()
    return kernel


def federation(
    *,
    platform: tuple[Sequence[Mapping[str, Any]], Mapping[str, list[dict[str, Any]]]],
    offboard: tuple[Sequence[Mapping[str, Any]], Mapping[str, list[dict[str, Any]]]],
    loss: LossModel | None = None,
    carried: Sequence[PipeName] = (UPLINK, DOWNLINK),
    platform_world: Mapping[str, str] = {},
    platform_vectors: Sequence[VectorSpec] = (),
    offboard_vectors: Sequence[VectorSpec] = (),
) -> tuple[Federation, list[Event], list[Event]]:
    up: list[Event] = []
    down: list[Event] = []
    to_offboard = LinkTransport(TOPOLOGY.link, pipes=frozenset(carried), loss=loss or LossModel())
    to_platform = LinkTransport(TOPOLOGY.link, pipes=frozenset(carried), loss=loss or LossModel())
    fed = Federation(topology=TOPOLOGY)
    fed.add(
        Node(
            "platform",
            kernel_for(
                "platform",
                platform[0],
                platform[1],
                link=to_offboard,
                events=up,
                vectors=platform_vectors,
                world=platform_world,
            ),
            to_offboard,
        )
    )
    fed.add(
        Node(
            "offboard",
            kernel_for(
                "offboard",
                offboard[0],
                offboard[1],
                link=to_platform,
                events=down,
                vectors=offboard_vectors,
            ),
            to_platform,
        )
    )
    return fed, up, down


def of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


REPORTER = {
    "name": "reporter",
    "priority": 50,
    "capabilities": [{"pipe": str(UPLINK), "min_integrity": 3}],
}
GOAL = {"name": "goal", "priority": 80, "pipes": {"stdin": str(UPLINK)}}


# --- the scheduling domain is the node ---------------------------------------


def test_the_two_kernels_have_separate_priority_spaces() -> None:
    """Each node keeps its own scheduler, ready queue and suspension stack.

    Priorities are not globally comparable and no attempt is made to make them so.
    """
    fed, _, _ = federation(
        platform=([REPORTER], {"reporter": [{"emit": "a"}, {"exit": True}]}),
        offboard=([GOAL], {"goal": [{"read": str(UPLINK)}, {"exit": True}]}),
    )
    assert fed.node("platform").kernel is not fed.node("offboard").kernel
    assert fed.node("platform").kernel.sched is not fed.node("offboard").kernel.sched


def test_a_write_crosses_the_link_and_arrives_an_rtt_later() -> None:
    """The seam. A job cannot observe which transport carries its pipe, so this is a
    latency change and not a semantic one."""
    fed, up, down = federation(
        platform=(
            [REPORTER],
            {"reporter": [{"write": {"pipe": str(UPLINK), "text": "gas high"}}, {"exit": True}]},
        ),
        offboard=([GOAL], {"goal": [{"read": str(UPLINK)}, {"emit": "seen"}, {"exit": True}]}),
    )
    fed.node("platform").kernel.spawn(DescriptorName("reporter"))
    goal = fed.node("offboard").kernel.spawn(DescriptorName("goal"))
    fed.run()

    sent = of(up, FrameSent)
    delivered = of(down, FrameDelivered)
    assert sent and delivered, "the write must leave one node and arrive at the other"
    assert delivered[0].pipe == UPLINK
    # One way is half the declared round trip.
    assert delivered[0].latency_ns >= TOPOLOGY.link.one_way_ns
    assert goal.state is JobState.DONE


def test_the_far_job_blocks_at_zero_cost_while_the_frame_is_in_flight() -> None:
    """An off-platform job whose pipes went quiet is descheduled -- the existing
    blocking semantics doing exactly what they were designed to do."""
    fed, _, down = federation(
        platform=(
            [REPORTER],
            {
                "reporter": [
                    {"emit": "thinking"},
                    {"emit": "still"},
                    {"write": {"pipe": str(UPLINK), "text": "x"}},
                    {"exit": True},
                ]
            },
        ),
        offboard=([GOAL], {"goal": [{"read": str(UPLINK)}, {"exit": True}]}),
    )
    fed.node("platform").kernel.spawn(DescriptorName("reporter"))
    goal = fed.node("offboard").kernel.spawn(DescriptorName("goal"))
    for _ in range(3):
        fed.step()
    assert goal.state is JobState.BLOCKED
    decodes = [e for e in down if type(e).KIND == "machine.decode"]
    assert not decodes, "a blocked job must consume no forward passes"


def test_preemption_does_not_cross_the_link() -> None:
    """The sharpest conclusion in ZEOS-Distributed.

    A high-priority on-platform handler cannot directly preempt an off-platform goal
    job. It writes to a pipe; the far kernel preempts *on receipt*, using its own
    priorities. The extra RTT is acceptable by construction, because a job that could
    not tolerate it would have failed the placement rule and been placed locally.
    """
    urgent = {
        "name": "urgent",
        "priority": 1,
        "capabilities": [{"pipe": str(UPLINK), "min_integrity": 3}],
    }
    handler = {"name": "handler", "priority": 5, "pipes": {"stdin": str(UPLINK)}}
    slow_goal = {"name": "goal", "priority": 80}

    fed, up, down = federation(
        platform=(
            [urgent],
            {"urgent": [{"write": {"pipe": str(UPLINK), "text": "alarm"}}, {"exit": True}]},
        ),
        offboard=(
            [slow_goal, handler],
            # Long enough to still be running when the frame lands. One way is 60 ms
            # and a step is 1 ms, so a goal job of five tokens would have finished
            # before the alarm ever arrived -- which is itself the point being made:
            # the RTT is real and the far kernel acts on receipt, not on send.
            {
                "goal": [{"emit": f"plan-{i}"} for i in range(120)] + [{"exit": True}],
                "handler": [{"read": str(UPLINK)}, {"emit": "responding"}, {"exit": True}],
            },
        ),
        offboard_vectors=[
            VectorSpec(
                name=VectorName("alarm"),
                source=UPLINK,
                handler=DescriptorName("handler"),
                priority=Priority(5),
            )
        ],
    )
    fed.node("offboard").kernel.spawn(DescriptorName("goal"))
    fed.node("platform").kernel.spawn(DescriptorName("urgent"))
    fed.run()

    sent = of(up, FrameSent)
    preempted = of(down, JobPreempted)
    assert sent, "the platform wrote to the link"
    assert preempted, "the far kernel preempted its own goal job on receipt"
    # The preemption happened at the far node, after the frame landed -- never before.
    delivered = of(down, FrameDelivered)
    assert delivered
    assert preempted[0].clock.virtual_ns >= delivered[0].clock.virtual_ns - fed.ns_per_step
    assert not of(up, JobPreempted), "nothing on the platform was preempted by the peer"


# --- replicas are stale, and say so ------------------------------------------


def test_authority_follows_the_sensors() -> None:
    assert TOPOLOGY.authority_for(ObjectName("robot.position")) == "platform"
    assert TOPOLOGY.authority_for(ObjectName("mission.goal")) == "offboard"
    assert TOPOLOGY.is_replica("offboard", ObjectName("robot.position"))
    assert not TOPOLOGY.is_replica("platform", ObjectName("robot.position"))


def test_a_replica_carries_an_age_and_an_owned_object_does_not() -> None:
    """An object you own has a value; an object you replicate has a value *and* an
    age. Conflating them is how a plan ends up safe against state nobody told it was
    old."""
    fed, _, down = federation(
        platform=([REPORTER], {"reporter": [{"exit": True}]}),
        offboard=([GOAL], {"goal": [{"exit": True}]}),
        platform_world={"robot.position": "bench-3"},
    )
    fed.step()
    fed.replicate(ObjectName("robot.position"))
    offboard = fed.node("offboard").kernel

    assert of(down, ReplicaRefreshed)
    for _ in range(400):
        fed.step()
    age = offboard.world.age_ns(ObjectName("robot.position"), now=offboard.clock)
    assert age is not None and age > 0, "a replica ages"
    assert offboard.world.age_ns(ObjectName("mission.goal"), now=offboard.clock) is None


def test_a_read_beyond_max_staleness_faults_rather_than_returning_it() -> None:
    """Fail loudly, per the standing rule. Quietly serving state a job declared it
    could not use defeats the point of the declaration."""
    picky = {
        "name": "picky",
        "priority": 50,
        "reads": ["robot.position"],
        "max_staleness": "50ms",
    }
    fed, _, down = federation(
        platform=([REPORTER], {"reporter": [{"exit": True}]}),
        offboard=([picky], {"picky": [{"emit": "planning"}, {"emit": "more"}, {"exit": True}]}),
        platform_world={"robot.position": "bench-3"},
    )
    fed.step()
    fed.replicate(ObjectName("robot.position"))
    # Let the replica go stale *before* the job starts. This is the realistic shape:
    # a job dispatched during a quiet spell finds the state it depends on is older
    # than it declared it could use.
    for _ in range(120):
        fed.step()
    fed.node("offboard").kernel.spawn(DescriptorName("picky"))
    for _ in range(10):
        fed.step()

    refused = of(down, StalenessRefused)
    assert refused, "a replica older than the declared tolerance must be refused"
    assert refused[0].obj == ObjectName("robot.position")
    assert refused[0].age_ns > refused[0].max_staleness_ns
    assert [f for f in of(down, FaultRaised) if "max_staleness" in f.detail]


def test_a_fresh_replica_is_used_without_complaint() -> None:
    """The pair that makes the previous test mean something."""
    picky = {"name": "picky", "priority": 50, "reads": ["robot.position"], "max_staleness": "5s"}
    fed, _, down = federation(
        platform=([REPORTER], {"reporter": [{"exit": True}]}),
        offboard=([picky], {"picky": [{"emit": "planning"}, {"exit": True}]}),
        platform_world={"robot.position": "bench-3"},
    )
    fed.step()
    fed.replicate(ObjectName("robot.position"))
    job = fed.node("offboard").kernel.spawn(DescriptorName("picky"))
    fed.run()
    assert not of(down, StalenessRefused)
    assert job.state is JobState.DONE


# --- a partition is a suspension ---------------------------------------------


def test_a_partition_is_a_world_state_change_with_a_handler() -> None:
    """No new mechanism. "The link went down" is an ordinary world write, so a vector
    can fire on it and an ordinary descriptor can respond."""
    fed, up, _ = federation(
        platform=([REPORTER], {"reporter": [{"exit": True}]}),
        offboard=([GOAL], {"goal": [{"exit": True}]}),
    )
    fed.step()
    fed.partition(reason="radio lost")

    changed = of(up, LinkStateChanged)
    assert changed and not changed[0].up and changed[0].reason == "radio lost"
    assert fed.node("platform").kernel.world.get(LINK_STATE) == "down"
    assert fed.node("offboard").kernel.world.get(LINK_STATE) == "down"


def test_writes_during_a_partition_are_dropped_not_queued() -> None:
    """A link that buffered indefinitely and flushed on reconnection would deliver a
    burst of stale instructions to a robot that has moved on."""
    fed, up, down = federation(
        platform=(
            [REPORTER],
            {"reporter": [{"write": {"pipe": str(UPLINK), "text": "late"}}, {"exit": True}]},
        ),
        offboard=([GOAL], {"goal": [{"read": str(UPLINK)}, {"exit": True}]}),
    )
    fed.partition()
    fed.node("platform").kernel.spawn(DescriptorName("reporter"))
    goal = fed.node("offboard").kernel.spawn(DescriptorName("goal"))
    for _ in range(30):
        fed.step()

    assert of(up, FrameSent), "the writer was not told; it wrote normally"
    assert not of(down, FrameDelivered), "nothing arrived"
    assert goal.state is JobState.BLOCKED, "the far job's pipes simply went quiet"


def test_the_writer_is_not_faulted_by_a_partition() -> None:
    """The design puts the consequence at the *reader*, as a job whose pipes went quiet.
    Faulting the writer would move the failure to the wrong place and lose the
    property that a partition is a suspension."""
    fed, up, _ = federation(
        platform=(
            [REPORTER],
            {
                "reporter": [
                    {"write": {"pipe": str(UPLINK), "text": "x"}},
                    {"emit": "carried on"},
                    {"exit": True},
                ]
            },
        ),
        offboard=([GOAL], {"goal": [{"exit": True}]}),
    )
    fed.partition()
    job = fed.node("platform").kernel.spawn(DescriptorName("reporter"))
    fed.run()
    assert job.state is JobState.DONE
    assert not of(up, FaultRaised)


def test_in_flight_frames_are_lost_when_the_wire_fails() -> None:
    """They were on the wire when the wire failed. Holding them models a link that
    pauses and resumes, which is a friendlier failure than the one the safety case
    has to survive."""
    fed, _, down = federation(
        platform=(
            [REPORTER],
            {"reporter": [{"write": {"pipe": str(UPLINK), "text": "x"}}, {"exit": True}]},
        ),
        offboard=([GOAL], {"goal": [{"read": str(UPLINK)}, {"exit": True}]}),
    )
    fed.node("platform").kernel.spawn(DescriptorName("reporter"))
    fed.step()
    fed.step()
    assert fed.node("platform").outbound.in_flight >= 1
    lost = fed.partition()
    assert lost >= 1
    for _ in range(30):
        fed.step()
    assert not of(down, FrameDelivered)


def test_a_reconnection_is_a_resume() -> None:
    """The slogan, as a test: *a partition is a suspension; a reconnection is a
    resume.* Restoring the link is an ordinary world write, and the platform keeps
    scheduling its own jobs throughout -- deliberation is what the link carried away,
    not safety."""
    local = {"name": "local-safety", "priority": 5}
    fed, up, _ = federation(
        platform=([local], {"local-safety": [{"emit": "holding safe"}, {"exit": True}]}),
        offboard=([GOAL], {"goal": [{"exit": True}]}),
    )
    fed.partition()
    job = fed.node("platform").kernel.spawn(DescriptorName("local-safety"))
    for _ in range(20):
        fed.step()
    assert job.state is JobState.DONE, "the platform stays safe standalone"

    fed.restore()
    changed = of(up, LinkStateChanged)
    assert len(changed) == 2 and changed[-1].up
    assert fed.node("platform").kernel.world.get(LINK_STATE) == "up"


# --- injected loss, deterministically ---------------------------------------


def test_loss_is_an_explicit_schedule_not_a_random_draw() -> None:
    """A seeded RNG would be reproducible only until someone added a frame upstream.
    An explicit schedule says exactly which delivery is lost, so the assertion reads
    as the scenario rather than as a consequence of a seed."""
    fed, up, down = federation(
        platform=(
            [REPORTER],
            {
                "reporter": [
                    {"write": {"pipe": str(UPLINK), "text": "one"}},
                    {"write": {"pipe": str(UPLINK), "text": "two"}},
                    {"exit": True},
                ]
            },
        ),
        offboard=([GOAL], {"goal": [{"read": str(UPLINK)}, {"exit": True}]}),
        loss=LossModel(drop_seqs=frozenset({1})),
    )
    fed.node("platform").kernel.spawn(DescriptorName("reporter"))
    fed.node("offboard").kernel.spawn(DescriptorName("goal"))
    fed.run()

    sent, delivered = of(up, FrameSent), of(down, FrameDelivered)
    assert len(sent) == 2, "both writes were attempted"
    assert len(delivered) == 1, "frame 1 was eaten; frame 2 arrived"
    assert delivered[0].frame == 2


def test_the_federation_is_deterministic() -> None:
    """The reason to build D1 before a real transport: same inputs, same journals."""

    def once() -> list[str]:
        fed, up, down = federation(
            platform=(
                [REPORTER],
                {"reporter": [{"write": {"pipe": str(UPLINK), "text": "x"}}, {"exit": True}]},
            ),
            offboard=([GOAL], {"goal": [{"read": str(UPLINK)}, {"emit": "ok"}, {"exit": True}]}),
        )
        fed.node("platform").kernel.spawn(DescriptorName("reporter"))
        fed.node("offboard").kernel.spawn(DescriptorName("goal"))
        fed.run()
        return [type(e).KIND for e in up] + ["|"] + [type(e).KIND for e in down]

    assert once() == once()
