# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""F0 -- the allocator, leases, and re-embodiment.

ZEOS-Fleet's slogan is the thing under test:

> **A robot is a body a job wears; changing bodies is a resume.**

The scenario is the Fleet design's beam example, reduced to what one kernel with mocked
bodies can show: carriers that need a gripper, a sweeper holding a doorway, a
battery fault mid-carry, and a spare to re-embody into.

Two structural notes carried forward from R0. First, **contention needs
preemption** -- with one running job a lower-priority claimant never runs while the
holder holds -- so the tests stage the holder and then introduce the urgent job.
Second, **a lease is an ordinary resource**, which is why waiting for a body
produces the same blocking, inheritance, and cycle detection as waiting for a
crane, with none of it written twice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from zeos.core.allocator import GreedyAllocator, ReleasePolicy
from zeos.core.embodiment import (
    EmbodimentRequirements,
    PlatformProfile,
    feasible_platforms,
    lease_name,
    match,
    unsatisfiable,
)
from zeos.core.events import (
    AllocationRefused,
    Disembodied,
    Embodied,
    Event,
    FaultRaised,
    GangAssembled,
    GangDissolved,
    JobResumed,
    PlatformJoined,
    PlatformWithdrawn,
    PriorityInherited,
    PriorityRestored,
    ResourceBlocked,
)
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    JobState,
    ObjectName,
    ResumeKind,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.resources import ResourceTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.lint import Severity, lint
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

CARRIER_7 = PlatformProfile(
    name="carrier-7",
    locomotion="wheeled",
    tooling=frozenset({"gripper-std", "lift-500kg"}),
    sensors=frozenset({"lidar", "smoke"}),
    battery=0.95,
    location="bay-3",
)
CARRIER_12 = PlatformProfile(
    name="carrier-12",
    locomotion="wheeled",
    tooling=frozenset({"gripper-std"}),
    sensors=frozenset({"lidar"}),
    battery=0.60,
    location="dock-2",
)
SWEEPER_2 = PlatformProfile(
    name="sweeper-2",
    locomotion="tracked",
    tooling=frozenset({"brush"}),
    battery=0.80,
    location="yard-east",
)
FLEET = (CARRIER_7, CARRIER_12, SWEEPER_2)


def build(
    descriptors: Sequence[Mapping[str, Any]],
    scripts: Mapping[str, list[dict[str, Any]]],
    platforms: Sequence[PlatformProfile] = FLEET,
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
        resources=ResourceTable(),
        platforms=platforms,
        journal_sink=events,
        config=KernelConfig(case="f0"),
    )
    kernel.start()
    return kernel, events


def of[E: Event](events: Sequence[Event], cls: type[E]) -> list[E]:
    return [e for e in events if isinstance(e, cls)]


CARRY = {
    "name": "carry-beam",
    "priority": 40,
    "requires": {"tooling": ["gripper-std"], "locomotion": "wheeled"},
    "release_policy": "at-action-boundary",
}
WORK = [{"emit": "lifting"}, {"emit": "moving"}, {"emit": "placed"}, {"exit": True}]


# --- matching: feasibility versus cost --------------------------------------


def test_a_body_without_the_tooling_is_not_a_choice() -> None:
    """Feasibility is a hard fact. Collapsing it into cost would let a sufficiently
    attractive score outvote a missing gripper."""
    needs = EmbodimentRequirements.parse({"tooling": ["gripper-std"]})
    result = match(needs, SWEEPER_2)
    assert not result.feasible
    assert "lacks tooling ['gripper-std']" in result.reasons


def test_locomotion_and_battery_are_feasibility() -> None:
    needs = EmbodimentRequirements.parse({"locomotion": "wheeled", "battery_min": "80%"})
    assert match(needs, CARRIER_7).feasible
    assert not match(needs, CARRIER_12).feasible  # 60% battery
    assert not match(needs, SWEEPER_2).feasible  # tracked


def test_near_is_cost_not_feasibility() -> None:
    """Fleet marks ``near:`` as "soft: feeds allocation cost, not feasibility". A
    distant body that can do the job is a legal answer, just a worse one."""
    needs = EmbodimentRequirements.parse({"tooling": ["gripper-std"], "near": "bay-3"})
    here, there = match(needs, CARRIER_7), match(needs, CARRIER_12)
    assert here.feasible and there.feasible
    assert here.cost < there.cost

    ranked = feasible_platforms(needs, FLEET)
    assert [r.platform for r in ranked] == ["carrier-7", "carrier-12"]


def test_unsatisfiable_explains_every_body() -> None:
    why = unsatisfiable(EmbodimentRequirements.parse({"tooling": ["welder"]}), FLEET)
    assert len(why) == 3 and all("lacks tooling" in w for w in why)


# --- leases -----------------------------------------------------------------


def test_a_job_that_needs_a_body_gets_one() -> None:
    kernel, events = build([CARRY], {"carry-beam": WORK})
    job = kernel.spawn(DescriptorName("carry-beam"))
    kernel.run_until_quiescent()

    embodied = of(events, Embodied)
    assert embodied and embodied[0].job == job.job_id
    assert embodied[0].platform == "carrier-7"  # cheapest feasible
    assert embodied[0].release_policy == ReleasePolicy.ACTION_BOUNDARY
    assert job.state is JobState.DONE


def test_a_deliberative_job_needs_no_body() -> None:
    """Most behaviours never touch flesh, and must not queue for it."""
    planner = {"name": "planner", "priority": 80}
    kernel, events = build([planner], {"planner": WORK})
    job = kernel.spawn(DescriptorName("planner"))
    kernel.run_until_quiescent()
    assert job.state is JobState.DONE
    assert not of(events, Embodied)


def test_the_lease_is_an_ordinary_resource() -> None:
    """The whole reason R0 came first."""
    kernel, _ = build([CARRY], {"carry-beam": [{"emit": "x"}, {"emit": "y"}]})
    job = kernel.spawn(DescriptorName("carry-beam"))
    kernel.tick()
    kernel.tick()
    name = lease_name("carrier-7")
    assert kernel.resources.has(name)
    assert kernel.resources.get(name).held_by(job.job_id)
    assert name in job.held_resources


def test_a_second_claimant_blocks_on_the_body() -> None:
    """And blocking on a body is blocking on a resource, so inheritance applies."""
    slow = dict(CARRY, name="slow-carry", priority=90)
    urgent = dict(CARRY, name="urgent-carry", priority=10)
    kernel, events = build(
        [slow, urgent],
        {
            "slow-carry": [{"emit": "a"}, {"emit": "b"}, {"emit": "c"}, {"exit": True}],
            "urgent-carry": WORK,
        },
        platforms=(CARRIER_7,),  # one body, two claimants
    )
    holder = kernel.spawn(DescriptorName("slow-carry"))
    kernel.tick()
    kernel.tick()
    kernel.spawn(DescriptorName("urgent-carry"))
    kernel.run_until_quiescent()

    assert of(events, ResourceBlocked), "the urgent job must wait for the body"
    donated = [e for e in of(events, PriorityInherited) if e.job == holder.job_id]
    assert donated, "the holder must inherit the waiting job's urgency"
    assert "body:carrier-7" in donated[0].resource


def test_completion_returns_the_body() -> None:
    kernel, events = build([CARRY], {"carry-beam": WORK})
    kernel.spawn(DescriptorName("carry-beam"))
    kernel.run_until_quiescent()
    released = [e for e in of(events, Disembodied) if e.reason == "job completed"]
    assert released
    assert kernel.leases.free_platforms() == ("carrier-12", "carrier-7", "sweeper-2")


def test_no_feasible_body_is_an_admission_fault() -> None:
    """The lint rejects this at load; reaching it means the fleet changed under a
    running system."""
    welder = dict(CARRY, name="weld", requires={"tooling": ["welder"]})
    kernel, events = build([welder], {"weld": WORK})
    kernel.spawn(DescriptorName("weld"))
    kernel.run_until_quiescent()

    assert of(events, AllocationRefused)
    faults = [f for f in of(events, FaultRaised) if f.fault is FaultKind.ADMISSION]
    assert faults and "no body satisfies" in faults[0].detail


# --- re-embodiment is a resume ----------------------------------------------


def test_revocation_is_a_suspension_not_a_loss() -> None:
    """A revocation is a suspension: the job keeps its transcript and goes on the stack."""
    kernel, events = build([CARRY], {"carry-beam": WORK}, platforms=(CARRIER_7,))
    job = kernel.spawn(DescriptorName("carry-beam"))
    kernel.tick()
    kernel.tick()
    assert kernel.leases.lease_of(job.job_id) is not None

    kernel.revoke_lease("carrier-7", reason="battery fault")
    assert job.state is JobState.SUSPENDED
    lost = of(events, Disembodied)
    assert lost and lost[0].reason == "battery fault"
    assert kernel.machine.transcript(job.job_id), "the transcript survives"


def test_re_embodiment_resumes_with_a_self_dirty_diff() -> None:
    """The dirty set is "dominated by ``self.*``" -- and the resume path needed
    no special case, because the kernel writes ``self.*`` on every embodiment."""
    kernel, events = build([CARRY], {"carry-beam": WORK})
    job = kernel.spawn(DescriptorName("carry-beam"))
    kernel.tick()
    kernel.tick()
    kernel.withdraw_platform("carrier-7", reason="battery fault")
    kernel.run_until_quiescent()

    resumed = [r for r in of(events, JobResumed) if r.job == job.job_id]
    assert resumed, "the job must resume in the spare"
    assert resumed[0].resume_kind is ResumeKind.DIRTY
    changed = {d.obj: d for d in resumed[0].dirty}
    assert ObjectName("self.platform") in changed
    assert changed[ObjectName("self.platform")].after == "carrier-12"
    assert ObjectName("self.battery") in changed


def test_nominally_identical_tooling_still_reports_a_note() -> None:
    """The audit finding this milestone owed. The re-embodiment diff includes

        self.tooling: gripper-std -> gripper-std (recalibrated; offsets differ)

    Same value, and the most important line in the diff. The old ``dirty_for``
    dropped it as an idempotent write.

    The fleet here is two bodies with *identical* tooling, which is what makes the
    case reachable at all: nothing about the value changes, only the machine it is
    bolted to.
    """
    spare = PlatformProfile(
        name="carrier-9",
        locomotion="wheeled",
        tooling=CARRIER_7.tooling,  # nominally the same gripper
        sensors=CARRIER_7.sensors,
        battery=0.70,
        location="dock-4",
    )
    kernel, events = build([CARRY], {"carry-beam": WORK}, platforms=(CARRIER_7, spare))
    job = kernel.spawn(DescriptorName("carry-beam"))
    kernel.tick()
    kernel.tick()
    kernel.withdraw_platform("carrier-7", reason="battery fault")
    kernel.run_until_quiescent()

    resumed = [r for r in of(events, JobResumed) if r.job == job.job_id]
    assert resumed
    tooling = next((d for d in resumed[0].dirty if d.obj == ObjectName("self.tooling")), None)
    assert tooling is not None, "a same-valued tooling change must survive the diff"
    assert tooling.before == tooling.after == "gripper-std,lift-500kg"
    assert "recalibrated" in tooling.note
    assert resumed[0].resume_kind is ResumeKind.DIRTY


def test_withdrawal_is_not_revocation() -> None:
    """The distinction F0 turned out to need.

    Revoking a lease frees a body for the next claimant, so a job whose lease is
    revoked and then re-embodied lands straight back in the same machine. That is
    correct for a scheduling decision and wrong for a flat battery. Withdrawal
    takes the body out of service.
    """
    kernel, _ = build([CARRY], {"carry-beam": WORK})
    job = kernel.spawn(DescriptorName("carry-beam"))
    kernel.tick()
    kernel.tick()

    kernel.revoke_lease("carrier-7", reason="reallocated")
    kernel.tick()
    lease = kernel.leases.lease_of(job.job_id)
    assert lease is not None and lease.platform == "carrier-7", "still a candidate"

    kernel.withdraw_platform("carrier-7", reason="battery fault")
    kernel.tick()
    lease = kernel.leases.lease_of(job.job_id)
    assert lease is not None and lease.platform != "carrier-7"
    assert "carrier-7" not in kernel.leases.platform_names()


def test_withdrawing_a_body_wakes_whoever_was_waiting_for_it() -> None:
    """The failure mode a resource table must not have: a job blocked on a body
    that has left the fleet would wait forever.

    One gripper, two claimants, and then the gripper goes to maintenance. The
    waiter must come back and be told there is nothing for it -- not sit on a
    resource that no longer exists.
    """
    holder = dict(CARRY, name="holder", priority=90)
    waiter = dict(CARRY, name="waiter", priority=10)  # urgent enough to preempt
    kernel, events = build(
        [holder, waiter],
        {
            "holder": [{"emit": "a"}, {"emit": "b"}, {"emit": "c"}, {"exit": True}],
            "waiter": WORK,
        },
        platforms=(CARRIER_7,),
    )
    kernel.spawn(DescriptorName("holder"))
    kernel.tick()
    kernel.tick()
    blocked = kernel.spawn(DescriptorName("waiter"))
    kernel.tick()
    assert blocked.state is JobState.BLOCKED, "queued behind the only gripper"

    kernel.withdraw_platform("carrier-7", reason="maintenance")
    kernel.run_until_quiescent()

    withdrawn = of(events, PlatformWithdrawn)
    assert withdrawn and withdrawn[0].reason == "maintenance"
    assert blocked.state is not JobState.BLOCKED, "the waiter must not be stranded"
    assert [f for f in of(events, FaultRaised) if f.job == blocked.job_id]


def test_a_joining_body_unblocks_a_waiting_job() -> None:
    """Joining the fleet is device hotplug, so it has to be able to happen
    while a job is already queued for exactly that capability.

    A job blocked on ``body:carrier-7`` is not really waiting for *that* body; it is
    waiting for any body that fits, and blocked on one lease because a resource is
    what the kernel knows how to wait on. An arrival has to reopen the question.
    """
    holder = dict(CARRY, name="holder", priority=90)
    kernel, events = build(
        [holder, CARRY],
        {
            "holder": [{"emit": "a"}, {"emit": "b"}, {"emit": "c"}, {"exit": True}],
            "carry-beam": WORK,
        },
        platforms=(CARRIER_7,),
    )
    kernel.spawn(DescriptorName("holder"))
    kernel.tick()
    kernel.tick()
    waiting = kernel.spawn(DescriptorName("carry-beam"))
    kernel.tick()
    assert of(events, ResourceBlocked), "queued for the only gripper in the fleet"

    kernel.join_platform(CARRIER_12)
    kernel.run_until_quiescent()

    joined = of(events, PlatformJoined)
    assert joined and joined[0].platform == "carrier-12"
    assert "gripper-std" in joined[0].profile
    embodied = [e for e in of(events, Embodied) if e.job == waiting.job_id]
    assert embodied and embodied[0].platform == "carrier-12"


def test_hotplug_returns_the_priority_the_waiter_had_donated() -> None:
    """Waking a waiter has to give the holder's borrowed urgency back.

    A job queued behind a lease donates its priority to the holder so the holder runs
    and releases. Hotplug unblocks that waiter a different way -- a new body arrives,
    and it stops waiting on the lease at all -- but the donation was released only on
    the release path. The holder went on carrying urgency borrowed from a job that no
    longer wanted anything from it, which is precisely the state inheritance exists to
    be temporary about, and it outranked that job's peers on the strength of it.
    """
    holder = dict(CARRY, name="holder", priority=90)
    kernel, events = build(
        [holder, CARRY],
        {
            "holder": [{"emit": "a"}, {"emit": "b"}, {"emit": "c"}, {"exit": True}],
            "carry-beam": WORK,
        },
        platforms=(CARRIER_7,),
    )
    held = kernel.spawn(DescriptorName("holder"))
    kernel.tick()
    kernel.tick()
    kernel.spawn(DescriptorName("carry-beam"))
    kernel.tick()
    assert of(events, PriorityInherited), "the waiter must have donated to begin with"
    assert held.current_priority == 40, "donated down to the waiter's priority"

    kernel.join_platform(CARRIER_12)

    assert held.current_priority == 90, "a body arrived; nothing is waiting on the lease"
    restored = [e for e in of(events, PriorityRestored) if e.job == held.job_id]
    assert restored and restored[0].to_priority == 90


def test_a_capability_no_body_has_faults_rather_than_waiting_for_hotplug() -> None:
    """The neighbouring case, and deliberately *not* the same one.

    Nothing feasible ever existing is a design error the lint catches at load
    Reaching it at runtime means the fleet shrank under a running system,
    and the kernel says so rather than parking the job against the possibility that
    a welder turns up one day.
    """
    welder = dict(CARRY, name="weld", requires={"tooling": ["welder"]})
    kernel, events = build([welder], {"weld": WORK}, platforms=(SWEEPER_2,))
    job = kernel.spawn(DescriptorName("weld"))
    kernel.run_until_quiescent()

    assert not of(events, ResourceBlocked), "it must not queue for a body that isn't"
    assert [f for f in of(events, FaultRaised) if f.fault is FaultKind.ADMISSION]
    assert job.state is not JobState.BLOCKED


def test_round_trips_are_still_suppressed() -> None:
    """The note mechanism must not have reopened the round-trip hole it sits beside."""
    from zeos.core.clock import Clock
    from zeos.world.store import ObjectSet
    from zeos.world.store import WorldStore as WS

    store = WS()
    t0, t1 = Clock(), Clock(token_clock=5)
    store.set(ObjectName("plant.valve"), "shut", at=t0)
    store.set(ObjectName("plant.valve"), "open", at=t1)
    store.set(ObjectName("plant.valve"), "shut", at=t1)
    assert store.dirty_for(ObjectSet.of(["plant.*"]), since=t1) == ()


def test_repeated_eviction_raises_starvation() -> None:
    """Per the Fleet design: "a mission preempted off bodies more than K times raises the same
    starvation fault" -- loud, not silently retried."""
    kernel, events = build([CARRY], {"carry-beam": WORK}, platforms=(CARRIER_7,))
    job = kernel.spawn(DescriptorName("carry-beam"))
    kernel.tick()
    kernel.tick()
    for _ in range(kernel.config.starvation_limit + 2):
        if kernel.leases.lease_of(job.job_id) is None:
            kernel._embody(job)  # pyright: ignore[reportPrivateUsage]
        kernel.revoke_lease("carrier-7", reason="churn")

    faults = [f for f in of(events, FaultRaised) if f.fault is FaultKind.STARVATION]
    assert faults and "evicted from a body" in faults[0].detail


# --- gangs ------------------------------------------------------------------

LEAD = {
    "name": "carry-lead",
    "priority": 40,
    "requires": {"tooling": ["gripper-std"]},
    "gang": {"name": "beam", "members": ["carry-lead", "carry-follow"], "coupling": "loose"},
}
FOLLOW = {
    "name": "carry-follow",
    "priority": 40,
    "requires": {"tooling": ["gripper-std"]},
}


def test_a_gang_is_dispatched_all_or_none() -> None:
    kernel, events = build([LEAD, FOLLOW], {"carry-lead": WORK, "carry-follow": WORK})
    spec = kernel.descriptors[DescriptorName("carry-lead")].gang
    assert spec is not None
    members = kernel.assemble_gang(spec)

    assert len(members) == 2
    assembled = of(events, GangAssembled)
    assert assembled and assembled[0].gang == "beam"
    assert set(assembled[0].platforms) == {"carrier-7", "carrier-12"}


def test_a_partial_gang_never_starts() -> None:
    """Gangs are all-or-none: a lone carrier lifting one end is worse than nothing happening."""
    kernel, events = build(
        [LEAD, FOLLOW],
        {"carry-lead": WORK, "carry-follow": WORK},
        platforms=(CARRIER_7,),  # only one gripper-capable body
    )
    spec = kernel.descriptors[DescriptorName("carry-lead")].gang
    assert spec is not None
    assert kernel.assemble_gang(spec) == ()

    assert not of(events, GangAssembled)
    dissolved = of(events, GangDissolved)
    assert dissolved and "partial gangs never start" in dissolved[0].reason
    assert kernel.leases.free_platforms() == ("carrier-7",), "no body was taken"


def test_evicting_one_member_dissolves_the_gang() -> None:
    """All-or-none preemption: preempting one carrier mid-lift while the other
    continues is strictly worse than preempting both."""
    kernel, events = build([LEAD, FOLLOW], {"carry-lead": WORK, "carry-follow": WORK})
    spec = kernel.descriptors[DescriptorName("carry-lead")].gang
    assert spec is not None
    members = kernel.assemble_gang(spec)
    kernel.tick()

    kernel.revoke_lease("carrier-7", reason="battery fault")

    dissolved = of(events, GangDissolved)
    assert dissolved and "member evicted" in dissolved[-1].reason
    assert all(kernel.leases.lease_of(m.job_id) is None for m in members), (
        "both carriers must lose their bodies, not just the faulted one"
    )


# --- load-time checks -------------------------------------------------------


def test_unsatisfiable_requires_is_rejected_at_load() -> None:
    """ "Rejected at load, not discovered at allocation"."""
    d = Descriptor.from_frontmatter(
        {"name": "weld", "priority": 50, "requires": {"tooling": ["welder"]}}
    )
    findings = lint({d.name: d}, platforms=FLEET)
    unsat = [f for f in findings if f.rule == "unsatisfiable-requires"]
    assert unsat and unsat[0].severity is Severity.ERROR
    assert "lacks tooling" in unsat[0].detail


def test_satisfiable_requires_passes() -> None:
    d = Descriptor.from_frontmatter(
        {"name": "carry", "priority": 50, "requires": {"tooling": ["gripper-std"]}}
    )
    assert not [f for f in lint({d.name: d}, platforms=FLEET) if f.rule == "unsatisfiable-requires"]


def test_rigid_gang_without_mesh_links_is_rejected() -> None:
    """Rigid coupling needs direct platform-to-platform links for phase sync. A
    rigid gang across robots that cannot talk directly is not slow, it is incorrect."""
    d = Descriptor.from_frontmatter(
        {
            "name": "carry-lead",
            "priority": 40,
            "requires": {"tooling": ["gripper-std"]},
            "gang": {"members": ["carry-lead"], "coupling": "rigid"},
        }
    )
    findings = lint({d.name: d}, platforms=FLEET)  # no profile declares mesh peers
    rigid = [f for f in findings if f.rule == "rigid-gang-without-mesh"]
    assert rigid and rigid[0].severity is Severity.ERROR


def test_unknown_gang_member_is_rejected() -> None:
    d = Descriptor.from_frontmatter(
        {
            "name": "carry-lead",
            "priority": 40,
            "gang": {"members": ["carry-lead", "ghost"], "coupling": "loose"},
        }
    )
    findings = lint({d.name: d})
    assert [f.rule for f in findings if f.rule == "unknown-gang-member"]


def test_an_invalid_release_policy_is_a_load_error() -> None:
    from zeos.descriptor.schema import DescriptorError

    with pytest.raises(DescriptorError, match="unknown release_policy"):
        Descriptor.from_frontmatter({"name": "x", "priority": 50, "release_policy": "whenever"})


def test_release_policy_grace_is_ordered() -> None:
    """The physical analogue of a masking budget: how long a body may be held
    against a more urgent claim."""
    assert ReleasePolicy.grace(ReleasePolicy.IMMEDIATE) == 0
    assert (
        ReleasePolicy.grace(ReleasePolicy.IMMEDIATE)
        < ReleasePolicy.grace(ReleasePolicy.ACTION_BOUNDARY)
        < ReleasePolicy.grace(ReleasePolicy.PLAN_STEP)
    )


# --- the allocator is a policy module ---------------------------------------


def test_allocator_is_swappable() -> None:
    """Per the Fleet design: "the allocator is a policy module with a kernel-defined
    contract"; ZEOS-Fleet "claims no new allocation algorithm"."""
    from zeos.core.allocator import Allocation, AllocationRequest, AllocatorPolicy

    class LastResort:
        """Deliberately perverse: always picks the *worst* feasible body."""

        def choose(
            self,
            request: AllocationRequest,
            *,
            profiles: Sequence[PlatformProfile],
            free: Sequence[str],
        ) -> Allocation:
            ranked = feasible_platforms(request.requirements, profiles)
            usable = [r for r in ranked if r.platform in set(free)]
            if not usable:
                return Allocation(request=request, candidates=ranked, reason="none free")
            return Allocation(request=request, platform=usable[-1].platform, candidates=ranked)

    assert isinstance(LastResort(), AllocatorPolicy)

    events: list[Event] = []
    kernel = Kernel(
        descriptors={DescriptorName("carry-beam"): Descriptor.from_frontmatter(CARRY)},
        machine=ScriptedMachine({"carry-beam": Script.from_spec(WORK)}, block_size=8),
        pipes=PipeTable(),
        vectors=VectorTable(),
        world=WorldStore(),
        platforms=FLEET,
        allocator=LastResort(),
        journal_sink=events,
    )
    kernel.start()
    kernel.spawn(DescriptorName("carry-beam"))
    kernel.run_until_quiescent()

    embodied = of(events, Embodied)
    assert embodied and embodied[0].platform == "carrier-12", "policy was honoured"


def test_greedy_allocator_reports_why_it_could_not_grant() -> None:
    allocator = GreedyAllocator()
    from zeos.core.allocator import AllocationRequest as Req
    from zeos.core.ids import JobId

    request = Req(
        job=JobId(1),
        descriptor=DescriptorName("carry"),
        requirements=EmbodimentRequirements.parse({"tooling": ["gripper-std"]}),
    )
    refused = allocator.choose(request, profiles=FLEET, free=[])
    assert not refused.granted and "every feasible body is leased" in refused.reason

    impossible = allocator.choose(
        Req(
            job=JobId(1),
            descriptor=DescriptorName("weld"),
            requirements=EmbodimentRequirements.parse({"tooling": ["welder"]}),
        ),
        profiles=FLEET,
        free=["carrier-7"],
    )
    assert not impossible.granted and "no body in the fleet" in impossible.reason
