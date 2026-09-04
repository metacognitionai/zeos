# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The allocator: which body serves which task, and on what terms.

The Fleet design is unusually explicit about what this module is *not*:

> Allocation -- which body serves which task -- is a matching problem with fifty
> years of literature… ZEOS-Fleet deliberately does not pick a winner. The
> **allocator is a policy module** with a kernel-defined contract.

So ``AllocatorPolicy`` is a protocol with a deliberately dull default, and what the
kernel owns is the set of invariants *around* allocation:

* every grant is a **lease**, and every lease is revocable under its declared
  release policy;
* every revocation is a **suspension, not a loss** -- the job keeps its transcript
  and resumes in another body;
* allocation respects the priority space: preempting a body follows the same rule
  as preempting tokens;
* deadlock over bodies and locks is detectable, because leases are ordinary
  resources in the one table (R0);
* starvation is surfaced -- a mission bounced off bodies more than K times raises
  the same fault a starved job does.

**A lease is a resource, not a new mechanism.** That is the whole reason R0 came
first: ``body:carrier-7`` is a capacity-1 resource, so it inherits blocking,
priority inheritance, cycle detection, and the wait-for graph without any of them
being re-implemented here. What this module adds is *matching*, *revocation*, and
*gangs*.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from zeos.core.embodiment import (
    EmbodimentRequirements,
    MatchResult,
    PlatformProfile,
    feasible_platforms,
    lease_name,
)
from zeos.core.ids import DescriptorName, JobId, ResourceName

__all__ = [
    "ReleasePolicy",
    "Lease",
    "GangSpec",
    "AllocatorPolicy",
    "GreedyAllocator",
    "AllocationRequest",
    "Allocation",
    "LeaseBook",
]


class ReleasePolicy:
    """At what granularity a job can be cleanly evicted from a body.

    Fleet calls this "the preemptibility declaration, applied to flesh instead of
    tokens", and the parallel is exact: ``preemptible: false`` masks interrupts for
    a bounded number of tokens, and a release policy bounds how long a body can be
    held against a more urgent claim.
    """

    IMMEDIATE = "immediate"
    ACTION_BOUNDARY = "at-action-boundary"
    PLAN_STEP = "at-plan-step"

    ALL = (IMMEDIATE, ACTION_BOUNDARY, PLAN_STEP)

    #: How many ticks a holder may keep the body after revocation is requested.
    #: A number is needed to make "at the current action boundary" mean anything in
    #: a kernel with no actions; these are the M0 stand-ins, and F2 replaces them
    #: with real action boundaries reported by the platform.
    GRACE_TICKS: Mapping[str, int] = {
        IMMEDIATE: 0,
        ACTION_BOUNDARY: 1,
        PLAN_STEP: 3,
    }

    @staticmethod
    def grace(policy: str) -> int:
        return ReleasePolicy.GRACE_TICKS.get(policy, 0)


@dataclass(frozen=True, slots=True)
class Lease:
    """A grant of one body to one job."""

    job: JobId
    platform: str
    release_policy: str = ReleasePolicy.ACTION_BOUNDARY
    gang: str = ""

    @property
    def resource(self) -> ResourceName:
        return lease_name(self.platform)


@dataclass(frozen=True, slots=True)
class GangSpec:
    """A coupled manoeuvre: members dispatched and preempted together."""

    name: str
    members: tuple[DescriptorName, ...] = ()
    coupling: str = "loose"  # rigid | loose
    sync_bound_ns: int | None = None
    on_member_fault: str = ""

    @property
    def is_rigid(self) -> bool:
        return self.coupling == "rigid"


@dataclass(frozen=True, slots=True)
class AllocationRequest:
    job: JobId
    descriptor: DescriptorName
    requirements: EmbodimentRequirements
    release_policy: str = ReleasePolicy.ACTION_BOUNDARY
    gang: str = ""


@dataclass(frozen=True, slots=True)
class Allocation:
    """What the allocator decided. ``platform`` empty means nothing was available."""

    request: AllocationRequest
    platform: str = ""
    candidates: tuple[MatchResult, ...] = ()
    reason: str = ""

    @property
    def granted(self) -> bool:
        return bool(self.platform)


@runtime_checkable
class AllocatorPolicy(Protocol):
    """The kernel-defined contract. Swap in an auction or an optimal assignment."""

    def choose(
        self,
        request: AllocationRequest,
        *,
        profiles: Sequence[PlatformProfile],
        free: Sequence[str],
    ) -> Allocation:
        """Pick a body for one request from those currently unleased."""
        ...


class GreedyAllocator:
    """Cheapest feasible free body, ties broken by name.

    Deliberately the dullest thing that satisfies the contract. Fleet's **OQ-F2**
    asks which allocator class wins *under lease revocability and priority
    preemption* -- whether revocable allocation changes the classical MRTA
    trade-offs -- and answering that needs a baseline to beat, not a clever default
    that muddies the comparison.
    """

    name = "greedy"

    def choose(
        self,
        request: AllocationRequest,
        *,
        profiles: Sequence[PlatformProfile],
        free: Sequence[str],
    ) -> Allocation:
        ranked = feasible_platforms(request.requirements, profiles)
        if not ranked:
            return Allocation(
                request=request,
                candidates=(),
                reason="no body in the fleet satisfies the requirements",
            )
        available = [r for r in ranked if r.platform in set(free)]
        if not available:
            return Allocation(
                request=request,
                candidates=ranked,
                reason=(f"every feasible body is leased: {[r.platform for r in ranked]}"),
            )
        return Allocation(request=request, platform=available[0].platform, candidates=ranked)


@dataclass
class LeaseBook:
    """Who holds which body, and which gangs are assembled.

    Holds no locking of its own -- the leases *are* resources in R0's table, so
    blocking and inheritance happen there. This is the index that makes revocation
    and gang assembly answerable.
    """

    profiles: dict[str, PlatformProfile] = field(default_factory=dict[str, PlatformProfile])
    leases: dict[str, Lease] = field(default_factory=dict[str, Lease])
    #: platform → tick at which a requested revocation becomes enforceable.
    revoking: dict[str, int] = field(default_factory=dict[str, int])
    #: How many times each job has been bounced off a body, for the starvation rule.
    evictions: dict[JobId, int] = field(default_factory=dict[JobId, int])

    def add_profile(self, profile: PlatformProfile) -> None:
        self.profiles[profile.name] = profile

    def remove_profile(self, platform: str) -> PlatformProfile | None:
        """Take a body out of service. It stops being an allocation candidate.

        Not the same as revoking its lease: a revoked body is free for the next
        claimant, a withdrawn one is gone. Keeping the eviction counts is
        deliberate -- a job bounced off three bodies that then left the fleet is
        still a job that has been bounced three times.
        """
        self.revoking.pop(platform, None)
        return self.profiles.pop(platform, None)

    def platform_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.profiles))

    def all_profiles(self) -> tuple[PlatformProfile, ...]:
        return tuple(self.profiles[n] for n in self.platform_names())

    def free_platforms(self) -> tuple[str, ...]:
        return tuple(n for n in self.platform_names() if n not in self.leases)

    def lease_of(self, job: JobId) -> Lease | None:
        return next((lease for lease in self.leases.values() if lease.job == job), None)

    def grant(self, lease: Lease) -> None:
        self.leases[lease.platform] = lease

    def revoke(self, platform: str) -> Lease | None:
        self.revoking.pop(platform, None)
        return self.leases.pop(platform, None)

    def gang_members(self, gang: str) -> tuple[Lease, ...]:
        return tuple(self.leases[p] for p in sorted(self.leases) if self.leases[p].gang == gang)

    def note_eviction(self, job: JobId) -> int:
        self.evictions[job] = self.evictions.get(job, 0) + 1
        return self.evictions[job]


def self_state(profile: PlatformProfile) -> dict[str, str]:
    """The ``self.*`` world state a job sees once embodied.

    A re-embodiment diff is "dominated by ``self.*``", so those objects
    have to exist and be written by the kernel on every embodiment change -- which
    is what makes changing bodies produce a resume notice rather than a silent
    substitution.
    """
    return {
        "self.platform": profile.name,
        "self.position": profile.location or "unknown",
        "self.tooling": ",".join(sorted(profile.tooling)) or "none",
        "self.locomotion": profile.locomotion or "none",
        "self.battery": f"{profile.battery:.0%}",
    }
