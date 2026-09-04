# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Holdable resources: mutexes, semaphores, and the things leases will be built on.

Core §2.2 names four kinds of resource a job can block on -- "tool, actuator, pipe,
lock" -- and until now the kernel modelled exactly one of them. Priority inheritance
was therefore wired to pipes and nothing else, and three separate parts of the
design were waiting on the same missing subsystem:

* the design's **multi-threaded extension** needs mutexes, semaphores, and atomics
  for concurrent jobs;
* **ZEOS-Fleet** needs physical mutexes -- doorways, corridors, cranes, charging
  docks -- with capacities and a *global* lock graph;
* **ZEOS-Fleet** defines an embodiment lease as "a resource in the core §2.2
  sense -- held, blocked-on, and subject to priority inheritance".

One table serves all three. That is the whole argument for building it before the
allocator: leases are defined in terms of resources, so building F0 first would
mean building the resource concept *inside* the allocator, where the multi-threaded
extension and the fleet layer cannot reach it.

Two properties are load-bearing and easy to lose:

**Capacity, not just exclusion.** A corridor might admit two. A mutex is the
degenerate case where capacity is one, not a separate mechanism.

**Deadlock is detected, not avoided by hope.** Because every hold and every wait
lives in one table, the wait-for graph is complete and cycles are findable. The
Fleet design is explicit that the response is "a loud scheduler fault naming the cycle" --
so the cycle itself is part of the fault, not something an operator reconstructs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import JobId, Priority, ResourceKind, ResourceName

__all__ = [
    "ResourceSpec",
    "Resource",
    "ResourceTable",
    "ResourceError",
    "Deadlock",
]


class ResourceError(RuntimeError):
    """Structural misuse -- an unknown resource, or a capacity below one."""


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """A declaration of something jobs can hold."""

    name: ResourceName
    #: How many holders at once. 1 is a mutex; N is a semaphore or a corridor that
    #: admits N; a *lease* is capacity 1 with an owner that can be revoked (F0).
    capacity: int = 1
    kind: ResourceKind = ResourceKind.MUTEX
    #: The node that owns this resource, for ZEOS-Fleet. Recorded and journalled in
    #: this phase; acted on when there is more than one kernel.
    authority: str = "local"
    description: str = ""

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ResourceError(f"resource {self.name}: capacity must be >= 1")


@dataclass(frozen=True, slots=True)
class Deadlock:
    """A cycle in the wait-for graph, with the victim already chosen.

    ``cycle`` is the jobs in wait order. ``victim`` is the one that must back out:
    the **lowest-priority** member, per the Fleet design's "lower mission priority backs
    out". Choosing here rather than at the call site keeps the policy in one place
    and makes it testable on its own.

    The comparison uses **base** priority, not inherited. A mission's importance is
    what it was declared at; an inherited priority is a temporary loan, and by the
    time a cycle forms the participants have usually lent priority to each other,
    which would flatten the comparison to a tie-break on job id.
    """

    cycle: tuple[JobId, ...]
    resources: tuple[ResourceName, ...]
    victim: JobId

    def render(self) -> str:
        hops = " -> ".join(
            f"job {job} waits on {res}" for job, res in zip(self.cycle, self.resources, strict=True)
        )
        return f"{hops} -> job {self.cycle[0]} (cycle); victim is job {self.victim}"


@dataclass
class Resource:
    """Runtime state of one resource."""

    spec: ResourceSpec
    #: Holders in acquisition order, so releases and journals are deterministic.
    holders: list[JobId] = field(default_factory=list[JobId])
    #: Jobs blocked wanting it, in arrival order. Wake order is by *priority*, not
    #: arrival -- arrival order only breaks ties, so that a queue cannot invert the
    #: scheduler's decisions.
    waiters: list[JobId] = field(default_factory=list[JobId])

    @property
    def name(self) -> ResourceName:
        return self.spec.name

    @property
    def free(self) -> int:
        return max(0, self.spec.capacity - len(self.holders))

    @property
    def available(self) -> bool:
        return self.free > 0

    def held_by(self, job: JobId) -> bool:
        return job in self.holders

    def acquire(self, job: JobId) -> bool:
        if self.held_by(job):
            return True  # re-acquisition is a no-op, not an error
        if not self.available:
            return False
        self.holders.append(job)
        if job in self.waiters:
            self.waiters.remove(job)
        return True

    def release(self, job: JobId) -> bool:
        if job not in self.holders:
            return False
        self.holders.remove(job)
        return True

    def add_waiter(self, job: JobId) -> None:
        if job not in self.waiters and not self.held_by(job):
            self.waiters.append(job)

    def drop_waiter(self, job: JobId) -> None:
        if job in self.waiters:
            self.waiters.remove(job)


class ResourceTable:
    """Every resource in one kernel, plus the wait-for graph over them."""

    def __init__(self, specs: Iterable[ResourceSpec] = ()) -> None:
        self._resources: dict[ResourceName, Resource] = {}
        for spec in specs:
            self.declare(spec)

    # -- declaration ---------------------------------------------------------

    def declare(self, spec: ResourceSpec) -> Resource:
        existing = self._resources.get(spec.name)
        if existing is not None:
            if existing.spec.capacity != spec.capacity:
                raise ResourceError(
                    f"resource {spec.name}: conflicting capacities "
                    f"({existing.spec.capacity} vs {spec.capacity})"
                )
            return existing
        resource = Resource(spec=spec)
        self._resources[spec.name] = resource
        return resource

    def forget(self, name: ResourceName) -> tuple[JobId, ...]:
        """Drop a resource that no longer exists, returning its stranded waiters.

        Only correct for resources that can genuinely disappear -- a leased body that
        left the fleet. A doorway does not stop existing because nobody is standing
        in it, so nothing else calls this. The waiters come back because a job
        blocked on a resource that has been deleted would otherwise wait forever,
        which is the one failure mode the table must not have.
        """
        resource = self._resources.pop(name, None)
        if resource is None:
            return ()
        return tuple(resource.waiters)

    def get(self, name: ResourceName) -> Resource:
        resource = self._resources.get(name)
        if resource is None:
            raise ResourceError(f"unknown resource {name!r}")
        return resource

    def has(self, name: ResourceName) -> bool:
        return name in self._resources

    def names(self) -> tuple[ResourceName, ...]:
        return tuple(sorted(self._resources))

    def all(self) -> tuple[Resource, ...]:
        return tuple(self._resources[n] for n in self.names())

    def __len__(self) -> int:
        return len(self._resources)

    # -- holding -------------------------------------------------------------

    def held_by(self, job: JobId) -> tuple[ResourceName, ...]:
        return tuple(r.name for r in self.all() if r.held_by(job))

    def release_all(self, job: JobId) -> tuple[ResourceName, ...]:
        """Release everything a job holds -- used when it completes or faults.

        A job that dies holding a lock would otherwise block every waiter forever,
        which is the one failure mode a resource table must not have. Note this is
        deliberately *not* what ZEOS-Fleet wants for a lost robot's **physical**
        locks: a dead robot may still be blocking the corridor it holds, so those
        are released only by reconciliation. That distinction arrives with F0.
        """
        released: list[ResourceName] = []
        for resource in self.all():
            if resource.release(job):
                released.append(resource.name)
            resource.drop_waiter(job)
        return tuple(released)

    def waiting_on(self, job: JobId) -> ResourceName | None:
        for resource in self.all():
            if job in resource.waiters:
                return resource.name
        return None

    # -- the wait-for graph --------------------------------------------------

    def _wait_edges(self) -> dict[JobId, tuple[ResourceName, tuple[JobId, ...]]]:
        """job → (resource it waits on, jobs holding that resource)."""
        edges: dict[JobId, tuple[ResourceName, tuple[JobId, ...]]] = {}
        for resource in self.all():
            for waiter in resource.waiters:
                edges[waiter] = (resource.name, tuple(resource.holders))
        return edges

    def find_deadlock(
        self,
        *,
        priorities: Mapping[JobId, Priority],
        extra: tuple[JobId, ResourceName] | None = None,
    ) -> Deadlock | None:
        """Find a cycle in the wait-for graph, if one exists.

        ``extra`` lets the kernel ask *"would this wait create a cycle?"* before
        actually blocking the job -- so a deadlock is reported at the moment it
        would form, naming the job that was about to close the loop, rather than
        after everything has already stopped.
        """
        edges = self._wait_edges()
        if extra is not None:
            job, name = extra
            edges[job] = (name, tuple(self.get(name).holders))

        for start in sorted(edges):
            path: list[JobId] = []
            resources: list[ResourceName] = []
            seen: set[JobId] = set()
            current = start
            while current in edges and current not in seen:
                seen.add(current)
                resource_name, holders = edges[current]
                path.append(current)
                resources.append(resource_name)
                # Follow the lowest-numbered holder for determinism. A cycle
                # through any holder is a cycle; picking a stable one keeps the
                # reported path reproducible across runs.
                nexts = [h for h in sorted(holders) if h != current]
                if not nexts:
                    break
                current = nexts[0]
                if current == start:
                    victim = max(
                        path, key=lambda j: (int(priorities.get(j, Priority(999))), int(j))
                    )
                    return Deadlock(
                        cycle=tuple(path),
                        resources=tuple(resources),
                        victim=victim,
                    )
        return None

    # -- wake order ----------------------------------------------------------

    def next_waiter(self, name: ResourceName, priorities: Mapping[JobId, Priority]) -> JobId | None:
        """The waiter that should get the resource next.

        Highest priority first, arrival order breaking ties. A FIFO queue here
        would let a low-priority job that happened to arrive first hold up an
        urgent one -- reintroducing, at the resource layer, exactly the inversion
        priority inheritance exists to prevent.
        """
        resource = self.get(name)
        if not resource.waiters:
            return None
        ordered = sorted(
            enumerate(resource.waiters),
            key=lambda pair: (int(priorities.get(pair[1], Priority(999))), pair[0]),
        )
        return ordered[0][1]


def lock_order_violations(
    orders: Mapping[str, Sequence[ResourceName]],
) -> tuple[tuple[str, str, ResourceName, ResourceName], ...]:
    """Find pairs of declared lock orders that disagree.

    Classic lock-ordering discipline: if every job acquires resources in one
    global order, cycles are impossible. Two descriptors that declare ``[a, b]``
    and ``[b, a]`` can deadlock, and that is knowable at load time rather than at
    3am in a corridor.

    Returns ``(descriptor_a, descriptor_b, first, second)`` for each disagreement:
    A takes ``first`` before ``second`` and B takes them the other way round.
    """
    violations: list[tuple[str, str, ResourceName, ResourceName]] = []
    names = sorted(orders)
    for i, left_name in enumerate(names):
        left = list(orders[left_name])
        left_rank = {r: n for n, r in enumerate(left)}
        for right_name in names[i + 1 :]:
            right = list(orders[right_name])
            right_rank = {r: n for n, r in enumerate(right)}
            shared = sorted(set(left_rank) & set(right_rank), key=lambda r: left_rank[r])
            for a_index, first in enumerate(shared):
                for second in shared[a_index + 1 :]:
                    if right_rank[first] > right_rank[second]:
                        violations.append((left_name, right_name, first, second))
    return tuple(violations)
