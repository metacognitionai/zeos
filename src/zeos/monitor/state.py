# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The fold: a journal becomes a picture of the system.

The design already commits to *"kernel state is a fold over this sequence"* (tech
spec's "journal is the source of truth" rule). A monitor is that fold,
rendered -- which is why this module adds no
kernel instrumentation at all. Three things follow, and they are the reason to build
it this way rather than by reaching into a live ``Kernel``:

* **Live and post-mortem are the same operation.** Tailing a running kernel's journal
  and replaying a recorded one differ only in where the lines come from.
* **A recorded run can be scrubbed.** Determinism is already a gate, so stepping
  backwards and forwards through an incident is exact rather than approximate. That
  is the difference between a dashboard and something you can drive.
* **The monitor is testable like everything else.** It is a pure function from a list
  of events to a snapshot, so it is covered by pytest rather than by looking at it.

**What this is not.** It is not a second source of truth. Anything the monitor
reports that the journal does not carry would be a fact nobody can verify after the
fact, so if a pane needs a number, the number needs an event. Building this surfaced
two places where that was not true -- ``JobSpawned`` carried no owner, and a job's
starting integrity appeared nowhere -- and the fix was to journal them, not to reach
around the journal for them.

The headline the top strip is built around is deliberate:

    12 jobs alive · 1 decoding · 1 forward pass/tick

That line *is* the design's central economic claim, made visible. Classical ``top``
leads with %CPU because a blocked process still occupies a scheduling slot; here a
blocked job costs nothing, and the gap between "alive" and "decoding" is the whole
argument.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import cast

from zeos.core.clock import Clock
from zeos.core.events import (
    AllocationRefused,
    AttentionDenied,
    AuthorityNarrowed,
    CapabilityChecked,
    CompilationRefused,
    DeadlockDetected,
    Decoded,
    DescriptorLoaded,
    Disembodied,
    EchoedBack,
    Elevated,
    ElevationEnded,
    Embodied,
    Endorsed,
    Event,
    FaultRaised,
    GangAssembled,
    GangDissolved,
    GateAnswered,
    GateConsulted,
    Injected,
    IntegrityDemoted,
    JobBlocked,
    JobCancelled,
    JobCompleted,
    JobDispatched,
    JobPreempted,
    JobResumed,
    JobSpawned,
    JobStateChanged,
    JobWoken,
    KernelStarted,
    PagedIn,
    PageFaultRaised,
    PipeBackpressure,
    PipeCreated,
    PipeReadEvent,
    PipeWritten,
    PlatformJoined,
    PlatformWithdrawn,
    PriorityInherited,
    PriorityRestored,
    Refaulted,
    ResourceAcquired,
    ResourceBlocked,
    ResourceReleased,
    SegmentEvicted,
    SegmentOpened,
    SpoofDetected,
    UtteranceCompiled,
    UtteranceReceived,
    VectorCoalesced,
    VectorFired,
    VectorThrottled,
    WorkingSetSampled,
    WorldWritten,
)
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    Integrity,
    JobId,
    JobState,
    ObjectName,
    PipeName,
    Principal,
    PrincipalId,
    Priority,
    ResourceName,
    ResumeKind,
    Ring,
)

__all__ = [
    "JobView",
    "PipeView",
    "VectorView",
    "ResourceView",
    "SystemView",
    "Counters",
    "PIPE_PREVIEW",
    "Monitor",
    "Timeline",
    "fold",
]


#: How much of a pipe's buffer a frame carries. A frame is delta-encoded and a
#: pipe's row is re-sent whenever it changes, so carrying an unbounded buffer makes
#: a payload quadratic in a deep pipe's traffic. ``PipeView.depth`` stays exact --
#: this bounds only what is shown, the same bargain ``render_event`` makes at 60
#: characters.
PIPE_PREVIEW = 64


# --- the pieces -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobView:
    """One row of the jobs table."""

    job: JobId
    descriptor: DescriptorName
    state: JobState = JobState.READY
    owner: PrincipalId = PrincipalId("kernel")
    base_priority: Priority = Priority(500)
    priority: Priority = Priority(500)
    #: Set while this job is running on borrowed urgency. The row should show the
    #: donation, because a job running at a priority its descriptor never declared
    #: is exactly the thing an operator needs explaining.
    inherited_from: JobId | None = None
    parent: JobId | None = None
    integrity: Integrity = Integrity(2)
    tokens: int = 0
    decodes: int = 0
    blocked_on: str = ""
    blocked_reason: str = ""
    platform: str = ""
    #: VM: resident tokens as last sampled, and the paging history that explains it.
    working_set: int = 0
    segments: int = 0
    evictions: int = 0
    refaults: int = 0
    page_faults: int = 0
    preempted: int = 0
    resumes_clean: int = 0
    resumes_dirty: int = 0
    faults: tuple[str, ...] = ()
    capability_denials: int = 0
    #: Resources held right now, and the one it is parked on.
    holding: tuple[ResourceName, ...] = ()
    waiting_for: ResourceName | None = None

    @property
    def alive(self) -> bool:
        return self.state not in (JobState.DONE, JobState.FAULTED)

    @property
    def inherited(self) -> bool:
        return self.inherited_from is not None

    @property
    def dirty(self) -> bool:
        """Watermark worse than the cleanest level. Worth colouring."""
        return int(self.integrity) > 0


@dataclass(frozen=True, slots=True)
class PipeView:
    name: PipeName
    ring: Ring = Ring.TRUSTED
    principal: Principal = Principal.PEER_JOB
    capacity_tokens: int = 0
    transport: str = "local"
    depth: int = 0
    written: int = 0
    read: int = 0
    backpressure_events: int = 0
    blocked_readers: tuple[JobId, ...] = ()
    #: True when a gate stands between this pipe and whoever writes to it.
    gated_by: str = ""
    #: What is sitting in the buffer right now, oldest first -- the head, because
    #: that is what the next read takes. Bounded by ``PIPE_PREVIEW``: ``depth`` is
    #: the exact count, and a pipe deeper than the preview shows its front.
    contents: tuple[str, ...] = ()

    @property
    def utilisation(self) -> float:
        if not self.capacity_tokens:
            return 0.0
        return min(1.0, self.depth / self.capacity_tokens)


@dataclass(frozen=True, slots=True)
class VectorView:
    name: str
    pipe: PipeName = PipeName("")
    handler: DescriptorName = DescriptorName("")
    priority: Priority = Priority(500)
    policy: str = ""
    fired: int = 0
    coalesced: int = 0
    throttled: int = 0


@dataclass(frozen=True, slots=True)
class ResourceView:
    name: ResourceName
    kind: str = "mutex"
    capacity: int = 1
    holders: tuple[JobId, ...] = ()
    waiters: tuple[JobId, ...] = ()

    @property
    def contended(self) -> bool:
        return bool(self.waiters)


@dataclass(frozen=True, slots=True)
class FaultRecord:
    at: Clock
    job: JobId | None
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class Counters:
    """The top strip.

    ``alive`` versus ``decoding`` is the headline: a hundred jobs can be in flight
    with one of them costing anything.
    """

    ticks: int = 0
    tokens: int = 0
    forward_passes: int = 0
    spawned: int = 0
    completed: int = 0
    faulted: int = 0
    preemptions: int = 0
    stack_depth: int = 0
    max_stack_depth: int = 0
    evictions: int = 0
    refaults: int = 0
    page_faults: int = 0
    capability_checks: int = 0
    capability_denials: int = 0
    demotions: int = 0
    endorsements: int = 0
    utterances: int = 0
    gate_vetoes: int = 0
    deadlocks: int = 0

    @property
    def blocked_cost(self) -> str:
        """The claim, as a phrase. Blocked jobs consume no forward passes."""
        return "blocked jobs cost nothing"


@dataclass(frozen=True, slots=True)
class SystemView:
    """One frame. Everything a pane needs, at one point in the journal."""

    seq: int = -1
    clock: Clock = field(default_factory=Clock)
    #: Which virtual-time instant this frame sits in, counting from 0. The journal
    #: does not record boundaries, but a driver moves virtual time between them and
    #: never during one, so each new instant is one driver tick -- the ``tN`` a
    #: driver's own printout counts.
    tick: int = 0
    case: str = ""
    seed: int = 0
    block_size: int = 0
    counters: Counters = field(default_factory=Counters)
    jobs: tuple[JobView, ...] = ()
    pipes: tuple[PipeView, ...] = ()
    vectors: tuple[VectorView, ...] = ()
    resources: tuple[ResourceView, ...] = ()
    stack: tuple[JobId, ...] = ()
    running: JobId | None = None
    leases: Mapping[str, JobId] = field(default_factory=dict[str, JobId])
    platforms: tuple[str, ...] = ()
    elevations: Mapping[PrincipalId, str] = field(default_factory=dict[PrincipalId, str])
    world: Mapping[ObjectName, str] = field(default_factory=dict[ObjectName, str])
    faults: tuple[FaultRecord, ...] = ()
    #: The event that produced this frame, rendered. Drives the ticker and gives the
    #: scrubber something to label each step with.
    last_event: str = ""
    last_kind: str = ""

    # -- derived, so panes do not each reinvent them -------------------------

    @property
    def alive(self) -> tuple[JobView, ...]:
        return tuple(j for j in self.jobs if j.alive)

    @property
    def decoding(self) -> JobView | None:
        return next((j for j in self.jobs if j.job == self.running), None)

    @property
    def blocked(self) -> tuple[JobView, ...]:
        return tuple(j for j in self.jobs if j.state is JobState.BLOCKED)

    @property
    def suspended(self) -> tuple[JobView, ...]:
        return tuple(j for j in self.jobs if j.state is JobState.SUSPENDED)

    def job(self, job_id: JobId) -> JobView | None:
        return next((j for j in self.jobs if j.job == job_id), None)

    def headline(self) -> str:
        n_alive = len(self.alive)
        return (
            f"{n_alive} job{'s' if n_alive != 1 else ''} alive · "
            f"{1 if self.running is not None else 0} decoding · "
            f"{len(self.blocked)} blocked · {len(self.suspended)} suspended"
        )


# --- the fold ---------------------------------------------------------------


class Monitor:
    """Applies journal events one at a time and can snapshot at any point.

    Mutable internally, immutable outward: ``snapshot`` returns a frozen ``SystemView``
    so a scrubber can keep every frame without them aliasing each other.
    """

    def __init__(self) -> None:
        self.seq = -1
        self.clock = Clock()
        self.tick = 0
        self.case = ""
        self.seed = 0
        self.block_size = 0
        self._jobs: dict[JobId, JobView] = {}
        self._pipes: dict[PipeName, PipeView] = {}
        #: The reconstructed buffer per pipe, in full. ``PipeView.contents`` is its
        #: head; keeping the whole thing here is what makes a later read show the
        #: right tokens. Bounded by the pipe's declared capacity.
        self._buffers: dict[PipeName, deque[str]] = {}
        self._vectors: dict[str, VectorView] = {}
        self._resources: dict[ResourceName, ResourceView] = {}
        self._descriptors: dict[DescriptorName, Priority] = {}
        self._stack: list[JobId] = []
        self._running: JobId | None = None
        self._leases: dict[str, JobId] = {}
        self._platforms: set[str] = set()
        self._elevations: dict[PrincipalId, str] = {}
        self._world: dict[ObjectName, str] = {}
        self._faults: list[FaultRecord] = []
        self._counters = Counters()
        self._last_event = ""
        self._last_kind = ""

    # -- job helpers ---------------------------------------------------------

    def _job(self, job_id: JobId) -> JobView:
        existing = self._jobs.get(job_id)
        if existing is None:
            # A journal that starts mid-run is legal -- the driver streams, and a
            # viewer may attach late. An unknown job gets a placeholder rather than
            # a KeyError, because a monitor that crashes on a partial journal is
            # useless exactly when it is most wanted.
            existing = JobView(job=job_id, descriptor=DescriptorName(f"job-{job_id}"))
            self._jobs[job_id] = existing
        return existing

    def _update(self, job_id: JobId, **changes: object) -> None:
        self._jobs[job_id] = replace(self._job(job_id), **changes)  # pyright: ignore[reportArgumentType]

    def _bump(self, **changes: int) -> None:
        current = {k: getattr(self._counters, k) + v for k, v in changes.items()}
        self._counters = replace(self._counters, **current)

    # -- the dispatch --------------------------------------------------------

    def apply(self, event: Event, *, seq: int | None = None) -> None:
        if seq is not None:
            self.seq = seq
        else:
            self.seq += 1
        if event.clock.virtual_ns > self.clock.virtual_ns:
            self.tick += 1
        self.clock = event.clock
        self._last_kind = type(event).KIND
        self._last_event = render_event(event)

        match event:
            case KernelStarted():
                self.case, self.seed = event.case, event.seed
                self.block_size = event.block_size
            case DescriptorLoaded():
                self._descriptors[event.descriptor] = event.priority

            # -- job lifecycle -------------------------------------------------
            case JobSpawned():
                self._jobs[event.job] = JobView(
                    job=event.job,
                    descriptor=event.descriptor,
                    owner=event.owner,
                    base_priority=event.priority,
                    priority=event.priority,
                    parent=event.parent,
                    integrity=event.integrity,
                )
                self._bump(spawned=1)
            case JobStateChanged():
                self._update(event.job, state=event.to_state)
                if event.to_state is JobState.RUNNING:
                    self._running = event.job
                elif self._running == event.job:
                    self._running = None
                if event.to_state is JobState.SUSPENDED:
                    if event.job not in self._stack:
                        self._stack.append(event.job)
                elif event.job in self._stack:
                    self._stack.remove(event.job)
                depth = len(self._stack)
                self._counters = replace(
                    self._counters,
                    stack_depth=depth,
                    max_stack_depth=max(self._counters.max_stack_depth, depth),
                )
            case JobDispatched():
                self._update(event.job, priority=event.priority)
                self._running = event.job
            case JobPreempted():
                job = self._job(event.job)
                self._update(event.job, preempted=job.preempted + 1)
                self._bump(preemptions=1)
            case JobBlocked():
                self._update(event.job, blocked_on=str(event.pipe), blocked_reason=event.reason)
            case JobWoken():
                self._update(event.job, blocked_on="", blocked_reason="")
            case JobResumed():
                job = self._job(event.job)
                if event.resume_kind is ResumeKind.DIRTY:
                    self._update(event.job, resumes_dirty=job.resumes_dirty + 1)
                else:
                    self._update(event.job, resumes_clean=job.resumes_clean + 1)
            case JobCompleted():
                self._update(
                    event.job,
                    state=JobState.DONE,
                    tokens=event.tokens_used,
                    blocked_on="",
                    blocked_reason="",
                )
                if self._running == event.job:
                    self._running = None
                self._bump(completed=1)
            case JobCancelled():
                self._update(event.job, state=JobState.DONE)
                if self._running == event.job:
                    self._running = None

            # -- priority inheritance -----------------------------------------
            case PriorityInherited():
                self._update(
                    event.job,
                    priority=event.to_priority,
                    inherited_from=event.blocked_job,
                )
            case PriorityRestored():
                self._update(event.job, priority=event.to_priority, inherited_from=None)

            # -- machine -------------------------------------------------------
            case Decoded():
                job = self._job(event.job)
                self._update(event.job, tokens=job.tokens + event.tokens, decodes=job.decodes + 1)
                self._bump(tokens=event.tokens, forward_passes=event.tokens, ticks=1)
            case Injected():
                self._bump(tokens=event.tokens)
            case SegmentOpened():
                job = self._job(event.job)
                self._update(event.job, segments=job.segments + 1)

            # -- protection ----------------------------------------------------
            case IntegrityDemoted():
                self._update(event.job, integrity=event.to_integrity)
                self._bump(demotions=1)
            case Endorsed():
                self._bump(endorsements=1)
            case CapabilityChecked():
                self._bump(capability_checks=1)
                if not event.allowed:
                    job = self._job(event.job)
                    self._update(event.job, capability_denials=job.capability_denials + 1)
                    self._bump(capability_denials=1)
            case AttentionDenied() | SpoofDetected():
                pass  # counted through the fault they raise

            # -- virtual context ------------------------------------------------
            case SegmentEvicted():
                job = self._job(event.job)
                self._update(event.job, evictions=job.evictions + 1)
                self._bump(evictions=1)
            case Refaulted():
                job = self._job(event.job)
                self._update(event.job, refaults=job.refaults + 1)
                self._bump(refaults=1)
            case PageFaultRaised():
                job = self._job(event.job)
                self._update(event.job, page_faults=job.page_faults + 1)
                self._bump(page_faults=1)
            case PagedIn():
                pass
            case WorkingSetSampled():
                self._update(event.job, working_set=event.size_tokens)

            # -- pipes ----------------------------------------------------------
            case PipeCreated():
                self._pipes[event.pipe] = PipeView(
                    name=event.pipe,
                    ring=event.ring,
                    principal=event.principal,
                    capacity_tokens=event.capacity_tokens,
                    transport=event.transport,
                )
            case PipeWritten():
                pipe = self._pipe(event.pipe)
                buffer = self._buffer(event.pipe)
                # A latched write replaces the buffer rather than appending to it:
                # an actuator pipe holds the current value, not a backlog. Treating
                # it as an append over-reports depth on every actuator pipe, which
                # is what this fold did before ``PipeWritten`` carried the fact.
                if event.latched:
                    buffer.clear()
                buffer.extend(event.text)
                self._pipes[event.pipe] = replace(
                    pipe,
                    depth=(0 if event.latched else pipe.depth) + event.tokens,
                    written=pipe.written + event.tokens,
                    contents=self._preview(event.pipe),
                )
            case PipeReadEvent():
                pipe = self._pipe(event.pipe)
                buffer = self._buffer(event.pipe)
                for _ in range(min(event.tokens, len(buffer))):
                    buffer.popleft()
                self._pipes[event.pipe] = replace(
                    pipe,
                    depth=max(0, pipe.depth - event.tokens),
                    read=pipe.read + event.tokens,
                    contents=self._preview(event.pipe),
                )
            case PipeBackpressure():
                pipe = self._pipe(event.pipe)
                self._pipes[event.pipe] = replace(
                    pipe, backpressure_events=pipe.backpressure_events + 1
                )

            # -- vectors ---------------------------------------------------------
            case VectorFired():
                v = self._vector(event.vector)
                self._vectors[event.vector] = replace(
                    v,
                    pipe=event.pipe,
                    handler=event.handler,
                    priority=event.priority,
                    policy=event.policy,
                    fired=v.fired + 1,
                )
            case VectorCoalesced():
                v = self._vector(event.vector)
                self._vectors[event.vector] = replace(v, coalesced=v.coalesced + event.collapsed)
            case VectorThrottled():
                v = self._vector(event.vector)
                self._vectors[event.vector] = replace(v, throttled=v.throttled + 1)

            # -- resources --------------------------------------------------------
            case ResourceAcquired():
                r = self._resource(event.resource)
                job = self._job(event.job)
                self._resources[event.resource] = replace(
                    r,
                    kind=str(event.resource_kind),
                    capacity=event.capacity,
                    holders=tuple(sorted({*r.holders, event.job})),
                    waiters=tuple(w for w in r.waiters if w != event.job),
                )
                self._update(
                    event.job,
                    holding=tuple(sorted({*job.holding, event.resource})),
                    waiting_for=None,
                )
            case ResourceBlocked():
                r = self._resource(event.resource)
                self._resources[event.resource] = replace(
                    r,
                    holders=event.holders,
                    waiters=tuple(sorted({*r.waiters, event.job})),
                )
                self._update(event.job, waiting_for=event.resource)
            case ResourceReleased():
                r = self._resource(event.resource)
                job = self._job(event.job)
                self._resources[event.resource] = replace(
                    r, holders=tuple(h for h in r.holders if h != event.job)
                )
                self._update(
                    event.job,
                    holding=tuple(h for h in job.holding if h != event.resource),
                )
            case DeadlockDetected():
                self._bump(deadlocks=1)

            # -- fleet -------------------------------------------------------------
            case PlatformJoined():
                self._platforms.add(event.platform)
            case PlatformWithdrawn():
                self._platforms.discard(event.platform)
                self._leases.pop(event.platform, None)
            case Embodied():
                self._platforms.add(event.platform)
                self._leases[event.platform] = event.job
                self._update(event.job, platform=event.platform)
            case Disembodied():
                self._leases.pop(event.platform, None)
                self._update(event.job, platform="")
            case GangAssembled() | GangDissolved() | AllocationRefused():
                pass

            # -- nli ----------------------------------------------------------------
            case UtteranceReceived():
                self._bump(utterances=1)
            case AuthorityNarrowed():
                self._update(event.job, owner=event.principal)
            case Elevated():
                self._elevations[event.principal] = ", ".join(str(c) for c in event.capabilities)
            case ElevationEnded():
                self._elevations.pop(event.principal, None)
            case GateConsulted():
                pipe = self._pipe(event.pipe)
                self._pipes[event.pipe] = replace(pipe, gated_by=str(event.gate))
            case GateAnswered():
                if not event.allowed:
                    self._bump(gate_vetoes=1)
            case UtteranceCompiled() | CompilationRefused() | EchoedBack():
                pass

            # -- world ---------------------------------------------------------------
            case WorldWritten():
                self._world[event.obj] = event.after

            # -- faults ---------------------------------------------------------------
            case FaultRaised():
                job = self._job(event.job)
                self._update(event.job, faults=(*job.faults, str(event.fault)))
                self._faults.append(
                    FaultRecord(
                        at=event.clock,
                        job=event.job,
                        kind=str(event.fault),
                        detail=event.detail,
                    )
                )
                if event.fault in (FaultKind.BUDGET, FaultKind.DEADLINE):
                    self._bump(faulted=1)

            case _:
                pass

    def _pipe(self, name: PipeName) -> PipeView:
        existing = self._pipes.get(name)
        if existing is None:
            existing = PipeView(name=name)
            self._pipes[name] = existing
        return existing

    def _buffer(self, name: PipeName) -> deque[str]:
        existing = self._buffers.get(name)
        if existing is None:
            existing = deque[str]()
            self._buffers[name] = existing
        return existing

    def _preview(self, name: PipeName) -> tuple[str, ...]:
        buffer = self._buffer(name)
        return tuple(buffer)[:PIPE_PREVIEW]

    def _vector(self, name: str) -> VectorView:
        existing = self._vectors.get(name)
        if existing is None:
            existing = VectorView(name=name)
            self._vectors[name] = existing
        return existing

    def _resource(self, name: ResourceName) -> ResourceView:
        existing = self._resources.get(name)
        if existing is None:
            existing = ResourceView(name=name)
            self._resources[name] = existing
        return existing

    # -- output ---------------------------------------------------------------

    def snapshot(self) -> SystemView:
        """Freeze the current state.

        Everything is sorted, because a monitor whose rows reorder between frames is
        unreadable -- and because a deterministic kernel deserves a deterministic view
        of itself.
        """
        return SystemView(
            seq=self.seq,
            clock=self.clock,
            tick=self.tick,
            case=self.case,
            seed=self.seed,
            block_size=self.block_size,
            counters=self._counters,
            jobs=tuple(self._jobs[j] for j in sorted(self._jobs)),
            pipes=tuple(self._pipes[p] for p in sorted(self._pipes)),
            vectors=tuple(self._vectors[v] for v in sorted(self._vectors)),
            resources=tuple(self._resources[r] for r in sorted(self._resources)),
            stack=tuple(self._stack),
            running=self._running,
            leases=dict(sorted(self._leases.items())),
            platforms=tuple(sorted(self._platforms)),
            elevations=dict(sorted(self._elevations.items())),
            world=dict(sorted(self._world.items())),
            faults=tuple(self._faults),
            last_event=self._last_event,
            last_kind=self._last_kind,
        )


@dataclass(frozen=True, slots=True)
class Timeline:
    """Every frame of a recorded run, plus the interesting moments.

    ``marks`` is what makes a scrubber usable: a run of a few thousand frames is
    unnavigable without somewhere to jump to, and the events worth jumping to are
    exactly the ones the design makes claims about -- a preemption, a fault, a
    demotion, a dirty resume, a gate veto.
    """

    frames: tuple[SystemView, ...] = ()
    marks: tuple[tuple[int, str], ...] = ()

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def final(self) -> SystemView:
        return self.frames[-1] if self.frames else SystemView()


#: Event kinds worth offering as scrubber marks. Each is a moment the specs make a
#: claim about, which is the criterion -- not "rare" and not "red".
MARKED_KINDS: Mapping[str, str] = {
    "job.preempted": "preemption",
    "job.resumed": "resume",
    "fault.raised": "fault",
    "integrity.demoted": "demotion",
    "vm.evicted": "eviction",
    "vm.refault": "refault",
    "resource.deadlock": "deadlock",
    "nli.gate_answered": "gate",
    "security.spoof": "spoof",
    "fleet.disembodied": "body lost",
    "job.priority_inherited": "inheritance",
}


def fold(events: Iterable[Event], *, every: int = 1) -> Timeline:
    """Replay a journal into a scrubbable timeline.

    ``every`` decimates: one frame per N events. The default keeps every frame, which
    is right for anything up to a few thousand events and is what makes stepping
    token-by-token through an incident possible. A long soak run would set it higher
    and lose only the ability to stop between two adjacent events.
    """
    monitor = Monitor()
    frames: list[SystemView] = []
    marks: list[tuple[int, str]] = []
    for index, event in enumerate(events):
        monitor.apply(event, seq=index)
        kind = type(event).KIND
        if kind in MARKED_KINDS:
            marks.append((len(frames), MARKED_KINDS[kind]))
        if index % every == 0:
            frames.append(monitor.snapshot())
    if not frames or frames[-1].seq != monitor.seq:
        frames.append(monitor.snapshot())
    return Timeline(frames=tuple(frames), marks=tuple(marks))


def render_event(event: Event) -> str:
    """One line for the ticker.

    Deliberately terse and deliberately uniform: the ticker is read by scanning, so a
    line that formats itself specially for its own event type is a line that breaks
    the scan.
    """
    kind = type(event).KIND
    parts: list[str] = []
    for name in fields(event):
        if name.name == "clock":
            continue
        value: object = getattr(event, name.name)
        if value is None or value == "" or value == () or value == 0:
            continue
        if isinstance(value, tuple):
            items = [str(v) for v in cast("tuple[object, ...]", value)]
            rendered = f"[{len(items)} items]" if len(items) > 3 else ", ".join(items)
        else:
            rendered = str(value)
        if len(rendered) > 60:
            rendered = rendered[:57] + "..."
        parts.append(f"{name.name}={rendered}")
    return f"{kind} {' '.join(parts)}".strip()


def from_records(records: Sequence[object]) -> Timeline:
    """Fold journal records (``seq`` + ``event``) as read from a file."""
    # getattr, not r.event: records is Sequence[object], and hasattr is the guard.
    return fold([getattr(r, "event") for r in records if hasattr(r, "event")])  # noqa: B009
