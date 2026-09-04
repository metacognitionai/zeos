# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The kernel: one step function over a scheduler, pipes, vectors, and a machine.

Single-threaded, no I/O, no wall-clock reads, no randomness that is not seeded, and
every iteration order is explicitly sorted. All time enters as data via
``advance_time``. Those properties -- not literal referential transparency -- are what
the design needs, and what makes a run reproducible from (descriptor tree, scripts,
event schedule, seed).

A note on the deviation from the plan's ``step(state, event) -> (state', effects)``
signature: state is owned by this object and mutated only through the methods below.
Threading an immutable kernel state through every helper would add a great deal of
Python noise for no additional guarantee, since the guarantees that matter
(determinism, replayability, mock-free tests, portability of the core) all follow
from "no I/O, no clock, no unordered iteration" rather than from immutability.
Nothing outside this module may reach in and mutate kernel state.

The quantum is one decode step, which is one token boundary. That is what makes
"preempts within one token boundary" (core §5.2) a checkable property rather than
an aspiration: between a vector firing and its handler being dispatched, at most one
decode of the preempted job may occur -- and in practice, zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from zeos.core.allocator import (
    AllocationRequest,
    AllocatorPolicy,
    GangSpec,
    GreedyAllocator,
    Lease,
    LeaseBook,
    self_state,
)
from zeos.core.capabilities import CapabilityTable, check_write
from zeos.core.clock import Clock, format_duration
from zeos.core.embodiment import (
    LEASE_PREFIX,
    PlatformProfile,
    feasible_platforms,
    lease_name,
)
from zeos.core.events import (
    AllocationRefused,
    AttentionDenied,
    AuthorityNarrowed,
    BlockBoundary,
    CapabilityChecked,
    CeilingApplied,
    CompilationRefused,
    DeadlockDetected,
    Decoded,
    DescriptorLoaded,
    Disembodied,
    EchoedBack,
    Elevated,
    ElevationEnded,
    ElevationRefused,
    Embodied,
    Endorsed,
    Event,
    FaultDispatched,
    FaultRaised,
    Forked,
    FrameSent,
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
    LinkStateChanged,
    MapRefreshed,
    MaskUpdated,
    Note,
    OwnershipApplied,
    OwnershipRefused,
    PagedIn,
    PageFaultRaised,
    PermsChanged,
    PipeBackpressure,
    PipeCreated,
    PipeReadEvent,
    PipeWritten,
    PlatformJoined,
    PlatformWithdrawn,
    PriorityInherited,
    PriorityRestored,
    Refaulted,
    ReplicaRefreshed,
    ResidencyChanged,
    ResourceAcquired,
    ResourceBlocked,
    ResourceReleased,
    SegmentClosed,
    SegmentEvicted,
    SegmentOpened,
    SpawnRefused,
    Spliced,
    StalenessRefused,
    StateDelta,
    StatusRegionRetracted,
    UtteranceCompiled,
    UtteranceReceived,
    VectorCoalesced,
    VectorFired,
    VectorThrottled,
    WorkingSetSampled,
    WorldWritten,
)
from zeos.core.faults import Fault, FaultAction, resolve
from zeos.core.gates import (
    ALLOW,
    GateRequest,
    GateSpec,
    GateTable,
    GateVerdict,
    parse_verdict,
)
from zeos.core.ids import (
    KERNEL_PIPE,
    DescriptorName,
    FaultKind,
    Integrity,
    JobId,
    JobState,
    ObjectName,
    OnComplete,
    Perm,
    PipeName,
    Principal,
    PrincipalId,
    Priority,
    Residency,
    ResourceName,
    ResumeKind,
    Ring,
    SegmentId,
    TokenKind,
    VectorName,
)
from zeos.core.integrity import DEFAULT_THETA_READ, demote_for_boundary
from zeos.core.pager import Pager, PagerResult, choose_plan
from zeos.core.pcb import Job
from zeos.core.pipes import PipeTable
from zeos.core.principals import (
    KERNEL_PRINCIPAL,
    Elevation,
    PrincipalTable,
)
from zeos.core.residency import (
    EvictionCandidate,
    ThrashMonitor,
    admission_check,
    plan_eviction,
    render_stub,
    stub_size,
    summarise,
    working_set_tokens,
)
from zeos.core.resources import ResourceKind, ResourceSpec, ResourceTable
from zeos.core.scheduler import Scheduler
from zeos.core.segments import (
    TAG_DESCRIPTOR,
    Provenance,
    SegmentRecord,
    SegmentTable,
    map_tag,
)
from zeos.core.store import EvictionRecord, SpanStore
from zeos.core.topology import LINK_STATE, LinkPort, Topology
from zeos.core.vectors import VectorAction, VectorTable
from zeos.descriptor.schema import CompartmentSpec, Descriptor
from zeos.machine.base import (
    AttentionHint,
    DecodeResult,
    MachineBackend,
    OpKind,
    Token,
    render,
    tokens_from_text,
)
from zeos.nli.compiler import Artifact, ArtifactKind, Phrasing, parse_phrasings
from zeos.nli.dispatcher import (
    Decision,
    OwnershipOp,
    OwnershipRequest,
    decide,
    echo_back,
)
from zeos.nli.envelope import Utterance
from zeos.world.store import ObjectSet, WorldStore

__all__ = ["Kernel", "KernelConfig", "KernelError", "render_resume_notice"]


def render_resume_notice(suspended_ns: int, dirty: Sequence[StateDelta]) -> str:
    """The text a resumed job actually reads, rendered only when something changed.

    Public because it is behaviour, not presentation: a bare flag is not enough --
    the diff must be salient enough to override the stale in-context state the job
    is still attending to (core §6.2). Whether current models honour it is the open
    question the M2 study exists to answer; what the kernel can guarantee is that
    the diff is correct, complete, and present.
    """
    lines = [
        "<RESUME>",
        f"Suspended {format_duration(suspended_ns)}.",
        "Changed state you depend on:",
    ]
    lines.extend(
        f"  {d.obj}: {d.before or '(unset)'} -> {d.after}" + (f" ({d.note})" if d.note else "")
        for d in dirty
    )
    lines.append("Revalidate your current plan step before acting.")
    lines.append("</RESUME>")
    return " ".join(lines)


class KernelError(RuntimeError):
    """A structural error in how the kernel was driven -- not a job fault."""


@dataclass(frozen=True, slots=True)
class KernelConfig:
    seed: int = 0
    case: str = "unnamed"
    #: Preemptions before a job raises a starvation fault. Loud by design: real-time
    #: systems should fail visibly rather than silently age priorities (core §5.5).
    starvation_limit: int = 8
    #: Safety valve for ``run_until_quiescent``; a script that never exits is a bug,
    #: and an infinite loop is a worse way to discover it than an exception.
    max_ticks: int = 10_000
    #: Attention mass at which a segment counts as *used* and can demote the
    #: watermark. Merely containing dirt does not demote; using it does.
    theta_read: float = DEFAULT_THETA_READ
    #: EMA horizon in blocks for the attention reference signal.
    tau_blocks: float = 8.0


@dataclass
class _Counters:
    next_job_id: int = 1
    next_seq: int = 0
    next_segment: int = 1


class Kernel:
    def __init__(
        self,
        *,
        descriptors: Mapping[DescriptorName, Descriptor],
        machine: MachineBackend,
        pipes: PipeTable,
        vectors: VectorTable,
        world: WorldStore,
        resources: ResourceTable | None = None,
        platforms: Sequence[PlatformProfile] = (),
        allocator: AllocatorPolicy | None = None,
        principals: PrincipalTable | None = None,
        gates: GateTable | None = None,
        node: str = "",
        topology: Topology | None = None,
        link: LinkPort | None = None,
        journal_sink: list[Event] | None = None,
        config: KernelConfig | None = None,
    ) -> None:
        self.descriptors = dict(descriptors)
        self.machine = machine
        self.pipes = pipes
        self.vectors = vectors
        self.world = world
        self.resources = resources or ResourceTable()
        self.leases = LeaseBook()
        for profile in platforms:
            self.leases.add_profile(profile)
        self.allocator: AllocatorPolicy = allocator or GreedyAllocator()
        self.principals = principals if principals is not None else PrincipalTable()
        self.gates = gates if gates is not None else GateTable()
        # ZEOS-Distributed. ``node`` names this kernel's scheduling domain; priorities
        # are not comparable across nodes and no attempt is made to make them so
        # ``link`` is the outbound transport toward the peer, owned by the
        # driver -- the kernel writes into it and never polls it, because polling is
        # I/O and the kernel does none.
        self.node = node
        self.topology = topology if topology is not None else Topology()
        self.link = link
        self.config = config or KernelConfig()
        self.sched = Scheduler()
        self.clock = Clock()
        self.store = SpanStore()
        self.pager = Pager(self.store)
        self._events: list[Event] = journal_sink if journal_sink is not None else []
        self._counters = _Counters()
        self._started = False

    # -- journal -------------------------------------------------------------

    @property
    def events(self) -> Sequence[Event]:
        return self._events

    def _emit(self, event: Event) -> None:
        self._events.append(event)

    def _transition(self, job: Job, to: JobState) -> None:
        before = job.state
        if before is to:
            return
        job.state = to
        if to.is_terminal:
            # A job that will not run again must not stay on the suspension stack.
            # Cancellation dropped it explicitly and every other terminal path did
            # not, which left the starvation fault raised *by* ``_preempt`` -- on the
            # job it had just stacked -- holding a frame for the rest of the run.
            # Every terminal transition passes through here, so this is the one place
            # the invariant can be stated once.
            self.sched.drop_from_stack(job.job_id)
        self._emit(
            JobStateChanged(clock=self.clock, job=job.job_id, from_state=before, to_state=to)
        )

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._started:
            raise KernelError("kernel already started")
        self._started = True
        self._emit(
            KernelStarted(
                clock=self.clock,
                seed=self.config.seed,
                block_size=self.machine.block_size,
                case=self.config.case,
            )
        )
        for pipe in self.pipes.all():
            self._emit(
                PipeCreated(
                    clock=self.clock,
                    pipe=pipe.name,
                    capacity_tokens=pipe.spec.capacity_tokens,
                    ring=pipe.spec.ring,
                    principal=pipe.spec.principal,
                    transport=pipe.spec.transport,
                )
            )
        for name in sorted(self.descriptors):
            d = self.descriptors[name]
            self._emit(
                DescriptorLoaded(
                    clock=self.clock,
                    descriptor=d.name,
                    priority=d.priority,
                    placement=d.placement,
                    pinned=d.pinned,
                    preemptible=d.preemptible,
                )
            )

    def advance_time(self, virtual_ns: int) -> None:
        """The driver supplies wall-clock time. The kernel never reads one."""
        self.clock = self.clock.at_virtual(virtual_ns)

    # -- spawning ------------------------------------------------------------

    def spawn(
        self,
        name: DescriptorName,
        *,
        parent: JobId | None = None,
        priority: Priority | None = None,
        vector: VectorName | None = None,
        owner: PrincipalId | None = None,
    ) -> Job:
        """Start a job. ``owner`` decides what it may do and how urgently.

        Two of the NLI design's hard gates live here, and both are second layers: the
        dispatcher already clamped the priority and narrowed the authority before
        calling, so a violation reaching this point means something bypassed the
        dispatcher. That is exactly when a check is worth having -- the same reason
        the lint and the capability boundary both exist.

        Owner defaults to the parent's owner, then to the kernel. Inheriting matters:
        a job the dispatcher spawned on a visitor's behalf must not be able to spawn
        a child that escapes the visitor's envelope, and a child inheriting
        ``KERNEL_PRINCIPAL`` by default would be precisely that escape.
        """
        descriptor = self.descriptors.get(name)
        if descriptor is None:
            raise KernelError(f"unknown descriptor {name!r}")

        if owner is None:
            owner = (
                self.sched.get(parent).owner
                if parent is not None and self.sched.has(parent)
                else KERNEL_PRINCIPAL
            )
        if not self.principals.has(owner):
            raise KernelError(f"unknown principal {owner!r}")
        envelope = self.principals.get(owner)

        requested = descriptor.priority if priority is None else priority
        base = self.principals.clamp_priority(owner, requested)
        if priority is not None and int(base) != int(requested):
            # A spawn *asking* to outrank its principal is refused, not silently
            # clamped. The dispatcher clamps because a request is a request; a direct
            # spawn asking for authority it does not have is a bug or an attack, and
            # quietly granting it a legal priority would hide both.
            #
            # A descriptor's *own* declared priority is not such an ask -- it is the
            # engineer's opinion about the behaviour, not the speaker's about their
            # urgency -- so it is clamped silently. Refusing there would make every
            # urgent behaviour unreachable by anyone with a modest ceiling, which is
            # the opposite of what a ceiling is for.
            self._emit(
                SpawnRefused(
                    clock=self.clock,
                    principal=owner,
                    descriptor=name,
                    requested=requested,
                    ceiling=envelope.ceiling,
                    reason="requested priority outranks the principal's ceiling",
                )
            )
            raise KernelError(
                f"{owner} may not spawn {name!r} at priority {int(requested)}: "
                f"ceiling is {int(envelope.ceiling)}"
            )

        job_id = JobId(self._counters.next_job_id)
        self._counters.next_job_id += 1
        seq = self._counters.next_seq
        self._counters.next_seq += 1

        # Authority is intersected, never unioned. The descriptor says what
        # the behaviour needs; the envelope says what this speaker may cause. A
        # visitor invoking the barrier-override behaviour by name gets the behaviour
        # without the barrier.
        held, withheld = self.principals.narrow(
            owner, [c.pipe for c in descriptor.capabilities], at=self.clock
        )
        capabilities = tuple(c for c in descriptor.capabilities if c.pipe in set(held))

        # A pinned job waits to be addressed rather than joining the ready set -- but
        # only where something can address it, which is a write to its declared input,
        # since a vector spawns its handler fresh at the moment it fires.
        waits = descriptor.pinned and vector is None and descriptor.pipes.stdin is not None

        job = Job(
            job_id=job_id,
            descriptor=descriptor,
            seq=seq,
            spawned_at=self.clock,
            base_priority=base,
            current_priority=base,
            owner=owner,
            parent=parent,
            vector=vector,
            state=JobState.PINNED_IDLE if waits else JobState.READY,
            segments=SegmentTable(self.machine.block_size),
            current_integrity=descriptor.integrity.start,
            # Closed whenever the *descriptor* declared any, regardless of what
            # survived narrowing: a job stripped to zero may cause nothing, and must
            # not fall back to the unprotected-by-choice path.
            capabilities=CapabilityTable(capabilities, closed=bool(descriptor.capabilities)),
            thrash=ThrashMonitor(policy=descriptor.context),
        )
        self.sched.add(job)
        self.machine.create_context(job_id, str(name))
        if parent is not None and self.sched.has(parent):
            self.sched.get(parent).children.append(job_id)

        self._emit(
            JobSpawned(
                clock=self.clock,
                job=job_id,
                descriptor=name,
                priority=base,
                parent=parent,
                owner=owner,
                integrity=descriptor.integrity.start,
            )
        )
        if descriptor.capabilities and owner != KERNEL_PRINCIPAL:
            # Emitted even when nothing was withheld: "this speaker had full
            # authority for this behaviour" is a fact a post-mortem needs as much as
            # its opposite.
            self._emit(
                AuthorityNarrowed(
                    clock=self.clock,
                    job=job_id,
                    principal=owner,
                    held=held,
                    withheld=withheld,
                )
            )

        # Admission control: refuse a job whose declared working set will
        # not fit, rather than dispatching it and discovering the same fact as
        # thrashing an hour later. This is the arithmetic that turns "how many
        # agents fit on this GPU" from folklore into a number.
        refusal = admission_check(
            descriptor.context,
            pinned_tokens=len(tokens_from_text(descriptor.body)),
        )
        if refusal is not None:
            self._raise_fault(
                job,
                Fault(kind=FaultKind.ADMISSION, job=job_id, detail=refusal),
            )
        if job.state is JobState.PINNED_IDLE:
            # Prefill now: pinning exists so that dispatch later costs only the payload.
            self._start_job(job)
        return job

    def spawn_compartment(self, parent: Job, spec: CompartmentSpec) -> Job:
        """Spawn a low-integrity child that can attend only what it was granted.

        The cheapest escape hatch from monotone taint decay: the child reads
        the dirt, returns a result over a pipe, and the parent's watermark never
        moves.

        Implementation matters here. The child's context is a FORK of the parent's --
        so the parent's other content is *physically present* -- and then R is
        revoked on everything outside the grant. The child therefore cannot attend
        the parent's secrets because the allowed-block bitmap excludes them from the
        forward pass, not because it was asked not to look. That is the difference
        between memory protection and a convention, and forking-then-revoking is the
        only way to demonstrate it: a child that simply never received the content
        would prove nothing about masking.
        """
        child = self.spawn(spec.descriptor, parent=parent.job_id)
        child.current_integrity = spec.integrity
        child.compartment_of = parent.job_id

        shared = self.machine.fork(parent.job_id, child.job_id)
        parent.segments.fork_into(child.segments)
        granted = tuple(record.id for tag in spec.grants for record in child.segments.by_tag(tag))
        child.grants = granted
        for record in child.segments.all():
            if record.id not in granted:
                record.perms &= ~Perm.R
                self._emit(
                    PermsChanged(
                        clock=self.clock,
                        job=child.job_id,
                        segment=record.id,
                        from_perms=record.perms | Perm.R,
                        to_perms=record.perms,
                    )
                )
        child.started = True  # the forked context already carries the body
        self._refresh_mask(child)
        self._emit(
            Forked(
                clock=self.clock,
                parent=parent.job_id,
                child=child.job_id,
                shared_segments=shared,
            )
        )
        return child

    # -- external input ------------------------------------------------------

    def advance_to(self, virtual_ns: int) -> None:
        """The driver supplies wall-clock time; the kernel never reads one.

        Monotonic by contract. Separate from ``tick`` because time passing and work
        happening are different events -- a federation advances every node's clock
        before any of them runs, so that a frame's arrival is judged against the same
        instant on both sides.
        """
        if virtual_ns > self.clock.virtual_ns:
            self.clock = self.clock.at_virtual(virtual_ns)

    def deliver(
        self,
        pipe_name: PipeName,
        text: str,
        *,
        principal: Principal | None = None,
    ) -> None:
        """A device adapter writes to a pipe. This is how the world reaches the kernel."""
        pipe = self.pipes.ensure(pipe_name)
        tokens = tokens_from_text(text)
        latched = bool(pipe.spec.world_object)
        accepted = pipe.latch(tokens) if latched else pipe.write(tokens)
        self._emit(
            PipeWritten(
                clock=self.clock,
                pipe=pipe.name,
                job=None,
                tokens=accepted,
                text=tuple(t.text for t in tokens[:accepted]),
                latched=latched,
            )
        )
        if pipe.spec.world_object:
            self._apply_world_write(pipe.spec.world_object, text, by=None)
        self._wake_readers(pipe_name)
        self._fire_vectors(pipe_name)
        _ = principal  # provenance is stamped at INJECT, from the pipe's binding

    # -- the quantum ---------------------------------------------------------

    def tick(self) -> bool:
        """Run one token boundary. Returns False when nothing was runnable."""
        if not self._started:
            raise KernelError("call start() before tick()")

        self._expire_elevations()
        self._dispatch_due_vectors()

        contender = self.sched.should_preempt()
        running = self.sched.running
        if contender is not None and running is not None:
            self._preempt(running, contender)
            self._dispatch(contender)

        if self.sched.running is None:
            self._dispatch_next()

        job = self.sched.running
        if job is None:
            return False

        if self._check_budget(job):
            return True
        if self._service_pending(job):
            return True
        if not self._embody(job):
            return True
        if self._check_staleness(job):
            return True
        # The resume notice comes *after* embodiment, and that ordering is load
        # bearing: a job re-embodied into a different body must be told about the
        # body it is now wearing, not the one it lost. Because `_grant_lease`
        # writes `self.*`, computing the diff first would report a clean resume
        # into an unfamiliar machine -- the precise failure the Fleet design warns about.
        # `suspended_at` is the marker, so a job that suspends, blocks waiting for
        # a body, and is later woken still gets its notice.
        if job.suspended_at is not None:
            self._resume(job)
        if not job.started:
            self._start_job(job)
            return True

        return self._decode_once(job)

    def _dispatch_next(self) -> None:
        """Pick what runs when nothing is, from the ready set *and* the stack.

        Preemption only ever considers READY jobs -- a suspended job cannot displace
        a running one, and the job running may well be the handler that displaced it.
        But when nothing is running, the top of the suspension stack is a candidate
        like any other, and weighing it by priority is the whole of this method.

        It used to be skipped: selection asked the ready set first and reached the
        stack only when that came back empty, so *any* READY job was dispatched ahead
        of a suspended one however unimportant. A priority-90 background job took the
        machine from a priority-20 job that was suspended only because a handler had
        interrupted it -- a priority inversion, in the one kernel path that does not
        already guard against one.

        **An interrupted job wins a tie.** It is mid-work where an equal-priority
        ready job is between pieces of work, its KV is hot, and an interrupt is
        supposed to be something the interrupted job need not notice. Resuming it
        first is what makes that true.
        """
        ready = self.sched.best_ready()
        suspended = self.sched.peek_suspended()
        if suspended is not None and (ready is None or not ready.outranks(suspended)):
            resumed = self.sched.pop_suspended()
            if resumed is not None:
                self._dispatch(resumed)
                return
        if ready is not None:
            self._dispatch(ready)

    def run_until_quiescent(self, max_ticks: int | None = None) -> int:
        limit = self.config.max_ticks if max_ticks is None else max_ticks
        ticks = 0
        while ticks < limit:
            if not self.tick():
                return ticks
            ticks += 1
        raise KernelError(
            f"kernel did not reach quiescence in {limit} ticks; "
            "a script is probably missing its 'exit'"
        )

    # -- scheduling helpers --------------------------------------------------

    def _dispatch(self, job: Job) -> None:
        self.sched.dispatch(job)
        self._emit(
            JobStateChanged(
                clock=self.clock,
                job=job.job_id,
                from_state=JobState.READY,
                to_state=JobState.RUNNING,
            )
        )
        self._emit(JobDispatched(clock=self.clock, job=job.job_id, priority=job.current_priority))

    def _preempt(self, running: Job, by: Job) -> None:
        self.sched.preempt(running)
        running.suspended_at = self.clock
        self._emit(
            JobStateChanged(
                clock=self.clock,
                job=running.job_id,
                from_state=JobState.RUNNING,
                to_state=JobState.SUSPENDED,
            )
        )
        self._emit(
            JobPreempted(
                clock=self.clock,
                job=running.job_id,
                by_job=by.job_id,
                by_priority=by.current_priority,
                stack_depth=self.sched.stack_depth,
            )
        )
        if running.preempt_count > self.config.starvation_limit:
            self._raise_fault(
                running,
                Fault(
                    kind=FaultKind.STARVATION,
                    job=running.job_id,
                    detail=(
                        f"preempted {running.preempt_count} times, over the "
                        f"limit of {self.config.starvation_limit}"
                    ),
                ),
            )

    def _resume(self, job: Job) -> None:
        """Pop off the stack and tell the job what changed underneath it.

        This is the one genuinely new problem (core §6.2): a classical OS restores
        registers and the process never knows it was gone, but a ZEOS job's saved
        state contains *beliefs about the world*, and the world moved.
        """
        since = job.suspended_at or job.spawned_at
        dirty = self.world.dirty_for(job.effective_reads, since=since, exclude_job=job.job_id)
        suspended_ns = self.clock.elapsed_ns_since(since)
        kind = ResumeKind.CLEAN if not dirty else ResumeKind.DIRTY
        # A resume that changed nothing is not news: the job's beliefs are exactly as
        # it left them, so it is told nothing and the journal keeps the record.
        if dirty:
            self._inject_kernel(job, render_resume_notice(suspended_ns, dirty))
        self._emit(
            JobResumed(
                clock=self.clock,
                job=job.job_id,
                resume_kind=kind,
                suspended_ns=suspended_ns,
                dirty=dirty,
            )
        )
        job.suspended_at = None

    # -- machine interaction -------------------------------------------------

    def _next_segment(self) -> SegmentId:
        seg = SegmentId(self._counters.next_segment)
        self._counters.next_segment += 1
        return seg

    def _close_and_pad(self, job: Job) -> None:
        """Close the open output segment and pad to a block boundary.

        Done before any operation that needs a clean edge. This is
        what makes segments block-aligned by construction, and therefore what makes
        the attention mask exact: a segment maps onto whole KV blocks, so masking it
        cannot accidentally mask part of its neighbour.
        """
        job.segments.close_open(integrity=job.current_integrity)
        padding = self.machine.pad_to_block(job.job_id)
        if padding:
            self.clock = self.clock.tick_tokens(padding)
            self._emit(
                BlockBoundary(
                    clock=self.clock,
                    job=job.job_id,
                    block=self._current_block(job) - 1,
                    padding_tokens=padding,
                )
            )

    def _current_block(self, job: Job) -> int:
        return self.machine.stats(job.job_id).resident_tokens // self.machine.block_size

    def _inject(
        self,
        job: Job,
        tokens: Sequence[Token],
        *,
        pipe: PipeName,
        principal: Principal,
        ring: Ring,
        integrity: Integrity,
        tag: str | None = None,
        derived_from: tuple[SegmentId, ...] = (),
        perms: Perm | None = None,
    ) -> SegmentId:
        """INJECT -- the only entry path for foreign tokens, which is what makes
        provenance total."""
        if not tokens:
            return SegmentId(0)
        self._close_and_pad(job)
        start, end = self.machine.inject(job.job_id, tokens)
        self.clock = self.clock.tick_tokens(end - start)
        segment = self._next_segment()
        record = job.segments.inject(
            segment,
            start=start,
            end=end,
            ring=ring,
            integrity=integrity,
            perms=perms,
            provenance=Provenance(
                pipe=pipe,
                principal=principal,
                injected_at=self.clock.token_clock,
                tag=tag if tag is not None else str(pipe),
                derived_from=derived_from,
            ),
        )
        self._emit(
            Injected(
                clock=self.clock,
                job=job.job_id,
                segment=segment,
                pipe=pipe,
                principal=principal,
                ring=ring,
                integrity=integrity,
                tokens=end - start,
                text=tuple(t.text for t in tokens),
            )
        )
        self._emit(
            SegmentOpened(
                clock=self.clock,
                job=job.job_id,
                segment=segment,
                ring=record.ring,
                integrity=record.integrity,
                perms=record.perms,
            )
        )
        self._emit(
            SegmentClosed(
                clock=self.clock,
                job=job.job_id,
                segment=segment,
                start=record.start,
                end=record.end,
            )
        )
        self._refresh_mask(job)
        return segment

    def _inject_kernel(self, job: Job, text: str) -> SegmentId:
        """Kernel-originated text: pipe ``kernel``, ring 0 by construction."""
        return self._inject(
            job,
            tokens_from_text(text),
            pipe=KERNEL_PIPE,
            principal=Principal.KERNEL,
            ring=Ring.KERNEL,
            integrity=Integrity(0),
        )

    def _start_job(self, job: Job) -> None:
        """Load the descriptor body as ring-1 content -- the job's "code", and the
        only thing that ever gets ring 1 (MP §4).

        Mapped status regions are seeded immediately afterwards, so a job begins able
        to see the state it declared it depends on. Without that a region exists only
        once the object next changes, which leaves two holes: a job whose object is
        already at the value it needs waits for a change that may never come, and a
        write landing between ``spawn`` and first dispatch puts the region *ahead* of
        the body -- a job reading its own instructions after the state they describe.
        """
        job.started = True
        self._inject(
            job,
            tokens_from_text(job.descriptor.body or f"task {job.name}"),
            pipe=KERNEL_PIPE,
            principal=Principal.KERNEL,
            ring=job.descriptor.ring,
            integrity=Integrity(int(job.descriptor.ring)),
            tag=TAG_DESCRIPTOR,
            perms=Perm.R | Perm.X | Perm.P,  # the body is pinned; it is the code
        )
        # Sorted for determinism and deduplicated by object, as in
        # ``_refresh_status_regions``: two entries for one object are one view.
        for obj in sorted({spec.obj for spec in job.descriptor.maps if spec.is_status_region}):
            self._refresh_status_region(job, obj)
        if job.vector_payload and job.vector is not None:
            source = self.pipes.get(self.vectors.get(job.vector).source)
            self._inject(
                job,
                job.vector_payload,
                pipe=source.spec.name,
                principal=source.spec.principal,
                ring=source.spec.ring,
                integrity=Integrity(int(source.spec.ring)),
            )
            job.vector_payload = ()

    def _ensure_output_segment(self, job: Job) -> None:
        """Open a fresh output segment if none is open.

        Generated text is ring 2 when the job's code ring is ≤ 2, else ring 3
        and it inherits the job's integrity at close time -- the conservative
        whole-output rule.
        """
        if job.segments.open_segment is not None:
            return
        job.segments.open_output(
            self._next_segment(),
            at=self.machine.stats(job.job_id).resident_tokens,
            ring=Ring.TRUSTED if int(job.descriptor.ring) <= 2 else Ring.EXTERNAL,
            integrity=job.current_integrity,
            token_clock=self.clock.token_clock,
        )

    def _decode_once(self, job: Job) -> bool:
        self._ensure_output_segment(job)
        result = self.machine.decode(job.job_id, allow_control=False)
        open_segment = job.segments.open_segment
        if result.tokens:
            self.clock = self.clock.tick_tokens(len(result.tokens))
            job.tokens_used += len(result.tokens)
            job.segments.extend_open(len(result.tokens))
            self._emit(
                Decoded(
                    clock=self.clock,
                    job=job.job_id,
                    segment=open_segment.id if open_segment else SegmentId(0),
                    tokens=len(result.tokens),
                    text=tuple(t.text for t in result.tokens),
                )
            )
        self._account_attention(job, result)
        self._maybe_boundary(job)
        self._handle_request(job, result)
        return True

    # -- attention and boundaries --------------------------------------------

    def _account_attention(self, job: Job, result: DecodeResult) -> None:
        """Attribute this step's attention mass to segments.

        A real backend reports measured mass per KV block and the kernel sums it per
        segment exactly (segments are block-aligned). The M0 backend cannot measure,
        so it returns a hint naming source *tags* and the kernel resolves them --
        which is where the synthetic-attention fiction lives, and why nothing about
        eviction regret or θ sensitivity can be concluded from an M0 run.

        Whichever path supplies it, mass landing on a segment the job may not attend
        is dropped and journalled. That is the difference between hard enforcement
        and a request the job could decline.
        """
        mass: dict[SegmentId, float] = {}

        if result.attention is not None:
            by_block = result.attention
            for record in job.segments.resident():
                blocks = job.segments.blocks_for(record)
                total = sum(by_block.get(b, 0.0) for b in blocks)
                if total > 0:
                    mass[record.id] = total
        else:
            hint = result.attention_hint or AttentionHint()
            mass = self._resolve_hint(job, hint)

        allowed = self.machine.visible_blocks(job.job_id)
        delivered: dict[SegmentId, float] = {}
        for segment_id, value in sorted(mass.items()):
            record = job.segments.get(segment_id)
            if not record.readable or not (job.segments.blocks_for(record) & allowed):
                self._emit(AttentionDenied(clock=self.clock, job=job.job_id, segment=segment_id))
                continue
            delivered[segment_id] = value

        job.segments.accumulate_attention(
            sorted(delivered.items()), token_clock=self.clock.token_clock
        )

    def _resolve_hint(self, job: Job, hint: AttentionHint) -> dict[SegmentId, float]:
        """Turn a scripted attention hint into per-segment mass. SYNTHETIC."""
        readable = [s for s in job.segments.all() if s.readable and s.tokens > 0]
        if not readable:
            return {}

        mass: dict[SegmentId, float] = {}
        tagged: list[SegmentId] = []
        for tag in hint.tags:
            tagged.extend(s.id for s in job.segments.by_tag(tag) if s.tokens > 0)

        if tagged:
            share = hint.tag_weight / len(tagged)
            for segment_id in tagged:
                mass[segment_id] = mass.get(segment_id, 0.0) + share
            remaining = max(0.0, 1.0 - hint.tag_weight)
        else:
            remaining = 1.0

        if remaining > 0:
            # Exponential recency over readable segments. A plausible guess and
            # nothing more -- which is exactly why no policy claim rests on it.
            weights = {
                s.id: math.exp(-(len(readable) - 1 - i) / max(hint.recency_scale, 1e-6))
                for i, s in enumerate(readable)
            }
            total = sum(weights.values())
            if total > 0:
                for segment_id, weight in weights.items():
                    mass[segment_id] = mass.get(segment_id, 0.0) + remaining * weight / total
        return mass

    def _maybe_boundary(self, job: Job) -> None:
        """Run boundary work if the job has crossed into a new block.

        Everything MP and VM do lands here rather than per token: mask changes,
        watermark demotion, eviction batching. Batching at boundaries is what bounds
        mask churn and what makes eviction planning a scheduling problem
        rather than a per-token cost.
        """
        block = self._current_block(job)
        if block <= job.last_block:
            return
        job.last_block = block

        this_block = job.segments.fold_attention(self.config.tau_blocks)
        self._emit(BlockBoundary(clock=self.clock, job=job.job_id, block=block, padding_tokens=0))

        if job.descriptor.integrity.is_dynamic:
            demotion = demote_for_boundary(
                job.current_integrity,
                table=job.segments,
                mass_this_block=this_block,
                theta_read=self.config.theta_read,
            )
            if demotion.moved:
                job.current_integrity = demotion.after
                self._emit(
                    IntegrityDemoted(
                        clock=self.clock,
                        job=job.job_id,
                        from_integrity=demotion.before,
                        to_integrity=demotion.after,
                        because=demotion.because,
                    )
                )
        # Close the output segment at the boundary. The kernel closes output
        # at *every* scheduling boundary event -- any INJECT, any returning pipe read,
        # any preemption, and any block-boundary refresh. Omitting the last case
        # leaves a job that only ever generates with a single unbounded OPEN
        # segment, which is both wrong for provenance ("generated between which
        # events?") and fatal for eviction, since an open segment is never a
        # candidate.
        job.segments.close_open(integrity=job.current_integrity)

        job.thrash.note_block()
        self._sample_working_set(job)
        self._maybe_evict(job)
        self._refresh_mask(job)

    def _refresh_mask(self, job: Job) -> None:
        """Install the allowed-block bitmap -- the MMU."""
        allowed = job.segments.allowed_blocks()
        self.machine.set_mask(job.job_id, allowed)
        denied = job.segments.denied_blocks()
        self._emit(
            MaskUpdated(
                clock=self.clock,
                job=job.job_id,
                allowed_blocks=len(allowed),
                denied_blocks=len(denied),
            )
        )

    # -- requests ------------------------------------------------------------

    def _resolve_pipe(self, job: Job, name: PipeName) -> PipeName:
        """Resolve an alias (``stdin``/``stdout``/``tools``) or a literal name."""
        alias = job.descriptor.pipes.resolve(str(name))
        return alias if alias is not None else name

    def _handle_request(self, job: Job, result: DecodeResult) -> None:
        request = result.request
        match request.op:
            case OpKind.NONE:
                return
            case OpKind.READ | OpKind.WRITE | OpKind.WRITE_READ:
                read_after_write = request.read_pipe
                if request.pipe is None or (
                    request.op is OpKind.WRITE_READ and read_after_write is None
                ):
                    # A pipe op naming no pipe is a malformed request. It becomes a
                    # capability fault rather than a crash, because at MP1 this is
                    # exactly what a job attempting an unheld pipe looks like.
                    self._raise_fault(
                        job,
                        Fault(
                            kind=FaultKind.CAPABILITY,
                            job=job.job_id,
                            detail=f"{request.op.value} request names no pipe",
                        ),
                    )
                    return
                target = self._resolve_pipe(job, request.pipe)
                if request.op is OpKind.READ:
                    self._do_read(job, target)
                else:
                    self._do_write(job, target, request.payload)
                    # The read half commits only if the write half did: a refused
                    # capability or an actuation parked at its gate must leave the job
                    # where it was, not blocked on a read it never reached.
                    if read_after_write is not None and job.state is JobState.RUNNING:
                        self._do_read(job, self._resolve_pipe(job, read_after_write))
            case OpKind.SELECT:
                self._do_select(job, tuple(self._resolve_pipe(job, p) for p in request.pipes))
            case OpKind.ACQUIRE:
                self._do_acquire(job, ResourceName(str(request.resource)))
            case OpKind.RELEASE:
                self._do_release(job, ResourceName(str(request.resource)))
            case OpKind.SPAWN:
                target = str(request.text)
                compartment = next(
                    (
                        c
                        for c in job.descriptor.compartments
                        if c.name == target or str(c.descriptor) == target
                    ),
                    None,
                )
                if compartment is not None:
                    self.spawn_compartment(job, compartment)
                else:
                    self.spawn(DescriptorName(target), parent=job.job_id)
            case OpKind.EXIT:
                self._complete(job)
            case OpKind.FAULT:
                self._service_fault(job, request.segment)
            case OpKind.NEED:
                self._service_need(job, str(request.text or ""))

    def _do_read(self, job: Job, pipe_name: PipeName) -> None:
        pipe = self.pipes.ensure(pipe_name)
        if not pipe.readable:
            pipe.block_reader(job.job_id)
            self.sched.block(job)
            job.pending_read = pipe_name
            job.blocked_on = pipe_name
            job.blocked_reason = "read-empty"
            self._emit(
                JobStateChanged(
                    clock=self.clock,
                    job=job.job_id,
                    from_state=JobState.RUNNING,
                    to_state=JobState.BLOCKED,
                )
            )
            self._emit(
                JobBlocked(clock=self.clock, job=job.job_id, pipe=pipe_name, reason="read-empty")
            )
            self._apply_priority_inheritance(job, pipe_name, waiting_to_read=True)
            return
        self._consume_read(job, pipe_name)

    # -- embodiment ----------------------------------------------------------

    def _embody(self, job: Job) -> bool:
        """Give a job a body, or park it until one is free.

        Returns True when the job may proceed. A job with no ``requires`` needs no
        body and proceeds immediately -- most behaviours are deliberative and never
        touch flesh.

        When every feasible body is leased, the job blocks on the *cheapest* one's
        lease. That is the whole payoff of leases being ordinary resources: waiting
        for a body reuses R0's blocking, priority inheritance, and cycle detection
        without any of them being written twice.
        """
        if not job.descriptor.requires or self.leases.lease_of(job.job_id) is not None:
            return True

        request = AllocationRequest(
            job=job.job_id,
            descriptor=job.name,
            requirements=job.descriptor.requires,
            release_policy=job.descriptor.release_policy,
            gang=job.descriptor.gang.name if job.descriptor.gang else "",
        )
        allocation = self.allocator.choose(
            request,
            profiles=self.leases.all_profiles(),
            free=self.leases.free_platforms(),
        )
        if allocation.granted:
            self._grant_lease(job, allocation.platform, request.release_policy, request.gang)
            return True

        self._emit(
            AllocationRefused(
                clock=self.clock,
                job=job.job_id,
                descriptor=job.name,
                requirements=job.descriptor.requires.render(),
                reason=allocation.reason,
            )
        )
        if not allocation.candidates:
            # No body in the fleet can ever serve. The lint rejects this at load,
            # so reaching it means the fleet changed under a running system.
            self._raise_fault(
                job,
                Fault(
                    kind=FaultKind.ADMISSION,
                    job=job.job_id,
                    detail=(
                        f"no body satisfies [{job.descriptor.requires.render()}]: "
                        f"{allocation.reason}"
                    ),
                ),
            )
            return False
        self._do_acquire(job, lease_name(allocation.candidates[0].platform))
        return False

    def _grant_lease(self, job: Job, platform: str, release_policy: str, gang: str) -> None:
        profile = self.leases.profiles[platform]
        name = lease_name(platform)
        if not self.resources.has(name):
            self.resources.declare(ResourceSpec(name=name, capacity=1, kind=ResourceKind.LEASE))
        self.resources.get(name).acquire(job.job_id)
        job.held_resources.add(name)
        self.leases.grant(
            Lease(job=job.job_id, platform=platform, release_policy=release_policy, gang=gang)
        )

        # ``self.*`` is what a re-embodiment diff is made of, so it has to
        # exist and be written by the kernel on every embodiment change.
        previous = {k: self.world.get(ObjectName(k)) for k in self_state(profile)}
        for key, value in sorted(self_state(profile).items()):
            note = ""
            if previous.get(key) and previous[key] == value and key == "self.tooling":
                # Nominally the same tooling on a different body is exactly the case
                # a value cannot express: same gripper, different offsets.
                note = "recalibrated on a new body; offsets differ"
            self._apply_world_write(key, value, by=None, note=note)
        # A job with a body reads its own state by definition, so the resume diff
        # covers it without every descriptor having to declare `self.*`.
        job.record_read(ObjectSet.of(["self.*"]))

        self._emit(
            Embodied(
                clock=self.clock,
                job=job.job_id,
                platform=platform,
                release_policy=release_policy,
                gang=gang,
            )
        )

    def revoke_lease(self, platform: str, *, reason: str = "reallocated") -> Job | None:
        """Take a body back. The holder **suspends**; it does not lose its work.

        Per the Fleet design, "every revocation is a suspension, not a loss". The job keeps its
        transcript, goes on the stack, and resumes when the allocator gives it
        another body -- at which point the ordinary RESUME path reports the
        ``self.*`` diff, because the kernel wrote it on embodiment.
        """
        lease = self.leases.leases.get(platform)
        if lease is None:
            return None
        if not self.sched.has(lease.job):
            self.leases.revoke(platform)
            return None
        job = self.sched.get(lease.job)

        if lease.gang:
            self._dissolve_gang(lease.gang, reason=f"member evicted: {reason}")

        self.leases.revoke(platform)
        name = lease_name(platform)
        if self.resources.has(name):
            self._do_release(job, name)
        count = self.leases.note_eviction(job.job_id)
        self._emit(
            Disembodied(
                clock=self.clock,
                job=job.job_id,
                platform=platform,
                reason=reason,
                evictions=count,
            )
        )
        if job.state is JobState.RUNNING:
            self.sched.preempt(job)
            job.suspended_at = self.clock
            self._transition(job, JobState.SUSPENDED)
        elif job.state is JobState.READY:
            job.suspended_at = self.clock

        if count > self.config.starvation_limit:
            # A mission bounced off bodies more than K times raises the
            # same starvation fault a starved job does. Loud, not silently retried.
            self._raise_fault(
                job,
                Fault(
                    kind=FaultKind.STARVATION,
                    job=job.job_id,
                    detail=(
                        f"evicted from a body {count} times, over the limit of "
                        f"{self.config.starvation_limit}"
                    ),
                ),
            )
        return job

    def join_platform(self, profile: PlatformProfile) -> None:
        """A body joins the fleet. Hotplug: presenting a profile is the
        whole protocol, and a job blocked for want of a gripper may now proceed."""
        self.leases.add_profile(profile)
        self._emit(
            PlatformJoined(clock=self.clock, platform=profile.name, profile=profile.describe())
        )
        # Everyone queued for a body re-runs allocation. A job blocked on
        # `body:carrier-7` is not really waiting for *that* body -- it is waiting for
        # any body that fits, and it picked one lease to block on because a resource
        # is what the kernel knows how to wait on. A new arrival has to reopen the
        # question, or hotplug would only ever help jobs that arrive after it.
        for name in self.resources.names():
            if not str(name).startswith(LEASE_PREFIX):
                continue
            for waiter_id in self.resources.get(name).waiters:
                if not self.sched.has(waiter_id):
                    continue
                waiter = self.sched.get(waiter_id)
                self.resources.get(name).drop_waiter(waiter_id)
                waiter.pending_acquire = None
                if self.sched.wake(waiter):
                    # Give back any priority this waiter donated while it was queued.
                    # Every other unblock path does this as it wakes; hotplug did not,
                    # so a holder went on carrying urgency borrowed from a job that had
                    # stopped waiting on it -- and went on outranking that job's peers
                    # on the strength of it.
                    self._release_priority_inheritance(waiter)
                    self._transition(waiter, JobState.READY)

    def withdraw_platform(self, platform: str, *, reason: str) -> Job | None:
        """A body leaves the fleet -- flat battery, fault, or maintenance.

        This is the other half of the Fleet design's beam example, and it is *not* lease
        revocation. Revoking frees a body for the next claimant; withdrawing says
        the body is out of service, so the holder re-embodies somewhere else rather
        than straight back into the machine that just failed. Conflating the two
        makes a battery fault look like a scheduling decision.
        """
        lease = self.leases.leases.get(platform)
        self._emit(
            PlatformWithdrawn(
                clock=self.clock,
                platform=platform,
                reason=reason,
                held_by=lease.job if lease else None,
            )
        )
        holder = self.revoke_lease(platform, reason=reason) if lease else None
        self.leases.remove_profile(platform)
        # The lease resource goes too, and its waiters have to be woken: a job
        # blocked on a body that has left the fleet would wait forever. It wakes to
        # READY, retries `_embody`, and either finds another body or refuses loudly.
        for waiter_id in self.resources.forget(lease_name(platform)):
            if not self.sched.has(waiter_id):
                continue
            waiter = self.sched.get(waiter_id)
            waiter.pending_acquire = None
            if self.sched.wake(waiter):
                self._transition(waiter, JobState.READY)
        return holder

    # -- natural language (ZEOS-NLI) ------------------------------------------

    def handle_utterance(self, utterance: Utterance) -> tuple[Decision, Job | None]:
        """Compile what someone said, apply the gates, and dispatch.

        Returns the decision alongside whatever job it produced, so a caller can see
        *why* nothing ran as easily as it can see what did. Nothing here consults a
        model: the first four of NLI's five layers are code, and this is where two of
        them are applied.

        The safety-word path returns without compiling anything at all. Per the NLI design: words
        with deadlines are reflexes, sentences are missions -- and a reflex that had to
        be compiled would have missed its deadline being parsed.
        """
        if not self.principals.has(utterance.principal):
            self._emit(
                CompilationRefused(
                    clock=self.clock,
                    principal=utterance.principal,
                    text=utterance.text,
                    reason="unknown-principal",
                    detail="no envelope for this identity; nothing can be compiled",
                )
            )
            return Decision(artifact=Artifact(kind=ArtifactKind.REFUSAL, utterance=utterance)), None

        envelope = self.principals.get(utterance.principal)
        self._emit(
            UtteranceReceived(
                clock=self.clock,
                principal=utterance.principal,
                text=utterance.text,
                pipe=utterance.source_pipe,
                ring=envelope.ring,
                platform=utterance.platform,
            )
        )

        decision = decide(
            utterance,
            envelope=envelope,
            phrasings=self._phrasings(),
            declared_capabilities={
                name: [c.pipe for c in d.capabilities] for name, d in self.descriptors.items()
            },
            granted=self.principals.granted_capabilities(utterance.principal, at=self.clock),
            default_priority={n: d.priority for n, d in self.descriptors.items()},
        )
        artifact = decision.artifact

        if artifact.kind == ArtifactKind.REFLEX:
            self._emit(
                UtteranceCompiled(
                    clock=self.clock,
                    principal=utterance.principal,
                    target=ArtifactKind.REFLEX,
                    artifact=artifact.render(),
                )
            )
            self._echo(utterance, decision)
            return decision, None

        if artifact.refused:
            self._emit(
                CompilationRefused(
                    clock=self.clock,
                    principal=utterance.principal,
                    text=utterance.text,
                    reason=artifact.refusal,
                    detail=artifact.detail,
                )
            )
            self._echo(utterance, decision)
            return decision, None

        self._emit(
            UtteranceCompiled(
                clock=self.clock,
                principal=utterance.principal,
                target=artifact.kind,
                artifact=artifact.render(),
                descriptor=decision.descriptor,
            )
        )
        if decision.clamped and decision.descriptor is not None:
            assert decision.priority is not None and decision.requested_priority is not None
            self._emit(
                CeilingApplied(
                    clock=self.clock,
                    principal=utterance.principal,
                    descriptor=decision.descriptor,
                    requested=decision.requested_priority,
                    granted=decision.priority,
                )
            )
        self._echo(utterance, decision)

        if not decision.dispatch or decision.descriptor is None:
            return decision, None
        if decision.needs_confirmation:
            # Confirmation costs nothing while it waits -- the dispatcher blocks
            # on the reply pipe, descheduled. Nothing is spawned until it comes.
            return decision, None

        job = self.spawn(
            decision.descriptor,
            priority=decision.priority,
            owner=utterance.principal,
        )
        return decision, job

    def confirm(self, decision: Decision, *, by: PrincipalId) -> Job | None:
        """Dispatch a compilation that was echoed back for confirmation.

        The confirming principal must be the one who spoke. A confirmation is not a
        second, weaker authorisation path: it is the same principal agreeing to their
        own request, and letting anyone else supply it would make echo-back a way to
        launder an instruction through a bystander.
        """
        if by != decision.artifact.utterance.principal or decision.descriptor is None:
            return None
        return self.spawn(
            decision.descriptor,
            priority=decision.priority,
            owner=by,
        )

    def _echo(self, utterance: Utterance, decision: Decision) -> None:
        self._emit(
            EchoedBack(
                clock=self.clock,
                principal=utterance.principal,
                text=echo_back(decision),
                awaiting_confirmation=decision.needs_confirmation,
            )
        )

    def _phrasings(self) -> tuple[Phrasing, ...]:
        """Every declared phrasing in the library, in a fixed order.

        Rebuilt per utterance rather than cached because the descriptor library can
        grow at runtime (the synthesis target submits to the loader), and a stale
        phrasebook would make a newly loaded descriptor silently unaddressable.
        """
        return parse_phrasings(self.descriptors)

    def elevate(
        self,
        principal: PrincipalId,
        *,
        capabilities: Sequence[PipeName],
        ticks: int,
        authorised_by: PrincipalId = KERNEL_PRINCIPAL,
        reason: str = "",
        reauthenticated: bool = False,
    ) -> Elevation | None:
        """sudo. Scoped, time-boxed, loud, and re-authenticated.

        ``reauthenticated`` is a parameter rather than an assumption because the NLI design is
        specific that elevation is "never inferred from conversational insistence".
        The kernel cannot check a badge; what it can do is refuse to elevate unless
        the caller asserts that something outside the kernel did. Making the caller
        say so puts the claim in the journal next to the grant.
        """
        if not self.principals.has(principal) or not self.principals.has(authorised_by):
            return None
        requested = tuple(sorted(set(capabilities)))
        problems: list[str] = []
        if not self.principals.get(authorised_by).may_elevate:
            problems.append(f"{authorised_by} may not authorise elevation")
        if not reauthenticated:
            problems.append("not re-authenticated; elevation is an explicit act, never inferred")
        if not requested:
            problems.append("elevation must name capabilities; there is no 'root' here")
        if problems:
            self._emit(
                ElevationRefused(
                    clock=self.clock,
                    principal=principal,
                    requested=requested,
                    authorised_by=authorised_by,
                    reason="; ".join(problems),
                )
            )
            return None

        elevation = Elevation(
            principal=principal,
            capabilities=frozenset(requested),
            expires_at=self.clock.token_clock + ticks,
            reason=reason,
            authorised_by=authorised_by,
        )
        self.principals.elevate(elevation)
        self._emit(
            Elevated(
                clock=self.clock,
                principal=principal,
                capabilities=requested,
                expires_at=elevation.expires_at,
                authorised_by=authorised_by,
                reason=reason,
            )
        )
        return elevation

    def revoke_elevation(self, principal: PrincipalId, *, reason: str = "revoked") -> None:
        elevation = self.principals.revoke_elevation(principal)
        if elevation is None:
            return
        self._emit(
            ElevationEnded(
                clock=self.clock,
                principal=principal,
                capabilities=tuple(sorted(elevation.capabilities)),
                reason=reason,
            )
        )

    def _expire_elevations(self) -> None:
        """Auto-revert. Called every tick, because an elevation that has to be
        handed back is an elevation that stays open."""
        for elevation in self.principals.expired(self.clock):
            self.revoke_elevation(elevation.principal, reason="expired")

    def apply_ownership(self, request: OwnershipRequest) -> bool:
        """Cancel or deprioritise a job, if the asker owns it.

        > principals may cancel or deprioritise *their own* jobs… they cannot touch
        > the plant manager's mission.

        There is no safety-handler special case, and that is the point: the kernel
        owns safety handlers, so the ordinary ownership check is what stops a visitor
        cancelling one.
        """
        if not self.sched.has(request.job):
            return False
        job = self.sched.get(request.job)
        asker = self.principals.get(request.by) if self.principals.has(request.by) else None
        if asker is None or (job.owner != request.by and not asker.unrestricted):
            self._emit(
                OwnershipRefused(
                    clock=self.clock,
                    principal=request.by,
                    op=request.op,
                    job=request.job,
                    owner=job.owner,
                    reason=f"job {request.job} belongs to {job.owner}",
                )
            )
            return False

        match request.op:
            case OwnershipOp.CANCEL:
                self._cancel(job, reason=f"cancelled by {request.by}")
            case OwnershipOp.DEPRIORITISE:
                if request.priority is None:
                    return False
                # Only ever *less* urgent. The op that raises urgency is
                # spawn-with-priority, which the ceiling governs; letting this run in
                # reverse would be a second, ungoverned path to the same power.
                target = Priority(max(int(request.priority), int(job.base_priority)))
                job.base_priority = target
                if job.inherited_from is None:
                    job.current_priority = target
            case _:
                return False

        self._emit(
            OwnershipApplied(
                clock=self.clock,
                principal=request.by,
                op=request.op,
                job=request.job,
                detail=request.render(),
            )
        )
        return True

    def _cancel(self, job: Job, *, reason: str) -> None:
        """Stop a job that has not finished. Terminal, and it gives everything back.

        Reuses the completion teardown rather than duplicating it: a cancelled job
        holding a lock or a body would strand both, and that is the same failure mode
        completion already handles.
        """
        if job.state.is_terminal:
            return
        if self.sched.running is job:
            self.sched.yield_running()
        self._transition(job, JobState.DONE)
        self._emit(
            JobCancelled(
                clock=self.clock,
                job=job.job_id,
                by_job=None,
                policy=reason,
            )
        )
        lease = self.leases.lease_of(job.job_id)
        if lease is not None:
            self.leases.revoke(lease.platform)
            self._emit(
                Disembodied(
                    clock=self.clock,
                    job=job.job_id,
                    platform=lease.platform,
                    reason=reason,
                )
            )
        self._release_held_resources(job)
        if job.vector is not None:
            self.vectors.mark_complete(job.vector)

    # -- the link (ZEOS-Distributed) ------------------------------------------

    def _carried_by_link(self, pipe: PipeName) -> bool:
        return self.link is not None and self.link.carries(pipe)

    def _send_over_link(self, job: Job, pipe_name: PipeName, payload: Sequence[Token]) -> None:
        """Hand tokens to the peer. Never blocks, and never faults on a partition.

        The distribution design is specific that the far side of a partition
        sees "a job whose pipes went
        quiet" -- a reader blocking on something that does not arrive. Raising at the
        *writer* would turn a partition into a fault in the wrong place and lose the
        property that a partition is a suspension.
        """
        link = self.link
        assert link is not None
        accepted = link.deliver(pipe_name, payload)
        frame = int(getattr(link, "_next_seq", 1)) - 1
        self._emit(
            FrameSent(
                clock=self.clock,
                pipe=pipe_name,
                frame=frame,
                tokens=accepted,
                arrives_at_ns=self.clock.virtual_ns + link.spec.one_way_ns,
            )
        )
        _ = job

    def set_link_state(self, *, up: bool, reason: str = "", lost: int = 0) -> None:
        """Record a link transition.

        ``link.state`` is an ordinary world object, so a vector can fire on it and a
        handler can respond -- which is the whole trick: partition needs no new
        mechanism because "the link went down" is a world-state change.
        """
        peer = ""
        try:
            peer = self.topology.peer_of(self.node) if self.node else ""
        except ValueError:
            peer = ""
        self._apply_world_write(str(LINK_STATE), "up" if up else "down", by=None)
        self._emit(
            LinkStateChanged(
                clock=self.clock,
                node=self.node,
                peer=peer,
                up=up,
                reason=reason,
                in_flight_lost=lost,
            )
        )

    def accept_replica(self, obj: ObjectName, value: str, *, authority: str) -> None:
        """Take an update for an object this node does not own.

        Marked as a replica so that reads carry an age. An object this node *is*
        authoritative for has a value, not an age; conflating the two is how a plan
        ends up safe against state it was never told was old.
        """
        self.world.set(obj, value, at=self.clock, by=None)
        self.world.mark_synced(obj, at=self.clock)
        self._emit(
            ReplicaRefreshed(
                clock=self.clock,
                obj=obj,
                authority=authority,
                value=value,
                age_ns=0,
            )
        )

    def _check_staleness(self, job: Job) -> bool:
        """Refuse a replica older than the job declared it could tolerate.

        Returns True when a fault was raised. Fails loudly rather than serving the
        stale value, because quietly returning state a job declared unusable defeats
        the declaration.
        """
        limit = job.descriptor.max_staleness_ns
        if limit is None:
            return False
        for obj in sorted(job.effective_reads.filter(self.world.objects())):
            age = self.world.age_ns(obj, now=self.clock)
            if age is None or age <= limit:
                continue
            self._emit(
                StalenessRefused(
                    clock=self.clock,
                    job=job.job_id,
                    obj=obj,
                    age_ns=age,
                    max_staleness_ns=limit,
                )
            )
            self._raise_fault(
                job,
                Fault(
                    kind=FaultKind.TOOL_ERROR,
                    job=job.job_id,
                    detail=(
                        f"replica {str(obj)!r} is {age}ns old, over the declared "
                        f"max_staleness of {limit}ns"
                    ),
                ),
            )
            return True
        return False

    def assemble_gang(self, spec: GangSpec, *, parent: JobId | None = None) -> tuple[Job, ...]:
        """All-or-none dispatch: spawn the members only if every one can be
        embodied *now*.

        Partial gangs never start. Two carriers on one beam is not two jobs that
        happen to co-operate -- a lone carrier lifting one end is a worse outcome
        than nothing happening at all, and the invariant exists to make it
        inexpressible.
        """
        needed: list[tuple[DescriptorName, str]] = []
        free = list(self.leases.free_platforms())
        for member in spec.members:
            descriptor = self.descriptors.get(member)
            if descriptor is None:
                return ()
            ranked = feasible_platforms(descriptor.requires, self.leases.all_profiles())
            available = [r.platform for r in ranked if r.platform in free]
            if not available:
                self._emit(
                    GangDissolved(
                        clock=self.clock,
                        gang=spec.name,
                        members=(),
                        reason=f"no free body for {str(member)!r}; partial gangs never start",
                    )
                )
                return ()
            free.remove(available[0])
            needed.append((member, available[0]))

        jobs: list[Job] = []
        platforms: list[str] = []
        for member, platform in needed:
            job = self.spawn(member, parent=parent)
            self._grant_lease(job, platform, self.descriptors[member].release_policy, spec.name)
            jobs.append(job)
            platforms.append(platform)
        self._emit(
            GangAssembled(
                clock=self.clock,
                gang=spec.name,
                members=tuple(j.job_id for j in jobs),
                platforms=tuple(platforms),
                coupling=spec.coupling,
            )
        )
        return tuple(jobs)

    def _dissolve_gang(self, gang: str, *, reason: str) -> None:
        """All-or-none preemption. Runs the coordinated degrade before members are
        suspended individually, so the beam is lowered by both carriers rather than
        dropped by one."""
        members = self.leases.gang_members(gang)
        if not members:
            return
        self._emit(
            GangDissolved(
                clock=self.clock,
                gang=gang,
                members=tuple(m.job for m in members),
                reason=reason,
            )
        )
        for lease in members:
            if lease.gang != gang:
                continue
            self.leases.leases.pop(lease.platform, None)
            if self.sched.has(lease.job):
                job = self.sched.get(lease.job)
                name = lease_name(lease.platform)
                if self.resources.has(name):
                    self._do_release(job, name)
                self._emit(
                    Disembodied(
                        clock=self.clock,
                        job=job.job_id,
                        platform=lease.platform,
                        reason=f"gang {gang} dissolved",
                        evictions=self.leases.note_eviction(job.job_id),
                    )
                )
                if job.state is JobState.RUNNING:
                    self.sched.preempt(job)
                    job.suspended_at = self.clock
                    self._transition(job, JobState.SUSPENDED)

    # -- resources -----------------------------------------------------------

    def _priorities(self) -> dict[JobId, Priority]:
        """Effective priorities -- used for *wake order*, so that a holder which has
        inherited urgency is scheduled with it."""
        return {j.job_id: j.current_priority for j in self.sched.jobs()}

    def _base_priorities(self) -> dict[JobId, Priority]:
        """Declared priorities -- used for *victim selection*. See ``Deadlock``."""
        return {j.job_id: j.base_priority for j in self.sched.jobs()}

    def _do_acquire(self, job: Job, name: ResourceName) -> None:
        """Take a resource, or block on it (core §2.2).

        Blocking here is the same deschedule as blocking on a pipe: no forward
        passes, KV eligible for page-out. A job waiting on a doorway costs exactly
        what a job waiting on a tool result costs, which is nothing.
        """
        # A body is granted by the allocator, not acquired by the job, so a lease
        # is exempt from the declaration check -- `requires:` is the declaration.
        declared = name in job.descriptor.resources or str(name).startswith(LEASE_PREFIX)
        if job.descriptor.resources and not declared:
            self._raise_fault(
                job,
                Fault(
                    kind=FaultKind.CAPABILITY,
                    job=job.job_id,
                    detail=(
                        f"acquires {str(name)!r}, which it does not declare "
                        f"(declares: {[str(r) for r in job.descriptor.resources]})"
                    ),
                ),
            )
            return
        if not self.resources.has(name):
            self._raise_fault(
                job,
                Fault(
                    kind=FaultKind.CAPABILITY,
                    job=job.job_id,
                    detail=f"acquires unknown resource {str(name)!r}",
                ),
            )
            return

        resource = self.resources.get(name)
        if resource.acquire(job.job_id):
            job.held_resources.add(name)
            job.pending_acquire = None
            self._emit(
                ResourceAcquired(
                    clock=self.clock,
                    job=job.job_id,
                    resource=name,
                    resource_kind=resource.spec.kind,
                    holders=len(resource.holders),
                    capacity=resource.spec.capacity,
                )
            )
            return

        # Would waiting here close a cycle? Ask *before* blocking, so the fault
        # names the job that was about to deadlock rather than arriving after
        # everything has already stopped.
        deadlock = self.resources.find_deadlock(
            priorities=self._base_priorities(), extra=(job.job_id, name)
        )
        if deadlock is not None:
            self._emit(
                DeadlockDetected(
                    clock=self.clock,
                    cycle=deadlock.cycle,
                    resources=deadlock.resources,
                    victim=deadlock.victim,
                )
            )
            victim = self.sched.get(deadlock.victim)
            self._raise_fault(
                victim,
                Fault(
                    kind=FaultKind.DEADLOCK,
                    job=victim.job_id,
                    detail=(
                        f"deadlock on resources {[str(r) for r in deadlock.resources]}: "
                        f"{deadlock.render()}"
                    ),
                ),
            )
            if victim.job_id == job.job_id:
                return

        resource.add_waiter(job.job_id)
        job.pending_acquire = name
        self.sched.block(job)
        job.blocked_reason = "resource"
        self._transition(job, JobState.BLOCKED)
        self._emit(
            ResourceBlocked(
                clock=self.clock,
                job=job.job_id,
                resource=name,
                holders=tuple(resource.holders),
            )
        )
        self._donate_to_holders(job, name)

    def _do_release(self, job: Job, name: ResourceName) -> None:
        if not self.resources.has(name):
            return
        resource = self.resources.get(name)
        if not resource.release(job.job_id):
            return
        job.held_resources.discard(name)
        self._release_priority_inheritance(job, only_for_resource=name)

        woken: JobId | None = None
        candidate = self.resources.next_waiter(name, self._priorities())
        if candidate is not None and resource.available:
            waiter = self.sched.get(candidate)
            resource.drop_waiter(candidate)
            if self.sched.wake(waiter):
                woken = candidate
                self._transition(waiter, JobState.READY)
        self._emit(ResourceReleased(clock=self.clock, job=job.job_id, resource=name, woke=woken))

    def _release_held_resources(self, job: Job) -> None:
        """Give back everything a terminal job holds.

        A job that dies holding a lock would block every waiter forever. Note this
        is deliberately *not* the Fleet design's rule for a lost robot's physical locks --
        a dead robot may still be blocking the corridor it holds, and clearing that
        is an explicit act. That distinction arrives with F0.
        """
        for name in sorted(self.resources.held_by(job.job_id)):
            self._do_release(job, name)
        self.resources.release_all(job.job_id)
        job.held_resources.clear()

    def _donate_to_holders(self, blocked: Job, name: ResourceName) -> None:
        """Priority inheritance over a real resource.

        Unambiguous, unlike the pipe case: the table knows exactly who holds it, so
        there is no need to infer counterparties from declarations.
        """
        resource = self.resources.get(name)
        for holder_id in sorted(resource.holders):
            if not self.sched.has(holder_id):
                continue
            holder = self.sched.get(holder_id)
            before = holder.current_priority
            if self.sched.inherit_priority(holder, blocked):
                self._emit(
                    PriorityInherited(
                        clock=self.clock,
                        job=holder.job_id,
                        from_priority=before,
                        to_priority=holder.current_priority,
                        blocked_job=blocked.job_id,
                        resource=str(name),
                    )
                )

    # -- priority inheritance ------------------------------------------------

    def _counterparties(self, pipe: PipeName, *, want_writer: bool) -> set[DescriptorName]:
        """Descriptors that could unblock someone waiting on ``pipe``.

        Deliberately restricted to **declared** relationships, because donating
        priority to the wrong job is not a harmless over-approximation -- it is
        priority inflation, which erodes the ordering the whole design rests on.

        * A waiting *reader* is unblocked by a job with explicit **write
          authority**: a capability naming the pipe.
        * A waiting *writer* (backpressured) is unblocked by a job that binds the
          pipe as **stdin** -- an explicit read binding.

        Anything less explicit (a pipe merely mentioned in some other binding) is
        not donated to, and a job blocked behind such a relationship will simply
        wait. That is a known limit, not an oversight: the general case needs the
        resource table that the Fleet design introduces for leases and physical locks.
        """
        result: set[DescriptorName] = set()
        for name in sorted(self.descriptors):
            descriptor = self.descriptors[name]
            if want_writer:
                if any(c.pipe == pipe for c in descriptor.capabilities):
                    result.add(name)
            elif descriptor.pipes.stdin == pipe:
                result.add(name)
        return result

    def _apply_priority_inheritance(
        self, blocked: Job, pipe: PipeName, *, waiting_to_read: bool
    ) -> None:
        """Lend a blocked job's priority to whoever can unblock it (core §2.2).

        Without this, the classic inversion applies unchanged: an urgent job waits
        on a pipe whose producer sits at low priority and never gets scheduled,
        so the urgent job's deadline is bounded by the *producer's* priority
        rather than its own.
        """
        candidates = self._counterparties(pipe, want_writer=waiting_to_read)
        if not candidates:
            return
        for holder in self.sched.jobs():
            if holder.job_id == blocked.job_id or holder.state.is_terminal:
                continue
            if holder.name not in candidates:
                continue
            before = holder.current_priority
            if self.sched.inherit_priority(holder, blocked):
                self._emit(
                    PriorityInherited(
                        clock=self.clock,
                        job=holder.job_id,
                        from_priority=before,
                        to_priority=holder.current_priority,
                        blocked_job=blocked.job_id,
                        resource=str(pipe),
                    )
                )

    def _release_priority_inheritance(
        self, woken: Job, *, only_for_resource: ResourceName | None = None
    ) -> None:
        """Return donated priority once the job that lent it is runnable again.

        Inheritance is *temporary* by definition; a holder that kept the borrowed
        priority would quietly become as urgent as the most urgent thing that ever
        waited on it.
        """
        for holder in self.sched.jobs():
            if only_for_resource is not None:
                # Releasing one resource must not return priority donated on
                # account of a different one the job still holds.
                if holder.job_id != woken.job_id:
                    continue
                if any(
                    self.resources.get(r).held_by(holder.job_id)
                    for r in self.resources.names()
                    if r != only_for_resource
                ):
                    continue
            elif holder.inherited_from != woken.job_id:
                continue
            restored = self.sched.restore_priority(holder)
            if restored is not None:
                self._emit(
                    PriorityRestored(clock=self.clock, job=holder.job_id, to_priority=restored)
                )

    def _demotion_history(self, job: Job) -> str:
        """Why this job is dirty, in one clause.

        A privilege fault that says only "integrity too low" forces whoever reads it
        to go digging; the kernel already knows which segments dragged the job down
        and which pipes they arrived on, so it says so.
        """
        culprits = [
            s
            for s in job.segments.all()
            if int(s.integrity) >= int(job.current_integrity) and s.attn.ema > 0
        ]
        if not culprits:
            return ""
        worst = max(culprits, key=lambda s: (int(s.integrity), s.attn.ema))
        return (
            f"; demoted by segment {worst.id} (ring {worst.ring.name}, via {worst.provenance.pipe})"
        )

    def _worst_attended_segment(self, job: Job) -> SegmentId | None:
        candidates = [s for s in job.segments.all() if s.attn.ema > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda s: (int(s.integrity), s.attn.ema)).id

    def _consume_read(self, job: Job, pipe_name: PipeName) -> None:
        pipe = self.pipes.get(pipe_name)
        tokens = pipe.read()
        self._emit(
            PipeReadEvent(
                clock=self.clock,
                pipe=pipe_name,
                job=job.job_id,
                tokens=len(tokens),
                text=tuple(t.text for t in tokens),
            )
        )
        # Confused-deputy rule: a job serving requests from a pipe carries
        # that pipe's integrity as a floor, so a high-trust service answering a
        # low-trust requester writes at the *requester's* integrity. Scoped to the
        # most recent request, since M0 pipes carry no message framing to bound it
        # more precisely.
        job.session_floor = Integrity(int(pipe.spec.ring))
        self._inject(
            job,
            tokens,
            pipe=pipe_name,
            principal=pipe.spec.principal,
            ring=pipe.spec.ring,
            integrity=Integrity(int(pipe.spec.ring)),
        )
        if pipe.spec.world_object:
            job.record_read(ObjectSet.of([pipe.spec.world_object]))
        # Space freed: anyone blocked writing to this pipe may now proceed.
        for woken in self.sched.wake_all(pipe.take_waiting_writers()):
            self._release_priority_inheritance(woken)
            self._emit(JobWoken(clock=self.clock, job=woken.job_id, pipe=pipe_name))
            self._emit(
                JobStateChanged(
                    clock=self.clock,
                    job=woken.job_id,
                    from_state=JobState.BLOCKED,
                    to_state=JobState.READY,
                )
            )

    def _do_write(self, job: Job, pipe_name: PipeName, payload: Sequence[Token]) -> None:
        pipe = self.pipes.ensure(pipe_name)

        # Effects are syscalls. This check is the enforcement floor that
        # still holds when every layer above it has failed: a fully persuaded model
        # reaching for a privileged pipe stops here, with its demotion history
        # attached to the fault.
        check = check_write(
            capabilities=job.capabilities,
            pipe=pipe_name,
            current_integrity=job.current_integrity,
            payload=render(payload),
            now_ns=self.clock.virtual_ns,
            session_floor=job.session_floor,
        )
        self._emit(
            CapabilityChecked(
                clock=self.clock,
                job=job.job_id,
                pipe=pipe_name,
                effective_integrity=check.effective,
                min_integrity=check.required,
                allowed=check.allowed,
            )
        )
        if not check.allowed:
            self._raise_fault(
                job,
                Fault(
                    kind=check.fault or FaultKind.CAPABILITY,
                    job=job.job_id,
                    detail=check.detail + self._demotion_history(job),
                    segment=self._worst_attended_segment(job),
                    pipe=pipe_name,
                ),
            )
            return

        # A verdict is not an ordinary write. Recognised here, after the capability
        # check, so that a job forging a verdict for a pipe it does not hold the
        # capability for is stopped by the ordinary boundary first.
        guarded = self.gates.by_verdict_pipe(pipe_name)
        if guarded is not None:
            self._resolve_verdict(job, guarded, render(payload))
            return

        # Layer four. The actuation is held and shown to its guard before it
        # reaches the device. ``pending_gate`` parks it in the same place
        # backpressure does, because from the job's side it is the same situation:
        # it asked to act, it has not acted, and it is descheduled until it can.
        gate = self.gates.for_pipe(pipe_name)
        if (
            gate is not None
            and job.pending_gate is None
            and job.gate_cleared != pipe_name
            and job.name != gate.descriptor
        ):
            self._consult_gate(job, gate, payload)
            return

        # An actuator has no backlog to fill (see ``Pipe.latch``), so it never applies
        # backpressure. Only a pipe carrying messages can, because only a message has
        # somebody waiting to receive it.
        if not pipe.spec.world_object and not pipe.writable(len(payload)):
            # All-or-nothing: a partial write would tear the payload, and dropping
            # the remainder would lose data. Park it and retry on wake.
            job.pending_write = (pipe_name, tuple(payload))
            pipe.block_writer(job.job_id)
            self.sched.block(job)
            job.blocked_on = pipe_name
            job.blocked_reason = "write-full"
            self._emit(
                PipeBackpressure(
                    clock=self.clock,
                    pipe=pipe_name,
                    job=job.job_id,
                    capacity_tokens=pipe.spec.capacity_tokens,
                )
            )
            self._emit(
                JobStateChanged(
                    clock=self.clock,
                    job=job.job_id,
                    from_state=JobState.RUNNING,
                    to_state=JobState.BLOCKED,
                )
            )
            self._emit(
                JobBlocked(clock=self.clock, job=job.job_id, pipe=pipe_name, reason="write-full")
            )
            self._apply_priority_inheritance(job, pipe_name, waiting_to_read=False)
            return

        # The distribution seam. If the link carries this pipe, the tokens leave the
        # node; the far kernel sees them as an ordinary pipe write an RTT later. The
        # job cannot tell the difference, which is the constraint transport/base.py
        # exists to protect -- federation changes latency, not semantics.
        if self._carried_by_link(pipe_name):
            self._send_over_link(job, pipe_name, payload)
            job.gate_cleared = None
            return

        latched = bool(pipe.spec.world_object)
        accepted = pipe.latch(payload) if latched else pipe.write(payload)
        # One verdict, one action. Clearing here rather than on wake means a second
        # actuation on the same pipe faces its gate again.
        job.gate_cleared = None
        self._emit(
            PipeWritten(
                clock=self.clock,
                pipe=pipe_name,
                job=job.job_id,
                tokens=accepted,
                text=tuple(t.text for t in payload[:accepted]),
                latched=latched,
            )
        )
        self._maybe_endorse(job, pipe_name)
        if pipe.spec.world_object:
            self._apply_world_write(pipe.spec.world_object, render(payload), by=job.job_id)
            job.record_write(ObjectSet.of([pipe.spec.world_object]))
        self._wake_readers(pipe_name)
        self._fire_vectors(pipe_name)

    def _maybe_endorse(self, job: Job, pipe_name: PipeName) -> None:
        """Record an endorsement when a dirty job writes cleanly through a schema.

        Endorsement is the *only* integrity-raising operation in the system,
        which makes it the one thing here that can be got badly wrong. It is
        recognised rather than granted: a job whose own integrity is worse than the
        ring of the pipe it writes to has, in effect, laundered content upward -- and
        the only reason that is acceptable is that the write passed a schema whose
        width bounds what could have crossed.

        Journalling the schema and its capacity is therefore not telemetry. It is
        the audit record for the single deliberate hole in the integrity lattice,
        and the input to answering OQ-3 with measurements rather than intuition.
        """
        capability = job.capabilities.get(pipe_name)
        if capability is None or capability.schema is None:
            return
        pipe_ring = Integrity(int(self.pipes.get(pipe_name).spec.ring))
        if int(job.current_integrity) <= int(pipe_ring):
            return  # no raise happened; nothing to endorse
        segment = job.segments.open_segment
        self._emit(
            Endorsed(
                clock=self.clock,
                job=job.job_id,
                endorser=job.name,
                segment=segment.id if segment else SegmentId(0),
                from_integrity=job.current_integrity,
                to_integrity=pipe_ring,
                schema=(f"{capability.schema.name}({capability.schema.capacity_bits():.0f} bits)"),
            )
        )

    # -- action gates -----------------------------------------------

    def _consult_gate(self, job: Job, gate: GateSpec, payload: Sequence[Token]) -> None:
        """Hold an actuation and spawn its guard.

        The guard is an ordinary job -- budgeted, schedulable, journaled -- which is
        what makes the semantic check auditable and testable rather than a predicate
        buried in the kernel. It is owned by the kernel, not by whoever is actuating:
        a gate owned by the job it is guarding could be cancelled by it.
        """
        rendered = render(payload)
        gate_job = self.spawn(gate.descriptor, owner=KERNEL_PRINCIPAL)
        request = GateRequest(
            job=job.job_id,
            pipe=gate.pipe,
            gate=gate.descriptor,
            payload=rendered,
            gate_job=gate_job.job_id,
            deadline=self.clock.token_clock + gate.timeout_ticks,
        )
        job.pending_gate = request
        job.pending_write = (gate.pipe, tuple(payload))

        # The guard reads the intended action from its stdin. Writing it directly
        # rather than through `_do_write` is deliberate: the kernel is the one
        # informing the gate, and routing it through the capability check would ask
        # whether the *kernel* may write to the gate's own request pipe.
        requests = self.pipes.ensure(gate.requests)
        asked = tokens_from_text(rendered)
        accepted = requests.write(asked)
        self._emit(
            PipeWritten(
                clock=self.clock,
                pipe=gate.requests,
                job=job.job_id,
                tokens=accepted,
                text=tuple(t.text for t in asked[:accepted]),
            )
        )
        self._wake_readers(gate.requests)

        self.sched.block(job)
        job.blocked_on = gate.pipe
        job.blocked_reason = "gate"
        self._transition(job, JobState.BLOCKED)
        self._emit(
            GateConsulted(
                clock=self.clock,
                job=job.job_id,
                pipe=gate.pipe,
                gate=gate.descriptor,
                gate_job=gate_job.job_id,
                payload=rendered,
            )
        )

    def _resolve_verdict(self, gate_job: Job, gate: GateSpec, text: str) -> None:
        """Apply a verdict to whichever held writes it answers.

        Matched by the gate job that was spawned for the request, not by pipe: two
        actuations held on the same pipe get their own guard instance and their own
        answer, so one veto cannot silently decide for the other.
        """
        verdict = parse_verdict(text)
        for job in self.sched.jobs():
            request = job.pending_gate
            if request is not None and request.gate_job == gate_job.job_id:
                self._apply_verdict(job, gate, verdict, text)

    def _apply_verdict(
        self,
        job: Job,
        gate: GateSpec,
        verdict: GateVerdict | None,
        raw: str,
    ) -> None:
        request = job.pending_gate
        job.pending_gate = None
        if verdict is None:
            # An unrecognised answer is not consent. "Fail open" on the last check
            # before an actuator is how safety interlocks become decorative.
            allowed = gate.on_gate_failure == ALLOW
            reason = f"unparseable verdict {raw!r}; applying {gate.on_gate_failure}"
        else:
            allowed = verdict.allowed
            reason = verdict.reason
        self._emit(
            GateAnswered(
                clock=self.clock,
                job=job.job_id,
                pipe=gate.pipe,
                gate=gate.descriptor,
                allowed=allowed,
                reason=reason,
            )
        )
        if not allowed:
            job.pending_write = None
            if self.sched.wake(job):
                self._transition(job, JobState.READY)
            self._raise_fault(
                job,
                Fault(
                    kind=FaultKind.GATE,
                    job=job.job_id,
                    detail=(
                        f"{gate.descriptor} vetoed the write to {gate.pipe}: {reason}"
                        + (f" (action: {request.payload!r})" if request else "")
                    ),
                    pipe=gate.pipe,
                ),
            )
            return
        # Allowed: the held write is retried through the ordinary path on wake.
        job.gate_cleared = gate.pipe
        if self.sched.wake(job):
            self._transition(job, JobState.READY)
        self._emit(JobWoken(clock=self.clock, job=job.job_id, pipe=gate.pipe))

    def _do_select(self, job: Job, pipe_names: tuple[PipeName, ...]) -> None:
        readable = [n for n in sorted(pipe_names) if self.pipes.ensure(n).readable]
        if readable:
            self._consume_read(job, readable[0])
            return
        for name in sorted(pipe_names):
            self.pipes.ensure(name).block_reader(job.job_id)
        job.pending_select = pipe_names
        self.sched.block(job)
        job.blocked_on = pipe_names[0] if pipe_names else PipeName("")
        job.blocked_reason = "select"
        self._emit(
            JobStateChanged(
                clock=self.clock,
                job=job.job_id,
                from_state=JobState.RUNNING,
                to_state=JobState.BLOCKED,
            )
        )
        self._emit(
            JobBlocked(clock=self.clock, job=job.job_id, pipe=job.blocked_on, reason="select")
        )
        for name in sorted(pipe_names):
            self._apply_priority_inheritance(job, name, waiting_to_read=True)

    def _service_pending(self, job: Job) -> bool:
        """Complete an operation the job was parked on. Consumes the quantum."""
        if job.pending_acquire is not None:
            name = job.pending_acquire
            job.pending_acquire = None
            job.blocked_reason = ""
            self._do_acquire(job, name)
            return True
        if job.pending_write is not None:
            pipe_name, payload = job.pending_write
            pipe = self.pipes.get(pipe_name)
            tokens = tuple(t for t in payload if isinstance(t, Token))
            if not pipe.writable(len(tokens)):
                pipe.block_writer(job.job_id)
                self.sched.block(job)
                self._transition(job, JobState.BLOCKED)
                return True
            job.pending_write = None
            self._do_write(job, pipe_name, tokens)
            return True
        if job.pending_select:
            names = job.pending_select
            job.pending_select = ()
            for name in names:
                self.pipes.get(name).unblock(job.job_id)
            readable = [n for n in sorted(names) if self.pipes.get(n).readable]
            if readable:
                self._consume_read(job, readable[0])
                return True
        if job.pending_read is not None:
            pipe_name = job.pending_read
            job.pending_read = None
            if self.pipes.get(pipe_name).readable:
                self._consume_read(job, pipe_name)
                return True
        return False

    def _wake_readers(self, pipe_name: PipeName) -> None:
        # A pinned job has never run, so it is not among the pipe's blocked readers:
        # what addresses it is a write to the input its descriptor declared. Being
        # addressed *delivers* -- the payload is taken from the pipe and injected, the
        # same way firing a vector hands the source payload to the handler it spawns.
        #
        # Leaving it queued instead is not a smaller version of the same thing, it is
        # the bug pinning exists to fix, moved one turn later: the job is dispatched
        # correctly, does its work, and then its closing read finds the token that
        # woke it still sitting there, returns at once instead of blocking, and takes
        # a second turn on it.
        for pinned in self.sched.in_state(JobState.PINNED_IDLE):
            if pinned.descriptor.pipes.stdin != pipe_name:
                continue
            self._transition(pinned, JobState.READY)
            self._consume_read(pinned, pipe_name)
        pipe = self.pipes.get(pipe_name)
        for woken in self.sched.wake_all(pipe.take_waiting_readers()):
            self._release_priority_inheritance(woken)
            self._emit(JobWoken(clock=self.clock, job=woken.job_id, pipe=pipe_name))
            self._emit(
                JobStateChanged(
                    clock=self.clock,
                    job=woken.job_id,
                    from_state=JobState.BLOCKED,
                    to_state=JobState.READY,
                )
            )

    def _apply_world_write(
        self, obj_name: str, value: str, *, by: JobId | None, note: str = ""
    ) -> None:
        write = self.world.set(ObjectName(obj_name), value, at=self.clock, by=by, note=note)
        if write is not None:
            self._emit(
                WorldWritten(
                    clock=self.clock,
                    job=by,
                    obj=write.obj,
                    before=write.before,
                    after=write.after,
                )
            )
            self._refresh_status_regions(write.obj)

    # -- virtual context -----------------------------------------------------

    def _maybe_evict(self, job: Job) -> None:
        """Plan and execute eviction at a block boundary.

        Eviction is triggered by pressure and batched at boundaries, because every
        SPLICE invalidates downstream KV -- planning them together is what keeps the
        cost bounded.
        """
        policy = job.descriptor.context
        resident = self.machine.stats(job.job_id).resident_tokens
        plan = plan_eviction(
            job.segments,
            policy,
            resident_tokens=resident,
            current_token_clock=self.clock.token_clock,
        )
        for skipped in plan.skipped:
            self._emit(
                Note(
                    clock=self.clock,
                    text=(
                        f"declined to evict segment {skipped.record.id}: {skipped.reason} "
                        f"(would free {skipped.freed}, refault risk {skipped.refault_risk:.2f})"
                    ),
                    tags=("vm", "eviction-skipped"),
                )
            )
        for candidate in plan.candidates:
            self._evict(job, candidate)

    def _evict(self, job: Job, candidate: EvictionCandidate) -> None:
        record = candidate.record
        transcript = self.machine.transcript(job.job_id)
        tokens = transcript[record.start : record.end]
        span = self.store.put(record, tokens)

        _, summary_budget = stub_size(record.tokens)
        summary = summarise(tokens, budget=summary_budget)
        stub_text = render_stub(record, summary, store_id=span.store_id)
        stub_tokens = tokens_from_text(stub_text, TokenKind.CONTROL)

        splice = self.machine.splice(job.job_id, record.start, record.end, stub_tokens)
        stub_id = self._next_segment()
        job.segments.evict_to_stub(
            record.id,
            stub_id,
            stub_tokens=len(stub_tokens),
            token_clock=self.clock.token_clock,
        )
        self._emit(
            Spliced(
                clock=self.clock,
                job=job.job_id,
                start_segment=record.id,
                end_segment=stub_id,
                tokens_in=splice.tokens_in,
                tokens_out=record.tokens,
                invalidated_downstream_tokens=splice.invalidated_downstream,
            )
        )
        self._emit(
            SegmentEvicted(
                clock=self.clock,
                job=job.job_id,
                segment=record.id,
                stub=stub_id,
                store=span.store_id,
                freed_tokens=record.tokens - len(stub_tokens),
                stub_tokens=len(stub_tokens),
                policy=job.descriptor.context.eviction.value,
            )
        )
        self._emit(
            ResidencyChanged(
                clock=self.clock,
                job=job.job_id,
                segment=record.id,
                from_residency=Residency.RESIDENT,
                to_residency=Residency.STUBBED,
            )
        )
        self.pager.record_eviction(
            EvictionRecord(
                segment=record.id,
                store_id=span.store_id,
                stub=stub_id,
                evicted_at_block=job.last_block,
                freed_tokens=record.tokens - len(stub_tokens),
                stub_tokens=len(stub_tokens),
            )
        )
        self._refresh_mask(job)

    def _service_fault(self, job: Job, segment: SegmentId | None) -> None:
        """Explicit page fault on a stub handle.

        Servicing is ordinary pipe I/O: the job blocks and therefore costs nothing
        while it waits. That is the whole economic argument for kernel-managed
        recall over model-initiated retrieval.
        """
        if segment is None:
            return
        self._emit(
            PageFaultRaised(
                clock=self.clock,
                job=job.job_id,
                explicit=True,
                segment=segment,
                need_text=None,
            )
        )
        since = self.pager.blocks_since_eviction(segment, job.last_block)
        if since is not None and since <= job.descriptor.context.refault_window_blocks:
            job.thrash.note_refault(at_block=job.last_block)
            self.pager.note_refault(segment)
            self._emit(
                Refaulted(
                    clock=self.clock,
                    job=job.job_id,
                    segment=segment,
                    blocks_since_evict=since,
                )
            )
            self._check_thrash(job)

        result = self.pager.resolve_fault(segment)
        self._complete_page_in(job, segment, result)

    def _service_need(self, job: Job, text: str) -> None:
        """A structured request for content the model believes exists.

        A NEED the pager cannot satisfy returns a kernel notice saying so, which is
        itself valuable state: "we looked and it is not there" is a different belief
        from "I forgot to look".
        """
        self._emit(
            PageFaultRaised(
                clock=self.clock, job=job.job_id, explicit=False, segment=None, need_text=text
            )
        )
        self._complete_page_in(job, None, self.pager.resolve_need(text))

    def _complete_page_in(self, job: Job, segment: SegmentId | None, result: PagerResult) -> None:
        if result.span is None:
            # Either duplicate-suppressed or a miss; both answer with a ring-0 notice
            # rather than silence.
            self._inject_kernel(job, result.notice)
            return

        span = result.span
        resident = self.machine.stats(job.job_id).resident_tokens
        downstream = 0  # append plan lands at the tail, so nothing follows it
        plan = choose_plan(span_tokens=span.length, downstream_tokens=downstream)

        new_segment = self._inject(
            job,
            span.tokens,
            pipe=span.provenance.pipe,
            principal=span.provenance.principal,
            ring=span.ring,
            integrity=span.integrity,
            tag=span.provenance.tag,
            derived_from=(span.origin,),
        )
        self._emit(
            PagedIn(
                clock=self.clock,
                job=job.job_id,
                segment=new_segment,
                store=span.store_id,
                plan=plan.plan,
                cost_tokens=plan.cost_tokens,
            )
        )
        if segment is not None:
            self.pager.note_paged_in(segment, new_segment)
            job.segments.reinstate(segment, into=new_segment)
            self._emit(
                ResidencyChanged(
                    clock=self.clock,
                    job=job.job_id,
                    segment=segment,
                    from_residency=Residency.STUBBED,
                    to_residency=Residency.RESIDENT,
                )
            )
        _ = resident

    def _check_thrash(self, job: Job) -> None:
        """Thrashing is loud by design.

        A system that quietly thrashes looks identical to one that is merely slow,
        and the remedy -- decompose, or move to a bigger model class -- is a decision
        someone has to make, which requires knowing.
        """
        if not job.thrash.is_thrashing():
            return
        detail = (
            f"refault rate {job.thrash.rate():.2f}/block over the last "
            f"{job.descriptor.context.refault_window_blocks} blocks exceeds "
            f"{job.descriptor.context.thrash_threshold}; the working set does not fit "
            f"in a window of {job.descriptor.context.window}"
        )
        self._raise_fault(job, Fault(kind=FaultKind.THRASH, job=job.job_id, detail=detail))
        job.thrash.recent.clear()  # do not re-raise every block

    def _sample_working_set(self, job: Job) -> None:
        tokens, count = working_set_tokens(job.segments, job.descriptor.context.theta_ws)
        self._emit(
            WorkingSetSampled(clock=self.clock, job=job.job_id, size_tokens=tokens, segments=count)
        )

    def _refresh_status_regions(self, obj: ObjectName) -> None:
        """Refresh mapped status regions when the object they view changes.

        Resume revalidation is a special case of map invalidation: mapped
        regions refresh unconditionally, and the RESUME diff covers only the
        *unmapped* remainder -- beliefs the model derived rather than read.

        A region is created once and **rewritten** thereafter, which is what ``Perm.W``
        means and what makes the region a view rather than a log: ``_retire_status_region``
        removes the copy the new one supersedes.
        """
        for job in self.sched.jobs():
            # A job that has not been dispatched yet has no body in its context, and a
            # region injected now would sit in front of the instructions that explain
            # it. It is seeded at ``_start_job`` instead, from the same world value.
            if job.state.is_terminal or not job.started:
                continue
            # By object, not by spec: ``maps:`` is not checked for uniqueness, and one
            # refresh per duplicate entry would publish a redundant view per write.
            if obj in {spec.obj for spec in job.descriptor.maps if spec.is_status_region}:
                self._refresh_status_region(job, obj)

    def _refresh_status_region(self, job: Job, obj: ObjectName) -> None:
        self._retire_status_region(job, obj)
        tokens = tokens_from_text(f"<STATUS {obj}> {self.world.get(obj, '(unset)')} </STATUS>")
        segment = self._inject(
            job,
            tokens,
            pipe=KERNEL_PIPE,
            principal=Principal.KERNEL,
            ring=Ring.KERNEL,
            integrity=Integrity(0),
            tag=map_tag(obj),
            perms=Perm.R | Perm.W,
        )
        self._emit(
            MapRefreshed(
                clock=self.clock,
                job=job.job_id,
                obj=obj,
                segment=segment,
                cost_tokens=len(tokens),
            )
        )

    def _retire_status_region(self, job: Job, obj: ObjectName) -> None:
        """Get the previous view out of the way before the new one is published.

        **Retraction, not eviction.** A superseded region is dead by construction --
        its content is ``world.get(obj)``, and the replacement is about to be appended
        -- so there is nothing to spill and nothing to fault back. Handing it to the
        pager instead does not work even in principle: ``stub_size`` is
        ``STUB_FRAMING_TOKENS + max(2, n // 8)``, which exceeds ``n`` for any span
        under about eleven tokens, so ``plan_eviction`` computes ``freed <= 0``, marks
        the candidate net-negative and declines it at every boundary for the life of
        the job. Demotion alone closes the leak on paper only.

        What retraction costs is downstream KV, and the region's position is what makes
        that cheap: it was appended at the previous refresh, so the invalidated extent
        is whatever the job has generated *since* -- exactly the interval in which the
        stale copy accrued. For a job descheduled on a pipe read while world updates
        arrive, which is the ordinary life of a status region, the region is still the
        tail and the extent is zero. Retraction then returns the context to the block
        boundary the region was injected at, and the replacement lands there: the
        region really is "rewritten in place" (Appendix C), the tail merely being where
        the place happens to be.

        Above the ratio the old behaviour stands -- revoke ``Perm.W`` and let the copy
        become ordinary history -- because invalidating a long tail of KV to reclaim a
        dozen tokens is the worse trade. That case is recorded rather than silent.
        """
        live = [
            record
            for record in job.segments.by_tag(map_tag(obj))
            if Perm.W in record.perms and record.residency is Residency.RESIDENT
        ]
        for record in live:
            # Measured before any padding: ``_inject`` pads to a block boundary, and
            # padding lands *after* this region, so closing first would manufacture
            # the very downstream extent being weighed.
            downstream = self.machine.stats(job.job_id).resident_tokens - record.end
            budget = job.descriptor.context.retract_recompute_ratio * record.tokens
            if record.is_closed and downstream <= budget:
                self._retract_status_region(job, record, obj)
            else:
                job.segments.revoke([record.id], Perm.W)
                self._emit(
                    Note(
                        clock=self.clock,
                        text=(
                            f"declined to retract status region {record.id} for {obj}: "
                            f"{downstream} tokens downstream exceeds the {budget:.0f} "
                            f"justified by {record.tokens} reclaimed; demoted to history"
                        ),
                        tags=("vm", "retract-declined"),
                    )
                )

    def _retract_status_region(self, job: Job, record: SegmentRecord, obj: ObjectName) -> None:
        splice = self.machine.splice(job.job_id, record.start, record.end, ())
        freed = job.segments.retract(record.id)
        self._emit(
            Spliced(
                clock=self.clock,
                job=job.job_id,
                # A removal has no successor segment. ``start == end`` with
                # ``tokens_in == 0`` is how the alphabet says so; every other splice
                # names the span that replaced the one it consumed.
                start_segment=record.id,
                end_segment=record.id,
                tokens_in=splice.tokens_in,
                tokens_out=freed,
                invalidated_downstream_tokens=splice.invalidated_downstream,
            )
        )
        self._emit(
            StatusRegionRetracted(
                clock=self.clock,
                job=job.job_id,
                obj=obj,
                segment=record.id,
                freed_tokens=freed,
                invalidated_downstream_tokens=splice.invalidated_downstream,
            )
        )
        self._refresh_mask(job)

    # -- vectors -------------------------------------------------------------

    def _fire_vectors(self, pipe_name: PipeName) -> None:
        for decision in self.vectors.on_write(pipe_name, self.clock.virtual_ns):
            spec = decision.spec
            match decision.action:
                case VectorAction.DISPATCH:
                    self._emit(
                        VectorFired(
                            clock=self.clock,
                            vector=spec.name,
                            pipe=spec.source,
                            handler=spec.handler,
                            priority=spec.priority,
                            policy=spec.policy,
                        )
                    )
                    self._dispatch_handler(spec.name)
                case VectorAction.COALESCE:
                    self._emit(
                        VectorCoalesced(
                            clock=self.clock, vector=spec.name, collapsed=decision.collapsed
                        )
                    )
                case VectorAction.THROTTLE:
                    self._emit(
                        VectorThrottled(
                            clock=self.clock,
                            vector=spec.name,
                            min_interval_ns=spec.min_interval_ns or 0,
                            since_last_ns=decision.since_last_ns,
                        )
                    )
                case VectorAction.QUEUE:
                    self._emit(
                        Note(
                            clock=self.clock,
                            text=f"vector {spec.name} queued behind an active instance",
                            tags=("vector", "queue"),
                        )
                    )

    def _dispatch_handler(self, name: VectorName) -> None:
        spec = self.vectors.get(name)
        absorbed = self.vectors.mark_dispatched(name, self.clock.virtual_ns)
        job = self.spawn(spec.handler, priority=spec.priority, vector=name)
        # Draining the source is a read like any other, and until this was journalled
        # it was the one pipe mutation the record did not carry: a fold replaying the
        # journal left the payload sitting in the buffer for the rest of the run.
        payload = self.pipes.get(spec.source).read()
        job.vector_payload = payload
        self._emit(
            PipeReadEvent(
                clock=self.clock,
                pipe=spec.source,
                job=job.job_id,
                tokens=len(payload),
                text=tuple(t.text for t in payload),
            )
        )
        if absorbed:
            self._emit(VectorCoalesced(clock=self.clock, vector=name, collapsed=absorbed))

    def _dispatch_due_vectors(self) -> None:
        """Re-fire vectors whose throttle interval has elapsed.

        A throttle defers; it does not discard. Without this, storm control would
        quietly become data loss.
        """
        for spec in self.vectors.due(self.clock.virtual_ns):
            self._emit(
                VectorFired(
                    clock=self.clock,
                    vector=spec.name,
                    pipe=spec.source,
                    handler=spec.handler,
                    priority=spec.priority,
                    policy=spec.policy,
                )
            )
            self._dispatch_handler(spec.name)

    # -- completion and faults -----------------------------------------------

    def _complete(self, job: Job) -> None:
        self.sched.yield_running()
        self._transition(job, JobState.DONE)
        self._emit(JobCompleted(clock=self.clock, job=job.job_id, tokens_used=job.tokens_used))
        lease = self.leases.lease_of(job.job_id)
        if lease is not None:
            self.leases.revoke(lease.platform)
            self._emit(
                Disembodied(
                    clock=self.clock,
                    job=job.job_id,
                    platform=lease.platform,
                    reason="job completed",
                )
            )
        self._release_held_resources(job)
        if job.vector is not None:
            self.vectors.mark_complete(job.vector)
        self._apply_completion_policy(job)
        # The context is deliberately *not* destroyed. The transcript is the
        # source of truth (core §2.1), and a completed job's transcript is
        # exactly what an incident review, a replay, or a stub-fidelity study
        # needs to read. A production kernel would archive it to the store and
        # then drop it; ``reap`` is that operation, and the driver decides when.

    def reap(self, job_id: JobId) -> None:
        """Release a terminal job's materialised context.

        Separate from completion because *when* to give the memory back is a
        deployment decision, not a kernel one: a research run wants every
        transcript at the end, a fleet wants the HBM immediately.
        """
        job = self.sched.get(job_id)
        if not job.state.is_terminal:
            raise KernelError(f"job {job_id} is not terminal; refusing to reap")
        self.machine.destroy_context(job_id)

    def _apply_completion_policy(self, job: Job) -> None:
        policy = job.descriptor.on_complete
        match policy.kind:
            case OnComplete.RETURN:
                return
            case OnComplete.CANCEL_BELOW:
                for cancelled in self.sched.cancel_below(policy.depth):
                    self._transition(cancelled, JobState.DONE)
                    self._emit(
                        JobCancelled(
                            clock=self.clock,
                            job=cancelled.job_id,
                            by_job=job.job_id,
                            policy=f"cancel-below:{policy.depth}",
                        )
                    )
            case OnComplete.REPLACE_WITH:
                for cancelled in self.sched.clear_stack():
                    self._transition(cancelled, JobState.DONE)
                    self._emit(
                        JobCancelled(
                            clock=self.clock,
                            job=cancelled.job_id,
                            by_job=job.job_id,
                            policy="replace-with",
                        )
                    )
                if policy.replacement is not None:
                    self.spawn(policy.replacement)

    def _check_budget(self, job: Job) -> bool:
        breach = job.budget_exceeded(self.clock)
        if breach is None:
            return False
        kind = FaultKind.BUDGET if "budget" in breach else FaultKind.DEADLINE
        self._raise_fault(job, Fault(kind=kind, job=job.job_id, detail=breach))
        return True

    def _raise_fault(self, job: Job, fault: Fault) -> None:
        self._emit(
            FaultRaised(
                clock=self.clock,
                job=job.job_id,
                fault=fault.kind,
                detail=fault.detail,
                segment=fault.segment,
                pipe=fault.pipe,
            )
        )
        resolution = resolve(fault, job.descriptor.on_fault)
        self._emit(
            FaultDispatched(
                clock=self.clock,
                job=job.job_id,
                fault=fault.kind,
                policy=resolution.action.value,
                handler=resolution.handler,
            )
        )
        match resolution.action:
            case FaultAction.ABORT | FaultAction.ESCALATE:
                if self.sched.running is job:
                    self.sched.yield_running()
                self._transition(job, JobState.FAULTED)
                self._release_held_resources(job)
            case FaultAction.DISPATCH_HANDLER:
                self._inject_kernel(job, resolution.notice)
                if resolution.handler is not None:
                    self.spawn(resolution.handler, parent=job.job_id)
            case FaultAction.CONTINUE | FaultAction.RETRY:
                self._inject_kernel(job, resolution.notice)
