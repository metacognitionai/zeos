# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The journal alphabet: every fact the kernel can record.

The journal is the source of truth. Kernel state is a fold
over this sequence, which is what gives us three things at once:

* **Determinism gate** -- same tree + same schedule + same seed must produce a
  byte-identical journal.
* **Testing** -- integration assertions are made against journal *properties*
  ("the alarm preempted supervision within one token boundary"; "supervision
  resumed with a dirty notice naming ``plant.unit_a``") rather than by inspecting
  transcripts, which are statistical.
* **Provenance and audit** -- for any effect, the derivation chain back to the
  external inputs that could have influenced it.

Events are pure data with no behaviour. Every event carries the ``Clock`` at which
it occurred; sequence numbers are assigned by the journal on append, not here, so
that an event is a fact rather than a position.

World-state values are ``str`` at the kernel boundary. That is deliberate rather
than lazy: threshold evaluation belongs to device adapters on the driver side, and
what the kernel needs the value *for* -- read/write-set diffing, RESUME text,
and status-region rendering -- is exactly the string the model will read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Final

from zeos.core.clock import Clock
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    Integrity,
    JobId,
    JobState,
    ObjectName,
    Perm,
    PipeName,
    Placement,
    Principal,
    PrincipalId,
    Priority,
    Residency,
    ResourceKind,
    ResourceName,
    ResumeKind,
    Ring,
    SegmentId,
    StoreId,
    VectorName,
    VectorPolicy,
)

__all__ = ["Event", "EVENT_REGISTRY", "event_class", "StateDelta"]

#: Duplicated from ``core.principals`` rather than imported: ``events`` sits below
#: it in the dependency order, and one string constant is a smaller price than a
#: cycle. ``principals`` asserts they agree.
KERNEL_OWNER = PrincipalId("kernel")


@dataclass(frozen=True)
class Event:
    """Base for every journalled fact. Subclasses add payload fields after ``clock``."""

    KIND: ClassVar[str] = "event"

    clock: Clock


EVENT_REGISTRY: Final[dict[str, type[Event]]] = {}


def event_class[E: Event](cls: type[E]) -> type[E]:
    """Register an event type under its ``KIND`` so replay can reconstruct it.

    Generic so that decorating a subclass preserves its type: a non-generic
    ``type[Event] -> type[Event]`` would erase every event down to ``Event`` at the
    use site, which silently defeats both type checking and exhaustiveness analysis
    over the alphabet.
    """
    kind = cls.KIND
    if kind == Event.KIND:
        raise ValueError(f"{cls.__name__} must declare its own KIND")
    existing = EVENT_REGISTRY.get(kind)
    if existing is not None:
        raise ValueError(f"duplicate event KIND {kind!r}: {existing.__name__} / {cls.__name__}")
    EVENT_REGISTRY[kind] = cls
    return cls


@dataclass(frozen=True)
class StateDelta:
    """One line of a RESUME diff: world state the resumed job depends on that
    changed while it was suspended.

    ``note`` carries a change the *value* cannot express. ZEOS-Fleet's
    re-embodiment example is the motivating case::

        self.tooling: gripper-std -> gripper-std (recalibrated; offsets differ)

    Before and after are identical, and it is still the most important line in the
    diff. Without a note field the kernel would drop it as an idempotent write and
    the resumed job would plan against a gripper it has not been calibrated for.
    """

    obj: ObjectName
    before: str
    after: str
    note: str = ""


# ---------------------------------------------------------------------------
# Kernel lifecycle and descriptor loading
# ---------------------------------------------------------------------------


@event_class
@dataclass(frozen=True)
class KernelStarted(Event):
    KIND: ClassVar[str] = "kernel.started"

    seed: int
    block_size: int
    case: str


@event_class
@dataclass(frozen=True)
class DescriptorLoaded(Event):
    KIND: ClassVar[str] = "descriptor.loaded"

    descriptor: DescriptorName
    priority: Priority
    placement: Placement
    pinned: bool
    preemptible: bool


@event_class
@dataclass(frozen=True)
class DescriptorRejected(Event):
    """Load-time lint failure -- the compiler error that arrives before anything runs
    ."""

    KIND: ClassVar[str] = "descriptor.rejected"

    descriptor: DescriptorName
    rule: str
    detail: str


# ---------------------------------------------------------------------------
# Job lifecycle and scheduling
# ---------------------------------------------------------------------------


@event_class
@dataclass(frozen=True)
class JobSpawned(Event):
    """A job started.

    ``owner`` and ``integrity`` are here because a monitor folding this journal
    could not otherwise answer two questions the kernel definitely knows the answer
    to: *whose job is this?* and *where did its watermark start?* Ownership was
    reconstructible only from ``AuthorityNarrowed``, which is emitted only when the
    descriptor declares capabilities -- so kernel-owned and capability-free jobs were
    simply unattributable. The starting integrity appeared nowhere at all, which made
    every later demotion a delta from an unknown baseline.

    Both are the same lesson the project keeps relearning: if no journal event proves
    it, it cannot be asserted from outside.
    """

    KIND: ClassVar[str] = "job.spawned"

    job: JobId
    descriptor: DescriptorName
    priority: Priority
    parent: JobId | None
    owner: PrincipalId = KERNEL_OWNER
    integrity: Integrity = Integrity(2)


@event_class
@dataclass(frozen=True)
class JobStateChanged(Event):
    """The canonical low-level transition record. The semantic events below
    (preempted, blocked, resumed) annotate *why*; this records *that*."""

    KIND: ClassVar[str] = "job.state"

    job: JobId
    from_state: JobState
    to_state: JobState


@event_class
@dataclass(frozen=True)
class JobDispatched(Event):
    KIND: ClassVar[str] = "job.dispatched"

    job: JobId
    priority: Priority


@event_class
@dataclass(frozen=True)
class JobPreempted(Event):
    KIND: ClassVar[str] = "job.preempted"

    job: JobId
    by_job: JobId
    by_priority: Priority
    stack_depth: int


@event_class
@dataclass(frozen=True)
class JobResumed(Event):
    """Pop from the suspension stack. ``dirty`` is the computed intersection of the
    job's read-set with the write-sets of everything that ran above it (core §6.2)."""

    KIND: ClassVar[str] = "job.resumed"

    job: JobId
    resume_kind: ResumeKind  # not ``kind``: that name belongs to the record envelope
    suspended_ns: int
    dirty: tuple[StateDelta, ...] = ()


@event_class
@dataclass(frozen=True)
class JobBlocked(Event):
    KIND: ClassVar[str] = "job.blocked"

    job: JobId
    pipe: PipeName
    reason: str  # "read-empty" | "write-full" | "page-fault"


@event_class
@dataclass(frozen=True)
class JobWoken(Event):
    """Wakes go to READY, never straight to RUNNING -- the scheduler decides
    (core Appendix A, rule 3)."""

    KIND: ClassVar[str] = "job.woken"

    job: JobId
    pipe: PipeName


@event_class
@dataclass(frozen=True)
class ResourceAcquired(Event):
    KIND: ClassVar[str] = "resource.acquired"

    job: JobId
    resource: ResourceName
    #: not ``kind`` -- that name belongs to the journal record envelope, as the
    #: codec guard points out the moment you forget.
    resource_kind: ResourceKind
    holders: int
    capacity: int


@event_class
@dataclass(frozen=True)
class ResourceReleased(Event):
    KIND: ClassVar[str] = "resource.released"

    job: JobId
    resource: ResourceName
    woke: JobId | None = None


@event_class
@dataclass(frozen=True)
class ResourceBlocked(Event):
    """A job waiting on a full resource. Blocking here costs nothing, exactly as
    blocking on a pipe does -- it is the same deschedule."""

    KIND: ClassVar[str] = "resource.blocked"

    job: JobId
    resource: ResourceName
    holders: tuple[JobId, ...]


@event_class
@dataclass(frozen=True)
class DeadlockDetected(Event):
    """A cycle in the wait-for graph, named in full.

    The cycle is part of the event because an operator should not have to
    reconstruct it from a heap of blocked jobs.
    """

    KIND: ClassVar[str] = "resource.deadlock"

    cycle: tuple[JobId, ...]
    resources: tuple[ResourceName, ...]
    victim: JobId


@event_class
@dataclass(frozen=True)
class Embodied(Event):
    """A job took a body. The lease is an ordinary resource."""

    KIND: ClassVar[str] = "fleet.embodied"

    job: JobId
    platform: str
    release_policy: str
    gang: str = ""


@event_class
@dataclass(frozen=True)
class Disembodied(Event):
    """A job lost a body. Every revocation is a **suspension, not a loss** -- the
    transcript survives and the job resumes in whatever body comes next."""

    KIND: ClassVar[str] = "fleet.disembodied"

    job: JobId
    platform: str
    reason: str
    evictions: int = 0


@event_class
@dataclass(frozen=True)
class PlatformJoined(Event):
    """A body joined the fleet. Per ZEOS-Fleet: "joining the fleet is device
    hotplug" -- the profile is the driver, and presenting one is the whole protocol.
    """

    KIND: ClassVar[str] = "fleet.platform_joined"

    platform: str
    profile: str


@event_class
@dataclass(frozen=True)
class PlatformWithdrawn(Event):
    """A body left the fleet, by fault or by hand.

    Distinct from ``Disembodied`` on purpose: revoking a lease frees a body for the
    next claimant, whereas withdrawing one says the body is out of service and must
    not be allocated again. A flat battery is the second, not the first.
    """

    KIND: ClassVar[str] = "fleet.platform_withdrawn"

    platform: str
    reason: str
    held_by: JobId | None = None


@event_class
@dataclass(frozen=True)
class AllocationRefused(Event):
    KIND: ClassVar[str] = "fleet.allocation_refused"

    job: JobId
    descriptor: DescriptorName
    requirements: str
    reason: str


@event_class
@dataclass(frozen=True)
class GangAssembled(Event):
    """All-or-none dispatch: members become runnable together, each on its own
    body, only once every required embodiment is leased."""

    KIND: ClassVar[str] = "fleet.gang_assembled"

    gang: str
    members: tuple[JobId, ...]
    platforms: tuple[str, ...]
    coupling: str


@event_class
@dataclass(frozen=True)
class GangDissolved(Event):
    """All-or-none preemption. Preempting one carrier mid-lift while the other
    continues is strictly worse than preempting both."""

    KIND: ClassVar[str] = "fleet.gang_dissolved"

    gang: str
    members: tuple[JobId, ...]
    reason: str


@event_class
@dataclass(frozen=True)
class PriorityInherited(Event):
    """Priority inheritance on a held resource, preventing classic inversion
    (core §2.2)."""

    KIND: ClassVar[str] = "job.priority_inherited"

    job: JobId
    from_priority: Priority
    to_priority: Priority
    blocked_job: JobId
    resource: str


@event_class
@dataclass(frozen=True)
class PriorityRestored(Event):
    KIND: ClassVar[str] = "job.priority_restored"

    job: JobId
    to_priority: Priority


@event_class
@dataclass(frozen=True)
class JobCompleted(Event):
    KIND: ClassVar[str] = "job.completed"

    job: JobId
    tokens_used: int


@event_class
@dataclass(frozen=True)
class JobCancelled(Event):
    """A job stopped before it finished.

    Either a handler's ``cancel-below`` / ``replace-with`` policy unwinding the stack,
    or a principal cancelling a job it owns. ``by_job`` is None in the
    second case -- a user-initiated cancel has a cancelling *principal*, not a
    cancelling job, and ``policy`` carries which."""

    KIND: ClassVar[str] = "job.cancelled"

    job: JobId
    by_job: JobId | None
    policy: str


# ---------------------------------------------------------------------------
# Pipes
# ---------------------------------------------------------------------------


@event_class
@dataclass(frozen=True)
class PipeCreated(Event):
    KIND: ClassVar[str] = "pipe.created"

    pipe: PipeName
    capacity_tokens: int
    ring: Ring
    principal: Principal
    transport: str  # "local" in phase 1; the distribution seam records it anyway


@event_class
@dataclass(frozen=True)
class PipeWritten(Event):
    KIND: ClassVar[str] = "pipe.written"

    pipe: PipeName
    job: JobId | None  # None when a device adapter or the kernel writes
    tokens: int
    #: The accepted tokens' text, in order. A short write journals only what was
    #: accepted, so replaying writes and reads reconstructs the buffer exactly.
    text: tuple[str, ...] = ()
    #: True when the write replaced the buffer rather than appending to it -- what
    #: ``Pipe.latch`` does for an actuator pipe, where a write is an effect and the
    #: buffer holds the current value rather than a backlog. Without this a fold
    #: cannot tell the two apart, and over-reports depth on every actuator pipe.
    latched: bool = False


@event_class
@dataclass(frozen=True)
class PipeReadEvent(Event):
    KIND: ClassVar[str] = "pipe.read"

    pipe: PipeName
    job: JobId
    tokens: int
    #: What was taken, in order -- the head of the buffer this read consumed.
    text: tuple[str, ...] = ()


@event_class
@dataclass(frozen=True)
class PipeBackpressure(Event):
    """A write to a full pipe. Automatic rate matching between models of different
    speeds, with zero logic in either descriptor (core §4.1)."""

    KIND: ClassVar[str] = "pipe.backpressure"

    pipe: PipeName
    job: JobId
    capacity_tokens: int


# ---------------------------------------------------------------------------
# Interrupt vectors
# ---------------------------------------------------------------------------


@event_class
@dataclass(frozen=True)
class VectorFired(Event):
    KIND: ClassVar[str] = "vector.fired"

    vector: VectorName
    pipe: PipeName
    handler: DescriptorName
    priority: Priority
    policy: VectorPolicy


@event_class
@dataclass(frozen=True)
class VectorCoalesced(Event):
    """Level-triggered, not edge-triggered: N pending firings collapse into one
    dispatch that reads the latest value (core §5.5)."""

    KIND: ClassVar[str] = "vector.coalesced"

    vector: VectorName
    collapsed: int


@event_class
@dataclass(frozen=True)
class VectorThrottled(Event):
    KIND: ClassVar[str] = "vector.throttled"

    vector: VectorName
    min_interval_ns: int
    since_last_ns: int


# ---------------------------------------------------------------------------
# Machine ops -- the five, plus the block boundary that MP/VM logic hangs off
# ---------------------------------------------------------------------------


@event_class
@dataclass(frozen=True)
class Decoded(Event):
    KIND: ClassVar[str] = "machine.decode"

    job: JobId
    segment: SegmentId
    tokens: int
    #: What the model produced at this boundary, one string per token.
    #:
    #: A tuple rather than a rendered line because a real tokenizer's tokens are
    #: sub-word pieces: joining them on whitespace would be lossy, and the debugger
    #: has to show what the model actually emitted. The kernel still never *reads*
    #: this -- no kernel decision consults token text (ZEOS-AM §3.1); it is recorded
    #: so that what a job generated is answerable from the journal rather than only
    #: from a live machine.
    text: tuple[str, ...] = ()


@event_class
@dataclass(frozen=True)
class Injected(Event):
    """The only entry path for foreign tokens -- which is what makes provenance
    total."""

    KIND: ClassVar[str] = "machine.inject"

    job: JobId
    segment: SegmentId
    pipe: PipeName
    principal: Principal
    ring: Ring
    integrity: Integrity
    tokens: int
    #: The foreign tokens themselves. This is the entry path provenance is stamped
    #: on, so it is also the one place the content that entered a context is
    #: recoverable after the run.
    text: tuple[str, ...] = ()


@event_class
@dataclass(frozen=True)
class Truncated(Event):
    KIND: ClassVar[str] = "machine.trunc"

    job: JobId
    at: int
    dropped_segments: tuple[SegmentId, ...]


@event_class
@dataclass(frozen=True)
class Forked(Event):
    KIND: ClassVar[str] = "machine.fork"

    parent: JobId
    child: JobId
    shared_segments: int


@event_class
@dataclass(frozen=True)
class Spliced(Event):
    KIND: ClassVar[str] = "machine.splice"

    job: JobId
    start_segment: SegmentId
    end_segment: SegmentId
    tokens_in: int
    #: Tokens the splice removed. Carried so that resident-size accounting folds
    #: exactly from the journal alone -- without it, replay cannot reconstruct
    #: context length and the determinism gate becomes unverifiable.
    tokens_out: int
    invalidated_downstream_tokens: int


@event_class
@dataclass(frozen=True)
class BlockBoundary(Event):
    """Segments are block-aligned by construction. Simulated in M0
    so that boundary-batched logic is exercised from the start rather than written
    against a machine that does not exist."""

    KIND: ClassVar[str] = "machine.block_boundary"

    job: JobId
    block: int
    padding_tokens: int


# ---------------------------------------------------------------------------
# Protected Mode
# ---------------------------------------------------------------------------


@event_class
@dataclass(frozen=True)
class SegmentOpened(Event):
    KIND: ClassVar[str] = "segment.opened"

    job: JobId
    segment: SegmentId
    ring: Ring
    integrity: Integrity
    perms: Perm


@event_class
@dataclass(frozen=True)
class SegmentClosed(Event):
    KIND: ClassVar[str] = "segment.closed"

    job: JobId
    segment: SegmentId
    start: int
    end: int


@event_class
@dataclass(frozen=True)
class PermsChanged(Event):
    KIND: ClassVar[str] = "segment.perms"

    job: JobId
    segment: SegmentId
    from_perms: Perm
    to_perms: Perm


@event_class
@dataclass(frozen=True)
class MaskUpdated(Event):
    """The allowed-block bitmap -- the MMU. Changes land only at
    block boundaries, which is what bounds mask churn."""

    KIND: ClassVar[str] = "mask.updated"

    job: JobId
    allowed_blocks: int
    denied_blocks: int


@event_class
@dataclass(frozen=True)
class AttentionDenied(Event):
    """Hard enforcement firing: the machine refused an attention the job attempted.

    Evidence for acceptance criterion 3 -- masking must be asserted by the machine
    refusing, not by the job declining.
    """

    KIND: ClassVar[str] = "mask.denied"

    job: JobId
    segment: SegmentId


@event_class
@dataclass(frozen=True)
class IntegrityDemoted(Event):
    """Biba low-water-mark: reading dirt makes you dirty.

    ``because`` names the segments whose attention mass crossed θ_read -- which is
    what makes a later PRIVILEGE_FAULT explainable rather than mysterious.
    """

    KIND: ClassVar[str] = "integrity.demoted"

    job: JobId
    from_integrity: Integrity
    to_integrity: Integrity
    because: tuple[SegmentId, ...]


@event_class
@dataclass(frozen=True)
class CapabilityChecked(Event):
    """Effects are syscalls: every pipe write is checked."""

    KIND: ClassVar[str] = "capability.checked"

    job: JobId
    pipe: PipeName
    effective_integrity: Integrity
    min_integrity: Integrity
    allowed: bool


@event_class
@dataclass(frozen=True)
class Endorsed(Event):
    """The only integrity-raising operation. Schema width is the
    security dial, so it is journalled."""

    KIND: ClassVar[str] = "integrity.endorsed"

    job: JobId
    endorser: DescriptorName
    segment: SegmentId
    from_integrity: Integrity
    to_integrity: Integrity
    schema: str


# ---------------------------------------------------------------------------
# Virtual Context
# ---------------------------------------------------------------------------


@event_class
@dataclass(frozen=True)
class SegmentEvicted(Event):
    KIND: ClassVar[str] = "vm.evicted"

    job: JobId
    segment: SegmentId
    stub: SegmentId
    store: StoreId
    freed_tokens: int
    stub_tokens: int
    policy: str


@event_class
@dataclass(frozen=True)
class ResidencyChanged(Event):
    KIND: ClassVar[str] = "vm.residency"

    job: JobId
    segment: SegmentId
    from_residency: Residency
    to_residency: Residency


@event_class
@dataclass(frozen=True)
class PageFaultRaised(Event):
    """Explicit fault: the model referenced a stub handle, or emitted a NEED.

    Servicing is ordinary pipe I/O -- the job blocks and therefore costs nothing
    while it waits.
    """

    KIND: ClassVar[str] = "vm.fault"

    job: JobId
    explicit: bool
    segment: SegmentId | None
    need_text: str | None


@event_class
@dataclass(frozen=True)
class PagedIn(Event):
    KIND: ClassVar[str] = "vm.paged_in"

    job: JobId
    segment: SegmentId
    store: StoreId
    plan: str  # "append" | "splice"
    cost_tokens: int


@event_class
@dataclass(frozen=True)
class Refaulted(Event):
    """A fault on a segment evicted within the last F blocks. Feeds thrash
    detection."""

    KIND: ClassVar[str] = "vm.refault"

    job: JobId
    segment: SegmentId
    blocks_since_evict: int


@event_class
@dataclass(frozen=True)
class WorkingSetSampled(Event):
    KIND: ClassVar[str] = "vm.working_set"

    job: JobId
    size_tokens: int
    segments: int


@event_class
@dataclass(frozen=True)
class MapRefreshed(Event):
    """Status-region tail refresh. Resume revalidation is a special case of map
    invalidation."""

    KIND: ClassVar[str] = "vm.map_refreshed"

    job: JobId
    obj: ObjectName
    segment: SegmentId
    cost_tokens: int


@event_class
@dataclass(frozen=True)
class StatusRegionRetracted(Event):
    """A superseded status region was removed from the window rather than left to the
    pager.

    The accompanying ``Spliced`` carries the token arithmetic, as it does for eviction;
    this event carries *why* the span went, which no other event in the alphabet can
    express -- ``SegmentEvicted`` would claim a store span and a stub handle that do
    not exist. ``invalidated_downstream_tokens`` is the price paid, and is zero
    whenever the region was still the tail.
    """

    KIND: ClassVar[str] = "vm.status_region_retracted"

    job: JobId
    obj: ObjectName
    segment: SegmentId
    freed_tokens: int
    invalidated_downstream_tokens: int


# ---------------------------------------------------------------------------
# World state
# ---------------------------------------------------------------------------


@event_class
@dataclass(frozen=True)
class WorldWritten(Event):
    KIND: ClassVar[str] = "world.written"

    job: JobId | None
    obj: ObjectName
    before: str
    after: str


# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------


@event_class
@dataclass(frozen=True)
class FaultRaised(Event):
    """Every enforcement action that changes behaviour emits one of these -- the
    "fail loudly" rule."""

    KIND: ClassVar[str] = "fault.raised"

    job: JobId
    fault: FaultKind
    detail: str
    segment: SegmentId | None = None
    pipe: PipeName | None = None


@event_class
@dataclass(frozen=True)
class FaultDispatched(Event):
    KIND: ClassVar[str] = "fault.dispatched"

    job: JobId
    fault: FaultKind
    policy: str
    handler: DescriptorName | None = None


@event_class
@dataclass(frozen=True)
class SpoofDetected(Event):
    """Attempted mimicry of control framing. The imposter is already inert by
    construction; this exists because attempted injection is worth alarming on even
    when it fails."""

    KIND: ClassVar[str] = "security.spoof"

    job: JobId
    pipe: PipeName
    rendered: str


@event_class
@dataclass(frozen=True)
class Note(Event):
    """Free-form kernel annotation. Deliberately last, and deliberately rare: a
    fact worth asserting on deserves its own event type."""

    KIND: ClassVar[str] = "note"

    text: str
    tags: tuple[str, ...] = field(default_factory=tuple)


# --- ZEOS-NLI: instruction as compilation ------------------------------------
#
# NLI wants the journal to answer "who spoke, what was compiled, which gate
# answered". These events exist so that answer can be assembled from the record
# rather than reconstructed from a transcript -- which is the difference between an
# audit trail and a story.


@event_class
@dataclass(frozen=True)
class UtteranceReceived(Event):
    """A human said something. The words are data on a pipe, carrying a principal.

    ``ring`` is the honest field (OQ-N4): an unauthenticated open-air microphone is a
    ring-3 source, and recording that here is what lets a later question about a
    compilation's trustworthiness be answered rather than assumed.
    """

    KIND: ClassVar[str] = "nli.utterance"

    principal: PrincipalId
    text: str
    pipe: PipeName
    ring: Ring
    platform: str = ""


@event_class
@dataclass(frozen=True)
class UtteranceCompiled(Event):
    """The artifact. A thing that can be read, which is what makes echo-back
    possible and audit meaningful."""

    KIND: ClassVar[str] = "nli.compiled"

    principal: PrincipalId
    target: str  # invocation | mission | synthesis | reflex
    artifact: str
    descriptor: DescriptorName | None = None


@event_class
@dataclass(frozen=True)
class CompilationRefused(Event):
    """Nothing was dispatched, and the reason is named.

    The distinction between "no compilation target" and every other refusal is the
    one worth keeping: it means the request was not denied, it was *unheard* -- the
    addressability property.
    """

    KIND: ClassVar[str] = "nli.refused"

    principal: PrincipalId
    text: str
    reason: str
    detail: str = ""


@event_class
@dataclass(frozen=True)
class AuthorityNarrowed(Event):
    """Layer one. The job runs at the speaker's envelope, not the
    descriptor's ambition.

    Emitted even when nothing was withheld, because "this speaker had full authority
    for this behaviour" is a fact a post-mortem needs as much as its opposite.
    """

    KIND: ClassVar[str] = "nli.authority_narrowed"

    job: JobId
    principal: PrincipalId
    held: tuple[PipeName, ...]
    withheld: tuple[PipeName, ...]


@event_class
@dataclass(frozen=True)
class CeilingApplied(Event):
    """Layer two. "NOW, drop everything" is a request."""

    KIND: ClassVar[str] = "nli.ceiling_applied"

    principal: PrincipalId
    descriptor: DescriptorName
    requested: Priority
    granted: Priority


@event_class
@dataclass(frozen=True)
class SpawnRefused(Event):
    """A spawn that would have outranked the principal's ceiling.

    The compiler clamps, so reaching this means something bypassed the compiler. Kept
    as the second of two layers for the same reason the lint and the capability
    boundary both exist: one catches the design error, the other catches the one
    nobody linted.
    """

    KIND: ClassVar[str] = "nli.spawn_refused"

    principal: PrincipalId
    descriptor: DescriptorName
    requested: Priority
    ceiling: Priority
    reason: str = ""


@event_class
@dataclass(frozen=True)
class EchoedBack(Event):
    """What the speaker was told before dispatch."""

    KIND: ClassVar[str] = "nli.echo"

    principal: PrincipalId
    text: str
    awaiting_confirmation: bool = False


@event_class
@dataclass(frozen=True)
class Elevated(Event):
    """Scoped, time-boxed, and **loud**.

    Loudness is a design requirement, not telemetry: "optionally a broadcast so
    humans present know the envelope changed". A silent elevation is
    indistinguishable from a compromised one.
    """

    KIND: ClassVar[str] = "nli.elevated"

    principal: PrincipalId
    capabilities: tuple[PipeName, ...]
    expires_at: int
    authorised_by: PrincipalId
    reason: str = ""


@event_class
@dataclass(frozen=True)
class ElevationEnded(Event):
    """Auto-revert. An elevation that has to be handed back is one that stays open."""

    KIND: ClassVar[str] = "nli.elevation_ended"

    principal: PrincipalId
    capabilities: tuple[PipeName, ...]
    reason: str = "expired"


@event_class
@dataclass(frozen=True)
class ElevationRefused(Event):
    KIND: ClassVar[str] = "nli.elevation_refused"

    principal: PrincipalId
    requested: tuple[PipeName, ...]
    authorised_by: PrincipalId
    reason: str = ""


@event_class
@dataclass(frozen=True)
class OwnershipApplied(Event):
    """A principal acted on its own job."""

    KIND: ClassVar[str] = "nli.ownership"

    principal: PrincipalId
    op: str
    job: JobId
    detail: str = ""


@event_class
@dataclass(frozen=True)
class OwnershipRefused(Event):
    """A principal tried to act on someone else's job.

    Not a special case for safety handlers: the kernel owns those, so this is the
    same check that stops a visitor cancelling the plant manager's mission.
    """

    KIND: ClassVar[str] = "nli.ownership_refused"

    principal: PrincipalId
    op: str
    job: JobId
    owner: PrincipalId
    reason: str = ""


@event_class
@dataclass(frozen=True)
class GateConsulted(Event):
    """Layer four. An intended actuation was held and shown to its guard."""

    KIND: ClassVar[str] = "nli.gate_consulted"

    job: JobId
    pipe: PipeName
    gate: DescriptorName
    gate_job: JobId
    payload: str


@event_class
@dataclass(frozen=True)
class GateAnswered(Event):
    """Allowed or vetoed, with the reason, and **which gate answered**.

    Both outcomes are journaled. A record that only kept vetoes would make the gate
    look like it fires rarely rather than like it is consulted every time.
    """

    KIND: ClassVar[str] = "nli.gate_answered"

    job: JobId
    pipe: PipeName
    gate: DescriptorName
    allowed: bool
    reason: str = ""


# --- ZEOS-Distributed: the link -----------------------------------------------
#
# The distribution design makes partition an ordinary world-state change with an
# ordinary handler, so
# these events describe the *transport*, not a new scheduling concept. The kernel
# never polls a link; it learns about arrivals as pipe writes, exactly as it learns
# about a sensor.


@event_class
@dataclass(frozen=True)
class LinkStateChanged(Event):
    """The link came up or went down.

    > **A partition is a suspension; a reconnection is a resume.**

    Journalled rather than merely reflected in ``link.state`` because the world write
    says *what* is true and this says *when it changed*, which is what a resume diff
    is computed against.
    """

    KIND: ClassVar[str] = "link.state"

    node: str
    peer: str
    up: bool
    reason: str = ""
    in_flight_lost: int = 0


@event_class
@dataclass(frozen=True)
class FrameSent(Event):
    """Tokens handed to the link for the peer. Not yet delivered -- that is the point."""

    KIND: ClassVar[str] = "link.sent"

    pipe: PipeName
    #: not ``seq`` -- that name belongs to the journal record envelope, as the codec
    #: guard points out the moment you forget. Fourth time it has caught this.
    frame: int
    tokens: int
    arrives_at_ns: int


@event_class
@dataclass(frozen=True)
class FrameDelivered(Event):
    KIND: ClassVar[str] = "link.delivered"

    pipe: PipeName
    frame: int
    tokens: int
    latency_ns: int
    ring: Ring


@event_class
@dataclass(frozen=True)
class FrameDropped(Event):
    """Lost to injected loss or to a partition that ate it in flight."""

    KIND: ClassVar[str] = "link.dropped"

    pipe: PipeName
    frame: int
    reason: str


@event_class
@dataclass(frozen=True)
class ReplicaRefreshed(Event):
    """A non-authoritative copy was updated from its authority.

    ``age_ns`` is what the reading job is entitled to be told: a plan that would be
    safe against fresh state may not be safe against state that is 400 ms old.
    """

    KIND: ClassVar[str] = "link.replica"

    obj: ObjectName
    authority: str
    value: str
    age_ns: int


@event_class
@dataclass(frozen=True)
class StalenessRefused(Event):
    """A read of a replica older than the descriptor's ``max_staleness``.

    Fails loudly rather than returning a stale value, per the standing rule. The
    alternative -- quietly serving state the job declared it could not use -- is the
    failure mode the declaration exists to prevent.
    """

    KIND: ClassVar[str] = "link.stale"

    job: JobId
    obj: ObjectName
    age_ns: int
    max_staleness_ns: int
