# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The process control block.

Descriptor + transcript + kernel metadata = the PCB. KV is a *cache* of the PCB,
not part of it (core §2.1) -- which is why nothing here refers to KV blocks, and why
a job survives a model swap: the transcript is portable state, the KV is a
materialisation of it for one specific model.

The read/write sets carried here are the *observed* ones, accumulated from what the
job actually touched. The union of declared and observed is what resume
invalidation uses (core §3) -- declared sets will be wrong, and the kernel is
expected to notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zeos.core.capabilities import CapabilityTable
from zeos.core.clock import Clock
from zeos.core.gates import GateRequest
from zeos.core.ids import (
    DescriptorName,
    Integrity,
    JobId,
    JobState,
    PipeName,
    PrincipalId,
    Priority,
    ResourceName,
    SegmentId,
    VectorName,
)
from zeos.core.principals import KERNEL_PRINCIPAL
from zeos.core.residency import ContextPolicy, ThrashMonitor
from zeos.core.segments import SegmentTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.base import Token
from zeos.world.store import ObjectSet

__all__ = ["Job"]


@dataclass
class Job:
    """One transformer-mediated task in flight."""

    job_id: JobId
    descriptor: Descriptor
    #: Spawn order. Breaks priority ties FIFO (core Appendix A, rule 1) -- and
    #: because it is an integer rather than a set-iteration accident, tie-breaking
    #: is deterministic and therefore replayable.
    seq: int
    state: JobState = JobState.READY
    spawned_at: Clock = field(default_factory=Clock)

    #: Static base priority from the descriptor, and the possibly-inherited
    #: effective priority. Inheritance is temporary; base never moves.
    base_priority: Priority = Priority(500)
    current_priority: Priority = Priority(500)
    inherited_from: JobId | None = None

    #: Whose job this is. The kernel owns boot jobs and vector-dispatched
    #: handlers, which is what lets the ownership rule have no exceptions: a user
    #: cannot cancel a safety handler because they do not own it, not because safety
    #: handlers are special-cased.
    owner: PrincipalId = KERNEL_PRINCIPAL
    parent: JobId | None = None
    children: list[JobId] = field(default_factory=list[JobId])
    #: Set when this job is a handler instance dispatched by a vector, so the
    #: kernel can release the vector's active count on completion.
    vector: VectorName | None = None

    tokens_used: int = 0
    blocked_on: PipeName | None = None
    blocked_reason: str = ""
    #: Clock at which the job was suspended, for the "suspended 94s" line and for
    #: computing which writes landed while it was gone.
    suspended_at: Clock | None = None
    preempt_count: int = 0

    observed_reads: ObjectSet = field(default_factory=ObjectSet)
    observed_writes: ObjectSet = field(default_factory=ObjectSet)

    #: A pipe operation the job started but could not complete, retried when it
    #: wakes. Without this a backpressured write would have to be either dropped
    #: (data loss) or partially applied (torn payload) -- both worse than parking it.
    pending_write: tuple[PipeName, tuple[object, ...]] | None = None
    #: A blocking READ the job is parked on, completed when it wakes. It has to be
    #: its own field rather than being recovered from ``blocked_on`` /
    #: ``blocked_reason``, because ``Scheduler.wake`` clears both on the way out of
    #: BLOCKED -- so a woken reader that had only those to go on would be dispatched
    #: with its read never completed, leaving the token it waited for sitting in the
    #: pipe. Every other parked operation already has such a field for the same reason.
    pending_read: PipeName | None = None
    #: Pipes a SELECT is waiting on; the job wakes when any becomes readable.
    pending_select: tuple[PipeName, ...] = ()
    #: A resource acquisition the job is parked on, retried when it wakes.
    pending_acquire: ResourceName | None = None
    #: An actuation held while its action gate decides. Reuses
    #: ``pending_write`` for the payload -- a write waiting for a verdict and a write
    #: waiting for buffer space are the same situation from the job's side.
    pending_gate: GateRequest | None = None
    #: A pipe whose gate has already answered *allow* for the write now parked in
    #: ``pending_write``. Without it the retried write walks back into its own gate
    #: and the two spin forever -- the gate says yes, the write restarts, the gate is
    #: consulted again. Cleared once the write actually lands, so a *second*
    #: actuation on the same pipe is gated afresh: a verdict is about one action, not
    #: a standing permission.
    gate_cleared: PipeName | None = None
    #: Resources currently held. Released on completion or fault, because a job
    #: that died holding a lock would block every waiter forever.
    held_resources: set[ResourceName] = field(default_factory=set[ResourceName])

    #: True once the kernel has appended the descriptor body as ring-1 content.
    started: bool = False
    #: Source-pipe payload taken when this job's vector fired, injected at `_start_job`
    #: so it lands after the body rather than ahead of it.
    vector_payload: tuple[Token, ...] = ()

    # -- Protected Mode ------------------------------------------------------

    #: The segment table. Kernel state, part of the PCB -- never part of the context,
    #: which is what makes provenance unforgeable rather than merely conventional.
    segments: SegmentTable = field(default_factory=lambda: SegmentTable(16))
    #: Biba watermark. Starts at the descriptor's declared level and only ever rises
    #: (worsens) until a compartment, endorser, or fork-and-discard resets it.
    current_integrity: Integrity = Integrity(2)
    capabilities: CapabilityTable = field(default_factory=CapabilityTable)
    #: Integrity floor while serving a request from a lower-trust pipe -- the
    #: confused-deputy rule, applied for the duration of one request.
    session_floor: Integrity | None = None
    #: Segments this job may attend when it is a compartment child. Empty means
    #: "everything of its own", which is the normal case.
    grants: tuple[SegmentId, ...] = ()
    compartment_of: JobId | None = None
    #: Refault accounting, and the loud fault it eventually raises.
    thrash: ThrashMonitor = field(default_factory=lambda: ThrashMonitor(policy=ContextPolicy()))
    #: Highest block index for which boundary work has run. Boundary work is
    #: batched here rather than per token because that is where the specs put it --
    #: mask churn, watermark demotion, and eviction all land at boundaries.
    last_block: int = -1

    @property
    def name(self) -> DescriptorName:
        return self.descriptor.name

    @property
    def effective_reads(self) -> ObjectSet:
        """Declared ∪ observed -- what resume invalidation is computed against."""
        return self.descriptor.reads | self.observed_reads

    @property
    def effective_writes(self) -> ObjectSet:
        return self.descriptor.writes | self.observed_writes

    @property
    def is_pinned(self) -> bool:
        return self.descriptor.pinned

    @property
    def is_preemptible(self) -> bool:
        """``preemptible: false`` is interrupt masking -- the ``cli``/``sti``
        equivalent, and just as dangerous, which is why the lint caps its budget."""
        return self.descriptor.preemptible

    def outranks(self, other: Job) -> bool:
        """Strictly higher priority, ties broken FIFO by spawn order.

        Strictness matters: an equal-priority job must not preempt, or two
        same-priority jobs would ping-pong at every boundary and make no progress.
        """
        if self.current_priority != other.current_priority:
            return self.current_priority < other.current_priority
        return False

    def budget_exceeded(self, now: Clock) -> str | None:
        """Returns a description of the breached budget, or None."""
        budget = self.descriptor.budget
        if budget.tokens is not None and self.tokens_used > budget.tokens:
            return f"token budget {budget.tokens} exceeded ({self.tokens_used} used)"
        if budget.deadline_ns is not None:
            elapsed = now.elapsed_ns_since(self.spawned_at)
            if elapsed > budget.deadline_ns:
                return f"deadline {budget.deadline_ns}ns missed (elapsed {elapsed}ns)"
        return None

    def record_read(self, patterns: ObjectSet) -> None:
        self.observed_reads = self.observed_reads | patterns

    def record_write(self, patterns: ObjectSet) -> None:
        self.observed_writes = self.observed_writes | patterns
