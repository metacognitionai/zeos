# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Identifiers and cross-cutting enumerations.

A leaf module: it imports nothing else from ``zeos.core``. Every other core module
imports it, which is what keeps the core free of import cycles. Enums that are used
by exactly one module still live here when the journal alphabet needs to name them --
``events.py`` would otherwise have to import half the kernel.

Numeric conventions follow the specs and are load-bearing, not cosmetic:

* **Priority** -- lower is more urgent (RTOS convention, core §2.2). Arbitrary
  integers, not fixed tiers.
* **Ring** -- 0 is most privileged (Multics convention, MP §4).
* **Integrity** -- 0 is most trusted, and *rises* as trust falls, so the Biba
  low-water-mark rule is ``max()``. Deliberately matches the
  priority and ring conventions so all three read the same way.
"""

from __future__ import annotations

import enum
from typing import Final, NewType

__all__ = [
    "JobId",
    "SegmentId",
    "PipeName",
    "DescriptorName",
    "StoreId",
    "BlockId",
    "VectorName",
    "ObjectName",
    "ResourceName",
    "ResourceKind",
    "Priority",
    "Integrity",
    "JobState",
    "Ring",
    "Perm",
    "Residency",
    "SegmentState",
    "TokenKind",
    "Principal",
    "PrincipalId",
    "FaultKind",
    "Placement",
    "VectorPolicy",
    "OnComplete",
    "OnFault",
    "EvictionPolicy",
    "ResumeKind",
    "RING_COUNT",
    "KERNEL_PIPE",
]

JobId = NewType("JobId", int)
SegmentId = NewType("SegmentId", int)
BlockId = NewType("BlockId", int)

# Pipes are addressed by *name*, never by object reference or node address. This is
# the distribution seam: a name is resolved through a transport, and a job cannot
# observe whether the far end is in-process or on another node.
PipeName = NewType("PipeName", str)

DescriptorName = NewType("DescriptorName", str)
VectorName = NewType("VectorName", str)
ObjectName = NewType("ObjectName", str)  # namespaced world state, e.g. "plant.unit_a"
#: A holdable resource: a mutex, a semaphore, a doorway, or (at F0) a body.
ResourceName = NewType("ResourceName", str)
#: A principal *identity*, e.g. ``badge:hengel-a``. Distinct from
#: ``Principal`` below, which is a provenance *class*: a pipe has a class, an
#: utterance has both, and the two answer different questions -- the class decides
#: what ring the words are content at, the identity decides what the compiled job
#: may do.
PrincipalId = NewType("PrincipalId", str)
StoreId = NewType("StoreId", str)  # content-addressed span hash

Priority = NewType("Priority", int)
Integrity = NewType("Integrity", int)

RING_COUNT: Final = 4

#: The one pipe whose writes are ring 0 by construction.
KERNEL_PIPE: Final = PipeName("kernel")


class JobState(enum.StrEnum):
    """The six states of core Appendix A, plus PINNED_IDLE for resident handlers."""

    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"  # on a pipe read/write; deschedules, costs nothing
    SUSPENDED = "suspended"  # preempted, on the stack
    PINNED_IDLE = "pinned_idle"  # resident and prefilled, waiting to be addressed
    DONE = "done"
    FAULTED = "faulted"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.DONE, JobState.FAULTED)

    @property
    def is_schedulable(self) -> bool:
        """States from which the scheduler may dispatch."""
        return self is JobState.READY


class Ring(enum.IntEnum):
    """Privilege of *content*, assigned by the kernel from provenance -- never
    claimed by the content itself (MP §4)."""

    KERNEL = 0  # kernel notices, stubs' framing, vector preambles
    DESCRIPTOR = 1  # the descriptor body at spawn; nothing else, ever
    TRUSTED = 2  # pipes explicitly marked trusted; endorser output
    EXTERNAL = 3  # tools, web, sensors, third-party messages


class Perm(enum.Flag):
    """Per-segment permission bits."""

    NONE = 0
    R = enum.auto()  # attendable -- hard-enforced by the allowed-block bitmap
    X = enum.auto()  # directive -- advisory to the model, binding at the boundary
    W = enum.auto()  # rewritable region -- only valid on status regions
    P = enum.auto()  # pinned -- exempt from eviction

    @classmethod
    def parse(cls, text: str) -> Perm:
        """Parse a descriptor's permission string, e.g. ``"RX"`` or ``"R"``."""
        perms = cls.NONE
        for ch in text.upper():
            if ch in " ,-":
                continue
            member = cls.__members__.get(ch)
            if member is None:
                raise ValueError(f"unknown permission bit {ch!r} in {text!r}")
            perms |= member
        return perms

    def render(self) -> str:
        return (
            "".join(str(bit.name) for bit in (Perm.R, Perm.X, Perm.W, Perm.P) if bit in self) or "-"
        )


class Residency(enum.StrEnum):
    """Content residency. Orthogonal to KV residency, which the core
    kernel tracks separately -- the kernel owns both axes."""

    RESIDENT = "resident"  # in the window
    STUBBED = "stubbed"  # stub in window, content in the store, fault path back
    ARCHIVED = "archived"  # not in the window at all; pager/search only


class SegmentState(enum.StrEnum):
    OPEN = "open"  # still being extended -- the active output tail
    CLOSED = "closed"  # immutable


class TokenKind(enum.StrEnum):
    """M0 stands in for a real tokenizer with a token *kind*.

    Unforgeability of kernel framing is a property of the
    tokenizer, not of a scanner. With no real tokenizer to configure, the
    equivalent structural guarantee is that a job's token stream simply cannot
    carry CONTROL tokens unless the kernel has explicitly enabled them for that
    step. This maps directly onto tokenizer control-ID disabling and sampler-side
    reservation at MP1 -- rather than onto string matching, which would teach the
    wrong lesson.
    """

    NORMAL = "normal"
    CONTROL = "control"


class Principal(enum.StrEnum):
    """Who is behind a pipe. Stamped onto every INJECT."""

    KERNEL = "kernel"
    USER = "user"
    TOOL = "tool"
    DEVICE = "device"
    PEER_JOB = "peer_job"


class FaultKind(enum.StrEnum):
    """The fault taxonomy plus the core's scheduling faults.

    All of these dispatch through the same interrupt mechanism, so error handling
    is also just descriptors (core §6.3).
    """

    # Protected Mode
    ATTENTION = "attention_fault"
    PRIVILEGE = "privilege_fault"
    SPOOF = "spoof_fault"
    CAPABILITY = "capability_fault"
    X_VIOLATION = "x_violation"  # advisory/soft; telemetry by default
    # Core
    BUDGET = "budget_fault"
    DEADLINE = "deadline_fault"
    STARVATION = "scheduler_fault_starvation"
    TOOL_ERROR = "tool_error"
    DEADLOCK = "scheduler_fault_deadlock"
    # ZEOS-NLI
    #: An action gate refused an actuation on semantic grounds. Distinct
    #: from CAPABILITY on purpose: the job held the capability and the gate still
    #: said no, which is the whole reason gates exist. Collapsing the two would make
    #: "you may not touch this pipe" and "you may not do this particular thing with
    #: it" indistinguishable in the journal.
    GATE = "gate_veto"
    # Virtual Context
    THRASH = "scheduler_fault_thrash"
    ADMISSION = "admission_refused"


class ResourceKind(enum.StrEnum):
    """What a resource represents. Capacity does the mechanical work; the kind is
    what an operator reads in a fault, and what F0 dispatches on."""

    MUTEX = "mutex"  # capacity 1: a doorway, a crane, a work cell
    SEMAPHORE = "semaphore"  # capacity N: a corridor admitting two
    LEASE = "lease"  # an embodiment lease; revocable at F0


class Placement(enum.StrEnum):
    """Where a job is permitted to run.

    Phase 1 records, validates, and journals this but does not act on it: only
    ``LocalTransport`` ships. It exists now so that federation is a transport
    addition rather than a descriptor-format change.
    """

    PLATFORM = "platform"  # on-robot / on-site; latency-critical
    OFFBOARD = "offboard"  # datacentre; deliberative, latency-tolerant
    ANY = "any"


class VectorPolicy(enum.StrEnum):
    """Repeat-firing policy for an interrupt vector (core §5.1)."""

    QUEUE = "queue"  # serialise repeat firings
    COALESCE = "coalesce"  # collapse pending firings; read the latest value
    REENTRANT = "reentrant"  # parallel handler instances; disjoint r/w sets only


class OnComplete(enum.StrEnum):
    """What a handler does to the stack beneath it when it finishes (core §6.3)."""

    RETURN = "return"  # default: LIFO pop
    CANCEL_BELOW = "cancel-below"
    REPLACE_WITH = "replace-with"


class OnFault(enum.StrEnum):
    ESCALATE = "escalate"
    RETRY = "retry"
    ABORT = "abort"
    HANDLER = "handler"


class EvictionPolicy(enum.StrEnum):
    ATTENTION_CLOCK = "attention-clock"
    FIFO = "fifo"
    PINNED_ONLY = "pinned-only"


class ResumeKind(enum.StrEnum):
    """Which notice the kernel appends when popping a job off the stack."""

    CLEAN = "resume"  # dirty set empty; no revalidation cost
    DIRTY = "resume_dirty"  # carries the diff
