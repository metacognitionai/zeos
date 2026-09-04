# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Bounded token buffers, and the scheduling primitive that eliminates polling.

A pipe is two things at once (core §4), and conflating them deliberately is the
whole trick: it is a channel between jobs, *and* it is how the kernel knows when a
job has nothing to do. A job that blocks on a read runs no forward passes and its
KV becomes eligible for page-out; a write to an empty pipe is a wake event for its
reader. One mechanism serves dataflow, tool completion, and interrupts.

Backpressure is not an error path. A bounded buffer that blocks its writer gives
automatic rate matching between producers and consumers of different speeds, with
zero logic in either descriptor.

Pipes are addressed by **name** and resolved through a transport, so a job cannot
observe whether its peer is in-process or on another node. Phase 1 ships only
``LocalTransport``; this is the seam that makes federation an addition rather than
a redesign.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import JobId, PipeName, Principal, Ring
from zeos.machine.base import Token

__all__ = ["PipeSpec", "Pipe", "PipeTable", "PipeError", "DEFAULT_CAPACITY"]

DEFAULT_CAPACITY = 4096


class PipeError(RuntimeError):
    """Structural misuse of a pipe -- an unknown name, or a capacity of zero."""


@dataclass(frozen=True, slots=True)
class PipeSpec:
    """A pipe's declaration. Both ends must agree on ring and principal, and the
    kernel refuses mismatches at load time."""

    name: PipeName
    ring: Ring = Ring.TRUSTED
    principal: Principal = Principal.PEER_JOB
    capacity_tokens: int = DEFAULT_CAPACITY
    transport: str = "local"
    #: Device pipes are written by adapters rather than by jobs. Recorded so that a
    #: reader/writer reachability lint could tell "nobody writes this" from "the world
    #: writes this" -- no such rule exists yet, so this is currently journalling and
    #: provenance only.
    device: bool = False
    #: Actuator pipes: a write here changes world state. This is the mirror of
    #: "device drivers are adapters that turn external events into pipe writes"
    #: (core §4.3) -- the outbound direction, where a pipe write becomes an effect.
    #: It is what lets a job's writes show up in another job's resume diff.
    world_object: str | None = None


@dataclass
class Pipe:
    """Runtime state of one pipe."""

    spec: PipeSpec
    buffer: deque[Token] = field(default_factory=deque[Token])
    #: Jobs blocked on read, in arrival order. FIFO so that wake order is
    #: deterministic -- the scheduler still decides who *runs*, but who becomes
    #: runnable must not depend on set iteration order.
    waiting_readers: deque[JobId] = field(default_factory=deque[JobId])
    waiting_writers: deque[JobId] = field(default_factory=deque[JobId])
    #: Total tokens ever written. Feeds vector coalescing and telemetry.
    total_written: int = 0

    @property
    def name(self) -> PipeName:
        return self.spec.name

    @property
    def available(self) -> int:
        return len(self.buffer)

    @property
    def free(self) -> int:
        return max(0, self.spec.capacity_tokens - len(self.buffer))

    @property
    def readable(self) -> bool:
        """A read unblocks on **any** available token, not on a full request
        (core §4.1)."""
        return bool(self.buffer)

    def writable(self, n: int = 1) -> bool:
        return self.free >= n

    def latch(self, tokens: Sequence[Token]) -> int:
        """Replace the buffer with ``tokens``: the current value, not a history.

        For a pipe backed by a ``world_object``. A write to one of those is an *effect*
        rather than a message -- the world already holds the result, and the buffer is
        only there so the value can be read back. Keeping a backlog instead makes
        capacity a countdown: nothing obliges a reader to exist, and the only wake for a
        blocked writer is another job's read, so a job that actuates once a turn parks
        against its own buffer and is never woken.
        """
        self.buffer.clear()
        return self.write(tokens)

    def write(self, tokens: Sequence[Token]) -> int:
        """Append what fits. Returns the number accepted; a short write means the
        writer must block for the remainder (backpressure)."""
        accepted = min(len(tokens), self.free)
        for token in tokens[:accepted]:
            self.buffer.append(token)
        self.total_written += accepted
        return accepted

    def read(self, n: int | None = None) -> tuple[Token, ...]:
        """Take up to ``n`` tokens (all available if ``None``)."""
        count = self.available if n is None else min(n, self.available)
        return tuple(self.buffer.popleft() for _ in range(count))

    def peek(self) -> tuple[Token, ...]:
        return tuple(self.buffer)

    def block_reader(self, job: JobId) -> None:
        if job not in self.waiting_readers:
            self.waiting_readers.append(job)

    def block_writer(self, job: JobId) -> None:
        if job not in self.waiting_writers:
            self.waiting_writers.append(job)

    def take_waiting_readers(self) -> tuple[JobId, ...]:
        woken = tuple(self.waiting_readers)
        self.waiting_readers.clear()
        return woken

    def take_waiting_writers(self) -> tuple[JobId, ...]:
        woken = tuple(self.waiting_writers)
        self.waiting_writers.clear()
        return woken

    def unblock(self, job: JobId) -> None:
        """Remove a job from both wait queues -- used when it is cancelled."""
        if job in self.waiting_readers:
            self.waiting_readers.remove(job)
        if job in self.waiting_writers:
            self.waiting_writers.remove(job)


class PipeTable:
    """All pipes in one kernel, by name."""

    def __init__(self, specs: Iterable[PipeSpec] = ()) -> None:
        self._pipes: dict[PipeName, Pipe] = {}
        for spec in specs:
            self.declare(spec)

    def declare(self, spec: PipeSpec) -> Pipe:
        if spec.capacity_tokens < 1:
            raise PipeError(f"pipe {spec.name}: capacity must be >= 1")
        existing = self._pipes.get(spec.name)
        if existing is not None:
            # Both ends declaring the same pipe is normal; disagreeing is not.
            if existing.spec.ring != spec.ring or existing.spec.principal != spec.principal:
                raise PipeError(
                    f"pipe {spec.name}: conflicting declarations "
                    f"({existing.spec.ring.name}/{existing.spec.principal.value} vs "
                    f"{spec.ring.name}/{spec.principal.value})"
                )
            return existing
        pipe = Pipe(spec=spec)
        self._pipes[spec.name] = pipe
        return pipe

    def get(self, name: PipeName) -> Pipe:
        pipe = self._pipes.get(name)
        if pipe is None:
            raise PipeError(f"unknown pipe {name!r}")
        return pipe

    def has(self, name: PipeName) -> bool:
        return name in self._pipes

    def ensure(self, name: PipeName) -> Pipe:
        """Get, declaring a default trusted pipe if absent.

        Used for pipes that appear in a descriptor but not in ``system/pipes.yaml``.
        The lint reports these; the kernel still runs, because failing to start over
        an undeclared report pipe would be a worse default than running with a
        conservative one.
        """
        if name not in self._pipes:
            self.declare(PipeSpec(name=name))
        return self._pipes[name]

    def names(self) -> tuple[PipeName, ...]:
        return tuple(sorted(self._pipes))

    def all(self) -> tuple[Pipe, ...]:
        return tuple(self._pipes[n] for n in self.names())

    def __contains__(self, name: PipeName) -> bool:
        return name in self._pipes

    def __len__(self) -> int:
        return len(self._pipes)
