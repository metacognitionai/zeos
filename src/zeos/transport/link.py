# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The link: an in-process transport with injected latency and injected loss.

ZEOS-Distributed's D1 milestone, exactly as specified:

> Two kernel instances with distinct priority spaces bridged by an in-process
> transport with injected latency and injected loss. Validates the design's claims (no cross-link
> preemption, replica staleness, partition as suspension) with no
> networking involved -- **and, being deterministic, is replayable, which a real
> network is not.**

That last clause is the reason to build this before D2 rather than after. A real
network reproduces a partition bug approximately; this reproduces it exactly, and
the difference is whether "the reconnection resumed with the wrong dirty set" is a
regression test or an anecdote.

**Determinism therefore constrains the design.** Latency is a function of the
virtual clock the driver supplies, and loss is either an explicit schedule of frame
indices or a seeded counter -- never ``random``. A transport that consulted a global
RNG would make every federated journal unreproducible, which would cost more than
the realism it bought.

**Nothing here is on the kernel's side of the seam.** The kernel writes to a pipe;
if that pipe is carried by the link, the frame lands in this queue; the *driver*
advances the queue and delivers what has arrived. The kernel does no I/O and reads
no clock, so a job still cannot observe which transport carries its pipe -- which is
the constraint ``transport/base.py`` exists to protect.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import PipeName, Principal, Ring
from zeos.core.topology import LinkSpec
from zeos.machine.base import Token

__all__ = ["Frame", "LinkTransport", "LossModel", "PartitionError"]


class PartitionError(RuntimeError):
    """Raised only by explicit inspection helpers, never by ``deliver``.

    A partitioned link does not *error* on write -- the design is specific that the
    off-platform side sees "a job whose pipes went quiet", which is a job blocking
    on a read that does not return. Raising here would turn a partition into a fault
    at the writer, which is a different and much less useful failure mode.
    """


@dataclass(frozen=True, slots=True)
class Frame:
    """One delivery in flight.

    Carries its provenance because protection requires it: traffic arriving from the peer
    enters through INJECT like everything else, so the ring and principal must cross
    the link with the tokens or provenance stops being total at the node boundary.
    """

    pipe: PipeName
    tokens: tuple[Token, ...]
    #: Virtual nanosecond at which this becomes visible to the far side.
    arrives_at_ns: int
    #: Monotonic per-link counter. The identity a loss schedule refers to, and the
    #: tiebreak that keeps delivery order deterministic when two frames land in the
    #: same nanosecond.
    seq: int
    ring: Ring = Ring.EXTERNAL
    principal: Principal = Principal.PEER_JOB
    sent_at_ns: int = 0

    def render(self) -> str:
        return f"#{self.seq} {self.pipe} ({len(self.tokens)} tok) @{self.arrives_at_ns}ns"


@dataclass
class LossModel:
    """Which frames the link eats. Deterministic by construction.

    Two shapes, both reproducible:

    * ``drop_seqs`` -- an explicit set of frame numbers. Use this in tests: it says
      exactly which delivery is lost, so the assertion reads as the scenario rather
      than as a consequence of a seed.
    * ``drop_every`` -- every Nth frame. Use this for soak runs where the *pattern*
      matters more than the individual loss.

    There is no probabilistic mode on purpose. A seeded RNG would be reproducible
    too, but only as long as nobody changed the number of calls before it -- and
    "the loss pattern moved because an unrelated frame was added upstream" is a
    debugging experience worth designing out.
    """

    drop_seqs: frozenset[int] = frozenset()
    drop_every: int = 0

    def drops(self, seq: int) -> bool:
        if seq in self.drop_seqs:
            return True
        return bool(self.drop_every) and seq % self.drop_every == 0

    def __bool__(self) -> bool:
        return bool(self.drop_seqs or self.drop_every)


@dataclass
class LinkTransport:
    """One direction of the link, from this node toward its peer.

    Each node holds its own outbound ``LinkTransport``; the pair is bridged by the
    federated driver. Modelling one direction per object rather than a single
    bidirectional pipe keeps asymmetric loss expressible -- which matters, because a
    radio link that can hear but not speak is a real failure mode and a symmetric
    model cannot represent the platform continuing to report while ignoring
    instructions.
    """

    spec: LinkSpec
    #: Pipes this transport is responsible for. Empty means "none" rather than
    #: "all": the local transport is the catch-all, and a link that silently claimed
    #: every pipe would make a misconfigured topology look like a working one.
    pipes: frozenset[PipeName] = frozenset()
    loss: LossModel = field(default_factory=LossModel)
    #: False while partitioned. Frames written during a partition are dropped, not
    #: queued -- a link that buffered indefinitely and flushed on reconnection would
    #: deliver a burst of stale instructions to a robot that has moved on.
    up: bool = True

    _in_flight: list[Frame] = field(default_factory=list[Frame])
    _next_seq: int = 1
    _sent: int = 0
    _dropped: int = 0
    _delivered: int = 0
    _now_ns: int = 0

    @property
    def name(self) -> str:
        return "link"

    def carries(self, pipe: PipeName) -> bool:
        return pipe in self.pipes

    # -- the sending side ----------------------------------------------------

    def deliver(self, pipe: PipeName, tokens: Sequence[Token]) -> int:
        """Accept tokens for transmission. Returns the count accepted.

        Always accepts in full when the link is up: there is no cross-link
        backpressure in D1, because backpressure is a property of the far pipe's
        buffer and learning about it costs a round trip. A real transport would
        carry a credit scheme; the honest D1 position is that this is unmodelled,
        and the tests do not depend on it.
        """
        if not self.carries(pipe):
            raise ValueError(f"link does not carry pipe {pipe!r}")
        seq = self._next_seq
        self._next_seq += 1
        self._sent += 1

        if not self.up or self.loss.drops(seq):
            self._dropped += 1
            return 0

        self._in_flight.append(
            Frame(
                pipe=pipe,
                tokens=tuple(tokens),
                arrives_at_ns=self._now_ns + self.spec.one_way_ns,
                seq=seq,
                ring=self.spec.ring,
                sent_at_ns=self._now_ns,
            )
        )
        return len(tokens)

    # -- the receiving side --------------------------------------------------

    def advance(self, now_ns: int) -> None:
        """Move virtual time forward. Called by the driver, never the kernel."""
        if now_ns < self._now_ns:
            raise ValueError(f"link clock moved backwards: {self._now_ns} -> {now_ns}")
        self._now_ns = now_ns

    def poll(self) -> Iterable[Frame]:
        """Frames that have arrived. Sorted, so delivery order is reproducible."""
        arrived = [f for f in self._in_flight if f.arrives_at_ns <= self._now_ns]
        if not arrived:
            return ()
        remaining = [f for f in self._in_flight if f.arrives_at_ns > self._now_ns]
        self._in_flight = remaining
        arrived.sort(key=lambda f: (f.arrives_at_ns, f.seq))
        self._delivered += len(arrived)
        return arrived

    # -- partition -----------------------------------------------------------

    def partition(self) -> int:
        """Cut the link. Returns how many in-flight frames were lost with it.

        In-flight frames are discarded rather than held: they were on the wire when
        the wire failed. Keeping them would model a link that pauses and resumes,
        which is a different and much friendlier failure than the one the safety
        case has to survive.
        """
        lost = len(self._in_flight)
        self._in_flight = []
        self._dropped += lost
        self.up = False
        return lost

    def restore(self) -> None:
        self.up = True

    # -- telemetry -----------------------------------------------------------

    @property
    def in_flight(self) -> int:
        return len(self._in_flight)

    @property
    def stats(self) -> tuple[int, int, int]:
        """``(sent, delivered, dropped)``."""
        return (self._sent, self._delivered, self._dropped)

    def describe(self) -> str:
        sent, delivered, dropped = self.stats
        state = "up" if self.up else "PARTITIONED"
        return (
            f"link {state}: {sent} sent, {delivered} delivered, {dropped} dropped, "
            f"{self.in_flight} in flight"
        )
