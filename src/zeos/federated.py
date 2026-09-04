# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Two kernels, one process -- ZEOS-Distributed's D1.

> Two kernel instances with distinct priority spaces bridged by an in-process
> transport with injected latency and injected loss. Validates the design's claims (no cross-link
> preemption, replica staleness, partition as suspension) with no
> networking involved -- **and, being deterministic, is replayable, which a real
> network is not.**

This module is the harness, and it lives beside ``driver.py`` rather than in ``core``
because advancing a link and moving frames between two kernels is I/O in every sense
that matters: it is where time is supplied and where arrival order is decided.

**The kernels do not know about each other.** Each holds its own scheduler, its own
suspension stack, and its own priority space; neither can preempt the other, because
preemption is defined at a token boundary and a token boundary is an event on one
node's machine. The only thing crossing is a pipe write, which arrives an RTT late
and lands as ordinary input. That is not a limitation to engineer away -- it is the
same structure as a classical OS, where you do not preempt a process on another
machine, you send it a message.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from zeos.core.clock import Clock
from zeos.core.events import FrameDelivered, FrameDropped
from zeos.core.ids import ObjectName, PipeName
from zeos.core.kernel import Kernel
from zeos.core.topology import Topology
from zeos.machine.base import render
from zeos.transport.link import LinkTransport

__all__ = ["Federation", "Node"]


@dataclass
class Node:
    """One scheduling domain: a kernel plus the outbound half of its link."""

    name: str
    kernel: Kernel
    outbound: LinkTransport

    @property
    def clock(self) -> Clock:
        return self.kernel.clock


@dataclass
class Federation:
    """Drives two kernels and the link between them.

    Deterministic by construction: virtual time advances in fixed steps supplied
    here, delivery order is sorted, and loss is an explicit schedule rather than a
    random draw. Same inputs, same journals, byte for byte -- which is what makes a
    partition bug a regression test instead of an anecdote.
    """

    topology: Topology
    nodes: dict[str, Node] = field(default_factory=dict[str, Node])
    #: Virtual nanoseconds per federation step. The link's one-way delay is measured
    #: against this, so it is what makes an RTT mean anything in an M0 run.
    ns_per_step: int = 1_000_000
    _now_ns: int = 0

    def add(self, node: Node) -> None:
        self.nodes[node.name] = node

    def node(self, name: str) -> Node:
        return self.nodes[name]

    @property
    def now_ns(self) -> int:
        return self._now_ns

    # -- the loop ------------------------------------------------------------

    def step(self) -> bool:
        """Advance one quantum on every node, then move whatever the link delivered.

        Returns False when nothing ran anywhere and nothing is in flight -- the
        federated analogue of quiescence. In flight matters: a federation with an
        empty ready set but a frame on the wire is not finished, it is waiting.
        """
        self._now_ns += self.ns_per_step
        ran = False

        for name in sorted(self.nodes):
            node = self.nodes[name]
            node.kernel.advance_to(self._now_ns)
            node.outbound.advance(self._now_ns)
            if node.kernel.tick():
                ran = True

        delivered = self._pump()
        return ran or delivered > 0 or self._in_flight() > 0

    def _pump(self) -> int:
        """Deliver arrived frames to the far kernel as ordinary pipe writes."""
        moved = 0
        for name in sorted(self.nodes):
            source = self.nodes[name]
            try:
                peer_name = self.topology.peer_of(name)
            except ValueError:
                continue
            peer = self.nodes.get(peer_name)
            if peer is None:
                continue
            for frame in source.outbound.poll():
                # Provenance survives the hop: the frame carries the ring the
                # link was declared at, and enters the far kernel through the
                # ordinary delivery path so it is INJECTed like any other input.
                peer.kernel.deliver(frame.pipe, render(frame.tokens))
                peer.kernel._emit(  # pyright: ignore[reportPrivateUsage]
                    FrameDelivered(
                        clock=peer.kernel.clock,
                        pipe=frame.pipe,
                        frame=frame.seq,
                        tokens=len(frame.tokens),
                        latency_ns=self._now_ns - frame.sent_at_ns,
                        ring=frame.ring,
                    )
                )
                moved += 1
        return moved

    def _in_flight(self) -> int:
        return sum(n.outbound.in_flight for n in self.nodes.values())

    def run(self, steps: int = 2000) -> int:
        """Run until nothing is runnable and nothing is on the wire."""
        for i in range(steps):
            if not self.step():
                return i
        raise RuntimeError(
            f"federation did not settle in {steps} steps; a script is probably "
            "missing its 'exit', or a frame is looping"
        )

    # -- partition -----------------------------------------------------------

    def partition(self, *, reason: str = "link lost") -> int:
        """Cut the link in both directions.

        In-flight frames are discarded, not held. They were on the wire when the wire
        failed; keeping them would model a link that pauses and resumes, which is a
        friendlier failure than the one the safety case has to survive.
        """
        lost = 0
        for name in sorted(self.nodes):
            node = self.nodes[name]
            dropped = node.outbound.partition()
            lost += dropped
            if dropped:
                node.kernel._emit(  # pyright: ignore[reportPrivateUsage]
                    FrameDropped(
                        clock=node.kernel.clock,
                        pipe=PipeName("*"),
                        frame=0,
                        reason=f"{dropped} frame(s) in flight when the link failed",
                    )
                )
            node.kernel.set_link_state(up=False, reason=reason, lost=dropped)
        return lost

    def restore(self, *, reason: str = "link restored") -> None:
        """Reconnect. A reconnection is a resume -- and the resume is the ordinary
        one, computed by each kernel against its own suspended jobs."""
        for name in sorted(self.nodes):
            node = self.nodes[name]
            node.outbound.restore()
            node.kernel.set_link_state(up=True, reason=reason)

    # -- replication ---------------------------------------------------------

    def replicate(self, obj: ObjectName) -> None:
        """Push an authoritative value to the node that only holds a replica.

        Called by whoever owns the refresh schedule. Deliberately explicit rather
        than automatic: the refresh interval is part of the staleness budget a
        descriptor declares against, so hiding it would hide the thing being measured.
        """
        authority = self.topology.authority_for(obj)
        if authority is None or authority not in self.nodes:
            return
        value = self.nodes[authority].kernel.world.get(obj)
        for name in sorted(self.nodes):
            if name == authority:
                continue
            self.nodes[name].kernel.accept_replica(obj, value, authority=authority)

    def journals(self) -> dict[str, Sequence[object]]:
        """Per-node journals. Separate on purpose -- each node is its own history."""
        return {n: list(self.nodes[n].kernel.events) for n in sorted(self.nodes)}
