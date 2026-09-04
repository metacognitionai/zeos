# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Nodes, object authority, and the link between them.

The core design describes one kernel on one node. That is enough for supervisory
control of fixed infrastructure, where the compute can sit wherever is convenient,
and it is not enough for robots, for one reason with no way around it: the
interrupt-latency budget targets tens of milliseconds for a pinned handler, and no
off-platform link meets that.

This module holds the declarations that make the split decidable rather than a
matter of taste:

* a **node** is a scheduling domain -- its own priority space, ready queue, and
  suspension stack. Priorities are not comparable across nodes and no attempt
  is made to make them so;
* an **object authority** is which node owns the truth for a piece of world state.
  Authority follows the sensors and actuators: ``robot.*`` is authoritative where
  the robot is;
* a **link** is characterised by one number the network owns, ``rtt_p99``, which is
  compared against a number the safety engineer owns, a vector's deadline. That
  comparison is the whole placement rule.

**RTT_p99, not mean.** The placement rule is explicit, and the reason is worth keeping in view: a
handler whose budget is met on average and missed at the tail has not met its
budget. Radio-linked platforms have heavy-tailed latency distributions, so the
honest consequence is that more runs on-platform than a mean-latency analysis would
suggest.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from zeos.core.ids import ObjectName, PipeName, Ring
from zeos.machine.base import Token
from zeos.world.store import ObjectSet

__all__ = [
    "LinkPort",
    "NodeSpec",
    "LinkSpec",
    "Topology",
    "PLATFORM",
    "OFFBOARD",
    "LINK_STATE",
]


@runtime_checkable
class LinkPort(Protocol):
    """What the kernel needs from a link, and nothing more.

    Declared here rather than imported from ``transport`` because ``core`` depends on
    the standard library, ``core``, ``world`` and ``descriptor/schema`` and nothing
    else -- and because this *is* the whole contract. The kernel hands tokens toward a
    peer and asks whether a pipe is its business. Everything else about a transport --
    latency, loss, partition, polling -- belongs to the driver, which is the only side
    that does I/O.
    """

    @property
    def spec(self) -> LinkSpec: ...

    def carries(self, pipe: PipeName) -> bool: ...

    def deliver(self, pipe: PipeName, tokens: Sequence[Token]) -> int: ...


#: Conventional node names. ZEOS-Distributed is written throughout for the two-node
#: case; N>2 is ZEOS-Fleet's problem, and Fleet answers it by keeping exactly one
#: deliberative tier -- so "the other side" stays singular even in a fleet.
PLATFORM = "platform"
OFFBOARD = "offboard"

#: The world object carrying link health. The design makes this an ordinary interrupt
#: source with an ordinary handler descriptor, which is the whole trick: partition
#: needs no new mechanism because "the link went down" is just a world-state change
#: that a vector can fire on.
LINK_STATE = ObjectName("link.state")


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """One scheduling domain.

    ``window_tokens`` is what makes the placement load-time check possible: a
    ``placement: platform`` descriptor whose declared working set exceeds the local
    model class's window is rejected, which is the existing admission rule (tech
    spec) applied per node rather than globally.
    """

    name: str
    model_class: str = "large"
    #: World state this node owns the truth for. Authority follows the sensors and
    #: actuators that produce it.
    authoritative_objects: ObjectSet = field(default_factory=ObjectSet)
    #: Context window of this node's model class, in tokens. 0 means "unbounded",
    #: which is the honest default for the deliberative tier at M0.
    window_tokens: int = 0

    def owns(self, obj: ObjectName) -> bool:
        return self.authoritative_objects.matches(obj)

    def describe(self) -> str:
        return (
            f"{self.name} ({self.model_class}, "
            f"authoritative over {self.authoritative_objects.render() or 'nothing'})"
        )


@dataclass(frozen=True, slots=True)
class LinkSpec:
    """The link between two nodes.

    ``ring`` is a deployment declaration checked at load time, exactly as pipe rings
    already are. A federated peer is ring 2 *only* if both ends are under the
    same operator's authority and the channel is authenticated; otherwise ring 3.
    Getting this wrong is the one genuinely new attack surface distribution
    introduces, so the default is the paranoid one.
    """

    rtt_p99_ns: int = 0
    ring: Ring = Ring.EXTERNAL

    #: One-way delivery delay. Half the round trip, which is the useful figure for
    #: a transport that models each direction separately.
    @property
    def one_way_ns(self) -> int:
        return self.rtt_p99_ns // 2

    def describe(self) -> str:
        return f"rtt_p99={self.rtt_p99_ns}ns, ring {int(self.ring)}"


@dataclass
class Topology:
    """Which nodes exist, what each owns, and what connects them.

    Deliberately a *declaration* rather than something discovered. The placement rule makes
    placement arithmetic over two declared numbers; discovering either at runtime
    would put a scheduling decision behind a measurement that can move, and the
    whole point is that the placement of a safety handler is a load-time property.
    """

    nodes: dict[str, NodeSpec] = field(default_factory=dict[str, NodeSpec])
    link: LinkSpec = field(default_factory=LinkSpec)

    def __init__(self, nodes: Iterable[NodeSpec] = (), link: LinkSpec | None = None) -> None:
        self.nodes = {}
        for node in nodes:
            self.nodes[node.name] = node
        self.link = link or LinkSpec()

    def has(self, name: str) -> bool:
        return name in self.nodes

    def get(self, name: str) -> NodeSpec:
        node = self.nodes.get(name)
        if node is None:
            raise KeyError(f"unknown node {name!r}")
        return node

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.nodes))

    def all(self) -> tuple[NodeSpec, ...]:
        return tuple(self.nodes[n] for n in self.names())

    def peer_of(self, node: str) -> str:
        """The other side. Two-node by construction; fleets generalise this."""
        others = [n for n in self.names() if n != node]
        if len(others) != 1:
            raise ValueError(
                f"peer_of expects exactly two nodes, found {self.names()}; "
                "N>2 is ZEOS-Fleet's problem, and it answers it with one "
                "deliberative tier rather than a mesh of peers"
            )
        return others[0]

    # -- authority -----------------------------------------------------------

    def authority_for(self, obj: ObjectName) -> str | None:
        """Which node owns the truth for this object, if any is declared.

        Deterministic when two nodes both claim it -- sorted by name -- but that is a
        load-time error, not a tie to be broken quietly. See ``authority_conflicts``.
        """
        for node in self.all():
            if node.owns(obj):
                return node.name
        return None

    def is_replica(self, node: str, obj: ObjectName) -> bool:
        """True when ``node`` reads this object from a replica rather than owning it.

        An object nobody claims is *not* a replica: it is local-only state, which is
        the normal case for anything the topology does not mention.
        """
        authority = self.authority_for(obj)
        return authority is not None and authority != node

    def authority_conflicts(self) -> tuple[str, ...]:
        """Objects claimed by more than one node.

        "Each object has exactly one authoritative node." Two claimants is not a
        precedence question -- it means two nodes will each believe they hold the
        truth, and a write on one will be silently overwritten by a refresh from the
        other.

        ``ObjectSet.intersects`` is deliberately conservative about wildcards, which
        is the right bias here for the same reason it is right for write conflicts:
        a false positive is a review conversation, a false negative is an unflagged
        race over who owns the truth.
        """
        conflicts: list[str] = []
        nodes = self.all()
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                if a.authoritative_objects.intersects(b.authoritative_objects):
                    conflicts.append(
                        f"{a.name} ({a.authoritative_objects.render()}) and "
                        f"{b.name} ({b.authoritative_objects.render()}) both claim "
                        "authority over the same state; each object needs exactly "
                        "one owner"
                    )
        return tuple(conflicts)

    def describe(self) -> str:
        return f"{len(self.nodes)} nodes [{', '.join(self.names())}], link {self.link.describe()}"


def unplaceable_vectors(
    *,
    deadlines: Sequence[tuple[str, int, str]],
    link: LinkSpec,
) -> tuple[str, ...]:
    """Vectors whose deadline the link cannot meet.

    Each entry is ``(vector name, deadline_ns, handler placement)``. The rule is one
    comparison: a handler placed off-platform is reachable only after a round trip,
    so a deadline tighter than ``rtt_p99`` is unmeetable by construction -- not
    unlikely, not load-dependent, unmeetable.
    """
    if not link.rtt_p99_ns:
        return ()
    return tuple(
        f"{name}: deadline {deadline}ns is under the link's rtt_p99 "
        f"({link.rtt_p99_ns}ns), but its handler is placed offboard"
        for name, deadline, placement in deadlines
        if deadline and deadline < link.rtt_p99_ns and placement == "offboard"
    )
