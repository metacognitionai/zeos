# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Principals: who is asking, and what they are allowed to cause.

The kernel had no answer to "whose job is this?". Every job had a descriptor, a
priority, and a capability table taken wholesale from that descriptor -- which is
fine while the only thing that can start a job is a boot file written by an
engineer, and useless the moment a human can ask for one. ZEOS-NLI puts four
hard gates in front of an utterance and the first two are **authority** and
**priority**; neither has anything to attach to without this module.

The model is deliberately the one from multi-user operating systems, because that
is where the property comes from:

> A dangerous instruction fails the way ``rm -rf /`` fails for a non-root user --
> permission denied, not moral disapproval. The model can be talked into *wanting*
> to comply. It cannot be talked into being *able* to.

Three pieces:

* a **principal envelope** -- an identity plus the authority it carries: which
  pipes it may cause writes to, the most urgent priority it may ask for, and the
  ring its utterances arrive at;
* **elevation** -- sudo. Scoped, time-boxed, loud, and re-authenticated. The
  dangerous operation was never impossible, it was gated on authority instead of
  rhetoric;
* **ownership** -- principals may cancel or deprioritise their *own* jobs and not
  anyone else's, which is what makes "actually, stop" a safe thing to say.

**Two different things are called "principal" in this codebase**, and conflating
them would be a real bug. ``ids.Principal`` is a provenance *class* -- kernel, user,
tool, device, peer job -- stamped onto every INJECT so that content's ring can be
assigned from where it came from. ``PrincipalId`` here is an
*identity*: ``badge:hengel-a``. A pipe has a class; an utterance has both. The class
decides what ring the words are content at; the identity decides what the compiled
job may do.

**Authority is intersected, never unioned.** A compiled job holds the capabilities
its descriptor declares *and* its owner's envelope permits. The descriptor says what
the behaviour needs; the envelope says what this speaker may cause. Taking the union
would mean a visitor could run the barrier-override behaviour by asking for it by
name, which is exactly the property NLI exists to deny.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from zeos.core.clock import Clock
from zeos.core.events import KERNEL_OWNER as _EVENTS_KERNEL_OWNER
from zeos.core.ids import Integrity, PipeName, PrincipalId, Priority, Ring

__all__ = [
    "PrincipalEnvelope",
    "Elevation",
    "PrincipalTable",
    "KERNEL_PRINCIPAL",
    "SAFETY_TIER",
    "CompilationTarget",
]

#: The kernel's own identity. Boot jobs and vector-dispatched handlers are owned by
#: it, which is what lets the ownership rule be stated without exceptions: a user
#: cannot cancel a safety handler because they do not own it, not because the
#: kernel special-cases safety handlers.
KERNEL_PRINCIPAL = PrincipalId("kernel")

#: ``events.py`` sits below this module and carries its own copy, so that a journal
#: event can default to the kernel without importing the principal model. They must
#: agree, and this is where that is checked rather than assumed.
assert KERNEL_PRINCIPAL == _EVENTS_KERNEL_OWNER

#: Priorities at or above this urgency (numerically at or below) are the safety
#: tier. Per ZEOS-NLI: "No user-spawned job can outrank the safety tier, so
#: danger-by-urgency -- making the system too busy obeying to stay safe -- is
#: inexpressible." A policy dial, and a load-time check enforces it against every
#: declared ceiling rather than trusting the site to get it right.
SAFETY_TIER = Priority(10)


class CompilationTarget:
    """How far up the compilation ladder a principal may go.

    > Deployments choose how far up this ladder they enable. An industrial site may
    > permit invocation only; a research platform may permit synthesis with human
    > co-sign above a capability threshold.

    Ordered, and the order is the point: invocation is the safest because the
    executable vocabulary of the system stays the reviewed descriptor library.
    """

    INVOCATION = "invocation"
    MISSION = "mission"
    SYNTHESIS = "synthesis"

    ALL = (INVOCATION, MISSION, SYNTHESIS)

    @staticmethod
    def rank(target: str) -> int:
        try:
            return CompilationTarget.ALL.index(target)
        except ValueError:
            return -1

    @staticmethod
    def permits(allowed: str, wanted: str) -> bool:
        return CompilationTarget.rank(wanted) <= CompilationTarget.rank(allowed)


@dataclass(frozen=True, slots=True)
class Elevation:
    """A time-boxed grant of named capabilities.

    Scoped, not root: ``capabilities`` is an explicit list of pipes, so "elevate to
    open the barrier" cannot become "elevate to everything". Time-boxed: ``expires_at``
    is a token-clock deadline, and the kernel reverts it without being asked, because
    an elevation that has to be handed back is an elevation that stays open.
    """

    principal: PrincipalId
    capabilities: frozenset[PipeName]
    expires_at: int
    reason: str = ""
    #: Whoever authorised it. Elevation is "an explicit act by an authorised
    #: principal, re-authenticated at request time" -- recording who is half of what
    #: makes it loud.
    authorised_by: PrincipalId = KERNEL_PRINCIPAL

    def active_at(self, clock: Clock) -> bool:
        return clock.token_clock < self.expires_at

    def render(self) -> str:
        pipes = ", ".join(sorted(str(p) for p in self.capabilities))
        return f"{self.principal} +[{pipes}] until t={self.expires_at}"


@dataclass(frozen=True, slots=True)
class PrincipalEnvelope:
    """An identity and the authority it carries.

    ``ring`` is the honest part. OQ-N4:

    > an unauthenticated open-air microphone is a **ring-3 source**, and its
    > compilations deserve the ring-3 envelope.

    So authentication strength is not a boolean that gates entry; it is the ring the
    words arrive at, which flows through the ordinary integrity machinery. A badge
    reader is ring 2; a microphone anyone can shout into is ring 3, and a job
    compiled from it starts demoted.
    """

    id: PrincipalId
    #: The most urgent priority this principal may ask for. Numerically *larger* is
    #: less urgent, so a ceiling is a lower bound on the number.
    ceiling: Priority = Priority(100)
    #: Pipes this principal may cause writes to. A compiled job's capabilities are
    #: intersected with this.
    capabilities: frozenset[PipeName] = frozenset()
    ring: Ring = Ring.EXTERNAL
    integrity: Integrity = Integrity(3)
    #: How far up the compilation ladder this principal may go.
    max_target: str = CompilationTarget.INVOCATION
    #: Descriptors this principal may invoke. Empty means "any in the library" --
    #: authority still narrows what the invocation can *do*, so this is a
    #: convenience for tight deployments rather than the load-bearing check.
    invocable: frozenset[str] = frozenset()
    #: May authorise elevation for others.
    may_elevate: bool = False
    #: Bypasses the intersection entirely -- the root case.
    #:
    #: Root in a real OS is not "has every capability enumerated", it is "the check
    #: does not apply", and modelling it as an enumeration would be wrong in a way
    #: that bites: a capability added to a descriptor next year would silently fail
    #: for the kernel's own boot jobs until someone remembered to widen a set. The
    #: lint warns about any *non*-kernel principal declared this way, because a site
    #: that grants it to a human has opted out of layer one.
    unrestricted: bool = False
    label: str = ""

    def describe(self) -> str:
        return (
            f"{self.id} (ring {int(self.ring)}, ceiling {int(self.ceiling)}, "
            f"{len(self.capabilities)} capabilit{'y' if len(self.capabilities) == 1 else 'ies'})"
        )

    def may_invoke(self, descriptor: str) -> bool:
        return not self.invocable or descriptor in self.invocable


@dataclass
class PrincipalTable:
    """The declared principals, plus whatever elevations are currently live.

    Holds no policy of its own beyond the intersection rule. Which principals exist
    and what they may do is site configuration -- ``system/principals.yaml`` -- for the
    same reason the allocator is a policy module: NLI claims no contribution in
    identity infrastructure (OQ-N4), only in what the kernel does with it.
    """

    principals: dict[PrincipalId, PrincipalEnvelope] = field(
        default_factory=dict[PrincipalId, PrincipalEnvelope]
    )
    elevations: dict[PrincipalId, Elevation] = field(default_factory=dict[PrincipalId, Elevation])

    def __init__(self, envelopes: Iterable[PrincipalEnvelope] = ()) -> None:
        self.principals = {}
        self.elevations = {}
        for envelope in envelopes:
            self.declare(envelope)
        if KERNEL_PRINCIPAL not in self.principals:
            # The kernel owns boot jobs and vector-dispatched handlers. Without an
            # envelope for it, every ownership and ceiling check would need a
            # special case for "no owner", and special cases in an authority check
            # are where authority checks go wrong.
            self.declare(
                PrincipalEnvelope(
                    id=KERNEL_PRINCIPAL,
                    ceiling=Priority(0),
                    ring=Ring.KERNEL,
                    integrity=Integrity(0),
                    max_target=CompilationTarget.SYNTHESIS,
                    may_elevate=True,
                    unrestricted=True,
                    label="the kernel itself",
                )
            )

    def declare(self, envelope: PrincipalEnvelope) -> None:
        self.principals[envelope.id] = envelope

    def has(self, principal: PrincipalId) -> bool:
        return principal in self.principals

    def get(self, principal: PrincipalId) -> PrincipalEnvelope:
        envelope = self.principals.get(principal)
        if envelope is None:
            raise KeyError(f"unknown principal {principal!r}")
        return envelope

    def names(self) -> tuple[PrincipalId, ...]:
        return tuple(sorted(self.principals))

    def all(self) -> tuple[PrincipalEnvelope, ...]:
        return tuple(self.principals[n] for n in self.names())

    # -- elevation -----------------------------------------------------------

    def elevate(self, elevation: Elevation) -> None:
        self.elevations[elevation.principal] = elevation

    def revoke_elevation(self, principal: PrincipalId) -> Elevation | None:
        return self.elevations.pop(principal, None)

    def expired(self, clock: Clock) -> tuple[Elevation, ...]:
        """Elevations whose lease has run out. The kernel reverts these unasked."""
        return tuple(
            self.elevations[p]
            for p in sorted(self.elevations)
            if not self.elevations[p].active_at(clock)
        )

    def granted_capabilities(self, principal: PrincipalId, *, at: Clock) -> frozenset[PipeName]:
        """Everything this principal may currently cause a write to: their standing
        envelope, plus any live elevation.

        Meaningless for an unrestricted principal -- ask ``narrow`` instead, which
        knows that the check does not apply rather than trying to enumerate it.
        """
        base = self.get(principal).capabilities
        elevation = self.elevations.get(principal)
        if elevation is not None and elevation.active_at(at):
            return base | elevation.capabilities
        return base

    # -- the two hard gates --------------------------------------------------

    def clamp_priority(self, principal: PrincipalId, wanted: Priority) -> Priority:
        """Apply the ceiling. "NOW, drop everything" is a request.

        Numerically: a smaller number is more urgent, so the ceiling is a floor on
        the value. Returns the priority the job will actually run at.
        """
        ceiling = self.get(principal).ceiling
        return Priority(max(int(wanted), int(ceiling)))

    def narrow(
        self,
        principal: PrincipalId,
        wanted: Sequence[PipeName],
        *,
        at: Clock,
    ) -> tuple[tuple[PipeName, ...], tuple[PipeName, ...]]:
        """Intersect a descriptor's declared capabilities with the speaker's.

        Returns ``(held, withheld)``. Both halves matter: the job runs with ``held``,
        and ``withheld`` is what the journal and the echo-back report, so a refusal
        can say *which* authority was missing rather than only that something was.
        """
        if self.get(principal).unrestricted:
            return tuple(wanted), ()
        allowed = self.granted_capabilities(principal, at=at)
        held = tuple(p for p in wanted if p in allowed)
        withheld = tuple(p for p in wanted if p not in allowed)
        return held, withheld


def ceiling_violations(
    table: PrincipalTable, *, safety_tier: Priority = SAFETY_TIER
) -> tuple[str, ...]:
    """Principals whose ceiling would let them outrank the safety tier.

    A load-time check, not a runtime one, because a site that has declared such a
    ceiling has already lost the property -- every compilation from that principal
    would be legal. The kernel principal is exempt: it *is* the safety tier.
    """
    return tuple(
        f"{e.id}: ceiling {int(e.ceiling)} is at or above the safety tier "
        f"({int(safety_tier)}); no user principal may outrank it"
        for e in table.all()
        if e.id != KERNEL_PRINCIPAL and int(e.ceiling) <= int(safety_tier)
    )


def unrestricted_principals(table: PrincipalTable) -> tuple[str, ...]:
    """Non-kernel principals that bypass the authority check.

    Sometimes legitimate -- a commissioning console, a site's own boot author -- and
    always worth saying out loud, because such a principal has opted out of the NLI design's
    first layer and every compilation from it runs at full authority.
    """
    return tuple(
        f"{e.id}: declared unrestricted; layer one (authority) does not apply to it"
        for e in table.all()
        if e.id != KERNEL_PRINCIPAL and e.unrestricted
    )


def unknown_capabilities(table: PrincipalTable, *, declared: Sequence[PipeName]) -> tuple[str, ...]:
    """Principal capabilities naming pipes that do not exist.

    Silent in effect -- an envelope granting ``actuators.fan_4`` where only
    ``unit_a`` exists simply never matches -- and that silence is the problem: it
    reads as authority granted and behaves as authority withheld.
    """
    if not declared:
        return ()
    known = set(declared)
    return tuple(
        f"{e.id}: grants unknown pipe {str(pipe)!r}"
        for e in table.all()
        for pipe in sorted(e.capabilities)
        if pipe not in known
    )
