# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The dispatcher: compile, gate, echo back, dispatch.

An ordinary descriptor in the real design -- "pinned at moderate priority in the fleet
kernel". Here it is a kernel-side function for the same reason the allocator's default
is a plain class: N0 is about whether the *gates* hold, and putting the dispatcher
inside a job would add a scheduling story without adding a safety one. What it must
share with the eventual job version is that it holds no authority of its own; it reads
an envelope and asks the kernel.

The ordering of the five layers is the substance of this module, so it is worth
restating why it is an ordering rather than a set. The layers run hardest-first, and
the first four never consult the model's opinion:

1. **Authority** -- the compiled job runs at the *speaker's* capability envelope.
   Eloquence is not an input to this check.
2. **Priority** -- the ceiling is configuration. "NOW, drop everything" is a request.
3. **Addressability** -- some things have no compilation target at all.
4. **Device gates** -- schemas, rate limits, interlocks, and action gates on the way
   out.
5. **Judgment** -- the dispatcher may refuse, confirm, or escalate. Valuable,
   trainable, and explicitly not load-bearing.

Addressability is checked during compilation (there is nothing to dispatch), device
gates are checked at the write boundary (that is where the actuation is), and this
module owns authority and priority. Judgment appears here as ``confirm`` and as the
right to refuse -- deliberately last, deliberately soft, and deliberately not the thing
the safety case rests on.

**Authority narrowing is not refusal.** A job whose speaker lacks the barrier
capability is dispatched *without* it, and faults when it reaches for the actuator.
That is the faithful reading of the authority layer -- "the job does not hold the capability and the
write raises a capability fault" -- and it is also the stronger demonstration: the
barrier stays shut because of the capability boundary, not because the dispatcher had
an opinion. A dispatcher that refused up front would be doing the work of layer five
and hiding whether layer one works at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import DescriptorName, JobId, PipeName, PrincipalId, Priority
from zeos.core.principals import PrincipalEnvelope
from zeos.nli.compiler import Artifact, ArtifactKind, Phrasing, compile_utterance
from zeos.nli.envelope import Utterance

__all__ = [
    "Decision",
    "OwnershipOp",
    "OwnershipRequest",
    "decide",
    "echo_back",
]


class OwnershipOp:
    """What a principal may do to a job it owns.

    > Ownership works as in any multi-user OS: principals may cancel or deprioritise
    > *their own* jobs… they cannot touch the plant manager's mission.

    ``DEPRIORITISE`` only ever moves a job to a *less* urgent priority. The op that
    would raise urgency is spawn-with-priority, which the ceiling already governs;
    letting deprioritise run in reverse would be a second, ungoverned path to the
    same power.
    """

    CANCEL = "cancel"
    DEPRIORITISE = "deprioritise"

    ALL = (CANCEL, DEPRIORITISE)


@dataclass(frozen=True, slots=True)
class OwnershipRequest:
    op: str
    job: JobId
    by: PrincipalId
    priority: Priority | None = None

    def render(self) -> str:
        if self.op == OwnershipOp.DEPRIORITISE and self.priority is not None:
            return f"{self.op} job {self.job} to priority {int(self.priority)}"
        return f"{self.op} job {self.job}"


@dataclass(frozen=True, slots=True)
class Decision:
    """The dispatcher's output: an artifact, the authority it will run with, and the
    priority it will run at.

    Both halves of the narrowing are kept. ``withheld`` is what makes an echo-back
    able to say *which* authority was missing, and what makes the journal usable in
    a post-mortem -- "refused" alone is an answer nobody can act on.
    """

    artifact: Artifact
    descriptor: DescriptorName | None = None
    #: Capabilities the job will hold: declared ∩ envelope.
    held: tuple[PipeName, ...] = ()
    #: Declared but not permitted to this speaker.
    withheld: tuple[PipeName, ...] = ()
    #: Priority after the ceiling.
    priority: Priority | None = None
    #: Priority asked for, when the ceiling moved it.
    requested_priority: Priority | None = None
    #: Set when layer five declined. Not load-bearing.
    judgment: str = ""
    references: tuple[str, ...] = ()

    @property
    def clamped(self) -> bool:
        return (
            self.priority is not None
            and self.requested_priority is not None
            and int(self.priority) != int(self.requested_priority)
        )

    @property
    def dispatch(self) -> bool:
        return self.artifact.dispatched and not self.judgment

    @property
    def needs_confirmation(self) -> bool:
        return self.dispatch and self.artifact.confirm


def decide(
    utterance: Utterance,
    *,
    envelope: PrincipalEnvelope,
    phrasings: Sequence[Phrasing],
    declared_capabilities: Mapping[DescriptorName, Sequence[PipeName]],
    granted: frozenset[PipeName],
    default_priority: Mapping[DescriptorName, Priority] = {},
    judgment: Mapping[str, str] = {},
) -> Decision:
    """Compile an utterance and apply layers one and two.

    ``granted`` is passed in rather than read from the envelope so that a live
    elevation participates through the ordinary mechanism. The design is specific that
    the elevated capability "attaches to the principal's envelope, so subsequently
    compiled jobs hold it through the ordinary mechanism -- no special path through
    the dispatcher, nothing for persuasion to target." A separate elevation branch
    here would be exactly that special path.
    """
    artifact = compile_utterance(utterance, envelope=envelope, phrasings=phrasings)
    if not artifact.dispatched:
        return Decision(artifact=artifact)

    invocation = artifact.invocation
    descriptor = invocation.descriptor if invocation is not None else None
    if descriptor is None:
        return Decision(artifact=artifact)

    declared = tuple(declared_capabilities.get(descriptor, ()))
    held = tuple(p for p in declared if p in granted)
    withheld = tuple(p for p in declared if p not in granted)

    wanted = artifact.requested_priority
    if wanted is None:
        wanted = default_priority.get(descriptor, envelope.ceiling)
    # Numerically larger is less urgent, so the ceiling is a floor on the value.
    effective = Priority(max(int(wanted), int(envelope.ceiling)))

    return Decision(
        artifact=artifact,
        descriptor=descriptor,
        held=held,
        withheld=withheld,
        priority=effective,
        requested_priority=wanted,
        judgment=judgment.get(str(descriptor), ""),
        references=tuple(r.render() for r in artifact.references),
    )


def echo_back(decision: Decision) -> str:
    """The sentence read back before dispatch.

    > "spawning beam-carry: two carriers, priority 40, est. 12 min, door-south will be
    > locked for 4 min"

    Possible only because the compilation is an artifact. The withheld capabilities
    are included deliberately: a speaker who is about to watch a job fail at the
    actuator should be told that before it starts, and telling them is a courtesy the
    guarantee does not depend on.
    """
    artifact = decision.artifact
    if artifact.kind == ArtifactKind.REFLEX:
        return f"reflex {artifact.reflex}: acting locally, not compiling"
    if artifact.refused:
        return f"cannot: {artifact.detail}"
    if decision.judgment:
        return f"declining: {decision.judgment}"

    parts = [artifact.render()]
    if decision.clamped and decision.priority is not None:
        assert decision.requested_priority is not None
        parts.append(
            f"priority {int(decision.requested_priority)} requested, "
            f"running at {int(decision.priority)} (your ceiling)"
        )
    if decision.withheld:
        parts.append(
            "without "
            + ", ".join(sorted(str(p) for p in decision.withheld))
            + " (not in your authority)"
        )
    if artifact.confirm:
        parts.append("confirm?")
    return "; ".join(parts)


@dataclass
class DispatchLog:
    """What was said, what it compiled to, and which gate answered.

    "Every attempt, allowed or blocked, is journaled with provenance: who spoke,
    what was compiled, which gate answered." This is the in-memory index over that;
    the journal remains the record.
    """

    entries: list[tuple[PrincipalId, str, str]] = field(
        default_factory=list[tuple[PrincipalId, str, str]]
    )

    def record(self, decision: Decision) -> None:
        self.entries.append(
            (
                decision.artifact.utterance.principal,
                decision.artifact.utterance.text,
                echo_back(decision),
            )
        )

    def by_principal(self, principal: PrincipalId) -> tuple[str, ...]:
        return tuple(e[2] for e in self.entries if e[0] == principal)
