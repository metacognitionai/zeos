# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Compilation: from what someone said to an artifact that faces the gates.

> **Instructions are compiled, not obeyed.**

The compiler turns an utterance into a discrete, inspectable artifact -- an invocation
of an existing descriptor, a mission tree, or a candidate descriptor for the loader --
and that artifact then faces every gate any program faces.

**This module is deliberately the untrusted component**, and that is worth being
explicit about, because it is easy to mistake for a weakness. In N0 compilation is a
template match over descriptor-declared phrasings; at N1 it is a language model. The
demonstration NLI exists to make is not about the compiler being reliable -- it is
about what happens *after* compilation, which is why judgment is the fifth of five
layers and the first four "do not consult the model's opinion". A trivially fooled
compiler and a persuaded LLM produce the same class of artifact, and both meet the
same four gates. Making the N0 compiler cleverer would demonstrate nothing extra;
making it *trusted* would demonstrate nothing at all.

**Addressability is opt-in, per descriptor.** A descriptor becomes reachable by voice
only by declaring ``utterances:``. This is the structural form of addressability:

> "Disable collision avoidance" has no compilation target… the reflex tier contains
> no LLM -- there is nothing there to persuade. The most safety-critical layer is not
> protected from natural language; it is *deaf* to it.

An opt-in list makes deafness the default and hearing the declared exception, which
is the right way round: a blacklist of forbidden phrasings would have to anticipate
every way of asking, and the interesting attacks are the phrasings nobody
anticipated. There is nothing to anticipate if the only reachable descriptors are the
ones that put their hand up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import DescriptorName, ObjectName, Priority
from zeos.core.principals import CompilationTarget, PrincipalEnvelope
from zeos.nli.envelope import (
    InvocationSpec,
    MissionSpec,
    Phrasing,
    Reference,
    Utterance,
    safety_word,
)

__all__ = [
    "Phrasing",
    "Phrasebook",
    "Artifact",
    "ArtifactKind",
    "RefusalReason",
    "compile_utterance",
    "parse_phrasings",
]

#: Words that mean "the thing I am pointing at" and therefore need deixis.
#:
#: ``the`` is deliberately absent. It marks *definiteness*, not deixis: "the north
#: door" is a name, "that crate" is a gesture. Including it made every definite noun
#: phrase demand a visible object, so an operator saying "hold traffic on the north
#: door" was told nothing was visible. Found by using the console, not by a test.
DEMONSTRATIVES = ("that ", "this ", "those ", "these ")


class ArtifactKind:
    INVOCATION = "invocation"
    MISSION = "mission"
    SYNTHESIS = "synthesis"
    #: A safety word. Not a compilation at all -- it never became language.
    REFLEX = "reflex"
    REFUSAL = "refusal"


class RefusalReason:
    """Why nothing was dispatched. Named, because the journal has to say
    *which gate answered* -- "refused" on its own is the answer a jailbreak
    post-mortem cannot use."""

    #: The strongest refusal: there is no descriptor this could target.
    NO_TARGET = "no-compilation-target"
    #: The principal may not go this far up the compilation ladder.
    TARGET_NOT_PERMITTED = "target-not-permitted"
    #: A named descriptor exists but this principal may not invoke it.
    NOT_INVOCABLE = "not-invocable"
    #: OQ-N2: deixis could not be grounded, and guessing is worse than asking.
    AMBIGUOUS_REFERENCE = "ambiguous-reference"
    UNRESOLVED_REFERENCE = "unresolved-reference"
    #: The soft layer. Present, and explicitly not load-bearing.
    JUDGMENT = "dispatcher-judgment"


@dataclass(frozen=True, slots=True)
class Artifact:
    """The compilation. A thing that can be read, which is the whole point.

    Echo-back "is possible *because the compilation is an artifact*, a thing
    that can be read, not an intention smeared through a transcript."
    """

    kind: str
    utterance: Utterance
    invocation: InvocationSpec | None = None
    mission: MissionSpec | None = None
    reflex: str = ""
    #: Priority *requested*. The ceiling is applied by the dispatcher, not here, so
    #: that the journal can show the clamp happening rather than only its result.
    requested_priority: Priority | None = None
    confirm: bool = False
    refusal: str = ""
    detail: str = ""
    references: tuple[Reference, ...] = ()

    @property
    def dispatched(self) -> bool:
        return self.kind in (ArtifactKind.INVOCATION, ArtifactKind.MISSION)

    @property
    def refused(self) -> bool:
        return self.kind == ArtifactKind.REFUSAL

    def render(self) -> str:
        """What echo-back reads out. The design's example is a sentence, not a struct."""
        if self.kind == ArtifactKind.REFLEX:
            return f"reflex: {self.reflex}"
        if self.kind == ArtifactKind.REFUSAL:
            return f"refused ({self.refusal}): {self.detail}"
        if self.invocation is not None:
            parts = [f"spawning {self.invocation.render()}"]
            if self.requested_priority is not None:
                parts.append(f"priority {int(self.requested_priority)}")
            # Only pointing is worth reading back. ``'bench-3' → bench-3`` is noise,
            # and noise in an echo-back is how people stop reading echo-backs.
            for ref in self.references:
                if str(ref.resolved) != ref.hint:
                    parts.append(ref.render())
            return ", ".join(parts)
        if self.mission is not None:
            return f"spawning mission {self.mission.render()}"
        return self.kind


def parse_phrasings(
    descriptors: Mapping[DescriptorName, object],
) -> tuple[Phrasing, ...]:
    """Collect every declared phrasing across the library.

    Order is by descriptor name then declaration order, so a text matching two
    phrasings compiles the same way on every run. Ambiguity across descriptors is a
    load-time lint, not a runtime coin toss.
    """
    out: list[Phrasing] = []
    for name in sorted(descriptors, key=str):
        declared = getattr(descriptors[name], "utterances", ())
        for phrasing in declared:
            out.append(phrasing)
    return tuple(out)


def _resolve(hint: str, utterance: Utterance) -> Reference:
    """Ground a referring expression against platform-attached deixis.

    The platform sent what it could see; the fleet has the map. If the hint is not
    demonstrative it is taken literally -- "zone bench-3" is an argument, not an act
    of pointing.
    """
    lowered = hint.lower()
    if not lowered.startswith(DEMONSTRATIVES):
        return Reference(hint=hint, resolved=ObjectName(hint))

    # Strip the demonstrative and use the remainder as the class hint: "that crate"
    # searches the visible objects for a crate.
    for word in DEMONSTRATIVES:
        if lowered.startswith(word):
            lowered = lowered[len(word) :]
            break
    candidates = utterance.deixis.candidates(lowered.strip())
    if len(candidates) == 1:
        return Reference(hint=hint, resolved=candidates[0], candidates=candidates)
    return Reference(hint=hint, candidates=candidates)


def compile_utterance(
    utterance: Utterance,
    *,
    envelope: PrincipalEnvelope,
    phrasings: Sequence[Phrasing],
) -> Artifact:
    """Utterance to artifact. Pure, deterministic, and untrusted.

    The order of checks is chosen so that the *most structural* refusal wins. If
    there is no compilation target at all, that is what the journal should say --
    reporting "not invocable" or "ambiguous reference" for a phrase that names no
    descriptor would suggest the door was locked when in fact there is no door.
    """
    reflex = safety_word(utterance.text)
    if reflex is not None:
        # Never went near the language path. Words with deadlines are
        # reflexes; sentences are missions.
        return Artifact(kind=ArtifactKind.REFLEX, utterance=utterance, reflex=reflex)

    matched: tuple[Phrasing, Mapping[str, str]] | None = None
    for phrasing in phrasings:
        arguments = phrasing.match(utterance.text)
        if arguments is not None:
            matched = (phrasing, arguments)
            break

    if matched is None:
        return Artifact(
            kind=ArtifactKind.REFUSAL,
            utterance=utterance,
            refusal=RefusalReason.NO_TARGET,
            detail=(
                "no descriptor in the library declares a phrasing for this; nothing is listening"
            ),
        )

    phrasing, raw_arguments = matched
    if not CompilationTarget.permits(envelope.max_target, CompilationTarget.INVOCATION):
        return Artifact(
            kind=ArtifactKind.REFUSAL,
            utterance=utterance,
            refusal=RefusalReason.TARGET_NOT_PERMITTED,
            detail=f"{envelope.id} may not compile to invocations",
        )
    if not envelope.may_invoke(str(phrasing.descriptor)):
        return Artifact(
            kind=ArtifactKind.REFUSAL,
            utterance=utterance,
            refusal=RefusalReason.NOT_INVOCABLE,
            detail=(
                f"{envelope.id} may not invoke {str(phrasing.descriptor)!r} "
                f"(permitted: {sorted(envelope.invocable)})"
            ),
        )

    references = tuple(_resolve(value, utterance) for _, value in sorted(raw_arguments.items()))
    ambiguous = [r for r in references if r.ambiguous]
    if ambiguous:
        # OQ-N2's interaction question. Clarifying costs nothing while it waits --
        # the dispatcher blocks on the reply pipe, descheduled -- and a best guess
        # that turns out wrong has already moved a crate.
        return Artifact(
            kind=ArtifactKind.REFUSAL,
            utterance=utterance,
            refusal=RefusalReason.AMBIGUOUS_REFERENCE,
            detail="; ".join(r.render() for r in ambiguous),
            references=references,
        )
    unresolved = [r for r in references if r.unresolved]
    if unresolved:
        return Artifact(
            kind=ArtifactKind.REFUSAL,
            utterance=utterance,
            refusal=RefusalReason.UNRESOLVED_REFERENCE,
            detail="; ".join(r.render() for r in unresolved),
            references=references,
        )

    resolved_arguments = {
        key: str(_resolve(value, utterance).resolved or value)
        for key, value in raw_arguments.items()
    }
    return Artifact(
        kind=ArtifactKind.INVOCATION,
        utterance=utterance,
        invocation=InvocationSpec(
            descriptor=phrasing.descriptor,
            arguments=resolved_arguments,
            references=references,
        ),
        requested_priority=phrasing.priority,
        confirm=phrasing.confirm,
        references=references,
    )


@dataclass
class Phrasebook:
    """Every phrasing in the library, with the load-time ambiguity check.

    Two descriptors declaring the same phrasing is a design error the site should
    hear about at load: at runtime it would resolve deterministically (by descriptor
    name) and therefore silently, and "silently, deterministically wrong" is the
    hardest kind of wrong to notice.
    """

    phrasings: tuple[Phrasing, ...] = ()
    by_pattern: Mapping[str, tuple[DescriptorName, ...]] = field(
        default_factory=dict[str, tuple[DescriptorName, ...]]
    )

    @staticmethod
    def build(phrasings: Sequence[Phrasing]) -> Phrasebook:
        index: dict[str, list[DescriptorName]] = {}
        for phrasing in phrasings:
            key = " ".join(phrasing.pattern.lower().split())
            index.setdefault(key, []).append(phrasing.descriptor)
        return Phrasebook(
            phrasings=tuple(phrasings),
            by_pattern={k: tuple(v) for k, v in index.items()},
        )

    def collisions(self) -> tuple[str, ...]:
        return tuple(
            f"{pattern!r} is declared by {[str(d) for d in owners]}"
            for pattern, owners in sorted(self.by_pattern.items())
            if len(set(owners)) > 1
        )
