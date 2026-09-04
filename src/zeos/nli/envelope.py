# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The utterance envelope -- what arrives when a human speaks.

The envelope is the whole of the platform's job. ZEOS-Fleet is one kernel with many
bodies, so a robot that hears an instruction does not interpret it:

> **All parsing happens at the fleet kernel.** A robot receiving an instruction does
> not interpret it; it ships it back to the controller inside an utterance envelope.

The consequences are worth stating because they are the reason for the design rather
than side effects of it:

* **One authority point.** N robots do not mean N differently-persuadable front
  doors.
* **One interpretation.** The same instruction to any robot compiles identically --
  no "robot 3 understood me better".
* **Addressee is not executor.** "Clean the whole yard", said to carrier-7, is not
  carrier-7's job. The robot contributes its microphone and its viewpoint.
* **The latency is legal.** Parsing is deliberation, seconds-tolerant, so the
  placement rule ``D(v) < RTT_p99`` sends it off-platform without controversy.

The one exception is the local vocabulary -- words with deadlines. ``STOP`` has a
deadline and therefore cannot round-trip. That set is recognised here (so the
compiler can route it away from the language path entirely) but its *on-platform*
binding is N2; in this kernel it dispatches locally in the sense that it never
becomes an invocation.

Envelopes carry no wall-clock. ``t`` in the spec's YAML is an ISO timestamp; here it
is the driver-supplied token clock, because the core reads no clocks and a replay has
to reproduce the same compilation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import DescriptorName, ObjectName, PipeName, PrincipalId, Priority

__all__ = [
    "Deixis",
    "Utterance",
    "Phrasing",
    "InvocationSpec",
    "MissionSpec",
    "Reference",
    "SAFETY_WORDS",
    "safety_word",
]

_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")

#: The closed local vocabulary. Single words, safety only, deliberately tiny.
#:
#: OQ-N3 asks how large this can grow "before it stops being a compiled reflex and
#: starts being an un-audited second dispatcher", and takes the position that growth
#: pressure should be resisted. So this is a frozen mapping to reflex actions, not an
#: extension point: anything with a sentence structure belongs at the fleet.
SAFETY_WORDS: Mapping[str, str] = {
    "stop": "halt",
    "halt": "halt",
    "hold": "hold",
    "freeze": "hold",
    "back away": "retreat",
    "back off": "retreat",
    "release": "release",
    "let go": "release",
}


def _mapping(value: object) -> Mapping[str, object]:
    """Coerce an untyped YAML node to a mapping. A malformed envelope loses the field
    rather than raising: the platform packaged what it had, and an envelope with no
    deixis is a legal envelope."""
    if isinstance(value, Mapping):
        items: list[tuple[object, object]] = list(value.items())  # pyright: ignore[reportUnknownArgumentType]
        return {str(k): v for k, v in items}
    return {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, list | tuple):
        return list(value)  # pyright: ignore[reportUnknownArgumentType]
    return ()


def safety_word(text: str) -> str | None:
    """Recognise a safety word by exact match on the normalised utterance.

    Exact match, not substring: "don't stop until it's done" contains ``stop`` and
    means the opposite. A keyword spotter that fires on substrings is a keyword
    spotter that can be talked into firing, which defeats the point of having a
    layer with no language model in it.
    """
    normalised = " ".join(text.lower().strip().strip(".!?,").split())
    return SAFETY_WORDS.get(normalised)


@dataclass(frozen=True, slots=True)
class Deixis:
    """Pointing, attached by the platform and resolved at the fleet.

    The platform sends raw observation -- where the speaker was looking, what was
    visible. Grounding "that crate" needs the site map, which is the fleet kernel's
    canonical shared segment and which no single platform holds. OQ-N2 is about how
    well this works; N0 only needs it to exist and be resolvable enough to fail
    honestly when it is ambiguous.
    """

    speaker_pose: str = ""
    gaze_ray: str = ""
    visible_objects: tuple[ObjectName, ...] = ()

    def candidates(self, hint: str) -> tuple[ObjectName, ...]:
        """Visible objects whose name matches a referring expression's hint."""
        if not hint:
            return self.visible_objects
        needle = hint.lower()
        return tuple(o for o in self.visible_objects if needle in str(o).lower())


@dataclass(frozen=True, slots=True)
class Utterance:
    """One thing a human said, packaged for transport to the kernel."""

    text: str
    principal: PrincipalId
    #: Which body heard it. Contributes a viewpoint, not an assignment.
    platform: str = ""
    mic: str = ""
    deixis: Deixis = field(default_factory=Deixis)
    #: The platform's own state at the moment of speaking, for grounding.
    platform_state: Mapping[str, str] = field(default_factory=dict[str, str])
    #: The pipe the utterance arrived on. Its ring is what the words are content
    #: at -- an open-air microphone is a ring-3 source (OQ-N4).
    source_pipe: PipeName = PipeName("user.commands")
    #: Where a reply, clarification, or echo-back goes.
    reply_pipe: PipeName = PipeName("user.replies")
    #: Token-clock reading, supplied by the driver. Never read from a wall clock.
    at: int = 0

    @property
    def is_safety_word(self) -> bool:
        return safety_word(self.text) is not None

    def render(self) -> str:
        where = f" on {self.platform}" if self.platform else ""
        return f"{self.principal}{where}: {self.text!r}"

    @staticmethod
    def parse(raw: Mapping[str, object]) -> Utterance:
        """Build an envelope from the spec's YAML shape."""
        source = _mapping(raw.get("source"))
        deixis_raw = _mapping(raw.get("deixis"))
        return Utterance(
            text=str(raw.get("text", "")),
            principal=PrincipalId(str(raw.get("principal", ""))),
            platform=str(source.get("platform", "") or ""),
            mic=str(source.get("mic", "") or ""),
            deixis=Deixis(
                speaker_pose=str(deixis_raw.get("speaker_pose", "") or ""),
                gaze_ray=str(deixis_raw.get("gaze_ray", "") or ""),
                visible_objects=tuple(
                    ObjectName(str(o)) for o in _sequence(deixis_raw.get("visible_objects"))
                ),
            ),
            platform_state={str(k): str(v) for k, v in _mapping(raw.get("platform_state")).items()},
            source_pipe=PipeName(str(raw.get("source_pipe", "user.commands"))),
            reply_pipe=PipeName(str(raw.get("reply_pipe", "user.replies"))),
            at=int(str(raw.get("t", 0) or 0)),
        )


@dataclass(frozen=True, slots=True)
class Reference:
    """A referring expression the compiler had to resolve, and what it resolved to.

    Kept as an artifact rather than folded silently into the arguments, because
    OQ-N2's interaction question -- clarify versus best-guess-with-echo -- needs the
    ambiguity to still be visible at echo-back time.
    """

    hint: str
    resolved: ObjectName | None = None
    candidates: tuple[ObjectName, ...] = ()

    @property
    def ambiguous(self) -> bool:
        return self.resolved is None and len(self.candidates) > 1

    @property
    def unresolved(self) -> bool:
        return self.resolved is None

    def render(self) -> str:
        if self.resolved is not None:
            return f"{self.hint!r} → {self.resolved}"
        if self.candidates:
            return f"{self.hint!r} → ambiguous: {[str(c) for c in self.candidates]}"
        return f"{self.hint!r} → nothing visible"


#: ``Phrasing`` lives here rather than in ``compiler`` so that loading a descriptor
#: does not require importing the compiler. The declaration is descriptor data; the
#: matching is the compiler's business.
@dataclass(frozen=True, slots=True)
class Phrasing:
    """One way of asking for one descriptor.

    ``pattern`` is a template with ``{name}`` placeholders -- ``"tidy the workshop in
    {zone}"``. Matching is exact modulo whitespace and case; there is no fuzzy
    matching, because a fuzzy matcher is a thing that can be argued with.
    """

    descriptor: DescriptorName
    pattern: str
    #: Maximum urgency this phrasing may request, before the principal's ceiling is
    #: applied on top. Lets a descriptor say "even a plant manager saying this gets
    #: priority 40", which is the descriptor's own opinion about its urgency.
    priority: Priority | None = None
    #: Requires echo-back confirmation before dispatch. Consequential
    #: missions declare this; OQ-N1 exists to measure what it buys.
    confirm: bool = False

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(_PLACEHOLDER.findall(self.pattern))

    def regex(self) -> re.Pattern[str]:
        parts: list[str] = []
        cursor = 0
        for match in _PLACEHOLDER.finditer(self.pattern):
            parts.append(re.escape(self.pattern[cursor : match.start()]))
            parts.append(f"(?P<{match.group(1)}>.+?)")
            cursor = match.end()
        parts.append(re.escape(self.pattern[cursor:]))
        body = "".join(parts).replace(r"\ ", r"\s+")
        return re.compile(rf"^\s*{body}\s*[.!?]*\s*$", re.IGNORECASE)

    def match(self, text: str) -> Mapping[str, str] | None:
        found = self.regex().match(" ".join(text.split()))
        if found is None:
            return None
        return {k: v.strip() for k, v in found.groupdict().items()}


@dataclass(frozen=True, slots=True)
class InvocationSpec:
    """What the compiler produced: a descriptor and its arguments.

    "The common case, and the safest: the executable vocabulary of the system is the
    reviewed descriptor library, so users can *request* anything but the system can
    only *do* things someone engineered."
    """

    descriptor: DescriptorName
    arguments: Mapping[str, str] = field(default_factory=dict[str, str])
    references: tuple[Reference, ...] = ()

    def render(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in sorted(self.arguments.items()))
        return f"{self.descriptor}({args})"


@dataclass(frozen=True, slots=True)
class MissionSpec:
    """A composition of existing descriptors with declared pipes between them
    ."""

    name: str
    members: tuple[InvocationSpec, ...] = ()
    pipes: Sequence[tuple[PipeName, DescriptorName, DescriptorName]] = ()

    def render(self) -> str:
        return f"{self.name}[{', '.join(m.render() for m in self.members)}]"
