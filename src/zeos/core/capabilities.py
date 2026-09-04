# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Capabilities: a job's only effects are pipe writes, and every write is checked.

Pipes are held as capabilities granted in the descriptor, not as ambient authority
The check at every write is the analogue of a syscall permission check, and
it is the layer that still holds when everything above it has failed -- a fully
persuaded model reaching for the mail pipe hits this and stops.

The check::

    check_write(job, cap, payload):
      assert cap in job.capabilities                       else CAPABILITY_FAULT
      eff = max(job.current_integrity, cap.session_floor)  # deputy rule
      assert eff <= cap.min_integrity                      else PRIVILEGE_FAULT
      assert payload matches cap.schema                    else CAPABILITY_FAULT
      assert cap.rate.admit(now)                           else CAPABILITY_FAULT

**Schema width is the security dial.** An endorser that may emit only
``{price: number, in_stock: bool}`` smuggles almost nothing; one that may emit
``{summary: string(2000)}`` smuggles plenty. Because that is a
quantitative claim, ``Schema.capacity_bits`` computes an upper bound on how much
attacker-chosen information can cross the boundary -- turning "narrow enough" from a
judgement call into a number a reviewer can argue with. This is a first cut at
OQ-3, not a settled answer: the bound is an information-theoretic ceiling, not a
demonstration that an attacker can achieve it.

Schemas are validated here rather than by a JSON-schema dependency because the core
stays stdlib-only, and because a deliberately small schema language is the point --
an expressive one would make the capacity bound incomputable, which would defeat
the mechanism it exists to support.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from zeos.core.ids import FaultKind, Integrity, PipeName
from zeos.core.integrity import can_write_at, effective_integrity

__all__ = [
    "FieldKind",
    "FieldSpec",
    "Schema",
    "RateLimit",
    "Capability",
    "CapabilityTable",
    "CheckResult",
    "check_write",
    "parse_payload",
]


class FieldKind(enum.StrEnum):
    NUMBER = "number"
    BOOL = "bool"
    STRING = "string"
    ENUM = "enum"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    kind: FieldKind
    max_length: int | None = None  # STRING only
    choices: tuple[str, ...] = ()  # ENUM only

    def validate(self, raw: str) -> str | None:
        """Returns an error message, or None if the value conforms."""
        match self.kind:
            case FieldKind.NUMBER:
                try:
                    float(raw)
                except ValueError:
                    return f"expected a number, got {raw!r}"
            case FieldKind.BOOL:
                if raw.lower() not in ("true", "false", "yes", "no", "1", "0"):
                    return f"expected a boolean, got {raw!r}"
            case FieldKind.ENUM:
                if raw not in self.choices:
                    return f"expected one of {list(self.choices)}, got {raw!r}"
            case FieldKind.STRING:
                if self.max_length is not None and len(raw) > self.max_length:
                    return f"string of {len(raw)} exceeds max_length {self.max_length}"
        return None

    def capacity_bits(self) -> float:
        """Upper bound on attacker-chosen information this field can carry.

        Deliberately generous to the attacker: a number is treated as a full 64-bit
        double, and a string as printable ASCII throughout. An under-estimate here
        would make a schema look safer than it is, which is the wrong direction for
        a security bound to be wrong in.
        """
        match self.kind:
            case FieldKind.BOOL:
                return 1.0
            case FieldKind.ENUM:
                return math.log2(len(self.choices)) if self.choices else 0.0
            case FieldKind.NUMBER:
                return 64.0
            case FieldKind.STRING:
                # ~95 printable ASCII characters.
                return (self.max_length or 0) * math.log2(95)


@dataclass(frozen=True, slots=True)
class Schema:
    """A fixed set of named fields. No nesting, no optionality, no free-form keys --
    each of which would widen the channel and blur the bound.

    ``bare_choices`` is the degenerate and most valuable case: an actuator accepts a
    *value*, not a record. ``{idle, normal, max}`` is a channel of 1.6 bits, which
    is about as narrow as an effect can be while still being useful -- and it is the
    shape most physical actuators actually have.
    """

    name: str
    fields: Mapping[str, FieldSpec] = field(default_factory=dict[str, FieldSpec])
    bare_choices: tuple[str, ...] = ()

    @staticmethod
    def of_values(name: str, choices: Sequence[str]) -> Schema:
        return Schema(name=name, bare_choices=tuple(choices))

    def validate(self, payload: str) -> str | None:
        if self.bare_choices:
            value = payload.strip()
            if value not in self.bare_choices:
                return f"schema {self.name!r}: {value!r} is not one of {list(self.bare_choices)}"
            return None
        values = parse_payload(payload)
        expected = set(self.fields)
        actual = set(values)
        if missing := sorted(expected - actual):
            return f"schema {self.name!r}: missing field(s) {missing}"
        if extra := sorted(actual - expected):
            return f"schema {self.name!r}: unexpected field(s) {extra}"
        for key in sorted(expected):
            if error := self.fields[key].validate(values[key]):
                return f"schema {self.name!r}: field {key!r}: {error}"
        return None

    def capacity_bits(self) -> float:
        """Total upper bound on smuggling capacity through this schema."""
        if self.bare_choices:
            return math.log2(len(self.bare_choices)) if self.bare_choices else 0.0
        return sum(spec.capacity_bits() for spec in self.fields.values())

    @staticmethod
    def parse(name: str, spec: Mapping[str, object]) -> Schema:
        fields: dict[str, FieldSpec] = {}
        for key, raw in spec.items():
            text = str(raw).strip()
            if text.startswith("string"):
                inner = text[len("string") :].strip("() ")
                fields[str(key)] = FieldSpec(
                    kind=FieldKind.STRING, max_length=int(inner) if inner else None
                )
            elif text.startswith("enum"):
                inner = text[len("enum") :].strip("() ")
                choices = tuple(c.strip() for c in inner.split(",") if c.strip())
                fields[str(key)] = FieldSpec(kind=FieldKind.ENUM, choices=choices)
            else:
                fields[str(key)] = FieldSpec(kind=FieldKind(text))
        return Schema(name=name, fields=fields)


def parse_payload(payload: str) -> dict[str, str]:
    """Parse ``key=value key=value`` into a mapping.

    A stand-in for the grammar-constrained decoding a real endorser would use
    The shape does not matter; that the channel is *enumerable* does.
    """
    values: dict[str, str] = {}
    for token in payload.split():
        key, sep, value = token.partition("=")
        if sep:
            values[key.strip()] = value.strip()
    return values


@dataclass
class RateLimit:
    """A sliding-window cap on writes through one capability."""

    max_events: int
    window_ns: int
    _events: list[int] = field(default_factory=list[int])

    def admit(self, now_ns: int) -> bool:
        cutoff = now_ns - self.window_ns
        self._events = [t for t in self._events if t > cutoff]
        if len(self._events) >= self.max_events:
            return False
        self._events.append(now_ns)
        return True


@dataclass
class Capability:
    """Permission to write to one pipe, with the conditions attached."""

    pipe: PipeName
    #: Minimum integrity required to write. Lower is more trusted, so a capability
    #: with ``min_integrity=0`` is the most privileged thing in the system.
    min_integrity: Integrity = Integrity(3)
    schema: Schema | None = None
    rate: RateLimit | None = None
    #: Integrity floor imposed while serving a request from a lower-trust pipe.
    session_floor: Integrity | None = None

    def describe(self) -> str:
        parts = [f"pipe={self.pipe}", f"min_integrity={self.min_integrity}"]
        if self.schema is not None:
            parts.append(f"schema={self.schema.name}({self.schema.capacity_bits():.0f} bits)")
        if self.rate is not None:
            parts.append(f"rate={self.rate.max_events}/{self.rate.window_ns}ns")
        return " ".join(parts)


class CapabilityTable:
    """A job's held capabilities, by pipe.

    ``closed`` distinguishes two kinds of empty, and the distinction is load bearing.
    A table that is empty because the descriptor declared nothing means "this
    behaviour opted out of the capability model" -- phase 1 allows that so MP adoption
    is not all-or-nothing. A table that is empty because a principal's envelope
    narrowed it to nothing means "this job may cause nothing at all".

    Conflating them inverts the authority check: the *more* authority you strip from
    a job, the more it can do, and a job stripped to zero becomes omnipotent. Which is
    exactly what happened the first time narrowing was wired up.
    """

    def __init__(self, capabilities: Iterable[Capability] = (), *, closed: bool = False) -> None:
        self._by_pipe: dict[PipeName, Capability] = {c.pipe: c for c in capabilities}
        self.closed = closed or bool(self._by_pipe)

    def get(self, pipe: PipeName) -> Capability | None:
        return self._by_pipe.get(pipe)

    def has(self, pipe: PipeName) -> bool:
        return pipe in self._by_pipe

    def all(self) -> tuple[Capability, ...]:
        return tuple(self._by_pipe[p] for p in sorted(self._by_pipe))

    def pipes(self) -> tuple[PipeName, ...]:
        return tuple(sorted(self._by_pipe))

    def __len__(self) -> int:
        return len(self._by_pipe)


@dataclass(frozen=True, slots=True)
class CheckResult:
    allowed: bool
    effective: Integrity
    required: Integrity
    fault: FaultKind | None = None
    detail: str = ""


def check_write(
    *,
    capabilities: CapabilityTable,
    pipe: PipeName,
    current_integrity: Integrity,
    payload: str,
    now_ns: int,
    session_floor: Integrity | None = None,
    unheld_is_fault: bool = True,
) -> CheckResult:
    """Run the boundary check for one pipe write.

    ``unheld_is_fault`` exists because not every pipe in a case is a guarded one:
    a descriptor that declares no capabilities at all is running unprotected by
    choice, and phase 1 lets it, rather than making MP adoption all-or-nothing. When
    a job *does* declare capabilities, writing to a pipe outside them is a fault.

    "Declares no capabilities" is ``CapabilityTable.closed`` being False, not the table
    being empty -- see that class. A job narrowed to zero capabilities by its owner's
    envelope has a closed, empty table and may write nothing.
    """
    capability = capabilities.get(pipe)
    if capability is None:
        if unheld_is_fault and capabilities.closed:
            return CheckResult(
                allowed=False,
                effective=current_integrity,
                required=Integrity(0),
                fault=FaultKind.CAPABILITY,
                detail=(
                    f"job holds no capability for pipe {pipe!r} "
                    f"(holds: {[str(p) for p in capabilities.pipes()]})"
                ),
            )
        return CheckResult(allowed=True, effective=current_integrity, required=Integrity(3))

    floor = session_floor if session_floor is not None else capability.session_floor
    effective = effective_integrity(current_integrity, floor)

    if not can_write_at(effective, capability.min_integrity):
        return CheckResult(
            allowed=False,
            effective=effective,
            required=capability.min_integrity,
            fault=FaultKind.PRIVILEGE,
            detail=(
                f"write to {pipe!r} requires integrity <= {capability.min_integrity} "
                f"but the job is at {effective}"
            ),
        )

    if capability.schema is not None:
        if error := capability.schema.validate(payload):
            return CheckResult(
                allowed=False,
                effective=effective,
                required=capability.min_integrity,
                fault=FaultKind.CAPABILITY,
                detail=error,
            )

    if capability.rate is not None and not capability.rate.admit(now_ns):
        return CheckResult(
            allowed=False,
            effective=effective,
            required=capability.min_integrity,
            fault=FaultKind.CAPABILITY,
            detail=(
                f"rate limit exceeded for {pipe!r}: "
                f"{capability.rate.max_events} per {capability.rate.window_ns}ns"
            ),
        )

    return CheckResult(allowed=True, effective=effective, required=capability.min_integrity)


def schemas_from_spec(raw: Mapping[str, object]) -> dict[str, Schema]:
    """Parse a ``schemas:`` block from case configuration."""
    schemas: dict[str, Schema] = {}
    for name, spec in raw.items():
        if isinstance(spec, Mapping):
            schemas[str(name)] = Schema.parse(str(name), spec)  # pyright: ignore[reportUnknownArgumentType]
    return schemas


def capabilities_from_spec(
    entries: Sequence[Mapping[str, object]], schemas: Mapping[str, Schema]
) -> list[Capability]:
    """Parse a descriptor's ``capabilities:`` list."""
    result: list[Capability] = []
    for entry in entries:
        pipe = entry.get("pipe")
        if not pipe:
            raise ValueError("capability entry missing 'pipe'")
        schema_name = entry.get("schema")
        schema = schemas.get(str(schema_name)) if schema_name else None
        if schema_name and schema is None:
            raise ValueError(f"capability for {pipe!r} names unknown schema {schema_name!r}")
        rate_spec = entry.get("rate")
        rate: RateLimit | None = None
        if isinstance(rate_spec, Mapping):
            typed_rate: Mapping[str, object] = rate_spec  # pyright: ignore[reportUnknownVariableType]
            rate = RateLimit(
                max_events=int(str(typed_rate.get("max", 1))),
                window_ns=int(str(typed_rate.get("window_ns", 1_000_000_000))),
            )
        result.append(
            Capability(
                pipe=PipeName(str(pipe)),
                min_integrity=Integrity(int(str(entry.get("min_integrity", 3)))),
                schema=schema,
                rate=rate,
            )
        )
    return result
